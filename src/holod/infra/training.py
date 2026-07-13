from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from holod.infra.util.training_help import (
    TrainImprovement,
    init_epoch,
    init_training,
    save_loss_to_file,
)
from holod.infra.util.types import (
    AnalysisType,
    check_dtype,
)

if TYPE_CHECKING:
    from rich.progress import TaskID

    from holod.infra.util.prog_helper import ProgressLike

from holod.core.plots import PlotPred
from holod.infra.dataclasses import (
    AutoConfig,
    CoreTrainer,
    EpochMetric,
    ModelCheckpoint,
    TrainingOutput,
    create_training_setup,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import get_logger
from holod.infra.util.prog_helper import setup_training_progress

logger = get_logger(__name__)


def train_epoch_loop(
    core_trainer: CoreTrainer, progress_bar: ProgressLike, task_id: TaskID, device: str
) -> EpochMetric:
    """Do one loop of training."""
    loader = core_trainer.train_loader
    _ = core_trainer.model.to(device)
    logger.debug(f"Using: {device}, for training.")

    epoch_metric, abs_err_sum, total_samples_for_metric, bin_centers_tens, phys_err_sum = (
        init_epoch(core_trainer, device)
    )
    imgs: Tensor
    labels: Tensor
    for imgs, labels in loader:
        # sending the images to the device is ensures the model, images, and labels
        # are on the same device, with datatype of float32 tensors on [0,1]
        # PERF: Non-blocking speed up
        imgs_tens: Tensor = imgs.to(device, non_blocking=True)
        labels_tens: Tensor = labels.to(device, non_blocking=True)

        # TODO: test check, move to tests instead? <07-24-25, luxShrine>
        assert (
            next(core_trainer.model.parameters()).device == imgs_tens.device == labels_tens.device
        ), (
            f"Images {imgs_tens.device}, labels {labels_tens.device}, or model "
            f"{next(core_trainer.model.parameters()).device} not on same device."
        )

        # -- pass images to model ----------------------------------------------------------------
        # get output of tensor data
        pred: Tensor = core_trainer.model(imgs_tens)
        if not check_dtype("dtype", torch.float32, predictions=pred, images=imgs_tens):
            raise Exception(f"dtypes do not match {__file__}")

        # -- refine preds/labels -----------------------------------------------------------------
        # if we are doing regression and the output tensor is of size [Batch, 1]
        # we need to "squeeze" it into a one dimentisonal tensor of [Batch]
        if core_trainer.analysis == AnalysisType.REG and pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        # check loss fn type
        labels_tens = core_trainer.check_loss(labels_tens, pred)

        # -- calculate loss-----------------------------------------------------------------------
        # this loss is the current average over the batch
        # to find the total loss over the epoch we must sum over
        # each mean loss times the number of images
        loss_fn_current = core_trainer.loss_fn(pred, labels_tens)

        loss_fn_current.backward()  # compute the gradient of the loss
        core_trainer.optimizer.step()  # compute one step of the optimization algorithm
        core_trainer.optimizer.zero_grad()  # reset gradients each loop

        # sum the value of the loss, scaled to the size of image tensor
        epoch_metric.loss_epoch += cast("float", loss_fn_current.item())
        epoch_metric.total_samples += imgs_tens.size(0)

        # -- absolute error sum & labels tensor --------------------------------------------------
        if core_trainer.analysis == AnalysisType.REG:
            # de-normalise as tensors only for regression
            (z_true, z_pred) = core_trainer.denormalize_tens(pred, labels_tens)
            # error is the distance from the correct value, z_pred -> z_true
            abs_err_sum += torch.sum(torch.abs(z_pred - z_true)).item()
            total_samples_for_metric += z_true.numel()  # num of elements
        else:
            # Get predicted class indices
            pred_classes = torch.argmax(pred, dim=1)  # Shape: [B]
            # Labels size should be [B] and long type
            # Ensure labels are the same shape as pred_classes for comparison
            # TODO: test check, move to tests instead? <07-24-25, luxShrine>
            assert pred_classes.shape == labels_tens.shape, (
                f"Shape mismatch: pred_classes {pred_classes.shape}, labels {labels_tens.shape}"
            )

            # error is the number of "misses" in which the model places the image in the wrong depth
            abs_err_sum += (pred_classes == labels_tens).sum().item()
            # number of correct predictions
            total_samples_for_metric += labels_tens.size(0)

            if bin_centers_tens is not None:
                # physical distance between predicted and true depth bins; unlike accuracy
                # this distinguishes a 1-bin miss from a many-bin miss (mm, quantized to
                # bin centers so within-bin error is excluded)
                phys_err_sum += (
                    torch.sum(
                        torch.abs(bin_centers_tens[pred_classes] - bin_centers_tens[labels_tens])
                    )
                ).item()

        # using bin value for classification or # continuous value for regression
        epoch_metric.labels_tensor = torch.as_tensor(
            core_trainer.evaluation_metric,
            dtype=torch.float32,
        )
        epoch_metric.metric_val = abs_err_sum / total_samples_for_metric
        if bin_centers_tens is not None:
            epoch_metric.mae_mm = phys_err_sum / total_samples_for_metric

        progress_bar.update(
            task_id,
            advance=1,
            train_loss=epoch_metric.loss_epoch / epoch_metric.total_samples,
        )

    return epoch_metric


def eval_epoch_loop(
    core_trainer: CoreTrainer, progress_bar: ProgressLike, task_id: TaskID, device: str
) -> EpochMetric:
    """Do one loop of evaluation."""
    loader = core_trainer.val_loader
    # reduces memory consumption when doing inference, as is the case for validation
    _ = core_trainer.model.eval()
    logger.debug(f"Using: {device}, for training.")

    epoch_metric, abs_err_sum, total_samples_for_metric, bin_centers_tens, phys_err_sum = (
        init_epoch(core_trainer, device)
    )

    imgs: Tensor
    labels: Tensor
    for imgs, labels in loader:
        # sending the images to the device is ensures the model, images, and labels
        # are on the same device, with datatype of float32 tensors on [0,1]
        # PERF: Non-blocking speed up
        imgs_tens: Tensor = imgs.to(device, non_blocking=True)
        labels_tens: Tensor = labels.to(device, non_blocking=True)

        # TODO: test check, move to tests instead? <07-24-25, luxShrine>
        assert (
            next(core_trainer.model.parameters()).device == imgs_tens.device == labels_tens.device
        ), (
            f"Images {imgs_tens.device}, labels {labels_tens.device}, or model "
            f"{next(core_trainer.model.parameters()).device} not on same device."
        )

        # -- pass images to model ----------------------------------------------------------------
        pred: Tensor = core_trainer.model(imgs_tens)
        if not check_dtype("dtype", torch.float32, predictions=pred, images=imgs_tens):
            raise Exception(f"dtypes do not match {__file__}")

        # -- refine preds/labels -----------------------------------------------------------------
        # if we are doing regression and the output tensor is of size [Batch, 1]
        # we need to "squeeze" it into a one dimentisonal tensor of [Batch]
        if core_trainer.analysis == AnalysisType.REG and pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        # check loss fn type
        labels_tens = core_trainer.check_loss(labels_tens, pred)

        # -- calculate loss-----------------------------------------------------------------------
        # this loss is the current average over the batch
        # to find the total loss over the epoch we must sum over
        # each mean loss times the number of images
        loss_fn_current = core_trainer.loss_fn(pred, labels_tens)

        # sum the value of the loss, scaled to the size of image tensor
        epoch_metric.loss_epoch += cast("float", loss_fn_current.item())
        epoch_metric.total_samples += imgs_tens.size(0)

        # -- absolute error sum & labels tensor --------------------------------------------------
        if core_trainer.analysis == AnalysisType.REG:
            # de-normalise as tensors only for regression
            (z_true, z_pred) = core_trainer.denormalize_tens(pred, labels_tens)
            # error is the distance from the correct value, z_pred -> z_true
            abs_err_sum += torch.sum(torch.abs(z_pred - z_true)).item()
            total_samples_for_metric += z_true.numel()  # num of elements
        else:
            # Get predicted class indices
            pred_classes = torch.argmax(pred, dim=1)  # Shape: [B]
            # Labels size should be [B] and long type
            # Ensure labels are the same shape as pred_classes for comparison
            # TODO: test check, move to tests instead? <07-24-25, luxShrine>
            assert pred_classes.shape == labels_tens.shape, (
                f"Shape mismatch: pred_classes {pred_classes.shape}, labels {labels_tens.shape}"
            )

            # error is the number of "misses" in which the model places the image in the wrong depth
            abs_err_sum += (pred_classes == labels_tens).sum().item()
            # number of correct predictions
            total_samples_for_metric += labels_tens.size(0)
            if bin_centers_tens is not None:
                # physical distance between predicted and true depth bins; unlike accuracy
                # this distinguishes a 1-bin miss from a many-bin miss (mm, quantized to
                # bin centers so within-bin error is excluded)
                phys_err_sum += (
                    torch.sum(
                        torch.abs(bin_centers_tens[pred_classes] - bin_centers_tens[labels_tens])
                    )
                ).item()

        # using bin value for classification or continuous value for regression
        epoch_metric.labels_tensor = torch.as_tensor(
            core_trainer.evaluation_metric,
            dtype=torch.float32,
        )
        epoch_metric.metric_val = abs_err_sum / total_samples_for_metric
        if bin_centers_tens is not None:
            epoch_metric.mae_mm = phys_err_sum / total_samples_for_metric

        # update progress bar
        progress_bar.update(
            task_id,
            advance=1,
            val_loss=epoch_metric.loss_epoch / epoch_metric.total_samples,
        )

    return epoch_metric


def train_eval_epoch(
    core_trainer: CoreTrainer,
    best_val_metric: float,
    ckpt: ModelCheckpoint | None,
) -> TrainingOutput:
    """Trains model over one epoch.

    Returns:
        float: Average loss of the model after epoch completes.

    """
    checkpoint_dir, labels_tensor, epoch_metric, avg_loss_train, avg_loss_val, device = (
        init_training(core_trainer, ckpt)
    )
    metric_val_hist: list[float] = []
    # one epoch-history file per run; see save_loss_to_file
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    val_metric_name = "accuracy" if core_trainer.analysis == AnalysisType.CLASS else "mae_mm"

    path_to_checkpoint: Path = checkpoint_dir / Path("latest_checkpoint.tar")
    path_to_model: Path = checkpoint_dir / Path(f"checkpoint_{core_trainer.backbone.name}.pth")
    # querying the CUDA device name raises on CPU-only hosts (e.g. a Colab CPU runtime)
    device_name = torch.cuda.get_device_name() if core_trainer.device == "cuda" else "cpu"
    progress_bar, train_task, val_task, epoch_task = setup_training_progress(
        core_trainer, avg_loss_train, avg_loss_val, device_name
    )

    _ = core_trainer.model.train()

    with progress_bar:  # allow for tracking of progress
        for epoch in range(core_trainer.a_cfg.epoch_count):
            progress_bar.reset(val_task, total=len(core_trainer.val_loader), val_loss=0)
            progress_bar.reset(train_task, total=len(core_trainer.train_loader), train_loss=0)

            # -- Training ----------------------------------------------------------------
            epoch_metric = train_epoch_loop(core_trainer, progress_bar, train_task, device)
            avg_loss_train = epoch_metric.average_loss()

            # -- Evaluation Loop ---------------------------------------------------------
            with torch.no_grad():
                epoch_metric = eval_epoch_loop(core_trainer, progress_bar, val_task, device)
                avg_loss_val = epoch_metric.average_loss()

            # store metric val from epochs previous
            metric_val_hist.append(epoch_metric.metric_val)
            # -- Save best checkpoint, if metric is better ---------------------------------------
            train_improvement = TrainImprovement.check_model_improvement(
                metric_val_hist,
                epoch_metric,
                best_val_metric,
                core_trainer,
                labels_tensor,
                path_to_model,
                epoch,
            )
            if train_improvement is None:
                break
            best_val_metric = train_improvement.best_val

            # Always save latest model, after going through both loaders
            model_checkpoint = ModelCheckpoint.from_epoch(
                epoch,
                core_trainer,
                labels_tensor,
                best_val_metric,
                core_trainer.optimizer.state_dict(),
                avg_loss_train,
                avg_loss_val,
            )
            model_checkpoint.torch_save(path_to_checkpoint)
            _ = save_loss_to_file(
                core_trainer.backbone.name,
                run_stamp,
                epoch,
                avg_loss_train,
                avg_loss_val,
                epoch_metric.metric_val,
                val_metric_name,
                epoch_metric.mae_mm,
            )

            progress_bar.update(
                epoch_task,
                advance=1,
                train_loss=avg_loss_train,
                val_loss=avg_loss_val,
                val_err=epoch_metric.metric_val,
                lr=core_trainer.optimizer.param_groups[0]["lr"],
            )

    return TrainingOutput(
        avg_loss_train,
        avg_loss_val,
        epoch_metric.metric_val,
        best_val_metric,
        # epoch_metric here is from the final validation pass
        val_mae_mm=epoch_metric.mae_mm,
    )


def train_autofocus(a_config: AutoConfig, path_ckpt: str | None = None) -> PlotPred:
    """Train the autofocus model and return plotting data."""
    # load data
    holo_base_ds = HologramFocusDataset(
        mode=a_config.analysis,
        num_classes=a_config.num_classes,
        csv_file_strpath=a_config.meta_csv_strpath,
    )

    # transform that data
    # distrubute to dataloaders
    # get information to train/validate
    core_trainer: CoreTrainer = create_training_setup(holo_base_ds, a_config)

    # with coretrainer and autoconfig created, we can load checkpoint
    best_val_metric: int | float
    if path_ckpt is not None:
        ckpt = ModelCheckpoint.load_ckpt(
            path_ckpt, core_trainer.optimizer, core_trainer.model, core_trainer.device
        )
        best_val_metric = ckpt.checkpoint.val_metric
    else:
        # For measuring evaluation: classificaton is maximizing correct bins,
        # regression is minimizing the error from expected
        best_val_metric = (
            float("inf") if core_trainer.analysis == AnalysisType.REG else -float("inf")
        )
        ckpt = None

    # train/validate, for all epochs
    training_output = train_eval_epoch(core_trainer, best_val_metric, ckpt)
    training_output.display_loss()

    # For class, train_cfg.evaluation_metric is base.bin_centers
    bin_centers = holo_base_ds.bin_centers if core_trainer.analysis == AnalysisType.CLASS else None
    # regression needs std and average
    (z_avg, z_std) = core_trainer.get_std_avg(core_trainer.analysis)

    # -- Get & Save Training Data for Plotting ---------------------------------------------------
    # TODO: Save the object to json/pickle/torch file to have access to
    # predictions for debugging/inspection purposes.
    return PlotPred.from_z_preds(
        core_trainer=core_trainer,
        bin_centers=bin_centers,
        training_output=training_output,
        bin_edges=holo_base_ds.bin_edges,
        z_avg=z_avg,
        z_std=z_std,
    )
