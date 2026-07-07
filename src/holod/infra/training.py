from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from holod.infra.util.types import (
    AnalysisType,
    Arr32,
    Arr64,
    ModelType,
    check_dtype,
)

if TYPE_CHECKING:
    from rich.progress import TaskID
    from torch.optim import Optimizer

    from holod.infra.util.prog_helper import ProgressLike

from holod.core.plots import PlotPred, TrainingRepeatConfig
from holod.infra.dataclasses import (
    AutoConfig,
    CoreTrainer,
    EpochMetric,
    TrainingOutput,
    transform_ds,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import get_logger
from holod.infra.util.paths import checkpoints_loc
from holod.infra.util.prog_helper import setup_training_progress

logger = get_logger(__name__)

# -- utils -------------------------------------------------------
# how many models to be kept
MAX_MODEL_HISTORY: int = 50
EXPECTED_IMPROVEMENT_PERCENT: float = 1e-3


def remove_oldest_checkpoint(path_to_model_detail: Path) -> None:
    """Remove the oldest checkpoint if max model history is reached."""
    # get files in out directory
    files_in_out_dir: list[Path] = list(path_to_model_detail.parent.iterdir())
    if len(files_in_out_dir) > (MAX_MODEL_HISTORY + 1):
        # clean up directory if needed to preserve storage
        # if checkpoint folder has > MAX_MODEL_HISTORY, remove oldest
        # find files that end in ".pth"
        file_count_pth: int = 0
        oldest_mod_time: float = np.inf
        oldest_file: Path | None = None
        for out_file in files_in_out_dir:
            if out_file.as_posix().endswith(".pth"):
                file_count_pth += 1
                current_mod_time: float = out_file.stat().st_mtime
                if current_mod_time < oldest_mod_time:
                    oldest_file = out_file
                    oldest_mod_time = current_mod_time
        # remove oldest_file if limit reached
        if file_count_pth > MAX_MODEL_HISTORY and oldest_file is not None:
            logger.debug(
                f"Max model history limit of {MAX_MODEL_HISTORY} "
                + f"reached, deleting {oldest_file}"
            )
            oldest_file.unlink()


# ---------------------------------------------------------


@dataclass
class TrainProgress:
    best_val: float
    save_best: bool


type TrainStatus = TrainProgress | None


def _check_model_improvement(
    metric_val_hist: list[float],
    epoch_metric: EpochMetric,
    a_cfg: AutoConfig,
    best_val: float,
    core_trainer: CoreTrainer,
    labels_tensor: Tensor,
    path_to_model: Path,
    epoch: int,
    short_range: int = 3,
    long_range: int = 5,
) -> TrainStatus:
    if a_cfg.analysis == AnalysisType.REG:
        # Lower MAE is better
        save_best_model_flag = epoch_metric.metric_val < best_val
        best_val_out = epoch_metric.metric_val if save_best_model_flag else best_val
        logger.debug(f"At {epoch} / {a_cfg.epoch_count} Val MAE: {epoch_metric.metric_val:.9f} mm")
    else:
        # Higher Accuracy is better
        save_best_model_flag = epoch_metric.metric_val > best_val
        best_val_out = epoch_metric.metric_val if save_best_model_flag else best_val
        logger.debug(
            f"At {epoch} / {a_cfg.epoch_count} Val Acc: {epoch_metric.metric_val * 100:.2f} %"
        )

    if epoch > (a_cfg.epoch_count / 5) and epoch >= 10:
        # get percent diff to measure improvement, desire current < previous
        percent_diff_history = get_percent_diff_history(
            epoch_metric, metric_val_hist, a_cfg.analysis
        )

        diff_short = abs(percent_diff_history[-short_range] - percent_diff_history[-1])
        diff_long = abs(percent_diff_history[-long_range] - percent_diff_history[-1])

        # if epoch is > N/5 and no improvement over three epochs, display error
        if diff_short <= EXPECTED_IMPROVEMENT_PERCENT:
            print(
                f"Small or no improvement of metric val: {diff_short}"
                + f" from epoch {epoch - 3} to epoch {epoch}",
            )
            # if after 5 epochs, stop training
            if diff_long < EXPECTED_IMPROVEMENT_PERCENT:
                print(
                    "Training stopping, little to no improvement after " + f"{long_range} epochs",
                )
                return None

    if save_best_model_flag:
        # convert best_val_metric form of 5 numbers, in scientific notation
        # create file with name that is unique to evaluation
        best_model_name: str = (
            path_to_model.name.removesuffix(".pth") + f"{best_val_out:3e}" + ".pth"
        )
        path_to_model_detail = path_to_model.parent / Path(best_model_name)

        # check if files in directory has potential amount of
        # files to reach limit before loop.
        remove_oldest_checkpoint(path_to_model_detail)
        checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, a_cfg, best_val_out)
        checkpoint.torch_save(path_to_model_detail)
    return TrainProgress(best_val_out, save_best_model_flag)


# TODO: this is recreating an array each time, only need to calculate the newest values
def get_percent_diff_history(
    epoch_metric: EpochMetric,
    metric_val_hist: list[float],
    analysis_type: AnalysisType,
) -> Arr32:
    """Calculate percentage difference across epochs for finding when improvement stalls."""
    try:
        average = [(x + epoch_metric.metric_val) / 2 for x in metric_val_hist]
        match analysis_type:
            case AnalysisType.REG:
                return np.asarray(
                    [
                        (x - epoch_metric.metric_val) / average[i]
                        for (i, x) in enumerate(metric_val_hist)
                    ],
                    dtype=np.float32,
                )
            case AnalysisType.CLASS:
                return np.asarray(
                    [
                        (epoch_metric.metric_val - x) / average[i]
                        for (i, x) in enumerate(metric_val_hist)
                    ],
                    dtype=np.float32,
                )

    except Exception as e:
        logger.error("Could not calculate the percent difference.")
        raise e


def epoch_loop(
    a_cfg: AutoConfig,
    core_trainer: CoreTrainer,
    progress_bar: ProgressLike,
    task_id: TaskID,
    step: str,
) -> EpochMetric:
    """Do one loop of evaluation and training."""
    device = torch.device("cpu") if a_cfg.device() == "cpu" else torch.device("cuda")
    if step == "train":
        loader = core_trainer.train_loader
        _ = core_trainer.model.to(device)
    else:
        loader = core_trainer.val_loader
        # reduces memory consumption when doing inference, as is the case for validation
        _ = core_trainer.model.eval()
    logger.debug(f"Using: {device}, for training.")

    epoch_metric: EpochMetric = EpochMetric()
    abs_err_sum: int | float = 0
    total_samples_for_metric: int = 0  # Denominator for MAE/Accuracy
    # bin-center lookup for the physical-distance MAE (classification only)
    bin_centers_tens: Tensor | None = None
    phys_err_sum: float = 0.0
    if a_cfg.analysis == AnalysisType.CLASS and core_trainer.bin_centers is not None:
        bin_centers_tens = torch.as_tensor(
            core_trainer.bin_centers, dtype=torch.float32, device=device
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
        if a_cfg.analysis == AnalysisType.REG and pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        # check loss fn type
        if isinstance(core_trainer.loss_fn, nn.MSELoss):
            labels_tens = labels_tens.float()  # expects float
        elif isinstance(core_trainer.loss_fn, nn.BCEWithLogitsLoss):
            labels_tens = labels_tens.float()  # BCE expects float 0/1
        elif isinstance(core_trainer.loss_fn, nn.CrossEntropyLoss):
            labels_tens = labels_tens.long()
            # fail fast with a readable error; a mismatch here otherwise surfaces as an
            # opaque CUDA device-side assert in backward()
            if pred.ndim != 2 or pred.shape[1] != a_cfg.num_classes:
                raise ValueError(
                    f"Model '{a_cfg.backbone.value}' output shape {tuple(pred.shape)} is "
                    f"incompatible with CrossEntropyLoss over {a_cfg.num_classes} classes; "
                    "expected (batch, num_classes)."
                )
            if labels_tens.numel() > 0 and (
                int(labels_tens.min()) < 0 or int(labels_tens.max()) >= a_cfg.num_classes
            ):
                raise ValueError(
                    f"Class labels outside [0, {a_cfg.num_classes}): "
                    f"min {int(labels_tens.min())}, max {int(labels_tens.max())}."
                )
        else:
            raise Exception(f"Loss function is unknown: {core_trainer.loss_fn}")

        # -- calculate loss-----------------------------------------------------------------------
        # this loss is the current average over the batch
        # to find the total loss over the epoch we must sum over
        # each mean loss times the number of images
        loss_fn_current = core_trainer.loss_fn(pred, labels_tens)

        # valuation is set to no_grad, this will not work on said tensor
        if step == "train":
            loss_fn_current.backward()  # compute the gradient of the loss
            core_trainer.optimizer.step()  # compute one step of the optimization algorithm
            core_trainer.optimizer.zero_grad()  # reset gradients each loop

        # sum the value of the loss, scaled to the size of image tensor
        epoch_metric.loss_epoch += cast("float", loss_fn_current.item())
        epoch_metric.total_samples += imgs_tens.size(0)

        # -- absolute error sum & labels tensor --------------------------------------------------
        if a_cfg.analysis == AnalysisType.REG:
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

        # update progress bar
        if step == "val":
            progress_bar.update(
                task_id,
                advance=1,
                val_loss=epoch_metric.loss_epoch / epoch_metric.total_samples,
            )
        else:
            progress_bar.update(
                task_id,
                advance=1,
                train_loss=epoch_metric.loss_epoch / epoch_metric.total_samples,
            )

    return epoch_metric


type StateDict = dict[str, Any]


# TODO: migrate to use this dataclass, for both checkpoint and model, have model
# just subsume this class <10-31-25, >
@dataclass
class Checkpoint:
    """Serialized training state used for resuming."""

    epoch: int
    model_state_dict: StateDict
    labels: torch.Tensor
    bin_centers: Arr64 | None
    num_classes: int
    val_metric: float
    model_type: ModelType
    # z-label normalization stats (regression only); None on classification
    # checkpoints and on checkpoints written before these fields existed
    z_avg: float | None = None
    z_std: float | None = None

    @classmethod
    def from_dict(cls, torch_dict: dict[str, Any]) -> Checkpoint:
        """Rebuild a checkpoint from the dictionary written by torch_save."""
        return cls(
            epoch=torch_dict["epoch"],
            model_state_dict=torch_dict["model_state_dict"],
            labels=torch_dict["labels"],
            val_metric=torch_dict["val_metric"],
            bin_centers=torch_dict["bin_centers"],
            num_classes=torch_dict["num_classes"],
            model_type=torch_dict["model_type"],
            z_avg=torch_dict.get("z_avg"),
            z_std=torch_dict.get("z_std"),
        )

    @classmethod
    def from_epoch(
        cls,
        epoch: int,
        core_trainer: CoreTrainer,
        labels_tensor,
        a_cfg: AutoConfig,
        val_metric: float,
    ) -> Checkpoint:
        """Build a checkpoint from the current epoch's training state."""
        (z_avg, z_std) = core_trainer.get_std_avg(a_cfg.analysis)
        return cls(
            epoch=epoch,
            model_state_dict=core_trainer.model.state_dict(),
            labels=labels_tensor,
            val_metric=val_metric,
            bin_centers=core_trainer.bin_centers,
            num_classes=a_cfg.num_classes,
            model_type=a_cfg.backbone,
            z_avg=float(z_avg) if z_avg is not None else None,
            z_std=float(z_std) if z_std is not None else None,
        )

    def torch_save(self, path) -> None:
        torch.save(
            asdict(self),
            path,
        )


@dataclass
class ModelCheckpoint:
    """Serialized training state used for models."""

    checkpoint: Checkpoint
    optimizer_state_dict: StateDict
    train_loss: float
    val_loss: float

    # torch.load(weights_only=True) only unpickles allowlisted types; besides tensors
    # and primitives, the saved dict holds a ModelType enum and numpy bin_centers.
    SAFE_GLOBALS: ClassVar[list[Any]] = [
        ModelType,
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
        np._core.multiarray._reconstruct,  # numpy's ndarray pickle helper  # pyright: ignore[reportAttributeAccessIssue]
    ]

    @classmethod
    def from_epoch(
        cls,
        epoch: int,
        core_trainer: CoreTrainer,
        labels_tensor,
        a_cfg: AutoConfig,
        val_metric: float,
        optimizer_state_dict: StateDict,
        train_loss: float,
        val_loss: float,
    ) -> ModelCheckpoint:
        # Awlays save latest model, after going through both loaders
        checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, a_cfg, val_metric)
        return cls(
            checkpoint=checkpoint,
            optimizer_state_dict=optimizer_state_dict,
            train_loss=train_loss,
            val_loss=val_loss,
        )

    @classmethod
    def load_ckpt(
        cls, path_ckpt: str, optimizer: Optimizer, model: nn.Module, device: str
    ) -> ModelCheckpoint:
        """Load a training checkpoint, restore optimizer and model state."""
        with torch.serialization.safe_globals(cls.SAFE_GLOBALS):
            torch_dict: dict[str, Any] = torch.load(
                path_ckpt, weights_only=True, map_location=device
            )
        checkpoint = Checkpoint.from_dict(torch_dict["checkpoint"])
        # returns unexpected keys or missing keys for the model if either case occurs
        incompatible = model.load_state_dict(checkpoint.model_state_dict)
        # must move model onto expected device before loading optimiser
        _ = model.to(device)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(f"State dict load mismatch: {incompatible}")

        ckpt: ModelCheckpoint = cls(
            checkpoint=checkpoint,
            optimizer_state_dict=torch_dict["optimizer_state_dict"],
            train_loss=torch_dict["train_loss"],
            val_loss=torch_dict["val_loss"],
        )
        optimizer.load_state_dict(ckpt.optimizer_state_dict)
        logger.info(
            f"Resumed from checkpoint {path_ckpt} "
            + f"(epoch {checkpoint.epoch}, val_metric {checkpoint.val_metric:.4g})"
        )

        return ckpt

    def torch_save(self, path) -> None:
        torch.save(
            asdict(self),
            path,
        )


def init_training(a_cfg: AutoConfig, ckpt: ModelCheckpoint | None):
    if ckpt is None:
        labels_tensor: Tensor = torch.empty([1, 1])
        epoch_metric: EpochMetric = EpochMetric(metric_val=0)
        avg_loss_train: float = 0
        avg_loss_val: float = 0
    else:
        labels_tensor = ckpt.checkpoint.labels.to(a_cfg.device())
        epoch_metric = EpochMetric(ckpt.checkpoint.val_metric)
        avg_loss_train = ckpt.train_loss
        avg_loss_val = ckpt.val_loss

    # Paths
    checkpoint_dir = checkpoints_loc()
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir, labels_tensor, epoch_metric, avg_loss_train, avg_loss_val


def train_eval_epoch(
    core_trainer: CoreTrainer,
    a_cfg: AutoConfig,
    best_val_metric: float,
    ckpt: ModelCheckpoint | None,
) -> TrainingOutput:
    """Trains model over one epoch.

    Returns:
        float: Average loss of the model after epoch completes.

    """
    checkpoint_dir, labels_tensor, epoch_metric, avg_loss_train, avg_loss_val = init_training(
        a_cfg, ckpt
    )
    metric_val_hist: list[float] = []

    path_to_checkpoint: Path = checkpoint_dir / Path("latest_checkpoint.tar")
    path_to_model: Path = checkpoint_dir / Path(f"checkpoint_{a_cfg.backbone.name}.pth")
    # querying the CUDA device name raises on CPU-only hosts (e.g. a Colab CPU runtime)
    device_name = torch.cuda.get_device_name() if a_cfg.device() == "cuda" else "cpu"
    progress_bar, train_task, val_task, epoch_task = setup_training_progress(
        a_cfg, avg_loss_train, avg_loss_val, core_trainer, device_name
    )

    _ = core_trainer.model.train()

    with progress_bar:  # allow for tracking of progress
        for epoch in range(a_cfg.epoch_count):
            for loader in [core_trainer.train_loader, core_trainer.val_loader]:
                if loader is core_trainer.train_loader:
                    # -- Training ----------------------------------------------------------------
                    # ensure model is on proper device
                    progress_bar.reset(
                        train_task, total=len(core_trainer.train_loader), train_loss=0
                    )
                    epoch_metric = epoch_loop(
                        a_cfg, core_trainer, progress_bar, train_task, "train"
                    )
                    avg_loss_train = epoch_metric.average_loss()
                else:
                    # -- Evaluation Loop ---------------------------------------------------------
                    progress_bar.reset(val_task, total=len(core_trainer.val_loader), val_loss=0)
                    with torch.no_grad():
                        epoch_metric = epoch_loop(
                            a_cfg, core_trainer, progress_bar, val_task, "val"
                        )
                        avg_loss_val = epoch_metric.average_loss()

            # -- test if model is still improving ------------------------------------------------

            # store metric val from epochs previous
            metric_val_hist.append(epoch_metric.metric_val)
            # -- Save best checkpoint, if metric is better ---------------------------------------
            train_status = _check_model_improvement(
                metric_val_hist,
                epoch_metric,
                a_cfg,
                best_val_metric,
                core_trainer,
                labels_tensor,
                path_to_model,
                epoch,
            )
            if train_status is None:
                break
            best_val_metric = train_status.best_val

            # Always save latest model, after going through both loaders
            model_checkpoint = ModelCheckpoint.from_epoch(
                epoch,
                core_trainer,
                labels_tensor,
                a_cfg,
                best_val_metric,
                core_trainer.optimizer.state_dict(),
                avg_loss_train,
                avg_loss_val,
            )
            model_checkpoint.torch_save(path_to_checkpoint)

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
    train_cfg: CoreTrainer = transform_ds(holo_base_ds, a_config)

    # with coretrainer and autoconfig created, we can load checkpoint
    best_val_metric: int | float
    if path_ckpt is not None:
        ckpt = ModelCheckpoint.load_ckpt(
            path_ckpt, train_cfg.optimizer, train_cfg.model, a_config.device()
        )
        best_val_metric = ckpt.checkpoint.val_metric
    else:
        # For measuring evaluation: classificaton is maximizing correct bins,
        # regression is minimizing the error from expected
        best_val_metric = float("inf") if a_config.analysis == AnalysisType.REG else -float("inf")
        ckpt = None

    # train/validate, for all epochs
    training_output = train_eval_epoch(train_cfg, a_config, best_val_metric, ckpt)
    training_output.display_loss()

    # For class, train_cfg.evaluation_metric is base.bin_centers
    bin_centers = holo_base_ds.bin_centers if a_config.analysis == AnalysisType.CLASS else None
    # regression needs std and average
    (z_avg, z_std) = train_cfg.get_std_avg(a_config.analysis)

    # -- Get & Save Training Data for Plotting ---------------------------------------------------
    # TODO: Save the object to json/pickle/torch file to have access to
    # predictions for debugging/inspection purposes.
    return PlotPred.from_z_preds(
        auto_config=a_config,
        train_cfg=train_cfg,
        bin_centers=bin_centers,
        avg_train_loss=training_output.avg_train_loss,
        avg_val_loss=training_output.avg_val_loss,
        bin_edges=holo_base_ds.bin_edges,
        z_avg=z_avg,
        z_std=z_std,
        repeat_config=TrainingRepeatConfig(
            a_config.to_user_config(),
            training_output.avg_train_loss,
            training_output.avg_val_loss,
        ),
    )
