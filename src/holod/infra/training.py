from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image as ImageType
from rich.progress import (
    Progress,
    TaskID,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from holod.core.plots import PlotPred, TrainingRepeatConfig
from holod.infra.dataclasses import (
    AutoConfig,
    CoreTrainer,
    EpochMetric,
    TrainingOutput,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import console_ as console
from holod.infra.log import get_logger
from holod.infra.util.paths import checkpoints_loc
from holod.infra.util.prog_helper import setup_training_progress
from holod.infra.util.types import (
    Q_,
    AnalysisType,
    Arr32,
    ModelType,
    SubsetImageDepth,
    check_dtype,
    u,
)

logger = get_logger(__name__)

# -- utils -------------------------------------------------------
# how many models to be kept
MAX_MODEL_HISTORY: int = 4
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


class TransformedDataset(Dataset[tuple[ImageType, np.float64]]):
    """Transform the input dataset by applying transfomations to it's contents.

    By wrapping the underlying dataset in this class, we can ensure that attrbutes
    retrived from this new object will undergo the expected transfomrmations while
    preserving the original data.

    Attributes:
        subset_obj: Subset[tuple[ImageType, np.float64]] Subset of the original
                    dataset.
        img_transform: v2.Compose | None  Transformation(s) to be applied to images.
        label_transform: v2.Lambda | None Transformation(s) to be applied to labels.

    """

    def __init__(
        self,
        subset_obj: SubsetImageDepth,
        img_transform: v2.Compose | None = None,
        label_transform: v2.Lambda | None = None,
    ) -> None:
        """Initialize the wrapper with optional image and label transforms."""
        # NOTE: v2 transformation classes != typical torch transformation classes
        self.subset_obj: SubsetImageDepth = subset_obj
        self.img_transform: v2.Compose | None = img_transform
        self.label_transform: v2.Lambda | None = label_transform

    @override
    def __getitem__(self, idx: int) -> tuple[Any | ImageType, Any | np.float64]:
        """Retrieve the image and label from the subest object."""
        img, label = self.subset_obj[idx]  # Gets (PIL Image, raw_label)

        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        """Get number of entries in subset."""
        return len(self.subset_obj)

    @classmethod
    def from_subset(
        cls,
        eval_subset: SubsetImageDepth,
        train_subset: SubsetImageDepth,
        crop_size: int,
        final_label_transform: v2.Lambda,
        model: ModelType,
    ):
        """Construct a tuple of transformed datasets from subsets."""
        rgb_models = [ModelType.ENET, ModelType.RESNET, ModelType.VIT]

        # TODO: using composition, wrap models in class that retrieves their transformations
        # then easily grab transforms

        # TODO: evaluate implementing extra transforms
        extra_tf: list[nn.Module] = [
            # crop random area
            v2.RandomResizedCrop(size=crop_size, antialias=True),
            # flip image with some probability
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]

        common_tf: list[nn.Module] = [
            # convert PIL to tensor
            v2.PILToTensor(),
            # ToTensor preserves original datatype, this ensures it is proper input type
            v2.ToDtype(torch.uint8, scale=True),
            v2.CenterCrop(size=crop_size),
            # normalize across channels, expects float
            v2.ToDtype(torch.float32, scale=True),
        ]
        if model in rgb_models:
            logger.debug("Model type requires three input channels, appending extra transform.")
            # convert to gray before normalization and keep 3 channels for some backbones
            common_tf.extend(
                [
                    v2.Grayscale(num_output_channels=3),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        eval_transform: v2.Compose = v2.Compose(common_tf)
        train_transform: v2.Compose = v2.Compose(extra_tf + common_tf)
        logger.debug("Train and evaluation transformations composed successfully.")

        # providing the dataset with these transforms will create a new subset
        # dataset containing the transformed images
        tf_eval_ds = cls(
            subset_obj=eval_subset,
            img_transform=eval_transform,
            label_transform=final_label_transform,
        )
        tf_train_ds = cls(
            subset_obj=train_subset,
            img_transform=train_transform,
            label_transform=final_label_transform,
        )
        logger.debug("transformed datasets created")
        return (tf_eval_ds, tf_train_ds)


def transform_ds(base: HologramFocusDataset, a_cfg: AutoConfig) -> CoreTrainer:
    """Split the dataset, apply transforms and build dataloaders."""
    (train_subset, eval_subset) = a_cfg.setup_loader_indices(base)
    logger.debug("Train and evaluation subset created successfully")
    # Calculate avg and std from the training subset's physical z-values
    # Need to access original z_m from base dataset via indices of train_subset
    (z_avg, z_std) = base.z.subset_mean_std(subset_indices=train_subset.indices)

    # Define label transforms based on analysis type (handles normalization for REG)
    if a_cfg.analysis == AnalysisType.REG:
        # True physical Zs for validation
        evaluation_metric = base.z.z_array[eval_subset.indices]

        def _reg_label_transform_fn(z_raw: np.float32) -> Tensor:
            """Pass in physical value, return normalized z tensor."""
            z_tensor = torch.as_tensor(z_raw, dtype=torch.float32)
            return (z_tensor - z_avg) / z_std

        # apply local function above to z values to create proper label
        final_label_transform = v2.Lambda(_reg_label_transform_fn)
    else:
        # Physical values of bin centers
        evaluation_metric = base.z_bins[eval_subset.indices]

        # simply convert to tensor
        final_label_transform = v2.Lambda(
            lambda z_raw_idx: torch.as_tensor(z_raw_idx, dtype=torch.long)
        )

    (train_subset, eval_subset) = TransformedDataset.from_subset(
        train_subset, eval_subset, a_cfg.crop_size, final_label_transform, a_cfg.backbone
    )

    # dataset needs to be iterable in terms of pytorch, dataloader does such
    eval_dl: DataLoader[tuple[ImageType, np.float64]] = DataLoader(
        eval_subset,
        batch_size=a_cfg.batch_size,
        drop_last=True,
        num_workers=a_cfg.num_workers,
        pin_memory=(a_cfg.device() == "cuda"),
        prefetch_factor=2 if a_cfg.num_workers > 0 else None,
        shuffle=True,
    )
    train_dl: DataLoader[tuple[ImageType, np.float64]] = DataLoader(
        train_subset,
        batch_size=a_cfg.batch_size,
        drop_last=True,
        num_workers=a_cfg.num_workers,
        pin_memory=(a_cfg.device() == "cuda"),
        prefetch_factor=2 if a_cfg.num_workers > 0 else None,
        shuffle=True,
    )
    logger.debug("Dataloaders created successfully")

    # reg options: nn.BCEWithLogitsLoss(), nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss() if a_cfg.analysis == AnalysisType.CLASS else nn.MSELoss()
    model = a_cfg.create_model()

    # optimizer should only be created after model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a_cfg.opt_lr, weight_decay=a_cfg.opt_weight_decay
    )

    # Scheduler monitors val_metrics[1]: MAE (min) for REG, Accuracy (max) for CLASS
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if a_cfg.analysis == AnalysisType.REG else "max",
        factor=a_cfg.sch_factor,
        patience=a_cfg.sch_patience,
    )

    # Create CoreTrainer
    core_trainer_config = CoreTrainer(
        evaluation_metric=evaluation_metric,  # This is Arr64 of z_m or bin_centers_m
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_subset,
        train_loader=train_dl,
        val_ds=eval_subset,
        val_loader=eval_dl,
        z_std=Q_(z_std, u.m),
        z_avg=Q_(z_avg, u.m),
    )
    console.rule("[black on green] Epoch Variables Initialization Complete ")
    return core_trainer_config


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
    epoch: int,
    short_range: int = 3,
    long_range: int = 5,
) -> TrainStatus:
    if a_cfg.analysis == AnalysisType.REG:
        # Lower MAE is better
        best_val_out = epoch_metric.metric_val if epoch_metric.metric_val < best_val else best_val
        logger.debug(
            f"At {epoch} / {a_cfg.epoch_count} Val Acc: {epoch_metric.metric_val * 100:.2f} %"
        )
    else:
        # Higher Accuracy is better
        best_val_out = epoch_metric.metric_val if epoch_metric.metric_val > best_val else best_val
        logger.debug(f"At {epoch} / {a_cfg.epoch_count} Val MAE: {epoch_metric.metric_val:.9f} µm")

    if epoch > (a_cfg.epoch_count / 5) and epoch >= 10:
        if epoch_metric.metric_val < best_val_out:
            save_best_model_flag = True

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
    a_cfg: AutoConfig, core_trainer: CoreTrainer, progress_bar: Progress, task_id: TaskID, step: str
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

        # using bin value for classification or # continuous value for regression
        epoch_metric.labels_tensor = torch.as_tensor(
            core_trainer.evaluation_metric,
            dtype=torch.float32,
        )
        epoch_metric.metric_val = abs_err_sum / total_samples_for_metric

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
    model_statedict: StateDict
    labels: torch.Tensor
    bin_centers: Arr32 | None
    num_classes: int
    val_metric: float
    model_type: ModelType

    @classmethod
    def from_dict(cls, torch_dict: dict[str, Any]) -> Checkpoint:
        return cls(
            epoch=torch_dict["epoch"],
            model_statedict=torch_dict["model_state_dict"],
            labels=torch_dict["labels"],
            val_metric=torch_dict["val_metric"],
            bin_centers=torch_dict["bin_centers"],
            num_classes=torch_dict["num_bins"],
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
        cls(
            epoch=epoch,
            model_statedict=core_trainer.model.state_dict(),
            labels=labels_tensor,
            val_metric=getattr(core_trainer.train_loader, "bin_centers", None),
            bin_centers=a_cfg.bin_centers,
            num_classes=a_cfg.num_classes,
        )
        pass

    def torch_save(self, path) -> None:
        torch.save(
            asdict(self),
            path,
        )


@dataclass
class ModelCheckpoint:
    """Serialized training state used for models."""

    checkpoint: Checkpoint
    optimizer_statedict: StateDict
    train_loss: float
    val_loss: float

    @classmethod
    def from_epoch(
        cls,
        epoch: int,
        core_trainer: CoreTrainer,
        labels_tensor,
        a_cfg: AutoConfig,
        val_metric: float,
        optimizer_statedict: StateDict,
        train_loss: float,
        val_loss: float,
    ) -> ModelCheckpoint:
        # Awlays save latest model, after going through both loaders
        checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, a_cfg, val_metric)
        return cls(
            checkpoint=checkpoint,
            optimizer_state_dict=core_trainer.optimizer.state_dict(),
            train_loss=train_loss,
            val_loss=val_loss,
        )

    @classmethod
    def load_ckpt(
        cls, path_ckpt: str, optimizer: Optimizer, model: nn.Module, device: str
    ) -> ModelCheckpoint:
        """Load a training checkpoint, restore optimizer and model state."""
        torch_dict: dict[str, Any] = torch.load(path_ckpt, weights_only=True, map_location=device)
        checkpoint = Checkpoint.from_dict(torch_dict)
        # returns unexepected keys or missing keys for the model if either case occurs
        missing_keys = model.load_state_dict()
        # must move model onto expected device before loading optimiser
        _ = model.to(device)
        if len(missing_keys) > 0:
            logger.info(f"{missing_keys}")

        ckpt: ModelCheckpoint = cls(
            checkpoint=checkpoint,
            optimizer_statedict=torch_dict["optimizer_state_dict"],
            train_loss=torch_dict["train_loss"],
            val_loss=torch_dict["val_loss"],
        )
        optimizer.load_state_dict(ckpt.optimizer_statedict)

        return ckpt

    def torch_save(self, path) -> None:
        torch.save(
            asdict(self),
            path,
        )


def train_eval_epoch(
    core_trainer: CoreTrainer, a_cfg: AutoConfig, best_val_metric: float, ckpt: Checkpoint | None
) -> TrainingOutput:
    """Trains model over one epoch.

    Returns:
        float: Average loss of the model after epoch completes.

    """
    labels_tensor: Tensor = torch.empty([1, 1]) if ckpt is None else ckpt.l_tens.to(a_cfg.device())
    epoch_metric: EpochMetric = EpochMetric(metric_val=0 if ckpt is None else ckpt.val_metric)
    avg_loss_train: float = 0 if ckpt is None else ckpt.train_loss
    avg_loss_val: float = 0 if ckpt is None else ckpt.val_loss
    metric_val_hist: list[float] = []

    # Paths
    checkpoint_dir = checkpoints_loc()
    checkpoint_dir.mkdir(exist_ok=True)

    path_to_checkpoint: Path = checkpoint_dir / Path("latest_checkpoint.tar")
    path_to_model: Path = checkpoint_dir / Path(f"checkpoint_{a_cfg.backbone.name}.pth")
    progress_bar, train_task, val_task, epoch_task = setup_training_progress(
        a_cfg, avg_loss_train, avg_loss_val, core_trainer, torch.cuda.get_device_name()
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
            train_status = _check_model_improvement(
                metric_val_hist, epoch_metric, a_cfg, best_val_metric, epoch
            )
            if train_status is None:
                break
            # -- Save best checkpoint, if metric is better ---------------------------------------
            if train_status.save_best:
                best_val_metric = train_status.best_val
                # convert best_val_metric form of 5 numbers, in scientific notation
                # create file with name that is unique to evaluation
                best_model_name: str = (
                    path_to_model.name.removesuffix(".pth") + f"{best_val_metric:3e}" + ".pth"
                )
                path_to_model_detail = path_to_model.parent / Path(best_model_name)

                # check if files in directory has potential amount of
                # files to reach limit before loop.
                remove_oldest_checkpoint(path_to_model_detail)
                checkpoint = Checkpoint.from_epoch(
                    epoch, core_trainer, labels_tensor, a_cfg, best_val_metric
                )
                checkpoint.torch_save(path_to_model_detail)

            # Always save latest model, after going through both loaders
            model_checkpoint = ModelCheckpoint.from_epoch(
                epoch,
                core_trainer,
                labels_tensor,
                a_cfg,
                best_val_metric,
                core_trainer.optimizer.state_dict,
                avg_loss_train,
                avg_loss_val,
            )
            model_checkpoint.torch_save(path_to_model_detail)

            progress_bar.update(
                epoch_task,
                advance=1,
                train_loss=avg_loss_train,
                val_loss=avg_loss_val,
                val_err=epoch_metric.metric_val,
                lr=core_trainer.optimizer.param_groups[0]["lr"],
            )

    return TrainingOutput(avg_loss_train, avg_loss_val, epoch_metric.metric_val, best_val_metric)


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
        bin_centers = ckpt.checkpoint.bin_centers
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
