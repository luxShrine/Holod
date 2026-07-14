from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import click
import numpy as np
import numpy.typing as npt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from serde import Untagged, serde
from serde.toml import from_toml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from torchvision.transforms import v2

import holod.infra.util.paths as paths
from holod.infra.dataset import HologramFocusDataset, RegLabelTransform, TransformedDataset
from holod.infra.log import get_logger
from holod.infra.losses import SoftOrdinalCrossEntropy
from holod.infra.models import FocusNetTorch, NeuralNetwork
from holod.infra.util.types import (
    Q_,
    AnalysisType,
    Arr64,
    CreateCSVOption,
    Mean,
    ModelType,
    StandardDev,
    StateDict,
    SubsetList,
    UserDevice,
    u,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as ImageType
    from pint.facets.plain.quantity import PlainQuantity
    from torch.nn import Module
    from torch.optim import Optimizer


logger = get_logger(__name__)

HOLO_DEF = paths.data_spec("mw")
SAMPLE_PATH: Path = paths.data_root() / Path("MW_Dataset_Sample") / Path("ODP-DLHM-Database.csv")
type FieldsUnion = Paths | Train | Flags

# --- Helpers


def check_csv_exists(create_csv: bool, ds_root: Path, meta_csv_name: str) -> tuple[bool, str]:
    """Validate the dataset root and its metadata CSV, prompting the user as needed.

    ``ds_root`` must already be resolved (see ``paths.resolve_dataset_root``), so
    dataset folders outside ``src/data`` work the same as bundled ones.

    Returns:
        ``(use_sample_data, meta_csv_name)`` where ``meta_csv_name`` reflects any
        CSV created during the check.

    """
    use_sample_data = False
    # parse user input and whether csv or data exists
    create_csv_option: CreateCSVOption
    if create_csv:
        create_csv_option = CreateCSVOption.CSV_CREATE
    elif meta_csv_name and (ds_root / meta_csv_name).is_file():
        create_csv_option = CreateCSVOption.CSV_VALID
    elif ds_root.is_dir():
        create_csv_option = CreateCSVOption.CSV_MISSING
    else:
        create_csv_option = CreateCSVOption.CSV_DATA_MISSING

    match create_csv_option:
        case CreateCSVOption.CSV_DATA_MISSING:
            # with csv missing see if user wants to use sample data, if not abort
            logger.debug(f"Could not find dataset folder {ds_root}.")
            use_sample_data = click.confirm(
                f"Could not find dataset folder {ds_root}. Use sample data instead?",
                default=True,
                abort=True,
            )
        case CreateCSVOption.CSV_MISSING:
            logger.debug(f"Could not find {meta_csv_name} in root of dataset folder {ds_root}.")
            user_response: str = click.prompt(
                f"Could not find '{meta_csv_name}' in root of dataset folder: '{ds_root}'. "
                + "Attempt to create csv file from ds_root or use sample data?",
                show_choices=True,
                type=click.Choice(["sample", "create"], case_sensitive=False),
            )
            if user_response.lower() == "create":
                # run it through again with create csv option
                return check_csv_exists(True, ds_root, meta_csv_name)
            if user_response.lower() == "sample":
                use_sample_data = True
            else:
                raise RuntimeError("failed to parse user choice for creating CSV.")

        case CreateCSVOption.CSV_VALID:
            logger.info(f"Path to csv exists {meta_csv_name}, continuing...")

        case CreateCSVOption.CSV_CREATE:
            if not ds_root.is_dir():
                raise click.ClickException(
                    f"Cannot create a CSV: dataset folder does not exist at {ds_root}"
                )
            # parse the create input here as to allow for the user to select the desire to
            # create a csv if the data is missing
            csv_name: str = click.prompt(
                "Enter a filename for csv",
                type=click.STRING,
                default=meta_csv_name or "ODP-DLHM-Database.csv",
            )
            HologramFocusDataset.create_meta(ds_root, csv_name)
            meta_csv_name = csv_name

    return (use_sample_data, meta_csv_name)


# ----


# TODO: potentially compress some of these fields to utilize the flags, paths, and train classes
@dataclass
class AutoConfig:
    """Class for storing torch options in the autofocus functions.

    Attributes:
        num_workers: How many data loading subprocesess to use in parallel.
        batch_size: How many images to process per epoch.
        val_split: Value split between prediction training and validation,
        leftover percentage is given to testing.

        crop_size: Length/width to crop the image to.
        opt_lr: The optimizers defined learning rate.
        opt_weight_decay: The optimizers defined learning rate.
        sch_factor: Amount to reduce the learning rate upon a plateau of improvement.
        sch_patience: Number of epochs considered a plateau for the scheduler to
        reduce the learning rate.

        num_classes: Number of classifications that can be made (only one for regression).
        backbone: The model itself.
        auto_method: Classification or regression training for the model.
        soft_label_sigma: Std dev (in bins) of the soft ordinal target distribution
        for classification; 0 keeps hard one-hot labels.

    """

    batch_size: int = 16
    crop_size: int = 224
    epoch_count: int = 10
    num_classes: int = 1
    num_workers: int = 2
    val_split: float = 0.2
    analysis: AnalysisType = AnalysisType.CLASS
    backbone: ModelType = ModelType.ENET
    device_user: UserDevice = UserDevice.CUDA
    meta_csv_strpath: str = (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix()
    opt_lr: float = 5e-5
    opt_weight_decay: float = 1e-2
    sch_factor: float = 0.1
    sch_patience: int = 5
    soft_label_sigma: float = 0.0
    fixed_seed: bool = True
    checkpoint: bool = False

    # Use default_factory for mutable fields and populate in __post_init__
    data: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate dictionary fields after initialization."""
        self.data.update(
            {
                "analysis": self.analysis,
                "backbone": self.backbone,
                "batch_size": self.batch_size,
                "crop_size": self.crop_size,
                "meta_csv_strpath": self.meta_csv_strpath,
                "num_workers": self.num_workers,
                "num_classes": self.num_classes,
                "sch_factor": self.sch_factor,
                "sch_patience": self.sch_patience,
                "soft_label_sigma": self.soft_label_sigma,
                "val_split": self.val_split,
            }
        )
        self.optimizer.update(
            {
                "opt_lr": self.opt_lr,
                "opt_weight_decay": self.opt_weight_decay,
                "sch_factor": self.sch_factor,
                "sch_patience": self.sch_patience,
            }
        )

    def setup_loader_indices(self, base: HologramFocusDataset) -> SubsetList:
        """Create random split of training and evaluation loader indices."""
        num_labels: int = len(base)
        eval_len: int = int(self.val_split * num_labels)
        train_len: int = num_labels - eval_len
        if self.fixed_seed:
            generator = torch.Generator().manual_seed(42)
            return random_split(base, [train_len, eval_len], generator)
        return random_split(base, [train_len, eval_len])

    def device(self) -> Literal["cuda", "cpu"]:
        """Return the device to use in autofocus training."""
        actual_device: Literal["cuda", "cpu"] = "cpu"  # Default to CPU
        if self.device_user == UserDevice.CUDA:
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                logger.warning("CUDA specified but not available, using CPU instead.")
        logger.debug(f"Using device: {actual_device}")
        return actual_device

    def to_user_config(self) -> ModelConfig:
        """Convert from autoconfig to ModelConfig."""
        train = Train(
            self.backbone.value,
            self.opt_lr,
            self.opt_weight_decay,
            self.sch_factor,
            self.sch_patience,
        )
        return ModelConfig(train)

    @classmethod
    def default(cls, analysis: AnalysisType, backbone: ModelType, num_classes: int):
        """Build a config with conservative defaults for a given backbone and analysis."""
        return cls(
            analysis=analysis,
            backbone=backbone,
            num_classes=num_classes,
            batch_size=8,
            crop_size=224,
            epoch_count=5,
            num_workers=2,
            val_split=0.2,
        )

    @classmethod
    def from_user(
        cls,
        backbone: str,
        batch_size: int,
        crop_size: int,
        device_user: str,
        num_classes: int,
        num_workers: int,
        val_split: float,
        epoch_count: int,
        meta_csv: str,
        checkpoint: bool | None = None,
        fixed_seed: bool | None = None,
        opt_weight_decay: float | None = None,
        opt_lr: float | None = None,
        sch_factor: float | None = None,
        sch_patience: int | None = None,
        soft_label_sigma: float | None = None,
    ) -> AutoConfig:
        """Parse user input to ensure proper arguments are recieved."""
        analysis = AnalysisType.CLASS if num_classes != 1 else AnalysisType.REG
        logger.info(f"Training type: {analysis}.")

        backbone_type = ModelType.from_str(backbone) if backbone is not None else ModelType.ENET
        # WARN: not robust for all types, will need a change <05-09-25>
        if backbone_type == ModelType.VIT and crop_size != 224:
            logger.error(
                f"backbone of type {backbone_type} requires a crop size of 224,"
                + "defaulting to appropriate crop size"
            )
            crop_size = 224

        # ensure not None; fall back to the class-level defaults
        checkpoint = checkpoint if checkpoint is not None else False
        fixed_seed = fixed_seed if fixed_seed is not None else True
        opt_lr = opt_lr if opt_lr is not None else 5e-5
        opt_weight_decay = opt_weight_decay if opt_weight_decay is not None else 1e-2
        # ReduceLROnPlateau requires factor < 1.0
        sch_factor = sch_factor if sch_factor is not None else 0.1
        sch_patience = sch_patience if sch_patience is not None else 5
        soft_label_sigma = soft_label_sigma if soft_label_sigma is not None else 0.0
        if soft_label_sigma < 0:
            raise ValueError(f"soft_label_sigma must be >= 0, got {soft_label_sigma}.")

        return cls(
            analysis=analysis,
            backbone=backbone_type,
            batch_size=batch_size,
            checkpoint=checkpoint,
            crop_size=crop_size,
            device_user=UserDevice.determine(device_user),
            epoch_count=epoch_count,
            fixed_seed=fixed_seed,
            meta_csv_strpath=meta_csv,
            num_classes=num_classes,
            opt_lr=opt_lr,
            val_split=val_split,
            num_workers=num_workers,
            opt_weight_decay=opt_weight_decay,
            sch_factor=sch_factor,
            sch_patience=sch_patience,
            soft_label_sigma=soft_label_sigma,
        )


@serde
class Paths:
    """Dataset location settings shared by every configured model."""

    dataset_root: str = ""
    meta_csv_name: str = ""

    @classmethod
    def empty(cls):
        """Return a Paths instance with both fields unset (never overrides on merge)."""
        return cls(dataset_root="", meta_csv_name="")


@serde
class Train:
    """Per-model training settings (optimizer and scheduler)."""

    backbone: str
    learning_rate: float | None = None
    optimizer_weight_decay: float | None = None
    sch_factor: float | None = None
    sch_patience: int | None = None


@serde
class Flags:
    """Boolean toggles shared by every configured model; ``None`` means unset."""

    checkpoint: bool | None = None
    create_csv: bool | None = None
    fixed_seed: bool | None = None
    sample: bool | None = None

    @classmethod
    def empty(cls):
        """Return a Flags instance with every flag unset (never overrides on merge)."""
        return cls(checkpoint=None, create_csv=None, fixed_seed=None, sample=None)


@serde(tagging=Untagged)
class ModelConfig:
    """Dataclass that deserializes the config file via serde."""

    train: Train

    @override
    def __repr__(self):
        """Return the proprerly formatted string representation of config."""
        return f"ConfigFile('{self.train}')"


@serde(tagging=Untagged)
class CompareUserConfig:
    """Dataclass that deserializes a per-model config file via serde."""

    paths: Paths
    flags: Flags
    batch_size: int = 16
    crop_size: int = 224
    device: str = "cuda"
    epoch_count: int = 10
    num_classes: int = 10
    num_workers: int = 2
    val_split: float = 0.2
    soft_label_sigma: float = 0.0
    enet: ModelConfig | None = None
    focusnet: ModelConfig | None = None
    pcnn: ModelConfig | None = None
    resnet: ModelConfig | None = None
    vit: ModelConfig | None = None

    def __post_init__(self):
        """Ensure that at least one model is configured."""
        if not self.configured_backbones():
            raise Exception("No model is configured! Must have at least one backbone configured.")
        self._paths_resolved: bool = False

    def resolve_paths(self) -> CompareUserConfig:
        """Resolve the dataset root and metadata CSV into absolute paths.

        Enforced automatically at the end of ``merge`` and at the start of
        ``to_auto_config``, so a config can never reach dataset loading or training
        with unresolved paths. Validates that the dataset and CSV exist, prompting
        the user as needed, and rewrites ``paths`` in-place to the resolved absolute
        locations. Idempotent: repeated calls are no-ops.
        """
        if self._paths_resolved:
            return self

        create_csv = False if self.flags.create_csv is None else self.flags.create_csv
        meta_csv = self.paths.meta_csv_name
        # resolve the dataset root so folders outside src/data (absolute, ``~``, or
        # cwd-relative paths) load the same way bundled datasets do
        ds_root_path = paths.resolve_dataset_root(self.paths.dataset_root)
        use_sample_data = self.flags.sample
        if not self.flags.sample:
            (use_sample_data, meta_csv) = check_csv_exists(create_csv, ds_root_path, meta_csv)
        # ensure the user has the sample data
        meta_csv_path: Path
        if use_sample_data:
            logger.warning(f"Ensuring {SAMPLE_PATH} exists...")
            if SAMPLE_PATH.exists():
                meta_csv_path = SAMPLE_PATH
                logger.info(f"{meta_csv_path} exists, continuing with sample data...")
            else:
                raise Exception(f"Path to sample data does not exist: {SAMPLE_PATH}")
        else:
            meta_csv_path = ds_root_path / meta_csv
        self.paths.dataset_root = ds_root_path.as_posix()
        self.paths.meta_csv_name = meta_csv_path.as_posix()
        self._paths_resolved = True
        return self

    def _merge_group(self, target: FieldsUnion, source: FieldsUnion) -> None:
        for f in fields(source):
            name: str = f.name
            value: Any = getattr(source, name)
            # empty strings mean "unset" for Paths fields; never clobber with them
            if value is not None and value != "":
                setattr(target, name, value)

    def merge(
        self,
        flags: Flags | None = None,
        paths: Paths | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        crop_size: int | None = None,
        epoch_count: int | None = None,
        num_classes: int | None = None,
        num_workers: int | None = None,
        val_split: float | None = None,
        soft_label_sigma: float | None = None,
    ):
        """Merge commandline inputs with an existing user config instance."""
        if batch_size is not None:
            self.batch_size = batch_size
        if device is not None:
            self.device = device
        if crop_size is not None:
            self.crop_size = crop_size
        if epoch_count is not None:
            self.epoch_count = epoch_count
        if num_classes is not None:
            self.num_classes = num_classes
        if num_workers is not None:
            self.num_workers = num_workers
        if val_split is not None:
            self.val_split = val_split
        if soft_label_sigma is not None:
            self.soft_label_sigma = soft_label_sigma
        if paths is not None:
            self._merge_group(self.paths, paths)
        if flags is not None:
            self._merge_group(self.flags, flags)
        # resolve here so no caller can use a merged config with unresolved paths
        return self.resolve_paths()

    def model_config(self, backbone: ModelType) -> ModelConfig | None:
        """Return the per-model config for a backbone, or ``None`` if unconfigured."""
        match backbone:
            case ModelType.ENET:
                return self.enet
            case ModelType.FOCUSNET:
                return self.focusnet
            case ModelType.PCNN:
                return self.pcnn
            case ModelType.RESNET:
                return self.resnet
            case ModelType.VIT:
                return self.vit

    def configured_backbones(self) -> list[ModelType]:
        """Return the backbones that have a per-model config in this file."""
        return [m for m in ModelType if self.model_config(m) is not None]

    @staticmethod
    def from_toml(toml: Path):
        """Deserialize a config instance from a TOML file (see train_settings.toml)."""
        with open(toml) as config_file:
            return from_toml(CompareUserConfig, config_file.read())

    @classmethod
    def from_model_config(cls, model_config: ModelConfig):
        """Build a config around a single model, with paths and flags left unset."""
        backbone = ModelType.from_str(model_config.train.backbone)
        match backbone:
            case ModelType.ENET:
                return cls(Paths.empty(), Flags.empty(), enet=model_config)
            case ModelType.FOCUSNET:
                return cls(Paths.empty(), Flags.empty(), focusnet=model_config)
            case ModelType.PCNN:
                return cls(Paths.empty(), Flags.empty(), pcnn=model_config)
            case ModelType.RESNET:
                return cls(Paths.empty(), Flags.empty(), resnet=model_config)
            case ModelType.VIT:
                return cls(Paths.empty(), Flags.empty(), vit=model_config)

    def to_auto_config(self, backbone: ModelType) -> AutoConfig:
        """Build the runtime ``AutoConfig`` for one configured backbone.

        Resolves paths first (a no-op if ``merge`` already did), so the returned
        config always points at a validated dataset.
        """
        out = self.resolve_paths()
        t_cfg = out.model_config(backbone)
        if t_cfg is None:
            raise KeyError(
                f"Selected backbone: {backbone.value} is not configured, "
                f"available backbones: {[m.value for m in out.configured_backbones()]}"
            )
        return AutoConfig.from_user(
            batch_size=out.batch_size,
            crop_size=out.crop_size,
            device_user=out.device,
            epoch_count=out.epoch_count,
            num_classes=out.num_classes,
            num_workers=out.num_workers,
            val_split=out.val_split,
            soft_label_sigma=out.soft_label_sigma,
            # paths & flags
            meta_csv=out.paths.meta_csv_name,
            checkpoint=out.flags.checkpoint,
            fixed_seed=out.flags.fixed_seed,
            # train cfg
            backbone=t_cfg.train.backbone,
            opt_lr=t_cfg.train.learning_rate,
            opt_weight_decay=t_cfg.train.optimizer_weight_decay,
            sch_factor=t_cfg.train.sch_factor,
            sch_patience=t_cfg.train.sch_patience,
        )


@dataclass
class CoreTrainer:
    """Class to hold all the training information."""

    a_cfg: AutoConfig
    evaluation_metric: npt.NDArray[np.float64] | npt.NDArray[np.intp]
    bin_centers: npt.NDArray[np.float64] | None
    model: Module
    loss_fn: Any
    optimizer: Optimizer
    scheduler: Any
    train_ds: Dataset[tuple[ImageType, np.float64]]
    train_loader: DataLoader[tuple[ImageType, np.float64]]
    val_ds: Dataset[tuple[ImageType, np.float64]]
    val_loader: DataLoader[tuple[ImageType, np.float64]]
    z_std: PlainQuantity[float]
    z_avg: PlainQuantity[float]

    @property
    def device(self):
        return self.a_cfg.device()

    @property
    def analysis(self):
        return self.a_cfg.analysis

    @property
    def backbone(self):
        return self.a_cfg.backbone

    def get_std_avg(self, analysis: AnalysisType) -> tuple[Mean, StandardDev] | tuple[None, None]:
        """Return the standard deviation and average if necessitated by analysis."""
        match analysis:
            case AnalysisType.REG:
                return (Mean(self.z_avg.magnitude), StandardDev(self.z_std.magnitude))
            case _:
                return (None, None)

    def denormalize_tens(self, pred: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Denormalize tensors for use after regression training."""
        (z_avg, z_std) = self.get_std_avg(AnalysisType.REG)
        if z_avg and z_std is not None:
            try:
                # reverse the transformation
                z_pred: Tensor = (pred * z_std) + z_avg
                z_true: Tensor = (labels * z_std) + z_avg
                return (z_true, z_pred)
            except Exception as e:
                raise e
        else:
            raise Exception("z_avg, and z_std need to be used but are None.")

    def check_loss(self, labels_tens: Tensor, pred: Tensor):
        # TODO: this check is done every time, probably should set this once as an operation
        # to perform and reuse it
        if isinstance(self.loss_fn, nn.MSELoss):
            labels_tens = labels_tens.float()  # expects float
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            labels_tens = labels_tens.float()  # BCE expects float 0/1
        elif isinstance(self.loss_fn, nn.CrossEntropyLoss | SoftOrdinalCrossEntropy):
            # SoftOrdinalCrossEntropy also takes hard indices; it softens them internally
            labels_tens = labels_tens.long()
            # fail fast with a readable error; a mismatch here otherwise surfaces as an
            # opaque CUDA device-side assert in backward()
            num_classes = self.a_cfg.num_classes
            if pred.ndim != 2 or pred.shape[1] != num_classes:
                raise ValueError(
                    f"Model '{self.backbone.value}' output shape {tuple(pred.shape)} is "
                    f"incompatible with CrossEntropyLoss over {num_classes} classes; "
                    "expected (batch, num_classes)."
                )
            if labels_tens.numel() > 0 and (
                int(labels_tens.min()) < 0 or int(labels_tens.max()) >= num_classes
            ):
                raise ValueError(
                    f"Class labels outside [0, {num_classes}): "
                    f"min {int(labels_tens.min())}, max {int(labels_tens.max())}."
                )
        else:
            raise Exception(f"Loss function is unknown: {self.loss_fn}")

        return labels_tens


@dataclass
class EpochMetric:
    """Output of one epoch of autofocus training."""

    loss_epoch: float = 0
    metric_val: float = 0
    total_samples: int = 0
    labels_tensor: Tensor = torch.empty([1, 1])
    # MAE between predicted and true bin centers, in the dataset's z units (mm);
    # classification only   for regression metric_val is already the physical MAE
    mae_mm: float | None = None

    def average_loss(self) -> float:
        """Compute the average loss of an epoch."""
        try:
            # might be zero
            return self.loss_epoch / self.total_samples
        except Exception as e:
            logger.exception(f"Error computing average epoch loss: {e}")
            return self.loss_epoch / 1


@dataclass
class TrainingOutput:
    """Final metrics of all training outputs."""

    avg_train_loss: float
    avg_val_loss: float
    evaluation_metric: float
    best_val_metric: float
    # final-epoch validation MAE in bin-center distance (mm); classification only
    val_mae_mm: float | None = None

    def display_loss(self) -> None:
        """Display the relevant metrics generated after training."""
        logger.info(f"Evaluation metric: {self.evaluation_metric}")
        logger.info(f"Average loss for evaluation: {self.avg_val_loss}")
        logger.info(f"Average loss for training: {self.avg_train_loss}")


@serde
class TrainingRepeatConfig:
    """Stores the relevant inforamtion concerning the report for training."""

    user_config: ModelConfig
    avg_train_loss: float
    avg_val_loss: float


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
        val_metric: float,
    ) -> Checkpoint:
        """Build a checkpoint from the current epoch's training state."""
        (z_avg, z_std) = core_trainer.get_std_avg(core_trainer.a_cfg.analysis)
        return cls(
            epoch=epoch,
            model_state_dict=core_trainer.model.state_dict(),
            labels=labels_tensor,
            val_metric=val_metric,
            bin_centers=core_trainer.bin_centers,
            num_classes=core_trainer.a_cfg.num_classes,
            model_type=core_trainer.a_cfg.backbone,
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
        val_metric: float,
        optimizer_state_dict: StateDict,
        train_loss: float,
        val_loss: float,
    ) -> ModelCheckpoint:
        # Awlays save latest model, after going through both loaders
        checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, val_metric)
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


def create_autofocus_model(a_cfg: AutoConfig, quiet: bool = False) -> nn.Module:
    """Create and configure the model."""
    model: nn.Module
    num_model_outputs = 1 if a_cfg.analysis == AnalysisType.REG else a_cfg.num_classes
    match a_cfg.backbone:
        case ModelType.PCNN:
            model = NeuralNetwork(1, num_model_outputs)
        case ModelType.ENET:
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_model_outputs)  # pyright: ignore[reportArgumentType]
        case ModelType.RESNET:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_model_outputs)
        case ModelType.VIT:
            model = timm.create_model(
                "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                pretrained=True,
                num_classes=num_model_outputs,
            )
        case ModelType.FOCUSNET:
            model = FocusNetTorch(num_classes=num_model_outputs)
    if not quiet:
        logger.info(
            f"Model '{a_cfg.backbone.value}' configured with {num_model_outputs} output "
            + f"features for {a_cfg.analysis.value} analysis."
        )
    return model


def class_label_transform(z_raw_idx: np.int64) -> Tensor:
    """Convert a bin index into a long tensor."""
    return torch.as_tensor(z_raw_idx, dtype=torch.long)


def create_training_setup(
    base: HologramFocusDataset, a_cfg: AutoConfig, quiet: bool = False
) -> CoreTrainer:
    """Split the dataset, apply transforms and build dataloaders."""
    (train_subset, eval_subset) = a_cfg.setup_loader_indices(base)
    logger.debug("Train and evaluation subset created successfully")
    # Calculate avg and std from the training subset's physical z-values
    # Need to access original z_m from base dataset via indices of train_subset
    (z_avg, z_std) = base.z.subset_mean_std(subset_indices=train_subset.indices)

    # Define label transforms based on analysis type (handles normalization for REG)
    evaluation_metric: npt.NDArray[np.float64] | npt.NDArray[np.intp]
    if a_cfg.analysis == AnalysisType.REG:
        # True physical Zs for validation
        evaluation_metric = base.z.z_array[eval_subset.indices]

        # normalize z values to create proper label
        final_label_transform = v2.Lambda(RegLabelTransform(z_avg=float(z_avg), z_std=float(z_std)))
    else:
        # Physical values of bin centers
        evaluation_metric = base.z_bins[eval_subset.indices]

        # simply convert to tensor
        final_label_transform = v2.Lambda(class_label_transform)

    (train_subset, eval_subset) = TransformedDataset.from_subset(
        eval_subset=eval_subset,
        train_subset=train_subset,
        crop_size=a_cfg.crop_size,
        final_label_transform=final_label_transform,
        model=a_cfg.backbone,
    )

    # dataset needs to be iterable in terms of pytorch, dataloader does such
    eval_dl: DataLoader[tuple[ImageType, np.float64]] = DataLoader(
        eval_subset,
        batch_size=a_cfg.batch_size,
        drop_last=False,
        num_workers=a_cfg.num_workers,
        pin_memory=(a_cfg.device() == "cuda"),
        prefetch_factor=2 if a_cfg.num_workers > 0 else None,
        shuffle=False,
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
    loss_fn: nn.Module
    if a_cfg.analysis == AnalysisType.CLASS:
        if a_cfg.soft_label_sigma > 0:
            # soft ordinal targets (SORD): near-miss bins keep some probability
            # mass, so the loss respects the depth ordering of the bins
            loss_fn = SoftOrdinalCrossEntropy(a_cfg.num_classes, a_cfg.soft_label_sigma)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    model = create_autofocus_model(a_cfg, quiet)

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
        a_cfg=a_cfg,
        evaluation_metric=evaluation_metric,  # This is Arr64 of z_m or bin_centers_m
        bin_centers=base.bin_centers if a_cfg.analysis == AnalysisType.CLASS else None,
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
    if not quiet:
        logger.info("Training Setup Complete")
    return core_trainer_config
