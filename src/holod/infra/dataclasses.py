from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, override

import click
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL.Image import Image as ImageType
from serde import Untagged, serde
from timm import create_model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from torchvision.transforms import v2

import holod.infra.util.paths as paths
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import get_logger
from holod.infra.util.types import (
    Q_,
    AnalysisType,
    CreateCSVOption,
    Mean,
    ModelType,
    StandardDev,
    SubsetImageDepth,
    SubsetList,
    UserDevice,
    u,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from pint.facets.plain.quantity import PlainQuantity
    from torch.nn import Module
    from torch.optim import Optimizer


logger = get_logger(__name__)

HOLO_DEF = paths.data_spec("mw")
# TODO: add to git this default path with sample data
SAMPLE_PATH: Path = paths.data_root() / Path("MW_Dataset_Sample") / Path("ODP-DLHM-Database.csv")
type FieldsUnion = Paths | Train | Flags


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

    """

    analysis: AnalysisType = AnalysisType.CLASS
    backbone: ModelType = ModelType.ENET
    batch_size: int = 16
    crop_size: int = 224
    device_user: UserDevice = UserDevice.CUDA
    epoch_count: int = 10
    meta_csv_strpath: str = (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix()
    num_classes: int = 1
    num_workers: int = 2
    opt_lr: float = 5e-5
    opt_weight_decay: float = 1e-2
    sch_factor: float = 0.1
    sch_patience: int = 5
    val_split: float = 0.2
    fixed_seed: bool = True
    checkpoint: bool = False

    # Use default_factory for mutable fields and populate in __post_init__
    data: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate dictionary fields after initialization."""
        logger.info(f"Training type: {self.analysis}.")
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

    def create_model(self) -> nn.Module:
        """Create and configure the model."""
        model: nn.Module
        num_model_outputs = 1 if self.analysis == AnalysisType.REG else self.num_classes
        match self.backbone:
            case ModelType.NEW:
                model = NeuralNetwork(1, num_classes=num_model_outputs)
            case ModelType.ENET:
                model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_model_outputs)
            case ModelType.RESNET:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_model_outputs)
            case ModelType.VIT:
                model = create_model(
                    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                    pretrained=True,
                    num_classes=num_model_outputs,
                )
            case ModelType.FOCUSNET:
                model = FocusNetTorch(num_classes=num_model_outputs)
            case _:
                raise Exception(f"Could not create model specified: {self.backbone}")  # pyright: ignore[reportUnreachable]
        logger.info(
            f"Model '{self.backbone.value}' configured with {num_model_outputs} output "
            + f"features for {self.analysis.value} analysis."
        )
        return model

    def to_user_config(self) -> UserConfig:
        """Convert from autoconfig to UserConfig."""
        paths = Paths(
            Path(self.meta_csv_strpath).parent.as_posix(), Path(self.meta_csv_strpath).stem
        )
        train = Train(
            self.backbone.value,
            self.batch_size,
            self.crop_size,
            self.device(),
            self.epoch_count,
            self.opt_lr,
            self.num_classes,
            self.num_workers,
            self.opt_weight_decay,
            self.sch_factor,
            self.sch_patience,
            self.val_split,
        )
        flags = Flags(
            self.checkpoint,
            False,  # irrelevant to repeating
            self.fixed_seed,
            (self.meta_csv_strpath == SAMPLE_PATH.as_posix()),  # if path == sample path
        )
        return UserConfig(paths, train, flags)

    @classmethod
    def from_user(
        cls,
        backbone: str | None,
        batch_size: int | None,
        crop_size: int | None,
        checkpoint: bool | None,
        ds_root: str | None,
        device_user: str | None,
        fixed_seed: bool | None,
        meta_csv_name: str | None,
        num_classes: int | None,
        num_workers: int | None,
        opt_weight_decay: float | None,
        val_split: float | None,
        epoch_count: int | None,
        opt_lr: float | None,
        create_csv: bool | None,
        use_sample_data: bool | None,
        sch_factor: float | None,
        sch_patience: int | None,
    ) -> AutoConfig:
        """Parse user input to ensure proper arguments are recieved."""
        if backbone is None:
            logger.error("No backbone passed, using efficientnet.")
            backbone = "Enet"

        backbone = backbone.lower()
        if "vit" in backbone:
            backbone_type = ModelType.VIT
        elif "enet" in backbone or "efficient" in backbone:
            backbone_type = ModelType.ENET
        elif "res" in backbone:
            backbone_type = ModelType.RESNET
        elif "new" in backbone or "custom" in backbone:
            backbone_type = ModelType.NEW
        elif "focus" in backbone:
            backbone_type = ModelType.FOCUSNET
        else:
            raise Exception(f"Unknown backbone passed: {backbone}")

        # parse user input and whether csv or data exists
        create_csv_option: CreateCSVOption
        if create_csv:
            create_csv_option = CreateCSVOption.CSV_CREATE
        elif (paths.data_root() / Path(ds_root) / Path(meta_csv_name)).exists():
            create_csv_option = CreateCSVOption.CSV_VALID
        elif (paths.data_root() / Path(ds_root)).exists():
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
                    create_csv_option = CreateCSVOption.CSV_CREATE
                elif user_response.lower() == "sample":
                    use_sample_data = True
                else:
                    raise RuntimeError("failed to parse user choice for creating CSV.")

            case CreateCSVOption.CSV_VALID:
                logger.info(f"Path to csv exists {meta_csv_name}, continuing...")

        # parse the create input here as to allow for the user to select the desire to
        # create a csv if the data is missing
        if create_csv_option == CreateCSVOption.CSV_CREATE:
            csv_name: str = click.prompt("Enter a filename for csv", type=click.STRING)
            HologramFocusDataset.create_meta(Path(ds_root), csv_name)

        # WARN: not robust for all types, will need a change <05-09-25>
        if backbone_type == ModelType.VIT and crop_size != 224:
            logger.error(
                f"backbone of type {backbone_type} requires a crop size of 224,"
                + "defaulting to appropriate crop size"
            )
            crop_size = 224

        # ensure the user has the sample data
        if use_sample_data:
            logger.warning(f"Ensuring {SAMPLE_PATH} exists...")
            if SAMPLE_PATH.exists():
                meta_csv_strpath: str = SAMPLE_PATH.as_posix()
                logger.info(f"{meta_csv_strpath} exists, continuing with sample data...")
            else:
                raise Exception(f"Path to sample data does not exist: {SAMPLE_PATH}")
        else:
            meta_csv_strpath = (paths.data_root() / ds_root / Path(meta_csv_name)).as_posix()

        # ensure not None
        backbone_type = backbone_type if backbone_type is not None else ModelType.ENET
        batch_size = batch_size if batch_size is not None else 16
        checkpoint = checkpoint if checkpoint is not None else False
        crop_size = crop_size if crop_size is not None else 224
        epoch_count = epoch_count if epoch_count is not None else 10
        fixed_seed = fixed_seed if fixed_seed is not None else True
        meta_csv_strpath = (
            meta_csv_strpath
            if meta_csv_strpath is not None
            else (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix()
        )
        num_classes = num_classes if num_classes is not None else 10
        opt_lr = opt_lr if opt_lr is not None else 5e-5
        val_split = val_split if val_split is not None else 0.2
        num_workers = num_workers if num_workers is not None else 1
        opt_weight_decay = opt_weight_decay if opt_weight_decay is not None else 0.1
        sch_factor = sch_factor if sch_factor is not None else 5
        sch_patience = sch_patience if sch_patience is not None else 5

        return cls(
            analysis=AnalysisType.CLASS if num_classes != 1 else AnalysisType.REG,
            backbone=backbone_type,
            batch_size=batch_size,
            checkpoint=checkpoint,
            crop_size=crop_size,
            device_user=UserDevice(device_user),
            epoch_count=epoch_count,
            fixed_seed=fixed_seed,
            meta_csv_strpath=meta_csv_strpath,
            num_classes=num_classes,
            opt_lr=opt_lr,
            val_split=val_split,
            num_workers=num_workers,
            opt_weight_decay=opt_weight_decay,
            sch_factor=sch_factor,
            sch_patience=sch_patience,
        )


class NeuralNetwork(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        # TODO: write out the matrix size for each step <luxShrine>
        super().__init__()
        # conv2d (ks=3, s=1) -> (N, OC, h - 2, w - 2)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # avg pool to 1x1, allowing for size agnostic images passed in
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Any -> (1, 1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> (N, 32, h-2, w-2)
        x = F.relu(self.conv2(x))  # -> (N, 64, h-4, w-4)
        x = self.pool(x)  # -> (N, 64, (h-4)/2, (h-4)/2)
        x = self.dropout1(x)
        x = self.gap(x)  # -> (N, 64, 1, 1)
        x = torch.flatten(x, 1)  #  -> (N, 64 * 1 * 1)
        x = F.relu(self.fc1(x))  # -> (N, 128)
        x = self.dropout2(x)
        x = self.fc2(x)  # -> (N, num_classes)
        return F.log_softmax(x, dim=1)


@serde
class Paths:
    dataset_root: str | None
    meta_csv_name: str | None


@serde
class Train:
    backbone: str | None
    batch_size: int | None
    crop_size: int | None
    device: str | None
    epoch_count: int | None
    learning_rate: float | None
    num_classes: int | None
    num_workers: int | None
    optimizer_weight_decay: float | None
    sch_factor: float | None
    sch_patience: int | None
    val_split: float | None


@serde
class Flags:
    checkpoint: bool | None
    create_csv: bool | None
    fixed_seed: bool | None
    sample: bool | None


@serde(tagging=Untagged)
class UserConfig:
    """Dataclass that deserializes the config file via serde."""

    paths: Paths
    train: Train
    flags: Flags

    @override
    def __repr__(self):
        """Return the proprerly formatted string representation of config."""
        return f"ConfigFile('{self.paths}'\n'{self.train}'\n'{self.flags}')"

    def merge(self, paths: Paths | None, train: Train | None, flags: Flags | None) -> None:
        """Merge commandline inputs with an existing user config instance."""
        # check to see if any arguments have been passed
        if paths is not None:
            self._merge_group(self.paths, paths)
        if train is not None:
            self._merge_group(self.train, train)
        if flags is not None:
            self._merge_group(self.flags, flags)

    def _merge_group(self, target: FieldsUnion, source: FieldsUnion) -> None:
        for f in fields(source):
            name: str = f.name
            value: Any = getattr(source, name)
            if value is not None:
                setattr(target, name, value)

    def to_auto_config(self) -> AutoConfig:
        return AutoConfig.from_user(
            backbone=self.train.backbone,
            batch_size=self.train.batch_size,
            checkpoint=self.flags.checkpoint,
            crop_size=self.train.crop_size,
            create_csv=self.flags.create_csv,
            device_user=self.train.device,
            ds_root=self.paths.dataset_root,
            epoch_count=self.train.epoch_count,
            fixed_seed=self.flags.fixed_seed,
            meta_csv_name=self.paths.meta_csv_name,
            num_classes=self.train.num_classes,
            num_workers=self.train.num_workers,
            opt_lr=self.train.learning_rate,
            opt_weight_decay=self.train.optimizer_weight_decay,
            sch_factor=self.train.sch_factor,
            sch_patience=self.train.sch_patience,
            use_sample_data=self.flags.sample,
            val_split=self.train.val_split,
        )


@dataclass
class CoreTrainer:
    """Class to hold specifically all the training information."""

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

    user_config: UserConfig
    avg_train_loss: float
    avg_val_loss: float


# ---------- focusnet torch implementation ----------


@torch.no_grad()
def fft_mag2d(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Compute |FFT2| with centered spectrum, per-sample, per-channel.

    Args:
        x: (N, C, H, W) real input (hologram).
        eps: small bias for numerical stability.

    Returns:
        (N, C, H, W) nonnegative magnitude.

    """
    # FFT expects complex; cast real -> complex
    x_c = torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1))
    f = torch.fft.fft2(x_c, dim=(-2, -1), norm="backward")
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.abs(f)
    if eps:
        mag = mag + eps
    return mag.real  # still real


def make_input_2ch(holod: torch.Tensor, use_fft: bool = True) -> torch.Tensor:
    """Build the 2‑channel input described in the repo's Fourier2D layer.

    Args:
        holod: (N, 1, H, W) real hologram.

    """
    assert holod.ndim == 4 and holod.size(1) == 1, "expect (N,1,H,W)"
    if use_fft:
        mag = fft_mag2d(holod)
        x = torch.cat([holod, mag], dim=1)  # -> (N, 2, H, W)
    else:
        x = torch.cat([holod, holod], dim=1)
    return x


# ---------- Model ----------


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FocusNetTorch(nn.Module):
    """Compact CNN for DLHM depth, inspired by the TF/Keras FocusNET repo.

    - Input: (N, 1, H, W) hologram; expanded to 2 channels [holod, |FFT(holod)|] internally.
    - Output: (N, num_classes) logits for classification, or (N, 1) normalized depth
      for regression (num_classes=1), matching the other backbones.
    """

    def __init__(
        self,
        num_classes: int = 1,
        width: int = 32,
        use_fft: bool = True,
        head: Literal["mlp", "linear"] = "mlp",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_fft = use_fft

        # Feature extractor; stem sees the 2-channel input built in forward()
        self.stem = ConvBlock(2, width, k=3)
        self.layer1 = nn.Sequential(
            ConvBlock(width, width, k=3),
            ConvBlock(width, width, k=3),
        )
        self.down1 = ConvBlock(width, width * 2, k=3, s=2)  # 128x128
        self.layer2 = nn.Sequential(
            ConvBlock(width * 2, width * 2, k=3),
            ConvBlock(width * 2, width * 2, k=3),
        )
        self.down2 = ConvBlock(width * 2, width * 4, k=3, s=2)  # 64x64
        self.layer3 = nn.Sequential(
            ConvBlock(width * 4, width * 4, k=3),
            ConvBlock(width * 4, width * 4, k=3),
        )

        # Global pooling + head
        self.pool = nn.AdaptiveAvgPool2d(1)
        emb_dim = width * 4
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(emb_dim // 2, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(emb_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on a (N, 1, H, W) real hologram.

        Returns:
            (N, num_classes) raw logits (classification) or values (regression).

        """
        x = make_input_2ch(x, use_fft=self.use_fft)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


# Label transforms must be picklable so DataLoader workers (spawn on Windows)
# can serialize the dataset; locals/lambdas inside transform_ds break that.
@dataclass
class RegLabelTransform:
    """Normalize a physical z-value into a float tensor using training-set stats."""

    z_avg: float
    z_std: float

    def __call__(self, z_raw: np.float32) -> Tensor:
        """Pass in physical value, return normalized z tensor."""
        z_tensor = torch.as_tensor(z_raw, dtype=torch.float32)
        return (z_tensor - self.z_avg) / self.z_std


def class_label_transform(z_raw_idx: np.int64) -> Tensor:
    """Convert a bin index into a long tensor."""
    return torch.as_tensor(z_raw_idx, dtype=torch.long)


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
        else:
            logger.debug("Model type requires one input channel, appending grayscale transform.")
            # custom nets (NEW, FOCUSNET) take a single-channel hologram
            common_tf.append(v2.Grayscale(num_output_channels=1))

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
        return (tf_train_ds, tf_eval_ds)


def transform_ds(base: HologramFocusDataset, a_cfg: AutoConfig) -> CoreTrainer:
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
    logger.info("Hyperparameter Initialization Complete ")
    return core_trainer_config
