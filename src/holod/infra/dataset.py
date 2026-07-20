from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, override

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

import holod.infra.util.paths as paths
from holod.infra.log import get_logger
from holod.infra.util.image_processing import correct_data_csv, parse_info
from holod.infra.util.types import (
    SENSOR_PIXEL_PITCH_M,
    AnalysisType,
    Arr32,
    Arr64,
    Mean,
    ModelType,
    StandardDev,
    SubsetImageDepth,
)

if TYPE_CHECKING:
    import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger(__name__)
HOLO_DEF = paths.data_spec("mw")


class HologramDepths:
    """Store and manipulate the z values of holograms, in millimeters (as recorded in the CSV)."""

    def __init__(self, z_values: Arr64) -> None:
        """Create a depth value."""
        self.z_array: Arr64 = z_values

    def __len__(self) -> int:
        """Length of the depth array."""
        return len(self.z_array)

    def min(self) -> np.float64:
        """Length of the depth array."""
        return self.z_array.min()

    def max(self) -> np.float64:
        """Length of the depth array."""
        return self.z_array.max()

    def __iter__(self):
        """Iterate through z_array."""
        yield from self.z_array

    def __getitem__(self, idx: int | slice) -> Arr64 | np.float64:
        """Return the slice or index of the underlying array."""
        if isinstance(idx, int):
            return self.z_array[idx]
        if isinstance(idx, slice):
            return self.z_array[idx.start : idx.stop : idx.step]
        return None

    def subset_mean_std(self, subset_indices: Sequence[int]) -> tuple[Mean, StandardDev]:
        """Get the mean and standard deviation for a subset of the z depths."""
        subset_z: Arr64 = self.z_array[subset_indices]
        subset_z_mean: float = subset_z.mean()
        subset_z_std: float = subset_z.std()
        if subset_z_std < 1e-6:  # Avoid division by zero or very small std
            subset_z_std = 1.0
            logger.warning(
                "Training subset z standard deviation is near zero."
                + f"Setting to {subset_z_std} for normalization."
            )

        return (Mean(subset_z_mean), StandardDev(subset_z_std))


class HologramFocusDataset(Dataset[tuple[ImageType, np.float64 | int]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self,
        mode: AnalysisType,
        num_classes: None | int,
        csv_file_strpath: str = (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix(),
        existing_df: None | pl.DataFrame = None,
    ) -> None:
        """Load metadata and prepare dataset fields."""
        # inherit from torch dataset
        # super().__init__()
        self.mode: AnalysisType = mode
        self.num_classes: int | None = num_classes

        # Format set of records to draw from
        self.csv_file_path: Path = Path(csv_file_strpath)
        holo_dir: Path = self.csv_file_path.parent
        self.records: pl.DataFrame = self.read_records(existing_df)
        self.check_dataset_length()
        # TODO: make the init process not dependent on paths at all, at to allow for
        # smooth tests or simulated holograms without managing paths
        self.paths: list[Path] = [holo_dir / Path(p) for p in self.records["path"].to_list()]

        # sample group of each record (the digit_sample_name directory in the
        # dataset layout); lets the split keep whole samples in one loader
        self.sample_names: list[str] = [
            self._sample_name_from_path(p) for p in self.records["path"].to_list()
        ]

        # CSV z_value column is in millimeters
        self.z: HologramDepths = HologramDepths(self.records["z_value"].to_numpy())
        # CSV Wavelength column is in micrometers (e.g. 0.405 for a 405 nm laser)
        self.wavelength_um: Arr64 = self.records["Wavelength"].to_numpy()
        # NOTE: constant pixel size (meters)
        self.pixel_size_m: Arr32 = np.full(len(self.z), SENSOR_PIXEL_PITCH_M, dtype=np.float32)

        # the z depth on its own cannot be passed to the model, it must be
        # converted to a set of bins, as integers. to do so digitize array => bins
        z_uniq: Arr64 = np.unique(self.z.z_array)
        logger.debug(
            f"unique depths sample (mm): {z_uniq[:10]} \n total: {len(np.unique(self.z.z_array))}"
        )

        if self.mode == AnalysisType.CLASS:
            # TODO: Implement automatic bin creation logic in some form
            # bins ought to wide enough to be populated, but not too wide as to loose meaning
            if self.num_classes is None or self.num_classes <= 0:
                raise ValueError("num_classes must be a positive integer for classification.")
            # create as many bins as selected number of classes, thus num_classes + 1 edges
            bin_edges = np.linspace(
                self.z.min(),  # upper/lower bounds
                self.z.max(),
                self.num_classes + 1,
                dtype=np.float64,
            )

            # np.digitize returns 1-based indices for bins normally.
            # bins[i-1] <= x < bins[i] -> i.
            # x < bins[0] -> 0. x >= bins[num_classes] (last edge) -> num_classes + 1.
            raw_binned_indices: npt.NDArray[np.intp] = np.digitize(self.z.z_array, bin_edges)

            # Clip to ensure indices are within [1, num_classes] then subtract 1 for 0-indexing
            # This maps values < min_z to bin 0, and values >= max_z to bin num_classes-1
            self.z_bins: npt.NDArray[np.intp] = np.clip(raw_binned_indices, 1, self.num_classes) - 1
            self.bin_centers: Arr64 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            logger.info(
                f"Created {self.num_classes} bins for classification with "
                + f"centers: {self.bin_centers}"
            )
        else:
            # TODO: flesh out regression
            bin_edges = None

        self.bin_edges: Arr64 | None = bin_edges

    def __len__(self) -> int:
        """Return the lengh of entries in dataframe."""
        return len(self.records)

    @override
    def __getitem__(self, idx: int) -> tuple[ImageType, np.float64 | np.int64]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return image and z depth value."""
        # ensure image is read properly
        try:
            img: ImageType = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            logger.exception(f"Error loading image {self.paths[idx]}: {e}")
            raise
        # Continuous if regression, bins if classification
        label: np.float64 | np.int64 = (  # pyright: ignore[reportAssignmentType]
            self.z[idx] if self.mode == AnalysisType.REG else self.z_bins[idx]
        )

        # Return quantities for clarity
        return (img, label)

    @staticmethod
    def _sample_name_from_path(rel_path: str) -> str:
        """Extract the sample group name from a CSV-relative image path.

        The dataset layout is ``<top_level>/<wavelength>/<digit_sample_name>/<zlevel>/<img>``,
        so the sample directory is always two levels above the image regardless of
        how deeply the dataset root itself is nested.
        """
        parts = PurePosixPath(rel_path.replace("\\", "/")).parts
        if len(parts) >= 3:
            return parts[-3]
        # too shallow to carry a sample directory (e.g. flat test datasets):
        # treat each image as its own group so no records silently collapse together
        return rel_path

    def sample_group_indices(self) -> dict[str, list[int]]:
        """Map each sample group name to the record indices belonging to it.

        Records of the same sample captured at different wavelengths share a group,
        so a sample-wise split never leaks a specimen across train and eval.
        """
        groups: dict[str, list[int]] = {}
        for idx, name in enumerate(self.sample_names):
            groups.setdefault(name, []).append(idx)
        return groups

    def read_records(self, existing_df: None | pl.DataFrame) -> pl.DataFrame:
        """Read the values of the dataset, either from file or dataframe."""
        if existing_df is not None:
            return existing_df
        return correct_data_csv(self.csv_file_path.resolve())

    def check_dataset_length(self) -> None:
        """Check that dataset is not empty nor too small to train on."""
        length_ds = len(self)
        logger.debug(f"length of base Hologram dataset is {length_ds}")
        # -- Broad Tests -------------------------------------------------------------------------
        if length_ds == 0:
            raise RuntimeError("dataset length is zero.")
        if length_ds < 30:
            raise RuntimeError("dataset contains less than 30 images, too few to train on.")

    @classmethod
    def from_df(
        cls, mode: AnalysisType, num_classes: None | int, df: pl.DataFrame
    ) -> HologramFocusDataset:
        """Create a HologramFocusDataset from a supplied dataframe."""
        df_created_dataset: HologramFocusDataset = HologramFocusDataset(
            mode,
            num_classes,
            existing_df=df,
            csv_file_strpath="",
        )
        return df_created_dataset

    @staticmethod
    def create_meta(hologram_directory: Path, csv_name: str) -> pl.DataFrame:
        """Return a dataframe mapping path of each image file to its data from the info.txt file.

        Args:
            hologram_directory: directory of dataset
            csv_name: filename of csv to be output
        returns:
            dataframe containing dataset paths linked to image's information

        """
        logger.info("Creating proper CSV...")

        rows: list[dict[str, (str | float)]] = []
        for info in hologram_directory.rglob("info.txt"):
            meta = parse_info(info)
            for img in info.parent.rglob("*.[jp][np]g"):
                rows.append(
                    {
                        "path": str(img.relative_to(hologram_directory)),
                        **meta,  # unpacks a dictionary into keyword arguments
                    }
                )
        df = pl.DataFrame(rows)
        df.glimpse()

        # create full path to save to
        full_csv_path: Path = hologram_directory / Path(csv_name)
        df.write_csv(full_csv_path, separator=";")

        return df


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
        legacy_random_crop: bool = False,
    ):
        """Construct a tuple of transformed datasets from subsets."""
        rgb_models = [ModelType.ENET, ModelType.RESNET, ModelType.VIT]

        # TODO: using composition, wrap models in class that retrieves their transformations
        # then easily grab transforms

        # TODO: evaluate implementing extra transforms
        extra_tf: list[nn.Module] = [
            # flip image with some probability
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
        if legacy_random_crop:
            # pre-fix behavior for ablation runs: random scaled crop before the flips
            logger.info("Legacy [bold]RandomResizedCrop[/bold] augmentation enabled (ablation).")
            extra_tf.insert(0, v2.RandomResizedCrop(size=crop_size, antialias=True))

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
