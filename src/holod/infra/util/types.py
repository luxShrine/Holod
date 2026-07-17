from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any, NewType, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image as ImageType
from torch.utils.data import Subset

from holod.infra.log import get_logger

if TYPE_CHECKING:
    # heavy plotting deps stay import-free at runtime: `type` aliases are lazily
    # evaluated, and eagerly importing matplotlib crashes in notebook subprocesses
    # whose MPLBACKEND names a backend module this venv doesn't have
    import plotly.graph_objects as go
    from matplotlib.figure import Figure

_T_co = TypeVar("_T_co", covariant=True)

logger = get_logger(__name__)


# -- PHYSICAL CONSTANTS --------------------------------------------------------------------------
# Unit convention (units are carried in names, converted only at explicit boundaries):
#   - metadata CSV / info.txt: Wavelength in µm (0.405 = 405 nm), L_value/z_value in mm
#   - training + reporting (labels, z_avg_mm/z_std_mm, MAE): mm
#   - optics API (recon_inline, focus_score, torch_recon): meters (wavelength_m, z_m, dx_m)
# pixel pitch of the sensor the datasets were captured with (meters)
SENSOR_PIXEL_PITCH_M: float = 3.8e-6


# -- NUMPY HELPERS -------------------------------------------------------------------------------
type Np1Array64 = npt.NDArray[np.float64]
type Np1Array32 = npt.NDArray[np.float32]
# TODO: replace above
type Arr64 = npt.NDArray[np.float64]
type Arr32 = npt.NDArray[np.float32]

# -- Epoch Helper --------------------------------------------------------------------------------
type SubsetImageDepth = Subset[tuple[ImageType, np.float64 | int]]
type SubsetList = list[Subset[tuple[ImageType, np.float64 | int]]]
type StateDict = dict[str, Any]

StandardDev = NewType("StandardDev", float)
Mean = NewType("Mean", float)

# plotting shorthands
type Plots = Figure | go.Figure | list[go.Figure] | list[Figure]


class AnalysisType(Enum):
    """Restrict analysis variable to known strings."""

    CLASS = "class"
    REG = "reg"

    @staticmethod
    def determine(num_classes: int):
        if num_classes == 1:
            return AnalysisType.REG
        return AnalysisType.CLASS


class DisplayType(Enum):
    """Restrict display variable to known strings."""

    SHOW = "show"
    SAVE = "save"
    BOTH = "both"
    META = "meta"


class CreateCSVOption(IntEnum):
    """Group options for user input with regards to creating a CSV."""

    CSV_CREATE = auto()
    CSV_DATA_MISSING = auto()
    CSV_MISSING = auto()
    CSV_VALID = auto()


class UserDevice(Enum):
    """Restrict device variable to known device strings."""

    CPU = "cpu"
    CUDA = "cuda"

    @staticmethod
    def determine(desired_device: str = "cuda"):
        if desired_device == "cpu":
            device = UserDevice.CPU
        elif desired_device == "cuda":
            device = UserDevice.CUDA if torch.cuda.is_available() else UserDevice.CPU
        else:
            device = UserDevice.CPU
        return device


class ModelType(Enum):
    """Restrict model backbone to known models.

    ENET: 3 channels, pretrained
    VIT: 3 channels, pretrained
    RESNET: 3 channels, pretrained
    PHYSICSCNN: 1 channel, untrained
    FOCUSNET: 1 channel, untrained
    """

    ENET = "efficientnet"
    VIT = "vit"
    RESNET = "resnet50"
    PCNN = "new"
    FOCUSNET = "focusnet"

    @classmethod
    def from_str(cls, backbone: str):
        backbone = backbone.lower()
        if "vit" in backbone:
            return ModelType.VIT
        if "enet" in backbone or "efficient" in backbone:
            return ModelType.ENET
        if "res" in backbone:
            return ModelType.RESNET
        if "new" in backbone or "custom" in backbone or "pcnn" in backbone:
            return ModelType.PCNN
        if "focus" in backbone:
            return ModelType.FOCUSNET
        raise Exception(f"Unknown backbone passed: {backbone}")


class TrainingStage(IntEnum):
    """Limit stages of training."""

    TRAIN = auto()
    EVAL = auto()


class ReconstructionMethod(Enum):
    """Limit type of reconstruction."""

    FRESNEL = "fresnel"
    ANGULAR = "angular"


def check_dtype(type_check: str, expected_type, **kwargs) -> bool:
    """Check all inputs have same types/datatypse, else return exception."""
    # get first value to be checked against
    first_value: Any = next(iter(kwargs.values()))
    match type_check:
        # case for tensor items
        case "dtype":
            # if expected type is passed, check it against each item
            if expected_type is not None:
                all_correct_type = all(v.dtype == expected_type for v in kwargs.values())
            else:
                # otherwise ensure consistency with first dtype
                all_correct_type = all(v.dtype == first_value.dtype for v in kwargs.values())
            # not all the correct type, show each items dtype
            if all_correct_type is False:
                mismatch = ", ".join(f"{k}={v.dtype}" for k, v in kwargs.items())
                logger.error(
                    "[bold red]Dtype mismatch for '%s':[/] %s",
                    type_check,
                    mismatch,
                )
                return False
            return True

        case _:
            raise Exception(f"Unexpected type check argument {type_check}")
