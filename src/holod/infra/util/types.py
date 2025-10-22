from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Any, NewType, TypeVar

import numpy as np
import numpy.typing as npt
import pint
import plotly.graph_objects as go
from matplotlib.figure import Figure
from PIL.Image import Image as ImageType
from torch.utils.data import Subset

from holod.infra.log import get_logger

_T_co = TypeVar("_T_co", covariant=True)

logger = get_logger(__name__)


# -- SINGLETON UNIT REGISTRY ---------------------------------------------------------------------
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity  # short alias

u = ureg  # short namespace, for u.um, u.nm, u.m


# -- NUMPY HELPERS -------------------------------------------------------------------------------
type Np1Array64 = npt.NDArray[np.float64]
type Np1Array32 = npt.NDArray[np.float32]
# TODO: replace above
type Arr64 = npt.NDArray[np.float64]
type Arr32 = npt.NDArray[np.float32]

# -- Epoch Helper --------------------------------------------------------------------------------
type SubsetImageDepth = Subset[tuple[ImageType, np.float64 | int]]
type SubsetList = list[Subset[tuple[ImageType, np.float64 | int]]]

StandardDev = NewType("StandardDev", float)
Mean = NewType("Mean", float)

# -- RANGE GUARDS FROM BEARTYPE ------------------------------------------------------------------
# BUG: does not work as intended
# type Nanometers = Annotated[Q_, Is[lambda q: 200 * u.nm <= q <= 2000 * u.nm]]
# type Micrometers = Annotated[Q_, Is[lambda q: -1_000 * u.um <= q <= 1_000 * u.um]]

# plotting shorthands
type Plots = Figure | go.Figure | list[go.Figure] | list[Figure]


class AnalysisType(Enum):
    """Restrict analysis variable to known strings."""

    CLASS = "class"
    REG = "reg"


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


class ModelType(Enum):
    """Restrict model backbone to known models.

    ENET: 3 channels, pretrained
    VIT: 3 channels, pretrained
    RESNET: 3 channels, pretrained
    NEW: 1 channel, untrained
    FOCUSNET: 1 channel, untrained
    """

    ENET = "efficientnet"
    VIT = "vit"
    RESNET = "resnet50"
    NEW = "new"
    FOCUSNET = "focusnet"


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
