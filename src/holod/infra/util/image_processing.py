import logging
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image as ImageType

from holod.infra.log import get_logger
from holod.infra.util.types import Arr64

logger = get_logger(__name__)


def crop_center(pil_img: ImageType, crop_width: int, crop_height: int) -> ImageType:
    """Crop provided image into around its center."""
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


def crop_max_square(pil_img: ImageType) -> ImageType:
    """Find the dimensions of image, crop to the largest square around its center."""
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def norm(data: Arr64) -> NDArray[np.float64]:
    """Normalize input numpy array."""
    max: np.float64 = np.max(data, keepdims=True)
    min: np.float64 = np.min(data, keepdims=True)
    return (data - min) / (max - min)


# check image is not corrupted
def _is_valid(path: str) -> bool:
    """Return ``True`` if the image file is readable by PIL."""
    try:
        with Image.open(path) as im:
            im.verify()  # quickly check if okay
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def correct_data_csv(csv_path: Path, dataset_path: Path) -> pl.DataFrame:
    """Ensure input dataframe maps on properly to data and paths."""
    dataset_base_path: Path = dataset_path.parent.expanduser().resolve()
    unfiltered_path_df: pl.DataFrame = pl.read_csv(csv_path, separator=";")  # read metadata CSV

    # get each path, get rid of leading "./", prepend it with the path to dataset parent,
    # replace original path column
    # clean_abs_path_df: pl.DataFrame = unfiltered_path_df.with_columns(
    #     pl.col("path").str.replace(r"\./", dataset_parent)
    # )

    abs_path_df: pl.DataFrame = unfiltered_path_df.with_columns(
        pl.col("path")
        .map_elements(lambda p: str(dataset_base_path / Path(p)), return_dtype=pl.Utf8)
        .alias("path")  # Overwrite the original 'path' column
    )

    # check if row actually points to an existing file, must use Path's function, thus map_elements
    filtered_path_df: pl.DataFrame = abs_path_df.filter(
        pl.col("path").map_elements(lambda p: Path(p).is_file(), pl.Boolean)
    )

    # make sure that there are any files after filtering
    if filtered_path_df.is_empty():
        raise RuntimeError("No hologram files found after filtering non-existing ones")

    # check image is not corrupted, again using PIL, thus map_elements
    proper_image_df = filtered_path_df.filter(pl.col("path").map_elements(_is_valid, pl.Boolean))

    # lists how many images were dropped
    n_bad = unfiltered_path_df.height - proper_image_df.height
    if n_bad:
        logger.warning("[bold yellow]Dropped %d corrupt or non-image files[/]", n_bad)

    # debug only, print random sample of the dataset to ensure nothing is obviously wrong
    if logger.isEnabledFor(logging.DEBUG):
        with pl.Config(fmt_str_lengths=50):  # make it a little longer for path
            logger.debug(proper_image_df.sample(10, shuffle=True))

    # Ensure path column is treated as string
    # casted_proper_image_df = proper_image_df.with_columns(pl.col("path").cast(pl.Utf8))

    # return casted_proper_image_df
    return proper_image_df


def parse_info(info_file: Path) -> dict[str, float]:
    """For each info.txt, extract data from it."""
    info: dict[str, float] = {}
    for line in info_file.read_text().splitlines():
        if not line.strip():
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip("um")  # strip units
        info[key] = float(val)
    return info
