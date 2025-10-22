"""Test Training pipeline."""

# TODO: call test seperately via program CLI?

from pathlib import Path
from random import randint
from typing import NewType

import numpy as np
import polars as pl
from PIL import Image, ImageEnhance
from PIL.Image import Image as ImageType
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models

import holo.infra.util.paths as paths
from holo.infra.dataclasses import AutoConfig, CoreTrainer, FocusNetTorch, NeuralNetwork
from holo.infra.dataset import HologramFocusDataset
from holo.infra.log import get_logger
from holo.infra.training import transform_ds
from holo.infra.util.paths import t_loc
from holo.infra.util.types import AnalysisType, ModelType

logger = get_logger(__name__)

type TestLoader = DataLoader[tuple[ImageType, np.float64]]
TestPath = NewType("TestPath", str)
TEST_IMGS: Path = t_loc() / "images"


# sanity check
def f():
    return AutoConfig(num_workers=0), CoreTrainer


def test_f():
    assert f() == (AutoConfig(num_workers=0), CoreTrainer)


def create_test_dataset(mode: AnalysisType = AnalysisType.CLASS) -> HologramFocusDataset:
    """Create small dataset to test."""
    if TEST_IMGS.exists:
        images_path = list(TEST_IMGS.glob("*.jpg"))
        if len(images_path) >= 31:
            print("correct amount of images, continuing...")
        else:
            print("creating test images...")
            idx = 1
            for _j in range(8):
                for _i, img in enumerate(TEST_IMGS.glob("source*.jpg")):
                    idx += 1
                    img = Image.open(img)
                    enhancer = ImageEnhance.Sharpness(img)
                    factor = idx / 4.0
                    img = enhancer.enhance(factor)
                    img.save(f"{TEST_IMGS}/test_image_{idx}.jpg")

        # new images potentially
        paths = [f.as_posix() for f in TEST_IMGS.resolve().glob("*.jpg")]
        num_images = len(paths)
        z_values = np.linspace(start=1e-6, stop=5e-6, num=num_images, dtype=np.float64)

        wavelengths = np.full((num_images), 0.405)
        data = [
            pl.Series("path", paths, dtype=pl.String),
            pl.Series("z_value", z_values),
            pl.Series("Wavelength", wavelengths),
        ]
        existing_df: pl.DataFrame = pl.DataFrame(data)

        num_classes: int | None = 10
        holo_dataset: HologramFocusDataset = HologramFocusDataset.from_df(
            mode, num_classes, existing_df
        )
        return holo_dataset
    TEST_IMGS.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        "Testing images not found, ensure testing images are in src/tests/images"
    )


def test_dataset_class():
    """Test to make sure that the dataset class is created."""
    assert isinstance(create_test_dataset(), HologramFocusDataset)


def test_dataset() -> None:
    """Basic sanity checks on ``HologramFocusDataset``."""
    # -- Test Types in __get_item__ --------------------------------------------------------------
    ds = create_test_dataset()
    x, _ = ds[0]
    image, label = ds[randint(0, len(ds) - 1)]
    assert isinstance(image, ImageType)
    label_type = float if ds.mode == AnalysisType.REG else np.int64
    assert isinstance(label, label_type)

    # if not isinstance(wavelength_m, Q_) or not isinstance(z_m, Q_):
    #
    #     print(
    #         "==== Unexpected Types in Hologram ==== "
    #         f"wavelength not units, is {type(wavelength_m)}"
    #         f"z not units, is {type(z_m)}"
    #         f"pixel_size_m not units, is {type(pixel_size_m)}"
    #     )


def create_test_loader() -> tuple[Tensor, Tensor]:
    """Create a small dataloader to test."""
    auto_config: AutoConfig = AutoConfig(
        analysis=AnalysisType.CLASS,
        batch_size=8,
        crop_size=124,
        epoch_count=2,
        meta_csv_strpath=TEST_IMGS.as_posix(),
        num_classes=5,
        num_workers=0,  # must be zero to prevent warning
        val_split=0.4,
        fixed_seed=True,
    )
    core_trainer: CoreTrainer = transform_ds(create_test_dataset(), auto_config)
    train_loader: TestLoader = core_trainer.train_loader
    eval_loader: TestLoader = core_trainer.val_loader
    train_features: Tensor
    train_labels: Tensor
    eval_features, eval_labels = next(iter(eval_loader))
    train_features, train_labels = next(iter(train_loader))
    # logger.debug(f"Feature batch shape: {train_features.size()}")
    # logger.debug(f"Labels batch shape: {train_labels.size()}")
    img: Tensor = train_features[0].squeeze(1)
    assert train_features.shape[1] >= 1
    _img_arr = img[1, :, :].numpy()
    # if label.ndim == 0:
    #     logger.debug(f"Sample label value: {label.item()}")
    # else:
    #     logger.debug(f"Sample label tensor: {label}")
    return eval_labels, train_labels


def test_loader():
    """Attempt to grab image and label."""
    label: Tensor = create_test_loader()[0]
    # print(f"{label.min()} \n {label.max()}")
    assert isinstance(label.min(), Tensor)
    assert isinstance(label.max(), Tensor)


def test_evaluation_metric_class():
    from holo.infra.training import transform_ds

    # construct autoconfig
    auto_config_c = AutoConfig(num_workers=0, analysis=AnalysisType.CLASS)
    auto_config_r = AutoConfig(num_workers=0, analysis=AnalysisType.REG)

    # construct training config
    base_c = create_test_dataset()
    base_r = create_test_dataset()
    t_cfg_c = transform_ds(base_c, auto_config_c)
    t_cfg_r = transform_ds(base_r, auto_config_r)

    # must be bins
    assert isinstance(t_cfg_c.evaluation_metric, np.ndarray), (
        f"evaluation_metric is not NDArray, found {type(t_cfg_c.evaluation_metric)}"
    )
    assert isinstance(t_cfg_r.evaluation_metric, np.ndarray), (
        f"evaluation_metric is not np.ndarray, found {type(t_cfg_r.evaluation_metric)}"
    )


def test_model_creation():
    res = AutoConfig(backbone=ModelType.RESNET)
    new = AutoConfig(num_classes=10, backbone=ModelType.NEW)
    enet = AutoConfig(backbone=ModelType.ENET)
    focusnet = AutoConfig(backbone=ModelType.FOCUSNET)

    model_res = res.create_model()
    model_new = new.create_model()
    model_enet = enet.create_model()
    model_focusnet = focusnet.create_model()

    assert isinstance(model_res, models.ResNet)
    assert isinstance(model_new, NeuralNetwork)
    assert isinstance(model_enet, models.EfficientNet)
    assert isinstance(model_focusnet, FocusNetTorch)
    # assert model_vit =
    # assert model_null = Exception


def test_paths():
    src = paths.src_root()
    holo_root = paths.holo_root()
    reports = paths.report_path()
    checkpoints = paths.checkpoints_loc()
    for p in ["maynooth", "phase-only", "thalassemic", "brownian", "bridge"]:
        paths.data_spec(p)

    paths.path_check({"reports": reports, "checkpoints": checkpoints, "holo_root": holo_root})

    assert paths.tex_root() == src / Path("tex")
    assert paths.figures_tex() == src / Path("tex/figures")
    assert paths.output_tex() == src / Path("tex/output")


def test_image_processing():
    import holo.infra.util.image_processing as ip

    # crop image
    image = Image.new(mode="RGB", size=(700, 500))
    cropped = ip.crop_center(image, 500, 500)
    assert cropped.size == (500, 500)
    max_cropped = ip.crop_max_square(image)
    assert max_cropped == cropped

    # normalize array
    arr = np.array([10, 20, 30, 40, 50])
    n_arr = ip.norm(arr)
    assert np.array_equal(n_arr, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))

    # valid image
    temp_image = t_loc() / Path("images/test_image_temp.jpg")
    non_existant_image = t_loc() / Path("_fake.jpg")
    assert ip._is_valid(non_existant_image) is False
    max_cropped.save(temp_image)
    assert ip._is_valid(temp_image) is True
    temp_image.unlink()  # remove after creating

    # validate data
    temp_csv_path = t_loc() / Path("test_temp_df.csv")
    # TODO: use this function to create df and thus test it as well
    # ip.parse_info()
    paths = [f.as_posix() for f in TEST_IMGS.resolve().glob("*.jpg")]
    num_images = len(paths)
    z_values = np.linspace(start=1e-6, stop=5e-6, num=num_images, dtype=np.float64)
    wavelengths = np.full((num_images), 0.405)
    df = pl.DataFrame(
        [
            pl.Series("path", paths, dtype=pl.String),
            pl.Series("z_value", z_values),
            pl.Series("Wavelength", wavelengths),
        ]
    )
    df.write_csv(temp_csv_path, separator=";")
    ip.correct_data_csv(temp_csv_path, TEST_IMGS)
    # TODO: create test for this final df
    temp_csv_path.unlink()


# TODO: takes a long time
# def test_train_autofocus():
#     plot_pred = train_autofocus(AutoConfig(epoch_count=1, crop_size=124, num_workers=6), None)
#     assert isinstance(plot_pred, PlotPred)


# def test_loading_config():
#     from serde.toml import from_toml
# from holo.infra.dataclasses import (
#     Flags,
#     Paths,
#     Train,
#     UserConfig,
# )
#
#     with open("train_settings.toml") as config_file:
#         config = config_file.read()
#     config: UserConfig = from_toml(UserConfig, config)
#     # merge the config file (if it exists), by overwriting it with CLI args
#     # BUG: currently requires MW-dataset
#     config.merge(
#         paths=Paths(None, None),
#         train=Train(
#             None,
#             None,
#             224,
#             None,
#             None,
#             None,
#             None,
#             0,
#             None,
#             None,
#             None,
#             None,
#         ),
#         flags=Flags(False, False, True, True),
#     )
#     assert isinstance(config.to_auto_config(), AutoConfig)


# TODO:
# def test_labels():
#
#     # test each loader
#     for loader in (t_cfg.train_loader, t_cfg.val_loader):
#         label_min, label_max = label.min(), label.max()
#         # check that labels are in range expected
#         # n_classes = pred.size(1)
#         if (label_min < 0) or (label_max >= a_cfg.num_classes):
#             raise Exception(f"label out of range: {label_min} – {label_max}")


# TODO: within expected range, type
# def test_std_avg():
# t_cfg.z_sig
# t_cfg.z_mu

# TODO: match expected?
# def test_loss_functions():
# t_cfg.loss_fn
# t_cfg.optimizer
# t_cfg.scheduler

# TODO: test
# assert z_true.size() == z_pred.size(), (
#     "z_pred and z_true are not the same size, cannot be compared"
#     )

# TODO:
# def check_units(vars_to_check: dict[Q_, u]) -> bool:
#     """Returns ``True`` if all quantities have the expected units.
# Example usage:
# '''
# quant_dict = {
#     z_m: u.m,
#     wavelength_m: u.m,
#     pixel_size_m: u.m,
# }
# if not check_units(quant_dict):
#     raise RuntimeError
#    '''
# """
#     i = 0
#     for v, units in vars_to_check.items():
#         if v.u == units:
#             continue
#         i += 1
#         logger.error(f"found unexpected units, expected {units} but got {v.to_compact}")
#     return i != 1
