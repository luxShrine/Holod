"""Tests for dataset-root resolution and the CSV/dataset config loading cases.

Covers ``paths.resolve_dataset_root``, ``check_csv_exists`` (every
``CreateCSVOption`` branch), ``CompareUserConfig`` loading/merging/path
resolution, and loading a ``HologramFocusDataset`` from a folder outside
``src/data``.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import polars as pl
import pytest
from PIL import Image

import holod.infra.util.paths as paths
from holod.infra.dataclasses import (
    SAMPLE_PATH,
    CompareUserConfig,
    Flags,
    ModelConfig,
    Paths,
    Train,
    check_csv_exists,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.util.types import AnalysisType, ModelType

CSV_NAME = "meta.csv"
NUM_IMAGES = 32


def _ban_user_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any interactive click prompt fires."""

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("no interactive prompt should be needed for this case")

    monkeypatch.setattr(click, "prompt", _fail)
    monkeypatch.setattr(click, "confirm", _fail)


def make_external_dataset(root: Path, *, with_csv: bool = True, with_info: bool = False) -> Path:
    """Create a small dataset on disk (outside ``src/data``); return the CSV path."""
    root.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for idx in range(NUM_IMAGES):
        img_path = root / f"holo_{idx}.jpg"
        Image.new("L", (32, 32), color=(idx * 7) % 255).save(img_path)
        rel_paths.append(img_path.name)
    if with_info:
        (root / "info.txt").write_text("Wavelength = 0.405\nL_value = 18.96\nz_value = 0.521\n")
    csv_path = root / CSV_NAME
    if with_csv:
        df = pl.DataFrame(
            [
                pl.Series("path", rel_paths, dtype=pl.String),
                pl.Series("z_value", np.linspace(1.0, 5.0, NUM_IMAGES, dtype=np.float64)),
                pl.Series("Wavelength", np.full(NUM_IMAGES, 0.405)),
            ]
        )
        df.write_csv(csv_path, separator=";")
    return csv_path


def build_config(**overrides: object) -> CompareUserConfig:
    """Build a single-model ``CompareUserConfig`` with sane defaults, overriding per test."""
    kwargs: dict[str, object] = {
        "paths": Paths.empty(),
        "flags": Flags(checkpoint=False, create_csv=False, fixed_seed=True, sample=False),
        "batch_size": 4,
        "crop_size": 64,
        "device": "cpu",
        "epoch_count": 1,
        "num_classes": 5,
        "num_workers": 0,
        "val_split": 0.2,
        "enet": ModelConfig(Train("enet")),
    }
    kwargs.update(overrides)
    return CompareUserConfig(**kwargs)  # pyright: ignore[reportArgumentType]


# -- resolve_dataset_root ----------------------------------------------------------------------


def test_resolve_empty_falls_back_to_data_root():
    """An unset dataset root resolves to src/data."""
    assert paths.resolve_dataset_root("") == paths.data_root()


def test_resolve_absolute_path_outside_data_root(tmp_path: Path):
    """Absolute paths anywhere on disk are used as-is."""
    ds = tmp_path / "external_ds"
    ds.mkdir()
    assert paths.resolve_dataset_root(str(ds)) == ds


def test_resolve_cwd_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Paths relative to the working directory resolve outside src/data."""
    ds = tmp_path / "my_ds"
    ds.mkdir()
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_dataset_root("my_ds") == ds.resolve()


def test_resolve_name_under_data_root():
    """Bare dataset names keep resolving under src/data (historical behavior)."""
    expected = (paths.data_root() / "MW_Dataset_Sample").resolve()
    assert paths.resolve_dataset_root("MW_Dataset_Sample") == expected


def test_resolve_repo_root_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Repo-relative paths (as in the README) work from any working directory."""
    monkeypatch.chdir(tmp_path)  # ensure the cwd candidate cannot match
    expected = (paths.data_root() / "MW_Dataset_Sample").resolve()
    assert paths.resolve_dataset_root("src/data/MW_Dataset_Sample") == expected


def test_resolve_missing_falls_back_to_data_root_candidate():
    """Unknown names fall back to the src/data candidate for consistent errors."""
    name = "no_such_dataset_xyz"
    assert paths.resolve_dataset_root(name) == paths.data_root() / name


# -- check_csv_exists --------------------------------------------------------------------------


def test_check_csv_valid_external(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An existing CSV in an external dataset folder passes without prompting."""
    ds = tmp_path / "ds"
    make_external_dataset(ds)
    _ban_user_prompts(monkeypatch)
    assert check_csv_exists(False, ds, CSV_NAME) == (False, CSV_NAME)


def test_check_csv_missing_choose_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Missing CSV in an existing folder: choosing 'sample' opts into sample data."""
    ds = tmp_path / "ds"
    make_external_dataset(ds, with_csv=False)
    monkeypatch.setattr(click, "prompt", lambda *a, **k: "sample")
    (use_sample, name) = check_csv_exists(False, ds, "absent.csv")
    assert use_sample is True
    assert name == "absent.csv"


def test_check_csv_missing_choose_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Missing CSV: choosing 'create' builds one from info.txt and returns its name."""
    ds = tmp_path / "ds"
    make_external_dataset(ds, with_csv=False, with_info=True)
    answers = iter(["create", "generated.csv"])
    monkeypatch.setattr(click, "prompt", lambda *a, **k: next(answers))
    (use_sample, name) = check_csv_exists(False, ds, "absent.csv")
    assert use_sample is False
    assert name == "generated.csv"
    df = pl.read_csv(ds / "generated.csv", separator=";")
    assert df.height == NUM_IMAGES
    assert {"path", "Wavelength", "L_value", "z_value"} <= set(df.columns)


def test_check_csv_dataset_missing_confirms_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A dataset folder that does not exist offers the sample data fallback."""
    monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
    (use_sample, _) = check_csv_exists(False, tmp_path / "missing", "x.csv")
    assert use_sample is True


def test_check_csv_create_requires_existing_folder(tmp_path: Path):
    """--create-csv against a missing folder raises a clear CLI error."""
    with pytest.raises(click.ClickException):
        check_csv_exists(True, tmp_path / "missing", "x.csv")


def test_check_csv_empty_name_treated_as_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An empty CSV name never validates against the folder itself."""
    ds = tmp_path / "ds"
    make_external_dataset(ds, with_csv=False)
    monkeypatch.setattr(click, "prompt", lambda *a, **k: "sample")
    (use_sample, _) = check_csv_exists(False, ds, "")
    assert use_sample is True


# -- CompareUserConfig: resolve_paths / merge / to_auto_config ----------------------------------


def test_resolve_external_absolute_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An absolute dataset root outside src/data flows into meta_csv_strpath."""
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    cfg = build_config(paths=Paths(str(tmp_path / "ds"), CSV_NAME)).resolve_paths()
    a_cfg = cfg.to_auto_config(ModelType.ENET)
    assert a_cfg.meta_csv_strpath == csv_path.as_posix()
    assert a_cfg.analysis == AnalysisType.CLASS


def test_resolve_cwd_relative_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A dataset root relative to the working directory resolves correctly."""
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = build_config(paths=Paths("ds", CSV_NAME)).resolve_paths()
    assert Path(cfg.to_auto_config(ModelType.ENET).meta_csv_strpath) == csv_path.resolve()


def test_resolve_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Calling resolve_paths twice must not re-resolve or re-prompt."""
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    cfg = build_config(paths=Paths(str(tmp_path / "ds"), CSV_NAME)).resolve_paths()
    first = cfg.paths.meta_csv_name
    assert cfg.resolve_paths().paths.meta_csv_name == first == csv_path.as_posix()


def test_resolve_sample_flag_skips_dataset_checks(monkeypatch: pytest.MonkeyPatch):
    """--sample uses the bundled sample CSV and never prompts."""
    _ban_user_prompts(monkeypatch)
    flags = Flags(checkpoint=False, create_csv=False, fixed_seed=True, sample=True)
    cfg = build_config(flags=flags).resolve_paths()
    assert cfg.to_auto_config(ModelType.ENET).meta_csv_strpath == SAMPLE_PATH.as_posix()


def test_regression_analysis(monkeypatch: pytest.MonkeyPatch):
    """A single class selects regression analysis."""
    _ban_user_prompts(monkeypatch)
    flags = Flags(checkpoint=False, create_csv=False, fixed_seed=True, sample=True)
    cfg = build_config(flags=flags, num_classes=1).resolve_paths()
    assert cfg.to_auto_config(ModelType.ENET).analysis == AnalysisType.REG


def test_merge_keeps_config_paths_when_cli_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Empty CLI path strings must not clobber values from the config file.

    ``merge`` resolves paths on exit; if the empty CLI paths clobbered the
    config-file dataset, resolution would prompt (banned here) instead of
    resolving to the config-file dataset's CSV.
    """
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    cfg = build_config(paths=Paths(str(tmp_path / "ds"), CSV_NAME))
    _ = cfg.merge(paths=Paths.empty(), flags=Flags.empty(), batch_size=8)
    assert cfg.paths.dataset_root == (tmp_path / "ds").as_posix()
    assert cfg.paths.meta_csv_name == csv_path.as_posix()
    assert cfg.batch_size == 8


def test_from_toml_multi_model_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The multi-model TOML schema deserializes and maps onto AutoConfig."""
    ds = tmp_path / "ds"
    csv_path = make_external_dataset(ds)
    toml_file = tmp_path / "train_settings.toml"
    toml_file.write_text(
        f"""
batch_size = 4
crop_size = 64
device = "cpu"
epoch_count = 1
num_classes = 5
num_workers = 0
val_split = 0.2

[paths]
dataset_root = "{ds.as_posix()}"
meta_csv_name = "{CSV_NAME}"

[flags]
checkpoint = false
create_csv = false
fixed_seed = true
sample = false

[enet.train]
backbone = "enet"
learning_rate = 1e-4
optimizer_weight_decay = 1e-2
sch_factor = 0.1
sch_patience = 7

[focusnet.train]
backbone = "focusnet"
learning_rate = 1e-3
"""
    )
    _ban_user_prompts(monkeypatch)
    cfg = CompareUserConfig.from_toml(toml_file)
    assert set(cfg.configured_backbones()) == {ModelType.ENET, ModelType.FOCUSNET}
    a_cfg = cfg.resolve_paths().to_auto_config(ModelType.ENET)
    assert a_cfg.meta_csv_strpath == csv_path.as_posix()
    assert a_cfg.opt_lr == 1e-4
    assert a_cfg.sch_patience == 7
    # the focusnet section leaves scheduler fields unset; defaults must be valid
    focus_cfg = cfg.to_auto_config(ModelType.FOCUSNET)
    assert focus_cfg.opt_lr == 1e-3
    assert 0 < focus_cfg.sch_factor < 1  # ReduceLROnPlateau requires factor < 1


# -- end-to-end dataset load -------------------------------------------------------------------


def test_hologram_dataset_loads_external_folder(tmp_path: Path):
    """HologramFocusDataset loads images from a folder outside src/data."""
    csv_path = make_external_dataset(tmp_path / "ds")
    ds = HologramFocusDataset(AnalysisType.CLASS, 5, csv_file_strpath=str(csv_path))
    assert len(ds) == NUM_IMAGES
    assert all(p.is_file() for p in ds.paths)
    (img, label) = ds[0]
    assert img.size == (32, 32)
    assert 0 <= int(label) < 5
