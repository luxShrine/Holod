"""Tests for dataset-root resolution and the CSV/dataset config loading cases.

Covers ``paths.resolve_dataset_root``, ``check_csv_exists`` (every
``CreateCSVOption`` branch), ``AutoConfig.from_user`` path building, and
loading a ``HologramFocusDataset`` from a folder outside ``src/data``.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import polars as pl
import pytest
from PIL import Image

import holod.infra.util.paths as paths
from holod.infra.dataclasses import SAMPLE_PATH, AutoConfig, check_csv_exists
from holod.infra.dataset import HologramFocusDataset
from holod.infra.util.types import AnalysisType

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


def build_config(**overrides: object) -> AutoConfig:
    """Call ``AutoConfig.from_user`` with sane defaults, overriding per test."""
    kwargs: dict[str, object] = {
        "backbone": "enet",
        "batch_size": 4,
        "crop_size": 64,
        "checkpoint": False,
        "ds_root": None,
        "device_user": "cpu",
        "fixed_seed": True,
        "meta_csv_name": None,
        "num_classes": 5,
        "num_workers": 0,
        "opt_weight_decay": None,
        "val_split": 0.2,
        "epoch_count": 1,
        "opt_lr": None,
        "create_csv": False,
        "use_sample_data": False,
        "sch_factor": None,
        "sch_patience": None,
    }
    kwargs.update(overrides)
    return AutoConfig.from_user(**kwargs)  # pyright: ignore[reportArgumentType]


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


# -- AutoConfig.from_user ----------------------------------------------------------------------


def test_from_user_external_absolute_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An absolute dataset root outside src/data flows into meta_csv_strpath."""
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    cfg = build_config(ds_root=str(tmp_path / "ds"), meta_csv_name=CSV_NAME)
    assert cfg.meta_csv_strpath == csv_path.as_posix()
    assert cfg.analysis == AnalysisType.CLASS


def test_from_user_cwd_relative_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A dataset root relative to the working directory resolves correctly."""
    csv_path = make_external_dataset(tmp_path / "ds")
    _ban_user_prompts(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = build_config(ds_root="ds", meta_csv_name=CSV_NAME)
    assert Path(cfg.meta_csv_strpath) == csv_path.resolve()


def test_from_user_sample_flag_skips_dataset_checks(monkeypatch: pytest.MonkeyPatch):
    """--sample uses the bundled sample CSV and never prompts."""
    _ban_user_prompts(monkeypatch)
    cfg = build_config(use_sample_data=True)
    assert cfg.meta_csv_strpath == SAMPLE_PATH.as_posix()


def test_from_user_regression_analysis(monkeypatch: pytest.MonkeyPatch):
    """A single class selects regression analysis."""
    _ban_user_prompts(monkeypatch)
    cfg = build_config(use_sample_data=True, num_classes=1)
    assert cfg.analysis == AnalysisType.REG


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
