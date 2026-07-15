from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

# WARN: Path must be importable at runtime: pyserde wraps @serde class methods with
# beartype, which resolves the annotation on FixAblationReport.save at call time.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from PIL import Image
from rich.table import Table
from serde import serde
from serde.json import to_json

from holod.core.compare import (
    HologramEval,
    _fmt,
    _load_checkpoint_model,
    evaluate_checkpoint_on_hologram,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import console_ as console
from holod.infra.log import get_logger
from holod.infra.training import create_training_setup, train_eval_epoch
from holod.infra.util.image_processing import crop_max_square
from holod.infra.util.paths import checkpoints_loc, report_path
from holod.infra.util.types import SENSOR_PIXEL_PITCH_M, AnalysisType

if TYPE_CHECKING:
    from holod.infra.dataclasses import CompareUserConfig, TrainingOutput
    from holod.infra.util.types import ModelType

logger = get_logger(__name__)

# condition names, in report order
COND_BEFORE = "before"
COND_CROP_FIX_ONLY = "crop_fix_only"
COND_AFTER = "after"


@serde
@dataclass
class ConditionStats:
    """Metrics for one before/after condition of the two-fix ablation."""

    condition: str
    trained_with_random_crop: bool
    mag_corrected_focus: bool
    checkpoint: str = ""
    val_accuracy: float | None = None
    val_mae_mm: float | None = None
    avg_train_loss: float | None = None
    avg_val_loss: float | None = None
    train_time_s: float | None = None
    # gradient-Tamura sharpness over the evaluated holograms; higher is sharper
    focus_tc_mean: float | None = None
    focus_tc_std: float | None = None
    # mean |z_pred - z_true| over the evaluated holograms (mm)
    holo_mae_mm: float | None = None
    holo_count: int = 0
    error: str | None = None


@serde
@dataclass
class FixAblationReport:
    """Before/after metrics for the random-crop and magnification fixes."""

    backbone: str
    analysis: str
    num_classes: int
    crop_size: int
    batch_size: int
    epoch_count: int
    device: str
    dataset_csv: str
    seed: int
    holo_count: int
    created: str
    conditions: list[ConditionStats] = field(default_factory=list)
    # raw per-hologram evaluations for each trained model (focus_tc is scored at
    # the magnification-corrected depth, focus_tc_uncorrected at the raw depth)
    legacy_evals: list[HologramEval] = field(default_factory=list)
    current_evals: list[HologramEval] = field(default_factory=list)

    def to_table(self) -> Table:
        """Render the report as a Rich table, one row per condition."""
        table = Table(
            title=(
                f"Fix ablation: {self.backbone} ({self.analysis}, {self.device}, "
                + f"crop {self.crop_size}, {self.epoch_count} epochs)"
            ),
            caption=(
                f"'{COND_CROP_FIX_ONLY}' and '{COND_AFTER}' share one trained model; "
                + "only the focus-score reconstruction depth differs."
            ),
        )
        table.add_column("Condition", style="bold")
        table.add_column("Random crop", justify="center")
        table.add_column("Mag-corrected z", justify="center")
        table.add_column("Accuracy", justify="right")
        table.add_column("Val MAE (mm)", justify="right")
        table.add_column("Train loss", justify="right")
        table.add_column("Val loss", justify="right")
        table.add_column("Focus TC (mean)", justify="right")
        table.add_column("Focus TC (std)", justify="right")
        table.add_column("Holo MAE (mm)", justify="right")
        table.add_column("Error")

        for cond in self.conditions:
            table.add_row(
                cond.condition,
                "yes" if cond.trained_with_random_crop else "no",
                "yes" if cond.mag_corrected_focus else "no",
                _fmt(cond.val_accuracy, ".4g"),
                _fmt(cond.val_mae_mm, ".4g"),
                _fmt(cond.avg_train_loss, ".4g"),
                _fmt(cond.avg_val_loss, ".4g"),
                _fmt(cond.focus_tc_mean, ".4g"),
                _fmt(cond.focus_tc_std, ".3g"),
                _fmt(cond.holo_mae_mm, ".4g"),
                cond.error if cond.error is not None else "",
                style="bold green" if cond.condition == COND_AFTER else None,
            )
        return table

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        """Serialize the report to JSON and CSV, returning both file paths."""
        out_dir = directory if directory is not None else report_path()
        # stamp the filenames with the report's creation time so successive
        # ablation runs accumulate instead of overwriting the previous run
        stamp = datetime.fromisoformat(self.created).strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"fix_ablation_{stamp}.json"
        csv_path = out_dir / f"fix_ablation_{stamp}.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(to_json(self, FixAblationReport))
        pl.DataFrame([asdict(c) for c in self.conditions]).write_csv(csv_path)
        logger.info(f"Fix-ablation report saved to {json_path} and {csv_path}")
        return (json_path, csv_path)


def _seed_everything(seed: int) -> None:
    """Reset the global RNGs so both training runs share their initialization."""
    torch.manual_seed(seed)
    random.seed(seed)
    # deliberately seed the legacy global numpy RNG: third-party code (torch
    # workers, torchvision) draws from it, not from a Generator instance
    np.random.seed(seed)  # noqa: NPY002


def _train_condition(
    u_cfg: CompareUserConfig,
    backbone: ModelType,
    base: HologramFocusDataset,
    legacy_random_crop: bool,
    seed: int,
) -> tuple[TrainingOutput, Path, float]:
    """Train one condition from scratch and preserve its best checkpoint.

    Returns the training output, the preserved checkpoint path, and the
    wall-clock training time in seconds.
    """
    cfg = u_cfg.to_auto_config(backbone)
    cfg.legacy_random_crop = legacy_random_crop
    # both runs must draw the identical validation split
    cfg.fixed_seed = True

    _seed_everything(seed)
    core_trainer = create_training_setup(base, cfg)
    best_val = float("inf") if cfg.analysis == AnalysisType.REG else -float("inf")
    start = time.perf_counter()
    output = train_eval_epoch(core_trainer, best_val, None)
    elapsed = time.perf_counter() - start

    # train_eval_epoch hardcodes the best-model filename per backbone, so the
    # second run would overwrite the first; move it to a condition-tagged name
    best = checkpoints_loc() / f"checkpoint_{backbone.name}_best.pth"
    if not best.exists():
        raise FileNotFoundError(
            f"Training finished but no best checkpoint was written at {best}; "
            + "cannot evaluate this condition."
        )
    tag = "legacycrop" if legacy_random_crop else "current"
    preserved = checkpoints_loc() / f"checkpoint_{backbone.name}_ablation_{tag}.pth"
    _ = best.replace(preserved)
    logger.info(f"Preserved {'legacy' if legacy_random_crop else 'current'} model at {preserved}")
    return (output, preserved, elapsed)


def _evaluate_checkpoint(
    ckpt_path: Path,
    base: HologramFocusDataset,
    indices: list[int],
    device: str,
    crop_size: int,
) -> list[HologramEval]:
    """Score one preserved checkpoint on the given dataset indices.

    Each hologram is scored at both the magnification-corrected depth
    (``focus_tc``) and the raw predicted depth (``focus_tc_uncorrected``).
    Failures are logged and skipped so one bad image cannot abort the report.
    """
    preloaded = _load_checkpoint_model(ckpt_path, device)
    has_l_value = "L_value" in base.records.columns
    if not has_l_value:
        logger.warning(
            "Dataset CSV has no L_value column; the uncorrected focus score "
            + "cannot be distinguished from the corrected one."
        )
    evals: list[HologramEval] = []
    for i in indices:
        try:
            pil_image = crop_max_square(Image.open(base.paths[i]).convert("RGB"))
            entry = evaluate_checkpoint_on_hologram(
                ckpt_path,
                pil_image,
                device,
                runs=1,
                crop_size=crop_size,
                # CSV Wavelength column is in micrometers
                wavelength=float(base.wavelength[i]) * 1e-6,
                dx=SENSOR_PIXEL_PITCH_M,
                z_true_mm=float(base.z.z_array[i]),
                l_mm=float(base.records["L_value"][i]) if has_l_value else None,
                preloaded=preloaded,
            )
            evals.append(entry)
        except Exception:
            logger.exception(f"Hologram evaluation failed for {base.paths[i]}; skipping.")
    return evals


def _condition_stats(
    condition: str,
    output: TrainingOutput,
    evals: list[HologramEval],
    analysis: AnalysisType,
    checkpoint: Path,
    train_time_s: float,
    trained_with_random_crop: bool,
    mag_corrected_focus: bool,
) -> ConditionStats:
    """Aggregate one condition's training output and hologram evaluations."""
    stats = ConditionStats(
        condition=condition,
        trained_with_random_crop=trained_with_random_crop,
        mag_corrected_focus=mag_corrected_focus,
        checkpoint=checkpoint.name,
        avg_train_loss=output.avg_train_loss,
        avg_val_loss=output.avg_val_loss,
        train_time_s=train_time_s,
    )
    if analysis == AnalysisType.CLASS:
        stats.val_accuracy = output.best_val_metric
        stats.val_mae_mm = output.val_mae_mm
    else:
        stats.val_mae_mm = output.best_val_metric

    focus_vals: list[float] = []
    for e in evals:
        focus = e.focus_tc if mag_corrected_focus else e.focus_tc_uncorrected
        if focus is not None:
            focus_vals.append(focus)
    if focus_vals:
        stats.focus_tc_mean = float(np.mean(focus_vals))
        stats.focus_tc_std = float(np.std(focus_vals))
    errors = [e.abs_error_mm for e in evals if e.abs_error_mm is not None]
    if errors:
        stats.holo_mae_mm = float(np.mean(errors))
    stats.holo_count = len(evals)
    return stats


def run_fix_ablation(
    u_cfg: CompareUserConfig,
    backbone: ModelType,
    holo_count: int = 50,
    seed: int = 42,
) -> FixAblationReport:
    """Train and evaluate the before/after conditions of the two pipeline fixes.

    Trains ``backbone`` twice under ``u_cfg`` — once with the legacy random-crop
    augmentation and once with the current pipeline — then focus-scores both
    checkpoints on the first ``holo_count`` holograms of the shared seeded
    validation split, at both the raw and magnification-corrected depths.
    """
    u_cfg = u_cfg.resolve_paths()
    analysis = AnalysisType.determine(u_cfg.num_classes)
    base = HologramFocusDataset(
        mode=analysis,
        num_classes=u_cfg.num_classes,
        csv_file_strpath=u_cfg.paths.meta_csv_name,
    )

    # training clobbers the per-backbone best checkpoint; keep any existing one
    best = checkpoints_loc() / f"checkpoint_{backbone.name}_best.pth"
    if best.exists():
        backup = checkpoints_loc() / (
            f"checkpoint_{backbone.name}_best_pre_ablation_"
            + f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
        )
        _ = best.replace(backup)
        logger.info(f"Existing best checkpoint moved aside to {backup}")

    console.rule(f"[black on cyan] Ablation 1/2: {backbone.value} with legacy random crop ")
    (legacy_out, legacy_ckpt, legacy_time) = _train_condition(
        u_cfg, backbone, base, legacy_random_crop=True, seed=seed
    )
    console.rule(f"[black on cyan] Ablation 2/2: {backbone.value} with current pipeline ")
    (current_out, current_ckpt, current_time) = _train_condition(
        u_cfg, backbone, base, legacy_random_crop=False, seed=seed
    )

    # the same fixed-seed split both trainings used; indices are already a
    # random permutation, so the first holo_count form a uniform sample
    cfg = u_cfg.to_auto_config(backbone)
    cfg.fixed_seed = True
    (_train_subset, eval_subset) = cfg.setup_loader_indices(base)
    eval_indices = [int(i) for i in list(eval_subset.indices)[:holo_count]]
    device = cfg.device()

    console.rule("[black on cyan] Focus-scoring both checkpoints on validation holograms ")
    legacy_evals = _evaluate_checkpoint(legacy_ckpt, base, eval_indices, device, cfg.crop_size)
    current_evals = _evaluate_checkpoint(current_ckpt, base, eval_indices, device, cfg.crop_size)

    conditions = [
        _condition_stats(
            COND_BEFORE,
            legacy_out,
            legacy_evals,
            analysis,
            legacy_ckpt,
            legacy_time,
            trained_with_random_crop=True,
            mag_corrected_focus=False,
        ),
        _condition_stats(
            COND_CROP_FIX_ONLY,
            current_out,
            current_evals,
            analysis,
            current_ckpt,
            current_time,
            trained_with_random_crop=False,
            mag_corrected_focus=False,
        ),
        _condition_stats(
            COND_AFTER,
            current_out,
            current_evals,
            analysis,
            current_ckpt,
            current_time,
            trained_with_random_crop=False,
            mag_corrected_focus=True,
        ),
    ]

    return FixAblationReport(
        backbone=backbone.value,
        analysis=analysis.value,
        num_classes=u_cfg.num_classes,
        crop_size=u_cfg.crop_size,
        batch_size=u_cfg.batch_size,
        epoch_count=u_cfg.epoch_count,
        device=device,
        dataset_csv=u_cfg.paths.meta_csv_name,
        seed=seed,
        holo_count=holo_count,
        created=datetime.now().isoformat(timespec="seconds"),
        conditions=conditions,
        legacy_evals=legacy_evals,
        current_evals=current_evals,
    )
