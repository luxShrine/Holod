from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

# WARN: Path must be importable at runtime: pyserde wraps @serde class methods with
# beartype, which resolves the annotation on ComparisonReport.save at call time.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from PIL import Image
from rich.table import Table
from serde import serde
from serde.json import to_json
from torchvision.transforms import v2

from holod.core.optics.reconstruction import dlhm_effective_z_mm, focus_score
from holod.core.plots import PlotPred
from holod.infra.dataclasses import (
    AutoConfig,
    Checkpoint,
    CompareUserConfig,
    ModelCheckpoint,
    create_autofocus_model,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import console_ as console
from holod.infra.log import get_logger
from holod.infra.training import (
    create_training_setup,
    train_eval_epoch,
)
from holod.infra.util.image_processing import crop_max_square
from holod.infra.util.paths import checkpoints_loc, report_path
from holod.infra.util.types import (
    SENSOR_PIXEL_PITCH_M,
    AnalysisType,
    DisplayType,
    ModelType,
    UserDevice,
)

if TYPE_CHECKING:
    import torch.nn as nn
    from PIL.Image import Image as ImageType


logger = get_logger(__name__)

# 3-channel pretrained ImageNet backbones; everything else takes a 1-channel hologram
PRETRAINED_BACKBONES: frozenset[ModelType] = frozenset(
    {ModelType.ENET, ModelType.VIT, ModelType.RESNET}
)
VIT_CROP_SIZE: int = 224
# file extensions of trained-model artifacts under checkpoints_loc()
MODEL_FILE_EXTS: frozenset[str] = frozenset({".pth", ".tar"})

# emphasis palette for the hologram-comparison plot: the best model gets the
# accent hue, the rest recede to gray; identity is carried by axis/value labels
PLOT_ACCENT = "#2a78d6"
PLOT_MUTED = "#898781"
PLOT_INK = "#0b0b0b"
PLOT_INK_SOFT = "#52514e"
PLOT_GRID = "#e1e0d9"
PLOT_SURFACE = "#fcfcfb"


@serde
@dataclass
class BackboneStats:
    """Statistics gathered for one backbone under the shared comparison config."""

    backbone: str
    metric_name: str
    input_channels: int = 0
    pretrained: bool = False
    crop_size: int = 0
    total_params: int = 0
    trainable_params: int = 0
    model_size_mb: float = 0.0
    latency_ms_per_img: float | None = None
    throughput_img_per_s: float | None = None
    trained: bool = False
    train_time_s: float | None = None
    time_per_epoch_s: float | None = None
    avg_train_loss: float | None = None
    avg_val_loss: float | None = None
    best_val_metric: float | None = None
    # final-epoch validation MAE between predicted and true bin centers (mm);
    # classification only   for regression best_val_metric is already the MAE
    val_mae_mm: float | None = None
    error: str | None = None


@serde
@dataclass
class ComparisonReport:
    """Per-backbone statistics plus the shared configuration they were measured under."""

    analysis: str
    num_classes: int
    crop_size: int
    batch_size: int
    epoch_count: int
    device: str
    created: str
    stats: list[BackboneStats] = field(default_factory=list)

    def best_index(self) -> int | None:
        """Return the index of the best-performing trained backbone, if any."""
        scored = [
            (i, s.best_val_metric)
            for (i, s) in enumerate(self.stats)
            if s.best_val_metric is not None
        ]
        if not scored:
            return None
        if self.analysis == AnalysisType.CLASS.value:
            # accuracy: higher is better
            return max(scored, key=lambda pair: pair[1])[0]
        # MAE: lower is better
        return min(scored, key=lambda pair: pair[1])[0]

    def to_table(self) -> Table:
        """Render the report as a Rich table, highlighting the best trained backbone."""
        metric_header = "Accuracy" if self.analysis == AnalysisType.CLASS.value else "Val MAE (mm)"
        table = Table(
            title=f"Backbone comparison ({self.analysis}, {self.device}, crop {self.crop_size})"
        )
        table.add_column("Backbone", style="bold")
        table.add_column("Params (M)", justify="right")
        table.add_column("Trainable (M)", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("In ch", justify="right")
        table.add_column("Pretrained", justify="center")
        table.add_column("ms/img", justify="right")
        table.add_column("img/s", justify="right")
        table.add_column("Train time (s)", justify="right")
        table.add_column("s/epoch", justify="right")
        table.add_column("Train loss", justify="right")
        table.add_column("Val loss", justify="right")
        table.add_column(metric_header, justify="right")
        if self.analysis == AnalysisType.CLASS.value:
            # accuracy misses how far off wrong bins are; MAE in mm shows that
            table.add_column("Val MAE (mm)", justify="right")
        table.add_column("Error")

        best = self.best_index()
        for i, s in enumerate(self.stats):
            row: list[str] = [
                s.backbone,
                f"{s.total_params / 1e6:.2f}",
                f"{s.trainable_params / 1e6:.2f}",
                f"{s.model_size_mb:.1f}",
                str(s.input_channels),
                "yes" if s.pretrained else "no",
                _fmt(s.latency_ms_per_img, ".2f"),
                _fmt(s.throughput_img_per_s, ".1f"),
            ]
            row.extend(
                [
                    _fmt(s.train_time_s, ".1f"),
                    _fmt(s.time_per_epoch_s, ".1f"),
                    _fmt(s.avg_train_loss, ".4g"),
                    _fmt(s.avg_val_loss, ".4g"),
                    _fmt(s.best_val_metric, ".4g"),
                ]
            )
            if self.analysis == AnalysisType.CLASS.value:
                row.append(_fmt(s.val_mae_mm, ".4g"))
            row.append(s.error if s.error is not None else "")
            table.add_row(*row, style="bold green" if i == best else None)
        return table

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        """Serialize the report to JSON and CSV, returning both file paths."""
        out_dir = directory if directory is not None else report_path()
        # stamp the filenames with the report's creation time so successive compare
        # runs accumulate instead of overwriting the previous run's statistics
        stamp = datetime.fromisoformat(self.created).strftime("%Y%m%d-%H%M%S")
        json_path = out_dir / f"backbone_comparison_{stamp}.json"
        csv_path = out_dir / f"backbone_comparison_{stamp}.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(to_json(self, ComparisonReport))
        pl.DataFrame([asdict(s) for s in self.stats]).write_csv(csv_path)
        logger.info(f"Comparison report saved to {json_path} and {csv_path}")
        return (json_path, csv_path)

    def plot(self, display: DisplayType = DisplayType.SAVE) -> Path | None:
        """Plot per-backbone training statistics (or static stats) as bar charts."""
        # heavy dep, imported lazily like the CLI does
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plotted = [(i, s) for (i, s) in enumerate(self.stats) if s.error is None]
        if not plotted:
            logger.warning("No successful backbone runs to plot.")
            return None
        best = self.best_index()
        names = [s.backbone for (_i, s) in plotted]
        colors = [PLOT_ACCENT if i == best else PLOT_MUTED for (i, _e) in plotted]

        # one panel per statistic: (subplot title, per-backbone values, format spec)
        panels: list[tuple[str, list[float | None], str]]
        metric_title = (
            "Accuracy (higher is better)"
            if self.analysis == AnalysisType.CLASS.value
            else "Val MAE (mm, lower is better)"
        )
        panels = [
            ("Parameters (M)", [s.total_params / 1e6 for (_i, s) in plotted], ".2f"),
            ("Latency (ms/img)", [s.latency_ms_per_img for (_i, s) in plotted], ".2f"),
            ("Model size (MB)", [s.model_size_mb for (_i, s) in plotted], ".1f"),
            (metric_title, [s.best_val_metric for (_i, s) in plotted], ".4g"),
            (
                "Validation loss (lower is better)",
                [s.avg_val_loss for (_i, s) in plotted],
                ".4g",
            ),
            ("Time per epoch (s)", [s.time_per_epoch_s for (_i, s) in plotted], ".1f"),
        ]
        if any(s.val_mae_mm is not None for (_i, s) in plotted):
            panels.insert(
                1,
                (
                    "Val MAE (mm, lower is better)",
                    [s.val_mae_mm for (_i, s) in plotted],
                    ".4g",
                ),
            )

        fig = make_subplots(
            rows=1,
            cols=len(panels),
            shared_yaxes=True,
            horizontal_spacing=0.06,
            subplot_titles=[title for (title, _values, _spec) in panels],
        )
        for col, (title, values, spec) in enumerate(panels, start=1):
            _ = fig.add_trace(
                go.Bar(
                    y=names,  # pyright: ignore[reportArgumentType]
                    x=values,
                    orientation="h",
                    marker_color=colors,
                    text=[_fmt(v, spec) for v in values],
                    textposition="outside",
                    textfont={"color": PLOT_INK_SOFT},
                    cliponaxis=False,
                    showlegend=False,
                    name=title,
                ),
                row=1,
                col=col,
            )
        stats_kind = "training statistics"
        _ = fig.update_layout(
            title_text=f"Backbone {stats_kind}   {self.analysis}, {self.device}, "
            + f"crop {self.crop_size}",
            paper_bgcolor=PLOT_SURFACE,
            plot_bgcolor=PLOT_SURFACE,
            font={"family": "system-ui, sans-serif", "color": PLOT_INK_SOFT},
            bargap=0.35,
            margin={"t": 80},
        )
        _ = fig.update_xaxes(gridcolor=PLOT_GRID, zeroline=False)
        _ = fig.update_yaxes(gridcolor=PLOT_GRID, autorange="reversed")

        html_path: Path | None = None
        if display is not DisplayType.SHOW:
            stamp = datetime.now().strftime("%H%M%S")
            html_path = report_path(figure=True) / f"backbone_comparison_{stamp}.html"
            fig.write_html(html_path, full_html=True, include_plotlyjs="cdn")
            logger.info(f"Backbone comparison plot saved to {html_path}")
        if display in (DisplayType.SHOW, DisplayType.BOTH):
            fig.show()
        return html_path


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional float, showing a dash for missing values."""
    return "-" if value is None else format(value, spec)


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (total, trainable)


def _model_size_mb(model: nn.Module) -> float:
    """Approximate the size of a model's parameters and buffers in megabytes."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def _measure_inference(
    model: nn.Module,
    channels: int,
    crop_size: int,
    device: str,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[float, float]:
    """Time single-image forward passes, returning (latency ms/img, throughput img/s)."""
    dev = torch.device(device)
    _ = model.to(dev)
    _ = model.eval()
    dummy = torch.rand(1, channels, crop_size, crop_size, device=dev)
    with torch.no_grad():
        for _i in range(warmup):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _i in range(iters):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        per_img = (time.perf_counter() - start) / iters
    return (per_img * 1e3, 1.0 / per_img)


def resolve_crop_size(backbone: ModelType, crop_size: int) -> int:
    """Return a crop size compatible with the backbone (ViT is fixed to 224)."""
    if backbone == ModelType.VIT and crop_size != VIT_CROP_SIZE:
        logger.warning(
            f"Backbone {backbone.value} requires a crop size of {VIT_CROP_SIZE}, "
            + f"overriding {crop_size} for its comparison run."
        )
        return VIT_CROP_SIZE
    return crop_size


def collect_model_stats(a_cfg: AutoConfig) -> BackboneStats:
    """Collect static architecture and inference-speed statistics for one backbone."""
    channels = 3 if a_cfg.backbone in PRETRAINED_BACKBONES else 1
    model = create_autofocus_model(a_cfg)
    (total, trainable) = _count_parameters(model)
    entry = BackboneStats(
        backbone=a_cfg.backbone.value,
        metric_name=_metric_name(a_cfg.analysis),
        input_channels=channels,
        pretrained=a_cfg.backbone in PRETRAINED_BACKBONES,
        crop_size=a_cfg.crop_size,
        total_params=total,
        trainable_params=trainable,
        model_size_mb=_model_size_mb(model),
    )
    (entry.latency_ms_per_img, entry.throughput_img_per_s) = _measure_inference(
        model, channels, a_cfg.crop_size, a_cfg.device()
    )
    return entry


def _metric_name(analysis: AnalysisType) -> str:
    """Name of the validation metric reported for an analysis type."""
    return "accuracy" if analysis == AnalysisType.CLASS else "mae_mm"


def _fill_training_stats(
    entry: BackboneStats, a_cfg: AutoConfig, base: HologramFocusDataset
) -> None:
    """Train one backbone with the standard pipeline and record its results in-place."""
    core_trainer = create_training_setup(base, a_cfg)
    best_val_metric = float("inf") if a_cfg.analysis == AnalysisType.REG else -float("inf")
    start = time.perf_counter()
    output = train_eval_epoch(core_trainer, best_val_metric, None)
    elapsed = time.perf_counter() - start
    entry.trained = True
    entry.train_time_s = elapsed
    entry.time_per_epoch_s = elapsed / max(a_cfg.epoch_count, 1)
    entry.avg_train_loss = output.avg_train_loss
    entry.avg_val_loss = output.avg_val_loss
    entry.best_val_metric = output.best_val_metric
    entry.val_mae_mm = output.val_mae_mm
    # persist the per-sample predictions exactly like an individual `holod train`
    # run does, using the same dataset the backbone was trained on
    bin_centers = base.bin_centers if a_cfg.analysis == AnalysisType.CLASS else None
    (z_avg, z_std) = core_trainer.get_std_avg(a_cfg.analysis)
    plot_pred = PlotPred.from_z_preds(
        core_trainer=core_trainer,
        bin_centers=bin_centers,
        training_output=output,
        bin_edges=base.bin_edges,
        z_avg_mm=z_avg,
        z_std_mm=z_std,
    )
    plot_pred.save_to_file()


def compare_backbones(
    u_cfg: CompareUserConfig,
    backbones: list[ModelType] | None = None,
) -> ComparisonReport:
    """Compare backbones under one shared config, training each with its own model config.

    Args:
        u_cfg: Shared configuration plus the per-model sections; paths are
            resolved here (``resolve_paths`` is idempotent) before any use.
        backbones: Which backbones to compare; ``None`` compares every backbone
            configured in ``u_cfg``.

    Returns:
        A ``ComparisonReport`` with one ``BackboneStats`` entry per requested backbone.
        A backbone that fails is recorded with its ``error`` set instead of aborting
        the remaining runs.

    """
    u_cfg = u_cfg.resolve_paths()
    chosen = u_cfg.configured_backbones() if backbones is None else list(backbones)
    device = UserDevice.determine(u_cfg.device)
    analysis = AnalysisType.determine(u_cfg.num_classes)
    base = HologramFocusDataset(
        mode=analysis,
        num_classes=u_cfg.num_classes,
        csv_file_strpath=u_cfg.paths.meta_csv_name,
    )

    stats: list[BackboneStats] = []
    for backbone in chosen:
        console.rule(f"[black on cyan] Comparing backbone: {backbone.value} ")
        cfg = u_cfg.to_auto_config(backbone)
        entry = BackboneStats(backbone=backbone.value, metric_name=_metric_name(cfg.analysis))
        try:
            entry = collect_model_stats(cfg)
            _fill_training_stats(entry, cfg, base)
        except Exception as exc:
            entry.error = f"{type(exc).__name__}: {exc}"
            logger.exception(f"Comparison failed for backbone '{backbone.value}'.")
        stats.append(entry)

    return ComparisonReport(
        analysis=analysis.value,
        num_classes=u_cfg.num_classes,
        crop_size=u_cfg.crop_size,
        batch_size=u_cfg.batch_size,
        epoch_count=u_cfg.epoch_count,
        device=device.value,
        created=datetime.now().isoformat(timespec="seconds"),
        stats=stats,
    )


@serde
@dataclass
class HologramEval:
    """Result of evaluating one trained model checkpoint against a single hologram."""

    model_name: str
    backbone: str = ""
    analysis: str = ""
    num_classes: int = 0
    crop_size: int = 0
    runs: int = 0
    z_pred_mm: float | None = None
    latency_ms_per_run: float | None = None
    # gradient-Tamura sharpness of the reconstruction at z_pred; higher is sharper
    # (see focus_score in core/optics/reconstruction.py)
    focus_tc: float | None = None
    # sharpness of the reconstruction at the *raw* predicted depth, without the
    # magnification correction — what the pre-fix evaluation measured; populated
    # only when the source->screen distance L is known
    focus_tc_uncorrected: float | None = None
    abs_error_mm: float | None = None
    # signed prediction error (positive = overshoot); exposes systematic bias
    signed_error_mm: float | None = None
    # absolute error as a percentage of the true depth
    rel_error_pct: float | None = None
    # absolute error in units of the classifier's bin width, i.e. how many bins
    # off the expected-depth prediction is (classification only)
    err_in_bins: float | None = None
    error: str | None = None


@serde
@dataclass
class HologramComparisonReport:
    """Per-checkpoint hologram evaluations plus the shared measurement setup."""

    image: str
    device: str
    runs: int
    wavelength_m: float
    dx_m: float
    z_true_mm: float | None
    created: str
    # DLHM source->screen distance L (mm); when known, focus scoring reconstructs
    # at the plane-wave-equivalent depth M*(L - z) instead of z directly
    l_mm: float | None = None
    evals: list[HologramEval] = field(default_factory=list)

    def best_index(self) -> int | None:
        """Return the index of the best evaluation.

        Ranked by absolute depth error when a ground-truth depth is known (lower is
        better), otherwise by the label-free gradient-Tamura focus score (higher is
        sharper).
        """
        if self.z_true_mm is not None:
            scored = [
                (i, e.abs_error_mm)
                for (i, e) in enumerate(self.evals)
                if e.abs_error_mm is not None
            ]
            if not scored:
                return None
            return min(scored, key=lambda pair: pair[1])[0]
        scored = [(i, e.focus_tc) for (i, e) in enumerate(self.evals) if e.focus_tc is not None]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[1])[0]

    def to_table(self) -> Table:
        """Render the report as a Rich table, highlighting the best evaluation."""
        table = Table(
            title=f"Hologram evaluation ({Path(self.image).name}, {self.device}, "
            + f"{self.runs} runs/model)"
        )
        table.add_column("Model", style="bold")
        table.add_column("Backbone")
        table.add_column("Analysis", justify="center")
        table.add_column("z pred (mm)", justify="right")
        if self.z_true_mm is not None:
            table.add_column("|err| (mm)", justify="right")
            table.add_column("err (mm)", justify="right")
            table.add_column("err %", justify="right")
            table.add_column("err (bins)", justify="right")
        table.add_column("ms/run", justify="right")
        table.add_column("Focus TC", justify="right")
        table.add_column("Error")

        best = self.best_index()
        for i, e in enumerate(self.evals):
            row: list[str] = [
                e.model_name,
                e.backbone,
                e.analysis,
                _fmt(e.z_pred_mm, ".4g"),
            ]
            if self.z_true_mm is not None:
                row.extend(
                    [
                        _fmt(e.abs_error_mm, ".4g"),
                        _fmt(e.signed_error_mm, "+.4g"),
                        _fmt(e.rel_error_pct, ".1f"),
                        _fmt(e.err_in_bins, ".2f"),
                    ]
                )
            row.extend(
                [
                    _fmt(e.latency_ms_per_run, ".2f"),
                    _fmt(e.focus_tc, ".4g"),
                    e.error if e.error is not None else "",
                ]
            )
            table.add_row(*row, style="bold green" if i == best else None)
        return table

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        """Serialize the report to JSON and CSV, returning both file paths."""
        out_dir = directory if directory is not None else report_path()
        json_path = out_dir / "hologram_comparison.json"
        csv_path = out_dir / "hologram_comparison.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(to_json(self, HologramComparisonReport))
        pl.DataFrame([asdict(e) for e in self.evals]).write_csv(csv_path)
        logger.info(f"Hologram comparison report saved to {json_path} and {csv_path}")
        return (json_path, csv_path)

    def plot(self, display: DisplayType = DisplayType.SAVE) -> Path | None:
        """Plot per-model depth predictions and reconstruction error as bar charts."""
        # heavy dep, imported lazily like the CLI does
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plotted = [(i, e) for (i, e) in enumerate(self.evals) if e.z_pred_mm is not None]
        if not plotted:
            logger.warning("No successful evaluations to plot.")
            return None
        best = self.best_index()
        names = [e.model_name for (_i, e) in plotted]
        colors = [PLOT_ACCENT if i == best else PLOT_MUTED for (i, _e) in plotted]

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.06,
            subplot_titles=[
                "Predicted depth (mm)",
                "Focus sharpness (gradient Tamura, higher is better)",
            ],
        )
        _ = fig.add_trace(
            go.Bar(
                y=names,  # pyright: ignore[reportArgumentType]
                x=[e.z_pred_mm for (_i, e) in plotted],
                orientation="h",
                marker_color=colors,
                text=[_fmt(e.z_pred_mm, ".4g") for (_i, e) in plotted],
                textposition="outside",
                textfont={"color": PLOT_INK_SOFT},
                cliponaxis=False,
                showlegend=False,
                name="z pred (mm)",
            ),
            row=1,
            col=1,
        )
        _ = fig.add_trace(
            go.Bar(
                y=names,  # pyright: ignore[reportArgumentType]
                x=[e.focus_tc for (_i, e) in plotted],
                orientation="h",
                marker_color=colors,
                text=[_fmt(e.focus_tc, ".4g") for (_i, e) in plotted],
                textposition="outside",
                textfont={"color": PLOT_INK_SOFT},
                cliponaxis=False,
                showlegend=False,
                name="Tamura TC",
            ),
            row=1,
            col=2,
        )
        if self.z_true_mm is not None:
            _ = fig.add_vline(
                x=self.z_true_mm,
                line_dash="dash",
                line_color=PLOT_INK,
                line_width=1,
                annotation_text=f"ground truth {self.z_true_mm:g} mm",
                annotation_position="bottom right",
                row=1,
                col=1,
            )
        _ = fig.update_layout(
            title_text=f"Depth prediction per model   {Path(self.image).name} ({self.device})",
            paper_bgcolor=PLOT_SURFACE,
            plot_bgcolor=PLOT_SURFACE,
            font={"family": "system-ui, sans-serif", "color": PLOT_INK_SOFT},
            bargap=0.35,
            margin={"t": 80},
        )
        _ = fig.update_xaxes(gridcolor=PLOT_GRID, zeroline=False)
        _ = fig.update_yaxes(gridcolor=PLOT_GRID, autorange="reversed")

        html_path: Path | None = None
        if display is not DisplayType.SHOW:
            stamp = datetime.now().strftime("%H%M%S")
            html_path = report_path(figure=True) / f"hologram_comparison_{stamp}.html"
            fig.write_html(html_path, full_html=True, include_plotlyjs="cdn")
            logger.info(f"Hologram comparison plot saved to {html_path}")
        if display in (DisplayType.SHOW, DisplayType.BOTH):
            fig.show()
        return html_path


def _load_checkpoint_model(
    ckpt_path: Path, device: str
) -> tuple[nn.Module, AutoConfig, Checkpoint]:
    """Rebuild an eval-mode model from a saved ``Checkpoint``/``ModelCheckpoint`` file."""
    with torch.serialization.safe_globals(ModelCheckpoint.SAFE_GLOBALS):
        torch_dict: dict[str, Any] = torch.load(ckpt_path, weights_only=True, map_location=device)
    # best-model .pth files hold a flat Checkpoint dict; epoch .tar checkpoints nest it
    ckpt = Checkpoint.from_dict(torch_dict.get("checkpoint", torch_dict))
    a_cfg = AutoConfig.default(
        analysis=AnalysisType.REG if ckpt.bin_centers is None else AnalysisType.CLASS,
        backbone=ckpt.model_type,
        num_classes=ckpt.num_classes,
    )
    model = create_autofocus_model(a_cfg).to(torch.device(device))
    incompatible = model.load_state_dict(ckpt.model_state_dict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(f"State dict load mismatch for {ckpt_path.name}: {incompatible}")
    if a_cfg.analysis == AnalysisType.REG and (ckpt.z_avg is None or ckpt.z_std is None):
        # checkpoints written before z_avg/z_std were saved cannot be de-normalized
        logger.warning(
            f"Checkpoint {ckpt_path.name} lacks z_avg/z_std normalization stats; "
            + "regression output will be taken as-is."
        )
    _ = model.eval()
    return (model, a_cfg, ckpt)


def _hologram_tensor(
    pil_image: ImageType, backbone: ModelType, crop_size: int, device: str
) -> torch.Tensor:
    """Preprocess a hologram like ``TransformedDataset`` does for this backbone."""
    transforms: list[nn.Module] = [
        v2.PILToTensor(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(size=crop_size),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if backbone in PRETRAINED_BACKBONES:
        transforms.extend(
            [
                v2.Grayscale(num_output_channels=3),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transforms.append(v2.Grayscale(num_output_channels=1))
    # shape [batch=1, C, H, W]
    return v2.Compose(transforms)(pil_image).unsqueeze(0).to(torch.device(device))


def _predict_depth_mm(
    model: nn.Module,
    input_tensor: torch.Tensor,
    analysis: AnalysisType,
    ckpt: Checkpoint,
) -> float:
    """Run one forward pass and map the output to a depth in millimeters.

    Classification models are decoded to the probability-weighted mean of the bin
    centers rather than the argmax center: weakly separated checkpoints often share
    an argmax bin (collapsing them to identical predictions), while the expectation
    varies continuously with the logits and so discriminates between checkpoints.
    """
    with torch.no_grad():
        pred = model(input_tensor)  # shape [Batch, Classes]
    if analysis == AnalysisType.CLASS:
        probs = torch.softmax(pred, dim=1).squeeze(0).double().cpu().numpy()
        centers = np.asarray(ckpt.bin_centers, dtype=np.float64)
        return float(np.dot(probs, centers))
    if ckpt.z_avg is not None and ckpt.z_std is not None:
        # reverse the label normalization applied by RegLabelTransform during training
        return float(pred.squeeze()) * ckpt.z_std + ckpt.z_avg
    return float(pred.squeeze())


def evaluate_checkpoint_on_hologram(
    ckpt_path: Path,
    pil_image: ImageType,
    device: str,
    runs: int,
    crop_size: int,
    wavelength_m: float,
    dx_m: float,
    z_true_mm: float | None,
    l_mm: float | None = None,
    preloaded: tuple[nn.Module, AutoConfig, Checkpoint] | None = None,
) -> HologramEval:
    """Evaluate one checkpoint on a hologram, timing ``runs`` repeated predictions.

    The predicted depth is scored label-free with ``focus_score`` (gradient-Tamura
    sharpness of the reconstruction; higher is sharper). When the DLHM source->screen
    distance ``l_mm`` is known, the reconstruction is done at the plane-wave-equivalent
    depth ``dlhm_effective_z_mm(z_pred, l_mm)`` — without it the raw predicted depth is
    used, which does not refocus DLHM holograms and makes the score uninformative —
    and the raw-depth score is additionally recorded as ``focus_tc_uncorrected``.
    Pass ``preloaded`` to reuse an already-loaded model when scoring many holograms
    against the same checkpoint.
    """
    entry = HologramEval(model_name=ckpt_path.name, runs=runs)
    (model, cfg, ckpt) = (
        preloaded if preloaded is not None else _load_checkpoint_model(ckpt_path, device)
    )
    entry.backbone = cfg.backbone.value
    entry.analysis = cfg.analysis.value
    entry.num_classes = cfg.num_classes
    entry.crop_size = resolve_crop_size(cfg.backbone, crop_size)

    input_tensor = _hologram_tensor(pil_image, cfg.backbone, entry.crop_size, device)
    z_pred_mm = _predict_depth_mm(model, input_tensor, cfg.analysis, ckpt)  # warmup
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _i in range(runs):
        z_pred_mm = _predict_depth_mm(model, input_tensor, cfg.analysis, ckpt)
    if device == "cuda":
        torch.cuda.synchronize()
    entry.latency_ms_per_run = (time.perf_counter() - start) / max(runs, 1) * 1e3
    entry.z_pred_mm = z_pred_mm

    intensity = np.asarray(pil_image.convert("L"), np.float32) / 255.0
    z_recon_mm = dlhm_effective_z_mm(z_pred_mm, l_mm) if l_mm is not None else z_pred_mm
    entry.focus_tc = focus_score(intensity, wavelength_m, z_recon_mm * 1e-3, dx_m)
    if l_mm is not None:
        # score the raw depth too (the pre-fix behavior); reuse the corrected value
        # when the degenerate-geometry fallback made both depths identical
        entry.focus_tc_uncorrected = (
            entry.focus_tc
            if z_recon_mm == z_pred_mm
            else focus_score(intensity, wavelength_m, z_pred_mm * 1e-3, dx_m)
        )
    if z_true_mm is not None:
        entry.abs_error_mm = abs(z_pred_mm - z_true_mm)
        entry.signed_error_mm = z_pred_mm - z_true_mm
        if z_true_mm != 0:
            entry.rel_error_pct = entry.abs_error_mm / abs(z_true_mm) * 100
        if cfg.analysis == AnalysisType.CLASS and ckpt.bin_centers is not None:
            centers = np.asarray(ckpt.bin_centers, dtype=np.float64)
            if len(centers) >= 2:
                # bins come from a uniform linspace, so one gap is the bin width
                bin_width_mm = float(centers[1] - centers[0])
                if bin_width_mm > 0:
                    entry.err_in_bins = entry.abs_error_mm / bin_width_mm
    return entry


def compare_on_hologram(
    img_file_path: str,
    ckpt_paths: list[Path] | None = None,
    runs: int = 5,
    crop_size: int = 224,
    wavelength_m: float = 405e-9,
    dx_m: float = SENSOR_PIXEL_PITCH_M,
    z_true_mm: float | None = None,
    l_mm: float | None = None,
    device: str | None = None,
) -> HologramComparisonReport:
    """Run comparison test evaluations of multiple trained models on one hologram.

    Args:
        img_file_path: Path to the hologram image to evaluate on.
        ckpt_paths: Checkpoint files to compare; ``None`` discovers every ``.pth``/``.tar``
            under ``checkpoints_loc()``.
        runs: How many repeated predictions to time per model.
        crop_size: Center-crop applied before inference (ViT is forced to 224).
        wavelength_m: Wavelength of the capture illumination (meters, e.g. 405e-9 for
            the project's 405 nm laser), for focus scoring.
        dx_m: Pixel pitch of the sensor (meters), for focus scoring; defaults to the
            3.8 µm pitch of the project's capture sensor.
        z_true_mm: Optional ground-truth depth (mm); enables absolute-error ranking.
        l_mm: Optional DLHM source->screen distance L (mm, the dataset's ``L_value``).
            When given, focus scoring reconstructs at the plane-wave-equivalent depth
            ``M * (L - z)``   strongly recommended, as raw depths do not refocus
            DLHM holograms.
        device: ``"cuda"``/``"cpu"``; ``None`` picks CUDA when available.

    Returns:
        A ``HologramComparisonReport`` with one ``HologramEval`` per checkpoint. A
        checkpoint that fails is recorded with its ``error`` set instead of aborting
        the remaining evaluations.

    """
    if ckpt_paths is None:
        ckpt_paths = sorted(
            path for path in checkpoints_loc().iterdir() if path.suffix in MODEL_FILE_EXTS
        )
    if not ckpt_paths:
        raise FileNotFoundError(f"No model files ({set(MODEL_FILE_EXTS)}) in {checkpoints_loc()}")
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    pil_image = crop_max_square(Image.open(img_file_path).convert("RGB"))

    evals: list[HologramEval] = []
    for ckpt_path in ckpt_paths:
        console.rule(f"[black on cyan] Evaluating model: {ckpt_path.name} ")
        try:
            entry = evaluate_checkpoint_on_hologram(
                ckpt_path, pil_image, dev, runs, crop_size, wavelength_m, dx_m, z_true_mm, l_mm
            )
        except Exception as exc:
            entry = HologramEval(model_name=ckpt_path.name, runs=runs)
            entry.error = f"{type(exc).__name__}: {exc}"
            logger.exception(f"Hologram evaluation failed for '{ckpt_path.name}'.")
        evals.append(entry)

    return HologramComparisonReport(
        image=Path(img_file_path).as_posix(),
        device=dev,
        runs=runs,
        wavelength_m=wavelength_m,
        dx_m=dx_m,
        z_true_mm=z_true_mm,
        created=datetime.now().isoformat(timespec="seconds"),
        l_mm=l_mm,
        evals=evals,
    )
