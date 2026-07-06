from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
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

from holod.core.optics.reconstruction import recon_inline
from holod.infra.dataclasses import AutoConfig
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import console_ as console
from holod.infra.log import get_logger
from holod.infra.training import Checkpoint, ModelCheckpoint, train_eval_epoch, transform_ds
from holod.infra.util.image_processing import crop_max_square
from holod.infra.util.paths import checkpoints_loc, report_path
from holod.infra.util.types import AnalysisType, DisplayType, ModelType

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
    trained: bool
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
        metric_header = "Accuracy" if self.analysis == AnalysisType.CLASS.value else "Val MAE (m)"
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
        if self.trained:
            table.add_column("Train time (s)", justify="right")
            table.add_column("s/epoch", justify="right")
            table.add_column("Train loss", justify="right")
            table.add_column("Val loss", justify="right")
            table.add_column(metric_header, justify="right")
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
            if self.trained:
                row.extend(
                    [
                        _fmt(s.train_time_s, ".1f"),
                        _fmt(s.time_per_epoch_s, ".1f"),
                        _fmt(s.avg_train_loss, ".4g"),
                        _fmt(s.avg_val_loss, ".4g"),
                        _fmt(s.best_val_metric, ".4g"),
                    ]
                )
            row.append(s.error if s.error is not None else "")
            table.add_row(*row, style="bold green" if i == best else None)
        return table

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        """Serialize the report to JSON and CSV, returning both file paths."""
        out_dir = directory if directory is not None else report_path()
        json_path = out_dir / "backbone_comparison.json"
        csv_path = out_dir / "backbone_comparison.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(to_json(self, ComparisonReport))
        pl.DataFrame([asdict(s) for s in self.stats]).write_csv(csv_path)
        logger.info(f"Comparison report saved to {json_path} and {csv_path}")
        return (json_path, csv_path)


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
    model = a_cfg.create_model()
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
    return "accuracy" if analysis == AnalysisType.CLASS else "mae_m"


def _fill_training_stats(
    entry: BackboneStats, a_cfg: AutoConfig, base: HologramFocusDataset
) -> None:
    """Train one backbone with the standard pipeline and record its results in-place."""
    core_trainer = transform_ds(base, a_cfg)
    best_val_metric = float("inf") if a_cfg.analysis == AnalysisType.REG else -float("inf")
    start = time.perf_counter()
    output = train_eval_epoch(core_trainer, a_cfg, best_val_metric, None)
    elapsed = time.perf_counter() - start
    entry.trained = True
    entry.train_time_s = elapsed
    entry.time_per_epoch_s = elapsed / max(a_cfg.epoch_count, 1)
    entry.avg_train_loss = output.avg_train_loss
    entry.avg_val_loss = output.avg_val_loss
    entry.best_val_metric = output.best_val_metric


def compare_backbones(
    a_cfg: AutoConfig,
    backbones: list[ModelType] | None = None,
    run_training: bool = True,
) -> ComparisonReport:
    """Compare backbones under one shared config, optionally training each one.

    Args:
        a_cfg: Shared runtime configuration; its ``backbone`` field is replaced per run.
        backbones: Which backbones to compare; ``None`` compares every ``ModelType``.
        run_training: When ``False``, only architecture and inference statistics are
            collected — no dataset is loaded and no training happens.

    Returns:
        A ``ComparisonReport`` with one ``BackboneStats`` entry per requested backbone.
        A backbone that fails is recorded with its ``error`` set instead of aborting
        the remaining runs.

    """
    chosen = list(ModelType) if backbones is None else list(backbones)
    base: HologramFocusDataset | None = None
    if run_training:
        base = HologramFocusDataset(
            mode=a_cfg.analysis,
            num_classes=a_cfg.num_classes,
            csv_file_strpath=a_cfg.meta_csv_strpath,
        )

    stats: list[BackboneStats] = []
    for backbone in chosen:
        console.rule(f"[black on cyan] Comparing backbone: {backbone.value} ")
        cfg = replace(
            a_cfg,
            backbone=backbone,
            crop_size=resolve_crop_size(backbone, a_cfg.crop_size),
            data={},
            optimizer={},
        )
        entry = BackboneStats(backbone=backbone.value, metric_name=_metric_name(cfg.analysis))
        try:
            entry = collect_model_stats(cfg)
            if base is not None:
                _fill_training_stats(entry, cfg, base)
        except Exception as exc:
            entry.error = f"{type(exc).__name__}: {exc}"
            logger.exception(f"Comparison failed for backbone '{backbone.value}'.")
        stats.append(entry)

    return ComparisonReport(
        analysis=a_cfg.analysis.value,
        num_classes=a_cfg.num_classes,
        crop_size=a_cfg.crop_size,
        batch_size=a_cfg.batch_size,
        epoch_count=a_cfg.epoch_count if run_training else 0,
        device=a_cfg.device(),
        trained=run_training,
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
    nrmse: float | None = None
    psnr_db: float | None = None
    abs_error_mm: float | None = None
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
    evals: list[HologramEval] = field(default_factory=list)

    def best_index(self) -> int | None:
        """Return the index of the best evaluation.

        Ranked by absolute depth error when a ground-truth depth is known, otherwise by
        the label-free forward-reconstruction NRMSE (lower is better for both).
        """
        if self.z_true_mm is not None:
            scored = [
                (i, e.abs_error_mm)
                for (i, e) in enumerate(self.evals)
                if e.abs_error_mm is not None
            ]
        else:
            scored = [(i, e.nrmse) for (i, e) in enumerate(self.evals) if e.nrmse is not None]
        if not scored:
            return None
        return min(scored, key=lambda pair: pair[1])[0]

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
        table.add_column("ms/run", justify="right")
        table.add_column("NRMSE", justify="right")
        table.add_column("PSNR (dB)", justify="right")
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
                row.append(_fmt(e.abs_error_mm, ".4g"))
            row.extend(
                [
                    _fmt(e.latency_ms_per_run, ".2f"),
                    _fmt(e.nrmse, ".4g"),
                    _fmt(e.psnr_db, ".2f"),
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
            subplot_titles=("Predicted depth (mm)", "Reconstruction NRMSE (lower is better)"),
        )
        _ = fig.add_trace(
            go.Bar(
                y=names,
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
                y=names,
                x=[e.nrmse for (_i, e) in plotted],
                orientation="h",
                marker_color=colors,
                text=[_fmt(e.nrmse, ".4g") for (_i, e) in plotted],
                textposition="outside",
                textfont={"color": PLOT_INK_SOFT},
                cliponaxis=False,
                showlegend=False,
                name="NRMSE",
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
            title_text=f"Depth prediction per model — {Path(self.image).name} ({self.device})",
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
    # best-model .pth files hold a flat Checkpoint dict; latest_checkpoint.tar nests it
    ckpt = Checkpoint.from_dict(torch_dict.get("checkpoint", torch_dict))
    cfg = AutoConfig(
        analysis=AnalysisType.REG if ckpt.bin_centers is None else AnalysisType.CLASS,
        backbone=ckpt.model_type,
        num_classes=ckpt.num_classes,
    )
    model = cfg.create_model().to(torch.device(device))
    incompatible = model.load_state_dict(ckpt.model_state_dict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(f"State dict load mismatch for {ckpt_path.name}: {incompatible}")
    if cfg.analysis == AnalysisType.REG and (ckpt.z_avg is None or ckpt.z_std is None):
        # checkpoints written before z_avg/z_std were saved cannot be de-normalized
        logger.warning(
            f"Checkpoint {ckpt_path.name} lacks z_avg/z_std normalization stats; "
            + "regression output will be taken as-is."
        )
    _ = model.eval()
    return (model, cfg, ckpt)


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
    """Run one forward pass and map the output to a depth in millimeters."""
    with torch.no_grad():
        pred = model(input_tensor)  # shape [Batch, Classes]
    if analysis == AnalysisType.CLASS:
        probs = torch.softmax(pred, dim=1)
        index = int(probs.argmax(1).item())
        centers = np.asarray(ckpt.bin_centers, dtype=np.float64)
        return float(centers[index])
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
    wavelength: float,
    dx: float,
    z_true_mm: float | None,
) -> HologramEval:
    """Evaluate one checkpoint on a hologram, timing ``runs`` repeated predictions.

    The predicted depth is scored by propagating the hologram to that depth with
    ``recon_inline`` and comparing the synthesized hologram against the measured one.
    """
    entry = HologramEval(model_name=ckpt_path.name, runs=runs)
    (model, cfg, ckpt) = _load_checkpoint_model(ckpt_path, device)
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
    (_amp, _phase, entry.nrmse, entry.psnr_db) = recon_inline(
        intensity, wavelength=wavelength, z=z_pred_mm * 1e-3, px=dx
    )
    if z_true_mm is not None:
        entry.abs_error_mm = abs(z_pred_mm - z_true_mm)
    return entry


def compare_on_hologram(
    img_file_path: str,
    ckpt_paths: list[Path] | None = None,
    runs: int = 5,
    crop_size: int = 224,
    wavelength: float = 530e-9,
    dx: float = 1e-6,
    z_true_mm: float | None = None,
    device: str | None = None,
) -> HologramComparisonReport:
    """Run comparison test evaluations of multiple trained models on one hologram.

    Args:
        img_file_path: Path to the hologram image to evaluate on.
        ckpt_paths: Checkpoint files to compare; ``None`` discovers every ``.pth``/``.tar``
            under ``checkpoints_loc()``.
        runs: How many repeated predictions to time per model.
        crop_size: Center-crop applied before inference (ViT is forced to 224).
        wavelength: Wavelength of the capture illumination (m), for reconstruction scoring.
        dx: Pixel pitch of the sensor (m), for reconstruction scoring.
        z_true_mm: Optional ground-truth depth (mm); enables absolute-error ranking.
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
                ckpt_path, pil_image, dev, runs, crop_size, wavelength, dx, z_true_mm
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
        wavelength_m=wavelength,
        dx_m=dx,
        z_true_mm=z_true_mm,
        created=datetime.now().isoformat(timespec="seconds"),
        evals=evals,
    )
