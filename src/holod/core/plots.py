from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.figure_factory as ff  # type: ignore
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from polars import DataFrame
from scipy.stats import gaussian_kde
from serde import serde
from serde.json import from_json, to_json
from sklearn.metrics import confusion_matrix

from holod.infra.dataclasses import CoreTrainer, TrainingOutput, TrainingRepeatConfig
from holod.infra.log import get_logger
from holod.infra.util.paths import report_path
from holod.infra.util.prog_helper import track_progress
from holod.infra.util.types import AnalysisType, Arr64, DisplayType, Plots

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

logger = get_logger(__name__)


@serde
class PlotMeta:
    """Plotting metadata storage."""

    title: str
    caption: str
    tags: list[str]  # e.g., ["autofocus", "fft"]
    slug: str = field(init=False, default_factory=str)
    kind: Literal["result", "diagnostic"] = "result"
    thumb: str | None = field(init=False, default_factory=str)
    page: str | None = field(init=False, default_factory=str)

    def __post_init__(self):
        """Create filename based on title."""
        self.slug = (Path(self.title.replace(" ", "_"))).as_posix()
        self.thumb = f"/plots/{self.slug}.png"
        self.page = f"/plots/{self.slug}.html"


@dataclass
class PlotCollection:
    meta: PlotMeta
    figure: go.Figure


@serde
class PlotPred:
    """Class for storing plotting information."""

    z_val_pred: npt.NDArray[np.float64]
    z_val: npt.NDArray[np.float64]
    z_train_pred: npt.NDArray[np.float64]
    z_train: npt.NDArray[np.float64]
    zerr_train: npt.NDArray[np.float64]
    zerr_val: npt.NDArray[np.float64]
    bin_edges: npt.NDArray[np.float64] | None
    titles: dict[str, str]
    display: str
    analysis: str
    repeat_config: TrainingRepeatConfig

    def save_to_file(self):
        # backbone + full date in the name: back-to-back runs (e.g. `holod compare`
        # training several backbones) must not overwrite each other's predictions
        backbone = self.repeat_config.user_config.train.backbone or "unknown"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = report_path() / f"plot_info_{backbone}_{stamp}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            plot_info_json = to_json(self, PlotPred)
            f.write(plot_info_json)
        logger.info(f"Training predictions saved to {file_path}")

    @classmethod
    def load_from_file(cls, path_to_file: Path):
        json_s = path_to_file.read_text()
        return from_json(cls, json_s)

    # TODO: incorporate into the classification func <luxShrine>
    def plot_violin_depth_bins(self) -> list[PlotCollection] | None:
        """Plot a violin distribution of prediction errors grouped by depth bins."""
        z: npt.NDArray
        z_pred: npt.NDArray
        export: list[PlotCollection] = []
        for step in ["train", "validation"]:
            if step == "train":
                z = self.z_train
                z_pred = -self.z_train_pred
                meta = PlotMeta(
                    title=self.titles["reg_vio_train"],
                    caption="Violin plot concerning the training error of predictions.",
                    tags=["Error", "Training"],
                )
            elif step == "validation":
                z = self.z_val
                z_pred = -self.z_val_pred
                meta = PlotMeta(
                    title=self.titles["reg_vio_validation"],
                    caption="Violin plot concerning the evaluation error of predictions.",
                    tags=["Error", "Evalutation"],
                )
            else:
                raise Exception(f"Unknown step '{step}' passed.")

            assert z_pred.shape == z.shape, "Vectors must match"
            depth_um = z * 1e6
            err_um = (z_pred - z) * 1e6

            # choose depth bins so each violin has ~50 points, tweak divisor
            n_bins = max(10, len(depth_um) // 50)
            np.linspace(depth_um.min(), depth_um.max(), n_bins + 1, dtype=np.float64)
            bin_idx = (
                np.digitize(depth_um, self.bin_edges) - 1  # pyright: ignore[reportCallIssue, reportArgumentType]
            )  # → 0 … n_bins-1

            df = pl.DataFrame(
                {
                    "bin": bin_idx,
                    "err_um": err_um,
                }
            )

            fig = go.Figure(go.Violin(y=df["err_um"], x=df["bin"], width=0.9))

            _ = fig.update_layout(
                yaxis_zeroline=True,
                title_text=meta.title,
                xaxis_title_text="True focus depth (µm)",
                yaxis_title_text="(pred true) (µm)",
                legend_title_text="Legend",
                hovermode="closest",
            )

            export.append(PlotCollection(meta, fig))
        return export

    def confusion_matrix(self, model_name: str):
        """Confusion Matrix plot."""
        # Ensure bin_edges is not None and is a numpy array for classification
        if self.bin_edges is None:
            raise Exception("bin_edges not provided in PlotPred. Plotting failed.")
        bin_edges = np.array(self.bin_edges, dtype=np.float64)
        assert isinstance(bin_edges[0], np.float64), (
            f"bin_edges contains something other than np.float64, found {type(bin_edges[0])}"
        )

        # Convert physical values back to bin indices for the confusion matrix
        # We'll use the validation set for the confusion matrix as a common practice
        true_indices_val = np.clip(np.digitize(self.z_val, bin_edges), 1, len(bin_edges) - 1) - 1
        pred_indices_val = (
            np.clip(np.digitize(self.z_val_pred, bin_edges), 1, len(bin_edges) - 1) - 1
        )

        num_classes = len(bin_edges) - 1

        # Class labels for the confusion matrix axes
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Convert to µm for readability if original units are meters
        # class_labels_um = [f"{c * 1e6:.1f}µm" for c in bin_centers]
        class_labels_um = [f"{c:.1f}µm" for c in bin_centers]

        cm = confusion_matrix(true_indices_val, pred_indices_val, labels=list(range(num_classes)))

        # Create annotated heatmap for confusion matrix using Plotly Figure Factory
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=class_labels_um,
            y=class_labels_um,
            colorscale="Blues",
            showscale=True,  # Shows the color bar
            reversescale=False,
        )
        _ = fig_cm.update_layout(
            title_text=f"Confusion Matrix (Validation Set) | {model_name}",
            xaxis_title_text="Predicted Label (Physical Bin Center)",
            yaxis_title_text="True Label (Physical Bin Center)",
            xaxis={
                "tickmode": "array",
                "tickvals": list(range(num_classes)),
                "ticktext": class_labels_um,
                "side": "bottom",
            },
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(num_classes)),
                "ticktext": class_labels_um,
                "autorange": "reversed",
            },  # Reversed to match typical CM layout
        )
        return fig_cm

    def scatter(self, model_name: str):
        """Scatter plot."""
        z_val_pred_min = self.z_val_pred.min()
        z_train_pred_min = self.z_train_pred.min()
        data = {
            "z_val": self.z_val,
            "z_train": self.z_train,
            "z_val_pred": self.z_val_pred,
            "z_train_pred": self.z_train_pred,
        }
        # Convert to µm for plotting, concat creates null values automatically since
        # prediction and evaluation arrays aren't the same size
        df: DataFrame = pl.concat(
            # items=[pl.DataFrame({name: (val * 1e6)}) for name, val in data.items()],
            items=[pl.DataFrame({name: val}) for name, val in data.items()],
            how="horizontal",
        )
        # even if some values are null (due to above) it will not be passed into the
        # plot as long as there are values in each bin
        residual_val = pl.col("z_val_pred") - pl.col("z_val")
        residual_train = pl.col("z_train_pred") - pl.col("z_train")
        df = df.with_columns((residual_val).alias("residual_val"))
        df = df.with_columns((residual_train).alias("residual_train"))
        # libraries like scipy and numpy dont play nicely with null values,
        # separate df for numerical manipulation
        df_filled: DataFrame = df.fill_null(0)

        # scipy estimates the probability density function of random points using
        # Gaussian kernels. This is generated and then applied to the stacked
        # arrays: vstack([N], [B]) -> [[N], [B]]. Outputing continious value to
        # use for density
        kde_val = gaussian_kde(
            np.vstack(
                [
                    df_filled["z_val_pred"].to_numpy().clip(min=z_val_pred_min),
                    df_filled["residual_val"].to_numpy(),
                ]
            )
        )
        kde_train = gaussian_kde(
            np.vstack(
                [
                    df_filled["z_train_pred"].to_numpy().clip(min=z_train_pred_min),
                    df_filled["residual_train"].to_numpy(),
                ]
            )
        )
        density_val = kde_val(
            np.vstack(
                [
                    df_filled["z_val_pred"].to_numpy().clip(min=z_val_pred_min),
                    df_filled["residual_val"].to_numpy(),
                ]
            )
        )
        density_train = kde_train(
            np.vstack(
                [
                    df_filled["z_train_pred"].to_numpy().clip(min=z_train_pred_min),
                    df_filled["residual_train"].to_numpy(),
                ]
            )
        )

        # -- create (1×2) layout -----------------------------------------------------------------
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.80, 0.20],  # 80 % to the scatter | 20 % to the histogram
            shared_yaxes=True,  # axes line up
            horizontal_spacing=0.04,
        )
        # -- main scatter coloured by density ----------------------------------------------------
        # Use the non filled df as plotly will properly drop the null values
        _ = fig.add_trace(
            go.Scatter(
                x=df["z_val_pred"].to_numpy(),
                y=df["residual_val"].to_numpy(),
                mode="markers",
                marker={
                    "color": density_val,  # continuous array
                    "colorscale": "burg",
                    "showscale": True,
                    "colorbar": {"title": "val density", "x": 1},  # separate the two scales
                    "size": 9,
                    "opacity": 0.9,
                },
                name="Validation",  # legend text
                legendgroup="VAL",  # groups both traces
                showlegend=True,  # visible item in legend
            ),
            row=1,
            col=1,
        )
        _ = fig.add_trace(
            go.Scatter(
                x=df["z_train_pred"].to_numpy(),
                y=df["residual_train"].to_numpy(),
                mode="markers",
                marker={
                    "color": density_train,
                    "colorscale": "oryel",
                    "showscale": True,
                    "colorbar": {"title": "train density", "x": 1.1},
                    "size": 9,
                    "opacity": 0.3,
                },
                name="Train",
                legendgroup="TRAIN",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # -- add the marginal histogram (horizontal for y-axis) ----------------------------------
        _ = fig.add_trace(
            go.Histogram(
                y=df["residual_val"],
                marker_color="rgba(68,1,84,0.7)",
                name="Validation",
                legendgroup="VAL",  # same group -> toggles together
                showlegend=False,  # no second item
                opacity=0.8,
            ),
            row=1,
            col=2,
        )

        _ = fig.add_trace(
            go.Histogram(
                y=df["residual_train"],
                marker_color="rgba(255,183,76,0.7)",
                name="Train",
                legendgroup="TRAIN",
                showlegend=False,
                opacity=0.8,
            ),
            row=1,
            col=2,
        )

        # Scatter
        _ = fig.update_xaxes(title="Focus Depth (µm)", row=1, col=1)
        _ = fig.update_yaxes(title="Residual (µm)", row=1, col=1)
        # Histogram region
        _ = fig.update_xaxes(showticklabels=True, row=1, col=2)

        _ = fig.update_layout(
            # template="plotly_white",
            title_text=f"Scatter Residual Plot (Validation Set) | {model_name}",
            legend_title_text="Legend",
            hovermode="closest",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        )
        return fig

    def plot_classification(
        self, display_override: DisplayType | None = None, timestamp: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Plot actual vs. predicted values and a confusion matrix for classification."""
        # unpack/reconfigure data
        display = display_override if display_override is not None else DisplayType(self.display)
        model_name = self.repeat_config.user_config.train.backbone
        model_name = "" if model_name is None else model_name

        cm_title = self.titles["cl_cm"]
        scatter_title = self.titles["cl_res"]

        fig_cm = self.confusion_matrix(model_name)
        fig_scatter = self.scatter(model_name)

        meta_cm = PlotMeta(
            title=cm_title,
            caption="Confusion matrix for classification.",
            tags=["Classification", "Confusion Matrix"],  # pyright: ignore[reportArgumentType]
            kind="result",  # pyright: ignore[reportArgumentType]
        )
        meta_scatter = PlotMeta(
            title=scatter_title,
            caption="Scatter-plot for classification.",
            tags=["Classification", "Scatter", "Training", "Validation"],  # pyright: ignore[reportArgumentType]
            kind="result",  # pyright: ignore[reportArgumentType]
        )
        # self.plot_violin_depth_bins()
        return repeat(
            self.repeat_config,
            [fig_cm, fig_scatter],
            [meta_cm, meta_scatter],
            display,
            timestamp,
        )

    def plot_regression_residual(self) -> list[PlotCollection] | None:
        """Plot residual vs true depth."""
        # setup base variables per step first
        z: npt.NDArray
        z_pred: npt.NDArray
        export: list[PlotCollection] = []
        for step in ["train", "validation"]:
            if step == "train":
                z = self.z_train
                z_pred = -self.z_train_pred
                meta = PlotMeta(
                    title=self.titles["reg_res_train"],
                    caption="Training residual for auto-focus.",
                    tags=["Resdiual", "Training"],
                )
            elif step == "validation":
                z = self.z_val
                z_pred = -self.z_val_pred
                meta = PlotMeta(
                    title=self.titles["reg_res_validation"],
                    caption="Evaluation residual for auto-focus.",
                    tags=["Resdiual", "Evalutation"],
                )
            else:
                raise Exception(f"Unknown step '{step}' passed.")

            res: npt.NDArray = z - z_pred
            # Ensure bins has at least 2 elements for np.linspace and subsequent logic
            n_bins: int = max(10, len(z) // 50)
            bins_np: npt.NDArray = np.linspace(z.min(), z.max(), n_bins, dtype=np.float64)

            # Calculate running mean & ±σ
            bin_idx = np.digitize(np.array(z), bins_np)
            mu_list: list[np.float64] = []
            sd_list: list[np.float64] = []
            xc_list: list[np.float64] = []
            # The loop should go up to len(bins_np) to cover all bins defined by linspace.
            # np.digitize with n_bins points creates n_bins-1 intervals.
            # Iterating from 1 to len(bins_np) (or n_bins) means checking indices 1 to
            # n_bins-1 based on digitize's output.
            for i in track_progress(range(1, len(bins_np)), description="Bin checking (Plotly)..."):
                mask: np.intp = bin_idx == i
                if mask.any():  # at least one sample in the bin
                    mu_list.append(res[mask].mean())
                    sd_list.append(res[mask].std())
                    xc_list.append(0.5 * (bins_np[i] + bins_np[i - 1]))

            mu_np = np.array(mu_list, dtype=np.float64)
            sd_np = np.array(sd_list, dtype=np.float64)
            xc_np = np.array(xc_list, dtype=np.float64)

            sig_x = np.concatenate([xc_np, xc_np[::-1]])
            sig_y = np.concatenate([mu_np + sd_np, (mu_np - sd_np)[::-1]], dtype=np.float64)

            fig: go.Figure = go.Figure()

            # Scatter plot of residuals
            _ = fig.add_trace(
                go.Scatter(
                    x=z,
                    y=res,
                    mode="markers",
                    marker={"size": 6, "opacity": 0.3, "color": "grey"},
                    name="Residuals",
                )
            )

            # Running mean line
            if len(xc_np) > 0:  # only plot if there's data
                _ = fig.add_trace(
                    go.Scatter(
                        x=xc_np,
                        y=mu_np,
                        mode="lines",
                        line={"color": "blue", "width": 2},
                        name="Mean",
                    )
                )

                # ±1 sigma band
                _ = fig.add_trace(
                    go.Scatter(
                        x=sig_x,
                        y=sig_y,
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.15)",
                        line={"color": "rgba(255,255,255,0)"},
                        hoverinfo="skip",
                        name="±1 σ",
                    )
                )

            # Horizontal line at y=0
            _ = fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

            _ = fig.update_layout(
                title_text=meta.title,
                xaxis_title_text="True focus depth (µm)",
                yaxis_title_text="Residual (pred–true) (µm)",
                legend_title_text="Legend",
                hovermode="closest",
            )

            export.append(PlotCollection(meta, fig))
        return export

    def plot_hexbin_with_marginals(self) -> list[PlotCollection] | None:
        """Plot density of predictions of z."""
        z: npt.NDArray
        z_pred: npt.NDArray
        export: list[PlotCollection] = []
        for step in ["train", "validation"]:
            if step == "train":
                z = self.z_train
                z_pred = -self.z_train_pred
                meta = PlotMeta(
                    title=self.titles["reg_den_train"],
                    caption="A hexbin plot showing the density of training predictions.",
                    tags=["Density", "Training"],
                )
            elif step == "validation":
                z = self.z_val
                z_pred = -self.z_val_pred
                meta = PlotMeta(
                    title=self.titles["reg_den_validation"],
                    caption="A hexbin plot showing the density of evaluation predictions.",
                    tags=["Density", "Evalutation"],
                )
            else:
                raise Exception(f"Unknown step '{step}' passed.")
            # data
            mask = np.isfinite(z)
            mask_pred = np.isfinite(z_pred)
            z, z_pred = z[mask], z_pred[mask_pred]

            if z.size == 0:
                raise Exception("plot_hexbin: no finite points after filtering")

            z_um, z_pred_um = z * 1e6, z_pred * 1e6

            # figure & main hexbin
            df = pl.DataFrame(
                {
                    "z true": z_um,
                    "z pred": z_pred_um,
                }
            )
            fig = px.density_heatmap(df, x="z true", y="z pred", marginal_x="histogram")
            export.append(PlotCollection(meta, fig))

        return export

    def plot_regression(
        self, display_override: DisplayType | None = None, timestamp: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Combine plotting functions involved in regression to get one result."""
        display = display_override if display_override is not None else DisplayType(self.display)
        fig_list: list[go.Figure] = []
        meta_list: list[PlotMeta] = []
        residual = self.plot_regression_residual()
        hexbin = self.plot_hexbin_with_marginals()
        merged_list = []
        if residual is not None:
            merged_list.extend(residual)
        if hexbin is not None:
            merged_list.extend(hexbin)

        for p in merged_list:
            meta_list.append(p.meta)
            fig_list.append(p.figure)

        return repeat(self.repeat_config, fig_list, meta_list, display, timestamp)

    @classmethod
    def from_z_preds(
        cls,
        bin_edges: Arr64 | None,
        core_trainer: CoreTrainer,
        training_output: TrainingOutput,
        bin_centers: Arr64 | None = None,
        z_avg: float | None = None,
        z_std: float | None = None,
    ) -> PlotPred:
        device = core_trainer.device
        _: nn.Module = (
            core_trainer.model.eval()
        )  # set model into valuation rather than training mode
        _ = core_trainer.model.to(device)  # ensure model is on expected device
        val_z_pred_list: list[Arr64] = []
        val_z_true_list: list[Arr64] = []
        train_z_pred_list: list[Arr64] = []
        train_z_true_list: list[Arr64] = []
        analysis = core_trainer.analysis

        titles: dict[str, str] = {}
        if analysis == AnalysisType.CLASS:
            # set title for the plots as expected by plotting module
            titles["cl_res"] = "Classification Residual"
            titles["cl_cm"] = "Confusion Matrix (validation)"
        else:
            # repeat the naming scheme for each item
            for t in ["train", "validation"]:
                titles[f"reg_res_{t}"] = f"Residual vs True depth ({t})"
                titles[f"reg_vio_{t}"] = f"Signed error distribution per depth slice ({t})"
                titles[f"reg_den_{t}"] = f"Prediction density ({t})"

        with torch.no_grad():
            for loader in [core_trainer.train_loader, core_trainer.val_loader]:
                current_step = "training" if loader is core_trainer.train_loader else "evaluation"
                for imgs, labels in track_progress(
                    loader, f"Gathering z predictions from {current_step} using {device}..."
                ):
                    # non_blocking means allowing for multiple tensors to be sent to device
                    imgs_tens = imgs.to(device, non_blocking=True)
                    labels_tens = labels.to(device, non_blocking=True)
                    assert (
                        next(core_trainer.model.parameters()).device
                        == imgs_tens.device
                        == labels_tens.device
                    ), (
                        f"Images {imgs_tens.device}, labels {labels_tens.device}, or train model \
                        {next(core_trainer.model.parameters()).device} not on same device."
                    )

                    # pass in data to train_cfg.model
                    preds = core_trainer.model(imgs_tens)
                    # convert back to physical units
                    if analysis.value == AnalysisType.REG.value:
                        if z_avg is None or z_std is None:
                            raise ValueError(
                                "z_avg and z_std required for regression de-normalization"
                            )

                        # bring predictions back to cpu
                        preds_batch = preds.squeeze().cpu().numpy() * z_std + z_avg
                        true_batch = labels_tens.cpu().numpy() * z_std + z_avg

                    elif analysis.value == AnalysisType.CLASS.value:
                        if bin_centers is None:
                            raise ValueError("bin_centers required for classificaton conversion")

                        # argmax returns a tensor containing the indices that hold
                        # the maximimum values of the input tensor across the selected
                        # dimension/axis. Here it grabs the indicies of the predictions,
                        # which ought to correspond to the integer bins in the label.
                        preds_arr_indices = preds.argmax(dim=1).cpu().numpy()
                        labels_arr_indices = labels_tens.cpu().numpy()
                        preds_batch = bin_centers[preds_arr_indices]
                        true_batch = bin_centers[labels_arr_indices]
                    else:
                        raise Exception("Unknown analysis value")

                    if loader == core_trainer.train_loader:
                        train_z_pred_list.append(preds_batch)
                        train_z_true_list.append(true_batch)
                    else:
                        val_z_pred_list.append(preds_batch)
                        val_z_true_list.append(true_batch)

        # store each of these values
        val_z_pred: Arr64 = np.concatenate(val_z_pred_list, dtype=np.float64)
        val_z_true: Arr64 = np.concatenate(val_z_true_list, dtype=np.float64)
        train_z_pred: Arr64 = np.concatenate(train_z_pred_list, dtype=np.float64)
        train_z_true: Arr64 = np.concatenate(train_z_true_list, dtype=np.float64)

        # Actual vs Predicted diff
        train_err: Arr64 = np.abs(train_z_pred - train_z_true)
        val_err: Arr64 = np.abs(val_z_pred - val_z_true)

        repeat_config = TrainingRepeatConfig(
            core_trainer.a_cfg.to_user_config(),
            training_output.avg_train_loss,
            training_output.avg_val_loss,
        )

        return cls(
            val_z_pred,
            val_z_true,
            train_z_pred,
            train_z_true,
            train_err,
            val_err,
            bin_edges,
            titles,
            "save",
            core_trainer.analysis.value,
            repeat_config,
        )


def repeat(
    repeat_config: TrainingRepeatConfig,
    in_fig: Plots,
    meta: list[PlotMeta],
    display: DisplayType,
    timestamp: str | None = None,
) -> list[dict[str, Any]] | None:
    """Store the results of the training metrics in a repeatable format."""
    report: Path = report_path()
    # "E" prefix marks a timestamp generated at plot time, not tied to a source plot_info json
    current_time = timestamp if timestamp is not None else "E" + datetime.now().strftime("%H%M%S")
    current_results: Path = report / f"results_{current_time}.json"

    json_report: str = to_json(repeat_config, TrainingRepeatConfig)
    current_results.write_text(json_report)

    meta_list: list[dict[str, Any]] = []
    files: list[str] = []
    if not isinstance(in_fig, Iterable):
        in_fig = [in_fig]
    for fig, m in zip(in_fig, meta, strict=False):
        # matplotlib and plotly can have a png
        png_path: str = f"{report_path(True) / Path(m.slug)}_{current_time}.png"
        match fig:
            case go.Figure():  # plotly
                logger.debug("Processing plotly figure for meta...")
                html_path: str = f"{report_path(True) / Path(m.slug)}_{current_time}.html"
                if display is not DisplayType.SHOW:
                    # Self-contained HTML (includes Plotly JS via CDN for lighter page weight)
                    fig.write_html(html_path, full_html=True, include_plotlyjs="cdn")  # pyright: ignore[reportArgumentType]
                if display is not DisplayType.SAVE and display is not DisplayType.META:
                    logger.info("Displaying plot...")
                    fig.show()
                if display is DisplayType.META:
                    files.append(png_path)
                    meta_list.append(asdict(m))
            case Figure():  # matplotlib
                if display is not DisplayType.SHOW:
                    logger.info("Displaying matplotlib plot...")
                    plt.savefig(png_path)
                if display is not DisplayType.SAVE:
                    logger.info("Displaying matplotlib plot...")
                    plt.show()
            case _:
                logger.error("Unkown plot type passed to save plot function.")
                raise Exception("Plot type unknown.")

    if len(files) != 0:
        try:
            # WARN: needs external dependencies to create thumbnail
            # for the card grid (requires `pip install -U kaleido`)
            pio.get_chrome()  # type: ignore
            pio.write_images(  # type: ignore
                fig=in_fig, file=files, scale=0.5, width=1200, height=675
            )  # 16:9
            return meta_list
        except Exception as e:
            logger.error("Error saving image, ensure you have chromium installed.")
            raise e
    return None


def plot_amp_phase(
    img_file_path,
    wavelength,
    ckpt_file,
    crop_size,
    dx,
    amp_true: npt.NDArray[Any] | None = None,
    phase_true: npt.NDArray[Any] | None = None,
    path_to_plot: str = "phase_amp.png",
    display: DisplayType = DisplayType.SHOW,
) -> str | None:
    """Visualise amplitude & phase reconstruction."""
    from PIL import Image

    # from holod.core.metrics import error_metric, wrap_phase
    from holod.core.metrics import wrap_phase
    from holod.core.optics.reconstruction import torch_recon

    _hologram, amp_recon, phase_recon, focus_tc = torch_recon(
        img_file_path, wavelength, ckpt_file, crop_size, dx
    )

    # label-free gradient-Tamura focus score; higher is sharper (see focus_score)
    logger.info(f"Focus score of reconstruction (gradient Tamura): {focus_tc:.4f}")

    _save_path = Path(path_to_plot)

    # vmin, vmax = (-np.pi, np.pi)
    vmin, vmax = (phase_recon.min(), phase_recon.max())
    logger.info("Creating Plot...")
    if amp_true is not None and phase_true is not None:
        # 2 x 3: [amp_true, amp_recon, |amp_err] / [phase_true, phase_recon, |phase_err_wrapped]
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        (ax_at, ax_ar, ax_ae, ax_pt, ax_pr, ax_pe) = axes.flatten()

        # Amplitude GT / Recon / Error
        im0: AxesImage = ax_at.imshow(amp_true, cmap="gray")
        ax_at.set_title("Amplitude ground truth")
        fig.colorbar(im0, ax=ax_at, shrink=0.8)

        im1: AxesImage = ax_ar.imshow(amp_recon, cmap="gray")
        ax_ar.set_title("Amplitude reconstruction")
        fig.colorbar(im1, ax=ax_ar, shrink=0.8)

        amp_err = np.abs(amp_true - amp_recon)
        im2: AxesImage = ax_ae.imshow(amp_err, cmap="inferno")
        ax_ae.set_title("Amplitude error (|gt - recon|)")
        fig.colorbar(im2, ax=ax_ae, shrink=0.8)

        # Phase GT / Recon / Wrapped error
        im3: AxesImage = ax_pt.imshow(phase_true, cmap="twilight", vmin=vmin, vmax=vmax)
        ax_pt.set_title("Phase ground truth")
        fig.colorbar(im3, ax=ax_pt, shrink=0.8)

        im_phase_recon: AxesImage = ax_pr.imshow(phase_recon, cmap="twilight", vmin=vmin, vmax=vmax)
        ax_pr.set_title("Phase reconstruction")
        fig.colorbar(im_phase_recon, ax=ax_pr, shrink=0.8)

        phase_err = wrap_phase(phase_true - phase_recon)
        im5: AxesImage = ax_pe.imshow(phase_err, cmap="twilight", vmin=vmin, vmax=vmax)
        ax_pe.set_title("Phase error (wrapped)")
        fig.colorbar(im5, ax=ax_pe, shrink=0.8)

    else:
        # 2 x 2: [amp_recon, phase_recon] + keep two axes for future metrics
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        (ax_ar, ax_blank1, ax_pr, ax_blank2) = axes.flatten()

        im1 = ax_ar.imshow(amp_recon, cmap="gray")
        ax_ar.set_title("Amplitude reconstruction")
        cbar_ar = fig.colorbar(im1, ax=ax_ar, shrink=0.8)
        cbar_ar.set_label("Intensity value (0 to 1)")

        im_phase_recon = ax_pr.imshow(phase_recon, cmap="twilight", vmin=vmin, vmax=vmax)
        ax_pr.set_title("Phase reconstruction")
        cbar_phase = fig.colorbar(im_phase_recon, ax=ax_pr, shrink=0.8)
        cbar_phase.set_label("Phase value (radians)")

        # Hide unused panes
        ax_blank1.axis("off")
        ax_blank2.axis("off")

    # TODO: properly account for no x,y ticks on images but y ticks on colorbar
    for ax in fig.axes:
        ax.set_xticks([])
        # ax.set_yticks([])
        # only apply to image axes (keep colorbar ticks)
        # if hasattr(ax, "set_xticks"):
        #     ax.set_xticks([])
        #     ax.set_yticks([])

    fig.tight_layout()

    # TODO: handle meta type
    if display != DisplayType.SHOW:
        fig.savefig("plot.png", dpi=200, bbox_inches="tight")
        if amp_true is None and phase_true is None:
            # crop the image to only contain the reconstruction
            im = Image.open("plot.png")
            w, h = im.size
            cropped = im.crop((0, 0, w / 1.5, h))
            cropped.save("plot.png")
    if display != DisplayType.SAVE:
        plt.show()
