from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from holod.infra.dataclasses import (
    Checkpoint,
    CompareUserConfig,
    CoreTrainer,
    EpochMetric,
    ModelCheckpoint,
    create_training_setup,
)
from holod.infra.dataset import HologramFocusDataset
from holod.infra.log import get_logger
from holod.infra.util.paths import checkpoints_loc, report_path
from holod.infra.util.prog_helper import track_progress
from holod.infra.util.types import (
    AnalysisType,
    Arr32,
    ModelType,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

EXPECTED_IMPROVEMENT_PERCENT: float = 1e-3

type TrainStatus = TrainImprovement | None


@dataclass
class TrainImprovement:
    best_val: float
    save_best: bool

    @classmethod
    def check_model_improvement(
        cls,
        metric_val_hist: list[float],
        epoch_metric: EpochMetric,
        best_val: float,
        core_trainer: CoreTrainer,
        labels_tensor: Tensor,
        path_to_model: Path,
        epoch: int,
        short_range: int = 3,
        long_range: int = 5,
    ) -> TrainStatus:
        epoch_count = core_trainer.a_cfg.epoch_count
        if core_trainer.analysis == AnalysisType.REG:
            # Lower MAE is better
            save_best_model_flag = epoch_metric.metric_val < best_val
            best_val_out = epoch_metric.metric_val if save_best_model_flag else best_val
            logger.debug(f"At {epoch} / {epoch_count} Val MAE: {epoch_metric.metric_val:.9f} mm")
        else:
            # Higher Accuracy is better
            save_best_model_flag = epoch_metric.metric_val > best_val
            best_val_out = epoch_metric.metric_val if save_best_model_flag else best_val
            logger.debug(
                f"At {epoch} / {epoch_count} Val Acc: {epoch_metric.metric_val * 100:.2f} %"
            )

        if epoch > (epoch_count / 5) and epoch >= 10:
            # get percent diff to measure improvement, desire current < previous
            percent_diff_history = get_percent_diff_history(
                epoch_metric, metric_val_hist, core_trainer.analysis
            )

            diff_short = abs(percent_diff_history[-short_range] - percent_diff_history[-1])
            diff_long = abs(percent_diff_history[-long_range] - percent_diff_history[-1])

            # if epoch is > N/5 and no improvement over three epochs, display error
            if diff_short <= EXPECTED_IMPROVEMENT_PERCENT:
                print(
                    f"Small or no improvement of metric val: {diff_short}"
                    + f" from epoch {epoch - 3} to epoch {epoch}",
                )
                # if after 5 epochs, stop training
                if diff_long < EXPECTED_IMPROVEMENT_PERCENT:
                    print(
                        "Training stopping, little to no improvement after "
                        + f"{long_range} epochs",
                    )
                    return None

        if save_best_model_flag:
            # overwrite the single best-model file in place; the metric and epoch
            # are stored inside the checkpoint itself
            checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, best_val_out)
            checkpoint.torch_save(path_to_model)
        return TrainImprovement(best_val_out, save_best_model_flag)


# TODO: this is recreating an array each time, only need to calculate the newest values
def get_percent_diff_history(
    epoch_metric: EpochMetric,
    metric_val_hist: list[float],
    analysis_type: AnalysisType,
) -> Arr32:
    """Calculate percentage difference across epochs for finding when improvement stalls."""
    try:
        average = [(x + epoch_metric.metric_val) / 2 for x in metric_val_hist]
        match analysis_type:
            case AnalysisType.REG:
                return np.asarray(
                    [
                        (x - epoch_metric.metric_val) / average[i]
                        for (i, x) in enumerate(metric_val_hist)
                    ],
                    dtype=np.float32,
                )
            case AnalysisType.CLASS:
                return np.asarray(
                    [
                        (epoch_metric.metric_val - x) / average[i]
                        for (i, x) in enumerate(metric_val_hist)
                    ],
                    dtype=np.float32,
                )

    except Exception as e:
        logger.error("Could not calculate the percent difference.")
        raise e


def init_epoch(core_trainer: CoreTrainer, device: str):
    epoch_metric: EpochMetric = EpochMetric()
    abs_err_sum: int | float = 0
    total_samples_for_metric: int = 0  # Denominator for MAE/Accuracy
    # bin-center lookup for the physical-distance MAE (classification only)
    bin_centers_tens: Tensor | None = None
    phys_err_sum: float = 0.0
    if core_trainer.analysis == AnalysisType.CLASS and core_trainer.bin_centers is not None:
        bin_centers_tens = torch.as_tensor(
            core_trainer.bin_centers, dtype=torch.float32, device=device
        )
    return (
        epoch_metric,
        abs_err_sum,
        total_samples_for_metric,
        bin_centers_tens,
        phys_err_sum,
    )


def init_training(core_trainer: CoreTrainer, ckpt: ModelCheckpoint | None):
    if ckpt is None:
        labels_tensor: Tensor = torch.empty([1, 1])
        epoch_metric: EpochMetric = EpochMetric(metric_val=0)
        avg_loss_train: float = 0
        avg_loss_val: float = 0
    else:
        labels_tensor = ckpt.checkpoint.labels.to(core_trainer.device)
        epoch_metric = EpochMetric(ckpt.checkpoint.val_metric)
        avg_loss_train = ckpt.train_loss
        avg_loss_val = ckpt.val_loss

    # Paths
    checkpoint_dir = checkpoints_loc()
    checkpoint_dir.mkdir(exist_ok=True)
    device = "cuda" if core_trainer.device == "cuda" else "cpu"
    return checkpoint_dir, labels_tensor, epoch_metric, avg_loss_train, avg_loss_val, device


def save_loss_to_file(
    model_name: str,
    run_stamp: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    val_metric: float,
    val_metric_name: str,
    val_mae_mm: float | None,
):
    """Append one epoch's losses and validation metric to the run's history JSON.

    ``run_stamp`` must be constant for one training run and unique across runs so
    that back-to-back runs (e.g. several backbones inside ``holod compare``, or two
    runs of the same backbone on one day) never mix their epoch histories.
    """
    slug = f"{model_name}-{run_stamp}.json"
    json_p = report_path() / "loss" / slug
    try:
        json_s = json_p.read_text()
        json_dict = json.loads(json_s)
    except FileNotFoundError:
        # new file, create it
        json_dict = {
            "name": model_name,
            "val_metric_name": val_metric_name,
            "epochs": [],
            "train_loss": [],
            "eval_loss": [],
            "val_metric": [],
            "val_mae_mm": [],
        }
        json_p.parent.mkdir(parents=True, exist_ok=True)
        json_p.touch()
    json_dict["epochs"].append(epoch)
    json_dict["train_loss"].append(train_loss)
    json_dict["eval_loss"].append(eval_loss)
    json_dict["val_metric"].append(val_metric)
    json_dict["val_mae_mm"].append(val_mae_mm)
    new_json_s = json.dumps(json_dict)
    json_p.write_text(new_json_s)
    return json_p


# exercised per-backbone by the slow tests in src/tests/check_overfit.py
def overfit_single_batch(
    core_trainer: CoreTrainer, n: int = 100, avg_over_w: int = 5, rel_threshold=0.05
):
    """Train on one batch for ``n`` steps to sanity-check model capacity and wiring.

    Returns a dict with the per-step ``losses``, the ``start_avg``/``end_avg`` loss
    (each averaged over ``avg_over_w`` steps), their ``ratio``, and whether the run
    counts as an ``overfit`` (ratio at or below ``rel_threshold``). The trainer's
    model and optimizer are left untouched.
    """
    loader = core_trainer.train_loader
    # ensure that if the core_trainer is used after, the model/optimizer are untouched;
    # copied together so the optimizer copy still points at the copied model's
    # parameters (separate deepcopies sever that link and step() updates nothing)
    model, opt = deepcopy((core_trainer.model, core_trainer.optimizer))
    loss_fn = core_trainer.loss_fn
    losses = []
    device = core_trainer.device
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    for _ in range(n):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        # we want overfitting, dont really need regularizer
        losses.append(float(loss.item()))

        opt.step()
        opt.zero_grad()

    start_avg = np.mean(losses[:avg_over_w])
    end_avg = np.mean(losses[-avg_over_w:])
    ratio = end_avg / start_avg if start_avg > 0 else float("nan")
    overfit = ratio <= rel_threshold

    print(f"Start avg (first {n}): {start_avg:.6f}")
    print(f"End avg (last {n}):   {end_avg:.6f}")
    print(f"Ratio (end/start):    {ratio:.4f}  (threshold={rel_threshold})")
    print(f"Overfit: {'YES' if overfit else 'NO'}")
    if end_avg > start_avg:
        print("Loss increased! check LR, optimizer setup, or loss sign.")

    return {
        "losses": losses,
        "start_avg": start_avg,
        "end_avg": end_avg,
        "ratio": ratio,
        "overfit": overfit,
    }


def _create_loader_cycle(core_trainer: CoreTrainer):
    while True:
        yield from core_trainer.train_loader


def determine_learning_rate(
    u_cfg: CompareUserConfig,
    backbone: ModelType,
    *,
    learning_rate_lower: float = 1e-7,
    learning_rate_upper: float = 1e-2,
    n: int = 500,
    divergence_factor: float = 4.0,
    beta: float = 0.98,
):
    """Sweep ``n`` log-spaced learning rates and report the final-epoch loss for each."""
    # log-spaced: LR effects span decades, linear spacing would leave the
    # low decades almost unsampled
    learning_rates = np.geomspace(learning_rate_lower, learning_rate_upper, num=n)
    loss_per_each_lr: dict[float, tuple[float, float]] = {}

    _ = torch.manual_seed(42)
    u_cfg = u_cfg.resolve_paths()
    analysis = AnalysisType.determine(u_cfg.num_classes)
    base = HologramFocusDataset(
        mode=analysis,
        num_classes=u_cfg.num_classes,
        csv_file_strpath=u_cfg.paths.meta_csv_name,
    )
    a_cfg = u_cfg.to_auto_config(backbone)
    core_trainer = create_training_setup(base, a_cfg)
    # identical split and model init for every candidate, so loss differences
    # reflect the learning rate alone
    model, opt, loss_fn = core_trainer.model, core_trainer.optimizer, core_trainer.loss_fn
    device = core_trainer.device
    model.to(device)

    # zip retrieves tuples until the shortest is exhausted, so no need to
    # have the loader be the same length as the learning rates
    loader_cycle = _create_loader_cycle(core_trainer)
    lr_batches = zip(learning_rates, loader_cycle, strict=False)

    best_smoothed_loss: float = np.inf
    exp_moving_avg = 0.0  # running average

    for idx, (lr, (x, y)) in enumerate(track_progress(lr_batches, total=n), start=1):
        # each batch has a new LR
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        loss_current = float(loss.item())
        exp_moving_avg = (beta * exp_moving_avg) + ((1 - beta) * loss_current)
        # the above average is setup such that ~=(1-beta)% of the inital loss and
        # would be passed on to be the best avg loss. Thus, the following iterations
        # will increase the loss as the previous exp_moving_avg increases from the
        # initial 0.0. The solution is to temper the moving average based on the number
        # of iterations
        smoothed_loss = exp_moving_avg / (1 - (beta**idx))
        loss_per_each_lr[lr] = (loss_current, smoothed_loss)

        opt.step()
        opt.zero_grad()

        if best_smoothed_loss > smoothed_loss:
            best_smoothed_loss = smoothed_loss
        elif not np.isfinite(loss_current) or (
            smoothed_loss > (divergence_factor * best_smoothed_loss)
        ):
            print(f"LR {lr} diverged (loss {loss_current} vs best {best_smoothed_loss}), skipping")
            break

    # compute the curve of the learning rate loss curve, desired region
    # should be on a downward slope, negative derivative
    diff = np.diff([sl for (_l, sl) in loss_per_each_lr.values()])
    diff = np.insert(diff, 0, np.inf)  # pad the start such that indices line up
    for idx, ((lr, (m_loss, s_loss)), c_diff) in enumerate(
        zip(loss_per_each_lr.items(), diff, strict=False)
    ):
        if (idx % 100) == 0:
            print(
                f"For learning rate: {lr:.6e}, corresponding loss: {m_loss:.6e}, "
                + f"smoothed loss: {s_loss:.6e}, corresponding diff: {c_diff:.6e}"
            )
    ideal_lr_idx = np.argmin(diff)
    ideal_lr = learning_rates[ideal_lr_idx]
    ideal_lr_loss, ideal_lr_s_loss = loss_per_each_lr[ideal_lr]
    ideal_lr_diff = diff[ideal_lr_idx]
    print(
        f"Reccomended learning rate: {ideal_lr:.6e}, corresponding loss: {ideal_lr_loss:.6e}, "
        + f"smoothed loss: {ideal_lr_s_loss:.6e}, corresponding diff: {ideal_lr_diff:.6e}"
    )

    return loss_per_each_lr, ideal_lr
