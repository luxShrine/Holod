from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from holod.infra.dataclasses import (
    Checkpoint,
    CoreTrainer,
    EpochMetric,
    ModelCheckpoint,
)
from holod.infra.log import get_logger
from holod.infra.util.paths import checkpoints_loc, report_path
from holod.infra.util.types import (
    AnalysisType,
    Arr32,
)

logger = get_logger(__name__)

# how many models to be kept
MAX_MODEL_HISTORY: int = 20
EXPECTED_IMPROVEMENT_PERCENT: float = 1e-3

type TrainStatus = TrainImprovement | None


def _remove_oldest_checkpoint(path_to_model_detail: Path) -> None:
    """Remove the oldest checkpoint if max model history is reached."""
    # get files in out directory
    files_in_out_dir: list[Path] = list(path_to_model_detail.parent.iterdir())
    if len(files_in_out_dir) > (MAX_MODEL_HISTORY + 1):
        # clean up directory if needed to preserve storage
        # if checkpoint folder has > MAX_MODEL_HISTORY, remove oldest
        # find files that end in ".pth"
        file_count_pth: int = 0
        oldest_mod_time: float = np.inf
        oldest_file: Path | None = None
        for out_file in files_in_out_dir:
            if out_file.as_posix().endswith(".pth"):
                file_count_pth += 1
                current_mod_time: float = out_file.stat().st_mtime
                if current_mod_time < oldest_mod_time:
                    oldest_file = out_file
                    oldest_mod_time = current_mod_time
        # remove oldest_file if limit reached
        if file_count_pth > MAX_MODEL_HISTORY and oldest_file is not None:
            logger.debug(
                f"Max model history limit of {MAX_MODEL_HISTORY} "
                + f"reached, deleting {oldest_file}"
            )
            oldest_file.unlink()


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
            # convert best_val_metric form of 5 numbers, in scientific notation
            # create file with name that is unique to evaluation
            best_model_name: str = (
                path_to_model.name.removesuffix(".pth") + f"{best_val_out:3e}" + ".pth"
            )
            path_to_model_detail = path_to_model.parent / Path(best_model_name)

            # check if files in directory has potential amount of
            # files to reach limit before loop.
            _remove_oldest_checkpoint(path_to_model_detail)
            checkpoint = Checkpoint.from_epoch(epoch, core_trainer, labels_tensor, best_val_out)
            checkpoint.torch_save(path_to_model_detail)
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


# TODO: setup tests?
def overfit_single_batch(
    core_trainer: CoreTrainer, n: int = 100, avg_over_w: int = 5, rel_threshold=0.05
):
    loader = core_trainer.train_loader
    # ensure that if the core_trainer is used after, the model/optimizer are untouched
    model = deepcopy(core_trainer.model)
    opt = deepcopy(core_trainer.optimizer)
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
