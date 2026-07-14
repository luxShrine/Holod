"""Slow capacity tests: every backbone must overfit a single batch.

Each test builds the real dataloader pipeline (dataset -> transforms ->
``create_training_setup``) for one backbone and runs it through
``overfit_single_batch``. A healthy model/pipeline pair must be able to
memorize one batch; failure points at a wiring bug (labels, transforms,
optimizer, loss) or a model without enough capacity.

These train for ~100 optimizer steps per backbone, so the whole module is
marked ``slow`` and skipped by ``make test``; run it with ``make test-slow``
or ``uv run pytest -q -m slow src/tests/check_overfit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from check_training import create_test_dataset

from holod.infra.dataclasses import AutoConfig, CoreTrainer, create_training_setup
from holod.infra.util.training_help import overfit_single_batch
from holod.infra.util.types import AnalysisType, ModelType, UserDevice

pytestmark = pytest.mark.slow

# create_test_dataset digitizes depths into 10 bins, so the configs must match
NUM_CLASSES = 10
BATCH_SIZE = 8


@dataclass(frozen=True)
class OverfitCase:
    """Per-backbone knobs for the single-batch overfit run."""

    crop_size: int
    opt_lr: float
    steps: int
    # end/start loss ratio the run must reach; dropout-heavy nets keep some
    # irreducible train-mode noise, so they get a looser bound
    rel_threshold: float


CASES: dict[ModelType, OverfitCase] = {
    # EfficientNet-B4 trains through Dropout(0.4) plus stochastic depth, so it
    # needs a hotter LR / more steps and keeps a higher train-mode loss floor
    ModelType.ENET: OverfitCase(crop_size=64, opt_lr=5e-4, steps=150, rel_threshold=0.15),
    ModelType.RESNET: OverfitCase(crop_size=64, opt_lr=1e-4, steps=100, rel_threshold=0.05),
    # the ViT patch embedding is fixed to 224x224 inputs
    ModelType.VIT: OverfitCase(crop_size=224, opt_lr=1e-4, steps=100, rel_threshold=0.05),
    ModelType.FOCUSNET: OverfitCase(crop_size=64, opt_lr=1e-3, steps=100, rel_threshold=0.05),
    # PCNN trains through Dropout(0.5), which keeps the train-mode loss noisy
    ModelType.PCNN: OverfitCase(crop_size=64, opt_lr=1e-3, steps=300, rel_threshold=0.25),
}


def build_overfit_trainer(backbone: ModelType) -> CoreTrainer:
    """Build a CoreTrainer for one backbone via the real training pipeline."""
    # the loader shuffle, augmentations, head init, and dropout masks all draw
    # from the global RNG; seed it so each run trains on the same batch
    _ = torch.manual_seed(1234)
    case = CASES[backbone]
    a_cfg = AutoConfig(
        analysis=AnalysisType.CLASS,
        backbone=backbone,
        batch_size=BATCH_SIZE,
        crop_size=case.crop_size,
        epoch_count=1,
        num_classes=NUM_CLASSES,
        num_workers=0,
        val_split=0.2,
        fixed_seed=True,
        opt_lr=case.opt_lr,
        # regularization only fights memorization here
        opt_weight_decay=0.0,
        # GPU when available, otherwise the pipeline must work on CPU too
        device_user=UserDevice.determine("cuda"),
    )
    core_trainer = create_training_setup(create_test_dataset(AnalysisType.CLASS), a_cfg)
    # overfit_single_batch moves the batch to core_trainer.device but leaves the
    # model where create_training_setup built it, so align them here
    _ = core_trainer.model.to(core_trainer.device)
    return core_trainer


@pytest.mark.parametrize("backbone", list(CASES), ids=lambda m: m.value)
def test_overfit_single_batch(backbone: ModelType) -> None:
    """The full pipeline must let each backbone memorize one training batch."""
    case = CASES[backbone]
    core_trainer = build_overfit_trainer(backbone)

    result = overfit_single_batch(
        core_trainer, n=case.steps, avg_over_w=5, rel_threshold=case.rel_threshold
    )

    losses = result["losses"]
    assert len(losses) == case.steps
    assert all(loss == loss for loss in losses), f"{backbone.value}: loss went NaN: {losses[-5:]}"
    assert result["end_avg"] < result["start_avg"], (
        f"{backbone.value}: loss did not decrease at all "
        f"(start {result['start_avg']:.4f} -> end {result['end_avg']:.4f}); "
        "check LR, optimizer wiring, or loss sign."
    )
    assert result["overfit"], (
        f"{backbone.value}: failed to overfit a single batch of {BATCH_SIZE} images in "
        f"{case.steps} steps: end/start loss ratio {result['ratio']:.4f} "
        f"> threshold {case.rel_threshold}."
    )
