"""Loss functions used by the autofocus training pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from holod.infra.log import get_logger

logger = get_logger(__name__)


class SoftOrdinalCrossEntropy(nn.Module):
    """Cross entropy against soft, ordinal-aware targets (SORD).

    Expands each hard bin index ``y`` into a probability distribution over all
    bins, ``target_k = softmax_k(-(k - y)^2 / (2 * sigma^2))``, so bins near the
    true depth keep most of the probability mass and a 1-bin miss is penalized
    less than a many-bin miss. Depth bins are uniform (``np.linspace`` in
    ``HologramFocusDataset``), so bin-index distance is proportional to physical
    depth distance and ``sigma`` is expressed in bins.

    Predictions stay ordinary ``[batch, num_classes]`` logits and labels stay
    hard ``[batch]`` indices; the softening happens entirely inside ``forward``.

    Attributes:
        num_classes: Number of depth bins the model predicts over.
        sigma: Standard deviation of the target distribution, in bins.
        soft_targets: ``[num_classes, num_classes]`` lookup table where row ``y``
            is the soft target distribution for true bin ``y``.

    """

    def __init__(self, num_classes: int, sigma: float) -> None:
        """Precompute the soft-target lookup table for every possible true bin."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive to soften labels, got {sigma}.")
        self.num_classes: int = num_classes
        self.sigma: float = sigma

        bin_idx = torch.arange(num_classes, dtype=torch.float32)
        # squared ordinal distance between every (true bin, candidate bin) pair
        dist_sq = (bin_idx.unsqueeze(0) - bin_idx.unsqueeze(1)) ** 2
        self.soft_targets: Tensor
        self.register_buffer("soft_targets", torch.softmax(-dist_sq / (2.0 * sigma**2), dim=1))
        logger.debug(
            f"SoftOrdinalCrossEntropy over {num_classes} bins with sigma={sigma} bins; "
            f"true-bin target weight {self.soft_targets[0, 0]:.3f}"
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute cross entropy between logits and the soft targets for ``target``.

        Args:
            pred: ``[batch, num_classes]`` raw logits.
            target: ``[batch]`` hard bin indices (long).

        """
        # the loss module is never moved with the model, so follow the logits' device
        soft = self.soft_targets.to(device=pred.device, dtype=pred.dtype)
        return F.cross_entropy(pred, soft[target])
