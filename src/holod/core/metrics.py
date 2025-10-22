from typing import cast

import numpy as np

from holod.infra.util.types import Arr32, Arr64
from holod.infra.log import get_logger

logger = get_logger(__name__)


def wrap_phase(p: Arr32):
    """One liner to wrap value in pi -> - pi."""
    return (p + np.pi) % (2 * np.pi) - np.pi


def phase_metrics(org_phase: Arr32, recon_phase: Arr32):
    """Calculate the mean average error and the phase cosine similarity."""
    diff = wrap_phase(org_phase - recon_phase)
    mae: float = np.abs(diff).mean(dtype=float)
    cos_sim = np.mean(np.cos(diff), dtype=float)  # 1.0 -> perfect match
    return {"MAE_phase": mae, "CosSim": cos_sim}


def error_metric(expected: Arr64, observed: Arr64, max_px: float):
    """Find the normalized root mean square error, and peak noise to signal ratio of two quantities.

    Args:
        expected: Array of image being reconstructed.
        observed: Array of reconstsucted image.
        max_px: The maximum pixel value of an image, e.g. 255 for 8-bit images.

    Returns:
        The NRMSE, and PSNR of these two images.

    """
    # Mean squared error (MSE) -> Peak Signal to noise ratio (PSNR)
    # MSE = 1/n \sum_{i=1}^{n} ( x_i - \hat{x}_{i} )^{2}
    mse = np.mean((expected - observed) ** 2, dtype=np.float64)
    rmse = cast("np.float64", np.sqrt(mse))
    nrmse = rmse / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = cast("np.float64", 10 * np.log10((max_px**2) / mse))
    return nrmse, psnr
