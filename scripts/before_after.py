"""Render the before/after comparison figure for the DLHM focus correction.

Outputs (200 dpi, ready to drop into the slide):
    fig_before_after.png      -- side-by-side: sqrt(I) field at uncorrected z
                                 vs. DC-stripped contrast field at z_eff,
                                 each annotated with its gradient-Tamura score.

The "after" panel reconstructs the SAME field the focus metric sees
(contrast = I/mean(I) - 1).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from holod.core.optics.reconstruction import (
    dlhm_effective_z_mm,
    gradient_tamura,
    recon_inline,
)


def run_before_after(
    intensity: np.ndarray,
    z_pred_mm: float,
    l_mm: float,
    wavelength_m: float,
    dx_m: float,
    out: Path,
) -> None:
    """Write the two-panel before/after figure for one hologram to ``out``.

    The "before" panel reconstructs at the raw predicted depth ``z_pred_mm``;
    the "after" panel reconstructs the DC-stripped contrast field at the
    effective depth, and each panel is annotated with its gradient-Tamura score.
    """
    # BEFORE: sqrt(I) field (DC term present), uncorrected geometry
    amp_before, _ = recon_inline(
        intensity, wavelength_m=wavelength_m, z_m=z_pred_mm * 1e-3, px_m=dx_m
    )
    score_before = gradient_tamura(amp_before)

    # AFTER: DC-stripped contrast field at the magnification-corrected depth --
    # exactly the field focus_score() builds internally.
    z_eff_mm = dlhm_effective_z_mm(z_pred_mm, l_mm)
    contrast = (intensity / max(float(intensity.mean()), 1e-12) - 1.0).astype(np.complex64)
    amp_after, _ = recon_inline(
        intensity,
        wavelength_m=wavelength_m,
        z_m=z_eff_mm * 1e-3,
        px_m=dx_m,
        field0=contrast,
    )
    score_after = gradient_tamura(amp_after)

    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, amp, title, score in [
        (
            ax_b,
            amp_before,
            f"Before: DC-dominated, uncorrected z = {z_pred_mm:.3g} mm",
            score_before,
        ),
        (
            ax_a,
            amp_after,
            f"After: DC-stripped, corrected $z_{{eff}}$ = {z_eff_mm:.3g} mm",
            score_after,
        ),
    ]:
        # robust display range so the DC background doesn't wash out the panel
        lo, hi = np.percentile(amp, [1, 99])
        ax.imshow(amp, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"gradient-Tamura = {score:.4f}", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"z_eff = {z_eff_mm:.4f} mm | score before = {score_before:.4f} | "
        f"score after = {score_after:.4f}"
    )
