"""Scaled single-FT Fresnel propagation with a FIXED output pixel pitch.

Validity: the input pre-chirp exp(ik x^2 / 2z) must be sampled, which gives
the SAME bound as plain single-FT: z >= N*dx^2/lambda. All z_eff values in
the ODP-DLHM dataset satisfy it (min ~64 mm vs z_c ~23-37 mm); the function
asserts rather than silently degrading.

Because gradient-Tamura uses |U| only, the unit-modulus output phase factors
of the Fresnel integral are omitted.

The zoom DFT is computed by exact separable matrix multiplication (verified
against a brute-force DFT); at N = 512 this is a pair of 512^3 complex
matmuls, a few ms with BLAS. A Bluestein-CZT version would be O(N^2 log N)
if this ever becomes the bottleneck.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def critical_z_m(n_grid: int, px_m: float, wavelength_m: float) -> float:
    """Pre-chirp sampling bound: valid for z >= N * dx^2 / lambda."""
    return n_grid * px_m**2 / wavelength_m


def _zoom_dft2(
    g: npt.NDArray[np.complex64], delta_f: float, px_m: float
) -> npt.NDArray[np.complex64]:
    """Centered 2D DFT of g evaluated at output frequency pitch ``delta_f``.

    F[k, l] = sum_{n, m} g[n, m] exp(-2i pi delta_f px (k n + l m))
    with n, m, k, l centered on the grid.
    """
    n = g.shape[0]
    idx = np.arange(n) - n // 2
    e = np.exp(-2j * np.pi * delta_f * px_m * np.outer(idx, idx)).astype(np.complex64)
    return e @ g.astype(np.complex64) @ e.T


def recon_scaled(
    field0: npt.NDArray[np.complex64],
    wavelength_m: float,
    z_m: float,
    px_m: float,
    px_out_m: float | None = None,
) -> npt.NDArray[np.float32]:
    """Fresnel amplitude at distance z on a fixed output grid.

    ``px_out_m`` defaults to the input (sensor) pitch, so the magnified
    image renders on the same N x N grid for every z -- the property the
    focus-metric sweep requires. Returns amplitude only.
    """
    n = field0.shape[0]
    if field0.shape[0] != field0.shape[1]:
        raise ValueError("square input expected")
    z_c = critical_z_m(n, px_m, wavelength_m)
    if z_m < z_c:
        raise ValueError(
            f"z = {z_m * 1e3:.1f} mm below single-FT validity bound "
            f"z_c = {z_c * 1e3:.1f} mm; use the TF method (recon_inline) there."
        )
    if px_out_m is None:
        px_out_m = px_m

    k = 2.0 * np.pi / wavelength_m
    x = (np.arange(n) - n // 2) * px_m
    pre = np.exp(1j * k / (2.0 * z_m) * np.add.outer(x**2, x**2))
    delta_f = px_out_m / (wavelength_m * z_m)  # output sample x2 -> freq x2/(lam z)
    u1 = _zoom_dft2(field0 * pre, delta_f, px_m)
    return np.abs(u1).astype(np.float32)


def focus_score_far_field(
    intensity: npt.NDArray[np.float32],
    wavelength_m: float,
    z_m: float,
    px_m: float,
) -> float:
    """Label-free gradient-Tamura focus score, fixed-grid scaled propagator.

    Same signature as the v2 function, so validation_sweep.py needs only its
    import changed. Uses the same DC-stripped contrast field as holod's
    focus_score.
    """
    from holod.core.optics.reconstruction import gradient_tamura

    meas = np.asarray(intensity, dtype=np.float32)
    contrast = (meas / max(float(meas.mean()), 1e-12) - 1.0).astype(np.complex64)
    amplitude = recon_scaled(contrast, wavelength_m, z_m, px_m)
    return gradient_tamura(amplitude)
