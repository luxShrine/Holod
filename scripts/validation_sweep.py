"""Focus-metric validation sweep against ground-truth depth labels.

Acceptance test for the corrected label-free focus metric: for a stratified
sample of labeled holograms, sweep candidate source-sample z, map each
candidate through the DLHM magnification correction, score the reconstruction
with the regime-correct propagator, and check that the score peaks at (or
within tolerance of) the ground-truth z.

Outputs:
    sweep_grid.png    -- one panel per hologram: score vs candidate z,
                         vertical line at ground truth, peak marked
    sweep_results.csv -- per-hologram peak z, gt z, error, pass/fail,
                         and the corrected score at the gt depth
    console summary   -- pass rate + corrected-score stats to replace
                         the stale 0.81 +/- 0.05 on slides 8/9

CSV schema expected (per the dataset README):
    path        image path relative to the CSV
    Wavelength  illumination wavelength in MICROMETERS (e.g. 0.405)
    L_value     source-to-sensor distance, mm
    z_value     ground-truth source-to-sample distance, mm
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from holod.core.optics.reconstruction import dlhm_effective_z_mm, load_intensity
from holod.core.optics.scaled_fresnel import focus_score_far_field
from holod.infra.util.types import SENSOR_PIXEL_PITCH_M


def read_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            true_path = Path("./src/data/MW_Dataset_Sample") / r["path"]
            if not true_path.exists():
                continue
            rows.append(
                {
                    "path": r["path"],
                    "wavelength_m": float(r["Wavelength"]) * 1e-6,  # um -> m
                    "l_mm": float(r["L_value"]),
                    "z_mm": float(r["z_value"]),
                }
            )
    if not rows:
        raise SystemExit(f"No rows read from {csv_path}")
    return rows


def stratified_pick(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Spread the picks across wavelengths and the z range."""
    rng = np.random.default_rng(seed)
    by_lam: dict[float, list[dict]] = {}
    for r in rows:
        by_lam.setdefault(round(r["wavelength_m"], 9), []).append(r)
    picks: list[dict] = []
    lams = sorted(by_lam)
    per_lam = max(1, n // len(lams))
    for lam in lams:
        grp = sorted(by_lam[lam], key=lambda r: r["z_mm"])
        # z-quantile spread within this wavelength
        idx = np.linspace(0, len(grp) - 1, per_lam).round().astype(int)
        # jitter so reruns with a different seed sample different holograms
        idx = np.clip(idx + rng.integers(-2, 3, size=idx.size), 0, len(grp) - 1)
        picks.extend(grp[i] for i in sorted(set(idx.tolist())))
    return picks[:n] if len(picks) > n else picks


def sweep_one(
    intensity: np.ndarray,
    wavelength_m: float,
    l_mm: float,
    z_grid_mm: np.ndarray,
    px_m: float,
) -> np.ndarray:
    scores = np.empty(z_grid_mm.size)
    for j, z_mm in enumerate(z_grid_mm):
        z_eff_mm = dlhm_effective_z_mm(float(z_mm), l_mm)
        scores[j] = focus_score_far_field(intensity, wavelength_m, z_eff_mm * 1e-3, px_m)
    return scores


def do_sweep(
    ds_root,
    csv_name,
    n_holograms,
    crop,
    n_z,
    z_margin_mm,
    tol_mm,
    outdir: Path,
    seed: int = 42,
) -> None:

    rows = read_rows(ds_root / csv_name)
    z_all = np.array([r["z_mm"] for r in rows])
    z_grid = np.linspace(
        max(1e-3, z_all.min() - z_margin_mm),
        z_all.max() + z_margin_mm,
        n_z,
    )
    picks = stratified_pick(rows, n_holograms, seed)
    outdir.mkdir(parents=True, exist_ok=True)

    ncols = 3
    nrows_fig = math.ceil(len(picks) / ncols)
    fig, axes = plt.subplots(
        nrows_fig, ncols, figsize=(4.2 * ncols, 3.2 * nrows_fig), squeeze=False
    )
    results: list[dict] = []

    for ax_i, rec in enumerate(picks):
        img_path = ds_root / str(rec["path"])
        intensity = load_intensity(img_path, crop)
        scores = sweep_one(
            intensity,
            rec["wavelength_m"],
            rec["l_mm"],
            z_grid,
            SENSOR_PIXEL_PITCH_M,
        )
        z_peak = float(z_grid[int(np.argmax(scores))])
        z_gt = float(rec["z_mm"])
        err = z_peak - z_gt
        ok = abs(err) <= tol_mm

        # corrected score at the ground-truth depth -> replaces slide number
        z_eff_gt = dlhm_effective_z_mm(z_gt, float(rec["l_mm"]))
        score_at_gt = focus_score_far_field(
            intensity,
            float(rec["wavelength_m"]),
            z_eff_gt * 1e-3,
            SENSOR_PIXEL_PITCH_M,
        )
        results.append(
            {
                "path": rec["path"],
                "wavelength_nm": round(rec["wavelength_m"] * 1e9),
                "L_mm": rec["l_mm"],
                "z_gt_mm": z_gt,
                "z_peak_mm": round(z_peak, 4),
                "peak_error_mm": round(err, 4),
                "pass": ok,
                "score_at_gt": round(score_at_gt, 4),
            }
        )

        ax = axes[ax_i // ncols][ax_i % ncols]
        ax.plot(z_grid, scores, lw=1.4)
        ax.axvline(z_gt, color="k", ls="--", lw=1, label=f"gt {z_gt:.2f} mm")
        ax.axvline(
            z_peak, color="tab:green" if ok else "tab:red", lw=1, label=f"peak {z_peak:.2f} mm"
        )
        ax.set_title(
            f"{Path(str(rec['path'])).name}  "
            f"lam={rec['wavelength_m'] * 1e9:.0f}nm  "
            f"{'PASS' if ok else 'FAIL'} ({err:+.2f} mm)",
            fontsize=8,
        )
        ax.set_xlabel("candidate z (mm)", fontsize=8)
        ax.set_ylabel("focus score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for k in range(len(picks), nrows_fig * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "sweep_grid.png", dpi=170, bbox_inches="tight")

    with open(outdir / "sweep_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    n_pass = sum(r["pass"] for r in results)
    errs = np.array([r["peak_error_mm"] for r in results])
    gts = np.array([r["score_at_gt"] for r in results])
    print(f"\n{'=' * 62}")
    print(f"Pass: {n_pass}/{len(results)} within +/-{tol_mm} mm")
    print(
        f"Peak error: mean {errs.mean():+.3f} mm, "
        f"MAE {np.abs(errs).mean():.3f} mm, max |err| {np.abs(errs).max():.3f} mm"
    )
    print(
        f"Corrected score at gt depth: {gts.mean():.3f} +/- {gts.std():.3f} "
        f"(replaces the stale 0.81 +/- 0.05 -- expand n before quoting)"
    )
    print(f"Outputs: {outdir / 'sweep_grid.png'}, {outdir / 'sweep_results.csv'}")
    if n_pass < len(results):
        fails = [r for r in results if not r["pass"]]
        print("\nFailures to investigate (wavelength, z regime):")
        for r in fails:
            print(
                f"  {r['path']}  lam={r['wavelength_nm']}nm  "
                f"gt={r['z_gt_mm']:.2f}  peak={r['z_peak_mm']:.2f}"
            )
    print(f"{'=' * 62}")
