from __future__ import annotations

import argparse
from pathlib import Path

from holod.cli import TRAIN_SETTINGS_STR
from holod.core.optics.reconstruction import load_intensity
from holod.infra.dataclasses import CompareUserConfig, Flags
from holod.infra.util.prog_helper import console
from holod.infra.util.types import ModelType

from .ablation import run_fix_ablation
from .before_after import run_before_after
from .dlhm_sketch import geometry_sketch
from .validation_sweep import do_sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # before/after related
    p.add_argument("hologram", help="path to a hologram image")
    p.add_argument(
        "--z-pred-mm",
        type=float,
        required=True,
        help="model-predicted (or ground truth) source-sample z, mm",
    )
    p.add_argument("--l-mm", type=float, required=True, help="source-to-sensor distance L, mm")
    p.add_argument("--wavelength-m", type=float, default=5.1e-7)
    p.add_argument("--dx-m", type=float, default=3.8e-6)
    p.add_argument("--ba-crop", type=int, default=2048)
    p.add_argument("-geo", "--geometry", action="store_true", default=False)
    p.add_argument("-ba", "--before-after", action="store_true", default=True)

    # ablation report related
    p.add_argument(
        "-abl",
        "--ablation",
        action="store_true",
        help="Report before/after metrics for the random-crop and magnification fixes.",
        default=False,
    )
    p.add_argument("--model", default="efficientnet", help="Model backbone name.")
    p.add_argument(
        "--num_classes",
        default=10,
        help="Number of classifications, set to 1 for regression training.",
    )

    # val sweep
    p.add_argument(
        "-sw",
        "--sweep",
        action="store_true",
        help="Focus-metric validation sweep against ground-truth depth labels.",
        default=False,
    )
    p.add_argument("--n-z", type=int, default=45, help="sweep grid points")
    p.add_argument(
        "--z-margin-mm",
        type=float,
        default=0.1,
        help="extend sweep beyond dataset z min/max by this much",
    )
    p.add_argument(
        "--tol-mm", type=float, default=0.15, help="pass if |peak - gt| <= tol (~one bin width)"
    )

    # shared
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--eval-crop", default=224, help="Size to crop images to.")
    p.add_argument(
        "--holo-count",
        default=50,
        help="Validation holograms to focus-score per checkpoint (one FFT propagation each).",
    )

    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_settings = Path(TRAIN_SETTINGS_STR)

    if args.sweep or args.ablation:
        if not train_settings.exists():
            raise Exception("Config file not found, expected 'train_settings.toml' in repo root.")

        config = CompareUserConfig.from_toml(train_settings)
        flags = Flags()
        flags.fixed_seed = True  # both conditions must share the validation split
        flags.checkpoint = False  # each condition trains from scratch, never resumes
        selected = ModelType.from_str(args.model)
        num_classes: int = args.num_classes
        eval_crop: int = args.eval_crop
        holo_count: int = args.holo_count
        config.merge(
            flags=flags,
            crop_size=eval_crop,
            num_classes=num_classes,
        )

        if args.sweep is True:
            do_sweep(
                config.paths.dataset_root,
                config.paths.meta_csv_name,
                holo_count,
                eval_crop,
                args.n_z,
                args.z_margin_mm,
                args.tol_mm,
                args.outdir,
            )

        if args.ablation is True:
            report = run_fix_ablation(config, selected, holo_count=holo_count)
            console.print(report.to_table())
            report.save()

    ba = args.before_after
    geo = args.geometry
    if geo or ba:
        if geo is True:
            geometry_sketch(args.outdir / "fig_geometry_sketch.png")
        if ba is True:
            intensity = load_intensity(args.hologram, args.ba_crop)
            run_before_after(
                intensity,
                args.z_pred_mm,
                args.l_mm,
                args.wavelength_m,
                args.dx_m,
                args.outdir / "fig_before_after.png",
            )
        print(f"Wrote figures to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
