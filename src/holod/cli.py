import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy.typing as npt
from click.core import ParameterSource

from holod.infra.dataclasses import CompareUserConfig
from holod.infra.log import get_logger, init_logging
from holod.infra.util.paths import (
    checkpoints_loc,
    data_spec,
    latest_epoch_checkpoint,
    path_check,
    report_path,
)
from holod.infra.util.types import ModelType

if TYPE_CHECKING:
    from holod.core.plots import PlotPred

# Must be called before anything logs
init_logging()
logger = get_logger(__name__)
TRAIN_SETTINGS_STR = "train_settings.toml"


def _cli_overrides(**values: Any) -> dict[str, Any]:
    """Return only the options the user explicitly set on the command line.

    Keys must match the Click parameter names. Options left at their CLI
    default are dropped, so the defaults shown in ``--help`` never clobber
    values from train_settings.toml.
    """
    ctx = click.get_current_context()
    return {
        name: value
        for name, value in values.items()
        if ctx.get_parameter_source(name) != ParameterSource.DEFAULT
    }


def _resume_checkpoint(selected: ModelType) -> str | None:
    """Return the latest epoch checkpoint for a backbone, or ``None`` to start fresh."""
    latest_ckpt = latest_epoch_checkpoint(selected.name)
    if latest_ckpt is not None:
        return latest_ckpt.as_posix()
    logger.warning(
        f"No epoch checkpoint found for {selected.name} in {checkpoints_loc()}, starting fresh."
    )
    return None


@click.group()
def cli():
    """Entry point for the command line interface."""
    pass


@cli.command()
@click.argument("ds_root", required=False, default="", type=click.Path(file_okay=False))
@click.option(
    "--csv-name",
    "meta_csv_name",
    default="",
    type=click.Path(dir_okay=False),
    help="Path to the metadata CSV file.",
)
@click.option(
    "--bins",
    "num_classes",
    default=10,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Number of classifications, set to 1 for regression training.",
)
@click.option(
    "--model",
    "backbone",
    default="efficientnet",
    show_default=True,
    type=click.Choice(["efficientnet", "vit", "resnet50", "focusnet", "pcnn"]),
    help="Model backbone name.",
)
@click.option(
    "--crop",
    "crop_size",
    default=224,
    show_default=True,
    type=click.IntRange(20, 516),
    help="Size to crop images to.",
)
@click.option(
    "--split",
    "val_split",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0.001, 1),
    help="Fraction of data for validation.",
)
@click.option(
    "--batch",
    "batch_size",
    default=16,
    show_default=True,
    type=click.IntRange(1, 64),
    help="Training batch size.",
)
@click.option(
    "--ep",
    "epoch_count",
    default=10,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Number of training epochs.",
)
@click.option(
    "--lr",
    "learning_rate",
    default=5e-5,
    show_default=True,
    type=click.FloatRange(1e-6, 1e2),
    help="How fast should the model adapt epoch to epoch",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    type=click.Choice(["cuda", "cpu"]),
    help="Device for training.",
)
@click.option(
    "--soft-sigma",
    "soft_label_sigma",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, 100.0),
    help="Std dev (in bins) for soft ordinal classification labels; 0 keeps hard labels.",
)
@click.option(
    "--fixed-seed/--no-fixed-seed",
    "fixed_seed",
    default=True,
    show_default=True,
    help="Keep the random seed consistent.",
)
@click.option(
    "--continue",
    "checkpoint",
    is_flag=True,
    default=False,
    show_default=True,
    help="Continue from checkpoint",
)
@click.option(
    "--create-csv",
    "create_csv",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a CSV for data.",
)
@click.option(
    "--sample",
    "sample",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use sample data provided.",
)
def train(
    ds_root: str,
    meta_csv_name: str,
    num_classes: int,
    backbone: str,
    crop_size: int,
    val_split: float,
    batch_size: int,
    epoch_count: int,
    learning_rate: float,
    device: str,
    soft_label_sigma: float,
    fixed_seed: bool,
    checkpoint: bool,
    create_csv: bool,
    sample: bool,
) -> None:
    """Train the autofocus model based on supplied dataset.

    DS_ROOT is the dataset folder; it may be a name under src/data, a path
    relative to the current directory, or an absolute path anywhere on disk.
    When omitted, the dataset_root from train_settings.toml is used.
    """
    from holod.infra.dataclasses import AutoConfig, Flags, ModelConfig, Paths, Train
    from holod.infra.training import train_autofocus

    # TODO: make mutually exclusive pairs of the following for the user to prevent odd cases:
    # - use sample & create csv
    # - use sample & dataset path
    # - use sample & resume training
    # maybe prompt the user to utilize sample?

    autofocus_config: AutoConfig
    path_ckpt: str | None = None
    train_settings = Path(TRAIN_SETTINGS_STR)
    selected = ModelType.from_str(backbone)
    if train_settings.exists():
        config = CompareUserConfig.from_toml(train_settings)
        # config-file values are the base; only options the user explicitly
        # passed on the command line override them
        config.merge(
            flags=Flags(
                **_cli_overrides(
                    checkpoint=checkpoint,
                    create_csv=create_csv,
                    fixed_seed=fixed_seed,
                    sample=sample,
                )
            ),
            paths=Paths(ds_root, meta_csv_name),
            **_cli_overrides(
                batch_size=batch_size,
                crop_size=crop_size,
                device=device,
                epoch_count=epoch_count,
                val_split=val_split,
                num_classes=num_classes,
                soft_label_sigma=soft_label_sigma,
            ),
        )
        if config.flags.checkpoint:
            path_ckpt = _resume_checkpoint(selected)
        # --lr targets the selected model's config, since learning rate is per-model now
        model_config = config.model_config(selected)
        if model_config is not None and _cli_overrides(learning_rate=learning_rate):
            model_config.train.learning_rate = learning_rate
        autofocus_config = config.to_auto_config(selected)

    elif ds_root != "":
        # no config file, but a dataset was supplied on the CLI: build the config
        # purely from CLI arguments (including their defaults)
        logger.warning("Config file 'train_settings.toml' not found, using CLI arguments only.")
        model_config = ModelConfig(
            train=Train(
                backbone=backbone,
                learning_rate=learning_rate,
                optimizer_weight_decay=None,
                sch_factor=None,
                sch_patience=None,
            )
        )
        config = CompareUserConfig.from_model_config(model_config)
        config = config.merge(
            flags=Flags(checkpoint, create_csv, fixed_seed, sample),
            paths=Paths(ds_root, meta_csv_name),
            batch_size=batch_size,
            crop_size=crop_size,
            device=device,
            epoch_count=epoch_count,
            val_split=val_split,
            num_classes=num_classes,
            soft_label_sigma=soft_label_sigma,
        )
        if checkpoint:
            path_ckpt = _resume_checkpoint(selected)
        autofocus_config = config.to_auto_config(selected)

    else:
        raise Exception("Config file not found, expected 'train_settings.toml' in repo root.")

    plot_info: PlotPred = train_autofocus(autofocus_config, path_ckpt)
    plot_info.save_to_file()


@cli.command()
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(["efficientnet", "vit", "resnet50", "focusnet", "new"]),
    help="Backbone to include; repeat for several. Defaults to all backbones.",
)
@click.option(
    "--bins",
    "num_classes",
    default=10,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Number of classifications, set to 1 for regression training.",
)
@click.option(
    "--crop",
    "crop_size",
    default=224,
    show_default=True,
    type=click.IntRange(20, 516),
    help="Size to crop images to.",
)
@click.option(
    "--batch",
    "batch_size",
    default=16,
    show_default=True,
    type=click.IntRange(1, 64),
    help="Training batch size.",
)
@click.option(
    "--ep",
    "epoch_count",
    default=10,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Number of training epochs per backbone.",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    type=click.Choice(["cuda", "cpu"]),
    help="Device for training.",
)
@click.option(
    "--soft-sigma",
    "soft_label_sigma",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, 100.0),
    help="Std dev (in bins) for soft ordinal classification labels; 0 keeps hard labels.",
)
@click.option(
    "--sample",
    "sample",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use sample data provided.",
)
@click.option(
    "--display",
    default="save",
    show_default=True,
    type=click.Choice(["save", "show", "both"]),
    help="Save the backbone-comparison plot, show it, or both.",
)
def compare(
    models: tuple[str, ...],
    num_classes: int,
    crop_size: int,
    batch_size: int,
    epoch_count: int,
    device: str,
    soft_label_sigma: float,
    sample: bool,
    display: str,
) -> None:
    """Compare each configured model backbone under the shared configuration."""
    from holod.core.compare import compare_backbones
    from holod.infra.dataclasses import Flags, Paths
    from holod.infra.log import console_ as console
    from holod.infra.util.types import DisplayType, ModelType

    selected = [ModelType(m) for m in models] if models else None

    config: CompareUserConfig
    train_settings = Path(TRAIN_SETTINGS_STR)
    if train_settings.exists():
        config = CompareUserConfig.from_toml(train_settings)
        # config-file values are the base; only options the user explicitly
        # passed on the command line override them
        config.merge(
            flags=Flags(**_cli_overrides(sample=sample)),
            paths=Paths.empty(),
            **_cli_overrides(
                batch_size=batch_size,
                crop_size=crop_size,
                device=device,
                epoch_count=epoch_count,
                num_classes=num_classes,
                soft_label_sigma=soft_label_sigma,
            ),
        )
    else:
        raise Exception("Config file not found, expected 'train_settings.toml' in repo root.")

    report = compare_backbones(config, backbones=selected)
    console.print(report.to_table())
    report.save()
    report.plot(DisplayType(display))


@cli.command()
@click.argument("img_file_path")
@click.option(
    "--model-path",
    "model_paths",
    multiple=True,
    type=click.Path(dir_okay=False),
    help="Checkpoint to evaluate; repeat for several. Defaults to every model in src/checkpoints.",
)
@click.option(
    "--runs",
    default=5,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Repeated predictions to time per model.",
)
@click.option(
    "--crop_size",
    default=224,
    show_default=True,
    help="Pixel width and height to center-crop input to",
)
@click.option(
    "--wavelength",
    default=530e-9,
    show_default=True,
    help="Wavelength of light used to capture the image (m)",
)
# default matches SENSOR_PIXEL_PITCH_M (kept literal here: heavy types import stays lazy)
@click.option(
    "--dx",
    default=3.8e-6,
    show_default=True,
    help="Pixel pitch of the capture sensor (m)",
)
@click.option(
    "--z-true",
    "z_true_mm",
    default=None,
    type=float,
    help="Ground-truth depth (mm); enables ranking models by absolute error.",
)
@click.option(
    "--l-value",
    "l_mm",
    default=None,
    type=float,
    help="DLHM source-to-screen distance L (mm, the dataset's L_value); lets focus "
    "scoring reconstruct at the correct effective depth.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cuda", "cpu"]),
    help="Device for evaluation; 'auto' picks CUDA when available.",
)
@click.option(
    "--display",
    default="save",
    show_default=True,
    type=click.Choice(["save", "show", "both"]),
    help="Save the depth-prediction plot, show it, or both.",
)
def compare_holo(
    img_file_path: str,
    model_paths: tuple[str, ...],
    runs: int,
    crop_size: int,
    wavelength: float,
    dx: float,
    z_true_mm: float | None,
    l_mm: float | None,
    device: str,
    display: str,
) -> None:
    """Run comparison test evaluations of trained models on a single hologram."""
    from holod.core.compare import compare_on_hologram
    from holod.infra.log import console_ as console
    from holod.infra.util.types import DisplayType

    ckpt_paths: list[Path] | None = None
    if model_paths:
        # resolve bare filenames against the checkpoints directory
        ckpt_paths = [
            path if path.exists() else checkpoints_loc() / path
            for path in (Path(m) for m in model_paths)
        ]
    path_check({"img_file_path": Path(img_file_path)})

    report = compare_on_hologram(
        img_file_path,
        ckpt_paths=ckpt_paths,
        runs=runs,
        crop_size=crop_size,
        wavelength=wavelength,
        dx=dx,
        z_true_mm=z_true_mm,
        l_mm=l_mm,
        device=None if device == "auto" else device,
    )
    console.print(report.to_table())
    report.save()
    report.plot(DisplayType(display))


@cli.command()
@click.option(
    "--display",
    default="save",
    show_default=True,
    type=click.Choice(["save", "show", "both", "meta"]),
    help="Save the output plots, show them, both, or write plot metadata only.",
)
def plot_train(
    display: str,
):
    """Plot the data saved from autofocus training."""
    from holod.core.plots import PlotPred  # performance reasons, import locally in function
    from holod.infra.util.types import AnalysisType, DisplayType

    logger.info("plotting training data...")
    jsons = list(report_path().glob("plot_info_*.json"))

    for idx, path_to_json in enumerate(jsons):
        # tie output filenames to the source json's backbone+timestamp token so runs
        # (including several backbones saved by one `compare`) don't overwrite each other
        time_in_title = path_to_json.stem.removeprefix("plot_info_") or str(idx)

        try:
            plot_info = PlotPred.load_from_file(path_to_json)

            # update plot obj with desired values
            display_check = DisplayType(display)
            logger.debug(f"Plotting function with option: {display}")

            meta_list: list[dict[str, Any]] = []
            # TODO: automatically get analysis type
            if plot_info.analysis == AnalysisType.CLASS.value:
                cls_res_cm = plot_info.plot_classification(display_check, time_in_title)
                if cls_res_cm is not None:
                    meta_list = cls_res_cm
            elif plot_info.analysis == AnalysisType.REG.value:
                reg_plots = plot_info.plot_regression(display_check, time_in_title)
                if reg_plots is not None:
                    meta_list = reg_plots
        except Exception:
            logger.exception(f"Failed to plot {path_to_json.name}, skipping.")
            continue

        if meta_list:
            meta_dict = {"items": meta_list}
            with open(report_path() / f"meta_{time_in_title}.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, ensure_ascii=False, indent=2)


@cli.command()
@click.argument("img_file_path")
@click.option("--amp_true", default=None, help="True amplitude")
@click.option("--phase_true", default=None, help="True Phase")
@click.option(
    "--model_path",
    default="best_model.pth",
    show_default=True,
    help="Path to trained model to use for torch optics analysis",
)
@click.option(
    "--crop_size",
    default=512,
    show_default=True,
    help="Pixel width and height of image",
)
@click.option(
    "--wavelength",
    default=530e-9,
    show_default=True,
    help="Wavelength of light used to capture the image (m)",
)
# default matches SENSOR_PIXEL_PITCH_M (kept literal here: heavy types import stays lazy)
@click.option(
    "--dx",
    default=3.8e-6,
    show_default=True,
    help="Pixel pitch of the capture sensor (m)",
)
@click.option(
    "--display",
    default="save",
    show_default=True,
    type=click.Choice(["save", "show", "both"]),
    help="Show, Save or do both for resulting phase and amplitude images",
)
def reconstruction(
    img_file_path: str,
    model_path: str | Path,
    crop_size: int,
    wavelength: float,
    dx: float,
    display: str,
    amp_true: None | npt.NDArray[Any],
    phase_true: None | npt.NDArray[Any],
):
    """Perform reconstruction on an hologram."""
    import holod.core.plots as plots
    from holod.infra.util.types import DisplayType

    DEF_IMG_FILE_PATH = (data_spec("mw") / "510" / "10_Phase_USAF" / "z10" / "10.jpg").as_posix()
    if img_file_path == "":
        img_file_path = DEF_IMG_FILE_PATH

    display_c = DisplayType(display)
    model_path = checkpoints_loc() / Path(model_path)
    model_ext = ".pth"
    if not model_path.exists():
        try:
            existing_models = list(checkpoints_loc().glob(f"*{model_ext}"))
            existing_models_index = [*range(len(existing_models))]
            existing_models_index = [str(i) for i in existing_models_index]
            choices = existing_models_index + ["q"]

            print(f"Model at {model_path} does not exist, models that do exist are:")
            [print(idx, f.name) for (idx, f) in enumerate(existing_models)]

        except Exception as e:
            logger.exception(f"Failed to find any {model_ext} type files in {checkpoints_loc()}")
            raise e

        user_response: str = click.prompt(
            "Enter the number that corresponds to an an existing model or type 'q' to quit",
            show_choices=False,
            type=click.Choice(choices, case_sensitive=False),
        )

        if user_response in existing_models_index:
            model_path = existing_models[int(user_response)]
            path_check({"img_file_path": Path(img_file_path)})

            # perform reconstruction
            plots.plot_amp_phase(
                crop_size=crop_size,
                ckpt_file=model_path,
                display=display_c,
                dx=dx,
                img_file_path=img_file_path,
                wavelength=wavelength,
                amp_true=amp_true,
                phase_true=phase_true,
            )
        elif user_response == "q":
            logger.info("Exiting...")
            pass
        else:
            raise RuntimeError("failed to parse user choice using model.")

    else:
        path_check({"img_file_path": Path(img_file_path)})

        # perform reconstruction
        plots.plot_amp_phase(
            crop_size=crop_size,
            ckpt_file=model_path,
            display=display_c,
            dx=dx,
            img_file_path=img_file_path,
            wavelength=wavelength,
            amp_true=amp_true,
            phase_true=phase_true,
        )


@cli.command()
@click.argument("ds_root", required=False, default="", type=click.Path(file_okay=False))
@click.option(
    "--csv-name",
    "meta_csv_name",
    default="",
    type=click.Path(dir_okay=False),
    help="Path to the metadata CSV file.",
)
@click.option(
    "--bins",
    "num_classes",
    default=10,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Number of classifications, set to 1 for regression training.",
)
@click.option(
    "--model",
    "backbone",
    default="efficientnet",
    show_default=True,
    type=click.Choice(["efficientnet", "vit", "resnet50", "focusnet", "pcnn"]),
    help="Model backbone name.",
)
@click.option(
    "--crop",
    "crop_size",
    default=224,
    show_default=True,
    type=click.IntRange(20, 516),
    help="Size to crop images to.",
)
@click.option(
    "--split",
    "val_split",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0.001, 1),
    help="Fraction of data for validation.",
)
@click.option(
    "--batch",
    "batch_size",
    default=16,
    show_default=True,
    type=click.IntRange(1, 64),
    help="Training batch size.",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    type=click.Choice(["cuda", "cpu"]),
    help="Device for training.",
)
@click.option(
    "--soft-sigma",
    "soft_label_sigma",
    default=0.0,
    show_default=True,
    type=click.FloatRange(0.0, 100.0),
    help="Std dev (in bins) for soft ordinal classification labels; 0 keeps hard labels.",
)
@click.option(
    "--create-csv",
    "create_csv",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a CSV for data.",
)
@click.option(
    "--sample",
    "sample",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use sample data provided.",
)
def determine_lr(
    ds_root: str,
    meta_csv_name: str,
    num_classes: int,
    backbone: str,
    crop_size: int,
    val_split: float,
    batch_size: int,
    device: str,
    soft_label_sigma: float,
    create_csv: bool,
    sample: bool,
) -> None:
    """Report the performance of a model across various learning rates."""
    from holod.infra.dataclasses import Flags, Paths
    from holod.infra.util.training_help import determine_learning_rate

    train_settings = Path(TRAIN_SETTINGS_STR)
    selected = ModelType.from_str(backbone)
    if train_settings.exists():
        config = CompareUserConfig.from_toml(train_settings)
        # config-file values are the base; only options the user explicitly
        # passed on the command line override them
        flags = Flags(**_cli_overrides(create_csv=create_csv, sample=sample))
        flags.fixed_seed = True  # LR sweeps always use a fixed seed for comparability
        config.merge(
            flags=flags,
            paths=Paths(ds_root, meta_csv_name),
            **_cli_overrides(
                batch_size=batch_size,
                crop_size=crop_size,
                device=device,
                val_split=val_split,
                num_classes=num_classes,
                soft_label_sigma=soft_label_sigma,
            ),
        )
    else:
        raise Exception("Config file not found, expected 'train_settings.toml' in repo root.")

    _learning_rate_report = determine_learning_rate(config, selected)


cli.add_command(train)
if __name__ == "__main__":
    cli()
