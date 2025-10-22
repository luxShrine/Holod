import json
from pathlib import Path
from typing import Any

import click
import numpy.typing as npt

from holod.infra.log import get_logger, init_logging
from holod.infra.util.paths import checkpoints_loc, data_spec, path_check, report_path

# Must be called before anything logs
init_logging()
logger = get_logger(__name__)


@click.group()
def cli():
    """Entry point for the command line interface."""
    pass


@cli.command()
@click.option(
    "--csv-name",
    "meta_csv_name",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to the metadata CSV file.",
)
@click.option(
    "--bins",
    "num_classes",
    default=None,
    type=click.IntRange(1, 100),
    help="Number of classifications, set to 1 for regression training.",
)
@click.option(
    "--model",
    "backbone",
    default=None,
    type=click.Choice(["efficientnet", "vit", "resnet50"]),
    help="Model backbone name.",
)
@click.option(
    "--crop",
    "crop_size",
    default=None,
    type=click.IntRange(20, 516),
    help="Size to crop images to.",
)
@click.option(
    "--split",
    "val_split",
    default=None,
    type=click.FloatRange(0.001, 1),
    help="Fraction of data for validation.",
)
@click.option(
    "--batch",
    "batch_size",
    default=None,
    type=click.IntRange(1, 64),
    help="Training batch size.",
)
@click.option(
    "--ep",
    "epoch_count",
    default=None,
    type=click.IntRange(1, 1000),
    help="Number of training epochs.",
)
@click.option(
    "--lr",
    "learning_rate",
    default=None,
    type=click.FloatRange(1e-6, 1e2),
    help="How fast should the model adapt epoch to epoch",
)
@click.option(
    "--device",
    "device_user",
    default=None,
    type=click.Choice(["cuda", "cpu"]),
    help="Device for training.",
)
@click.option("--fixed-seed", "fixed_seed", default=None, help="Keep the random seed consistent.")
@click.option("--continue", "continue_train", default=None, help="Continue from checkpoint")
@click.option("--create-csv", "create_csv", default=None, help="Create a CSV for data.")
@click.option("--sample", "use_sample_data", default=None, help="Use sample data provided.")
def train(
    meta_csv_name: str | None,
    num_classes: int | None,
    backbone: str | None,
    crop_size: int | None,
    val_split: float | None,
    batch_size: int | None,
    epoch_count: int | None,
    learning_rate: float | None,
    device_user: str | None,
    fixed_seed: bool | None,
    continue_train: bool | None,
    create_csv: bool | None,
    use_sample_data: bool | None,
) -> None:
    """Train the autofocus model based on supplied dataset."""
    from datetime import datetime

    from serde.json import to_json
    from serde.toml import from_toml

    from holod.core.plots import PlotPred
    from holod.infra.dataclasses import AutoConfig, Flags, Paths, Train, UserConfig
    from holod.infra.training import train_autofocus

    # TODO: make mutually exclusive pairs of the following for the user to prevent odd cases:
    # - use sample & create csv
    # - use sample & dataset path
    # - use sample & resume training
    # maybe prompt the user to utilize sample?

    autofocus_config: AutoConfig
    path_ckpt: str | None
    if Path("train_settings.toml").exists():
        with open("train_settings.toml") as config_file:
            config = config_file.read()
        config = from_toml(UserConfig, config)
        # merge the config file (if it exists), by overwriting it with CLI args
        config.merge(
            paths=Paths(None, meta_csv_name),
            train=Train(
                backbone,
                batch_size,
                crop_size,
                device_user,
                epoch_count,
                learning_rate,
                num_classes,
                None,
                None,
                None,
                None,
                val_split,
            ),
            flags=Flags(continue_train, create_csv, fixed_seed, use_sample_data),
        )
        path_ckpt = (
            checkpoints_loc().as_posix()
            if (config.flags.checkpoint or continue_train) is True
            else None
        )
        autofocus_config = config.to_auto_config()

    else:
        logger.error(
            "Config file not found, expected 'train_settings.toml' in repo root. Using \
        default settings."
        )
        autofocus_config = AutoConfig()

    # plot_info = train_autofocus_lightning(autofocus_config, path_ckpt=None)
    plot_info: PlotPred = train_autofocus(autofocus_config, path_ckpt)
    current_time = datetime.now()
    current_time = current_time.strftime("%H%M%S")
    with open(report_path() / "plot_info.json", "w", encoding="utf-8") as f:
        plot_info_json = to_json(plot_info, PlotPred)
        f.write(plot_info_json)


@cli.command()
# TODO: make this utilize a proper enum, not true/false <luxShrine>
@click.option("--display", default="save", help="Save the output plots, show them, or both.")
def plot_train(
    display: str,
):
    """Plot the data saved from autofocus training."""
    from serde.json import from_json

    from holod.core.plots import PlotPred  # performance reasons, import locally in function
    from holod.infra.util.types import AnalysisType, DisplayType

    logger.info("plotting training data...")

    with open(report_path() / "plot_info.json") as f:
        plot_info: PlotPred = from_json(PlotPred, f.read())

    assert isinstance(plot_info, PlotPred), f"plot_info is not PlotPred, found {type(plot_info)}"
    # update plot obj with desired values
    display_check = DisplayType(display)
    logger.info(f"Plotting function with option: {display}")

    meta_list: list[str] = []
    # TODO: automatically get analysis type
    print(plot_info.analysis)
    if plot_info.analysis == AnalysisType.CLASS.value:
        cls_res_cm = plot_info.plot_classification(display_check)
        if cls_res_cm is not None:
            meta_list = cls_res_cm
    elif plot_info.analysis == AnalysisType.REG.value:
        reg_plots = plot_info.plot_regression(display_check)
        if reg_plots is not None:
            meta_list = reg_plots

    if all(meta_list) is not None:
        meta_dict = {"items": meta_list}
        with open(report_path() / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)


@cli.command()
@click.argument("img_file_path")
@click.option("--amp_true", default=None, help="True amplitude")
@click.option("--phase_true", default=None, help="True Phase")
@click.option(
    "--model_path",
    default="best_model.pth",
    help="Path to trained model to use for torch optics analysis",
)
@click.option("--crop_size", default=512, help="Pixel width and height of image")
@click.option(
    "--wavelength", default=530e-9, help="Wavelength of light used to capture the image (m)"
)
@click.option("--z", default=20e-3, help="Distance of measurement (m)")
@click.option("--dx", default=1e-6, help="Size of image px (m)")
@click.option(
    "--display",
    default="save",
    help="Show, Save or do both for resulting phase and amplitude images",
)
def reconstruction(
    img_file_path: str,
    model_path: str | Path,
    crop_size: int,
    wavelength: float,
    z: float,
    dx: float,
    display: str,
    amp_true: None | npt.NDArray[Any],
    phase_true: None | npt.NDArray[Any],
):
    """Perform reconstruction on an hologram."""
    import holod.core.plots as plots  # performance reasons, import locally in function
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


cli.add_command(train)
if __name__ == "__main__":
    cli()
