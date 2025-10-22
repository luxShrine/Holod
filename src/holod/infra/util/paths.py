"""Exposes common paths useful for manipulating datasets and generating figures.

Styled after https://github.com/showyourwork/showyourwork/tree/main.

"""

import logging
from pathlib import Path

import click

from holod.infra.log import get_logger, log_and_echo

logger = get_logger(__name__)


def path_check(paths: dict[str, Path], *, raise_click: bool = True) -> None:
    """Ensure all paths passed in exist and surface actionable errors.

    Args:
        paths: Mapping of symbolic names to filesystem locations.
        raise_click: When ``True``, missing paths are turned into ``ClickException``
            instances so CLI callers get concise error messages.
    """
    for variable, path_to_check in paths.items():
        try:
            exists = path_to_check.exists()
        except Exception as exc:
            log_and_echo(
                logging.ERROR,
                "[bold red]%s[/] could not be accessed at [underline]%s[/]: %s",
                variable,
                path_to_check,
                exc,
                logger=logger,
                console=not raise_click,
                echo=not raise_click,
            )
            if raise_click:
                raise click.ClickException(
                    f"Unable to access {variable!r} at {path_to_check}"
                ) from exc
            raise
        if not exists:
            log_and_echo(
                logging.ERROR,
                "[bold red]%s[/] not found at [underline]%s[/]",
                variable,
                path_to_check,
                logger=logger,
                console=not raise_click,
                echo=not raise_click,
            )
            if raise_click:
                raise click.ClickException(f"{variable!r} not found at {path_to_check}")
            raise FileNotFoundError(f"{variable!r} not found at {path_to_check}")


def repo_root() -> Path:
    """Return the path to the repository root."""
    return Path(__file__).resolve().parents[4].absolute()


def src_root() -> Path:
    """Return the path to the ``src`` directory."""
    return repo_root() / "src"


def holo_root() -> Path:
    """Return path to the holod package root (src/holod)."""
    return src_root() / "holod"


def report_path(figure: bool = False) -> Path:
    """Return path to the directory storing reports (reports/ or reports/figures)."""
    report = repo_root() / "reports"
    fig = report / "figures"
    report.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    if figure:
        return fig
    return report


def checkpoints_loc() -> Path:
    """Return path to the directory storing checkpoints and models (src/checkpoints)."""
    check = src_root() / "checkpoints"
    check.mkdir(exist_ok=True)
    return check


def t_loc() -> Path:
    """Return path to the directory containing the tests (src/tests)."""
    return src_root() / "tests"


def data_root() -> Path:
    """Return the directory that holds datasets (src/data)."""
    return src_root() / "data"


def data_spec(name: str) -> Path:
    """Path to the datasets in (src/data/)."""
    match name.lower():
        case "bridge":
            # Path to the ``bridge100k`` dataset (src/data/bridge100k).
            return data_root() / "bridge100k"
        case "brownian":
            # Path to the Brownian motion dataset.
            return data_root() / "Brownian_Motion_Strouhal_Analysis_Data"
        case "mw":
            # Path to the primary training dataset (src/data/MW-Dataset).
            return data_root() / "MW-Dataset"
        case "thalassemic":
            # Path to the normal and thalassemic cells dataset.
            return data_root() / "normal_and_thalassemic_cells"
        case "phase-only":
            # Path to the phase-only hologram dataset.
            return data_root() / "Phase_Only_Holograms"
        case "maynooth":
            # Path to the Maynooth dataset.
            return data_root() / "DHM_1"
        case _:
            raise Exception("Unknown path requested.")


def tex_root() -> Path:
    """Return path to the ``tex`` directory (src/tex)."""
    return src_root() / "tex"


def figures_tex() -> Path:
    """Path where LaTeX figures are written (src/tex/figures)."""
    return tex_root() / "figures"


def output_tex() -> Path:
    """Path for miscellaneous LaTeX output files (src/tex/output)."""
    return tex_root() / "output"
