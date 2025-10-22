"""Central logging setup (queue listener pattern)."""

from __future__ import annotations

import atexit
import logging
import logging.config
import logging.handlers as lh
import multiprocessing as mp
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from multiprocessing import Queue as MPQueue
from pathlib import Path
from typing import Any, Final

import click
import rich  # noqa
from pythonjsonlogger import json
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

_LOG_FILE: Final[Path] = Path("logs/holo_log.jsonl")
_LOG_LEVEL = "DEBUG"  # root level – override per logger if needed


# Single, process‑safe queue for all log records
_queue: MPQueue[logging.LogRecord] = mp.Queue(-1)


_LEVEL_STYLES: Final[dict[int, dict[str, object]]] = {
    logging.DEBUG: {"fg": "bright_black"},
    logging.INFO: {"fg": "cyan"},
    logging.WARNING: {"fg": "yellow"},
    logging.ERROR: {"fg": "bright_red"},
    logging.CRITICAL: {"fg": "bright_white", "bg": "red", "bold": True},
}


class _RichMarkupFilter(logging.Filter):
    """Ensure records have ``markup`` enabled for RichHandler by default."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if getattr(record, "markup", None) is None:
            record.markup = True
        return True


_MARKUP_FILTER = _RichMarkupFilter()


def get_logger(name: str | None = None, *, enable_markup: bool = True) -> logging.Logger:
    """Return a logger that is wired through the Rich logging configuration."""
    init_logging()
    logger = logging.getLogger("holod" if name is None else name)
    if enable_markup and not any(isinstance(f, _RichMarkupFilter) for f in logger.filters):
        logger.addFilter(_MARKUP_FILTER)
    return logger


def log_and_echo(
    level: int,
    message: str,
    *args: object,
    logger: logging.Logger | None = None,
    markup: bool = True,
    console: bool = True,
    echo: bool = True,
    err: bool | None = None,
    **kwargs: Any,
) -> None:
    """Log ``message`` and optionally mirror it through Click with Rich styling."""
    target = logger or get_logger()
    extra = kwargs.pop("extra", {})
    extra = {**extra, "markup": markup}
    if not console:
        extra["suppress_rich_console"] = True
    target.log(level, message, *args, extra=extra, **kwargs)

    if not echo:
        return

    formatted = message % args if args else message
    plain_text = _render_for_click(formatted, markup=markup)
    style = _LEVEL_STYLES.get(level, {})
    click_kwargs = {"err": err if err is not None else level >= logging.WARNING}
    click_kwargs.update(style)

    if style:
        click.secho(plain_text, **click_kwargs)
    else:
        click.echo(plain_text, **click_kwargs)


def _render_for_click(message: str, *, markup: bool) -> str:
    """Convert Rich markup messages into plain text for Click."""
    if not markup:
        return message
    capture = Console(record=True)
    capture.print(message)
    return capture.export_text(clear=True).rstrip("\n")


# -- Formatters -------------------------------------------------------------------------
class JsonFormatter(json.JsonFormatter):
    """Add milliseconds and process name."""

    def add_fields(self, log_record, record, message_dict):
        """Inject extra fields used by the JSON logger."""
        super().add_fields(log_record, record, message_dict)
        log_record["msecs"] = record.msecs
        log_record["line"] = record.lineno
        log_record["module"] = record.module


json_fmt = JsonFormatter("%(levelname)s %(name)s %(message)s %(asctime)s")
# Ensure log directory exists
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
# -- Handlers attached to the QueueListener ------------------------------------------------------
# Keep ~5 MB per file, plus three backups (≈ 20 MB total)
file_handler = RotatingFileHandler(
    _LOG_FILE,
    maxBytes=2 * 1024 * 1024,  # 2 MB
    backupCount=1,
    encoding="utf-8",
)
file_handler.setFormatter(json_fmt)
file_handler.setLevel(logging.DEBUG)


# Console formatter is handled inside RichHandler
# setup rich_tracebacks
_ = install()
rich_handler = RichHandler(
    console=Console(stderr=True, log_time_format="%H:%M:%S"),
    show_level=True,
    show_time=True,
    tracebacks_word_wrap=False,
    rich_tracebacks=True,
    locals_max_length=1,
    locals_max_string=20,
    tracebacks_code_width=10,
    tracebacks_extra_lines=1,
    tracebacks_max_frames=1,
    tracebacks_show_locals=True,
    # tracebacks_suppress=["click", "typer"],
    show_path=True,
)
rich_handler.setLevel(logging.INFO)


class _RichConsoleGate(logging.Filter):
    """Allow records to suppress Rich console output via ``suppress_rich_console``."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return not getattr(record, "suppress_rich_console", False)


rich_handler.addFilter(_RichConsoleGate())

_listener = lh.QueueListener(
    _queue,
    file_handler,
    rich_handler,
    respect_handler_level=True,
)

# ---------------------------------------------------------------------------


def init_logging() -> None:
    """Initialise root logger exactly once (idempotent)."""
    if getattr(init_logging, "_configured", False):  # type: ignore[attr-defined]
        return

    queue_handler = lh.QueueHandler(_queue)

    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(message)s",  # prevents double logging info for console
        handlers=[queue_handler],
        force=True,  # override anything Typer / other libs did
    )

    # Quiet noisy libraries
    # logging.getLogger("typer").setLevel(logging.WARNING)
    # logging.getLogger("click").setLevel(logging.WARNING)

    _listener.start()
    _ = atexit.register(_listener.stop)

    init_logging._configured = True  # type: ignore[attr-defined]


logger = get_logger(__name__)

# console shared by rich progress & logging
console_ = Console()  # reuse elsewhere


@contextmanager
def log_timing(label: str):
    """Wrap calls to time blocks of code.

    Example:
        with log_timing("Epoch validation"):
            validate(...)

    """
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info("[bold cyan]%s[/] took %.3fs", label, dur)
