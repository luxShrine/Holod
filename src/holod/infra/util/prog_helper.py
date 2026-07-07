"""Progress reporting that adapts to terminals, notebook kernels, and dumb consoles."""

from __future__ import annotations

import time
from collections.abc import Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from rich.text import Text

from holod.infra.log import console_ as console
from holod.infra.util.env import supports_live_display

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rich.progress import Task

    from holod.infra.dataclasses import AutoConfig, CoreTrainer

# used to help align items that are printed, allows for one central area of control
ALIGN: str = "\t  "

# seconds between status lines printed by the plain-text fallbacks
_PRINT_INTERVAL_S: float = 5.0


class RateColumn(ProgressColumn):
    """Custom class for creating rate column."""

    def render(self, task: Task) -> Text:
        """Render the speed of batch processing."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.2f} batch/s", style="progress.data")


class MetricColumn(ProgressColumn):
    """Render any numeric field kept in task.fields (e.g. 'loss', 'acc', 'lr')."""

    def __init__(self, name: str, fmt: str = "{:.4f}", style: str = "cyan"):
        """Store rendering parameters for the column."""
        super().__init__()
        self.name, self.fmt, self.style = name, fmt, style

    def render(self, task: Task):
        """Format the stored metric value for display."""
        val = task.fields.get(self.name)
        if val is None:
            return Text("–")
        return Text(self.fmt.format(val), style=self.style)


@dataclass
class _PlainTask:
    """State tracked for one PlainProgress task."""

    description: str
    total: float | None
    completed: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)
    last_print: float = 0.0


class PlainProgress:
    """Plain-text stand-in for ``rich.progress.Progress``.

    Rich live displays render nothing when stdout is neither a terminal nor a
    notebook display hook — e.g. output piped to a file, or a ``!holod train``
    shell cell in Jupyter/Colab. This implements the subset of the ``Progress``
    API the training loop uses and prints throttled status lines instead.
    """

    def __init__(self, print_interval_s: float = _PRINT_INTERVAL_S) -> None:
        """Store the minimum delay between printed status lines."""
        self._tasks: dict[int, _PlainTask] = {}
        self._next_id: int = 0
        self._interval: float = print_interval_s

    def __enter__(self) -> PlainProgress:
        """Match the context-manager interface of ``Progress``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """No live display to tear down."""

    def add_task(self, description: str, total: float | None = None, **fields: Any) -> TaskID:
        """Register a task and return its id (mirrors ``Progress.add_task``)."""
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = _PlainTask(description, total, fields=dict(fields))
        return TaskID(task_id)

    def reset(self, task_id: TaskID, *, total: float | None = None, **fields: Any) -> None:
        """Zero a task's progress, optionally updating its total and metric fields."""
        task = self._tasks[task_id]
        task.completed = 0.0
        if total is not None:
            task.total = total
        task.fields.update(fields)
        task.last_print = 0.0

    def update(
        self,
        task_id: TaskID,
        *,
        advance: float | None = None,
        completed: float | None = None,
        total: float | None = None,
        **fields: Any,
    ) -> None:
        """Advance a task and print a status line when the throttle interval elapses."""
        task = self._tasks[task_id]
        if total is not None:
            task.total = total
        if completed is not None:
            task.completed = completed
        if advance is not None:
            task.completed += advance
        task.fields.update(fields)

        finished = task.total is not None and task.completed >= task.total
        now = time.monotonic()
        if finished or now - task.last_print >= self._interval:
            task.last_print = now
            # flush so lines appear promptly through notebook shell pipes
            print(_format_status(task), flush=True)


def _format_status(task: _PlainTask) -> str:
    """Render one plain status line for a task."""
    if task.total:
        head = f"{task.description} {task.completed:g}/{task.total:g} "
        head += f"({task.completed / task.total:.0%})"
    else:
        head = f"{task.description} {task.completed:g}"
    parts = [head]
    for key, val in task.fields.items():
        parts.append(f"{key} {val:.4g}" if isinstance(val, int | float) else f"{key} {val}")
    return " | ".join(parts)


type ProgressLike = Progress | PlainProgress


def _live_progress(jupyter: bool) -> Progress:
    """Build the Rich progress bar; notebooks get a calmer refresh and no spinner."""
    columns: list[ProgressColumn | str] = [
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•Time Taken",  # Separator
        TimeElapsedColumn(),
        "•Time Remaining",
        TimeRemainingColumn(),
        "•",
        RateColumn(),
        "•Train Loss",
        MetricColumn("train_loss", fmt="{:.4f}", style="magenta"),
        "•Val Loss",
        MetricColumn("val_loss", fmt="{:.4f}", style="yellow"),
        "•LR",
        MetricColumn("lr", fmt="{:.1e}", style="dim cyan"),  # Shorter LR format
    ]
    if not jupyter:
        # the spinner only shows liveness in a terminal; in a notebook it forces
        # constant repaints of the whole display
        columns.append(SpinnerColumn())
    return Progress(
        *columns,
        console=console,
        # notebook cells repaint the entire display each refresh; keep that cheap
        refresh_per_second=2 if jupyter else 10,
        transient=False,  # Keep finished tasks visible
    )


def track_progress[T](
    sequence: Iterable[T], description: str = "Working...", total: float | None = None
) -> Iterator[T]:
    """Iterate with a progress display, printing plain lines when live rendering can't work."""
    if supports_live_display(console):
        yield from track(sequence, description=description, total=total, console=console)
        return

    if total is None and isinstance(sequence, Sized):
        total = len(sequence)
    last_print = time.monotonic()
    for count, item in enumerate(sequence, start=1):
        yield item
        now = time.monotonic()
        if (total is not None and count >= total) or now - last_print >= _PRINT_INTERVAL_S:
            last_print = now
            tail = f"/{total:g} ({count / total:.0%})" if total else ""
            print(f"{description} {count}{tail}", flush=True)


def setup_training_progress(
    a_cfg: AutoConfig,
    train_loss: float,
    val_loss: float,
    core_trainer: CoreTrainer,
    device_name: str,
) -> tuple[ProgressLike, TaskID, TaskID, TaskID]:
    """Create a progress reporter for training monitoring suited to the environment."""
    train_loss_start: float = train_loss
    val_loss_start: float = val_loss

    progress_bar: ProgressLike
    if supports_live_display(console):
        progress_bar = _live_progress(console.is_jupyter)
    else:
        progress_bar = PlainProgress()

    # progress
    epoch_task = progress_bar.add_task(
        "Epoch",
        total=a_cfg.epoch_count,
        train_loss=train_loss_start,
        val_loss=val_loss_start,
        accuracy_measure=0,
        lr=float(core_trainer.optimizer.param_groups[0]["lr"]),
    )
    train_task = progress_bar.add_task(
        "Train", total=len(core_trainer.train_loader), avg_loss=train_loss_start
    )
    val_task = progress_bar.add_task(
        "Evaluation", total=len(core_trainer.val_loader), avg_loss=val_loss_start
    )

    console.print(
        f"{ALIGN}Using: [bold green]{a_cfg.device()}[/]",
    )
    if a_cfg.device() == "cuda":
        console.print(
            f"{ALIGN}Cuda Device is: [bold green]{device_name}[/]",
        )

    return progress_bar, train_task, val_task, epoch_task
