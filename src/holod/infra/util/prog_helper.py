from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from holod.infra.dataclasses import AutoConfig, Checkpoint, CoreTrainer
from holod.infra.log import console_ as console

# used to help align items that are printed, allows for one central area of control
ALIGN: str = "\t  "


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


def setup_training_progress(
    a_cfg: AutoConfig,
    train_loss: float,
    val_loss: float,
    core_trainer: CoreTrainer,
    device_name: str,
) -> tuple[Progress, TaskID, TaskID, TaskID]:
    """Create and configure a Rich Progress bar for training monitoring."""
    train_loss_start: float = train_loss
    val_loss_start: float = val_loss

    progress_bar = Progress(
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
        SpinnerColumn(),
        transient=False,  # Keep finished tasks visible
    )

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
