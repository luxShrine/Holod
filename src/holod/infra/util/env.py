"""Detect what kind of console environment the process is attached to."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


def in_notebook_kernel() -> bool:
    """Return True when running inside a Jupyter/IPython kernel (incl. Google Colab)."""
    if "google.colab" in sys.modules:
        return True
    # a kernel always has IPython imported already; don't import it ourselves
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        return False
    shell = ipython_module.get_ipython()
    # ZMQInteractiveShell is Jupyter; Colab names its kernel shell "Shell"
    return shell is not None and type(shell).__name__ in {"ZMQInteractiveShell", "Shell"}


def in_colab_shell_cell() -> bool:
    """Return True when running as a subprocess of a Colab ``!``/``%%shell`` cell.

    Colab runs shell cells through a pseudo-terminal, so ``isatty()`` is True and
    Rich picks its terminal-mode live display — but the cell's output pane does
    not honour the cursor-movement codes ``Live`` repaints with, so each refresh
    is appended as a new frame. Colab's env vars are inherited by the subprocess
    while the kernel modules are not, which distinguishes this case from both the
    kernel itself and a real terminal.
    """
    return "COLAB_RELEASE_TAG" in os.environ and not in_notebook_kernel()


def supports_live_display(console: Console) -> bool:
    """Report whether Rich live displays (Progress, track) can render.

    Live displays need either a real terminal or a notebook display hook. With
    plain piped output they render nothing at all, and under a Colab shell-cell
    PTY they stack one frame per refresh; both need the printed fallback instead.
    """
    if in_colab_shell_cell():
        return False
    return console.is_terminal or console.is_jupyter or in_notebook_kernel()
