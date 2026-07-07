"""Detect what kind of console environment the process is attached to."""

from __future__ import annotations

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


def supports_live_display(console: Console) -> bool:
    """Report whether Rich live displays (Progress, track) can render.

    Live displays need either a real terminal or a notebook display hook; with
    plain piped output (e.g. a ``!holod train`` shell cell in Colab) they render
    nothing at all and a printed fallback must be used instead.
    """
    return console.is_terminal or console.is_jupyter or in_notebook_kernel()
