"""holod: train, evaluate, and visualize autofocus-depth models for digital holograms."""

from __future__ import annotations

import importlib.util
import os


def _sanitize_mpl_backend() -> None:
    """Drop an ``MPLBACKEND`` pointing at a module this environment cannot import.

    Notebook hosts (Colab, Jupyter) export ``MPLBACKEND=module://matplotlib_inline.
    backend_inline`` to every subprocess. Inside this project's venv that module does
    not exist, and ``import matplotlib`` raises ``ValueError: Key backend: ... is not
    a valid value for backend`` while validating it. Unsetting the variable lets
    matplotlib fall back to its normal backend auto-detection (Agg when headless).
    """
    backend = os.environ.get("MPLBACKEND", "")
    if backend.startswith("module://"):
        root_module = backend.removeprefix("module://").partition(".")[0]
        if importlib.util.find_spec(root_module) is None:
            del os.environ["MPLBACKEND"]


_sanitize_mpl_backend()
