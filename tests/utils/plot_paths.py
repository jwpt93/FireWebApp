from __future__ import annotations

from pathlib import Path


def get_test_plots_dir() -> Path:
    """Return the canonical flat output directory for test plots."""
    base = Path(__file__).resolve().parents[2] / "test_plots"
    base.mkdir(parents=True, exist_ok=True)
    return base
