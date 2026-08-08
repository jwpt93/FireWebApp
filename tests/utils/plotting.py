from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def _safe_series(series: Iterable[Tuple]):
    out = []
    for item in series:
        if len(item) == 3:
            label, t, y = item
            style = {}
        else:
            label, t, y, style = item
        if y is None:
            continue
        if len(t) == 0 or len(y) == 0:
            continue
        out.append((label, np.asarray(t, dtype=float), np.asarray(y, dtype=float), style or {}))
    return out


def plot_rate_temp(
    out_path: Path,
    title: str,
    rate_label: str,
    rate_series: Iterable[Tuple],
    temp_series: Iterable[Tuple],
    mlr_series: Optional[Iterable[Tuple]] = None,
    mlr_label: str = "MLR [kg/m2/s]",
) -> bool:
    """Plot rate and temperature time series. Returns True if plotted."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    rate_series = _safe_series(rate_series)
    temp_series = _safe_series(temp_series)
    mlr_series = _safe_series(mlr_series or [])
    if not rate_series and not temp_series and not mlr_series:
        return False

    panels = []
    if rate_series:
        panels.append((rate_label, rate_series))
    if mlr_series:
        panels.append((mlr_label, mlr_series))
    if temp_series:
        panels.append(("Tsurf [K]", temp_series))

    nrows = len(panels)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, max(4, int(3 * nrows))), sharex=True)
    if nrows == 1:
        axes = [axes]

    for idx, ((ylabel, series), ax) in enumerate(zip(panels, axes)):
        for label, t, y, style in series:
            ax.plot(t, y, label=label, **style)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        if idx == nrows - 1:
            ax.set_xlabel("Time [s]")

    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_xy(
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x: np.ndarray,
    y: np.ndarray,
) -> bool:
    """Plot Y vs X with a 1:1 reference line."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return False
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        return False

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, "o", ms=2, alpha=0.6)
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [lo, hi], "--", lw=1, color="gray")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True
