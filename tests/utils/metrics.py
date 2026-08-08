from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def time_to_threshold(t: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return t1
    frac = (threshold - y0) / (y1 - y0)
    return t0 + frac * (t1 - t0)


def peak_time_and_value(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmax(y))
    return float(t[idx]), float(y[idx])


def normalized_L2_error(y: np.ndarray, y_ref: np.ndarray) -> float:
    denom = np.linalg.norm(y_ref) + 1e-12
    return float(np.linalg.norm(y - y_ref) / denom)


def is_monotonic(values: Iterable[float], increasing: bool = True) -> bool:
    vals = np.asarray(list(values), dtype=float)
    diffs = np.diff(vals)
    if increasing:
        return bool(np.all(diffs >= -1e-12))
    return bool(np.all(diffs <= 1e-12))
