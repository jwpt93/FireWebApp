from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Metric:
    value: Optional[float]
    status: str


@dataclass
class Metrics:
    t_onset: Metric
    t_peak: Metric
    peak: Metric
    integral_60: Metric
    integral_120: Metric
    integral_exp: Metric
    l2_shape: Metric
    T_peak: Metric
    t_T_peak: Metric
    t_T_threshold: Metric
    m_frac_end: Metric
    t_mass_10: Metric
    t_mass_50: Metric


def _metric(value: Optional[float]) -> Metric:
    return Metric(value=value, status="OK") if value is not None else Metric(value=None, status="SKIPPED")


def _time_to_threshold(t: np.ndarray, y: np.ndarray, threshold: float) -> Optional[float]:
    idx = np.where(y <= threshold)[0]
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


def _time_to_threshold_rise(t: np.ndarray, y: np.ndarray, threshold: float) -> Optional[float]:
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


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz")
    return float(trapz_fn(y, x))


def _integral_to(t: np.ndarray, y: np.ndarray, t_end: float) -> float:
    if t_end <= t[0]:
        return 0.0
    if t_end >= t[-1]:
        return _trapz(y, t)
    y_end = np.interp(t_end, t, y)
    mask = t < t_end
    t_seg = np.concatenate([t[mask], [t_end]])
    y_seg = np.concatenate([y[mask], [y_end]])
    return _trapz(y_seg, t_seg)


def compute_metrics_rate(
    t: np.ndarray,
    rate: Optional[np.ndarray],
    Tsurf: Optional[np.ndarray],
    t_end: float,
    T_threshold: float,
) -> Metrics:
    t = np.asarray(t, dtype=float)

    t_onset = None
    t_peak = None
    peak = None
    integral_60 = None
    integral_120 = None
    integral_exp = None

    if rate is not None:
        rate = np.asarray(rate, dtype=float)
        max_rate = float(np.nanmax(rate)) if rate.size else 0.0
        if max_rate > 1e-12:
            t_onset = _time_to_threshold_rise(t, rate, 0.05 * max_rate)
            t_peak = float(t[int(np.nanargmax(rate))])
            peak = max_rate
            integral_60 = _integral_to(t, rate, min(60.0, t_end))
            integral_120 = _integral_to(t, rate, min(120.0, t_end))
            integral_exp = _integral_to(t, rate, t_end)

    T_peak = None
    t_T_peak = None
    t_T_threshold = None
    if Tsurf is not None:
        Tsurf = np.asarray(Tsurf, dtype=float)
        if Tsurf.size:
            idx = int(np.nanargmax(Tsurf))
            T_peak = float(Tsurf[idx])
            t_T_peak = float(t[idx])
            t_T_threshold = _time_to_threshold_rise(t, Tsurf, T_threshold)

    return Metrics(
        t_onset=_metric(t_onset),
        t_peak=_metric(t_peak),
        peak=_metric(peak),
        integral_60=_metric(integral_60),
        integral_120=_metric(integral_120),
        integral_exp=_metric(integral_exp),
        l2_shape=Metric(None, "SKIPPED"),
        T_peak=_metric(T_peak),
        t_T_peak=_metric(t_T_peak),
        t_T_threshold=_metric(t_T_threshold),
        m_frac_end=Metric(None, "SKIPPED"),
        t_mass_10=Metric(None, "SKIPPED"),
        t_mass_50=Metric(None, "SKIPPED"),
    )


def compute_mass_metrics(
    t: np.ndarray,
    mass_total: Optional[np.ndarray],
) -> tuple[Metric, Metric, Metric]:
    if mass_total is None:
        return Metric(None, "SKIPPED"), Metric(None, "SKIPPED"), Metric(None, "SKIPPED")

    mass = np.asarray(mass_total, dtype=float)
    valid = np.where(mass > 0)[0]
    if len(valid) == 0:
        return Metric(None, "SKIPPED"), Metric(None, "SKIPPED"), Metric(None, "SKIPPED")

    i0 = int(valid[0])
    m0 = float(mass[i0])
    if m0 <= 0:
        return Metric(None, "SKIPPED"), Metric(None, "SKIPPED"), Metric(None, "SKIPPED")

    t0 = t[i0:]
    m0_series = mass[i0:]

    m_frac_end = float(m0_series[-1] / m0)
    t_10 = _time_to_threshold(t0, m0_series, 0.9 * m0)
    t_50 = _time_to_threshold(t0, m0_series, 0.5 * m0)

    return _metric(m_frac_end), _metric(t_10), _metric(t_50)


def compute_l2_shape(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
    dt: float = 0.5,
) -> Metric:
    t_end = min(float(t_ref[-1]), float(t_other[-1]))
    grid = np.arange(0.0, t_end + dt, dt)
    ref = np.interp(grid, t_ref, y_ref)
    other = np.interp(grid, t_other, y_other)

    max_ref = float(np.nanmax(ref))
    max_other = float(np.nanmax(other))
    if max_ref <= 1e-12 or max_other <= 1e-12:
        return Metric(None, "SKIPPED")

    ref_n = ref / max_ref
    other_n = other / max_other
    denom = np.linalg.norm(ref_n) + 1e-12
    l2 = float(np.linalg.norm(other_n - ref_n) / denom)
    return Metric(l2, "OK")
