from __future__ import annotations

from ..config.schemas import Thresholds


def should_ignite(m_py: float, T1: float, thresholds: Thresholds) -> bool:
    """Return True if flame should ignite."""

    return m_py > thresholds.m_py_ignite or T1 > thresholds.T_ignite


def should_extinguish(m_py: float, T1: float, thresholds: Thresholds) -> bool:
    """Return True if flame should extinguish."""

    return m_py < thresholds.m_py_crit or T1 < thresholds.T_py
