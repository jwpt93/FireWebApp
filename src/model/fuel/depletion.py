from __future__ import annotations

import numpy as np


def apply_depletion(
    t: np.ndarray,
    m_py_raw: np.ndarray,
    m0_kg_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a simple fuel depletion model to a raw pyrolysis flux.

    Args:
        t: Time array [s].
        m_py_raw: Raw pyrolysis flux [kg/m^2/s].
        m0_kg_m2: Initial dry fuel mass per area [kg/m^2].

    Returns:
        (m_py, m_remaining) where:
          m_py is the depleted pyrolysis flux [kg/m^2/s]
          m_remaining is remaining dry fuel mass per area [kg/m^2]
    """

    if m0_kg_m2 <= 0.0:
        return m_py_raw, np.zeros_like(m_py_raw)

    m_py = np.zeros_like(m_py_raw, dtype=float)
    m_remaining = np.zeros_like(m_py_raw, dtype=float)

    remaining = float(m0_kg_m2)
    m_remaining[0] = remaining
    for i in range(len(t)):
        if i > 0:
            dt = float(t[i] - t[i - 1])
        else:
            dt = 0.0
        frac = 0.0 if m0_kg_m2 <= 0 else max(remaining / m0_kg_m2, 0.0)
        m_py[i] = float(m_py_raw[i]) * frac
        remaining = max(0.0, remaining - m_py[i] * dt)
        m_remaining[i] = remaining

    return m_py, m_remaining
