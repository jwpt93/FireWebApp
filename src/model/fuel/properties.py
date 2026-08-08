from __future__ import annotations

import math


def clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""

    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def safe_exp(x: float, max_exp: float = 50.0) -> float:
    """Exponent with clamp to avoid overflow."""

    return math.exp(min(x, max_exp))
