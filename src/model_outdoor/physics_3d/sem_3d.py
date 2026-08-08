"""3D Synthetic Eddy Method (Jarrin et al. 2006) for inlet turbulence.

Provides the tent shape function used to compose y-asymmetric u/v/w
perturbations from a set of randomly placed, signed eddies that advect
through the inlet plane.  RNG is consumed at init and during
deterministic eddy-recycling only — no randomness inside parallel
kernels (CLAUDE.md Rule #17).

References:
    Jarrin N. et al. (2006) Int. J. Heat Fluid Flow 27:585-593.
"""
from __future__ import annotations

import math

import numpy as np


SQRT_1_5 = math.sqrt(1.5)


def sem_tent(r: np.ndarray | float) -> np.ndarray | float:
    """Variance-normalized tent shape function: √1.5 · max(1 − |r|, 0).

    The factor √1.5 makes ∫ f² dr = 1 over r ∈ [−1, 1], so the
    aggregated eddy field has unit-variance after scaling by
    ``1/√N · amp``.
    """
    return SQRT_1_5 * np.maximum(1.0 - np.abs(r), 0.0)
