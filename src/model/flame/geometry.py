"""Flame geometry correlations for pool fires at finite bed scale.

Provides:
  heskestad_flame_height()   — mean flame height (Heskestad 1983)
  pool_fire_view_factor()    — radiation view factor: vertical flame cylinder → horizontal base
  bed_equivalent_diameter()  — fuel bed area → equivalent diameter

Physical basis
--------------
For a fuel bed of area A_bed [m²] burning at total HRR Q [kW]:
  1. Compute equivalent diameter D_eq = 2√(A_bed/π) [m]
  2. Compute mean flame height L_f via Heskestad (1983)
  3. Compute view factor F from flame geometry via Drysdale (1999) §4.3

Usage in ROM
------------
When flame.geometry_mode = "heskestad" and flame.area_m2 is set, the runner
replaces the static deck view_factor with F(t) computed from the current HRRPUA.
This makes flame feedback thermodynamically self-consistent: a stronger fire has a
taller flame, a larger view factor, and therefore stronger feedback.

References
----------
Heskestad, G. (1983). "Luminous heights of turbulent diffusion flames."
  Fire Safety Journal, 5(2), 103–108. doi:10.1016/0379-7112(83)90002-4

Drysdale, D. (1999). "An Introduction to Fire Dynamics." 2nd ed. Wiley.
  §4.3 View factors for pool fires, Eq. 4.16.
"""

from __future__ import annotations

import math


def bed_equivalent_diameter(area_m2: float) -> float:
    """Equivalent circular diameter for a fuel bed of given plan area [m].

    Converts rectangular or irregular bed to circle of equal area:
      D_eq = 2 × √(A/π)

    Parameters
    ----------
    area_m2 : float  Fuel bed plan area [m²]. Must be > 0.

    Returns
    -------
    float  Equivalent diameter D_eq [m].
    """
    return 2.0 * math.sqrt(max(area_m2, 1e-9) / math.pi)


def heskestad_flame_height(Q_kW: float, D_m: float) -> float:
    """Mean flame height via Heskestad (1983) correlation.

    L_f = max(0, −1.02 × D + 0.235 × Q^0.4)

    Valid domain: Q* > ~0.07 (Heskestad 1983 Fig. 2).  Below this, the flame
    is marginal and L_f → 0 (correctly clamped).

    Parameters
    ----------
    Q_kW : float  Total fire HRR [kW].  Clamped to ≥ 0.
    D_m  : float  Equivalent diameter of flame base [m].  Clamped to > 0.

    Returns
    -------
    float  Mean flame height L_f [m].  Always ≥ 0.
    """
    Q = max(float(Q_kW), 0.0)
    D = max(float(D_m), 1e-6)
    return max(0.0, -1.02 * D + 0.235 * (Q ** 0.4))


def pool_fire_view_factor(L_f: float, D_eq: float) -> float:
    """Radiation view factor from a vertical cylinder flame to its horizontal base.

    Models the flame as a solid vertical cylinder of height L_f and diameter D_eq.
    Configuration factor F_{flame→base}:
      tan θ = 2 L_f / D_eq
      F = 0.5 × (1 − cos θ)      (Drysdale 1999, Eq. 4.16)

    Limits:
      L_f → 0:  θ → 0,  F → 0      (no flame, no feedback)
      L_f → ∞:  θ → π/2, F → 0.5  (half-hemisphere, F_max = 0.5)

    Parameters
    ----------
    L_f   : float  Mean flame height [m].  Must be ≥ 0.
    D_eq  : float  Equivalent base diameter [m].  Must be > 0.

    Returns
    -------
    float  View factor F [-].  Range [0, 0.5].
    """
    if L_f <= 0.0:
        return 0.0
    D = max(float(D_eq), 1e-6)
    theta = math.atan(2.0 * float(L_f) / D)
    return 0.5 * (1.0 - math.cos(theta))
