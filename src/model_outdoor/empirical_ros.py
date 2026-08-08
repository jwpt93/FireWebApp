"""Empirical rate-of-spread (ROS) models for the level-set forcing.

PURPOSE
-------
At very low wind (U < ~1.4 m/s for grass), the resolved-physics propagation
mechanism in this code (RANS k-ε + EDC + bed coupling) cannot bridge the
source-patch to downstream cells.  Per Cheney 1998 IJWF 8:1 §3.2, fires in
that regime do propagate empirically but in an unsteady manner mediated by
intermittent flame tongues (Finney 2015 PNAS 112:9833-9838).  Our mean-field
closures structurally cannot represent that variance — see
Phase 18 bug-sweep memory `phase18_bug_sweep_lowU_null.md`.

The pattern used by every published production wildland-fire simulator
(WRF-Fire, CAWFE, PHOENIX, FARSITE, BehavePlus) at low U is to bypass
resolved combustion entirely and impose an empirical rate-of-spread at
the front.  This module provides the same option for our code, toggleable
via deck.

Three empirical models are implemented:

  * "cheney_eq6"       — Cheney 1998 Eq. 6 power-law (the fit our Cheney 1993
                         validation cases use as reference).  Inputs: a_ch
                         (fuel-dependent coefficient), U, moisture-fraction.
  * "marsden_smedley"  — Marsden-Smedley & Catchpole 1995 IJWF 5(4):215
                         Tasmanian buttongrass moorland regression.  Inputs:
                         U at 1.7 m, dead-fuel moisture (%), stand age (yr).
                         Wind exponent 1.312 (not 1.0), and includes an
                         age-asymptote fuel-build-up factor.  See Phase 22.
  * "rothermel"        — Rothermel 1972 (stub; not yet implemented).

USAGE (deck)
------------
  outdoor.empirical_ros_enable          = true
  outdoor.empirical_ros_model           = cheney_eq6
  outdoor.empirical_ros_a_ch            = 0.406       # Cheney Nat grass
                                                       # (Cut uses 0.343)
  outdoor.empirical_ros_u_threshold_m_s = 1.4         # Cheney 1998
                                                       # quasi-steady cutoff
  outdoor.empirical_ros_blend_width_m_s = 0.5         # smooth blend window

BLENDING RULE
-------------
At wind speed U:
  * U <= threshold - blend_width        : empirical only (resolved disabled)
  * threshold - blend_width < U < thresh: linear blend resolved/empirical
  * U >= threshold                       : resolved only

This preserves the validated high-U production behaviour while substituting
empirical propagation in the regime where resolved physics cannot.

REFERENCES
----------
  Cheney, N.P., Gould, J.S., Catchpole, W.R. (1998) "Prediction of fire
    spread in grasslands," Int. J. Wildland Fire 8:1-13.  Eq. 6 power-law.
  Finney, M.A. et al. (2015) "Role of buoyant flame dynamics in wildfire
    spread," PNAS 112:9833-9838.  Identifies the missing variance mechanism.
  Mandel, J., Beezley, J.D., Kochanski, A.K. (2011) "Coupled atmosphere-
    wildland fire modelling with WRF 3.3 and SFIRE 2011," Geosci. Model
    Dev. 4:591-610.  Reference implementation of empirical-ROS-at-front
    pattern in WRF-Fire.
  Rothermel, R.C. (1972) "A mathematical model for predicting fire spread
    in wildland fuels," USDA Forest Service Research Paper INT-115.
"""

from __future__ import annotations

import math


# Cheney 1998 Eq. 6 default exponents (calibrated against Annaburroo grass).
CHENEY_EQ6_U_EXP    = 0.987
CHENEY_EQ6_B_MF     = 0.0707
CHENEY_EQ6_U2_RATIO = 0.723   # U_2 = 0.723 · U_10 (Cheney 1993 convention)


# Marsden-Smedley & Catchpole 1995 IJWF 5(4):215 buttongrass moorland fit.
# Regression fit on 44 experimental Tasmanian fires, 0.25-1.0 ha plots.
# Wind measurement at 1.7 m above ground; regression uses U in km/h.
# Reproduced in Tasmania Parks planned-burning guidelines 2009 Appendix 2.
MS_1995_CONST       = 0.678 / 60.0    # (m/s per (km/h)^1.312 unit-of-U-M-age)
MS_1995_U_EXP       = 1.312
MS_1995_B_MF        = 0.0243
MS_1995_AGE_LAMBDA  = 0.116           # age-asymptote build-up rate (1/yr)


def cheney_eq6_ros_m_per_s(
    U_m_s: float,
    moisture_frac: float,
    a_ch: float,
) -> float:
    """Cheney 1998 Eq. 6 grass-fire rate of spread.

    Parameters
    ----------
    U_m_s : float
        Mean wind speed at standard reference height (m/s).
    moisture_frac : float
        Fuel moisture content as a mass fraction (e.g. 0.04 = 4%).
    a_ch : float
        Fuel-dependent coefficient.  Cheney 1998 reports:
          * Nat (natural / undisturbed grass): a_ch = 0.406
          * Cut (cut / mown grass):            a_ch = 0.343

    Returns
    -------
    ros_m_per_s : float
        Predicted rate of spread (m/s), guaranteed non-negative.
    """
    if U_m_s <= 0.0:
        return 0.0
    mf_pct = moisture_frac * 100.0
    u2 = max(0.0, CHENEY_EQ6_U2_RATIO * U_m_s)
    ros_m_per_min = a_ch * (u2 ** CHENEY_EQ6_U_EXP) * \
                    math.exp(-CHENEY_EQ6_B_MF * mf_pct) * 60.0
    return max(0.0, ros_m_per_min / 60.0)


def marsden_smedley_ros_m_per_s(
    U_1p7_m_s: float,
    moisture_frac: float,
    age_yr: float,
) -> float:
    """Marsden-Smedley & Catchpole 1995 buttongrass moorland ROS regression.

    Parameters
    ----------
    U_1p7_m_s : float
        Mean wind speed at 1.7 m above ground (m/s).  The M-S dataset was
        collected with sensors at 1.7 m; the regression is calibrated at
        this reference height, not the 10 m of Cheney.  Convert with a
        log-law if you have U_10 (typical over moorland with z0~0.03 m:
        U_1.7 = U_10 / 1.44).
    moisture_frac : float
        Dead-fuel moisture content as a mass fraction (e.g. 0.30 = 30%).
        Marsden-Smedley calibrated with dead-fuel moisture only; live
        buttongrass leaves are treated as a moisture sink, not a fuel.
    age_yr : float
        Stand age since last fire (years).  Fuel-build-up factor
        `(1 - exp(-0.116·age))` — asymptotes near 1.0 at age ≥ 40 yr.

    Returns
    -------
    ros_m_per_s : float
        Predicted head-fire rate of spread (m/s), non-negative.
    """
    if U_1p7_m_s <= 0.0 or age_yr <= 0.0:
        return 0.0
    U_kmh = 3.6 * U_1p7_m_s
    mf_pct = moisture_frac * 100.0
    ros_m_s = (
        MS_1995_CONST
        * (U_kmh ** MS_1995_U_EXP)
        * math.exp(-MS_1995_B_MF * mf_pct)
        * (1.0 - math.exp(-MS_1995_AGE_LAMBDA * age_yr))
    )
    return max(0.0, ros_m_s)


def marsden_smedley_p_sustain(
    U_1p7_m_s: float,
    moisture_frac: float,
    productivity: int = 1,
) -> float:
    """Marsden-Smedley 2001 IJWF 10(2):255 Part IV sustaining logistic.

    Returns probability that a buttongrass moorland fire will sustain
    propagation given wind, moisture, and site productivity.  Used as a
    validation target for extinction physics, NOT as a model driver.

    productivity : int
        1 = low-productivity (quartzite substrates),
        2 = medium-productivity (dolerite/limestone/glacial till).
    """
    U_kmh = 3.6 * U_1p7_m_s
    mf_pct = moisture_frac * 100.0
    a = (-1.0
         + 0.68 * U_kmh
         - 0.07 * mf_pct
         - 0.0037 * U_kmh * mf_pct
         + 2.1 * productivity)
    return 1.0 / (1.0 + math.exp(-a))


def rothermel_ros_m_per_s(*args, **kwargs) -> float:
    """Rothermel 1972 ROS (NOT IMPLEMENTED yet).

    Rothermel needs a full fuel-model parameter set (heat content, SAV,
    moisture of extinction, packing ratio, etc.).  Implement on demand
    when non-grass suppression-validation targets land.
    """
    raise NotImplementedError(
        "Rothermel 1972 not yet implemented.  Use 'cheney_eq6' for "
        "grass cases.  Open an issue if you need Rothermel."
    )


def blend_resolved_empirical(
    U_m_s: float,
    u_threshold_m_s: float,
    blend_width_m_s: float,
) -> float:
    """Return the weight applied to the EMPIRICAL ROS in the blended v_n.

    Returns 1.0 (empirical only) below threshold-blend_width.
    Returns 0.0 (resolved only)  at-or-above threshold.
    Linear ramp in the [threshold-blend_width, threshold] window.

    blend_width_m_s = 0.0 → hard step at threshold.
    """
    if U_m_s >= u_threshold_m_s:
        return 0.0
    if blend_width_m_s <= 0.0:
        return 1.0
    u_lo = u_threshold_m_s - blend_width_m_s
    if U_m_s <= u_lo:
        return 1.0
    # Linear blend: w_emp = (threshold - U) / blend_width
    return (u_threshold_m_s - U_m_s) / blend_width_m_s


# ── FuelModel class-based interface (Phase 22.5) ─────────────────────
# Purpose: a registry-based extension point so adding a new empirical
# model is one dataclass + one @register decorator, and every caller
# gets a consistent .ros() / .ros_samples() / .p_sustain() API.
#
# Also positions us for later Bayesian hierarchical model (BHM) work:
# .ros_samples(n) currently returns n identical delta-function samples,
# but a BHM-fitted subclass would return n posterior draws that
# propagate uncertainty downstream (e.g. into a Monte-Carlo v_n on the
# level-set).  See the "generalization" discussion in the 2026-07 Phase 22
# transcript for the full BHM plan.

from typing import ClassVar


class FuelModel:
    """Base class for empirical rate-of-spread models.

    Concrete subclasses declare ``name`` (registry key) and ``schema``
    (tuple of kwarg names they consume beyond ``U_m_s``), and implement
    ``ros(U_m_s, **kwargs)``.  Optional: override ``p_sustain`` to expose
    an extinction predictor, or ``ros_samples`` to expose posterior
    samples (default: n copies of the deterministic ``ros`` value).
    """
    name: ClassVar[str] = ""
    schema: ClassVar[tuple] = ()

    def ros(self, U_m_s: float, **kwargs) -> float:
        raise NotImplementedError

    def ros_samples(self, U_m_s: float, n: int = 1, **kwargs):
        """Return n samples of ROS in m/s.  Default: delta-function.

        BHM-fitted subclasses would override to return n draws from the
        posterior predictive distribution.
        """
        val = self.ros(U_m_s, **kwargs)
        return [val] * n

    def p_sustain(self, U_m_s: float, **kwargs) -> float:
        """Probability of sustained propagation given conditions.

        Default 1.0 (model assumes fire always sustains).  Override
        when the empirical family includes an extinction predictor
        (e.g. Marsden-Smedley 2001 logistic).
        """
        return 1.0


_REGISTRY: dict = {}


def register(cls):
    """Decorator: register a FuelModel subclass into the global registry."""
    _REGISTRY[cls.name] = cls()
    return cls


def get_model(name: str) -> FuelModel:
    """Look up a FuelModel by name.  Raises ValueError on unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown fuel model '{name}'.  Registered: {list_models()}"
        )
    return _REGISTRY[name]


def list_models() -> list:
    """Sorted list of registered fuel-model names."""
    return sorted(_REGISTRY.keys())


@register
class CheneyEq6(FuelModel):
    """Cheney 1998 Eq.6 grass-fire power law (Annaburroo NT calibration)."""
    name = "cheney_eq6"
    schema = ("moisture_frac", "a_ch")

    def ros(self, U_m_s: float, moisture_frac: float, a_ch: float,
            **_ignored) -> float:
        return cheney_eq6_ros_m_per_s(U_m_s, moisture_frac, a_ch)


@register
class MarsdenSmedley(FuelModel):
    """Marsden-Smedley & Catchpole 1995 buttongrass moorland (Tasmania)."""
    name = "marsden_smedley"
    schema = ("moisture_frac", "age_yr")

    def ros(self, U_m_s: float, moisture_frac: float,
            age_yr: float = 10.0, **_ignored) -> float:
        return marsden_smedley_ros_m_per_s(U_m_s, moisture_frac, age_yr)

    def p_sustain(self, U_m_s: float, moisture_frac: float,
                  productivity: int = 1, **_ignored) -> float:
        return marsden_smedley_p_sustain(U_m_s, moisture_frac, productivity)


@register
class RothermelStub(FuelModel):
    """Rothermel 1972 (not yet implemented — raises NotImplementedError)."""
    name = "rothermel"
    schema = ()

    def ros(self, U_m_s: float, **_ignored) -> float:
        return rothermel_ros_m_per_s()


def evaluate_empirical_ros(
    model: str,
    U_m_s: float,
    moisture_frac: float = None,
    a_ch: float = None,
    **kwargs,
) -> float:
    """Dispatch to the requested empirical-ROS model (backward-compat shim).

    New code should prefer ``get_model(name).ros(U, **kwargs)`` directly.
    This wrapper preserves the pre-registry positional-arg call sites
    (spread_3d.py, existing scripts, unit tests) so the class refactor
    is behaviour-preserving.

    Extra model-specific kwargs (e.g. ``age_yr`` for buttongrass) pass
    through via ``kwargs``; each model's ``ros()`` ignores kwargs it
    doesn't consume.

    Raises ValueError on unknown model.
    """
    if moisture_frac is not None:
        kwargs["moisture_frac"] = moisture_frac
    if a_ch is not None:
        kwargs["a_ch"] = a_ch
    return get_model(model).ros(U_m_s, **kwargs)
