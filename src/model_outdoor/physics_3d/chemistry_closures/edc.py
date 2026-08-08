"""Magnussen 1981 EDC (Eddy-Dissipation Concept) closure.

Cell-averaged rate ω_cell = γ*·ρ·min(Y_F, Y_O2/s)/τ* where
    γ* = (C_γ · (ν·ε/k²)^(1/4))^3      fine-structure volume fraction
    τ* = C_τ · (ν/ε)^(1/2)             fine-structure time scale

The fine structure is assumed to be at flame T (chemistry is fast
inside it) so reaction proceeds at the mixing-into-fine-structure rate
regardless of cell-averaged T.  Textbook fix for our coarse-grid
(10 cm vs ~mm flame) closure mismatch.

Lit refs:
- Magnussen, B.F. (1981) "On the structure of turbulence and a generalized
  eddy dissipation concept for chemical reaction in turbulent flow"
  AIAA-81-0042, 19th AIAA Aerospace Sci. Meeting
- Magnussen, B.F. (1989) "The eddy dissipation concept" XI Task Leaders
  Meeting Energy Conservation in Combustion, IEA
- Gran, I.R. & Magnussen, B.F. (1996) Combust. Sci. Tech. 119:191
- Implemented in OpenFOAM (EDM, EDMArrhenius), ANSYS Fluent (EDC)

Phase 15-0: moved from combustion_3d.py into the chemistry_closures
registry framework.  Numerics unchanged (Rule #17 verified by
tests/outdoor/test_chemistry_closure_registry.py).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._constants import S_STOICH, HOC_J


# Magnussen 1989 fine-structure constants.
C_GAMMA_EDC = 2.1377   # fine-structure volume-fraction constant
C_TAU_EDC   = 0.4083   # fine-structure time-scale constant
NU_GAS_EDC  = 1.5e-5   # [m²/s] kinematic viscosity, gas at ~ambient

K_TURB_FLOOR_EDC   = 1.0e-4   # [m²/s²] floor for k (avoid /0)
EPS_TURB_FLOOR_EDC = 1.0e-6   # [m²/s³] floor for ε

# Water-vapor specific heat for composition-dependent cp_mix
# (Phase 16 2026-06-18; NIST tables at 1500 K).
CP_VAPOR_EDC = 2000.0   # [J/kg/K]

# ── Extinction-threshold physics (Phase 16, 2026-06-18) ──────────────
# Three orthogonal extinction mechanisms.
#
# WARNING — natural-fire moisture: enabling A+B+C in the chemistry
# kernel WITHOUT a coupled fast T_g cooling pathway produces an
# ARTIFACT.  Per-cell omega quench leaves the cell's stored thermal
# energy ρ·cp·T_g intact (cooling timescale via radiation/diffusion is
# ~1 s, vs front-passage time ~100 ms).  The "extinguished" cell still
# radiates at T_g⁴ to neighbors so the front skips marginal cells and
# concentrates in robust-combustion cells → ROS becomes FLAT vs
# moisture content (verified: Catchpole M=5/10/20/30% gave 30.95 /
# 31.26 / 31.78 / 32.01 m/min with ext ON, vs proper 29.61 / 26.93 /
# 20.35 / 20.50 with ext OFF).  See memory note
# phase16_moisture_sensitivity_water_mass_limit.md.
#
# USE CASE — Rule #0 suppression validation: the mechanisms ARE
# physically correct for DIRECT suppressant injection (water mist,
# CO2, foam, N2).  When a suppressant injects mass into a cell,
# Y_H2O / Y_inert spike instantly AND the vapor sensible-heat debit
# drops T_g rapidly via the existing spread_3d Q_vapor_debit term.
# The two effects together engage mechanisms B (inert dilution) and
# C (cold-flame floor) WITH the cooling that makes them physically
# consistent.  Default OFF; opt in via edc_extinction_enable=True.
#
# (A) Heat-loss balance: combustion quenches when Q_rate < safety_factor
#     × Q_radiative_loss locally.  Linn 2002 FIRETEC §3.2 implements an
#     equivalent local energy-balance gate.
#
# (B) Inert-mass-fraction suppression: combustion rate ramps to zero
#     when Y_inert (= 1 − Y_F − Y_O2 − Y_H2O for our species set; we
#     compute (1 − Y_F − Y_O2) as the upper bound here since Y_H2O is
#     not passed through this kernel) exceeds Y_INERT_CRIT.  Captures
#     the FAST suppression dynamics of water mist, CO2, foam, N2
#     dilution — the rate-limiting Rule #0 mechanism.  Beyler 1992
#     "Major species production by diffusion flames in a two-layer
#     compartment fire environment" Fire Saf. J. 19:67-94 + Drysdale
#     §3.5: hydrocarbon-air flames quench when inert fraction exceeds
#     ~0.88 (i.e., < 12% O2 + fuel combined).
#
# (C) Cold-flame floor: omega forced to zero when T_g < T_IGNITION_MIN.
#     Westbrook & Dryer 1981 / Drysdale §3.4: hydrocarbon-air flame
#     ignition T ≈ 1100-1300 K; we use 1200 K as the midrange.
#
# All three are user-deck-configurable.  Defaults are literature values.
EXTINCTION_F_SAFETY  = 1.5      # [-] heat-loss safety factor (Linn 2002)
Y_INERT_CRIT         = 0.88     # [-] hydrocarbon-flame extinction limit
                                # (Beyler 1992; equivalent to limiting O2
                                # ≈ 12% in dry-air baseline)
T_IGNITION_MIN       = 1200.0   # [K] cellulose/CH4-class ignition floor
                                # (Westbrook & Dryer 1981; Drysdale §3.4)
# Stefan-Boltzmann constant (used in heat-loss-balance gate, mech A).
SIGMA_SB_EDC         = 5.67e-8  # [W/m²/K⁴]

# ── Phase 17a (2026-06-20) Y_H2O direct omega suppression ────────────
# Beyler 1992 "Major species production by diffusion flames in a two-layer
# compartment fire environment" Fire Saf. J. 19:67-94 — hydrocarbon flames
# extinguish when water vapor mass fraction exceeds the saturation/
# extinction limit.  Lit-bracketed Y_H2O_QUENCH ≈ 0.15-0.22 for grass-
# cellulose volatiles at typical fire conditions.  Reduces omega as
# Y_H2O grows; reaches zero at the lit-bracketed quench limit.  ALWAYS
# ON (not gated by extinction_enable) — captures the direct extinction
# mechanism that field-density Y_F·Y_O2 dilution alone misses.
Y_H2O_QUENCH         = 0.18    # [-] water-vapor mass fraction at which
                                # combustion fully quenches.  Beyler 1992
                                # Fig 6 (extinction limits for various
                                # diluents); ~12% O2 + Y_H2O → quench.

# ── Phase 16 (2026-06-18) extinction-coupled wet-bulb cooling ───────
# Closes the per-cell-quench artifact: when extinction fires in a cell,
# the gas is also cooled toward the local wet-bulb temperature (≈ T_BOIL
# when Y_H2O is high).  This propagates the extinction via the existing
# gas-energy diffusion + radiation mechanisms so the cascade can develop
# at the actual front-passage timescale (not the 1-second diffusion
# timescale that masks single-cell extinction).
#
# Lit hook: Drysdale 2011 §3.5 vapor cooling at flame extinction;
# ASHRAE 2017 Fundamentals Ch.1 psychrometric wet-bulb cooling.  The
# T_wb pin is the standard limit for water-saturated gas:
#   T_wb ≈ T_BOIL_WATER when Y_H2O > Y_H2O_SAT_THRESH
# Below saturation we use T_wb_approx as a linear interpolant which
# avoids requiring a full psychrometric solver in the hot loop.
T_BOIL_WATER_EDC      = 373.15   # [K] water saturation T at 1 atm
Y_H2O_SAT_THRESH      = 0.10     # [-] above this, gas approaches T_wb
                                 # rapidly; below, slower relax


@njit(cache=True, parallel=True)
def step_chemistry_ode_edc(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]  updated in place
    Y_fuel: np.ndarray,        # (Nz, Ny, Nx) [-]  updated in place
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-]  updated in place
    k_turb: np.ndarray,        # (Nz, Ny, Nx) [m²/s²] TKE
    eps_turb: np.ndarray,      # (Nz, Ny, Nx) [m²/s³] dissipation
    chi_rad: float,
    cp_g: float,               # [J/kg/K] dry-air baseline (1100); the
                               # effective cp_mix is computed per cell
                               # from Y_H2O when provided.
    dt: float,                 # [s] outer chemistry sub-step
    n_substeps: int,           # internal sub-cycle (≥1)
    omega_int_out: np.ndarray, # (Nz, Ny, Nx) [kg/m³/s] time-averaged ω over dt
    Y_H2O: np.ndarray,         # (Nz, Ny, Nx) [-] water-vapor mass fraction;
                               # pass zeros to get legacy constant-cp behavior.
                               # cp_mix = (1-Y_H2O)·cp_g + Y_H2O·CP_VAPOR_EDC
                               # is used in the T_g update (Phase 16 2026-06-18).
    extinction_enable: bool = False,  # Phase 16: opt-in extinction
                                       # thresholds A+B+C (default OFF
                                       # preserves backward-compat with
                                       # 0D startup tests).  Required ON
                                       # for natural-fire moisture and
                                       # Rule #0 suppression validation.
    # Phase 23: chemistry-family scalars (biomass defaults preserve
    # bit-exact behaviour for pre-Phase-23 callers per Rule #17).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
) -> None:
    """Magnussen 1981 EDC closure (no Arrhenius / cell-T gate, no bootstrap).

    Cell-averaged rate ω_cell = γ*·ρ·min(Y_F, Y_O2/s)/τ* where
        γ* = (C_γ · (ν·ε/k²)^(1/4))^3      fine-structure volume fraction
        τ* = C_τ · (ν/ε)^(1/2)             fine-structure time scale

    The fine structure is assumed to be at flame T (chemistry is fast
    inside it) so reaction proceeds at the mixing-into-fine-structure rate
    regardless of cell-averaged T.  This is the textbook fix for our
    coarse-grid (10 cm vs ~mm flame) closure mismatch.

    Same ODE form as step_chemistry_ode but with EDC ω in place of the
    4-rate min closure.  Explicit Euler is sufficient (EDC rate isn't
    stiff in T).
    """
    Nz, Ny, Nx = rho.shape
    h = dt / max(n_substeps, 1)
    HoC_eff = hoc_J * (1.0 - chi_rad)

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Yf  = Y_fuel[k, j, i]
                YO2 = Y_O2[k, j, i]
                Tg  = T_g[k, j, i]
                rho_i = rho[k, j, i]
                k_t = k_turb[k, j, i]
                if k_t < K_TURB_FLOOR_EDC:
                    k_t = K_TURB_FLOOR_EDC
                e_t = eps_turb[k, j, i]
                if e_t < EPS_TURB_FLOOR_EDC:
                    e_t = EPS_TURB_FLOOR_EDC

                # Magnussen fine-structure params (constant within step
                # since k, ε are frozen — operator-split with k-ε solver
                # outside the chemistry kernel).
                ratio = NU_GAS_EDC * e_t / (k_t * k_t)
                if ratio < 1e-30:
                    ratio = 1e-30
                gamma_star = (C_GAMMA_EDC * ratio ** 0.25) ** 3
                if gamma_star > 1.0:
                    gamma_star = 1.0     # clamp, can't exceed 1
                tau_star = C_TAU_EDC * (NU_GAS_EDC / e_t) ** 0.5

                omega_acc = 0.0
                for _ in range(max(n_substeps, 1)):
                    if Yf <= 1e-9 or YO2 <= 1e-9:
                        omega = 0.0
                    else:
                        Y_lim = Yf if Yf < (YO2 / s_stoich) else (YO2 / s_stoich)
                        omega_fine = rho_i * Y_lim / tau_star    # [kg/m³/s]
                        omega = gamma_star * omega_fine
                        # ── Phase 17a: Y_H2O direct omega suppression ──
                        # Beyler 1992 hydrocarbon-flame extinction by water
                        # vapor.  Reduces omega linearly from baseline at
                        # Y_H2O=0 to zero at Y_H2O=Y_H2O_QUENCH.  Captures
                        # the direct combustion-rate suppression that mass-
                        # fraction dilution of Y_F·Y_O2 alone underestimates
                        # at field grass density where Y_H2O is sparse.
                        Y_H2O_cell_for_suppr = Y_H2O[k, j, i]
                        _h2o_quench_substantial = False
                        if Y_H2O_cell_for_suppr > 0.0:
                            _h2o_factor = (1.0 - Y_H2O_cell_for_suppr
                                                  / Y_H2O_QUENCH)
                            if _h2o_factor < 0.0:
                                _h2o_factor = 0.0
                            omega = omega * _h2o_factor
                            # Tier 2-C trigger (2026-06-21): when Y_H2O
                            # quench reduces omega by ≥50%, route through
                            # the Drysdale §3.5 wet-bulb cascade below so
                            # the gas-phase T_g also cools.  This is what
                            # closes the loop between the moisture quench
                            # and the radiation feedback (lower T_g →
                            # lower σT⁴ forward emission → slower preheat
                            # ahead of front → lower ROS).  Earlier
                            # implementation gated wet-bulb on the A/B/C
                            # extinction state which (per Phase 16 memo)
                            # had natural-fire artifacts; using the Y_H2O
                            # trigger ties cooling to the moisture cause
                            # directly without touching A/B/C.
                            if _h2o_factor < 0.5:
                                _h2o_quench_substantial = True

                    # ── Extinction-threshold gates (Phase 16, 2026-06-18) ──
                    # Opt-in via extinction_enable.  Default OFF preserves
                    # 0D adiabatic startup tests where the flame must
                    # begin cold (below T_IGNITION_MIN) and heat itself.
                    # Production 3D runs (mickey, Cheney, Catchpole,
                    # suppression) opt IN via deck flag.
                    _extinction_fired = False
                    if extinction_enable and omega > 0.0:
                        # (B) Inert-fraction suppression (Beyler 1992):
                        # smoothly ramps omega → 0 as (Y_F + Y_O2) →
                        # (1 − Y_INERT_CRIT).  Captures water-mist / CO2
                        # / N2 dilution.  Y_inert upper bound is
                        # 1 − Y_F − Y_O2 (no Y_H2O access here).
                        Y_inert_bound = 1.0 - Yf - YO2
                        if Y_inert_bound > Y_INERT_CRIT:
                            _rate_supp = ((1.0 - Y_inert_bound)
                                          / (1.0 - Y_INERT_CRIT))
                            if _rate_supp < 0.0:
                                _rate_supp = 0.0
                            omega = omega * _rate_supp
                            if _rate_supp < 0.5:
                                _extinction_fired = True
                        # (C) Cold-flame floor (Westbrook & Dryer 1981):
                        # omega → 0 below hydrocarbon ignition T.
                        if omega > 0.0 and Tg < T_IGNITION_MIN:
                            omega = 0.0
                            _extinction_fired = True
                        # (A) Marginal heat-release-rate threshold
                        # (Linn 2002 / Drysdale §3.4 cellulose MEP):
                        # below this the flame can't overcome local
                        # radiative + convective losses.
                        Q_RATE_MIN_MARGINAL = 5.0e4   # [W/m³]
                        if omega > 0.0:
                            Q_rate = omega * HoC_eff
                            if Q_rate < (EXTINCTION_F_SAFETY
                                          * Q_RATE_MIN_MARGINAL):
                                omega = 0.0
                                _extinction_fired = True
                    # ── Wet-bulb cooling cascade (Drysdale §3.5) ──
                    # Tier 2-C (2026-06-21): always-on, gated by either the
                    # Y_H2O substantial quench OR (A/B/C) extinction firing.
                    # Rate-limited toward T_BOIL_WATER with τ_wb=0.5s.  Earlier
                    # implementation only fired on A/B/C; Phase 17a moves it
                    # out so Y_H2O quench drives gas cooling too, closing the
                    # quench→T_g→σT⁴→preheat→ROS feedback loop.
                    #
                    # Rate-limited form (not instant pin): otherwise momentum
                    # EoS blows up on rapid ρ jumps (verified empirically —
                    # instant pin caused u → 10⁸⁰ m/s in 0.3 s sim).
                    #
                    # Exponential relaxation per substep:
                    #   dT = (T_g − T_wb)·(1 − exp(−h/τ_wb)) · Y_H2O_strength
                    if (_extinction_fired or _h2o_quench_substantial) \
                            and Tg > T_BOIL_WATER_EDC:
                        Y_H2O_cell_for_wb = Y_H2O[k, j, i]
                        if Y_H2O_cell_for_wb > 0.0:
                            wb_strength = Y_H2O_cell_for_wb / Y_H2O_SAT_THRESH
                            if wb_strength > 1.0:
                                wb_strength = 1.0
                            TAU_WB = 0.5     # [s] wet-bulb relax timescale
                            relax_factor = (1.0
                                            - math.exp(-h / TAU_WB))
                            dT_wb = (Tg - T_BOIL_WATER_EDC) * relax_factor * wb_strength
                            Tg = Tg - dT_wb

                    # Explicit Euler
                    dY  = -omega * h / rho_i           # ΔY_F
                    Yf  = Yf + dY
                    if Yf < 0.0:
                        Yf = 0.0
                    YO2 = YO2 + s_stoich * dY          # ΔY_O2 = s · ΔY_F
                    if YO2 < 0.0:
                        YO2 = 0.0
                    # Composition-dependent cp_mix (Phase 16 2026-06-18):
                    # water vapor (cp≈2000 J/kg/K) adds thermal inertia
                    # in moisture-laden cells.  Y_H2O passed as zeros
                    # array → cp_mix == cp_g (legacy behavior).
                    Y_H2O_cell = Y_H2O[k, j, i]
                    cp_mix = (1.0 - Y_H2O_cell) * cp_g + Y_H2O_cell * CP_VAPOR_EDC
                    Tg  = Tg + omega * HoC_eff * h / (rho_i * cp_mix)
                    if Tg > 2400.0:                    # cap at T_ad-ish
                        Tg = 2400.0

                    omega_acc += omega * h

                Y_fuel[k, j, i] = Yf
                Y_O2[k, j, i]   = YO2
                T_g[k, j, i]    = Tg
                omega_int_out[k, j, i] = omega_acc / dt if dt > 0.0 else 0.0


def run(
    *,
    rho: np.ndarray,
    T_g: np.ndarray,
    Y_fuel: np.ndarray,
    Y_O2: np.ndarray,
    k_turb: np.ndarray,
    eps_turb: np.ndarray,
    chi_rad: float,
    cp_g: float,
    dt: float,
    n_substeps: int,
    omega_out: np.ndarray,
    Y_H2O: np.ndarray = None,
    extinction_enable: bool = False,
    # Phase 23 chemistry-family kwargs (biomass defaults preserve
    # pre-Phase-23 behaviour bit-exact).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
    **_unused,
) -> None:
    """Pluggable-closure entry point for Magnussen 1981 EDC.

    See chemistry_closures._interface for the contract.  Extra kwargs
    not listed in the signature (e.g., tau_mix, omega_O2) are silently
    ignored — they are needed by other closures.

    Y_H2O: per-cell water-vapor mass fraction.  When None, treated as
    zeros (constant cp_g — legacy behavior preserved for 0D tests).
    extinction_enable: opt-in extinction-threshold physics A+B+C.
    """
    if Y_H2O is None:
        Y_H2O = np.zeros_like(T_g)
    step_chemistry_ode_edc(
        rho, T_g, Y_fuel, Y_O2, k_turb, eps_turb,
        chi_rad, cp_g, dt, n_substeps, omega_out,
        Y_H2O,
        bool(extinction_enable),
        s_stoich=s_stoich, hoc_J=hoc_J,
    )
