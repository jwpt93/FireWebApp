"""Westbrook-Dryer 1981 2-step methane oxidation in the EDC framework.

Two sequential Arrhenius reactions:

  R1:  CH4 + 1.5 O2 → CO + 2 H2O          A1 = 2.8e9,  E1 = 202 kJ/mol
  R2:  CO  + 0.5 O2 → CO2                  A2 = 2.24e12, E2 = 170 kJ/mol

Reference: Westbrook & Dryer (1981) "Simplified reaction mechanisms for
the oxidation of hydrocarbon fuels in flames," Combust. Sci. Tech.
27:31-43.  R2 uses the "single-step CO oxidation" form (dropping the
[H2O]^0.5 fractional-order term the original paper includes) — the
same approximation FDS uses in its 2-step methane closure.

Why 2-step for MEC prediction
=============================
Real cup-burner extinction at MEC is set by CO oxidation kinetics.
The overall CH4 → CO2 pathway is fast at flame T, but CO → CO2 slows
sharply as O2 falls and T drops (because CO oxidation depends on OH-
radical chain regeneration, which N2 collisions scavenge).  Single-
step Arrhenius CH4 → CO2 misses this because it lumps R1 + R2 into
one rate with an intermediate-averaged activation energy.

In our EDC framework, ω_R1 and ω_R2 are computed independently as
min(ω_chem, ω_mix) — so when R2 becomes chemistry-limited near
extinction, R2 rate drops while R1 keeps going.  Heat release from
R2 (which is ~60% of the total) drops → gas cools → chain reaction →
extinction.  Whether that produces the correct MEC value is what
this closure tests.

Stoichiometry (mass basis)
==========================
R1: 1 kg CH4 + s_R1 kg O2 → b_CO kg CO + b_H2O kg H2O
    s_R1 = 1.5 × 32 / 16 = 3.0
    b_CO = 28 / 16 = 1.75
    b_H2O = 2 × 18 / 16 = 2.25
R2: 1 kg CO + s_R2 kg O2 → b_CO2 kg CO2
    s_R2 = 0.5 × 32 / 28 = 0.5714
    b_CO2 = 44 / 28 = 1.5714

Heat release (Westbrook & Dryer 1981 Table 1)
=============================================
    HR_R1 (per kg CH4) ≈ 5.02e6 J/kg     (small — CO not fully oxidized)
    HR_R2 (per kg CO)  ≈ 10.10e6 J/kg    (large — CO → CO2 releases the
                                          rest of the LHV)
Total per kg CH4: 5.02e6 + 1.75 × 10.10e6 = 22.7e6 → this equals the
"partial" LHV.  The full LHV (50e6 J/kg) is when H2O condenses to
liquid; for gaseous-water combustion products (our case), 50e6 is
still the total.  Our numbers give partial ≈ 22.7 MJ/kg because
Westbrook & Dryer's HR values are for the reaction enthalpies at
reference conditions.  For deck-consistency with the 1-step case
(which uses HOC_J = 50e6 for methane), we scale the reaction
enthalpies so their sum matches HOC_J:
    HR_R1_fraction = 0.221    (5.02 / (5.02 + 1.75×10.10))
    HR_R2_fraction = 0.779

Rule #18 (unit tests): included alongside this module.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._constants import _R_GAS
from .edc import (
    C_GAMMA_EDC, C_TAU_EDC, NU_GAS_EDC,
    K_TURB_FLOOR_EDC, EPS_TURB_FLOOR_EDC,
    CP_VAPOR_EDC,
)


# Westbrook-Dryer 1981 kinetics.
A_R1 = 2.8e9        # [1/s] CH4 oxidation pre-exponential (R1)
E_R1 = 202_000.0    # [J/mol] CH4 oxidation activation energy
A_R2 = 2.24e12      # [1/s] CO oxidation pre-exponential (R2)
E_R2 = 170_000.0    # [J/mol] CO oxidation activation energy

# Mass-basis stoichiometry.
S_STOICH_R1 = 3.0      # kg O2 / kg CH4 (R1)
S_STOICH_R2 = 4.0 / 7.0    # kg O2 / kg CO ≈ 0.5714
B_CO_R1     = 7.0 / 4.0    # kg CO / kg CH4 = 1.75
B_H2O_R1    = 9.0 / 4.0    # kg H2O / kg CH4 = 2.25
B_CO2_R2    = 11.0 / 7.0   # kg CO2 / kg CO ≈ 1.5714

# Heat-release fractions (sum to 1; multiply by hoc_J for per-reaction J/kg).
HR_R1_FRACTION = 5.02e6 / (5.02e6 + B_CO_R1 * 10.10e6)   # ≈ 0.221
HR_R2_FRACTION = 1.0 - HR_R1_FRACTION                    # ≈ 0.779


@njit(cache=True, parallel=True)
def step_chemistry_ode_edc_2step(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]  updated in place
    Y_fuel: np.ndarray,        # (Nz, Ny, Nx) [-]  updated in place
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-]  updated in place
    Y_CO: np.ndarray,          # (Nz, Ny, Nx) [-]  updated in place
    k_turb: np.ndarray,        # (Nz, Ny, Nx) [m²/s²] TKE
    eps_turb: np.ndarray,      # (Nz, Ny, Nx) [m²/s³] dissipation
    chi_rad: float,
    cp_g: float,               # [J/kg/K] dry-air baseline
    dt: float,
    n_substeps: int,
    omega_int_out: np.ndarray, # (Nz, Ny, Nx) [kg/m³/s] time-averaged TOTAL ω
    hoc_J:    float,           # [J/kg CH4] total heat of combustion (methane 50e6)
) -> None:
    """2-step Westbrook-Dryer methane oxidation in EDC.

    Each reaction rate = min(ω_chem, ω_mix) where ω_chem is the local
    Arrhenius rate and ω_mix is the EDC fine-structure mixing rate
    (same γ* + τ* form as the 1-step edc.step_chemistry_ode_edc).

    Species updates per outer substep:
      Y_F   -= ω_R1 · h / ρ
      Y_O2  -= (S_STOICH_R1 · ω_R1 + S_STOICH_R2 · ω_R2) · h / ρ
      Y_CO  += (B_CO_R1 · ω_R1 − ω_R2) · h / ρ
      T_g   += (HR_R1 · ω_R1 + HR_R2 · ω_R2) · (1 − χ_rad) · h / (ρ · cp_g)

    Uses the same γ*, τ* Magnussen EDC framework as edc.py.  omega_int_out
    accumulates the CH4-equivalent time-averaged rate for diagnostic
    parity with the 1-step closure (multiply by 2 to get O2 use rate).
    """
    Nz, Ny, Nx = rho.shape
    h = dt / max(n_substeps, 1)
    HR_R1_J_per_kgF = HR_R1_FRACTION * hoc_J
    HR_R2_J_per_kgCO = HR_R2_FRACTION * hoc_J / B_CO_R1

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Yf   = Y_fuel[k, j, i]
                YO2  = Y_O2[k, j, i]
                Yco  = Y_CO[k, j, i]
                Tg   = T_g[k, j, i]
                rho_i = rho[k, j, i]

                if Tg <= 0.0 or rho_i <= 0.0:
                    omega_int_out[k, j, i] = 0.0
                    continue

                inv_rho    = 1.0 / rho_i
                inv_rho_cp = 1.0 / (rho_i * cp_g)

                # ── EDC mixing timescale (Magnussen 1989 fine-structure) ──
                k_t   = k_turb[k, j, i]
                eps_t = eps_turb[k, j, i]
                if k_t < K_TURB_FLOOR_EDC:
                    k_t = K_TURB_FLOOR_EDC
                if eps_t < EPS_TURB_FLOOR_EDC:
                    eps_t = EPS_TURB_FLOOR_EDC
                # γ* = (C_γ · (ν·ε/k²)^(1/4))^3
                gamma_star = (C_GAMMA_EDC *
                              (NU_GAS_EDC * eps_t / (k_t * k_t)) ** 0.25) ** 3
                if gamma_star < 1.0e-6:
                    gamma_star = 1.0e-6
                if gamma_star > 0.99:
                    gamma_star = 0.99
                # τ* = C_τ · (ν/ε)^(1/2)
                tau_star = C_TAU_EDC * (NU_GAS_EDC / eps_t) ** 0.5
                if tau_star < 1.0e-6:
                    tau_star = 1.0e-6
                edc_prefac = gamma_star * rho_i / tau_star

                omega_R1_acc = 0.0
                omega_R2_acc = 0.0

                for _sub in range(n_substeps):
                    if Yf <= 0.0 and Yco <= 0.0:
                        break
                    if YO2 <= 0.0 or Tg <= 250.0:
                        break

                    # ── R1 rate: CH4 + 1.5 O2 → CO + 2 H2O ────────
                    if Yf > 0.0:
                        k_chem_R1 = A_R1 * math.exp(-E_R1 / (_R_GAS * Tg))
                        omega_chem_R1 = rho_i * k_chem_R1 * Yf * YO2
                        Y_lim_R1 = Yf if Yf < (YO2 / S_STOICH_R1) \
                                       else (YO2 / S_STOICH_R1)
                        omega_mix_R1 = edc_prefac * Y_lim_R1
                        omega_R1 = omega_chem_R1 if omega_chem_R1 < omega_mix_R1 \
                                                 else omega_mix_R1
                    else:
                        omega_R1 = 0.0

                    # ── R2 rate: CO + 0.5 O2 → CO2 ────────────────
                    if Yco > 0.0:
                        k_chem_R2 = A_R2 * math.exp(-E_R2 / (_R_GAS * Tg))
                        omega_chem_R2 = rho_i * k_chem_R2 * Yco * YO2
                        Y_lim_R2 = Yco if Yco < (YO2 / S_STOICH_R2) \
                                       else (YO2 / S_STOICH_R2)
                        omega_mix_R2 = edc_prefac * Y_lim_R2
                        omega_R2 = omega_chem_R2 if omega_chem_R2 < omega_mix_R2 \
                                                 else omega_mix_R2
                    else:
                        omega_R2 = 0.0

                    # Cap ω by species availability over the substep h.
                    # Prevents overshoot Y < 0 in the explicit update.
                    max_R1 = (Yf * rho_i) / h if h > 0.0 else 1.0e30
                    if omega_R1 > max_R1:
                        omega_R1 = max_R1
                    # For R2: CO consumption capped by min(Y_CO, Y_O2/s_R2)/h.
                    _r2_cap_CO  = (Yco * rho_i) / h if h > 0.0 else 1.0e30
                    _r2_cap_O2  = (YO2 * rho_i) / (S_STOICH_R2 * h) if h > 0.0 else 1.0e30
                    if omega_R2 > _r2_cap_CO:
                        omega_R2 = _r2_cap_CO
                    if omega_R2 > _r2_cap_O2:
                        omega_R2 = _r2_cap_O2
                    # Also cap R1 by O2 availability considering R2 also
                    # consumes O2:  s_R1·ω_R1 + s_R2·ω_R2 ≤ (Y_O2·ρ)/h
                    _o2_avail = (YO2 * rho_i) / h if h > 0.0 else 1.0e30
                    _o2_needed = S_STOICH_R1 * omega_R1 + S_STOICH_R2 * omega_R2
                    if _o2_needed > _o2_avail:
                        # Scale both back proportionally
                        _scale = _o2_avail / _o2_needed
                        omega_R1 *= _scale
                        omega_R2 *= _scale

                    # ── Advance state (explicit Euler substep) ─────
                    dYf   = -omega_R1 * h * inv_rho
                    dYO2  = -(S_STOICH_R1 * omega_R1 + S_STOICH_R2 * omega_R2) * h * inv_rho
                    dYco  = (B_CO_R1 * omega_R1 - omega_R2) * h * inv_rho
                    dTg   = (HR_R1_J_per_kgF * omega_R1 * (1.0 - chi_rad)
                             + HR_R2_J_per_kgCO * omega_R2 * (1.0 - chi_rad)) \
                            * h * inv_rho_cp

                    Yf  = Yf + dYf
                    if Yf < 0.0: Yf = 0.0
                    YO2 = YO2 + dYO2
                    if YO2 < 0.0: YO2 = 0.0
                    Yco = Yco + dYco
                    if Yco < 0.0: Yco = 0.0
                    Tg  = Tg + dTg
                    if Tg > 2400.0:
                        Tg = 2400.0

                    omega_R1_acc += omega_R1 * h
                    omega_R2_acc += omega_R2 * h

                Y_fuel[k, j, i] = Yf
                Y_O2[k, j, i]   = YO2
                Y_CO[k, j, i]   = Yco
                T_g[k, j, i]    = Tg
                # Report time-averaged CH4-consumption rate (R1 rate)
                # for diagnostic parity with the 1-step closure's
                # omega_int_out.
                omega_int_out[k, j, i] = omega_R1_acc / dt if dt > 0.0 else 0.0


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
    Y_CO: np.ndarray,          # required for this closure
    hoc_J: float = 50_000_000.0,
    **_unused,
) -> None:
    """Pluggable-closure entry point for Westbrook-Dryer 2-step methane.

    Extra kwargs beyond the base contract (Y_CO, hoc_J) MUST be supplied
    by the main loop.  See chemistry_closures._interface for the base
    contract; this closure additionally reads/writes Y_CO in place.
    """
    step_chemistry_ode_edc_2step(
        rho, T_g, Y_fuel, Y_O2, Y_CO, k_turb, eps_turb,
        chi_rad, cp_g, dt, n_substeps, omega_out, hoc_J,
    )
