"""Chomiak 1990 PaSR (Partially Stirred Reactor) closure.

Each cell partitioned into a reactive fraction γ_pasr and an inert
fraction (1-γ_pasr).  Reactive fraction has Arrhenius rate; γ_pasr
balances chemistry vs mixing time scales.

    ω_cell = γ_pasr · ω_arrhenius,  γ_pasr = τ_chem / (τ_chem + τ_mix)
    τ_chem = ρ · Y_F / ω_arrhenius

At low T, ω_arrh small → τ_chem large → γ_pasr → 1 → ω_cell ≈ ω_arrh
    (chemistry-limited)
At high T, ω_arrh large → τ_chem << τ_mix → γ_pasr → τ_chem/τ_mix → 0
    so ω_cell → ω_arrh · τ_chem/τ_mix (mixing-limited, EBU-like)

Lit refs:
- Chomiak, J. (1990) "Combustion: a study in theory, fact and application"
  §6.5 Partially-stirred reactor model
- Sabel'nikov, V. & Figueira da Silva, L.F. (2002) Combust. Theory Model.
  6:511 — PaSR formulation used in OpenFOAM reactingFoam

Phase 15-0: moved from combustion_3d.py into the chemistry_closures
registry framework.  Numerics unchanged (Rule #17 verified).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._constants import S_STOICH, HOC_J, A_COMB, E_COMB, _R_GAS


@njit(cache=True, parallel=True)
def step_chemistry_ode_pasr(
    rho: np.ndarray,
    T_g: np.ndarray,
    Y_fuel: np.ndarray,
    Y_O2: np.ndarray,
    tau_mix: np.ndarray,       # (Nz, Ny, Nx) [s] EBU mixing time = k/ε floored
    chi_rad: float,
    cp_g: float,
    dt: float,
    n_substeps: int,
    omega_int_out: np.ndarray,
    # Phase 23: chemistry-family kwargs (biomass defaults preserve Rule #17
    # bit-exact for pre-Phase-23 callers).  Cup burner deck sets methane
    # values via chemistry_closures.run(..., s_stoich=4.0, hoc_J=50e6, ...).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
    a_comb:   float = A_COMB,
    e_comb:   float = E_COMB,
) -> None:
    """Chomiak 1990 PaSR: ω_cell = γ_pasr · ω_arrhenius.

    Same ODE structure as step_chemistry_ode_edc.  Explicit Euler.
    See module docstring for the closure formula.
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
                tau_m = tau_mix[k, j, i]
                if tau_m < 1e-6:
                    tau_m = 1e-6

                omega_acc = 0.0
                for _ in range(max(n_substeps, 1)):
                    if Yf <= 1e-9 or YO2 <= 1e-9 or Tg <= 250.0:
                        omega = 0.0
                    else:
                        # Arrhenius rate (Westbrook & Dryer)
                        Y_lim = Yf if Yf < (YO2 / s_stoich) else (YO2 / s_stoich)
                        k_arrh = a_comb * math.exp(-e_comb / (_R_GAS * Tg))
                        omega_arrh = rho_i * k_arrh * Yf * YO2     # [kg/m³/s]

                        if omega_arrh <= 1e-30:
                            omega = 0.0
                        else:
                            # τ_chem = Y_F / (ω_arrh/ρ) = ρ·Y_F/ω_arrh
                            tau_chem = rho_i * Yf / omega_arrh
                            gamma_pasr = tau_chem / (tau_chem + tau_m)
                            omega = gamma_pasr * omega_arrh
                            # Cap at well-mixed EBU rate to avoid runaway
                            omega_ebu_cap = rho_i * Y_lim / tau_m
                            if omega > omega_ebu_cap:
                                omega = omega_ebu_cap

                    dY  = -omega * h / rho_i
                    Yf  = Yf + dY
                    if Yf < 0.0:
                        Yf = 0.0
                    YO2 = YO2 + s_stoich * dY
                    if YO2 < 0.0:
                        YO2 = 0.0
                    Tg  = Tg + omega * HoC_eff * h / (rho_i * cp_g)
                    if Tg > 2400.0:
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
    tau_mix: np.ndarray,
    chi_rad: float,
    cp_g: float,
    dt: float,
    n_substeps: int,
    omega_out: np.ndarray,
    # Phase 23 chemistry-family kwargs (biomass defaults preserve
    # pre-Phase-23 behaviour bit-exact).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
    a_comb:   float = A_COMB,
    e_comb:   float = E_COMB,
    **_unused,
) -> None:
    """Pluggable-closure entry point for Chomiak 1990 PaSR.

    See chemistry_closures._interface for the contract.  Extra kwargs
    not listed in the signature (e.g., k_turb, eps_turb, omega_O2) are
    silently ignored — they are needed by other closures.
    """
    step_chemistry_ode_pasr(
        rho, T_g, Y_fuel, Y_O2, tau_mix,
        chi_rad, cp_g, dt, n_substeps, omega_out,
        s_stoich=s_stoich, hoc_J=hoc_J, a_comb=a_comb, e_comb=e_comb,
    )
