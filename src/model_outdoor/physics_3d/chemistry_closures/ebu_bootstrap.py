"""Phase 14s 4-rate ODE closure (EBU + Arrhenius + O₂-supply + Damköhler).

ω = min(ω_chem, ω_EBU, ω_O2_supply, ω_max_T)
    ω_chem    = ρ·A·exp(-E/RT)·Y_F·Y_O2     [Westbrook & Dryer 1981]
    ω_EBU     = C·ρ·min(Y_F, Y_O2/s)/τ_mix   [Magnussen & Hjertager 1977]
    ω_O2      = frozen omega_O2_supply       [Pruyn et al. 2018 — bounds
                                              advective O₂ delivery]
    ω_max_T   = ρ·(S_L+u')/dx (frozen)       [Damköhler 1940; Williams 1985 —
                                              turbulent-flame-speed cap]

Method: linearly-implicit Euler (Rosenbrock-1).  Per inner substep
h = dt / n_substeps:

    (I − h·J) · Δy = h · f(y)
    y ← y + Δy

L-stable for any h (handles stiff Arrhenius).  Branch-aware Jacobian
for ω, 3×3 solve via Cramer's rule.

Used with the Phase 14x bootstrap mechanism (apply_bootstrap_heat in
flame_front_3d) for legacy 'ebu_bootstrap' compatibility.

Lit refs:
- Hairer & Wanner (1996) Solving ODE II §IV.7 — Rosenbrock-1
- Magnussen & Hjertager (1977) Symp. Combust. 16:719 — EDM
- Westbrook & Dryer (1981) Combust. Sci. Tech. 27:31 — Arrhenius
- Pruyn et al. (2018) Combust. Flame 187:182 — O₂-supply rate-limit
- Damköhler (1940) Z. Elektrochem. 46:601 — turbulent flame speed
- Williams (1985) Combust. Sci. Tech. 41:235 — laminar flame speed

Phase 15-0: moved from combustion_3d.py into the chemistry_closures
registry framework.  Numerics unchanged (Rule #17 verified).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._constants import S_STOICH, HOC_J, A_COMB, E_COMB, _R_GAS, C_EBU


@njit(cache=True, parallel=True)
def step_chemistry_ode(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]  updated in place
    Y_fuel: np.ndarray,        # (Nz, Ny, Nx) [-]  updated in place
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-]  updated in place
    tau_mix: np.ndarray,       # (Nz, Ny, Nx) [s]  EBU mixing time
    omega_O2_supply: np.ndarray,  # (Nz, Ny, Nx) [kg fuel/m³/s] frozen input
    omega_max_T: np.ndarray,   # (Nz, Ny, Nx) [kg/m³/s] Damköhler turbulent-flame-speed cap
    chi_rad: float,
    cp_g: float,               # [J/kg/K] gas specific heat
    dt: float,                 # [s] outer chemistry sub-step
    n_substeps: int,           # internal sub-cycle (≥1)
    omega_int_out: np.ndarray, # (Nz, Ny, Nx) [kg/m³/s] time-averaged ω over dt
    # Phase 23 chemistry-family scalars (biomass defaults preserve
    # bit-exact behaviour per Rule #17).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
    a_comb:   float = A_COMB,
    e_comb:   float = E_COMB,
    c_ebu:    float = C_EBU,
) -> None:
    """Phase 14s — operator-split chemistry: per-cell stiff ODE integration.

    Solves the local 3-state ODE [Y_F, Y_O2, T_g] over dt with no transport
    coupling.  Advances state in place and returns time-averaged ω.

        dY_F/dt   = -ω/ρ
        dY_O2/dt  = -s·ω/ρ
        dT_g/dt   =  ω·HoC·(1-χ_rad)/(ρ·cp_g)

    See module docstring for the 4-rate closure and integrator details.
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
                tau   = tau_mix[k, j, i]
                omO2  = omega_O2_supply[k, j, i]
                omT   = omega_max_T[k, j, i]   # Damköhler cap

                if Yf <= 0.0 or YO2 <= 0.0 or Tg <= 0.0 or rho_i <= 0.0:
                    omega_int_out[k, j, i] = 0.0
                    continue

                inv_rho = 1.0 / rho_i
                inv_rho_cp = 1.0 / (rho_i * cp_g)
                omega_acc = 0.0  # time-integrated ω · h, divided by dt at end

                for _sub in range(n_substeps):
                    if Yf <= 0.0 or YO2 <= 0.0:
                        break

                    # ── Compute ω and identify binding branch ──────────
                    k_chem = a_comb * math.exp(-e_comb / (_R_GAS * Tg))
                    omega_chem = rho_i * k_chem * Yf * YO2

                    if tau <= 0.0 or tau >= 1.0e30:
                        omega_ebu = 1.0e30
                    else:
                        Y_O2_avail = YO2 / s_stoich
                        Y_lim = Yf if Yf < Y_O2_avail else Y_O2_avail
                        omega_ebu = c_ebu * rho_i * Y_lim / tau

                    omega = omega_chem
                    branch = 0  # 0=chem, 1=ebu, 2=o2_supply, 3=damköhler
                    if omega_ebu < omega:
                        omega = omega_ebu
                        branch = 1
                    if omT < omega:
                        # Damköhler turbulent-flame-speed cap: ω ≤ ρ·(S_L+u')/dx.
                        omega = omT
                        branch = 3
                    if omO2 < omega:
                        omega = omO2
                        branch = 2

                    if omega <= 0.0:
                        break

                    # ── Branch-aware Jacobian rows for ω ──────────────
                    if branch == 0:
                        # Arrhenius binding
                        dom_dYf  = omega / Yf
                        dom_dYO2 = omega / YO2
                        dom_dT   = omega * e_comb / (_R_GAS * Tg * Tg)
                    elif branch == 1:
                        # EBU binding: ω = C·ρ·min(Y_F, Y_O2/s)/τ
                        if Yf < (YO2 / s_stoich):
                            dom_dYf  = c_ebu * rho_i / tau
                            dom_dYO2 = 0.0
                        else:
                            dom_dYf  = 0.0
                            dom_dYO2 = c_ebu * rho_i / (s_stoich * tau)
                        dom_dT = 0.0
                    else:
                        # O₂-supply (branch 2) or Damköhler cap (branch 3).
                        dom_dYf  = 0.0
                        dom_dYO2 = 0.0
                        dom_dT   = 0.0

                    # f(y) = [-ω/ρ, -s·ω/ρ, +ω·HoC_eff/(ρ·cp)]
                    f0 = -omega * inv_rho
                    f1 = -s_stoich * omega * inv_rho
                    f2 =  omega * HoC_eff * inv_rho_cp

                    # J = ∂f/∂y (3x3, sparse from ω-derivatives)
                    j00 = -dom_dYf  * inv_rho
                    j01 = -dom_dYO2 * inv_rho
                    j02 = -dom_dT   * inv_rho
                    j10 = -s_stoich * dom_dYf  * inv_rho
                    j11 = -s_stoich * dom_dYO2 * inv_rho
                    j12 = -s_stoich * dom_dT   * inv_rho
                    j20 =  dom_dYf  * HoC_eff * inv_rho_cp
                    j21 =  dom_dYO2 * HoC_eff * inv_rho_cp
                    j22 =  dom_dT   * HoC_eff * inv_rho_cp

                    # M = I − h·J
                    m00 = 1.0 - h * j00
                    m01 =     - h * j01
                    m02 =     - h * j02
                    m10 =     - h * j10
                    m11 = 1.0 - h * j11
                    m12 =     - h * j12
                    m20 =     - h * j20
                    m21 =     - h * j21
                    m22 = 1.0 - h * j22

                    # RHS = h · f(y)
                    r0 = h * f0
                    r1 = h * f1
                    r2 = h * f2

                    # 3x3 solve via Cramer's rule
                    det = (m00 * (m11 * m22 - m12 * m21)
                           - m01 * (m10 * m22 - m12 * m20)
                           + m02 * (m10 * m21 - m11 * m20))
                    if abs(det) < 1.0e-30:
                        # Fall back to forward Euler if singular
                        d0 = r0
                        d1 = r1
                        d2 = r2
                    else:
                        inv_det = 1.0 / det
                        d0 = ((r0 * (m11 * m22 - m12 * m21)
                              - m01 * (r1 * m22 - m12 * r2)
                              + m02 * (r1 * m21 - m11 * r2)) * inv_det)
                        d1 = ((m00 * (r1 * m22 - m12 * r2)
                              - r0  * (m10 * m22 - m12 * m20)
                              + m02 * (m10 * r2 - r1 * m20)) * inv_det)
                        d2 = ((m00 * (m11 * r2 - r1 * m21)
                              - m01 * (m10 * r2 - r1 * m20)
                              + r0  * (m10 * m21 - m11 * m20)) * inv_det)

                    Yf  = Yf  + d0
                    YO2 = YO2 + d1
                    Tg  = Tg  + d2

                    # Clamp
                    if Yf < 0.0:
                        Yf = 0.0
                    if YO2 < 0.0:
                        YO2 = 0.0
                    if YO2 > 0.232:
                        YO2 = 0.232
                    if Yf > 1.0:
                        Yf = 1.0

                    omega_acc += omega * h

                # Write back
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
    omega_O2: np.ndarray,
    omega_max_T: np.ndarray,
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
    c_ebu:    float = C_EBU,
    **_unused,
) -> None:
    """Pluggable-closure entry point for the 4-rate EBU+Arrhenius closure.

    See chemistry_closures._interface for the contract.  Extra kwargs
    not listed in the signature (e.g., k_turb, eps_turb) are silently
    ignored — they are needed by other closures.

    Caller (spread_3d.py main loop) must additionally apply the
    Phase 14x bootstrap heat via flame_front_3d.apply_bootstrap_heat
    when combustion_closure=='ebu_bootstrap'.  The bootstrap is NOT
    part of this closure module by design (it operates on Q_comb /
    flame_body_mask, which sit downstream of the chemistry ODE).
    """
    step_chemistry_ode(
        rho, T_g, Y_fuel, Y_O2, tau_mix, omega_O2, omega_max_T,
        chi_rad, cp_g, dt, n_substeps, omega_out,
        s_stoich=s_stoich, hoc_J=hoc_J, a_comb=a_comb,
        e_comb=e_comb, c_ebu=c_ebu,
    )
