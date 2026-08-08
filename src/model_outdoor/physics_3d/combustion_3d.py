"""Gas-phase combustion utilities and shared constants.

Phase 15-0 refactor (2026-06-05): chemistry-closure ODE kernels (EDC,
EBU+Arrhenius+O₂-supply+Damköhler, PaSR) moved from this file into the
``chemistry_closures`` package and dispatched through a registry — see
:mod:`model_outdoor.physics_3d.chemistry_closures` and its ``_interface.py``
for the plug-in contract.  The moved kernels are re-imported here at
module load for backward compatibility with existing test imports
(``tests/outdoor/test_edc_0d_adiabatic_validation.py``,
``tests/outdoor/test_chemistry_0d_validation.py``).

This module retains:

  - Shared physical constants (``S_STOICH``, ``HOC_J``, ``A_COMB``, …),
    re-exported from :mod:`chemistry_closures._constants` so existing
    callers in ``tests/outdoor/test_3d_components.py`` continue to work.

  - Rate-only kernel ``step_combustion`` — three-rate min(ω_chem, ω_mix)
    used by older isolation tests; not part of the production main loop.

  - O₂-supply rate kernel ``step_o2_supply_rate`` — face-flux-sum O₂
    delivery limit, called from ``spread_3d.py`` BEFORE the chemistry
    closure dispatch as a frozen array input.

  - T_g pin kernel ``apply_t_g_pin`` — Mell 2007 WFDS-pure flame-zone
    temperature pin used by some closures as a post-step energy-conserving
    way to maintain T_g ≥ T_flame in flame_body cells.

Module-docstring references (full lit trail used by closures and constants):
- Magnussen, B.F. & Hjertager, B.H. (1977) Symp. (Int.) Combust.
  16:719 — EDM with C_EBU = 4.0 for turbulent diffusion flames
- Westbrook & Dryer (1981) Combust. Sci. Tech. 27:31 — Arrhenius
- Spalding, D.B. (1971) Combust. Sci. Tech. 4:43 — three-rate min()
- Pruyn et al. (2018) Combust. Flame 187:182 — O₂-supply combustion in grass
- Grishin & Perminov (2002) Combust. Explos. Shock 38:131 — O₂ transport
- Susott, R.A. (1980) Forest Sci. 26:347 — grass biomass HoC + composition
- Sung et al. (2025) NIST TN 2314 — effective HoC (cone-cal) for grass
- Drysdale (2011) Fire Dynamics 3rd ed. §1.2.3 + Tab 1.13 — T_ad
- Pitts (1995) Prog. Energy Combust. Sci. 21:197 — cellulosic flame T
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ── Shared constants ────────────────────────────────────────────────────────
# Source of truth: chemistry_closures._constants.  Re-exported here so
# existing import sites (test_3d_components.py: step_combustion +
# constants) continue to work without modification.
from .chemistry_closures._constants import (
    S_STOICH,
    Y_O2_AIR,
    A_COMB,
    E_COMB,
    _R_GAS,
    C_EBU,
    HOC_J,
)

# ── Backward-compat re-exports for moved chemistry-closure kernels ──────────
# Phase 15-0: closures now live in chemistry_closures/<name>.py.  Existing
# tests import these from combustion_3d directly; the rebind below keeps
# that path working.  Production main loop (spread_3d.py) dispatches via
# chemistry_closures.run(closure_name, **kwargs) instead.
from .chemistry_closures.edc import step_chemistry_ode_edc
from .chemistry_closures.ebu_bootstrap import step_chemistry_ode
from .chemistry_closures.pasr import step_chemistry_ode_pasr


@njit(cache=True, parallel=True)
def step_combustion(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]
    Y_fuel: np.ndarray,        # (Nz, Ny, Nx) [-]
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-]  transported species
    tau_mix: np.ndarray,       # (Nz, Ny, Nx) [s]  +inf-equivalent for laminar
    chi_rad: float,            # [-] radiative fraction
    omega_out: np.ndarray,     # (Nz, Ny, Nx) [kg/m³/s] (overwritten)
    Q_comb_out: np.ndarray,    # (Nz, Ny, Nx) [W/m³] (overwritten)
    # Phase 23 chemistry-family scalars (biomass defaults preserve
    # bit-exact behaviour per Rule #17).
    s_stoich: float = S_STOICH,
    hoc_J:    float = HOC_J,
    a_comb:   float = A_COMB,
    e_comb:   float = E_COMB,
    c_ebu:    float = C_EBU,
) -> None:
    """Compute combustion rate ω and gas-phase heat release Q in place.

    Returns ω as min(ω_chem, ω_mix).  Caller may further bound ω by
    an O₂-supply rate (see :func:`step_o2_supply_rate`).

    Rate-only kernel — does NOT update species or T_g.  Used by isolation
    tests in ``tests/outdoor/test_3d_components.py``; production main loop
    uses one of the operator-split ODE closures dispatched from
    :mod:`chemistry_closures`.
    """
    Nz, Ny, Nx = rho.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Yf = Y_fuel[k, j, i]
                if Yf <= 0.0:
                    omega_out[k, j, i] = 0.0
                    Q_comb_out[k, j, i] = 0.0
                    continue
                YO2 = Y_O2[k, j, i]
                if YO2 <= 0.0:
                    omega_out[k, j, i] = 0.0
                    Q_comb_out[k, j, i] = 0.0
                    continue

                rho_i = rho[k, j, i]
                T = T_g[k, j, i]
                if T <= 0.0:
                    omega_out[k, j, i] = 0.0
                    Q_comb_out[k, j, i] = 0.0
                    continue

                # Chemistry rate (Arrhenius).
                k_chem = a_comb * math.exp(-e_comb / (_R_GAS * T))
                omega_chem = rho_i * k_chem * Yf * YO2     # [kg/m³/s]

                # Mixing rate (Magnussen-Hjertager EDM); τ_mix = +∞ → laminar.
                tau = tau_mix[k, j, i]
                if tau <= 0.0 or tau >= 1.0e30:
                    omega_mix = 1.0e30
                else:
                    Y_O2_avail = YO2 / s_stoich
                    Y_lim = Yf if Yf < Y_O2_avail else Y_O2_avail
                    omega_mix = c_ebu * rho_i * Y_lim / tau

                omega = omega_mix if omega_mix < omega_chem else omega_chem

                omega_out[k, j, i] = omega
                Q_comb_out[k, j, i] = omega * hoc_J * (1.0 - chi_rad)


# ── WFDS-pure T_g pin (Phase 14y O5) ─────────────────────────────────────────
# Force T_g ≥ T_FLAME_PIN in flame_body cells.  Replaces the bootstrap
# heat-source mechanism with an explicit-conservation-aware temperature pin.
# The pin energy is computed and tracked as a diagnostic
# (pin_energy_J_per_m3 * cell_volume = J added to gas per cell per step).
#
# Lit anchor: Mell et al. (2007) IJWF 16:1 §3.4 WFDS subgrid HRR distribution
# treats the gas at T_flame in the flame zone, not at cell-averaged T.

T_FLAME_PIN = 1100.0   # [K] cell-averaged flame T at coarse grid.  Drysdale
                       # 2011 §1.2.3 grass flame T = 1500K, but cell-averaged
                       # T over a 10cm cell that contains a ~mm flame layer
                       # plus surrounding cool gas is ~800-1200K (Mell 2007
                       # WFDS subgrid HRR distribution).  1100K is mid-range.


@njit(cache=True, parallel=True)
def apply_t_g_pin(
    T_g: np.ndarray,            # (Nz, Ny, Nx) [K] updated in place
    rho: np.ndarray,            # (Nz, Ny, Nx) [kg/m³]
    flame_body_mask: np.ndarray,# (Nz, Ny, Nx) bool
    cp_g: float,
    T_pin: float,
    pin_energy_per_m3_out: np.ndarray,  # (Nz, Ny, Nx) [J/m³] added energy
) -> None:
    """Force T_g ≥ T_pin in flame_body cells; record pin energy added.

    pin_energy_per_m3 = ρ·c_p·(T_pin − T_g_old) when T_g_old < T_pin else 0.
    """
    Nz, Ny, Nx = rho.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if flame_body_mask[k, j, i] and T_g[k, j, i] < T_pin:
                    pin_energy_per_m3_out[k, j, i] = (
                        rho[k, j, i] * cp_g * (T_pin - T_g[k, j, i])
                    )
                    T_g[k, j, i] = T_pin
                else:
                    pin_energy_per_m3_out[k, j, i] = 0.0


@njit(cache=True, parallel=True)
def step_o2_supply_rate(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    u: np.ndarray, v: np.ndarray, w: np.ndarray,  # gas velocity components
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-] transported O₂ mass fraction
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    omega_O2_out: np.ndarray,  # (Nz, Ny, Nx) [kg fuel/m³/s] (overwritten)
    # Phase 23: chemistry-family scalar (biomass default preserves
    # bit-exact behaviour per Rule #17).
    s_stoich: float = S_STOICH,
) -> None:
    """Compute per-cell combustion rate limit set by O₂ mass-flux supply.

    For each interior cell, sums the positive (incoming) O₂ mass-flux
    contributions across all 6 faces using a first-order upwind scheme:

        ṁ_O₂_in [kg/m³/s] = Σ_faces max(0, ρ_up · u_face) · Y_O2_up · 1/Δ

    The fuel-equivalent supply rate is ṁ_O₂_in / s_stoich.  A cell with
    no inflow has ω_O2 = 0 (no combustion possible without fresh O₂).
    A cell with high inflow has ω_O2 ≫ ω_chem so the chemistry branch
    is the rate limit (no spurious throttle on well-supplied cells).

    Boundary cells (i=0, i=Nx-1, etc.) get a partial sum from internal
    faces only; they default to "infinite supply" via the fill before
    this kernel is called, so combustion in pilot/inlet zones is
    governed by chemistry/mixing as before.

    Reference: Spalding (1971) Combust. Sci. Tech. 4:43; Pruyn et al.
    (2018) Combust. Flame 187:182.
    """
    Nz, Ny, Nx = rho.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    for k in prange(1, Nz - 1):
        inv_dz = 1.0 / dz_arr[k]   # per-cell z-spacing (Phase 14g)
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                m_in = 0.0
                # x-minus face: between cell (i-1) and (i).  Flow into
                # cell i if u_face > 0; upwind cell is (i-1).
                u_face = 0.5 * (u[k, j, i - 1] + u[k, j, i])
                if u_face > 0.0:
                    m_in += rho[k, j, i - 1] * u_face * Y_O2[k, j, i - 1] * inv_dx
                # x-plus face: between (i) and (i+1).  Flow into i if
                # u_face < 0; upwind cell is (i+1).
                u_face = 0.5 * (u[k, j, i] + u[k, j, i + 1])
                if u_face < 0.0:
                    m_in += rho[k, j, i + 1] * (-u_face) * Y_O2[k, j, i + 1] * inv_dx
                # y-minus
                v_face = 0.5 * (v[k, j - 1, i] + v[k, j, i])
                if v_face > 0.0:
                    m_in += rho[k, j - 1, i] * v_face * Y_O2[k, j - 1, i] * inv_dy
                # y-plus
                v_face = 0.5 * (v[k, j, i] + v[k, j + 1, i])
                if v_face < 0.0:
                    m_in += rho[k, j + 1, i] * (-v_face) * Y_O2[k, j + 1, i] * inv_dy
                # z-minus
                w_face = 0.5 * (w[k - 1, j, i] + w[k, j, i])
                if w_face > 0.0:
                    m_in += rho[k - 1, j, i] * w_face * Y_O2[k - 1, j, i] * inv_dz
                # z-plus
                w_face = 0.5 * (w[k, j, i] + w[k + 1, j, i])
                if w_face < 0.0:
                    m_in += rho[k + 1, j, i] * (-w_face) * Y_O2[k + 1, j, i] * inv_dz
                # ω_O2_supply = ṁ_O2_in / s_stoich
                omega_O2_out[k, j, i] = m_in / s_stoich
