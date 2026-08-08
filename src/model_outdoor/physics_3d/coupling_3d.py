"""Gas-solid energy coupling (convection + surface radiation losses).

Per-cell volumetric heat exchange:

    q_conv = h_p · a_v · (T_g − T_s)            [W/m³]

with Ranz-Marshall (1952) particle Nusselt number for gas flow over
spheres / cylinders:

    Nu = 2 + 0.6 · Re^0.5 · Pr^(1/3)
    h_p = Nu · k_gas / d_p ;   d_p = 4 / σ_SAV
    a_v = σ_SAV · α_s

Solid energy equation (per unit cell volume):

    ρ_s · c_p,s · α_s · dT_s/dt = q_rad_in + q_conv − q_loss − Q_pyro

where
  q_rad_in  = absorbed radiative flux per cell volume [W/m³]
              (computed externally by radiation_3d, then divided by dz)
  q_loss    = ε · σ · (T_s⁴ − T_amb⁴) · a_v          [W/m³]
              surface radiation from heated solid to ambient
  Q_pyro    = endothermic heat sink from pyrolysis [W/m³]
              (computed externally by pyrolysis_3d)

Gas energy contribution from this module: -q_conv (gas loses heat to
solid where T_g > T_s, gains it back where T_g < T_s).  Heat release
from combustion (Q_comb) is handled separately in combustion_3d.

References:
- Ranz & Marshall (1952) Chem. Eng. Prog. 48:141 — particle heat transfer
- Drysdale (2011) Fire Dynamics, Ch. 2 — gas property values
- Quintiere (2006) — emissivity values for char-laden surfaces
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


# Air properties (Drysdale 2011 Table 2.4).
_MU_GAS = 1.8e-5     # [Pa·s]
_K_GAS  = 0.026      # [W/m/K]
_PR_GAS = 0.7        # [-]

# Solid properties (dry cellulosic, Janssens 1993).
_RHO_SOLID = 500.0    # [kg/m³] particle density
_CP_SOLID  = 1300.0   # [J/kg/K]
_EPS_SOLID = 0.9      # [-] surface emissivity
_SIGMA_SB  = 5.67e-8  # [W/m²/K⁴]

# Phase 14r — ground-contact convective heat loss for k=0 bed cells.
# The bottom face of the bottom-most bed layer sits on cold soil at T_amb.
# DOM handles the radiative ground BC (Marshak ε_w=1.0 since Phase 14q-1),
# but the conductive/convective contact has no closure in the bed energy
# equation.  In wildland-fire DNS (Mell 2007 WFDS, Morvan-Dupuy 2004) the
# ground BC is treated with either a fixed-T soil contact via Fourier
# conduction or a fixed-h convective coefficient.  Without a T_soil(t)
# evolution model, the convective form is the lit-defensible choice.
#
# H_GROUND magnitude: Drysdale 2011 §2.2 — natural convection from a
# horizontal upward-cooling surface gives h ≈ 1.32·(ΔT/L)^(1/4), which
# evaluates to 5–10 W/m²/K for ΔT ~ 300–600 K and bed length L ~ 1 m.
# Soil-conduction equivalent (Drysdale §2.3, dry sand k_soil ≈ 0.27 W/m/K
# with thermal penetration δ ~ √(α t) ~ 3 mm at t = 60 s) gives an
# instantaneous coefficient of order 80 W/m²/K but decays as 1/√t.  Using
# h = 5 W/m²/K is the steady-state lower bound; it activates only the
# bottom (k=0) bed layer and damps low-wind cases where bed advection is
# small relative to conductive losses.
H_GROUND = 5.0  # [W/m²/K]


@njit(cache=True, parallel=True)
def step_gas_solid_coupling(
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K] — overwritten in place
    T_s: np.ndarray,           # (Nz, Ny, Nx) [K] — overwritten in place
    rho: np.ndarray,           # (Nz, Ny, Nx) gas density [kg/m³]
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    alpha_s: np.ndarray,       # (Nz, Ny, Nx) solid volume fraction
    sigma_sav: float,          # [1/m]
    q_rad_in: np.ndarray,      # (Nz, Ny, Nx) [W/m²] absorbed flux per cell
    Q_pyro: np.ndarray,        # (Nz, Ny, Nx) [W/m³] endothermic sink
    Q_comb: np.ndarray,        # (Nz, Ny, Nx) [W/m³] combustion heat to gas
    m_water: np.ndarray,       # (Nz, Ny, Nx) [kg/m³] moisture per cell — updated in place
    L_v: float,                # [J/kg] latent heat of vaporization (water)
    dt: float,
    dz_arr: np.ndarray,   # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    T_amb: float,
    q_loss_enable: bool = True,  # Phase 13.W: disable when q_rad_in is FVM net (includes self-emission)
    h_conv_mult: float = 1.0,    # Phase 15G: scale gas-solid h_p (Ranz-Marshall)
                                  # for sensitivity testing.  1.0 = unmodified.
) -> None:
    """One time step of gas-solid coupling + solid energy + gas Q_comb.

    Updates T_s, T_g, and m_water in place.  Other state arrays read-only.

    Phase 14h: moisture evaporation handled inside this kernel using the
    same heat budget that would otherwise raise T_s.  Energy supplied to
    the solid (q_rad + q_conv - q_loss) is first diverted to vaporize
    water (Frandsen 1971 §4; Albini 1985 §2.3 — wildland-fuel drying is
    driven by both radiative AND convective heating, not radiation alone).
    Pre-14h evap was radiation-only, so opaque dense beds (Cut grass)
    failed to dry their downstream cells and the moisture gate locked
    pyrolysis there indefinitely.

    Energy bookkeeping (per cell, when α_s > 0):
        q_in_solid = q_rad_volumetric + q_conv - q_loss
        q_evap_use = min(max(q_in_solid, 0), m_water * L_v / dt)
        dm_evap = q_evap_use * dt / L_v
        m_water -= dm_evap
        q_residual = q_in_solid - q_evap_use   # heats solid
        dT_s/dt = (q_residual - Q_pyro) / C_s
    Gas still loses q_conv (it supplies that energy regardless of whether
    it ends up in the solid or in latent heat — this preserves the gas
    energy balance unchanged).
    """
    Nz, Ny, Nx = T_g.shape

    if sigma_sav <= 0.0:
        return
    d_p = 4.0 / sigma_sav  # particle diameter

    cp_g = 1100.0  # gas cp; could be passed in to support T-dependent cp later

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                a_s = alpha_s[k, j, i]
                Tg  = T_g[k, j, i]
                Ts  = T_s[k, j, i]
                rho_g = rho[k, j, i]

                # ── Gas-phase combustion heat contribution ──────────
                Tg_new = Tg + Q_comb[k, j, i] / (rho_g * cp_g) * dt

                # ── Solid energy & convective coupling (only where α_s > 0) ──
                if a_s > 0.0:
                    speed = math.sqrt(u[k, j, i]**2 + v[k, j, i]**2 + w[k, j, i]**2)
                    Re = rho_g * speed * d_p / _MU_GAS
                    if Re < 0.1:
                        Re = 0.1   # natural-convection floor
                    Nu = 2.0 + 0.6 * (Re ** 0.5) * (_PR_GAS ** (1.0 / 3.0))
                    h_p = Nu * _K_GAS / d_p * h_conv_mult
                    a_v = sigma_sav * a_s

                    q_conv = h_p * a_v * (Tg_new - Ts)            # W/m³
                    if q_loss_enable:
                        q_loss = _EPS_SOLID * _SIGMA_SB * (Ts**4 - T_amb**4) * a_v
                    else:
                        q_loss = 0.0
                    q_rad_volumetric = q_rad_in[k, j, i] / dz_arr[k]

                    # Phase 14r — ground-contact convective heat loss (k=0 only).
                    # Bottom-face area per cell volume = 1/dz_arr[0]; loss to
                    # cold soil at T_amb via Newton's law cooling.  Activates
                    # only on the bottom-most bed layer.
                    if k == 0:
                        q_loss_ground = H_GROUND * (Ts - T_amb) / dz_arr[0]
                    else:
                        q_loss_ground = 0.0

                    # ── Phase 14h: moisture evaporation (rad+conv driven) ──
                    q_in_solid = q_rad_volumetric + q_conv - q_loss - q_loss_ground
                    mw = m_water[k, j, i]
                    if mw > 0.0 and L_v > 0.0 and dt > 0.0 and q_in_solid > 0.0:
                        q_evap_max = mw * L_v / dt   # cap by water available
                        q_evap_use = q_in_solid if q_in_solid < q_evap_max else q_evap_max
                        dm_evap = q_evap_use * dt / L_v
                        new_mw = mw - dm_evap
                        if new_mw < 0.0:
                            new_mw = 0.0
                        m_water[k, j, i] = new_mw
                        q_residual = q_in_solid - q_evap_use
                    else:
                        q_residual = q_in_solid

                    C_s = _RHO_SOLID * _CP_SOLID * a_s
                    if C_s > 0.0:
                        dTs_dt = (q_residual - Q_pyro[k, j, i]) / C_s
                        Ts_new = Ts + dTs_dt * dt
                    else:
                        Ts_new = Ts

                    # Gas loses q_conv regardless (supplied by gas to solid+water).
                    Tg_new = Tg_new - q_conv / (rho_g * cp_g) * dt

                    T_s[k, j, i] = Ts_new

                T_g[k, j, i] = Tg_new
