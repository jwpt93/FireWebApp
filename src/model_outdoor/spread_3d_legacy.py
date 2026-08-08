"""3D Boussinesq momentum + energy fire spread — Phase 12 (LEGACY).

DEPRECATED 2026-04-24: Renamed from spread_3d.py to spread_3d_legacy.py
during the bottom-up Phase 13 rebuild.  This file is preserved for
historical reference only.  Zero callsites; never validated against
EXP data.  See plans/a-is-what-we-snug-haven.md and the new
spread_3d.py for the rebuilt model.

Original docstring follows.

---

Extends the 2D (x,z) Phase 11 solver with a y-dimension to resolve the
breakup of 2D coherent convection rolls into 3D structures.  The 2D model
overpredicts ROS by ~2× (Morvan 2007, Linn 2010); the 3D model should
eliminate this structural limitation.

Grid: (Nz, Ny, Nx) — z vertical, y cross-wind (into page), x spread.
BCs in y: symmetry at y=0 (∂/∂y = 0), wall at y=L_y (v=0).

References:
  Morvan & Dupuy (2001) Combust. Flame 127:1981 — porous drag
  Linn et al. (2002) IJWF 11:233 — FIRETEC, 3D vs 2D
  Smagorinsky (1963) Mon. Weather Rev. 91:99 — sub-grid model
  Lilly (1967) NCAR MS 67-2 — C_s = 0.17
  Chorin (1967) J. Comput. Phys. 2:12 — pressure projection
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from model.io.text_input import load_text_input, RomInputs
from model_outdoor.fuel_element import run_outdoor_element
from model_outdoor.boundary import (
    byram_flame_length,
    flame_tilt_angle,
    midflame_wind_speed,
    wind_profile_in_bed,
)
from model_outdoor.config import outdoor_env_from_dict
from model_outdoor.spread import (
    SpreadConfig,
    SpreadResult,
    _build_poisson_matrix_3d,
    _RHO_GAS, _CP_GAS, _MU_GAS, _K_GAS, _PR_GAS, _RHO_PARTICLE,
)


def run_3d_momentum_spread(
    ri_or_path: Union[Path, str, RomInputs],
    spread_cfg: SpreadConfig,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
    pde_dx: float = 0.02,
    pde_domain_m: float = 10.0,
    n_z_bed: int = 4,
    n_z_buffer: int = 2,
    n_y: int = 4,
    variable_density: bool = False,
    low_mach: bool = False,
) -> SpreadResult:
    """3D Boussinesq momentum + energy — Phase 12.

    Parameters
    ----------
    n_y : int
        Number of cells in the y-direction (cross-wind).
        dy = h_bed / n_z_bed (matches vertical resolution).
        BCs: symmetry at y=0, wall at y = n_y × dy.
    """
    # ── Parse deck ───────────────────────────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri_base = load_text_input(Path(ri_or_path))
    else:
        ri_base = copy.deepcopy(ri_or_path)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    # ── Source element ───────────────────────────────────────────────
    _ri_Tamb = ri_base.Tamb
    ri_base.Tamb = spread_cfg.T_gas_spread_K
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_Tamb
    signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    src_t = np.asarray(signals_src.t, dtype=float)
    src_hrrpua = np.asarray(signals_src.hrrpua, dtype=float)
    _pk = int(np.argmax(src_hrrpua))
    _plat = float(src_hrrpua[_pk]) * 0.8
    src_hrrpua = src_hrrpua.copy()
    src_hrrpua[_pk:] = np.maximum(src_hrrpua[_pk:], _plat)

    # ── Physical parameters ──────────────────────────────────────────
    T_amb = outdoor_cfg.ambient_T_K if hasattr(outdoor_cfg, 'ambient_T_K') else 300.0
    T_ign = T_amb + 300.0
    _g = 9.81
    h_bed = outdoor_cfg.fuel_depth_m
    rho_bulk = outdoor_cfg.bulk_density_kg_m3
    cp_s = 1300.0; eps_s = 0.90; sigma_sb = 5.67e-8
    U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)

    rho0 = _RHO_GAS; cp_g = _CP_GAS
    nu = _MU_GAS / rho0
    alpha_th = nu / _PR_GAS
    _C_D_drag = 1.0
    _C_smag = 0.17
    _Pr_t = 0.5

    _HRRPUA_W = float(np.max(src_hrrpua)) * 1000.0 * 0.8
    _chi_rad = spread_cfg.chi_rad_spread
    T_flame_adiabatic = 1473.0
    _U_char = max(U_mf, math.sqrt(_g * h_bed * 0.5))
    _dT_gas = (1.0 - _chi_rad) * _HRRPUA_W / max(rho0 * cp_g * _U_char, 1.0)
    T_flame = min(300.0 + _dT_gas, T_flame_adiabatic)
    _hoc_J = 14900.0 * 1000.0

    # Per-cell Arrhenius (same as 2D solver).
    # TGA kinetics — hemi + cellulose + lignin (same as 2D solver).
    _R_gas = 8.314
    _A_hemi = 9.71e11; _E_hemi = 139800.0   # Orfão (1999) xylan
    _A_cell = 2.07e14; _E_cell = 178700.0   # Orfão (1999) cellulose
    _A_lign = 2.59e1;  _E_lign = 60800.0    # Orfão (1999) lignin
    _f_lignin = float(getattr(ri_base, 'seq_mr_frac0', 0) or 0.0)
    _f_hemi = 0.15
    _f_cell = max(0.0, 1.0 - _f_hemi - _f_lignin)
    _flame_vf = getattr(ri_base, 'flame_view_factor', None)
    if _flame_vf is None:
        _flame_vf = ri_base.fuel_overrides.get('flame_view_factor', 0.40)
    _flame_vf = float(_flame_vf)

    sigma_sav = outdoor_cfg.sav_ratio_1_m
    d_p = 4.0 / sigma_sav
    beta = rho_bulk / _RHO_PARTICLE
    a_v = sigma_sav * beta
    _Re_p = max(rho0 * max(U_mf, 0.1) * d_p / _MU_GAS, 0.1)
    _Nu_p = 2.0 + 0.6 * _Re_p**0.5 * _PR_GAS**(1.0 / 3.0)
    h_p = _Nu_p * _K_GAS / d_p
    hp_av = h_p * a_v

    chi_rad = _chi_rad
    _w_0 = rho_bulk * h_bed   # [kg/m²] fuel load
    _hoc_J = 14900.0 * 1000.0  # [J/kg]
    L_f = byram_flame_length(_HRRPUA_W, h_bed)
    theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f, outdoor_cfg.terrain)

    # ── Grid (Nz, Ny, Nx) ───────────────────────────────────────────
    Nx = max(10, int(math.ceil(pde_domain_m / pde_dx)))
    dx = pde_domain_m / Nx
    Nz = n_z_bed + n_z_buffer
    dz = h_bed / max(n_z_bed, 1)
    Ny = n_y
    dy = dz   # match vertical resolution
    x_mid = np.linspace(0.5 * dx, pde_domain_m - 0.5 * dx, Nx)
    # Solid thermal capacity including moisture evaporation energy.
    _M_f = outdoor_cfg.initial_moisture_frac if hasattr(outdoor_cfg, 'initial_moisture_frac') else 0.0
    _L_v = 2257000.0   # [J/kg]
    _cp_eff = cp_s + _M_f * _L_v / max(T_ign - T_amb, 1.0)
    C_s = rho_bulk * _cp_eff * dz
    _f_abs = 1.0 - math.exp(-a_v * dz)

    # Inlet wind profile (Cionco within bed, free-stream above).
    u_inlet = np.zeros(Nz)
    for k in range(Nz):
        u_inlet[k] = wind_profile_in_bed(dz * (k + 0.5), h_bed, U_mf) \
            if dz * (k + 0.5) <= h_bed else U_mf

    # ── State arrays (Nz, Ny, Nx) ───────────────────────────────────
    vel_u = np.zeros((Nz, Ny, Nx))   # x-velocity
    vel_v = np.zeros((Nz, Ny, Nx))   # y-velocity
    vel_w = np.zeros((Nz, Ny, Nx))   # z-velocity
    T_g = np.full((Nz, Ny, Nx), T_amb)
    T_s = np.full((Nz, Ny, Nx), T_amb)

    # Initial v-perturbation to seed 3D instability.
    # The 3D solver does NOT use the turbulent inflow BC (which is for
    # the 2D solver to compensate for the missing y-dimension).  The 3D
    # already breaks coherent rolls via the resolved y-flow.  A small
    # initial perturbation is sufficient to seed the instability.
    _rng = np.random.default_rng(42)
    for k in range(Nz):
        vel_u[k, :, :] = u_inlet[k]
    _pert_amp = max(U_mf, 0.01) * 0.01
    vel_v += _rng.uniform(-_pert_amp, _pert_amp, vel_v.shape)

    # Fuel mass per POOL per layer — sequential depletion.
    _m_total = rho_bulk * dz
    _m_hemi = np.full((n_z_bed, Nx), _f_hemi * _m_total)
    _m_cell = np.full((n_z_bed, Nx), _f_cell * _m_total)
    _m_lign = np.full((n_z_bed, Nx), _f_lignin * _m_total)
    _m_fuel_min = 1e-6 * _m_total

    # Source: 0.5m burning zone, all y-cells, all fuel layers.
    _n_source = max(1, int(0.5 / dx))
    _col_burning = np.zeros(Nx, dtype=bool)
    _col_burning[:_n_source] = True
    for k in range(n_z_bed):
        T_s[k, :, :_n_source] = T_ign + 100.0

    x_front = float(x_mid[min(_n_source - 1, Nx - 1)])
    front_history_t: List[float] = [0.0]
    front_history_x: List[float] = [x_front]

    # Per-cell Arrhenius — no schedule needed.
    _t_ign_col = np.full(Nx, np.inf)
    _t_ign_col[:_n_source] = 0.0

    # ── Poisson (3D) ─────────────────────────────────────────────────
    _, _poi_lu = _build_poisson_matrix_3d(Nz, Ny, Nx, dx, dy, dz)

    # ── Timestep ─────────────────────────────────────────────────────
    _Q_est = (1.0 - _chi_rad) * _HRRPUA_W / h_bed   # estimate for timestep
    _w_est = math.sqrt(_g * h_bed * max(_Q_est / max(hp_av, 1.0), 1.0) / T_amb)
    _u_est = max(U_mf, _w_est, 0.1)
    _dt_drag = 2.0 / (_C_D_drag * a_v * _u_est) if a_v > 0 else 1.0
    dt = max(min(0.3 * dx / max(U_mf, 0.01),
                 0.3 * dz / max(_w_est, 0.01),
                 0.3 * dy / max(_w_est, 0.01),
                 0.2 * min(dx, dy, dz)**2 / max(nu, 1e-8),
                 0.5 * _dt_drag,
                 0.1), 1e-6)

    # ── Helper: upwind advection for 3D scalar field ─────────────────
    def _upwind_3d(phi, u, v, w):
        """Sign-split upwind advection of scalar phi by (u,v,w)."""
        adv = np.zeros_like(phi)
        # x
        up = np.maximum(u, 0.0); un = np.minimum(u, 0.0)
        adv[:, :, 1:] -= up[:, :, 1:] * (phi[:, :, 1:] - phi[:, :, :-1]) / dx
        adv[:, :, :-1] -= un[:, :, :-1] * (phi[:, :, 1:] - phi[:, :, :-1]) / dx
        # y
        vp = np.maximum(v, 0.0); vn = np.minimum(v, 0.0)
        adv[:, 1:, :] -= vp[:, 1:, :] * (phi[:, 1:, :] - phi[:, :-1, :]) / dy
        adv[:, :-1, :] -= vn[:, :-1, :] * (phi[:, 1:, :] - phi[:, :-1, :]) / dy
        # z
        wp = np.maximum(w, 0.0); wn = np.minimum(w, 0.0)
        adv[1:, :, :] -= wp[1:, :, :] * (phi[1:, :, :] - phi[:-1, :, :]) / dz
        adv[:-1, :, :] -= wn[:-1, :, :] * (phi[1:, :, :] - phi[:-1, :, :]) / dz
        return adv

    def _diffuse_3d(phi, coeff):
        """Central-difference diffusion of scalar phi."""
        diff = np.zeros_like(phi)
        if isinstance(coeff, np.ndarray):
            diff[:, :, 1:-1] += coeff[:, :, 1:-1] * (phi[:, :, :-2] - 2*phi[:, :, 1:-1] + phi[:, :, 2:]) / dx**2
            diff[:, 1:-1, :] += coeff[:, 1:-1, :] * (phi[:, :-2, :] - 2*phi[:, 1:-1, :] + phi[:, 2:, :]) / dy**2
            diff[1:-1, :, :] += coeff[1:-1, :, :] * (phi[:-2, :, :] - 2*phi[1:-1, :, :] + phi[2:, :, :]) / dz**2
        else:
            diff[:, :, 1:-1] += coeff * (phi[:, :, :-2] - 2*phi[:, :, 1:-1] + phi[:, :, 2:]) / dx**2
            diff[:, 1:-1, :] += coeff * (phi[:, :-2, :] - 2*phi[:, 1:-1, :] + phi[:, 2:, :]) / dy**2
            diff[1:-1, :, :] += coeff * (phi[:-2, :, :] - 2*phi[1:-1, :, :] + phi[2:, :, :]) / dz**2
        return diff

    # ── Time loop ────────────────────────────────────────────────────
    _step = 0
    _print_every = max(1, int(10.0 / max(dt, 1e-6)))
    t = 0.0
    while t < max_wall_time_s:
        _step += 1
        if _step % _print_every == 0:
            _n_burn = int(np.sum(_col_burning))
            print(f"  t={t:.1f}s  front={x_front:.2f}m  burning={_n_burn}  "
                  f"dt={dt:.5f}s", flush=True)
        # Source cells maintain T at ignition (external heat, TGA-appropriate).
        for k in range(n_z_bed):
            T_s[k][:, :_n_source] = np.maximum(T_s[k][:, :_n_source], T_ign)

        # ── Smagorinsky ──────────────────────────────────────────
        _Delta = (dx * dy * dz) ** (1.0 / 3.0)
        # Strain rate components (central differences).
        _S11 = np.zeros_like(vel_u)
        _S22 = np.zeros_like(vel_v)
        _S33 = np.zeros_like(vel_w)
        _S12 = np.zeros_like(vel_u)
        _S13 = np.zeros_like(vel_u)
        _S23 = np.zeros_like(vel_u)
        _S11[:, :, 1:-1] = (vel_u[:, :, 2:] - vel_u[:, :, :-2]) / (2*dx)
        _S22[:, 1:-1, :] = (vel_v[:, 2:, :] - vel_v[:, :-2, :]) / (2*dy)
        _S33[1:-1, :, :] = (vel_w[2:, :, :] - vel_w[:-2, :, :]) / (2*dz)
        _S12[:, 1:-1, 1:-1] = 0.5 * (
            (vel_u[:, 2:, 1:-1] - vel_u[:, :-2, 1:-1]) / (2*dy) +
            (vel_v[:, 1:-1, 2:] - vel_v[:, 1:-1, :-2]) / (2*dx))
        _S13[1:-1, :, 1:-1] = 0.5 * (
            (vel_u[2:, :, 1:-1] - vel_u[:-2, :, 1:-1]) / (2*dz) +
            (vel_w[1:-1, :, 2:] - vel_w[1:-1, :, :-2]) / (2*dx))
        _S23[1:-1, 1:-1, :] = 0.5 * (
            (vel_v[2:, 1:-1, :] - vel_v[:-2, 1:-1, :]) / (2*dz) +
            (vel_w[1:-1, 2:, :] - vel_w[1:-1, :-2, :]) / (2*dy))
        _S_mag = np.sqrt(2*(_S11**2 + _S22**2 + _S33**2 +
                            2*(_S12**2 + _S13**2 + _S23**2)))
        _nu_t = (_C_smag * _Delta)**2 * _S_mag
        _nu_eff = nu + _nu_t

        # ── Momentum (advection + diffusion + drag + buoyancy) ───
        du = _upwind_3d(vel_u, vel_u, vel_v, vel_w) + _diffuse_3d(vel_u, _nu_eff)
        dv = _upwind_3d(vel_v, vel_u, vel_v, vel_w) + _diffuse_3d(vel_v, _nu_eff)
        dw = _upwind_3d(vel_w, vel_u, vel_v, vel_w) + _diffuse_3d(vel_w, _nu_eff)

        # Buoyancy (z-direction only).
        buoy = _g * (T_g - T_amb) / max(T_amb, 1.0)

        # Porous drag on perturbation velocity (fuel bed only).
        _u_pert = vel_u[:n_z_bed] - u_inlet[:n_z_bed, np.newaxis, np.newaxis]
        _v_pert = vel_v[:n_z_bed]
        _w_pert = vel_w[:n_z_bed]
        _sp = np.sqrt(_u_pert**2 + _v_pert**2 + _w_pert**2)
        _dc = _C_D_drag * a_v * 0.5
        du[:n_z_bed] -= _dc * _sp * _u_pert
        dv[:n_z_bed] -= _dc * _sp * _v_pert
        dw[:n_z_bed] -= _dc * _sp * _w_pert

        u_star = vel_u + dt * du
        v_star = vel_v + dt * dv
        w_star = vel_w + dt * (dw + buoy)

        # Velocity BCs.
        for j in range(Ny):
            u_star[:, j, 0] = u_inlet
        u_star[0, :, :] = 0.0
        u_star[-1, :, :] = u_star[-2, :, :]  # top: Neumann
        # y: periodic.
        v_star[:, 0, :] = v_star[:, -2, :]   # periodic
        v_star[:, -1, :] = v_star[:, 1, :]   # periodic
        u_star[:, 0, :] = u_star[:, -2, :]
        u_star[:, -1, :] = u_star[:, 1, :]
        w_star[:, 0, :] = w_star[:, -2, :]
        w_star[:, -1, :] = w_star[:, 1, :]
        # z: no-penetration bottom/top.
        w_star[0, :, :] = 0.0
        w_star[-1, :, :] = 0.0

        # ── Pressure projection (approximate variable-density) ────
        if variable_density:
            _rho_g = 101325.0 / (287.0 * np.maximum(T_g, T_amb))
        div = np.zeros((Nz, Ny, Nx))
        div[:, :, :-1] += (u_star[:, :, 1:] - u_star[:, :, :-1]) / dx
        div[:, :-1, :] += (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
        div[:-1, :, :] += (w_star[1:, :, :] - w_star[:-1, :, :]) / dz
        # Low-Mach: ∇·u = (1/T) DT/Dt (Rehm & Baum 1978).
        if low_mach:
            _Q_exp = np.zeros((Nz, Ny, Nx))
            _T_safe = np.maximum(T_g, T_amb)
            if variable_density:
                _Q_exp[:n_z_bed] = hp_av * (T_s[:n_z_bed] - T_g[:n_z_bed]) / (_rho_g[:n_z_bed] * cp_g)
                for k in range(n_z_bed):
                    _sl_q = _Q_exp[k]
                    _hd = np.maximum(T_flame_adiabatic - T_g[k], 0.0) / \
                        max(T_flame_adiabatic - T_amb, 1.0)
                    _sl_q[:, _col_burning] += _Q_est / (_rho_g[k][:, _col_burning] * cp_g) * _hd[:, _col_burning]
            else:
                _Q_exp[:n_z_bed] = hp_av * (T_s[:n_z_bed] - T_g[:n_z_bed]) / (rho0 * cp_g)
                for k in range(n_z_bed):
                    _sl_q = _Q_exp[k]
                    _hd = np.maximum(T_flame_adiabatic - T_g[k], 0.0) / \
                        max(T_flame_adiabatic - T_amb, 1.0)
                    _sl_q[:, _col_burning] += _Q_est / (rho0 * cp_g) * _hd[:, _col_burning]
            _Q_exp /= _T_safe
            div -= _Q_exp
        rhs = (rho0 / dt) * div.ravel()
        rhs[-1] = 0.0
        P_flat = _poi_lu.solve(rhs)
        P = P_flat.reshape(Nz, Ny, Nx)

        dPdx = np.zeros_like(vel_u)
        dPdx[:, :, 1:] = (P[:, :, 1:] - P[:, :, :-1]) / dx
        dPdy = np.zeros_like(vel_v)
        dPdy[:, 1:, :] = (P[:, 1:, :] - P[:, :-1, :]) / dy
        dPdz = np.zeros_like(vel_w)
        dPdz[1:, :, :] = (P[1:, :, :] - P[:-1, :, :]) / dz

        # Velocity correction: constant ρ₀ (see 2D solver comment).
        vel_u = u_star - (dt / rho0) * dPdx
        vel_v = v_star - (dt / rho0) * dPdy
        vel_w = w_star - (dt / rho0) * dPdz

        # Re-apply BCs after projection.
        for j in range(Ny):
            vel_u[:, j, 0] = u_inlet
        vel_u[0, :, :] = 0.0
        # y: periodic
        vel_v[:, 0, :] = vel_v[:, -2, :]
        vel_v[:, -1, :] = vel_v[:, 1, :]
        vel_u[:, 0, :] = vel_u[:, -2, :]
        vel_u[:, -1, :] = vel_u[:, 1, :]
        vel_w[:, 0, :] = vel_w[:, -2, :]
        vel_w[:, -1, :] = vel_w[:, 1, :]
        # z
        vel_w[0, :, :] = 0.0
        vel_w[-1, :, :] = 0.0

        # ── Energy ───────────────────────────────────────────────
        _alpha_eff = alpha_th + _nu_t / _Pr_t
        dTg = _upwind_3d(T_g, vel_u, vel_v, vel_w) + _diffuse_3d(T_g, _alpha_eff)
        # Gas-solid coupling (convective only).
        if variable_density:
            _rho_loc = 101325.0 / (287.0 * np.maximum(T_g, T_amb))
            _rcp = _rho_loc * cp_g
            dTg[:n_z_bed] -= hp_av * (T_g[:n_z_bed] - T_s[:n_z_bed]) / _rcp[:n_z_bed]
        else:
            dTg[:n_z_bed] -= hp_av * (T_g[:n_z_bed] - T_s[:n_z_bed]) / (rho0 * cp_g)
        # Per-cell Arrhenius — sequential pool depletion.
        _T_s_ymean = np.mean(T_s[:n_z_bed], axis=1)
        _T_safe = np.maximum(_T_s_ymean, T_amb)
        _mdot_hemi = _A_hemi * np.exp(-_E_hemi / (_R_gas * _T_safe)) * _m_hemi
        _mdot_cell = _A_cell * np.exp(-_E_cell / (_R_gas * _T_safe)) * _m_cell
        _mdot_lign = _A_lign * np.exp(-_E_lign / (_R_gas * _T_safe)) * _m_lign
        _m_dot_py = _mdot_hemi + _mdot_cell + _mdot_lign
        _Q_gas = _m_dot_py * _hoc_J * (1.0 - chi_rad) / h_bed   # [W/m³]
        for k in range(n_z_bed):
            _sl = dTg[k]
            _hd = np.maximum(T_flame_adiabatic - T_g[k], 0.0) / \
                max(T_flame_adiabatic - T_amb, 1.0)
            _Q_k = _Q_gas[k, :] * _hd  # (Ny, Nx) broadcast via _hd
            # _Q_k is (Nx,), _hd is (Ny, Nx), need to broadcast
            _Q_k_2d = _Q_gas[k, :][np.newaxis, :] * _hd  # (Ny, Nx)
            if variable_density:
                _sl[:, _col_burning] += _Q_k_2d[:, _col_burning] / _rcp[k][:, _col_burning]
            else:
                _sl[:, _col_burning] += _Q_k_2d[:, _col_burning] / (rho0 * cp_g)

        T_g += dTg * dt
        np.clip(T_g, T_amb, T_flame_adiabatic, out=T_g)
        T_g[-1, :, :] = T_amb   # top BC
        T_g[0, :, :] = T_g[1, :, :]  # ground: zero gradient
        # y: periodic
        T_g[:, 0, :] = T_g[:, -2, :]
        T_g[:, -1, :] = T_g[:, 1, :]

        # ── Solid ────────────────────────────────────────────────
        _Tl = 0.5 * (T_g[:n_z_bed] + T_s[:n_z_bed])
        _hr = 4.0 * eps_s * sigma_sb * _Tl**3
        _hpe = (h_p + _hr) * a_v
        q_gs = _hpe * dz * (T_g[:n_z_bed] - T_s[:n_z_bed])
        q_loss = eps_s * sigma_sb * (T_s[:n_z_bed]**4 - T_amb**4) * a_v * dz

        # Slab radiation (same as 2D, applied uniformly in y).
        q_rad = np.zeros((n_z_bed, Ny, Nx))
        _bc = np.where(_col_burning)[0]
        if len(_bc) > 0 and L_f > 0:
            _lb = int(_bc[-1])
            _fe = x_mid[_lb] + 0.5 * dx
            _be = x_mid[_bc[0]] - 0.5 * dx
            _is = _lb + 1
            if _is < Nx:
                _dn = np.maximum(x_mid[_is:] - _fe, 0.5 * dx)
                _df = x_mid[_is:] - _be
                _st = math.sin(theta_tilt)
                _rn = np.maximum(_dn - L_f * _st, 1e-3)
                _rf = np.maximum(_df - L_f * _st, 1e-3)
                _Fn = 0.5 * (1.0 - _rn / np.sqrt(L_f**2 + _rn**2))
                _Ff = 0.5 * (1.0 - _rf / np.sqrt(L_f**2 + _rf**2))
                _Fs = np.maximum(_Fn - _Ff, 0.0)
                _q_inc = chi_rad * _HRRPUA_W * _Fs
                _transmit = 1.0 - _f_abs
                for k in range(n_z_bed - 1, -1, -1):
                    # Broadcast over y.
                    q_rad[k, :, _is:] = _q_inc[np.newaxis, :] * _f_abs
                    _q_inc = _q_inc * _transmit

        # Flame feedback — upstream neighbor (same as 2D).
        # Flame feedback from own pyrolysis only (same as 2D).
        _q_flame_back = chi_rad * _m_dot_py * _hoc_J * _flame_vf
        _q_fb_3d = np.zeros_like(T_s[:n_z_bed])
        for k in range(n_z_bed):
            _q_fb_3d[k, :, :] = _q_flame_back[k, :][np.newaxis, :]  # broadcast y

        dTs = np.zeros_like(T_s)
        dTs[:n_z_bed] = (q_gs + q_rad + _q_fb_3d - q_loss) / C_s
        T_s += dTs * dt

        # Fuel depletion.
        _m_hemi -= _mdot_hemi * dt
        _m_cell -= _mdot_cell * dt
        _m_lign -= _mdot_lign * dt
        np.clip(_m_hemi, 0.0, None, out=_m_hemi)
        np.clip(_m_cell, 0.0, None, out=_m_cell)
        np.clip(_m_lign, 0.0, None, out=_m_lign)
        t += dt

        # ── Ignition (check y-averaged top-layer T_s) ────────────
        # Column ignites when the y-averaged top-layer T_s exceeds T_ign.
        _Ttop_ymean = np.mean(T_s[n_z_bed - 1], axis=0)  # (Nx,)
        newly = (_Ttop_ymean >= T_ign) & (~_col_burning)
        if np.any(newly):
            _col_burning |= newly
            _nc = np.where(newly)[0]
            _t_ign_col[_nc] = t
            _nf = float(x_mid[_nc[-1]])
            if _nf > x_front:
                x_front = _nf
                front_history_t.append(t)
                front_history_x.append(x_front)
                # Dynamic flame length (Byram 1959).
                if len(front_history_t) >= 3:
                    _ros_cur = (front_history_x[-1] - front_history_x[-2]) / \
                        max(front_history_t[-1] - front_history_t[-2], 1e-8)
                    _I_B_kW = _hoc_J * _w_0 * _ros_cur / 1000.0
                    if _I_B_kW > 0:
                        L_f = 0.0475 * _I_B_kW ** 0.493
                        theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f,
                                                      outdoor_cfg.terrain)

        # Burnout (mass-based).
        _col_fuel_sum = np.sum(_m_hemi + _m_cell + _m_lign, axis=0)
        _burned_out = _col_burning & (_col_fuel_sum < n_z_bed * _m_fuel_min)
        if np.any(_burned_out):
            _col_burning[_burned_out] = False

        if x_front > pde_domain_m * 0.9:
            break

    # ── ROS ──────────────────────────────────────────────────────────
    ft = np.array(front_history_t)
    fx = np.array(front_history_x)
    if len(ft) >= 3:
        nh = len(ft) // 2
        ros_m_s = (fx[-1] - fx[nh]) / max(ft[-1] - ft[nh], 1e-8)
    elif len(ft) >= 2:
        ros_m_s = (fx[-1] - fx[0]) / max(ft[-1] - ft[0], 1e-8)
    else:
        ros_m_s = 0.0

    return SpreadResult(
        t_ignition=[0.0], cell_t=[src_t], cell_hrrpua=[src_hrrpua],
        ros_m_s=ros_m_s, n_cells_ignited=int(np.sum(_col_burning)),
        spread_cfg=spread_cfg, n_jump_list=[],
    )
