"""Tentative-velocity update for the 3D momentum equation (B.3a).

This module advances velocity by all RHS terms EXCEPT the pressure
gradient: advection, viscous diffusion, gravity/buoyancy, and external
volumetric forces (e.g. drag from drag_3d.py).

    u* = u^n + dt · ( -(u·∇)u + ν∇²u + g_buoy + F_ext/ρ )

The projection step (Chorin 1967) that enforces ∇·u^{n+1} = 0 lives in
projection_3d.py (B.3b) and uses ``u*`` as input.

Boussinesq buoyancy (gravity acts in -z; positive z is up):

    g_buoy_z = -g · (T - T_amb) / T_amb

i.e. hot gas rises.  For our low-Mach formulation with EoS ρ=P/(R/M·T)
we apply this approximation in the laminar baseline; full
density-weighted form may be added in a non-Boussinesq variant later.

Advection: first-order upwind on a uniform Cartesian grid.  Diffusion:
central differences (Laplacian) with ν = μ/ρ.

References:
- Chorin (1967) J. Comput. Phys. 2:12 — fractional-step method
- Boussinesq (1903) Théorie analytique de la chaleur
- Patankar (1980) — finite-volume discretisation conventions
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from model_outdoor.physics_3d.muscl_3d import muscl_face_value


_G = 9.81
_MU_GAS = 1.8e-5  # [Pa·s]


@njit(cache=True, parallel=True)
def step_tentative_velocity(
    u: np.ndarray,             # (Nz, Ny, Nx) [m/s] — overwritten in place
    v: np.ndarray,
    w: np.ndarray,
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]
    Fx_ext: np.ndarray,        # (Nz, Ny, Nx) [N/m³] body forces (e.g. drag)
    Fy_ext: np.ndarray,
    Fz_ext: np.ndarray,
    dt: float,
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1 (face k+½)
    d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1 (face k-½)
    T_amb: float,
    u_inlet: np.ndarray,       # (Nz, Ny) [m/s] inlet face u — Phase 14v-bc Way B ghost
    v_inlet: np.ndarray,       # (Nz, Ny) [m/s] inlet face v — Phase 14ap SEM ghost (zero baseline)
    w_inlet: np.ndarray,       # (Nz, Ny) [m/s] inlet face w — Phase 14ap SEM ghost (zero baseline)
) -> None:
    """Advance u, v, w by one tentative step in place.

    Updates u[:], v[:], w[:].  Boundaries are NOT updated here — the
    caller applies BCs (inlet, walls, periodic, edge-loss) before
    and after this call.

    Time integrator: explicit Euler.  Spatial: first-order upwind for
    advection, central for diffusion.  Buoyancy applied to w only.

    Phase 14g: dz is now per-cell (dz_arr) to support non-uniform z.
    The z-derivatives use cell-center distance arrays (d_face_above /
    d_face_below) for correct finite-volume / finite-difference behavior
    on stretched grids.

    Phase 14v-bc: extended back to k=0.  The original 14r-wall extension
    failed because projection's k=0 div was forced to 0 by ghost reflection,
    so momentum's buoyancy at k=0 had no pressure-correction outlet — gave
    spurious downward w domain-wide.  Phase 14v-bc fixes the projection
    side (mirror ghost reflection makes div_z[0] = w[0]/dz_arr[0] respond
    to source/buoyancy), so extending momentum to k=0 is now safe AND
    necessary — without it, no-slip wall friction (the u_below=0 mirror
    ghost terms below) is never applied, since u[0] is no longer pinned
    in `_apply_velocity_bcs`.  Way B: BCs through ghosts only.
    """
    Nz, Ny, Nx = u.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy

    # Pre-allocated outputs to avoid in-place feedback.  We write into
    # a separate array and copy back at the end.  (Numba doesn't let
    # us allocate new arrays inside parallel loops cleanly, so we
    # iterate over interior cells and update with the OLD values.)
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dw = np.zeros_like(w)

    # Phase 14v-bc: full Way B ghost handling.
    #   k=0 (wall):     u_below = v_below = w_below = 0 (no-slip face)
    #   k=Nz-1 (top):   zero-gradient ghost = self
    #   i=0 (inlet):    u_left = u_inlet, v_left = w_left = 0 (uniform inflow)
    #   i=Nx-1 (outlet):zero-gradient ghost = self
    #   y (periodic):   modular indexing
    for k in prange(0, Nz):
        for j in range(Ny):
            jm2 = (j - 2) % Ny
            jm1 = (j - 1) % Ny
            jp1 = (j + 1) % Ny
            jp2 = (j + 2) % Ny
            for i in range(Nx):
                ui = u[k, j, i]; vi = v[k, j, i]; wi = w[k, j, i]
                rho_i = rho[k, j, i]
                nu_i = _MU_GAS / rho_i

                # Boundary ghost values (Way B on-the-fly).  Suffix L = lower
                # neighbor (i-1 / k-1), R = upper (i+1 / k+1).
                # x: inlet face Dirichlet, outlet zero-gradient
                if i == 0:
                    uL_x = u_inlet[k, j]
                    vL_x = v_inlet[k, j]
                    wL_x = w_inlet[k, j]
                else:
                    uL_x = u[k, j, i-1]; vL_x = v[k, j, i-1]; wL_x = w[k, j, i-1]
                if i == Nx - 1:
                    uR_x = ui; vR_x = vi; wR_x = wi
                else:
                    uR_x = u[k, j, i+1]; vR_x = v[k, j, i+1]; wR_x = w[k, j, i+1]
                # z: wall (face=0) at k=0, zero-gradient at k=Nz-1
                if k == 0:
                    uL_z = 0.0; vL_z = 0.0; wL_z = 0.0
                else:
                    uL_z = u[k-1, j, i]; vL_z = v[k-1, j, i]; wL_z = w[k-1, j, i]
                if k == Nz - 1:
                    uR_z = ui; vR_z = vi; wR_z = wi
                else:
                    uR_z = u[k+1, j, i]; vR_z = v[k+1, j, i]; wR_z = w[k+1, j, i]

                # ── MUSCL advection (Phase 14k, replaces 1st-order upwind) ─
                # Conservation form: u_c · (φ_face[i+½] − φ_face[i-½]) / dx,
                # equivalent to u_c · ∂φ/∂x for divergence-free flow (which u
                # is, post-projection from previous step).  φ ∈ {u, v, w}.
                # x-direction faces.  MUSCL only when 4-cell stencil fits;
                # otherwise 1st-order upwind using ghost-aware uL_x / uR_x.
                if 2 <= i <= Nx - 3:
                    u_xpf = muscl_face_value(u[k, j, i-1], ui,
                                             u[k, j, i+1], u[k, j, i+2], ui)
                    u_xmf = muscl_face_value(u[k, j, i-2], u[k, j, i-1],
                                             ui, u[k, j, i+1], ui)
                    v_xpf = muscl_face_value(v[k, j, i-1], vi,
                                             v[k, j, i+1], v[k, j, i+2], ui)
                    v_xmf = muscl_face_value(v[k, j, i-2], v[k, j, i-1],
                                             vi, v[k, j, i+1], ui)
                    w_xpf = muscl_face_value(w[k, j, i-1], wi,
                                             w[k, j, i+1], w[k, j, i+2], ui)
                    w_xmf = muscl_face_value(w[k, j, i-2], w[k, j, i-1],
                                             wi, w[k, j, i+1], ui)
                    dudx = (u_xpf - u_xmf) * inv_dx
                    dvdx = (v_xpf - v_xmf) * inv_dx
                    dwdx = (w_xpf - w_xmf) * inv_dx
                else:
                    if ui >= 0.0:
                        dudx = (ui - uL_x) * inv_dx
                        dvdx = (vi - vL_x) * inv_dx
                        dwdx = (wi - wL_x) * inv_dx
                    else:
                        dudx = (uR_x - ui) * inv_dx
                        dvdx = (vR_x - vi) * inv_dx
                        dwdx = (wR_x - wi) * inv_dx
                # y-direction faces (periodic wrap — MUSCL always available
                # via jm2/jm1/jp1/jp2 modular indices precomputed above).
                u_yp = muscl_face_value(u[k, jm1, i], ui,
                                         u[k, jp1, i], u[k, jp2, i], vi)
                u_ym = muscl_face_value(u[k, jm2, i], u[k, jm1, i],
                                         ui, u[k, jp1, i], vi)
                v_yp = muscl_face_value(v[k, jm1, i], vi,
                                         v[k, jp1, i], v[k, jp2, i], vi)
                v_ym = muscl_face_value(v[k, jm2, i], v[k, jm1, i],
                                         vi, v[k, jp1, i], vi)
                w_yp = muscl_face_value(w[k, jm1, i], wi,
                                         w[k, jp1, i], w[k, jp2, i], vi)
                w_ym = muscl_face_value(w[k, jm2, i], w[k, jm1, i],
                                         wi, w[k, jp1, i], vi)
                dudy = (u_yp - u_ym) * inv_dy
                dvdy = (v_yp - v_ym) * inv_dy
                dwdy = (w_yp - w_ym) * inv_dy
                # z-direction faces (non-uniform spacing).  Phase 14v-bc: full
                # range with wall ghost (=0) at k=0 and zero-grad at k=Nz-1.
                if 2 <= k <= Nz - 3:
                    inv_dz_eff = 1.0 / (0.5 * (d_face_above[k] + d_face_below[k]))
                    u_zpf = muscl_face_value(u[k-1, j, i], ui,
                                             u[k+1, j, i], u[k+2, j, i], wi)
                    u_zmf = muscl_face_value(u[k-2, j, i], u[k-1, j, i],
                                             ui, u[k+1, j, i], wi)
                    v_zpf = muscl_face_value(v[k-1, j, i], vi,
                                             v[k+1, j, i], v[k+2, j, i], wi)
                    v_zmf = muscl_face_value(v[k-2, j, i], v[k-1, j, i],
                                             vi, v[k+1, j, i], wi)
                    w_zpf = muscl_face_value(w[k-1, j, i], wi,
                                             w[k+1, j, i], w[k+2, j, i], wi)
                    w_zmf = muscl_face_value(w[k-2, j, i], w[k-1, j, i],
                                             wi, w[k+1, j, i], wi)
                    dudz = (u_zpf - u_zmf) * inv_dz_eff
                    dvdz = (v_zpf - v_zmf) * inv_dz_eff
                    dwdz = (w_zpf - w_zmf) * inv_dz_eff
                else:
                    if wi >= 0.0:
                        # Backward upwind — at k=0 wall ghost (uL_z=0) is used.
                        inv_d_below = 1.0 / d_face_below[k]
                        dudz = (ui - uL_z) * inv_d_below
                        dvdz = (vi - vL_z) * inv_d_below
                        dwdz = (wi - wL_z) * inv_d_below
                    else:
                        inv_d_above = 1.0 / d_face_above[k]
                        dudz = (uR_z - ui) * inv_d_above
                        dvdz = (vR_z - vi) * inv_d_above
                        dwdz = (wR_z - wi) * inv_d_above

                adv_u = -(ui * dudx + vi * dudy + wi * dudz)
                adv_v = -(ui * dvdx + vi * dvdy + wi * dvdz)
                adv_w = -(ui * dwdx + vi * dwdy + wi * dwdz)

                # ── Viscous diffusion (FV form for non-uniform dz) ──────
                # On a non-uniform grid the second derivative is:
                #   d²u/dz² = ((u[k+1]-u[k])/d_above − (u[k]-u[k-1])/d_below) / dz[k]
                # which reduces to the uniform-grid Laplacian when dz is constant.
                # Phase 14r: at k=0 the "below" neighbour is the wall (u_wall=0)
                # at half-cell distance d_face_below[0] = dz_arr[0]/2 — gives
                # the wall-shear contribution 2·u/dz that drives the no-slip BC.
                inv_dz_k = 1.0 / dz_arr[k]
                inv_d_above = 1.0 / d_face_above[k]
                inv_d_below = 1.0 / d_face_below[k]
                # Phase 14v-bc: ghost-aware diffusion stencils.
                # At k=0: uL_z = 0 (wall) at half-cell distance — naturally
                # produces 2·ν·u/dz wall shear via standard FV form.
                d2u_dx2 = (uR_x - 2.0 * ui + uL_x) * inv_dx2
                d2u_dy2 = (u[k, jp1, i] - 2.0 * ui + u[k, jm1, i]) * inv_dy2
                d2u_dz2 = (((uR_z - ui) * inv_d_above
                            - (ui - uL_z) * inv_d_below) * inv_dz_k)
                d2v_dx2 = (vR_x - 2.0 * vi + vL_x) * inv_dx2
                d2v_dy2 = (v[k, jp1, i] - 2.0 * vi + v[k, jm1, i]) * inv_dy2
                d2v_dz2 = (((vR_z - vi) * inv_d_above
                            - (vi - vL_z) * inv_d_below) * inv_dz_k)
                d2w_dx2 = (wR_x - 2.0 * wi + wL_x) * inv_dx2
                d2w_dy2 = (w[k, jp1, i] - 2.0 * wi + w[k, jm1, i]) * inv_dy2
                d2w_dz2 = (((wR_z - wi) * inv_d_above
                            - (wi - wL_z) * inv_d_below) * inv_dz_k)
                visc_u = nu_i * (d2u_dx2 + d2u_dy2 + d2u_dz2)
                visc_v = nu_i * (d2v_dx2 + d2v_dy2 + d2v_dz2)
                visc_w = nu_i * (d2w_dx2 + d2w_dy2 + d2w_dz2)

                # ── Buoyancy (Boussinesq, z only) ────────────────────
                # a_buoy_z = +g · (T - T_amb) / T_amb  (Boussinesq).
                # Hot gas (T > T_amb) gets positive acceleration → rises.
                # Derived from F = -(ρ − ρ_amb)·g_vec with g_vec = -g·ẑ:
                #   F_z = +(ρ_amb·g/ρ)·(T − T_amb)/T_amb ≈ g·ΔT/T_amb.
                # ABLATION RESULT: only +17K when disabled — small effect.
                buoy_w = _G * (T_g[k, j, i] - T_amb) / T_amb

                # ── External body force (drag, etc.) ─────────────────
                ext_u = Fx_ext[k, j, i] / rho_i
                ext_v = Fy_ext[k, j, i] / rho_i
                ext_w = Fz_ext[k, j, i] / rho_i

                du[k, j, i] = (adv_u + visc_u + ext_u) * dt
                dv[k, j, i] = (adv_v + visc_v + ext_v) * dt
                dw[k, j, i] = (adv_w + visc_w + buoy_w + ext_w) * dt

    # Apply updates over full grid (all cells are real; BCs via on-the-fly
    # ghost in the loop above — Way B).
    for k in prange(0, Nz):
        for j in range(Ny):
            for i in range(Nx):
                u[k, j, i] += du[k, j, i]
                v[k, j, i] += dv[k, j, i]
                w[k, j, i] += dw[k, j, i]


# ── Phase 14y: outflow sponge zone for u_x ──────────────────────────────────
# Damps u_x toward the inlet log-law profile in the last N_SPONGE cells
# near x=Lx.  Suppresses backflow at the open outflow boundary that the
# Dirichlet-pressure (p=0) BC otherwise admits.  Standard CFD technique
# for open boundaries with buoyancy-driven entrainment (Givelberg &
# Bunin 2004; ANSYS Fluent open-boundary §6.6).  Damping is a soft
# physical relaxation, not a BC pin — multiple time steps before u
# fully approaches u_target.
#
# Sponge profile: σ(i) = σ_max · ((i − i_start)/N_sponge)²  for the
# last N_sponge cells; zero elsewhere.  Quadratic ramp avoids artificial
# reflections (Israeli & Orszag 1981).

@njit(cache=True, parallel=True)
def apply_outflow_sponge(
    u: np.ndarray,            # (Nz, Ny, Nx)  modified in place
    u_target_2d: np.ndarray,  # (Nz, Ny)      target profile (typically u_inlet log-law)
    sigma_x: np.ndarray,      # (Nx,)         relaxation rate per i [1/s]; nonzero only in sponge
    Y_F: np.ndarray,          # (Nz, Ny, Nx)  fuel mass fraction — used to skip flame cells
    Y_F_skip: float,          # if Y_F[k,j,i] > Y_F_skip, sponge is skipped at that cell
    dt: float,
) -> None:
    """Flame-aware sponge: damps only in cells where Y_F is low (i.e., NOT in
    the active flame plume).  Suppresses backflow at the open outflow boundary
    without artificially damping the active flame body when it advances into
    the sponge zone.

    Necessary because cases with shorter Lx (e.g., Cheney sweep at Lx=10) have
    the active front reach the outlet region; an unconditional sponge would
    kill the flame there (observed Nat 4% U=2: 1.14 PASS → 0.13 FAIL).
    """
    Nz, Ny, Nx = u.shape
    for k in prange(Nz):
        for j in range(Ny):
            u_t = u_target_2d[k, j]
            for i in range(Nx):
                sig = sigma_x[i]
                if sig > 0.0 and Y_F[k, j, i] < Y_F_skip:
                    u[k, j, i] += sig * dt * (u_t - u[k, j, i])
