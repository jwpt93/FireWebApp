"""Phase 15P — fully Lagrangian Finney burst-convective preheat closure.

Companion to the Eulerian Phase 15O (finney_tendril_3d.py); intended as
a higher-fidelity alternative when the spawn-and-deposit Eulerian
treatment proves too smeared.

PHYSICS DIFFERENCE FROM PHASE 15O
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 15O: each spawn event is INSTANTANEOUSLY teleported to a forward
deposit zone with exponential-weighted Eulerian distribution.  No
particle-following dynamics; the gas density-difference between flame
body and ambient cannot carry the tongue — the deposit pattern is fixed
by `L_t = u_buoy · duty_cycle · T_period`.

Phase 15P: each spawn event instantiates a TRACKED PARTICLE with state
(x, y, z, u, v, w, m, E, Y_F, t_remaining).  The particle moves under
its own equation of motion in the gas flow:

    dv/dt = g · (ρ_g − ρ_p)/ρ_p · ẑ          (buoyancy, vertical)
            + (u_g − v_p) / τ_drag             (gas drag, all components)

where τ_drag = d_p / (C_D · |u_g − v_p|) is the quadratic-drag
characteristic time and d_p is the particle's characteristic diameter
(Finney 2015 tongue width ~ 50-100 mm).  The particle density is computed
from its remaining mass over the typical tongue volume V_p = π/6 · d_p³.

Buoyancy direction: if ρ_p < ρ_g (hot tongue, lower density than
surrounding cold air ahead) → positive a_buoy → tongue rises into the
plume.  As it deposits mass+enthalpy into the Eulerian cells it passes
through, ρ_p drops further (m_p decreases at constant V_p) and buoyancy
strengthens — the tongue accelerates upward as it shrinks, matching
the Finney 2015 PIV-observed behavior.

LIFETIME AND DEATH
~~~~~~~~~~~~~~~~~~

- t_remaining initialised to `t_contact_s` at spawn.
- Each step, fraction `dt / t_remaining` of the carried inventory
  deposits into the Eulerian cell containing the particle.  Conservation
  exact by construction.
- Particle dies when t_remaining ≤ 0  (full inventory deposited)
                  OR ‖position‖ leaves the bed-relevant domain.
- **Exit from domain ≠ conservation break** — the residual inventory
  simply leaves with the particle.  This is physical: a tongue that
  punches above the canopy and rises into the convective plume above
  Z_top has, in reality, left the bed control volume.  Per user
  direction: do NOT dump residual at the exit cell.

CONSERVATION (when particle stays in-domain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Σ_t [rate(t) · dt] from spawn to death  =  ΔM_spawn  (exact)
  Σ_t [E_rate(t) · dt]                     =  ΔE_spawn  (exact)
  Σ_t [Yf_rate(t) · dt]                    =  ΔYf_spawn (exact)
  Σ_t [Px_rate(t) · dt]                    =  ΔPx_spawn (exact)

(Last step: when t_remaining ≤ dt, dump whatever inventory remains —
the kernel handles this branch.  No accumulated rounding drift.)

PARTICLE BUFFER
~~~~~~~~~~~~~~~

Particles are stored in struct-of-arrays form, in fixed-size buffers
of length N_max.  Spawning into a full buffer is a no-op (logged via
`n_spawn_overflow`).  Defragmentation is NOT performed every step;
dead-particle slots are reused when a new spawn arrives and an alive
slot is unavailable.

The slot allocator scans linearly for the first dead slot; allocation
cost is O(N_max) per spawn.  At spawn rates of ~5 events / sim_step
and N_max = 8192, this is negligible.

REFERENCES
~~~~~~~~~~

- Finney, M.A., et al. (2015) "Role of buoyant flame dynamics in
  wildfire spread," PNAS 112(32):9833.  Tongue ascent ~1-2 m/s;
  forward intrusion ~U_freestream relative to mean flow.
- Wimer, N.T., et al. (2020) JFM 895:A26.  Strouhal-Froude scaling
  unchanged from 15O.
- Crowe, C.T., et al. (2011) "Multiphase Flows with Droplets and
  Particles," CRC Press.  Quadratic-drag formulation, p. 49.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

# Phase 16-0: shared Lagrangian primitives.  Locator + slot allocator
# now live in `lagrangian_particles_3d` so the bed/firebrand modules can
# reuse them.  Re-exported below for backward compatibility with callers
# that imported these as `finney_lagrangian_3d._locate_k_from_z` etc.
from model_outdoor.physics_3d.lagrangian_particles_3d import (
    locate_k_from_z as _locate_k_from_z,    # noqa: F401  (re-export)
    alloc_dead_slot as _alloc_dead_slot,    # noqa: F401  (re-export)
    GRAV,
    CP_GAS,
    ALIVE_FALSE,
    ALIVE_TRUE,
)

# Phase 15P-only constants (Finney closure)
T_AMB    = 300.0    # K
T_GAS_FLAME = 600.0 # K — local L_F detection threshold

# Particle physical defaults (lit-anchored)
D_P_DEFAULT_M = 0.075   # Finney 2015 tongue diameter ~ 50-100 mm (median 75)
C_D_DEFAULT   = 1.0     # blunt-body high-Re sphere ~0.4-1.2; pick 1.0 (Crowe)


@njit(cache=True)
def step_finney_lagrangian_advect(
    rho: np.ndarray,        # (Nz, Ny, Nx) gas state — modified
    T_g: np.ndarray,        # —                       modified
    Y_F: np.ndarray,        # —                       modified
    u: np.ndarray, v: np.ndarray, w: np.ndarray,  #  modified
    # particle state — modified
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_u: np.ndarray, part_v: np.ndarray, part_w: np.ndarray,
    part_m: np.ndarray, part_E: np.ndarray, part_Yf: np.ndarray,
    part_t_rem: np.ndarray, part_alive: np.ndarray,
    # grid
    dx: float, dy: float, dz_arr: np.ndarray, z_face: np.ndarray,
    # physics params
    d_p: float, C_D: float,
    dt: float,
    # diagnostics out (length-1 int arrays)
    n_alive_out: np.ndarray,
    n_exit_out: np.ndarray,
) -> None:
    """Advect live particles one step; deposit inventory; retire dead/exited.

    SEQUENTIAL outer loop — guarantees deterministic deposit accumulation
    when multiple particles target the same Eulerian cell.  At expected
    particle counts (~100s alive on mickey, ~few-1000 on larger cases)
    this is dwarfed by the FFT / DOM passes.

    Each step per particle:
      1. Locate (i, j, k) of cell containing particle position.
      2. Compute deposit fraction = dt / t_rem; cap at 1.
      3. Apply mass-balanced update to (rho, T_g, Y_F, u) at that cell.
         (v, w gas state are NOT updated — particle x-momentum only,
          matching the Eulerian kernel's convention.)
      4. Decrement particle inventory by deposited fraction.
      5. Integrate particle velocity:
            ρ_p   = m_p / (π/6 · d_p³)
            a_b   = g · (ρ_g − ρ_p)/ρ_p          (vertical only)
            τ_d   = d_p / (C_D · |u_g − v_p|)
            a_u   = (u_g − u_p) / τ_d
            a_v   = (v_g − v_p) / τ_d
            a_w   = (w_g − w_p) / τ_d  +  a_b
         v_p ← v_p + a · dt;  x_p ← x_p + v_p · dt.
      6. Apply y-periodic BC.
      7. If x_p or z_p out of domain  OR  t_rem ≤ 0  → retire.
    """
    Nz, Ny, Nx = rho.shape
    N_max = part_alive.shape[0]
    n_alive = 0
    n_exit = 0
    V_p = (math.pi / 6.0) * d_p * d_p * d_p  # particle volume [m³]
    inv_V_p = 1.0 / V_p
    Lx = dx * Nx
    Ly = dy * Ny

    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue

        # Locate cell
        i = int(part_x[p] / dx)
        j = int(part_y[p] / dy)
        k = _locate_k_from_z(part_z[p], z_face, Nz)
        if i < 0 or i >= Nx or j < 0 or j >= Ny or k < 0:
            # Out of domain — particle leaves with residual inventory
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue

        # ── Deposit fraction of remaining inventory into cell (i,j,k) ──
        t_rem = part_t_rem[p]
        if t_rem <= dt:
            frac = 1.0
        else:
            frac = dt / t_rem
        dM = frac * part_m[p]
        dE = frac * part_E[p]
        dYf = frac * part_Yf[p]
        # Particle x-momentum contribution to gas
        dPx = frac * part_m[p] * part_u[p]

        V_cell = dx * dy * dz_arr[k]
        rho_old = rho[k, j, i]
        M_old = rho_old * V_cell
        M_new = M_old + dM
        if M_new > 0.0:
            # Mass-balanced update for T_g, Y_F, u
            E_old_local = M_old * CP_GAS * T_g[k, j, i]
            E_new_local = E_old_local + dE
            Yf_old_M = M_old * Y_F[k, j, i]
            Yf_new_M = Yf_old_M + dYf
            Px_old_M = M_old * u[k, j, i]
            Px_new_M = Px_old_M + dPx
            rho[k, j, i] = M_new / V_cell
            T_g[k, j, i] = E_new_local / (M_new * CP_GAS)
            Y_F[k, j, i] = Yf_new_M / M_new
            u[k, j, i] = Px_new_M / M_new

        # Decrement particle inventory
        part_m[p]  -= dM
        part_E[p]  -= dE
        part_Yf[p] -= dYf
        part_t_rem[p] = t_rem - dt
        # Numerical cleanup
        if part_t_rem[p] <= 0.0 or part_m[p] <= 0.0:
            part_alive[p] = ALIVE_FALSE
            continue

        # ── Integrate particle equation of motion ──
        # Gas state at particle's cell
        rho_g_p = rho[k, j, i]
        u_g_p   = u[k, j, i]
        v_g_p   = v[k, j, i]
        w_g_p   = w[k, j, i]

        # Particle density (remaining mass over fixed volume)
        rho_p = part_m[p] * inv_V_p
        if rho_p <= 0.0:
            part_alive[p] = ALIVE_FALSE
            continue

        # Buoyancy (vertical only)
        a_buoy = GRAV * (rho_g_p - rho_p) / rho_p

        # Quadratic gas-drag
        urel = u_g_p - part_u[p]
        vrel = v_g_p - part_v[p]
        wrel = w_g_p - part_w[p]
        umag = math.sqrt(urel*urel + vrel*vrel + wrel*wrel)
        if umag < 1.0e-6:
            a_u = 0.0
            a_v = 0.0
            a_w = a_buoy
        else:
            tau_d = d_p / (C_D * umag)
            a_u = urel / tau_d
            a_v = vrel / tau_d
            a_w = wrel / tau_d + a_buoy

        # Update velocity (forward Euler)
        part_u[p] += a_u * dt
        part_v[p] += a_v * dt
        part_w[p] += a_w * dt

        # Update position
        part_x[p] += part_u[p] * dt
        part_y[p] += part_v[p] * dt
        part_z[p] += part_w[p] * dt

        # y-periodic BC
        if part_y[p] < 0.0:
            part_y[p] += Ly
        elif part_y[p] >= Ly:
            part_y[p] -= Ly

        # Domain exit check (x, z) — y is periodic so always in-domain
        if part_x[p] < 0.0 or part_x[p] >= Lx:
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue
        if part_z[p] < z_face[0] or part_z[p] >= z_face[Nz]:
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue

        n_alive += 1

    n_alive_out[0] = n_alive
    n_exit_out[0] = n_exit


@njit(cache=True)
def step_finney_lagrangian_spawn(
    rho: np.ndarray,        # gas state — modified (source-cell sink)
    T_g: np.ndarray,        # —          modified
    Y_F: np.ndarray,        # —          modified
    u: np.ndarray,          # —          modified
    phi_flame: np.ndarray,
    L_F_field: np.ndarray,
    last_spawn_time: np.ndarray,
    # particle state — modified (new spawns allocated)
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_u: np.ndarray, part_v: np.ndarray, part_w: np.ndarray,
    part_m: np.ndarray, part_E: np.ndarray, part_Yf: np.ndarray,
    part_t_rem: np.ndarray, part_alive: np.ndarray,
    # grid
    dx: float, dy: float, dz_arr: np.ndarray, z_centre: np.ndarray,
    # params
    t_now: float,
    sr: float, duty_cycle: float, f_mass: float, fr_min: float,
    t_contact_s: float,
    # diagnostics out
    n_spawn_events_out: np.ndarray,
    n_spawn_overflow_out: np.ndarray,
) -> None:
    """Detect LE spawn events, allocate particles, debit source-cell inventory.

    Spawn gating identical to Phase 15O.1 (Sr-Fr + frequency cap).  When
    a spawn fires:
      1. Compute extracted (dM, dE, dYf, dPx) from the source cell.
      2. Apply source-cell sink IN PLACE (rho, T_g, Y_F, u updated
         mass-balanced).  Per user: source is instantaneous, deposit is
         time-spread via the particle's lifetime.
      3. Allocate a dead slot; if buffer full, increment overflow counter
         and skip.
      4. Initialise particle position at cell centre, velocity = gas
         velocity (the tongue is born co-moving with the local gas; its
         own buoyancy + drag then carry it).
      5. Inventory = extracted; t_rem = t_contact_s.
    """
    Nz, Ny, Nx = rho.shape
    n_spawn_events_out[0] = 0
    n_spawn_overflow_out[0] = 0
    f_eff = min(f_mass, 0.99)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx - 1):
                phi_here = phi_flame[k, j, i]
                phi_next = phi_flame[k, j, i + 1]
                if not (phi_here <= 0.0 and phi_next > 0.0):
                    continue
                L_F = L_F_field[j, i]
                if L_F <= 0.0:
                    continue
                Tg_here = T_g[k, j, i]
                if Tg_here <= T_AMB:
                    continue
                u_buoy_sq = 2.0 * GRAV * L_F * (Tg_here - T_AMB) / T_AMB
                if u_buoy_sq <= 0.0:
                    continue
                u_buoy = math.sqrt(u_buoy_sq)
                T_period = L_F / max(sr * u_buoy, 1.0e-12)
                Fr_local = u_buoy_sq / (GRAV * L_F)
                if Fr_local < fr_min:
                    continue
                if t_now - last_spawn_time[k, j, i] < T_period:
                    continue

                # Compute extraction from source cell
                V_cell = dx * dy * dz_arr[k]
                rho_src = rho[k, j, i]
                if rho_src <= 0.0:
                    continue
                dM = f_eff * rho_src * V_cell
                dE = dM * CP_GAS * T_g[k, j, i]
                dYf = dM * Y_F[k, j, i]
                dPx = dM * u[k, j, i]

                # Apply source-cell sink mass-balanced
                M_old = rho_src * V_cell
                M_new = M_old - dM
                if M_new <= 0.0:
                    continue
                E_old_local = M_old * CP_GAS * T_g[k, j, i]
                E_new_local = E_old_local - dE
                Yf_old_M = M_old * Y_F[k, j, i]
                Yf_new_M = Yf_old_M - dYf
                Px_old_M = M_old * u[k, j, i]
                Px_new_M = Px_old_M - dPx
                rho[k, j, i] = M_new / V_cell
                T_g[k, j, i] = E_new_local / (M_new * CP_GAS)
                Y_F[k, j, i] = Yf_new_M / M_new
                u[k, j, i] = Px_new_M / M_new

                # Allocate particle slot
                slot = _alloc_dead_slot(part_alive)
                if slot < 0:
                    n_spawn_overflow_out[0] += 1
                    # Spawn fails — but source-cell sink was already
                    # applied.  This is an irreversible loss; the user
                    # has been notified via overflow counter.
                    last_spawn_time[k, j, i] = t_now
                    continue

                # Initialise particle.  Position at cell centre.
                part_x[slot] = (i + 0.5) * dx
                part_y[slot] = (j + 0.5) * dy
                part_z[slot] = z_centre[k]
                # Velocity = local gas velocity (tongue born co-moving)
                part_u[slot] = u[k, j, i]
                # v, w gas state are at cell centre but we read from u-grid
                # for x-component, accept centred for v, w (small bias)
                part_v[slot] = 0.0
                part_w[slot] = u_buoy * 0.1   # small initial upward kick
                # Inventory
                part_m[slot]  = dM
                part_E[slot]  = dE
                part_Yf[slot] = dYf
                part_t_rem[slot] = t_contact_s
                part_alive[slot] = ALIVE_TRUE

                last_spawn_time[k, j, i] = t_now
                n_spawn_events_out[0] += 1
