"""Phase 16-0 — Shared Lagrangian particle infrastructure.

Provides the kinematic / locator / buffer primitives that every
Lagrangian particle class in this codebase reuses, so module-specific
kernels can focus on their own physics (deposit, pyrolysis, char-ox,
firebrand-ignition transfer, etc.) without re-implementing motion and
cell lookups.

CONSUMERS (current and planned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - finney_lagrangian_3d.py    (Phase 15P) — gas-phase tongues; uses
                               locator + slot allocator + step_kinematics
                               with drag + buoyancy
  - lagrangian_bed_3d.py       (Phase 16) — solid-phase fuel particles
                               (drying, pyrolysis, char-ox, smolder);
                               stationary by default but can call
                               step_kinematics when lofted
  - lagrangian_firebrand_3d.py (Phase 16+) — mobile burning particles;
                               uses step_kinematics with drag + gravity

DESIGN
~~~~~~

Each consumer module owns its own arrays — there's NO shared
multi-class particle buffer.  Sharing happens at the function level:

  - locate_k_from_z(z, z_face, Nz)    → k-index of cell containing z
  - alloc_dead_slot(part_alive)       → first free slot (or -1)
  - step_kinematics(...)              → drag + buoyancy + gravity step

The shared kinematic kernel takes flag arguments
(use_drag, use_buoyancy, use_gravity) so different particle classes
can opt in or out of each force.  For STATIONARY particles (default
for bed), the consumer simply does not call step_kinematics at all
— the kinematic state arrays are still allocated (cheap) but never
modified.

CONSERVATION DISCIPLINE
~~~~~~~~~~~~~~~~~~~~~~~

This module does NOT touch gas state (rho, T_g, Y_F).  Conservation
is the consumer module's responsibility:

  - inventory carried by particles is decremented when the
    consumer applies it to gas cells (deposit step)
  - inventory leaves the simulation when a particle exits the
    domain — step_kinematics marks alive=0 and bumps n_exit; the
    consumer is expected to NOT double-account that loss
  - the kinematic step never modifies (m, E, Y_F, m_solid, ...) —
    only (x, y, z, u, v, w, alive)

DOMAIN EXIT POLICY
~~~~~~~~~~~~~~~~~~

Per project rule established in Phase 15P:
  "Leaving the domain ≠ conservation break.  Real tongues that punch
   above the canopy and rise into the convective plume above Z_top
   have, in reality, left the bed control volume.  Do NOT dump
   residual at the exit cell."

The kinematic step retires the particle (alive=0) and increments the
exit counter.  Inventory remains in the particle's slot until the slot
is re-used.

REFERENCES
~~~~~~~~~~

- Mell et al. 2007 "Numerical simulation and experiments of burning
  douglas fir trees," WFDS approach to Lagrangian vegetation
- McGrattan et al. 2017 "FDS Technical Reference vol 1" §8 particle
  motion + drag
- Crowe et al. 2011 "Multiphase Flows with Droplets and Particles,"
  CRC Press — quadratic drag formulation
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

# Shared physical constants
GRAV   = 9.81     # m/s²
CP_GAS = 1100.0   # J/kg/K — matches coupling_3d / finney_tendril_3d / coupling

# Alive flag canonical values (use int8 arrays)
ALIVE_FALSE = 0
ALIVE_TRUE  = 1


# ── Locator helpers ────────────────────────────────────────────────────


@njit(cache=True)
def locate_k_from_z(z: float, z_face: np.ndarray, Nz: int) -> int:
    """Return k-index of the cell containing z, or -1 if out of domain.

    z_face has length Nz+1; cell k spans z_face[k] .. z_face[k+1].
    Linear scan — at production Nz~60 this is O(Nz) per particle, which
    is fine; particle-step cost is dominated by gas-cell mass-balance
    updates, not the locator.
    """
    if z < z_face[0] or z >= z_face[Nz]:
        return -1
    for k in range(Nz):
        if z < z_face[k + 1]:
            return k
    return Nz - 1


@njit(cache=True)
def locate_cell(
    x: float, y: float, z: float,
    dx: float, dy: float, z_face: np.ndarray, Nz: int,
    Nx: int, Ny: int,
) -> tuple:
    """Return (i, j, k) of the cell containing (x, y, z).

    Any index = -1 → particle is out of domain in that axis.
    Caller is responsible for treating this as a domain-exit event.
    """
    if x < 0.0:
        i = -1
    else:
        i = int(x / dx)
        if i >= Nx:
            i = -1
    if y < 0.0:
        j = -1
    else:
        j = int(y / dy)
        if j >= Ny:
            j = -1
    k = locate_k_from_z(z, z_face, Nz)
    return i, j, k


# ── Slot allocator ─────────────────────────────────────────────────────


@njit(cache=True)
def alloc_dead_slot(part_alive: np.ndarray) -> int:
    """Return the index of the first dead particle slot, or -1 if full.

    Linear scan over the alive flag array.  O(N_max) per allocation; at
    expected spawn rates (~5-10 spawns/step) and N_max=8192 this is
    well under 1% of step cost.  For very large N_max with high spawn
    rates, a free-list could be added later; not premature-optimized now.
    """
    N_max = part_alive.shape[0]
    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            return p
    return -1


# ── Buffer factory ─────────────────────────────────────────────────────


def allocate_kinematic_buffers(N_max: int) -> dict:
    """Return a fresh dict of zero-initialised kinematic state arrays.

    Caller (each particle class) ADDS its own inventory arrays alongside
    these — this factory only handles the universal kinematic + alive
    state.  Returned arrays:

      x, y, z : float64[N_max]   position (m)
      u, v, w : float64[N_max]   velocity (m/s)
      alive   : int8 [N_max]     ALIVE_TRUE / ALIVE_FALSE
      age     : float64[N_max]   time since spawn (s)
    """
    if N_max < 0:
        raise ValueError(f"N_max must be ≥ 0; got {N_max}")
    return {
        "x":     np.zeros(N_max, dtype=np.float64),
        "y":     np.zeros(N_max, dtype=np.float64),
        "z":     np.zeros(N_max, dtype=np.float64),
        "u":     np.zeros(N_max, dtype=np.float64),
        "v":     np.zeros(N_max, dtype=np.float64),
        "w":     np.zeros(N_max, dtype=np.float64),
        "alive": np.zeros(N_max, dtype=np.int8),
        "age":   np.zeros(N_max, dtype=np.float64),
    }


# ── Shared kinematic step ──────────────────────────────────────────────


@njit(cache=True)
def step_kinematics(
    # particle state (modified in place)
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_u: np.ndarray, part_v: np.ndarray, part_w: np.ndarray,
    part_alive: np.ndarray, part_age: np.ndarray,
    # per-particle density (kg/m³, used by buoyancy)
    rho_p_arr: np.ndarray,
    # gas state (read-only at particle locations)
    rho_g_grid: np.ndarray,
    u_g_grid: np.ndarray,
    v_g_grid: np.ndarray,
    w_g_grid: np.ndarray,
    # grid
    dx: float, dy: float, dz_arr: np.ndarray, z_face: np.ndarray,
    # physics options
    d_p: float, C_D: float,
    use_drag: bool, use_buoyancy: bool, use_gravity: bool,
    y_periodic: bool,
    dt: float,
    # diagnostics (length-1 int64 arrays)
    n_alive_out: np.ndarray,
    n_exit_out: np.ndarray,
) -> None:
    """Advance position + velocity of every alive particle by one dt.

    Forces applied (configurable via flags):
        if use_drag:      a += (u_g − u_p) / τ_d,   τ_d = d_p / (C_D · |u_g − u_p|)
        if use_buoyancy:  a_z += g · (ρ_g − ρ_p) / ρ_p
        if use_gravity:   a_z += −g

    Integration: semi-implicit / symplectic Euler — velocity first,
    then position with the NEW velocity.  Bit-exact deterministic under
    repeat (sequential outer loop; no reductions).

    y BC:
        if y_periodic: wrap into [0, Ly)
        else:          domain exit on y < 0 or y >= Ly

    x and z BC: always domain exit.  Retired particles get alive=0;
    inventory left on the particle's slot is the consumer module's
    responsibility (Phase 15P convention: drop it, no exit-cell dump).

    Determinism: Rule #17 bit-exact under back-to-back calls — no
    parallel reductions, no thread-shared accumulators.
    """
    Nz, Ny, Nx = rho_g_grid.shape
    N_max = part_alive.shape[0]
    Lx = dx * Nx
    Ly = dy * Ny
    n_alive = 0
    n_exit = 0

    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue

        # Locate cell
        i, j, k = locate_cell(part_x[p], part_y[p], part_z[p],
                              dx, dy, z_face, Nz, Nx, Ny)
        if i < 0 or j < 0 or k < 0:
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue

        # Compute kinematic acceleration components
        a_u = 0.0
        a_v = 0.0
        a_w = 0.0

        if use_drag:
            urel = u_g_grid[k, j, i] - part_u[p]
            vrel = v_g_grid[k, j, i] - part_v[p]
            wrel = w_g_grid[k, j, i] - part_w[p]
            umag = math.sqrt(urel * urel + vrel * vrel + wrel * wrel)
            if umag > 1.0e-6:
                tau_d = d_p / (C_D * umag)
                a_u += urel / tau_d
                a_v += vrel / tau_d
                a_w += wrel / tau_d

        if use_buoyancy:
            rho_p = rho_p_arr[p]
            if rho_p > 0.0:
                rho_g_p = rho_g_grid[k, j, i]
                a_w += GRAV * (rho_g_p - rho_p) / rho_p

        if use_gravity:
            a_w -= GRAV

        # Symplectic Euler: velocity, then position
        part_u[p] += a_u * dt
        part_v[p] += a_v * dt
        part_w[p] += a_w * dt

        part_x[p] += part_u[p] * dt
        part_y[p] += part_v[p] * dt
        part_z[p] += part_w[p] * dt

        # y BC
        if y_periodic:
            if part_y[p] < 0.0:
                part_y[p] += Ly
            elif part_y[p] >= Ly:
                part_y[p] -= Ly
        else:
            if part_y[p] < 0.0 or part_y[p] >= Ly:
                part_alive[p] = ALIVE_FALSE
                n_exit += 1
                continue

        # x BC (always exit)
        if part_x[p] < 0.0 or part_x[p] >= Lx:
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue

        # z BC (always exit)
        if part_z[p] < z_face[0] or part_z[p] >= z_face[Nz]:
            part_alive[p] = ALIVE_FALSE
            n_exit += 1
            continue

        # Age update for retained particles
        part_age[p] += dt
        n_alive += 1

    n_alive_out[0] = n_alive
    n_exit_out[0] = n_exit


# ── Per-particle helpers (for consumer modules) ────────────────────────


@njit(cache=True)
def compute_rho_p_sphere(part_m: np.ndarray,
                         d_p: float,
                         part_alive: np.ndarray,
                         rho_p_out: np.ndarray) -> None:
    """Fill rho_p_out[p] = m[p] / (π/6 · d_p³) for alive slots; 0 otherwise.

    Convenience for consumer modules that use a uniform spherical-particle
    geometry (Finney tongues, char particles, etc.).  Bed-particle modules
    with non-spherical / per-particle d_p should compute rho_p directly
    instead of calling this helper.
    """
    N_max = part_alive.shape[0]
    V_p = (math.pi / 6.0) * d_p * d_p * d_p
    if V_p <= 0.0:
        for p in range(N_max):
            rho_p_out[p] = 0.0
        return
    inv_V_p = 1.0 / V_p
    for p in range(N_max):
        if part_alive[p] == ALIVE_TRUE:
            rho_p_out[p] = part_m[p] * inv_V_p
        else:
            rho_p_out[p] = 0.0
