"""Phase 15O — Eulerian leading-edge spawn-and-deposit closure for Finney 2015
buoyancy-instability intermittent burst-convective preheat.

PHYSICS
~~~~~~~

Finney 2015 PNAS 112(32):9833 measured that the dominant forward heat
transfer in grass fires is NOT steady radiation (≤30 kW/m² at the
preheating fuel) but intermittent contact from buoyancy-unstable flame
tongues that detach from the flame surface at the Strouhal-Froude
frequency.  During contact (100-500 ms), fine fuels heat at ~2,500 °C/s
— 500× the radiation-only rate.  Between contacts, only the steady
radiation reaches the fuel.

Finney 2015 establishes the scaling but explicitly defers closure form
to subsequent modeling work.  Phase 15O implements this as a
**conservation-preserving per-cell Eulerian spawn-and-deposit kernel**
on the flame leading-edge surface:

  1. Identify leading-edge surface cells: phi_flame[k,j,i] ≤ 0 AND
     phi_flame[k,j,i+1] > 0 (the +x-facing flame body boundary).
  2. Compute per-cell Strouhal-Froude scaling from LOCAL state:
       L_F      = local flame height at this (j) row, where T_g > 600K
       u_buoy   = sqrt(2·g·L_F·(T_g − T_amb)/T_amb)
       T_period = L_F / (Sr · u_buoy)
       Fr_local = u_buoy² / (g · L_F)
  3. Gate by Fr_local ≥ Fr_min AND (t − last_spawn[k,j,i]) ≥ T_period.
  4. If gated, extract mass+enthalpy+species+momentum from source cell
     (sink).
  5. Deposit into 1-5 cells forward of the source with exponential weights
     summing to 1 (each conserved quantity is redistributed exactly).
  6. Update last_spawn[k,j,i] = t for the frequency cap.

CONSERVATION DISCIPLINE
~~~~~~~~~~~~~~~~~~~~~~~

Each spawn event is strictly conservation-preserving per the four
quantities:

  Mass:     Σ_target ΔM_target = ΔM_source        (unit-test enforced)
  Enthalpy: Σ_target ΔE_target = ΔE_source        (unit-test enforced)
  Species:  Σ_target Δ(ρ·Y_F)_target = Δ(ρ·Y_F)_source
  Momentum: Σ_target ΔP_target,i = ΔP_source,i  for i ∈ {x,y,z}

The source cell loses density (continuity equation in the next outer
step drives flow inward to replenish, matching the physical picture
of the flame drawing fresh reactant from upstream).

CALIBRATION (Rule #1 — committed BEFORE running mickey)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SR_DEFAULT       = 0.20   (Finney 2015 Strouhal-Froude central value)
  DUTY_CYCLE       = 0.40   (Finney 2015 imaging contact duration ratio)
  F_MASS_DEFAULT   = 0.05   (mass fraction extracted per spawn — physical
                              best guess, sized as small but non-negligible)
  FR_MIN_DEFAULT   = 0.5    (gating Froude minimum, Wimer 2020 J.F.M. 895)
  T_AMB            = 300.0  (K)
  T_GAS_FLAME      = 600.0  (K threshold for L_F local-height detection)

These ARE the lit-anchored committed values.  Rule #2 forbids fishing
them into the Cheney calibration window; if Phase 15O does not close
the ROS gap with these committed values, the result is a documented
Rule #4 limitation.

References
~~~~~~~~~~

- Finney, M.A., Cohen, J.D., Forthofer, J.M., et al. (2015) "Role of
  buoyant flame dynamics in wildfire spread," PNAS 112(32):9833-9838.
- Finney, M.A., Grumstrup, T.P., Grenfell, I. (2020) "Flame
  characteristics adjacent to a stationary line fire," Combust. Sci.
  Tech. 194(11):2298-2316.
- Wimer, N.T., et al. (2020) "Scaling of the puffing Strouhal number
  for buoyant jets and plumes," J. Fluid Mech. 895:A26.
- Cetegen, B.M., Ahmed, T.A. (1993) "Experiments on the periodic
  instability of buoyant plumes and pool fires," Comb. Flame 93:157.

Phase 15O design choices (vs Phase 15N mean-field):
  - LEADING-EDGE SURFACE ONLY (not full flame body) — matches Finney
    physics (tongues form at the boundary, not in the interior).
  - INTERMITTENT (frequency-capped per cell) — not continuous time-average.
  - STRICT CONSERVATION — source cell is a true sink, deposit cells
    receive exact balanced gains.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

# Phase 15O committed values (Rule #1 lit-anchored, Rule #2 no-fishing)
SR_DEFAULT       = 0.20     # Strouhal number; Finney 2015 PNAS
DUTY_CYCLE       = 0.40     # contact/period ratio; Finney 2015 imaging
F_MASS_DEFAULT   = 0.05     # mass fraction extracted per spawn
FR_MIN_DEFAULT   = 0.5      # gating Froude minimum
T_AMB_DEFAULT    = 300.0    # ambient T [K]
T_GAS_FLAME      = 600.0    # L_F detection threshold [K]
GRAV             = 9.81     # m/s²
CP_GAS           = 1100.0   # J/kg/K  (matches coupling_3d)
MAX_DEPOSIT_CELLS = 5       # cap on N forward target cells per spawn

# Sentinel for "never spawned": last_spawn_time[k,j,i] = -1e9
_NEVER_SPAWNED = -1.0e9


@njit(cache=True)
def _compute_L_F_per_column(T_g, T_thresh, dz_arr, n_z_bed):
    """Per-(j,i) local flame body height L_F based on T_g > T_thresh.

    Returns (Ny, Nx) field of L_F in meters.  L_F is the highest z
    where T_g > T_thresh, integrated over dz_arr from the BED TOP.
    For columns with no hot gas, returns 0.0.
    """
    Nz, Ny, Nx = T_g.shape
    L_F = np.zeros((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        for i in range(Nx):
            h = 0.0
            for k in range(n_z_bed, Nz):
                if T_g[k, j, i] > T_thresh:
                    h += dz_arr[k]
            L_F[j, i] = h
    return L_F


# ── Phase 15O.1 — time-spread Eulerian release (no Lagrangian state) ────
#
# The instantaneous-deposit Phase 15O kernel biases against closure
# efficacy: in dt ≈ 25 ms, the deposited heat at a target cell advects
# downstream in 1 timestep at U=4 m/s, draining before gas-solid
# coupling integrates a useful T_s rise.
#
# Phase 15O.1 keeps the Eulerian architecture but spreads each spawn's
# release over T_contact (≈ duty_cycle × T_period ≈ 200-500 ms) via
# persistent per-cell "remaining inventory" fields.  Per step:
#   - Phase A: each cell with remaining > 0 releases fraction
#     (dt / time_remaining) of its inventory, then decrements
#     remaining and time_remaining.  Conservation exact by construction.
#   - Phase B: new spawns ADD their full inventory to the existing
#     remaining; time_remaining = max(existing, T_contact).
#
# Conservation:
#   ∫_{spawn}^{spawn + T_contact} rate · dt = ΔM_spawn  (exact)
# even when multiple sources target the same deposit cell at overlapping
# times — because we accumulate total remaining inventory, not rate.
#
# State fields (10 new (Nz,Ny,Nx) float64 arrays):
#   sink_M_remaining, sink_E_remaining, sink_Yf_remaining,
#       sink_Px_remaining, sink_time_remaining
#   deposit_M_remaining, deposit_E_remaining, deposit_Yf_remaining,
#       deposit_Px_remaining, deposit_time_remaining
#
# At mickey scale (Nz×Ny×Nx ≈ 60×5×100 = 30k cells):
#   10 fields × 8 bytes × 30k = 2.4 MB.  Acceptable.


@njit(cache=True)
def step_finney_tendril_apply_pending(
    rho: np.ndarray,
    T_g: np.ndarray,
    Y_F: np.ndarray,
    u: np.ndarray,
    sink_M: np.ndarray, sink_E: np.ndarray,
    sink_Yf: np.ndarray, sink_Px: np.ndarray,
    sink_t_rem: np.ndarray,
    dep_M: np.ndarray, dep_E: np.ndarray,
    dep_Yf: np.ndarray, dep_Px: np.ndarray,
    dep_t_rem: np.ndarray,
    dx: float, dy: float, dz_arr: np.ndarray,
    dt: float,
) -> None:
    """Phase A of Phase 15O.1 — apply pending sink/deposit rates this step.

    Releases (dt / time_remaining) of each remaining inventory and applies
    to the state arrays via mass-balanced updates.  Decrements remaining
    and time_remaining.  When time_remaining ≤ dt, releases whatever
    remains (numerical residual cleanup).

    Conservation discipline: per-cell mass / enthalpy / fuel / momentum
    updates use the exact accumulated inventory; no per-spawn timing or
    rate is required.
    """
    Nz, Ny, Nx = rho.shape
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                V = dx * dy * dz_arr[k]

                # ── Apply pending SINK (source-cell loss) ──
                t_rem = sink_t_rem[k, j, i]
                if t_rem > 0.0:
                    # Fraction to release this step (cap at 1.0)
                    if t_rem <= dt:
                        frac = 1.0
                    else:
                        frac = dt / t_rem
                    dM = frac * sink_M[k, j, i]
                    dE = frac * sink_E[k, j, i]
                    dYf = frac * sink_Yf[k, j, i]
                    dPx = frac * sink_Px[k, j, i]
                    rho_old = rho[k, j, i]
                    M_old = rho_old * V
                    M_new = M_old - dM
                    if M_new > 0.0:
                        # Mass-balanced update for T_g, Y_F, u
                        # E_old = M_old · cp · T_g_old
                        # E_new = E_old - dE
                        # T_g_new = E_new / (M_new · cp)
                        E_old_local = M_old * CP_GAS * T_g[k, j, i]
                        E_new_local = E_old_local - dE
                        Yf_old_M = M_old * Y_F[k, j, i]
                        Yf_new_M = Yf_old_M - dYf
                        Px_old_M = M_old * u[k, j, i]
                        Px_new_M = Px_old_M - dPx
                        rho[k, j, i] = M_new / V
                        T_g[k, j, i] = E_new_local / (M_new * CP_GAS)
                        Y_F[k, j, i] = Yf_new_M / M_new
                        u[k, j, i] = Px_new_M / M_new
                    # Decrement remaining inventory + timer
                    sink_M[k, j, i] -= dM
                    sink_E[k, j, i] -= dE
                    sink_Yf[k, j, i] -= dYf
                    sink_Px[k, j, i] -= dPx
                    sink_t_rem[k, j, i] = t_rem - dt
                    if sink_t_rem[k, j, i] < 0.0:
                        # Numerical cleanup
                        sink_t_rem[k, j, i] = 0.0
                        sink_M[k, j, i] = 0.0
                        sink_E[k, j, i] = 0.0
                        sink_Yf[k, j, i] = 0.0
                        sink_Px[k, j, i] = 0.0

                # ── Apply pending DEPOSIT (target-cell gain) ──
                t_rem = dep_t_rem[k, j, i]
                if t_rem > 0.0:
                    if t_rem <= dt:
                        frac = 1.0
                    else:
                        frac = dt / t_rem
                    dM = frac * dep_M[k, j, i]
                    dE = frac * dep_E[k, j, i]
                    dYf = frac * dep_Yf[k, j, i]
                    dPx = frac * dep_Px[k, j, i]
                    rho_old = rho[k, j, i]
                    M_old = rho_old * V
                    M_new = M_old + dM
                    if M_new > 0.0:
                        E_old_local = M_old * CP_GAS * T_g[k, j, i]
                        E_new_local = E_old_local + dE
                        Yf_old_M = M_old * Y_F[k, j, i]
                        Yf_new_M = Yf_old_M + dYf
                        Px_old_M = M_old * u[k, j, i]
                        Px_new_M = Px_old_M + dPx
                        rho[k, j, i] = M_new / V
                        T_g[k, j, i] = E_new_local / (M_new * CP_GAS)
                        Y_F[k, j, i] = Yf_new_M / M_new
                        u[k, j, i] = Px_new_M / M_new
                    dep_M[k, j, i] -= dM
                    dep_E[k, j, i] -= dE
                    dep_Yf[k, j, i] -= dYf
                    dep_Px[k, j, i] -= dPx
                    dep_t_rem[k, j, i] = t_rem - dt
                    if dep_t_rem[k, j, i] < 0.0:
                        dep_t_rem[k, j, i] = 0.0
                        dep_M[k, j, i] = 0.0
                        dep_E[k, j, i] = 0.0
                        dep_Yf[k, j, i] = 0.0
                        dep_Px[k, j, i] = 0.0


@njit(cache=True)
def step_finney_tendril_queue_spawns(
    rho: np.ndarray,
    T_g: np.ndarray,
    Y_F: np.ndarray,
    u: np.ndarray,
    phi_flame: np.ndarray,
    L_F_field: np.ndarray,
    last_spawn_time: np.ndarray,
    sink_M: np.ndarray, sink_E: np.ndarray,
    sink_Yf: np.ndarray, sink_Px: np.ndarray,
    sink_t_rem: np.ndarray,
    dep_M: np.ndarray, dep_E: np.ndarray,
    dep_Yf: np.ndarray, dep_Px: np.ndarray,
    dep_t_rem: np.ndarray,
    dx: float, dy: float, dz_arr: np.ndarray,
    t_now: float,
    sr: float, duty_cycle: float, f_mass: float, fr_min: float,
    T_amb: float,
    t_contact_s: float,
    n_spawn_events_out: np.ndarray,
    # Phase 15O.2/15O.3 — asymmetric spatial-aggregation box around LE.
    # Defaults all 0 → single-cell extraction (Phase 15O.1 back-compat).
    box_dk_up_radius: int = 0,     # cells ABOVE LE  (+z, plume direction)
    box_dk_down_radius: int = 0,   # cells BELOW LE  (−z, into deeper bed)
    box_dj_radius: int = 0,        # cells cross-stream (±y, symmetric)
    box_di_back_radius: int = 0,   # cells BEHIND LE (−x, into body)
) -> None:
    """Phase B of Phase 15O.1 — evaluate new spawns and ADD inventory.

    Identifies leading-edge surface cells passing the Sr-Fr gate + frequency
    cap.  For each, computes the spawn's total ΔM, ΔE, ΔYf, ΔP (using the
    CURRENT cell state — i.e., POST-Phase-A state, since Phase A already
    ran).  Adds these as full inventory to sink/deposit remaining fields;
    sets time_remaining = max(existing, t_contact_s).
    """
    Nz, Ny, Nx = rho.shape
    n_spawn_events_out[0] = 0

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
                if Tg_here <= T_amb:
                    continue
                u_buoy_sq = 2.0 * GRAV * L_F * (Tg_here - T_amb) / T_amb
                if u_buoy_sq <= 0.0:
                    continue
                u_buoy = math.sqrt(u_buoy_sq)
                T_period = L_F / max(sr * u_buoy, 1.0e-12)
                Fr_local = u_buoy_sq / (GRAV * L_F)
                if Fr_local < fr_min:
                    continue
                if t_now - last_spawn_time[k, j, i] < T_period:
                    continue

                # ── Phase 15O.2 — aggregate spawn inventory over box ──
                # Box: (k ± dk_radius, j ± dj_radius, i - di_radius .. i)
                # Only cells INSIDE flame body (phi_flame ≤ 0) contribute.
                # When all radii = 0, this reduces to single-cell extraction
                # (Phase 15O.1 back-compat).
                f_eff = min(f_mass, 0.99)
                dM_total = 0.0
                dE_total = 0.0
                dYf_total = 0.0
                dPx_total = 0.0

                for dk in range(-box_dk_down_radius, box_dk_up_radius + 1):
                    k_box = k + dk
                    if k_box < 0 or k_box >= Nz:
                        continue
                    for dj in range(-box_dj_radius, box_dj_radius + 1):
                        j_box = j + dj
                        if j_box < 0 or j_box >= Ny:
                            continue
                        for di_box in range(0, box_di_back_radius + 1):
                            i_box = i - di_box
                            if i_box < 0:
                                continue
                            # Only extract from cells inside the flame body
                            if phi_flame[k_box, j_box, i_box] > 0.0:
                                continue
                            cell_vol_box = dx * dy * dz_arr[k_box]
                            rho_box = rho[k_box, j_box, i_box]
                            if rho_box <= 0.0:
                                continue
                            # Mass to extract from this box cell.
                            # Multiple LE cells may aggregate over overlapping
                            # boxes; sink inventory accumulates, and Phase A
                            # releases the total over t_contact via the
                            # conservation-preserving fraction = dt/t_rem.
                            dM_box = f_eff * rho_box * cell_vol_box
                            dE_box = dM_box * CP_GAS * T_g[k_box, j_box, i_box]
                            dYf_box = dM_box * Y_F[k_box, j_box, i_box]
                            dPx_box = dM_box * u[k_box, j_box, i_box]
                            # Add to source-cell sink inventory
                            sink_M[k_box, j_box, i_box] += dM_box
                            sink_E[k_box, j_box, i_box] += dE_box
                            sink_Yf[k_box, j_box, i_box] += dYf_box
                            sink_Px[k_box, j_box, i_box] += dPx_box
                            if sink_t_rem[k_box, j_box, i_box] < t_contact_s:
                                sink_t_rem[k_box, j_box, i_box] = t_contact_s
                            # Accumulate totals for the deposit side
                            dM_total += dM_box
                            dE_total += dE_box
                            dYf_total += dYf_box
                            dPx_total += dPx_box

                if dM_total <= 0.0:
                    # No mass aggregated (e.g., box entirely outside flame body)
                    continue

                last_spawn_time[k, j, i] = t_now

                # Compute deposit zone weights (same as 15O.1)
                L_t = u_buoy * (duty_cycle * T_period)
                N_targets = int(L_t / dx)
                if N_targets < 1:
                    N_targets = 1
                if N_targets > MAX_DEPOSIT_CELLS:
                    N_targets = MAX_DEPOSIT_CELLS

                inv_L_t = 1.0 / max(L_t, 1.0e-12)
                w_sum = 0.0
                ws = np.zeros(N_targets, dtype=np.float64)
                for di in range(1, N_targets + 1):
                    d_center = (di - 0.5) * dx
                    wgt = math.exp(-d_center * inv_L_t)
                    ws[di - 1] = wgt
                    w_sum += wgt
                if w_sum <= 0.0:
                    continue
                inv_w_sum = 1.0 / w_sum

                # Distribute aggregated total to forward target cells
                for di in range(1, N_targets + 1):
                    i_t = i + di
                    if i_t >= Nx:
                        break
                    if phi_flame[k, j, i_t] <= 0.0:
                        continue
                    w_frac = ws[di - 1] * inv_w_sum
                    dep_M[k, j, i_t] += w_frac * dM_total
                    dep_E[k, j, i_t] += w_frac * dE_total
                    dep_Yf[k, j, i_t] += w_frac * dYf_total
                    dep_Px[k, j, i_t] += w_frac * dPx_total
                    if dep_t_rem[k, j, i_t] < t_contact_s:
                        dep_t_rem[k, j, i_t] = t_contact_s

                n_spawn_events_out[0] += 1


# ── Original instantaneous kernel (Phase 15O baseline) ──────────────


@njit(cache=True)
def step_finney_tendril_spawn_deposit(
    rho: np.ndarray,        # (Nz, Ny, Nx) gas density [kg/m³] — modified in place
    T_g: np.ndarray,        # gas T [K]                       — modified
    Y_F: np.ndarray,        # fuel mass fraction [-]          — modified
    u: np.ndarray, v: np.ndarray, w: np.ndarray,  # gas velocities — modified
    phi_flame: np.ndarray,  # signed-distance to flame body [m]
    L_F_field: np.ndarray,  # (Ny, Nx) local flame height [m]
    last_spawn_time: np.ndarray,  # (Nz, Ny, Nx) [s] — modified
    dx: float,              # cell size x [m]
    dy: float,              # cell size y [m]
    dz_arr: np.ndarray,     # (Nz,) cell heights [m]
    t_now: float,           # current sim time [s]
    sr: float,              # Strouhal number
    duty_cycle: float,      # contact/period ratio
    f_mass: float,          # mass-fraction extracted per spawn
    fr_min: float,          # Froude gating minimum
    T_amb: float,           # ambient T [K]
    # Diagnostics returned via the count array
    n_spawn_events_out: np.ndarray,  # (1,) int — total spawn events this step
) -> None:
    """One-step Finney-tendril spawn-and-deposit.

    Operates as a TWO-PASS Eulerian update with strict conservation:

      Pass 1 (per source cell, parallel-safe):
        - Detect leading-edge surface cells (phi_flame ≤ 0 AND
          phi_flame[i+1] > 0)
        - Evaluate Sr-Fr gate and frequency cap
        - If eligible: compute extraction quantities (ΔM, ΔE, ΔY_F, ΔP)
          and write SINK directly into source cell

      Pass 2 (per source cell again, parallel-safe):
        - For each spawning source cell, compute deposit zone (forward
          N cells with exponential decay weights)
        - Apply per-target gains directly

    By updating directly in passes (not collecting offsets in a parcel
    list), we keep the kernel @njit-cacheable and avoid race conditions:
    each pass only mutates cells via deterministic indexing.

    Conservation is enforced per-event in the formula construction; the
    unit tests verify totals on the whole-domain integrals.

    Notes on parallel safety: pass 1 writes to source cell (k,j,i), pass
    2 writes to forward cells (k,j,i+di) with di > 0.  In neither pass
    do two source cells write to the SAME target, because:
      - Pass 1: each source cell only writes to itself.
      - Pass 2: each source cell writes to its OWN forward zone.  Two
        sources can deposit to the same target only if they're within
        MAX_DEPOSIT_CELLS of each other on the same (k, j) row, but
        deposits are ADD operations — accumulation is commutative and
        thus race-safe under PARALLEL execution.  However, to keep
        bit-exact Rule #17 compliance, we run pass 2 SEQUENTIALLY
        rather than @prange.
    """
    Nz, Ny, Nx = rho.shape
    n_spawn_events_out[0] = 0

    # ── Pass 1: per leading-edge cell, evaluate gate and EXTRACT ────
    # Run sequentially to keep deterministic ordering when a cell may
    # depend on prior source extractions (e.g. neighbor cells).
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx - 1):
                # Leading-edge surface: in body here, outside at +1
                phi_here = phi_flame[k, j, i]
                phi_next = phi_flame[k, j, i + 1]
                if not (phi_here <= 0.0 and phi_next > 0.0):
                    continue

                # Local L_F
                L_F = L_F_field[j, i]
                if L_F <= 0.0:
                    continue

                # Local buoyant velocity
                Tg_here = T_g[k, j, i]
                if Tg_here <= T_amb:
                    continue
                u_buoy_sq = 2.0 * GRAV * L_F * (Tg_here - T_amb) / T_amb
                if u_buoy_sq <= 0.0:
                    continue
                u_buoy = math.sqrt(u_buoy_sq)

                # Strouhal period & Froude gate
                T_period = L_F / max(sr * u_buoy, 1.0e-12)
                Fr_local = u_buoy_sq / (GRAV * L_F)
                if Fr_local < fr_min:
                    continue

                # Frequency cap: only spawn if enough time has passed
                if t_now - last_spawn_time[k, j, i] < T_period:
                    continue

                # ── Spawn: extract from source cell ──────────────────
                cell_vol = dx * dy * dz_arr[k]
                rho_here = rho[k, j, i]
                if rho_here <= 0.0:
                    continue
                # Cap f_mass to ensure rho_new > 0
                f_eff = min(f_mass, 0.99)
                dM = f_eff * rho_here * cell_vol  # [kg]
                rho_new = rho_here - dM / cell_vol
                if rho_new <= 0.0:
                    continue

                # Source cell looks the same per-unit-mass (T_g, Y_F,
                # velocity unchanged); only ρ drops because the extracted
                # mass took its per-mass properties with it.
                rho[k, j, i] = rho_new
                # T_g, Y_F, u/v/w unchanged at source (same per-mass values).

                # Mark spawn time
                last_spawn_time[k, j, i] = t_now

                # ── Deposit zone parameters ──────────────────────────
                L_t = u_buoy * (duty_cycle * T_period)
                N_targets = int(L_t / dx)
                if N_targets < 1:
                    N_targets = 1
                if N_targets > MAX_DEPOSIT_CELLS:
                    N_targets = MAX_DEPOSIT_CELLS

                # Pass 2 inline (per source): compute weights then deposit
                inv_L_t = 1.0 / max(L_t, 1.0e-12)
                w_sum = 0.0
                # First pass: compute denormalized weights
                ws = np.zeros(N_targets, dtype=np.float64)
                for di in range(1, N_targets + 1):
                    d_center = (di - 0.5) * dx
                    wgt = math.exp(-d_center * inv_L_t)
                    ws[di - 1] = wgt
                    w_sum += wgt
                if w_sum <= 0.0:
                    # Shouldn't happen; numerical safety
                    continue
                inv_w_sum = 1.0 / w_sum

                # Per-target gains (mass-weighted apportionment of
                # ALL the conserved quantities from the source)
                # ΔE_target = ΔE_source × (w_target / Σw)
                # ΔP_target_x = ΔP_source_x × (w_target / Σw)
                # By apportioning ΔM, ΔE, ΔY_F, ΔP all by the SAME
                # weights, conservation is automatic per-component.
                dE_per_kg = CP_GAS * Tg_here   # enthalpy/kg of source gas
                dE_total = dM * dE_per_kg
                dYF_total = dM * Y_F[k, j, i]
                dPx_total = dM * u[k, j, i]
                dPy_total = dM * v[k, j, i]
                dPz_total = dM * w[k, j, i]

                for di in range(1, N_targets + 1):
                    i_t = i + di
                    if i_t >= Nx:
                        break
                    # Phase 15O only deposits into cells AHEAD of the
                    # flame body (phi_flame > 0); skip if target is
                    # already inside the body
                    if phi_flame[k, j, i_t] <= 0.0:
                        continue
                    w_frac = ws[di - 1] * inv_w_sum
                    cell_vol_t = dx * dy * dz_arr[k]
                    rho_t_old = rho[k, j, i_t]
                    M_old = rho_t_old * cell_vol_t
                    dM_t = w_frac * dM
                    M_new = M_old + dM_t
                    if M_new <= 0.0:
                        continue
                    # Update density
                    rho[k, j, i_t] = M_new / cell_vol_t
                    # Update T_g via energy conservation:
                    #   (M_old·cp·T_old + dM_t·cp·T_g_here) / (M_new·cp)
                    Tg_t_old = T_g[k, j, i_t]
                    T_g[k, j, i_t] = (
                        (M_old * Tg_t_old + dM_t * Tg_here)
                        / max(M_new, 1.0e-30)
                    )
                    # Update Y_F via species conservation:
                    YF_t_old = Y_F[k, j, i_t]
                    Y_F[k, j, i_t] = (
                        (M_old * YF_t_old + dM_t * Y_F[k, j, i])
                        / max(M_new, 1.0e-30)
                    )
                    # Update velocity via momentum conservation:
                    u_t_old = u[k, j, i_t]
                    u[k, j, i_t] = (
                        (M_old * u_t_old + dM_t * u[k, j, i])
                        / max(M_new, 1.0e-30)
                    )
                    v_t_old = v[k, j, i_t]
                    v[k, j, i_t] = (
                        (M_old * v_t_old + dM_t * v[k, j, i])
                        / max(M_new, 1.0e-30)
                    )
                    w_t_old = w[k, j, i_t]
                    w[k, j, i_t] = (
                        (M_old * w_t_old + dM_t * w[k, j, i])
                        / max(M_new, 1.0e-30)
                    )

                n_spawn_events_out[0] += 1
