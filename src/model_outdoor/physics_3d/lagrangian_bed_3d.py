"""Phase 16 — Lagrangian sub-grid bed-fuel particles.

Represents the solid bed as a population of discrete particles instead of
the Eulerian (Nz, Ny, Nx) `m_solid` / `m_water` / `m_char` fields, so
that bed-scale physics (pyrolysis, drying, char-ox, smolder) operates at
sub-grid resolution independent of dx.  This is the bed half of the
FDS / WFDS Lagrangian-vegetation approach (Mell et al. 2007).

WHY
~~~

Phase 15G / 15O / 15P investigations established that the level-set
FSD chemistry closure under-predicts mickey-scale Cheney ROS by ~5×,
and a sub-grid burst-convective preheat closure cannot fix this at any
defensible magnitude — the closure architecture caps at <1% of the
bulk pyrolysate flux (see memory phase15g, phase15op).  Going coarser
to FIRETEC's regime (dx=0.5 m) makes the model fizzle entirely because
each bed cell holds only a single, lumped pyrolysis state (memory
model_dx_floor_0p1m.md).

A Lagrangian bed treatment decouples bed pyrolysis resolution from
grid resolution: at dx=0.5 m one cell can hold 50+ particles, each
with its own (m_solid, m_water, T_s, char), so the volumetric S_pyro
source aggregates from many sub-cell pyrolysis trajectories.  This is
what WFDS uses to validate Cheney AU grasslands at meter-scale grids.

PARTICLE STATE
~~~~~~~~~~~~~~

Each bed particle has (in addition to the kinematic state allocated by
lagrangian_particles_3d.allocate_kinematic_buffers):

  m_solid     [kg]   dry biomass mass (decays via pyrolysis)
  m_water     [kg]   water mass (decays via drying)
  m_char      [kg]   accumulated char (grown by pyrolysis fraction)
  T_s         [K]    solid temperature
  m_solid_0   [kg]   initial dry-biomass mass (diagnostic; constant)
  m_water_0   [kg]   initial water mass (used for moisture gate)
  sav         [1/m]  surface-to-volume ratio (SAV) of the represented
                     vegetation element (e.g., ~2000/m for fine grass).
                     Drives heat-transfer area and form-drag (future).

INITIALISATION FROM EULERIAN α_s FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each Eulerian bed cell (k < n_z_bed, α_s > 0) is populated with
`n_per_cell` particles whose total mass matches the cell's Eulerian
inventory:

  m_solid_per_particle = ρ_b · V_cell / n_per_cell
  m_water_per_particle = M_init · m_solid_per_particle
                         (M = m_water/m_solid_dry, dimensionless)

Positions within the cell follow a deterministic low-discrepancy
distribution (coprime-mod packing — bit-exact under repeat per Rule
#17; no RNG involved).

GAS-CELL AGGREGATION
~~~~~~~~~~~~~~~~~~~~

Per step, each particle's drying + pyrolysis rates are scattered to the
gas cell containing it as VOLUMETRIC sources [kg/m³/s]:

  S_pyro[k,j,i]   += ETA · (dm_solid_p/dt) / V_cell
  S_drying[k,j,i] += (dm_water_p/dt) / V_cell
  Q_pyro[k,j,i]   += (rate · HOR_combined) / V_cell
  Q_drying[k,j,i] += (dm_water_p · L_VAP / dt) / V_cell
  Y_F_source[k,j,i] (= S_pyro × dt scaled into mass-fraction balance)

These match the Eulerian kernels' output signatures exactly, so the
existing gas-state coupling kernels can absorb the particle-derived
sources without modification.

T_s ENERGY BALANCE (Phase 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T_s update uses gas-particle convective exchange:

  dT_s/dt = h_conv · A_p · (T_g − T_s) / (m_p · c_p_s)
          − Q_endo_pyro / (m_p · c_p_s)
          − dm_water_p · L_VAP / (m_p · c_p_s · dt)

with:
  A_p   = σ · V_p_solid = σ · m_solid_p / ρ_solid_true
  h_conv ~ 25 W/m²/K (Mell 2007 §3.4, grass blade in light wind)
  c_p_s ~ 1500 J/kg/K (cellulose)
  ρ_solid_true ~ 380 kg/m³ (dry grass cellular density)

Radiation (DOM → particle) deferred to Phase 3.  Char_ox + smolder
also deferred to Phase 3.

CONSERVATION DISCIPLINE
~~~~~~~~~~~~~~~~~~~~~~~

This module preserves total bed mass + total system enthalpy across
each step, IN-DOMAIN:

  Σ_p [Δm_solid_p + Δm_water_p + Δm_char_p] = 0   per particle
  Δ(particle inventory)  =  Σ_cell [Δ(gas mass from sources)]

Out-of-domain loss only via particle motion (not used by stationary
bed-particle config); when motion is off, conservation is strict.

REFERENCES
~~~~~~~~~~

- Mell, W.E., Manzello, S.L., Maranghides, A. et al. (2007)
  "The wildland-urban interface fire problem — current approaches and
  research needs," IJWF 19:238.  Lagrangian-vegetation methodology.
- Susott, R.A. (1982) Combust. Flame 46:71 — grass SAV and pyrolysis
  yields.
- Lautenberger, C.W. (2007) Fire Saf. J. 42:215 — drying Arrhenius.
- Boonmee, N. & Quintiere, J.G. (2005) Combust. Flame 141:283 — char
  oxidation kinetics (used in Phase 3).
- Memory: phase15op_finney_closure_null.md, model_dx_floor_0p1m.md.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

# Shared infra: re-exported constants + helpers
from model_outdoor.physics_3d.lagrangian_particles_3d import (
    locate_k_from_z,
    locate_cell,
    ALIVE_FALSE,
    ALIVE_TRUE,
    allocate_kinematic_buffers,
)

# Pyrolysis / drying / char-ox / smolder constants — match the Eulerian
# kernels exactly so particle and grid paths are physically
# interchangeable.
from model_outdoor.physics_3d.pyrolysis_3d import (
    A_DRY, E_DRY, L_VAP_WATER,
    A_MD2004, E_MD2004,
    A_OP_MD2004, E_OP_MD2004, N_O2_OP, Y_O2_MIN_OP,
    ETA_MD2004, CHAR_YIELD_MD2004,
    HEAT_OF_PYROLYSIS, HOR_OP_MD2004,
    A_CHAR, E_CHAR, HOC_CHAR, T_CHAR_ONSET, Y_O2_MIN_CHAR, Q_CHAR_MAX,
    A_SMOLD, E_SMOLD, HOC_SMOLD, T_SMOLD_ONSET, Y_O2_MIN_SMOLD, Q_SMOLD_MAX,
    _R_GAS,
)

# Bed-particle physical constants
RHO_SOLID_TRUE_GRASS = 380.0   # [kg/m³] dry cellular density of grass
                                # (Susott 1982; cellulose ~ 380-500 kg/m³)
CP_SOLID_GRASS       = 1500.0  # [J/kg/K] specific heat of dry grass
                                # (Mell 2007 §3.4)
H_CONV_DEFAULT       = 25.0    # [W/m²/K] gas-particle convective coefficient
                                # for grass blade (Mell 2007).  Lit-bracketed
                                # 10-100 W/m²/K depending on wind; pick 25
                                # as conservative default; deck can override.
SAV_GRASS_DEFAULT    = 2000.0  # [1/m] grass surface-to-volume ratio
                                # (Cheney 1993 fine-fuel reference)
M_PARTICLE_BURNOUT   = 1.0e-8  # [kg] retire particle below this total
                                # mass (m_solid + m_water + m_char).
                                # Prevents zombie-particle dt waste.
# Stefan-Boltzmann radiation loss from particle surface.  At plateau the
# Σ Q_input balances Σ Q_rad_loss = ε·σ·A_p·T_s⁴, naturally capping T_s.
# Without this, no upper bound on particle T_s — once exothermic surface
# reactions fire (char_ox, smolder), T_s integrates without limit.
SIGMA_SB             = 5.67e-8 # [W/m²/K⁴] Stefan-Boltzmann
EPS_SOLID_DEFAULT    = 0.9     # [-] grass / char emissivity (Mell 2007;
                                # matches coupling_3d._EPS_SOLID)

# Equilibrium-drying boiling temperature.  FIRETEC (Linn 2002) pins
# vegetation T_s at this value while m_water > 0; excess heat above the
# pin goes to evaporation at rate Q_excess / L_VAP_WATER.  Recovers the
# Cheney 1993 moisture penalty (mass-time-scaling) that first-order
# Arrhenius drying cannot.
T_BOIL_WATER         = 373.15  # [K] water boiling at 1 atm

# Grass-tuned Arrhenius drying constants for the COMBINED drying mode.
# Below T_BOIL, water removal follows first-order Arrhenius with these
# (faster than Lautenberger's white-pine bound-water values); above
# T_BOIL the equilibrium override completes the evaporation.
# E = 30 kJ/mol: intermediate between Sano & Hasegawa 1995 capillary-
# water value (E=20 kJ/mol for rice straw, similar grass morphology)
# and Lautenberger 2009 white-pine bound-water value (E=44 kJ/mol).
# A = 4.29×10³ /s: matches Lautenberger's prefactor (same order as
# Stenseng et al 2001 biomass water removal A=1.3×10⁴ /s).
A_DRY_GRASS          = 4.29e3   # [1/s] grass capillary-water prefactor
E_DRY_GRASS          = 30_000.0 # [J/mol] grass capillary-water E_a

# Drying-mode tags (must match what the caller passes).
DRY_MODE_ARRHENIUS   = 0        # Lautenberger 2009 only (default; backward-compat)
DRY_MODE_EQUILIBRIUM = 1        # FIRETEC heat-rate-limited only
DRY_MODE_COMBINED    = 2        # grass Arrhenius (below T_BOIL) + equilibrium (above)


def allocate_bed_particle_buffers(N_max: int) -> dict:
    """Extend kinematic buffers with bed-specific state arrays.

    Returned dict adds to allocate_kinematic_buffers():
        m_solid:    [N_max]  current dry biomass mass [kg]
        m_water:    [N_max]  current water mass [kg]
        m_char:     [N_max]  accumulated char mass [kg]
        T_s:        [N_max]  solid temperature [K]
        m_solid_0:  [N_max]  initial dry mass (constant; diagnostic)
        m_water_0:  [N_max]  initial water mass (used for moisture gate)
        sav:        [N_max]  surface-to-volume ratio [1/m]
    """
    if N_max < 0:
        raise ValueError(f"N_max must be ≥ 0; got {N_max}")
    buf = allocate_kinematic_buffers(N_max)
    buf["m_solid"]    = np.zeros(N_max, dtype=np.float64)
    buf["m_water"]    = np.zeros(N_max, dtype=np.float64)
    buf["m_char"]     = np.zeros(N_max, dtype=np.float64)
    buf["T_s"]        = np.zeros(N_max, dtype=np.float64)
    buf["m_solid_0"]  = np.zeros(N_max, dtype=np.float64)
    buf["m_water_0"]  = np.zeros(N_max, dtype=np.float64)
    buf["sav"]        = np.zeros(N_max, dtype=np.float64)
    # Phase 20 C: peak char mass ever reached (ratcheted in pyrolysis).
    # Used as reference for ash-coverage penalty in char oxidation.
    buf["m_char_max"] = np.zeros(N_max, dtype=np.float64)
    return buf


def initialize_bed_particles_from_alpha_s(
    buf: dict,
    alpha_s: np.ndarray,        # (Nz, Ny, Nx)
    rho_b_dry: float,           # [kg/m³] bed dry density
    moisture_frac: float,       # M = m_water / m_solid_dry
    T_amb: float,
    dx: float, dy: float, dz_arr: np.ndarray,
    n_z_bed: int,
    n_per_cell: int,
    sav: float = SAV_GRASS_DEFAULT,
    i_lo: int | None = None, i_hi: int | None = None,
    j_lo: int | None = None, j_hi: int | None = None,
) -> int:
    """Populate buf with bed particles distributed across α_s > 0 cells.

    Returns: number of particles allocated.

    DESIGN:
      For each (k, j, i) with k < n_z_bed and α_s[k,j,i] > 0:
        - Allocate `n_per_cell` particles
        - Each particle gets: m_solid_p = ρ_b · V_cell / n_per_cell · α_s
                              m_water_p = M · m_solid_p
                              m_char_p  = 0
                              T_s_p     = T_amb
                              sav_p     = sav
        - Position: deterministic coprime-mod sub-cell packing
          (no RNG → bit-exact reproducible per Rule #17)

    BOUNDING (i_lo .. i_hi, j_lo .. j_hi):
      Optional x/y restriction to the actual fuel bed region (saves
      particles in non-bed columns where α_s would be 0 anyway).  If
      not specified, scans the full Eulerian grid; the α_s mask still
      filters non-bed cells.

    BUFFER OVERFLOW:
      If buf['alive'].shape[0] is smaller than (n_bed_cells × n_per_cell),
      raises ValueError.  Caller is expected to size the buffer correctly:
        N_max = (i_hi - i_lo) × (j_hi - j_lo) × n_z_bed × n_per_cell

    Returns the count actually allocated.
    """
    Nz, Ny, Nx = alpha_s.shape
    N_max = buf["alive"].shape[0]

    if i_lo is None: i_lo = 0
    if i_hi is None: i_hi = Nx
    if j_lo is None: j_lo = 0
    if j_hi is None: j_hi = Ny

    # Count bed cells with fuel
    n_bed_cells = 0
    for k in range(min(n_z_bed, Nz)):
        for j in range(j_lo, min(j_hi, Ny)):
            for i in range(i_lo, min(i_hi, Nx)):
                if alpha_s[k, j, i] > 0.0:
                    n_bed_cells += 1
    n_required = n_bed_cells * n_per_cell
    if n_required > N_max:
        raise ValueError(
            f"Buffer too small: need {n_required} slots "
            f"({n_bed_cells} bed cells × {n_per_cell} particles); have {N_max}"
        )

    slot = 0
    for k in range(min(n_z_bed, Nz)):
        V_cell = dx * dy * dz_arr[k]
        z_face_k = float(np.sum(dz_arr[:k]))
        for j in range(j_lo, min(j_hi, Ny)):
            for i in range(i_lo, min(i_hi, Nx)):
                a_s = alpha_s[k, j, i]
                if a_s <= 0.0:
                    continue
                # ρ_b is the BULK density (kg of solid per m³ of cell-
                # average volume — already accounts for porosity).  Total
                # solid mass in the cell = ρ_b × V_cell.  Do NOT multiply
                # by α_s (that's the solid volume fraction; it's already
                # baked into ρ_b in this codebase's convention).  Match
                # the Eulerian init pattern at spread_3d line ~932:
                #   state.m_hemi[k,:,i_bed] = rho_b
                m_solid_cell = rho_b_dry * V_cell
                m_solid_per_p = m_solid_cell / n_per_cell
                m_water_per_p = moisture_frac * m_solid_per_p

                # Coprime-mod sub-cell packing (deterministic, low-discrepancy)
                for p in range(n_per_cell):
                    fx = ((p * 13) % n_per_cell + 0.5) / n_per_cell
                    fy = ((p * 7)  % n_per_cell + 0.5) / n_per_cell
                    fz = (p + 0.5) / n_per_cell

                    buf["x"][slot] = (i + fx) * dx
                    buf["y"][slot] = (j + fy) * dy
                    buf["z"][slot] = z_face_k + fz * dz_arr[k]
                    buf["u"][slot] = 0.0
                    buf["v"][slot] = 0.0
                    buf["w"][slot] = 0.0
                    buf["alive"][slot] = ALIVE_TRUE
                    buf["age"][slot]   = 0.0

                    buf["m_solid"][slot]   = m_solid_per_p
                    buf["m_water"][slot]   = m_water_per_p
                    buf["m_char"][slot]    = 0.0
                    buf["T_s"][slot]       = T_amb
                    buf["m_solid_0"][slot] = m_solid_per_p
                    buf["m_water_0"][slot] = m_water_per_p
                    buf["sav"][slot]       = sav

                    slot += 1

    return slot


def apply_moisture_jump_zone(
    buf: dict, N: int,
    dx: float, dy: float, dz_arr: np.ndarray,
    n_z_bed: int,
    i_lo: int, i_hi: int,
    kz_mask: np.ndarray,        # (n_z_bed,) uint8: 1 = layer in zone
    delta_water_kg_m3: float,
) -> int:
    """Phase 24: sprinkler moisture-jump BC for the Lagrangian bed.

    Adds ``delta_water_kg_m3 · V_cell / n_per_cell`` per particle to
    every ALIVE particle whose (i, k) cell falls inside the zone:
    ``i ∈ [i_lo, i_hi)`` AND ``kz_mask[k] == 1``.  Particles outside
    the zone are untouched.  y-axis is entire domain (no j filter) —
    zone is (x, z) only, mirrors the Eulerian branch.

    The per-particle share is derived by counting alive particles per
    (i, k) cell once, so the total water added equals
    ``delta_water_kg_m3 · V_cell`` per cell in the zone regardless of
    how many particles that cell holds (Rule #17-safe by construction:
    single deterministic pass over the alive particles).

    Returns the number of particles updated (diagnostic).
    """
    if N <= 0 or i_hi <= i_lo:
        return 0

    part_x = buf["x"]
    part_y = buf["y"]
    part_z = buf["z"]
    part_alive = buf["alive"]
    part_m_water = buf["m_water"]
    part_m_water_0 = buf["m_water_0"]

    z_face = np.concatenate(([0.0], np.cumsum(dz_arr[:n_z_bed])))

    Nx_zone = i_hi - i_lo
    # Ny is unknown here — infer from max j in the particle set (cheap
    # single-pass; robust to sparse populations).  Fallback to 1 if the
    # buffer is empty.
    Ny_max = 1
    for p in range(N):
        if part_alive[p] == ALIVE_FALSE:
            continue
        jj = int(part_y[p] / dy)
        if jj + 1 > Ny_max:
            Ny_max = jj + 1

    counts = np.zeros((n_z_bed, Ny_max, Nx_zone), dtype=np.int64)
    cell_i = np.full(N, -1, dtype=np.int64)
    cell_j = np.full(N, -1, dtype=np.int64)
    cell_k = np.full(N, -1, dtype=np.int64)

    for p in range(N):
        if part_alive[p] == ALIVE_FALSE:
            continue
        i = int(part_x[p] / dx)
        if i < i_lo or i >= i_hi:
            continue
        j = int(part_y[p] / dy)
        if j < 0 or j >= Ny_max:
            continue
        # locate k in bed via z_face bisection (small n_z_bed → linear)
        z = part_z[p]
        k = -1
        for kk in range(n_z_bed):
            if z_face[kk] <= z < z_face[kk + 1]:
                k = kk
                break
        if k < 0 or kz_mask[k] == 0:
            continue
        cell_i[p] = i - i_lo
        cell_j[p] = j
        cell_k[p] = k
        counts[k, j, i - i_lo] += 1

    n_updated = 0
    for p in range(N):
        i_local = cell_i[p]
        j = cell_j[p]
        k = cell_k[p]
        if i_local < 0 or j < 0 or k < 0:
            continue
        c = counts[k, j, i_local]
        if c <= 0:
            continue
        V_cell = dx * dy * dz_arr[k]
        dw_per_particle = delta_water_kg_m3 * V_cell / float(c)
        part_m_water[p]   += dw_per_particle
        part_m_water_0[p] += dw_per_particle
        n_updated += 1

    return n_updated


@njit(cache=True)
def step_horizontal_solid_conduction_scatter(
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_alive: np.ndarray,
    part_m_solid: np.ndarray, part_m_water: np.ndarray,
    part_m_char: np.ndarray, part_T_s: np.ndarray,
    T_s_grid: np.ndarray,
    alpha_s_grid: np.ndarray,
    dx: float, dy: float, z_face: np.ndarray,
    k_solid: float, rho_solid_true: float, cp_solid: float,
    n_z_bed: int,
    dt: float,
) -> None:
    """Apply horizontal (x, y) solid conduction on the per-cell T_s grid,
    then scatter the delta_T back to particles.

    Physical rationale (per user direction):
      Grass conduction along the bed plane is small (α_solid ≈ 1.4e-7 m²/s,
      penetration distance √(αt) ≈ 12 µm in 1 ms) but provides the
      forward-spread pathway when gas-mediated radiation feedback is the
      only other forward heat-transfer term.  Match k_solid and ρ·cp from
      the existing vertical-conduction kernel (solid_conduction_3d).

    Sequential outer loop (Rule #17 deterministic).
    """
    Nz, Ny, Nx = T_s_grid.shape
    diff = k_solid / (rho_solid_true * cp_solid)   # m²/s
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    # 1. Compute conduction delta per cell.  Operate ONLY on bed cells
    #    (k < n_z_bed AND alpha_s > 0).  Use the OLD T_s for neighbors so
    #    no in-place hazard.
    T_s_old = T_s_grid.copy()
    delta_T_per_cell = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    for k in range(min(n_z_bed, Nz)):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s_grid[k, j, i] <= 0.0:
                    continue
                T_c = T_s_old[k, j, i]
                # x-direction (Neumann BC at edges via zero flux)
                Tx_p = T_s_old[k, j, i + 1] if (i + 1 < Nx and
                        alpha_s_grid[k, j, i + 1] > 0.0) else T_c
                Tx_m = T_s_old[k, j, i - 1] if (i - 1 >= 0 and
                        alpha_s_grid[k, j, i - 1] > 0.0) else T_c
                # y-direction (periodic)
                jp1 = j + 1 if j + 1 < Ny else 0
                jm1 = j - 1 if j - 1 >= 0 else Ny - 1
                Ty_p = T_s_old[k, jp1, i] if alpha_s_grid[k, jp1, i] > 0.0 else T_c
                Ty_m = T_s_old[k, jm1, i] if alpha_s_grid[k, jm1, i] > 0.0 else T_c
                # Laplacian
                lap = ((Tx_p - 2.0 * T_c + Tx_m) * inv_dx2
                       + (Ty_p - 2.0 * T_c + Ty_m) * inv_dy2)
                dT = diff * lap * dt
                delta_T_per_cell[k, j, i] = dT
                T_s_grid[k, j, i] = T_c + dT

    # 2. Scatter delta_T_per_cell back to each alive particle.
    N_max = part_alive.shape[0]
    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue
        i = int(part_x[p] / dx)
        j = int(part_y[p] / dy)
        if i < 0 or i >= Nx or j < 0 or j >= Ny:
            continue
        k = locate_k_from_z(part_z[p], z_face, Nz)
        if k < 0 or k >= n_z_bed:
            continue
        part_T_s[p] += delta_T_per_cell[k, j, i]


@njit(cache=True)
def aggregate_particles_to_T_s_grid(
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_alive: np.ndarray,
    part_m_solid: np.ndarray, part_m_water: np.ndarray,
    part_m_char: np.ndarray, part_T_s: np.ndarray,
    dx: float, dy: float, z_face: np.ndarray,
    T_s_grid: np.ndarray,
    T_amb: float,
) -> None:
    """Mirror per-particle T_s into per-cell Eulerian T_s grid.

    Mass-weighted average T_s within each cell that contains particles;
    cells WITHOUT particles are left at T_amb (downstream DOM kernel
    expects T_s ≥ T_amb).

    Used as a diagnostic + radiation feed when the bed-particle path
    is active.  Sequential outer loop for Rule #17 determinism.
    """
    Nz, Ny, Nx = T_s_grid.shape
    # Working arrays (allocated by caller could be better, but this is
    # called once per outer step so the alloc cost is acceptable).
    num = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    den = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    N_max = part_alive.shape[0]
    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue
        xi = int(part_x[p] / dx)
        yj = int(part_y[p] / dy)
        if xi < 0 or xi >= Nx or yj < 0 or yj >= Ny:
            continue
        zk = locate_k_from_z(part_z[p], z_face, Nz)
        if zk < 0:
            continue
        m_t = part_m_solid[p] + part_m_water[p] + part_m_char[p]
        if m_t <= 0.0:
            continue
        num[zk, yj, xi] += part_T_s[p] * m_t
        den[zk, yj, xi] += m_t
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if den[k, j, i] > 0.0:
                    T_s_grid[k, j, i] = num[k, j, i] / den[k, j, i]
                # else leave T_s_grid unchanged (non-bed cells keep
                # whatever value they had — typically T_amb)


@njit(cache=True)
def aggregate_particles_to_M_local_grid(
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_alive: np.ndarray,
    part_m_solid: np.ndarray, part_m_water: np.ndarray,
    dx: float, dy: float, z_face: np.ndarray,
    M_local_grid: np.ndarray,
) -> None:
    """Aggregate per-particle M = m_water/m_solid into per-cell grid.

    Sums water and dry-solid mass independently across particles within
    each cell, then computes cell-level M_local = Σm_water / Σm_solid.
    Cells without particles get M_local = 0 (no bed → no moisture).

    Used by DOM radiation solver to scale κ_solid by (1 + β·M_local)
    per Mell 2007 WFDS / Linn 2002 FIRETEC (wet bed absorbs more IR
    radiation per kg of solid).  Sequential outer loop for Rule #17
    determinism.
    """
    Nz, Ny, Nx = M_local_grid.shape
    num = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    den = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    N_max = part_alive.shape[0]
    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue
        xi = int(part_x[p] / dx)
        yj = int(part_y[p] / dy)
        if xi < 0 or xi >= Nx or yj < 0 or yj >= Ny:
            continue
        zk = locate_k_from_z(part_z[p], z_face, Nz)
        if zk < 0:
            continue
        num[zk, yj, xi] += part_m_water[p]
        den[zk, yj, xi] += part_m_solid[p]
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if den[k, j, i] > 0.0:
                    M_local_grid[k, j, i] = num[k, j, i] / den[k, j, i]
                else:
                    M_local_grid[k, j, i] = 0.0


@njit(cache=True)
def step_bed_particles(
    # particle state (modified)
    part_x: np.ndarray, part_y: np.ndarray, part_z: np.ndarray,
    part_alive: np.ndarray,
    part_m_solid: np.ndarray, part_m_water: np.ndarray,
    part_m_char: np.ndarray, part_T_s: np.ndarray,
    part_m_water_0: np.ndarray, part_sav: np.ndarray,
    # gas state (read at particle locations)
    T_g_grid: np.ndarray,
    Y_O2_grid: np.ndarray,
    # External solid heat source per cell (W/m³, positive = heat into solid).
    # Drip torch, bootstrap, radiation absorption, etc.  Distributed
    # uniformly across the n_per_cell particles in each cell — the
    # particle-side update is dT_s = Q_external * V_cell * dt / (n_per_cell * m_p * cp).
    Q_solid_ext_per_cell: np.ndarray,
    n_per_cell_for_split: int,
    # gas-cell aggregation sources (overwritten this step)
    S_pyro_per_cell: np.ndarray,
    S_drying_per_cell: np.ndarray,
    Q_pyro_per_cell: np.ndarray,
    Q_drying_per_cell: np.ndarray,
    Y_F_source_per_cell: np.ndarray,
    Q_char_per_cell: np.ndarray,
    Q_smold_per_cell: np.ndarray,
    Q_g_conv_per_cell: np.ndarray,   # gas → particle convective heat extraction
                                      # [W/m³, positive = gas loses heat]
    # grid
    dx: float, dy: float, dz_arr: np.ndarray, z_face: np.ndarray,
    # physics params
    h_conv: float,
    rho_solid_true: float,
    cp_solid: float,
    eps_solid: float,
    T_amb: float,
    view_factor: float,        # [0..1] scalar multiplier on Stefan-Boltzmann
                                # loss.  When view_factor_geometric=False,
                                # used as-is.  When True, MULTIPLIED with the
                                # per-particle Beer-Lambert f_geom (so 1.0 +
                                # geometric=True gives pure depth-based).
    view_factor_geometric: bool, # If True, compute f_geom_p = exp(-κ·(h_bed−z_p))
                                # per particle, where κ ≈ sav·α_s_avg ≈ effective
                                # bed absorption coefficient.  Particles deep in
                                # the bed get f_geom → 0 (emission reabsorbed);
                                # surface particles get f_geom → 1.
    h_bed: float,              # bed top z-coordinate (m), for geometric mode
    kappa_bed_eff: float,      # effective bed absorption (1/m), for geometric
    dt: float,
    # process toggles (default all True)
    do_drying: bool,
    do_pyrolysis: bool,
    do_char_ox: bool,
    do_smolder: bool,
    # Drying mode selector:
    #   0 = Lautenberger 2009 Arrhenius (white pine bound-water kinetics)
    #   1 = FIRETEC heat-rate-limited equilibrium (Linn 2002 / Pimont &
    #       Linn 2009).  Particle T_s pinned at T_BOIL while m_water>0;
    #       excess heat above pinning goes to evaporation.  Recovers
    #       Cheney 1993 moisture sensitivity (mass-time scaling) that
    #       first-order Arrhenius cannot reproduce.
    drying_mode: int,
    # Phase 20 char-ox knobs (Cheney sweep GIF diagnostic).
    #   char_ox_flux_cap_W_m2 : surface-area-scaled cap on char Q (default
    #     1.0e5 W/m² per Williams 1985 low-end; bump to 2.5e5 for
    #     Williams 1985 clean-char peak; see phase20_char_ox_investigation).
    #   char_ox_ash_exp : ash-coverage penalty exponent (0.0 = disabled;
    #     >0 = A_reactive = A_geom × (m_char / m_char_max)^ash_exp).
    #     Requires part_m_char_max tracking (updated in pyrolysis branch).
    char_ox_flux_cap_W_m2: float,
    char_ox_ash_exp: float,
    part_m_char_max: np.ndarray,   # per-particle, updated when pyrolysis
                                    # produces char (max ratchet).  Used
                                    # only if char_ox_ash_exp > 0.
    # diagnostics (length-1 int64)
    n_alive_out: np.ndarray,
    n_burned_out: np.ndarray,
    # Hot-particle diagnostics (length-16 float64) — captures budget
    # at the particle with max T_s after this step.  Layout:
    #  [0] T_s_max         [1] Q_conv      [2] Q_rxn        [3] Q_dry
    #  [4] Q_char_ox       [5] Q_smold     [6] Q_ext        [7] Q_rad_loss
    #  [8] Q_other_total   [9] T_g_local  [10] mc/dt       [11] C_rad
    # [12] Newton_F_final [13] Newton_max_F_global
    # [14] A_p            [15] m_total_p
    diag_max_out: np.ndarray,
) -> None:
    """One step of drying + pyrolysis + T_s convective update + aggregation.

    Per alive particle:
      1. Locate cell (i, j, k); skip + retire if out of domain
      2. Drying  (Arrhenius water evaporation, same A_DRY/E_DRY/L_VAP
         as Eulerian step_drying)
      3. Pyrolysis (MD2004 thermal + R_op, same constants as
         step_pyrolysis_md2004, with moisture gate)
      4. T_s update via gas-particle convection + endo/exo Q
      5. Aggregate sources to gas cell (volumetric)
      6. Retire if total mass < burnout threshold

    Gas-cell aggregation arrays are OVERWRITTEN at the start of this
    call — caller does NOT need to zero them.  This matches the Eulerian
    kernel convention (S_pyro_out etc. are overwritten).

    Determinism: sequential outer loop (Rule #17 bit-exact).  At expected
    particle counts (~1M for mickey-class at n_per_cell=50, ~few-hundred-k
    for FIRETEC-class) this is dominated by the per-particle Arrhenius
    eval, not the loop overhead.
    """
    Nz, Ny, Nx = T_g_grid.shape
    N_max = part_alive.shape[0]
    n_alive = 0
    n_burned = 0

    # Zero aggregation arrays (kernel overwrites; caller need not zero)
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                S_pyro_per_cell[k, j, i] = 0.0
                S_drying_per_cell[k, j, i] = 0.0
                Q_pyro_per_cell[k, j, i] = 0.0
                Q_drying_per_cell[k, j, i] = 0.0
                Y_F_source_per_cell[k, j, i] = 0.0
                Q_char_per_cell[k, j, i] = 0.0
                Q_smold_per_cell[k, j, i] = 0.0
                Q_g_conv_per_cell[k, j, i] = 0.0

    inv_dt = 1.0 / dt if dt > 0.0 else 0.0
    inv_cp_solid = 1.0 / cp_solid

    # Reset diag (track max T_s and the budget at that particle)
    for _di in range(diag_max_out.shape[0]):
        diag_max_out[_di] = 0.0
    _T_s_max_seen = 0.0
    _newton_max_F_global = 0.0

    for p in range(N_max):
        if part_alive[p] == ALIVE_FALSE:
            continue

        # Locate cell
        i, j, k = locate_cell(part_x[p], part_y[p], part_z[p],
                              dx, dy, z_face, Nz, Nx, Ny)
        if i < 0 or j < 0 or k < 0:
            part_alive[p] = ALIVE_FALSE
            continue

        V_cell = dx * dy * dz_arr[k]
        T_s = part_T_s[p]
        T_g = T_g_grid[k, j, i]
        Y_O2 = Y_O2_grid[k, j, i]

        m_solid_p = part_m_solid[p]
        m_water_p = part_m_water[p]
        m_char_p  = part_m_char[p]
        m_water_0 = part_m_water_0[p]
        sav_p     = part_sav[p]

        # ── (1) Drying — Arrhenius water evaporation ──
        # DRY_MODE_ARRHENIUS:   Lautenberger 2009 white-pine bound-water
        #                       kinetics (A_DRY, E_DRY).
        # DRY_MODE_EQUILIBRIUM: skip Arrhenius — equilibrium override
        #                       (step 6.5) does all evaporation.
        # DRY_MODE_COMBINED:    grass-tuned Arrhenius (A_DRY_GRASS,
        #                       E_DRY_GRASS) for sub-boil drying +
        #                       equilibrium override for above-boil.
        dm_evap = 0.0
        if do_drying and m_water_p > 0.0 and T_s > 0.0:
            if drying_mode == DRY_MODE_ARRHENIUS:
                k_dry = A_DRY * math.exp(-E_DRY / (_R_GAS * T_s))
                mw_new = m_water_p * math.exp(-k_dry * dt)
                dm_evap = m_water_p - mw_new
                m_water_p = mw_new
            elif drying_mode == DRY_MODE_COMBINED:
                k_dry = A_DRY_GRASS * math.exp(-E_DRY_GRASS / (_R_GAS * T_s))
                mw_new = m_water_p * math.exp(-k_dry * dt)
                dm_evap = m_water_p - mw_new
                m_water_p = mw_new

        # ── (2) Pyrolysis — MD2004 thermal + R_op with moisture gate ──
        dm_pyro = 0.0
        rate_pyro = 0.0
        f_thermal = 1.0
        f_op = 0.0
        if do_pyrolysis and m_solid_p > 0.0 and T_s > 0.0:
            # Moisture gate: linear soft-ramp suppression of pyrolysis
            # by water vapor competing for cell-wall sites + cooling
            # the particle (Mell 2007 WFDS pattern).  Phase 16 fix
            # (2026-06-18): switched from (1 - 100·wet) hard-cutoff to
            # linear (1 - wet).  The hard-cutoff blocked pyrolysis
            # entirely when wet > 1%, producing a discontinuous
            # ignition delay (M=0% ignites in 0.4s, M=5% takes 28s in
            # cone-density test).  Linear gate matches cone calorimeter
            # literature t_ig(M) across the full M range.
            if m_water_0 > 0.0:
                wet = m_water_p / m_water_0
                moist_gate = 1.0 - wet
                if moist_gate < 0.0:
                    moist_gate = 0.0
            else:
                moist_gate = 1.0

            if moist_gate > 0.0:
                k_thermal = A_MD2004 * math.exp(-E_MD2004 / (_R_GAS * T_s))
                if Y_O2 > Y_O2_MIN_OP:
                    k_op = (A_OP_MD2004
                            * math.exp(-E_OP_MD2004 / (_R_GAS * T_s))
                            * Y_O2 ** N_O2_OP)
                else:
                    k_op = 0.0
                k_total = k_thermal + k_op
                if k_total > 0.0:
                    m_solid_new = m_solid_p * math.exp(-k_total * dt)
                    dm_full = m_solid_p - m_solid_new
                    dm_pyro = dm_full * moist_gate
                    rate_pyro = dm_pyro * inv_dt
                    f_thermal = k_thermal / k_total
                    f_op = k_op / k_total
                    m_char_p  += CHAR_YIELD_MD2004 * dm_pyro
                    m_solid_p -= dm_pyro
                    # Phase 20 C: ratchet m_char_max as pyrolysis produces char.
                    # Char oxidation later uses this as ash-coverage reference.
                    if m_char_p > part_m_char_max[p]:
                        part_m_char_max[p] = m_char_p

        # Current condensed-phase surface area (used by char_ox / smolder
        # caps below and by T_s update).  Includes m_char — char retains
        # the geometric surface that participates in oxidation.
        _A_p_now = sav_p * (m_solid_p + m_char_p) / rho_solid_true
        # ── (3) Char oxidation — consumes m_char ──
        # Per-particle Arrhenius: m_dot = A_CHAR · exp(-E_CHAR/RT_s) · m_char · Y_O2
        # Exothermic: heats particle, gas cell gains Q.  No mass added to gas
        # (CO2 product not tracked separately — matches Eulerian convention).
        # Surface-area-scaled flux cap: Q_char ≤ Q_CHAR_FLUX_MAX × A_p.
        # The flux value (1×10⁵ W/m²) matches the equivalent of the
        # Eulerian volumetric cap (5×10⁵ W/m³) at the GR1 grass packing
        # (sav·α_s ≈ 5.6 1/m), and is consistent with char-oxidation
        # surface fluxes reported in Williams 1985.  Surface-area
        # scaling is required for the Lagrangian path — as char depletes,
        # A_p shrinks and the cap MUST shrink with it, else burnt-out
        # particles climb to 10,000 K under a constant volumetric cap.
        # Phase 20 B: Q_CHAR_FLUX_MAX is now runtime (default 1.0e5).
        # Phase 20 C: ash-coverage penalty on reactive surface area.  When
        # char_ox_ash_exp > 0: A_reactive = A_geom × (m_char/m_char_max)^exp.
        # As char depletes (m_char < m_char_max), ash accumulates and
        # blocks O2 access → effective A drops faster than mass.
        dm_char_ox = 0.0
        Q_char_ox_p = 0.0
        if (do_char_ox and m_char_p > 1.0e-12 and T_s >= T_CHAR_ONSET
                and Y_O2 >= Y_O2_MIN_CHAR):
            k_ch = A_CHAR * math.exp(-E_CHAR / (_R_GAS * T_s))
            m_dot_ch = k_ch * m_char_p * Y_O2
            m_cons_ch = m_dot_ch * dt
            if m_cons_ch > 0.5 * m_char_p:
                m_cons_ch = 0.5 * m_char_p
            _A_reactive = _A_p_now
            if char_ox_ash_exp > 0.0 and part_m_char_max[p] > 1.0e-15:
                _burn_frac = m_char_p / part_m_char_max[p]
                if _burn_frac < 0.0:
                    _burn_frac = 0.0
                _A_reactive *= _burn_frac ** char_ox_ash_exp
            _Q_cap_part = char_ox_flux_cap_W_m2 * _A_reactive
            _Q_arrh_part = (m_cons_ch * HOC_CHAR) * inv_dt
            if _Q_arrh_part > _Q_cap_part:
                m_cons_ch = _Q_cap_part / (HOC_CHAR * inv_dt)
            dm_char_ox = m_cons_ch
            Q_char_ox_p = (dm_char_ox * HOC_CHAR) * inv_dt
            m_char_p -= dm_char_ox

        # ── (4) Smoldering oxidation — consumes m_solid + m_char surface ──
        # Slow low-T surface oxidation; same Arrhenius form, lower E.
        # Consumes from m_solid (cellulosic fines) + m_char (already
        # oxidatively-prepared surface).  Eulerian path consumes m_solid only;
        # particle path includes m_char too because in the sub-grid view both
        # surfaces are available.
        # Q-magnitude cap (Q_SMOLD_MAX = 2×10⁵ W/m³ per cell) matches the
        # Eulerian step_smoldering_oxidation safeguard (pyrolysis_3d.py:562).
        dm_smold_solid = 0.0
        dm_smold_char  = 0.0
        Q_smold_p      = 0.0
        if (do_smolder and T_s >= T_SMOLD_ONSET and Y_O2 >= Y_O2_MIN_SMOLD):
            m_avail_sm = m_solid_p + m_char_p
            if m_avail_sm > 1.0e-12:
                k_sm = A_SMOLD * math.exp(-E_SMOLD / (_R_GAS * T_s))
                m_dot_sm = k_sm * m_avail_sm * Y_O2
                m_cons_sm = m_dot_sm * dt
                if m_cons_sm > 0.5 * m_avail_sm:
                    m_cons_sm = 0.5 * m_avail_sm
                # Surface-area-scaled Q cap (flux 4×10⁴ W/m², ~Q_SMOLD_MAX
                # equivalent for grass packing — smolder is gentler than
                # char_ox by the 2/5 volumetric ratio).
                Q_SMOLD_FLUX_MAX = 4.0e4
                _Q_cap_sm_part = Q_SMOLD_FLUX_MAX * _A_p_now
                _Q_arrh_sm = (m_cons_sm * HOC_SMOLD) * inv_dt
                if _Q_arrh_sm > _Q_cap_sm_part:
                    m_cons_sm = _Q_cap_sm_part / (HOC_SMOLD * inv_dt)
                # Split consumption across m_solid + m_char in their ratio
                if m_avail_sm > 0.0:
                    f_solid_sm = m_solid_p / m_avail_sm
                    dm_smold_solid = f_solid_sm * m_cons_sm
                    dm_smold_char  = (1.0 - f_solid_sm) * m_cons_sm
                    m_solid_p -= dm_smold_solid
                    m_char_p  -= dm_smold_char
                Q_smold_p = (m_cons_sm * HOC_SMOLD) * inv_dt   # W per particle

        # ── (5) Aggregate to gas cell as volumetric sources ──
        inv_V = 1.0 / V_cell
        S_pyro_per_cell[k, j, i]   += ETA_MD2004 * rate_pyro * inv_V
        S_drying_per_cell[k, j, i] += dm_evap * inv_dt * inv_V
        # Heat of reaction: thermal endothermic (positive Q absorbed by
        # solid), R_op exothermic (negative HOR → released).  Sign here
        # matches Eulerian step_pyrolysis_md2004 (Q_pyro_out[k,j,i] in
        # W/m³ sign convention: positive = solid heat sink).
        Q_pyro_per_cell[k, j, i] += (rate_pyro * (
            f_thermal * HEAT_OF_PYROLYSIS
            + f_op * HOR_OP_MD2004
        )) * inv_V
        Q_drying_per_cell[k, j, i] += dm_evap * L_VAP_WATER * inv_dt * inv_V
        # Y_F mass-fraction source: ETA fraction of pyrolysate is fuel gas
        Y_F_source_per_cell[k, j, i] += ETA_MD2004 * rate_pyro * inv_V
        # Char-ox + smolder volumetric heat release (W/m³).  These match
        # the Eulerian step_char_oxidation / step_smoldering_oxidation
        # sign convention (POSITIVE = heat released, transferred to gas
        # via T_g rise in coupling).
        Q_char_per_cell[k, j, i]  += Q_char_ox_p * inv_V
        Q_smold_per_cell[k, j, i] += Q_smold_p * inv_V
        # Gas → particle convective heat extraction.  Computed below in
        # the T_s update; aggregated here.  POSITIVE = gas loses heat
        # (which goes to the particle).

        # ── (6) T_s update — convection + reaction enthalpies ──
        # A_p = sav · V_condensed = sav · (m_solid + m_char) / rho_solid_true
        # Post-pyrolysis the particle is char (not gone) — char retains the
        # original geometric surface and radiates / convects identically.
        # Excluding m_char here was a bug: it zeroed A_p once pyrolysis
        # completed, killing Stefan-Boltzmann loss + Q_conv, and let Q_char_ox
        # + Q_smold heat the char unboundedly to the safety cap.
        m_total_p = m_solid_p + m_water_p + m_char_p
        Q_conv = 0.0
        if m_total_p > 0.0:
            A_p = sav_p * (m_solid_p + m_char_p) / rho_solid_true
            # Convection: positive when T_g > T_s (heats particle)
            Q_conv = h_conv * A_p * (T_g - T_s)
            # Endo/exo from pyrolysis (per particle, in W)
            # Sign: HEAT_OF_PYROLYSIS positive → endo → cools particle
            #       HOR_OP_MD2004 negative → exo → heats particle
            Q_rxn = -rate_pyro * (
                f_thermal * HEAT_OF_PYROLYSIS
                + f_op * HOR_OP_MD2004
            )
            # Drying: endothermic → cools particle
            Q_dry_part = -dm_evap * L_VAP_WATER * inv_dt
            # Char-ox + smolder: exothermic → heats particle (Q already W)
            # External cell heat → split uniformly across n_per_cell
            # particles in the cell.  Q_external [W/m³] × V_cell [m³] =
            # total W into the cell; / n_per_cell_for_split = W per particle.
            Q_ext_p = 0.0
            if n_per_cell_for_split > 0:
                Q_ext_p = Q_solid_ext_per_cell[k, j, i] * V_cell / n_per_cell_for_split
            # Stefan-Boltzmann radiation loss from particle surface,
            # scaled by an effective view factor.  view_factor (scalar)
            # always applies; if view_factor_geometric, additionally
            # multiply by per-particle Beer-Lambert based on depth
            # below bed top (h_bed - z_p).  Surface particles (z_p
            # near h_bed) get f_geom ≈ 1; deep particles get f_geom ≈ 0
            # (their emission is reabsorbed by neighbors via DOM).
            _f_geom = 1.0
            if view_factor_geometric:
                _h_above = h_bed - part_z[p]
                if _h_above < 0.0:
                    _h_above = 0.0
                _f_geom = math.exp(-kappa_bed_eff * _h_above)
            _C_rad = eps_solid * SIGMA_SB * A_p * view_factor * _f_geom
            # ── Newton-iterated implicit Stefan-Boltzmann ───────────────
            # Solves the nonlinear T_new equation:
            #   m·cp·(T_new − T_old)/dt = Q_other − C·(T_new⁴ − T_amb⁴)
            # via 5 Newton iterations.  Single-linearization (FIRESTAR-
            # style) is too weak when 4·C·T³ ≪ m·cp/dt (always true for
            # our particle sizes); Newton converges quadratically to the
            # true plateau set by Q_other = C·(T_plateau⁴ − T_amb⁴).
            _mc = m_total_p * cp_solid
            _T_amb4 = T_amb * T_amb * T_amb * T_amb
            _Q_other = (Q_conv + Q_rxn + Q_dry_part
                        + Q_char_ox_p + Q_smold_p + Q_ext_p)
            _mc_inv_dt = _mc / dt
            T_old_for_newton = T_s
            T_iter = T_s
            _F = 0.0
            for _newt in range(5):
                _T_iter2 = T_iter * T_iter
                _T_iter3 = _T_iter2 * T_iter
                _T_iter4 = _T_iter3 * T_iter
                # F(T_iter) = 0 at convergence
                _F = (_mc_inv_dt * (T_iter - T_old_for_newton)
                      - _Q_other + _C_rad * (_T_iter4 - _T_amb4))
                _Fp = _mc_inv_dt + 4.0 * _C_rad * _T_iter3
                T_iter = T_iter - _F / _Fp
                # Guard: keep iteration within physical range
                if T_iter < T_amb:
                    T_iter = T_amb
            T_s = T_iter
            Q_rad_loss_p = _C_rad * (T_s**4 - _T_amb4)
            # Track Newton residual magnitude (global max of final |F|)
            _abs_F = _F if _F >= 0.0 else -_F
            if _abs_F > _newton_max_F_global:
                _newton_max_F_global = _abs_F
            # Capture diagnostics if this is the hottest particle so far
            if T_s > _T_s_max_seen:
                _T_s_max_seen = T_s
                diag_max_out[0]  = T_s
                diag_max_out[1]  = Q_conv
                diag_max_out[2]  = Q_rxn
                diag_max_out[3]  = Q_dry_part
                diag_max_out[4]  = Q_char_ox_p
                diag_max_out[5]  = Q_smold_p
                diag_max_out[6]  = Q_ext_p
                diag_max_out[7]  = Q_rad_loss_p
                diag_max_out[8]  = _Q_other
                diag_max_out[9]  = T_g
                diag_max_out[10] = _mc_inv_dt
                diag_max_out[11] = _C_rad
                diag_max_out[12] = _abs_F
                diag_max_out[14] = A_p
                diag_max_out[15] = m_total_p
            # Hard safety cap at 10,000 K — should never bind if Newton
            # converged properly; backstop for pathological inputs.
            if T_s > 1.0e4:
                T_s = 1.0e4

            # ── (6.5) FIRETEC heat-rate-limited equilibrium drying ─────
            # When drying_mode == DRY_MODE_EQUILIBRIUM, pin T_s at T_BOIL
            # while m_water > 0 and divert any heat above the pin to
            # latent evaporation.  Linn 2002 / Pimont & Linn 2009.
            # Implementation: any excess energy above (T_BOIL × m·cp) is
            # the heat that would have raised T above boiling; redirect
            # it to evaporation at L_VAP cost per kg.  When water is
            # exhausted, residual energy continues heating the particle.
            if (do_drying
                    and (drying_mode == DRY_MODE_EQUILIBRIUM
                         or drying_mode == DRY_MODE_COMBINED)
                    and m_water_p > 0.0 and T_s > T_BOIL_WATER):
                _excess_J = _mc * (T_s - T_BOIL_WATER)
                _dm_eq = _excess_J / L_VAP_WATER
                if _dm_eq >= m_water_p:
                    # All water evaporates, residual heat raises T above boil
                    _residual_J = _excess_J - m_water_p * L_VAP_WATER
                    _dm_eq = m_water_p
                    m_water_p = 0.0
                    T_s = T_BOIL_WATER + (_residual_J / _mc if _mc > 0.0 else 0.0)
                else:
                    m_water_p -= _dm_eq
                    T_s = T_BOIL_WATER
                # Aggregate the evaporation flux to gas-source arrays
                # (matches the arrhenius-mode bookkeeping at line ~688)
                S_drying_per_cell[k, j, i]  += _dm_eq * inv_dt * inv_V
                Q_drying_per_cell[k, j, i]  += _dm_eq * L_VAP_WATER * inv_dt * inv_V
                # Also accumulate to dm_evap for the burnout / mass total
                dm_evap += _dm_eq

        # Aggregate gas heat sink (sign: positive = gas loses heat to
        # particle).  Convection moves Q_conv W of heat from gas to
        # particle; gas-side this is a heat sink, particle-side a gain.
        Q_g_conv_per_cell[k, j, i] += Q_conv * inv_V

        # Write back particle state
        part_T_s[p]    = T_s
        part_m_solid[p] = m_solid_p
        part_m_water[p] = m_water_p
        part_m_char[p]  = m_char_p

        # ── (7) Burnout check ──
        if m_total_p < M_PARTICLE_BURNOUT:
            part_alive[p] = ALIVE_FALSE
            n_burned += 1
        else:
            n_alive += 1

    n_alive_out[0] = n_alive
    n_burned_out[0] = n_burned
    # Global Newton residual max written after all particles done
    diag_max_out[13] = _newton_max_F_global
