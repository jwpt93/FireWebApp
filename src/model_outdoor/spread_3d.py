"""3D PDE fire-spread model — Phase 13 bottom-up rebuild.

This module is the integration entry point.  Physics is modularised
under physics_3d/ ; this file holds the cell-state allocation, grid
setup, time-loop integration, and a public ``run_3d_spread`` function.

Design (per plans/a-is-what-we-snug-haven.md):
- Lumped-species gas: every cell tracks alpha_solid, Y_fuel, T_g, u, v, w.
  Y_air = 1 - Y_fuel by closure.
- Solid phase: only where alpha_solid > 0 (bed cells).  Per-cell mass
  pools m_hemi, m_cell, m_lign and temperature T_s.
- Laminar baseline; turbulence modules are conditional Phase D2.
- y-BC selectable: 'periodic' (default; infinite-fire-line interpretation)
  or 'edge_loss' (ghost-cell damping for finite fire line).
- Compute: NumPy arrays, Numba @njit(parallel=True) loops in physics_3d.

Physics summary (governing equations) — see plan file for full form:
  Gas:      continuity, momentum (Boussinesq+drag), Y_fuel transport,
            energy (T_g) with combustion source.
  Solid:    3-pool Arrhenius pyrolysis, energy with rad/conv coupling.
  Combust:  omega = min(omega_mix, omega_chem)
            (Magnussen 1976; Westbrook & Dryer 1981).
  Radiat:   Beer-Lambert slab (Albini 1985).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from model.io.text_input import RomInputs, load_text_input
from model_outdoor.boundary import (
    byram_flame_length, flame_tilt_angle, midflame_wind_speed,
    raupach_d_z0,
    wind_profile_in_bed,
    wind_profile_log_law,
    wind_profile_canopy_bl,
)
from model_outdoor.config import outdoor_env_from_dict
from model_outdoor.physics_3d import (
    combustion_3d, coupling_3d, dom_3d, drag_3d, flame_front_3d,
    momentum_3d, projection_3d, pyrolysis_3d, radiation_3d,
    solid_conduction_3d, species_3d, turbulence_3d,
)
from model_outdoor.physics_3d import chemistry_closures
from model_outdoor.physics_3d.chemistry_closures._constants import (
    resolve_chemistry_family as _resolve_chemistry_family,
)
from model_outdoor.boundary_conditions import (
    get_bc_class as _get_bc_class,
    available as _bc_available,
)
from model_outdoor.physics_3d import level_set_fsd_3d
from model_outdoor.physics_3d import finney_burst_3d
from model_outdoor.physics_3d import finney_tendril_3d
from model_outdoor.physics_3d import finney_lagrangian_3d
from model_outdoor.physics_3d import lagrangian_bed_3d
from model_outdoor import empirical_ros as _empirical_ros


# ── Physical constants ────────────────────────────────────────────────────────
# Air properties at standard conditions (Drysdale 2011, Table 2.4).
_R_UNIV = 8.314           # [J/mol/K]
_M_AIR  = 0.029           # [kg/mol]
_R_AIR  = _R_UNIV / _M_AIR  # [J/kg/K]
_P0     = 101_325.0       # [Pa] thermodynamic pressure (low-Mach)
_RHO_GAS_REF = 1.2        # [kg/m³] reference air density at 300 K
_CP_GAS = 1100.0          # [J/kg/K] hot-air-mixture cp (Drysdale 2011)
_MU_GAS = 1.8e-5          # [Pa·s] dynamic viscosity (Drysdale 2011)
_K_GAS  = 0.026           # [W/m/K] air thermal conductivity
_PR_GAS = 0.7             # Prandtl number

# Solid (dry cellulosic biomass) reference properties.
# Drysdale (2011); Janssens (1993); Koufopanos (1991).
_RHO_PARTICLE = 500.0     # [kg/m³] dry biomass particle density
_CP_SOLID     = 1300.0    # [J/kg/K] dry biomass heat capacity
_HOC_J        = 14_900_000.0  # [J/kg] heat of combustion (Sung et al. 2025)
_T_AMB_DEFAULT = 300.0    # [K]
_G = 9.81                 # [m/s²]


# ── Cell-state container ──────────────────────────────────────────────────────
@dataclass
class CellState3D:
    """All state arrays of the 3D PDE on a (Nz, Ny, Nx) grid.

    Each field is a NumPy array shape (Nz, Ny, Nx), dtype float64.
    Numba JIT functions in physics_3d operate on these arrays directly
    (passed as named arguments, not the dataclass).
    """

    # Gas phase
    rho:    np.ndarray   # [kg/m³] gas density (from EoS each step)
    u:      np.ndarray   # [m/s] x-velocity
    v:      np.ndarray   # [m/s] y-velocity
    w:      np.ndarray   # [m/s] z-velocity
    p_dyn:  np.ndarray   # [Pa] dynamic pressure (projection)
    T_g:    np.ndarray   # [K] gas temperature
    Y_fuel: np.ndarray   # [-] fuel-gas mass fraction
    Y_O2:   np.ndarray   # [-] O₂ mass fraction (transported; init = 0.232 fresh air)
    Y_H2O:  np.ndarray   # [-] water-vapor mass fraction (transported; source
                          # = drying evaporation; required for Cheney moisture
                          # sensitivity AND Rule #0 suppression validation).
                          # Y_inert (N2 + combustion CO2) is implicit:
                          # 1 − Y_fuel − Y_O2 − Y_H2O.
    Y_CO:   np.ndarray   # [-] CO mass fraction (transported).  Phase 23
                          # 2-step Westbrook-Dryer intermediate.  Zero for
                          # all pre-Phase-23 cases (single-step biomass
                          # closure never produces CO); populated only by
                          # combustion_closure="edc_2step_methane".

    # Solid phase (zero in buffer cells)
    alpha_s: np.ndarray  # [-] solid volume fraction
    T_s:     np.ndarray  # [K] solid temperature
    m_hemi:  np.ndarray  # [kg/m³] hemicellulose mass per cell volume
    m_cell:  np.ndarray  # [kg/m³] cellulose
    m_lign:  np.ndarray  # [kg/m³] lignin
    m_char:  np.ndarray  # [kg/m³] accumulated char from pyrolysis (Phase 14
                          # Try 5b, Yang 2007 char yields).  step_char_oxidation
                          # and step_smoldering_oxidation consume m_char (not
                          # the lumped m_hemi bulk-fuel pool).

    @classmethod
    def allocate(cls, Nz: int, Ny: int, Nx: int, T_amb: float) -> "CellState3D":
        """Allocate all state arrays at ambient/zero initial conditions.

        Y_O2 is initialized to fresh-air value 0.232 (Drysdale 2011
        Table 2.4) — combustion products + N₂ are lumped via implicit
        Y_inert = 1 − Y_fuel − Y_O2.
        """
        shape = (Nz, Ny, Nx)
        zero = np.zeros(shape, dtype=np.float64)
        rho_amb = _P0 / (_R_AIR * T_amb)
        return cls(
            rho    = np.full(shape, rho_amb, dtype=np.float64),
            u      = zero.copy(),
            v      = zero.copy(),
            w      = zero.copy(),
            p_dyn  = zero.copy(),
            T_g    = np.full(shape, T_amb, dtype=np.float64),
            Y_fuel = zero.copy(),
            Y_O2   = np.full(shape, 0.232, dtype=np.float64),
            Y_H2O  = zero.copy(),
            Y_CO   = zero.copy(),
            alpha_s = zero.copy(),
            T_s     = np.full(shape, T_amb, dtype=np.float64),
            m_hemi  = zero.copy(),
            m_cell  = zero.copy(),
            m_lign  = zero.copy(),
            m_char  = zero.copy(),
        )


# ── Grid container ────────────────────────────────────────────────────────────
@dataclass
class Grid3D:
    """3D Cartesian grid metadata with optional non-uniform vertical spacing.

    Phase 14g: dz becomes a per-cell ARRAY (``dz_arr``) instead of a scalar
    to support geometric expansion of buffer cells above the bed (saves
    ~50% z-cells while preserving in-bed resolution).  ``dz`` is kept as a
    "nominal" scalar (= ``dz_arr[0]``, the bed-cell value) for backward-
    compatible external API.  Kernels that need per-cell spacing read
    ``dz_arr[k]``, ``inv_dz_arr[k]``, and the precomputed face distances.
    """

    Nx: int; Ny: int; Nz: int
    dx: float; dy: float; dz: float    # dz: nominal (bed-cell) spacing
    Lx: float; Ly: float; Lz: float
    n_z_bed: int  # number of bed layers (z indices [0, n_z_bed))

    x_mid: np.ndarray  # [Nx]   cell-center x coords
    y_mid: np.ndarray  # [Ny]   cell-center y coords
    z_mid: np.ndarray  # [Nz]   cell-center z coords (non-uniform if expansion)

    # Phase 14g — non-uniform-z support
    dz_arr: np.ndarray        # [Nz]   per-cell vertical spacing
    inv_dz_arr: np.ndarray    # [Nz]   1/dz_arr (precomputed)
    z_face: np.ndarray        # [Nz+1] cell face positions in z (z_face[0]=0)
    d_face_above: np.ndarray  # [Nz]   distance from cell k to cell k+1 center
                              #        (for k=Nz-1, distance to ghost = dz_arr[k]/2)
    d_face_below: np.ndarray  # [Nz]   distance from cell k to cell k-1 center
                              #        (for k=0, distance to ghost = dz_arr[k]/2)

    @classmethod
    def build(cls, Lx: float, Ly: float, Lz: float, dx: float,
              h_bed: float, n_z_bed: int,
              dy: float | None = None,
              dz_expansion: float = 1.0,
              dz_first: float | None = None,
              bl_growth: float = 1.0,
              dz_first_above: float | None = None,
              bl_growth_above: float = 1.3,
              bed_refine_top: bool = False,
              # Phase 14ag: explicit BL kwargs for the new mesh kernel.
              # When any of these are set (>0), Grid3D.build delegates the
              # z-axis construction to model_outdoor.mesh.build_z_axis_bed_atm
              # which composes a clean segment stack: optional wall BL inside
              # the bed at z=0, uniform bulk bed, optional inner-solid BL at
              # z=h_bed, optional outer-air BL above z=h_bed, and a bulk
              # atmosphere with growing cells (capped at atm_max_dz).
              wall_bl_N: int = 0,
              wall_bl_first_dz: float = 0.0,
              wall_bl_growth: float = 1.3,
              bed_top_inner_bl_N: int = 0,
              bed_top_inner_bl_first_dz: float = 0.0,
              bed_top_inner_bl_growth: float = 1.3,
              bed_top_outer_bl_N: int = 0,
              bed_top_outer_bl_first_dz: float = 0.0,
              bed_top_outer_bl_growth: float = 1.3,
              atm_max_dz: float | None = None,
              atm_growth: float = 1.3,
              atm_uniform_dz: float | None = None) -> "Grid3D":
        """Build a 3D grid with ``dx`` cell width and optional z-expansion.

        ``h_bed`` is the fuel-bed height; bed gets ``n_z_bed`` cells of
        uniform ``dz_bed = h_bed / n_z_bed``.  Above the bed, buffer cell
        spacing geometrically expands by ``dz_expansion`` per cell:
            dz[n_z_bed + j] = dz_bed * dz_expansion^j  (j ≥ 0)
        With ``dz_expansion = 1.0`` (default), dz is uniform — backward-compat.

        Phase 14t-A — boundary-layer refinement at z=0 (cold flow only):
        When ``n_z_bed = 0`` AND ``dz_first`` is set, the lowest gas cells
        are geometrically refined: first cell of thickness ``dz_first``,
        each subsequent cell grows by ``bl_growth`` until the cell size
        reaches ``dx``, after which standard ``dz_expansion`` continues.
        This resolves the viscous sublayer for canonical TBL validation.

        For fire cases (``n_z_bed > 0``): bed cells stay uniform; BL
        params are ignored.  Bed sits on the wall, so there is no
        gas-phase BL to resolve at z=0.

        ``Lz`` should exceed ``h_bed`` so a buffer/flame zone is resolved.
        ``dy`` is set so dy ≈ dx.
        """
        if Lz < h_bed:
            raise ValueError(f"Lz={Lz} must exceed h_bed={h_bed}")
        if dz_expansion < 1.0:
            raise ValueError(f"dz_expansion={dz_expansion} must be ≥ 1.0")
        if bl_growth < 1.0:
            raise ValueError(f"bl_growth={bl_growth} must be ≥ 1.0")
        Nx = max(2, int(round(Lx / dx)))
        _dy_target = dy if dy is not None else dx
        Ny = max(1, int(round(Ly / _dy_target)))
        dy = Ly / Ny if Ny > 0 else 1.0

        # Phase 14ag: if ANY of the new explicit-BL flags are set, delegate
        # z-axis construction to model_outdoor.mesh.build_z_axis_bed_atm.
        # Otherwise fall through to the legacy code path (uniform bed,
        # optional dz_first stretch, dz_first_above, bed_refine_top, etc.)
        # for bit-identical backward compatibility.
        use_new_kernel = (wall_bl_N > 0
                          or bed_top_inner_bl_N > 0
                          or bed_top_outer_bl_N > 0)
        if use_new_kernel:
            from model_outdoor.mesh import build_z_axis_bed_atm
            dz_arr, n_z_bed = build_z_axis_bed_atm(
                h_bed=h_bed, Lz=Lz, n_z_bed=n_z_bed,
                wall_bl_N=wall_bl_N,
                wall_bl_first_dz=wall_bl_first_dz,
                wall_bl_growth=wall_bl_growth,
                bed_top_inner_bl_N=bed_top_inner_bl_N,
                bed_top_inner_bl_first_dz=bed_top_inner_bl_first_dz,
                bed_top_inner_bl_growth=bed_top_inner_bl_growth,
                bed_top_outer_bl_N=bed_top_outer_bl_N,
                bed_top_outer_bl_first_dz=bed_top_outer_bl_first_dz,
                bed_top_outer_bl_growth=bed_top_outer_bl_growth,
                atm_max_dz=atm_max_dz,
                atm_growth=atm_growth,
                atm_uniform_dz=atm_uniform_dz,
            )
            Nz = len(dz_arr)
            dz_bed = float(h_bed) / max(n_z_bed, 1)
            z_face = np.concatenate([[0.0], np.cumsum(dz_arr)])
            z_mid = z_face[:-1] + 0.5 * dz_arr
            Lz_actual = z_face[-1]
            # Precompute face-to-face distances (cell-center to cell-center)
            d_face_above = np.empty(Nz, dtype=np.float64)
            d_face_below = np.empty(Nz, dtype=np.float64)
            for k in range(Nz):
                d_face_above[k] = (0.5 * (dz_arr[k] + dz_arr[k+1])
                                   if k + 1 < Nz else 0.5 * dz_arr[k])
                d_face_below[k] = (0.5 * (dz_arr[k] + dz_arr[k-1])
                                   if k - 1 >= 0 else 0.5 * dz_arr[k])
            x_mid = (np.arange(Nx) + 0.5) * (Lx / Nx)
            y_mid = (np.arange(Ny) + 0.5) * dy
            return cls(
                Nx=Nx, Ny=Ny, Nz=Nz,
                dx=Lx / Nx, dy=dy, dz=float(dz_bed),
                Lx=Lx, Ly=Ly, Lz=float(Lz_actual),
                n_z_bed=n_z_bed,
                x_mid=x_mid, y_mid=y_mid, z_mid=z_mid,
                dz_arr=dz_arr, inv_dz_arr=1.0 / dz_arr,
                z_face=z_face,
                d_face_above=d_face_above, d_face_below=d_face_below,
            )
        # ── Legacy path follows below (uniform bed default, etc.) ─────────

        # Bed cells: uniform OR geometrically-stretched at the wall.
        # Phase 14v-bc: when dz_first + bl_growth are set AND n_z_bed > 0,
        # the bed is partitioned with a thin first cell (dz_first at z=0)
        # growing geometrically by bl_growth until h_bed is filled.  This
        # resolves the wall layer for proper turbulent wall-stress capture
        # (k-ε / wall function) and replaces the over-restrictive cell pin
        # u[0,:,:] = 0 that had been compensating for the under-resolved BL.
        # n_z_bed is interpreted as a SOFT TARGET cell count when BL is on
        # (actual count chosen so the geometric stack fills h_bed exactly).
        if (n_z_bed > 0 and dz_first is not None
                and dz_first > 0.0 and bl_growth > 1.0 + 1e-12):
            bed_dz_list = [float(dz_first)]
            cumulative = float(dz_first)
            for _ in range(n_z_bed - 1):
                next_dz = bed_dz_list[-1] * bl_growth
                if cumulative + next_dz >= h_bed:
                    break
                bed_dz_list.append(next_dz)
                cumulative += next_dz
            # Last cell absorbs any remainder so bed thickness = h_bed exactly.
            tail = h_bed - cumulative
            if tail > 0.0:
                bed_dz_list.append(tail)
            bed_dz = np.array(bed_dz_list, dtype=np.float64)
            if bed_refine_top:
                # Phase 14af: invert the geometric stack so thin cells are
                # at the TOP of the bed instead of the bottom.  Used to
                # isolate which end of the bed drives the resolution gain
                # (top cells = downward DOM radiation lands there; bottom
                # cells = wind shear with ground).  Same total cells, same
                # h_bed, same dz_first/dz_last values — just reversed.
                bed_dz = bed_dz[::-1].copy()
            n_z_bed = len(bed_dz)        # update to actual cell count
            dz_bed = bed_dz[-1]          # bulk bed dz (used as buffer-start size)
        else:
            dz_bed = h_bed / max(n_z_bed, 1)
            bed_dz = np.full(n_z_bed, dz_bed, dtype=np.float64)

        # Phase 14t-A: BL refinement at z=0 (cold-flow only).
        bl_dz = np.array([], dtype=np.float64)
        if n_z_bed == 0 and dz_first is not None and dz_first > 0.0:
            if bl_growth <= 1.0 + 1e-12:
                # Uniform-thickness BL of dz_first (caller can stack multiple
                # to resolve the sublayer; rare — typically growth>1 is used).
                # To avoid building enormous numbers of cells, we cap at one.
                bl_dz = np.array([dz_first], dtype=np.float64)
            else:
                # Geometric BL growth: first cell = dz_first, each ×bl_growth,
                # stop when next cell would exceed dx (then transition to
                # the existing dz_expansion stack at dx).
                bl_list = []
                dz = float(dz_first)
                while dz < dx and len(bl_list) < 200:
                    bl_list.append(dz)
                    dz *= bl_growth
                bl_dz = np.array(bl_list, dtype=np.float64)
            dz_bed_for_buffer = dx   # buffer above BL starts from dx-sized cells
        else:
            dz_bed_for_buffer = dz_bed

        # Phase 14ad: BL refinement ABOVE the bed (n_z_bed > 0 case).
        # When dz_first_above < dz_bed, insert a stack of cells starting at
        # dz_first_above and geometrically growing by bl_growth_above until
        # they reach dz_bed (or dx).  Resolves the steep T_g gradient at the
        # bed-top / plume interface where downward radiation + advection
        # heat the bed canopy.  Without this, the cell immediately above
        # the bed is dz_bed thick — too coarse to resolve the gas-phase
        # boundary layer between flame body and bed surface.
        bl_above_dz = np.array([], dtype=np.float64)
        if (n_z_bed > 0 and dz_first_above is not None
                and dz_first_above > 0.0 and dz_first_above < dz_bed_for_buffer
                and bl_growth_above > 1.0 + 1e-12):
            bl_above_list = [float(dz_first_above)]
            while bl_above_list[-1] * bl_growth_above < dz_bed_for_buffer:
                bl_above_list.append(bl_above_list[-1] * bl_growth_above)
                if len(bl_above_list) >= 200:
                    break   # safety cap
            bl_above_dz = np.array(bl_above_list, dtype=np.float64)

        # Buffer cells: geometric expansion (or uniform if dz_expansion=1)
        target_buffer = Lz - h_bed - bl_dz.sum() - bl_above_dz.sum()
        if dz_expansion <= 1.0 + 1e-12:
            # Uniform dz throughout (legacy behavior)
            n_buf = max(1, int(round(target_buffer / dz_bed_for_buffer)))
            buf_dz = np.full(n_buf, dz_bed_for_buffer, dtype=np.float64)
        else:
            # Geometric expansion: find n_buf such that
            #   sum_{j=0..n_buf-1} dz_bed · expansion^j ≥ target_buffer
            #   = dz_bed · (expansion^n_buf - 1) / (expansion - 1) ≥ target_buffer
            ratio = target_buffer * (dz_expansion - 1.0) / dz_bed_for_buffer + 1.0
            n_buf = max(1, int(np.ceil(np.log(ratio) / np.log(dz_expansion))))
            buf_dz = dz_bed_for_buffer * np.power(dz_expansion, np.arange(n_buf, dtype=np.float64))

        dz_arr = np.concatenate([bed_dz, bl_dz, bl_above_dz, buf_dz])
        Nz = len(dz_arr)
        # Cumulative z-face positions: z_face[0] = 0, z_face[k+1] = z_face[k] + dz[k]
        z_face = np.concatenate([[0.0], np.cumsum(dz_arr)])
        z_mid = z_face[:-1] + 0.5 * dz_arr
        Lz_actual = z_face[-1]

        # Precompute face-to-face distances (cell-center to cell-center)
        d_face_above = np.empty(Nz, dtype=np.float64)
        d_face_below = np.empty(Nz, dtype=np.float64)
        for k in range(Nz):
            d_face_above[k] = (0.5 * (dz_arr[k] + dz_arr[k+1])
                               if k + 1 < Nz else 0.5 * dz_arr[k])
            d_face_below[k] = (0.5 * (dz_arr[k] + dz_arr[k-1])
                               if k - 1 >= 0 else 0.5 * dz_arr[k])

        x_mid = (np.arange(Nx) + 0.5) * (Lx / Nx)
        y_mid = (np.arange(Ny) + 0.5) * dy

        return cls(
            Nx=Nx, Ny=Ny, Nz=Nz,
            dx=Lx / Nx, dy=dy, dz=float(dz_bed),
            Lx=Lx, Ly=Ly, Lz=float(Lz_actual),
            n_z_bed=n_z_bed,
            x_mid=x_mid, y_mid=y_mid, z_mid=z_mid,
            dz_arr=dz_arr, inv_dz_arr=1.0 / dz_arr,
            z_face=z_face,
            d_face_above=d_face_above, d_face_below=d_face_below,
        )


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class Spread3DResult:
    """Output of run_3d_spread."""
    ros_m_s: float
    n_cells_ignited: int
    front_t: list                  # list[float]
    front_x: list                  # list[float]
    grid: Grid3D
    state_final: CellState3D
    # Per-step diagnostics (lists, len = n_steps + 1)
    diag_t: list = field(default_factory=list)
    diag_Tg_max: list = field(default_factory=list)
    diag_Ts_max: list = field(default_factory=list)
    diag_Sp_max: list = field(default_factory=list)
    diag_Qc_max: list = field(default_factory=list)
    diag_omega_max: list = field(default_factory=list)
    diag_Y_max: list = field(default_factory=list)
    diag_n_ign: list = field(default_factory=list)
    diag_proj_div_max: list = field(default_factory=list)
    diag_proj_n_iter: list = field(default_factory=list)
    # Phase 14u: final turbulence fields for diagnostics
    k_turb_final: "np.ndarray | None" = None
    eps_turb_final: "np.ndarray | None" = None
    nu_t_final: "np.ndarray | None" = None

    # ── Save/load ────────────────────────────────────────────────────────
    # Persists the result to a single .npz file so expensive runs can be
    # cached and replayed without re-simulating.  Stores all arrays + a
    # JSON-encoded metadata dict (run inputs, grid scalars).  Restored
    # results have identical fields but may not be re-runnable through
    # `run_3d_spread` (they're snapshots, not live state).

    def save(self, path: "Union[str, Path]", metadata: Optional[dict] = None) -> None:
        """Save result to ``path`` (a single .npz file).

        Parameters
        ----------
        path
            Destination path.  Adds ``.npz`` suffix if not present.
        metadata
            Optional dict of JSON-serialisable run inputs (wind speed,
            grid params, deck info) — preserved as a string in the npz.
        """
        import json
        path = Path(path)
        if not str(path).endswith(".npz"):
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        g = self.grid; s = self.state_final
        np.savez_compressed(
            path,
            # Scalars / metadata
            _meta_json=np.array(json.dumps({
                "ros_m_s": float(self.ros_m_s),
                "n_cells_ignited": int(self.n_cells_ignited),
                "Nx": int(g.Nx), "Ny": int(g.Ny), "Nz": int(g.Nz),
                "dx": float(g.dx), "dy": float(g.dy), "dz": float(g.dz),
                "Lx": float(g.Lx), "Ly": float(g.Ly), "Lz": float(g.Lz),
                "n_z_bed": int(g.n_z_bed),
                "user": metadata or {},
            })),
            # Grid arrays
            x_mid=g.x_mid, y_mid=g.y_mid, z_mid=g.z_mid,
            dz_arr=g.dz_arr, inv_dz_arr=g.inv_dz_arr, z_face=g.z_face,
            d_face_above=g.d_face_above, d_face_below=g.d_face_below,
            # State arrays
            rho=s.rho, u=s.u, v=s.v, w=s.w, p_dyn=s.p_dyn, T_g=s.T_g,
            Y_fuel=s.Y_fuel, Y_O2=s.Y_O2, Y_H2O=s.Y_H2O, Y_CO=s.Y_CO,
            alpha_s=s.alpha_s, T_s=s.T_s,
            m_hemi=s.m_hemi, m_cell=s.m_cell, m_lign=s.m_lign,
            m_char=s.m_char,
            # Front + diagnostics
            front_t=np.array(self.front_t), front_x=np.array(self.front_x),
            diag_t=np.array(self.diag_t),
            diag_Tg_max=np.array(self.diag_Tg_max),
            diag_Ts_max=np.array(self.diag_Ts_max),
            diag_Sp_max=np.array(self.diag_Sp_max),
            diag_Qc_max=np.array(self.diag_Qc_max),
            diag_omega_max=np.array(self.diag_omega_max),
            diag_Y_max=np.array(self.diag_Y_max),
            diag_n_ign=np.array(self.diag_n_ign),
        )

    @classmethod
    def load(cls, path: "Union[str, Path]") -> "Spread3DResult":
        """Load a previously-saved result from ``path``."""
        import json
        path = Path(path)
        if not str(path).endswith(".npz"):
            path = path.with_suffix(".npz")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta_json"]))

        grid = Grid3D(
            Nx=int(meta["Nx"]), Ny=int(meta["Ny"]), Nz=int(meta["Nz"]),
            dx=float(meta["dx"]), dy=float(meta["dy"]), dz=float(meta["dz"]),
            Lx=float(meta["Lx"]), Ly=float(meta["Ly"]), Lz=float(meta["Lz"]),
            n_z_bed=int(meta["n_z_bed"]),
            x_mid=z["x_mid"], y_mid=z["y_mid"], z_mid=z["z_mid"],
            dz_arr=z["dz_arr"], inv_dz_arr=z["inv_dz_arr"],
            z_face=z["z_face"],
            d_face_above=z["d_face_above"], d_face_below=z["d_face_below"],
        )
        state = CellState3D(
            rho=z["rho"], u=z["u"], v=z["v"], w=z["w"],
            p_dyn=z["p_dyn"], T_g=z["T_g"],
            Y_fuel=z["Y_fuel"], Y_O2=z["Y_O2"],
            Y_H2O=(z["Y_H2O"] if "Y_H2O" in z.files
                   else np.zeros_like(z["Y_O2"])),
            Y_CO=(z["Y_CO"] if "Y_CO" in z.files
                  else np.zeros_like(z["Y_O2"])),
            alpha_s=z["alpha_s"], T_s=z["T_s"],
            m_hemi=z["m_hemi"], m_cell=z["m_cell"], m_lign=z["m_lign"],
            m_char=z["m_char"] if "m_char" in z.files else np.zeros_like(z["m_hemi"]),
        )
        return cls(
            ros_m_s=float(meta["ros_m_s"]),
            n_cells_ignited=int(meta["n_cells_ignited"]),
            front_t=z["front_t"].tolist(),
            front_x=z["front_x"].tolist(),
            grid=grid,
            state_final=state,
            diag_t=z["diag_t"].tolist(),
            diag_Tg_max=z["diag_Tg_max"].tolist(),
            diag_Ts_max=z["diag_Ts_max"].tolist(),
            diag_Sp_max=z["diag_Sp_max"].tolist(),
            diag_Qc_max=z["diag_Qc_max"].tolist(),
            diag_omega_max=z["diag_omega_max"].tolist(),
            diag_Y_max=z["diag_Y_max"].tolist(),
            diag_n_ign=z["diag_n_ign"].tolist(),
        )


# ── Public entry point ────────────────────────────────────────────────────────
def run_3d_spread(
    ri_or_path: Union[Path, str, RomInputs],
    *,
    wind_speed_m_s: float = 0.0,
    Lx: float | None = None,
    Ly: float | None = None,
    Lz: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
    n_z_bed: int | None = None,
    dz_expansion: float | None = None,
    dz_first: float | None = None,
    bl_growth: float | None = None,
    dz_first_above: float | None = None,
    bl_growth_above: float = 1.3,
    bed_refine_top: bool = False,
    # Phase 14ag: new explicit BL kwargs.  Defaults None so _pick falls
    # through to outdoor_overrides (deck-driven mesh, per 2026-06-01 rule).
    # When BOTH the kwarg and the deck are unset, the legacy mesh path runs
    # (bit-exact backward compatible).
    wall_bl_N: int | None = None,
    wall_bl_first_dz: float | None = None,
    wall_bl_growth: float | None = None,
    bed_top_inner_bl_N: int | None = None,
    bed_top_inner_bl_first_dz: float | None = None,
    bed_top_inner_bl_growth: float | None = None,
    bed_top_outer_bl_N: int | None = None,
    bed_top_outer_bl_first_dz: float | None = None,
    bed_top_outer_bl_growth: float | None = None,
    atm_max_dz: float | None = None,
    atm_growth: float | None = None,
    atm_uniform_dz: float | None = None,
    max_wall_time_s: float = 60.0,
    y_bc: str = "periodic",
    turbulence_model: str = "laminar",
    radiation_solver: str = "dom",
    cfl_factor: float | None = None,
    # Adaptive-dt floor.  Adaptive dt recomputes each step from current
    # u_max / ν_t_max.  If pathological transients shrink dt below this
    # value, halt with a diagnostic — prevents wasted compute on a
    # numerically sick simulation.  Default 1×10⁻⁴ s (100 µs) is well
    # above the physically expected worst-case for our grid scales;
    # hitting the floor strongly indicates instability rather than
    # legitimate physics.  Set lower (e.g. 1e-5) only for very-fine grids.
    min_dt_s: float = 1.0e-4,
    # ── DIAGNOSTIC ignition mode: hold a hot-gas column above source ──
    # Default OFF.  When enabled, OVERRIDES T_g in a 30-cm column above
    # the source patch to the kerosene adiabatic-flame temperature for
    # the ignition window.  This is a SIMPLER and more DIRECT model of
    # the drip-torch flame than volumetric Q_solid_ext deposition (which
    # gets partially lost via particle Stefan-Boltzmann once T_s rises).
    # Note: the legacy "pilot ignition" pin was removed in Phase 14v as
    # unphysical infinite-heat reservoir — this opt-in flag re-enables
    # the mechanism specifically as a diagnostic for Phase 16 ignition
    # budget testing.
    ignition_T_pin_enable: bool = False,
    ignition_T_pin_K: float = 1500.0,    # K (kerosene flame with ~40% losses
                                          # giving practical flame temp;
                                          # stoichiometric AFT is ~2370 K)
    ignition_T_pin_height_m: float = 0.30, # column height above bed top
    ignition_T_pin_ramp_s: float = 0.5,   # ramp time from T_amb to T_pin
                                          # (smooth start avoids density shock)
    # Phase 18 scout: when True, place the gas pin column INSIDE the bed
    # (z ∈ [0, h_bed]) instead of above it (z ∈ [h_bed, h_bed + height_m]).
    # Rationale: at low U the gas-pin above the bed has its energy carried
    # straight up by buoyancy (out the top), leaving the bed-bottom lateral
    # transport pathway under-fed.  Putting the pin INSIDE the bed makes
    # the bed itself act as the thermal mass and heats the favorable
    # lateral-propagation z-band.
    ignition_T_pin_in_bed: bool = False,
    wall_function: bool | None = None,
    bed_x_start: float | None = None,
    bed_x_end: float | None = None,
    combustion_closure: str = "edc",
    # Phase 23: chemistry-family preset selects the 5-scalar constants
    # (S_STOICH, HOC_J, A_COMB, E_COMB, C_EBU) that the closure kernels
    # consume.  Default "biomass" preserves Rule #17 bit-exact behaviour
    # for all pre-Phase-23 validation cases (Cheney, Marsden-Smedley).
    # Registered families: see model_outdoor.physics_3d.chemistry_closures
    # ._constants.CHEMISTRY_FAMILIES.  Deck override:
    # ``outdoor.chemistry_family = methane``.
    chemistry_family: str = "biomass",
    # Phase 23 Refactor 2A: BC-registry hook.  Default "outdoor_wind"
    # is the wrap of the pre-Phase-23 wind-inlet pattern (bit-exact-
    # invariant for Cheney/Marsden-Smedley).  Registered kinds are in
    # model_outdoor.boundary_conditions.available().  Deck override:
    # ``outdoor.boundary_condition_kind = cup_burner``.
    boundary_condition_kind: str = "outdoor_wind",
    wind_profile_type: str = "log_law",
    alpha_s_profile_type: str = "uniform",
    alpha_s_decay_k: float = 1.5,
    snapshot_dir: "Path | str | None" = None,
    snapshot_interval_s: float = 1.0,
    # Phase 14ah: projection solver method
    projection_method: str = "pardiso",
    projection_cg_rtol: float = 1.0e-6,
    projection_amg_rebuild_every: int = 100,
    # Phase 14ah-3: DOM sub-cycling (radiation lags hydro)
    dom_subcycle_every: int = 1,
    # Phase 14aj: diagnostic — allow chi_rad override (default = None → 0.34)
    chi_rad_override: float | None = None,
    # Phase 15D sensitivity-sweep override for FSD's laminar flame speed.
    # When set, plumbs through to chemistry_closures.run as s_L kwarg.
    # None (default) leaves closure-internal default of 0.4 m/s untouched.
    s_L_fsd_override: float | None = None,
    # Phase 15D-SS — steady-state-driven termination.  When True, the
    # main loop also breaks early once the rolling slope d(front_x)/dt
    # converges within ``steady_state_tolerance`` across two consecutive
    # checks (after a ``steady_state_warmup_s`` ignition window).
    # ``max_wall_time_s`` remains the hard upper bound.  Recommended:
    # bump ``max_wall_time_s`` to 60-90s when this is on so steady-state
    # has time to develop.
    steady_state_detect: bool = False,
    steady_state_warmup_s: float = 10.0,
    steady_state_window_s: float = 5.0,
    steady_state_check_interval_s: float = 2.0,
    steady_state_tolerance: float = 0.05,
    # Phase 15E length-scale cap on ε.  When > 0, clamps
    # ε ≤ k^1.5 / L_min so the implied turbulence length scale cannot
    # fall below L_min.  Recommended L_min = h_bed.  0.0 disables.
    eps_realiz_L_min_m: float = 0.0,
    # Phase 15E-B Durbin 1996 T-bound on ε using local strain rate.
    # When > 0, clamps ε ≤ k · sqrt(3·|S|²) / α where |S|² = 2 S_ij S_ij.
    # Canonical α ≈ 0.6 (Pope 2000 §11.4).  0.0 disables.
    eps_realiz_durbin_alpha: float = 0.0,
    # Phase 18 scout: post-multiply nu_t output by a scalar. 1.0 = unchanged
    # (production default). Used to probe whether broader turbulent diffusion
    # of heat helps bridge low-U propagation. Applied to k-ε AND smagorinsky.
    # NOT a production knob — scout-only for closure-sensitivity analysis.
    nu_t_multiplier: float = 1.0,
    # Phase 15G — Damköhler 1 turbulent flame-speed for FSD branch.
    # When True: s_T = s_L·(1+u'/s_L) in the level_set_fsd closure.
    # u' = √(2k/3) from k-ε; capped at s_T_cap_factor × s_L.
    # Standard turbulent-diffusion-flame correction (Williams 1985).
    turbulent_s_T_fsd: bool = False,
    s_T_cap_factor: float = 5.0,
    # Phase 15G under-prediction diagnostic — multiplier on gas-solid
    # convective heat-transfer coefficient h_p (Ranz-Marshall).
    # 1.0 (default) = no change.  Sensitivity-testing only; Rule #1
    # forbids landing a calibrated multiplier without literature support.
    h_conv_mult: float = 1.0,
    # Phase 15H — Charlette 2002 sub-grid wrinkling factor Ξ applied to
    # the FSD branch chemistry rate:  ω_TFM = Ξ · ρ · s_L · |∇c| · f_av.
    # 1.0 (default) = unmodified back-compat.
    tfm_xi: float = 1.0,
    # Phase 15I — diagnostic override for MD2004 thermal R_p Arrhenius
    # constants.  Both must be set together (kinetic-compensation: never
    # change A alone without E, per Antal-Varhegyi 1998).  Defaults None →
    # use MD2004 lumped triplet.
    pyrolysis_A_p_override: float | None = None,
    pyrolysis_E_p_override: float | None = None,
    # Phase 15J — when True, route inner-body cells through EDC instead of
    # FSD in the level_set_fsd hybrid closure.  Effective chemistry then
    # matches Linn 2002 FIRETEC / Mell 2007 WFDS mixing-limited fast-
    # chemistry practice; level-set v_n still tracks the front kinematics.
    # Default False = current Phase 15D hybrid (FSD inside body).
    inner_body_edc: bool = False,
    # Phase 15K — diagnostic override for DOM hot-gas soot extinction
    # ceiling κ_gas_max [1/m].  Default None → use library default
    # (Tien 1998 SFPE Handbook 0.5 1/m for cellulose smoke).
    # Mell 2007 WFDS 1.0; WSGGM full-spectrum 0.5–3 at flame T.
    # Diagnostic only per Rule #2 until literature support is added in
    # the calling deck/worker.
    dom_kappa_gas_max_override: float | None = None,
    # Phase 15L — diagnostic ignition-kick multipliers.  Probe whether the
    # 5× ROS shortfall is an attractor problem: does a stronger initial
    # kick lift the system to a higher self-sustaining propagation rate,
    # or does it always collapse back to the 8.58 m/min attractor?
    #   ignition_q_mult     — multiplier on the 240 kW/m² pulse intensity
    #   ignition_width_mult — multiplier on the 0.5 m source x-width
    # Defaults 1.0 = unmodified (Quintiere-style piloted ignition).
    # Diagnostic only per Rule #2; production decks must keep these at 1.0.
    ignition_q_mult: float = 1.0,
    ignition_width_mult: float = 1.0,
    # Phase 15N — Finney 2015 parameterized burst-convective preheat
    # closure.  When True, adds a sub-grid forward convective heat-flux
    # term to the v_n driver's q_in_at_front, parameterized as
    #   q_burst_conv(d) = q_0 · exp(-d / L_burst) · gate(I_fire)
    # on ahead-band cells.  Default False = unmodified back-compat.
    # finney_q_0 / finney_L_burst, when set, override the committed
    # Phase 15N lit-anchored values (Finney 2015 PNAS, conservative end).
    # Per Rule #2 these must not be fished to match Cheney.
    finney_burst_enable: bool = False,
    finney_q_0: float | None = None,
    finney_L_burst: float | None = None,
    finney_I_thresh: float | None = None,
    # Phase 15O — Eulerian leading-edge spawn-and-deposit closure.
    # When True, fires the conservation-preserving sub-grid burst-convective
    # mechanism on flame leading-edge surface cells.  Committed lit-anchored
    # values:  Sr=0.20, duty=0.40, f_mass=0.05, fr_min=0.5.
    finney_tendril_enable: bool = False,
    finney_tendril_sr: float | None = None,
    finney_tendril_duty: float | None = None,
    finney_tendril_f_mass: float | None = None,
    finney_tendril_fr_min: float | None = None,
    # Phase 15O.1 — time-spread Eulerian release.  When > 0, each spawn's
    # mass/heat/species/momentum extraction is spread linearly over
    # t_contact seconds via persistent per-cell inventory fields instead
    # of being applied instantaneously.  Conservation is exact even
    # under overlapping spawns at the same source/target cell.
    # Default 0.0 (instantaneous, identical to Phase 15O).
    # Recommended Phase 15O.1 value: ~ 0.3 s (matches Finney 2015
    # measured T_contact range 100-500 ms; aligns with duty_cycle ×
    # T_period for typical mickey conditions).
    finney_tendril_t_contact_s: float = 0.0,
    # Phase 15O.2 — spatial aggregation box-radii.  When > 0, each spawn
    # at a leading-edge cell aggregates source mass/heat/etc. from a 3D
    # box of flame-body cells around the LE anchor: (k ± dk, j ± dj,
    # i - di_radius .. i).  Each box cell contributes f_mass × its mass.
    # Total per spawn is 27× larger than single-cell at (1,1,1) box,
    # representing a more physical "tongue-scale" extraction.
    # Defaults 0 = single-cell (Phase 15O.1 back-compat).
    finney_tendril_box_dk_up: int = 0,    # cells above LE (+z)
    finney_tendril_box_dk_down: int = 0,  # cells below LE (−z)
    finney_tendril_box_dj: int = 0,        # cells cross-stream (±y)
    finney_tendril_box_di_back: int = 0,   # cells behind LE (−x, into body)
    # Phase 15P — Lagrangian Finney burst-convective preheat (tracked
    # particles with buoyancy + drag).  Exclusive with the Eulerian
    # tendril path above; if both are enabled, the Eulerian path runs.
    finney_lagrangian_enable: bool = False,
    finney_lagrangian_N_max: int = 8192,
    finney_lagrangian_t_contact_s: float = 0.3,
    finney_lagrangian_d_p_m: float = 0.075,
    finney_lagrangian_C_D: float = 1.0,
    finney_lagrangian_sr: float | None = None,
    finney_lagrangian_duty: float | None = None,
    finney_lagrangian_f_mass: float | None = None,
    finney_lagrangian_fr_min: float | None = None,
    # Phase 16 — Lagrangian sub-grid bed particles.  When enabled, the
    # Eulerian bed pyrolysis/drying/char_ox/smolder kernels are SKIPPED
    # and the bed is represented as a population of discrete particles
    # (one per α_s > 0 cell × N_per_cell).  Mass migrates from m_hemi /
    # m_water to particle inventory at init; m_hemi / m_water are zeroed.
    # Gas-solid coupling kernel is also skipped — particles handle their
    # own T_s update and emit Q_g_conv to extract heat from gas directly.
    lagrangian_bed_enable: bool = False,
    lagrangian_bed_N_per_cell: int = 50,
    lagrangian_bed_sav_1_m: float | None = None,    # None → use deck SAV
    lagrangian_bed_h_conv: float = 25.0,            # W/m²/K, Mell 2007
    lagrangian_bed_rho_solid_true: float = 380.0,   # kg/m³ dry grass (Susott 1982)
    lagrangian_bed_cp_solid: float = 1500.0,        # J/kg/K (Mell 2007)
    lagrangian_bed_eps_solid: float = 0.9,          # particle radiation emissivity
                                                     # (grass/char ε ~ 0.85-0.95)
    lagrangian_bed_view_factor: float = 1.0,        # [0..1] scalar multiplier on
                                                     # particle radiation loss.
                                                     # When view_factor_geometric
                                                     # is True (default), this is
                                                     # an additional tuning knob
                                                     # on top of per-particle
                                                     # Beer-Lambert; default 1.0
                                                     # = pure geometric.
    lagrangian_bed_view_factor_geometric: bool = True, # per-particle f_geom =
                                                     # exp(-κ_bed·(h_bed−z_p)):
                                                     # surface particles emit
                                                     # fully, deep particles
                                                     # ≈ 0 (reabsorbed by neighbors).
    lagrangian_bed_do_drying: bool = True,
    lagrangian_bed_do_pyrolysis: bool = True,
    lagrangian_bed_do_char_ox: bool = True,
    lagrangian_bed_do_smolder: bool = True,
    # Drying-kinetics mode for the Lagrangian bed kernel.
    #   "arrhenius"   = Lautenberger 2009 bound-water Arrhenius only
    #                   (default; backward-compat).  Under-predicts grass
    #                   moisture sensitivity.
    #   "equilibrium" = FIRETEC heat-rate-limited only (Linn 2002).
    #                   Pins T_s at 373 K while water remains.
    #   "combined"    = grass-tuned Arrhenius (A_DRY_GRASS, E_DRY_GRASS
    #                   per Sano & Hasegawa 1995 capillary-water) below
    #                   T_BOIL + FIRETEC equilibrium above T_BOIL.
    #                   Combines preheat-zone water removal with mass-
    #                   dependent boil-phase delay; best estimate of
    #                   Cheney moisture sensitivity from first principles.
    lagrangian_bed_drying_mode: str = "arrhenius",
    # Phase 16 EDC extinction-threshold physics (mechanisms A+B+C):
    #   A — marginal heat-release-rate quench (Linn 2002 / Drysdale §3.4)
    #   B — inert-fraction combustion suppression (Beyler 1992)
    #   C — cold-flame floor (Westbrook & Dryer 1981)
    # Defaults OFF — backward-compat with 0D startup tests.  Required ON
    # for natural-fire moisture sensitivity AND Rule #0 suppression
    # validation (water mist, CO2 extinguisher, foam).
    edc_extinction_enable: bool = False,
    # Phase 17c test (2026-06-23): zero the level-set v_n forcing and let
    # the bed self-ignite via gas-side CFD (advection + DOM + h_conv) +
    # bed coupling alone.  Tests whether the kinematic v_n = q_in / E_ign
    # surrogate (Mell 2007 §3.4) was masking working CFD physics.  Level-
    # set still tracks regions via the source-patch initialization but
    # does not propagate kinematically.
    level_set_passive: bool = False,
    # Phase 17d (2026-06-25): solid-phase ignition.  Pre-heat the source-
    # patch bed particles to T_s_seed at t=0 and skip the artificial gas
    # T_g pin.  Avoids the 5s × 1500K gas heating that creates a coherent
    # buoyant plume strong enough to reverse the bed wind at low U.
    solid_phase_ignition_enable: bool = False,
    solid_phase_ignition_T_s_K: float = 800.0,
    # Phase 19 (2026-06-29): empirical-ROS hybrid for low-U regimes.
    # When U < empirical_ros_u_threshold_m_s, the level-set front advances
    # at an EMPIRICAL rate (Cheney 1998 Eq. 6 by default) instead of the
    # resolved-physics v_n.  Linear blend in [threshold - blend_width,
    # threshold].  This matches the WRF-Fire / CAWFE / PHOENIX pattern of
    # using Rothermel-style ROS at the front in regimes where resolved
    # closures cannot capture the intermittent-tongue propagation
    # mechanism (Finney 2015 PNAS).  Default disabled.  See
    # `model_outdoor/empirical_ros.py` and Phase 18 bug-sweep memo.
    empirical_ros_enable: bool = False,
    empirical_ros_model: str = "cheney_eq6",
    empirical_ros_a_ch: float = 0.406,         # Cheney Nat grass default
                                                # (Cut grass = 0.343)
    empirical_ros_u_threshold_m_s: float = 1.4, # Cheney 1998 quasi-steady
                                                # cutoff (5 km/h)
    empirical_ros_blend_width_m_s: float = 0.5, # smooth blend window
    # Phase 22: buttongrass (Marsden-Smedley 1995) age kwarg.  Ignored by
    # cheney_eq6.  Used by empirical_ros_model="marsden_smedley".
    empirical_ros_age_yr: float = 10.0,
    # Phase 20 char-ox knobs (Phase 19 sweep GIF diagnostic — persistent
    # hot char block).  Both default to preserving pre-Phase-20 behavior.
    #   char_ox_flux_cap_W_m2 (B):  surface-area-scaled cap on Q_char per
    #     particle.  Default 1.0e5 (Williams 1985 low end).  Bump to
    #     2.5e5 for Williams clean-char peak.
    #   char_ox_ash_exp (C):  ash-coverage penalty exponent on reactive
    #     surface area.  0.0 = disabled.  Typical 0.5-1.0 (random-pore
    #     model, Bhatia & Perlmutter 1980).  Uses ratcheted m_char_max
    #     per particle as the reference for burn fraction.
    char_ox_flux_cap_W_m2: float = 1.0e5,
    char_ox_ash_exp: float = 0.0,
    # Phase 24 — sprinkler-activation moisture-jump BC.  At t = t_s, adds
    # rho_b * delta_frac to m_water in a (x, z) zone of the bed.  Works
    # for both Eulerian and Lagrangian bed.  One-shot per run.  Off by
    # default; when off the code path is a no-op and preserves bit-exact
    # results of every prior validated case (Rule #17).
    # See docs/plans/phase24_sprinkler_moisture_jump.md.
    moisture_jump_enable: bool = False,
    moisture_jump_t_s: float = 0.0,
    moisture_jump_delta_frac: float = 0.0,
    moisture_jump_x_lo_m: float | None = None,   # default 0.0
    moisture_jump_x_hi_m: float | None = None,   # default Lx
    moisture_jump_z_lo_m: float | None = None,   # default 0.0
    moisture_jump_z_hi_m: float | None = None,   # default h_bed
) -> Spread3DResult:
    """Run the bottom-up 3D PDE spread model.

    Mesh, bed extent, and numerical params can be specified in the deck
    via ``outdoor.Lx``, ``outdoor.dx``, ``outdoor.bed_x_start`` etc.
    Kwargs override deck values when explicit; otherwise deck values are
    used; otherwise hardcoded fallback defaults apply.

    Parameters
    ----------
    ri_or_path
        Path to deck file, deck string, or pre-parsed ``RomInputs``.
    wind_speed_m_s
        10-m reference wind speed (m/s).  Inlet profile is applied at x=0.
    Lx, Ly, Lz
        Domain extents [m].  ``Lx`` is along wind, ``Ly`` cross-wind,
        ``Lz`` vertical (must exceed ``h_bed``).
    dx
        Horizontal cell size [m].  ``dy`` is rounded to be near ``dx``.
    n_z_bed
        Number of vertical cells inside the fuel bed.
    bed_x_start, bed_x_end
        Bed leading and trailing edges [m].  Default: 0.0 and Lx (full
        bed).  When ``bed_x_start > 0``, the upstream region [0, bed_x_start]
        is bare ground (no fuel, no drag), letting the wind develop a
        natural profile before encountering the bed leading edge — matches
        Cheney 1993 experimental setup (open ground upstream of fuel patch).
    max_wall_time_s
        Simulated time horizon [s].
    y_bc
        ``'periodic'`` (default; infinite-fire-line) or ``'edge_loss'``.
    turbulence_model
        ``'laminar'`` (default), ``'k_epsilon'``, or ``'smagorinsky'``.
    wall_function
        Apply log-law wall function at z=0 (Phase 14t-B).
    combustion_closure
        Selects the gas-phase combustion closure.

        ``'ebu_bootstrap'`` (default) — Magnussen-Hjertager EBU + Arrhenius
            with 4-rate min closure, plus the Phase 14x bootstrap heat
            applied in flame_body cells for ``cell_age < t_bootstrap``.
            This is the legacy Phase 14x closure.

        ``'edc'`` — Magnussen 1981 Eddy-Dissipation Concept.  Replaces
            the EBU+Arrhenius closure with a fine-structure rate
            ω = γ*·ρ·min(Y_F, Y_O2/s)/τ* where γ* and τ* are derived
            from local k, ε.  No bootstrap.

        ``'pasr'`` — Chomiak 1990 Partially Stirred Reactor.  Cell rate
            is γ_pasr·ω_EBU where γ_pasr = τ_chem/(τ_chem + τ_mix).  No
            bootstrap.
    """
    # Phase 15-0: closure name is validated by the chemistry_closures
    # registry — see model_outdoor.physics_3d.chemistry_closures.
    if combustion_closure not in chemistry_closures.available():
        raise ValueError(
            f"combustion_closure={combustion_closure!r} not in "
            f"{chemistry_closures.available()!r}"
        )

    # ── Parse deck and pull outdoor params ───────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri = load_text_input(Path(ri_or_path))
    else:
        ri = ri_or_path
    outdoor_cfg = outdoor_env_from_dict(ri.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    # ── Mesh + bed extent resolution: kwarg → deck → fallback default ────
    # Deck values come from outdoor_overrides dict (parser is generic for
    # outdoor.* keys, see model/io/text_input.py).
    _ovr = ri.outdoor_overrides
    def _pick(kw_val, deck_key, default):
        if kw_val is not None:
            return kw_val
        # Parser stores keys lowercase; tolerate either case from callers.
        for key in (deck_key, deck_key.lower()):
            if key in _ovr and _ovr[key] is not None:
                return _ovr[key]
        return default
    Lx           = float(_pick(Lx,           "Lx",           10.0))
    Ly           = float(_pick(Ly,           "Ly",            2.0))
    Lz           = float(_pick(Lz,           "Lz",            1.5))
    dx           = float(_pick(dx,           "dx",            0.05))
    n_z_bed      = int  (_pick(n_z_bed,      "n_z_bed",       4))
    dz_expansion = float(_pick(dz_expansion, "dz_expansion",  1.0))
    bl_growth    = float(_pick(bl_growth,    "bl_growth",     1.0))
    cfl_factor   = float(_pick(cfl_factor,   "cfl_factor",    0.4))
    wall_function = bool(_pick(wall_function, "wall_function", False))
    # Projection iteration controls.  Threshold-based termination: loop
    # runs until divmax < proj_div_tol or n_iter == proj_max_iter (safety
    # cap).  proj_max_iter should be set high enough that the threshold
    # is the binding constraint under normal conditions.
    proj_max_iter = int  (_pick(None, "proj_max_iter", 50))
    proj_div_tol  = float(_pick(None, "proj_div_tol",  1.0e-3))
    # dz_first stays None when unset (BL refinement off)
    if dz_first is None:
        dz_first = _ovr.get("dz_first", None)
    if dz_first is not None:
        dz_first = float(dz_first)
    # Phase 14ag segmented mesh kernel — deck-drivable.
    # When any wall_bl_N / bed_top_*_N is > 0, Grid3D.build delegates to
    # build_z_axis_bed_atm which composes: wall BL + uniform bed +
    # optional inner-solid BL + optional outer-air BL + atm with growth.
    # Defaults preserve legacy bit-exact behavior (all params 0/None).
    wall_bl_N                  = int  (_pick(wall_bl_N,                  "wall_bl_N",                  0))
    wall_bl_first_dz           = float(_pick(wall_bl_first_dz,           "wall_bl_first_dz",           0.0))
    wall_bl_growth             = float(_pick(wall_bl_growth,             "wall_bl_growth",             1.3))
    bed_top_inner_bl_N         = int  (_pick(bed_top_inner_bl_N,         "bed_top_inner_bl_N",         0))
    bed_top_inner_bl_first_dz  = float(_pick(bed_top_inner_bl_first_dz,  "bed_top_inner_bl_first_dz",  0.0))
    bed_top_inner_bl_growth    = float(_pick(bed_top_inner_bl_growth,    "bed_top_inner_bl_growth",    1.3))
    bed_top_outer_bl_N         = int  (_pick(bed_top_outer_bl_N,         "bed_top_outer_bl_N",         0))
    bed_top_outer_bl_first_dz  = float(_pick(bed_top_outer_bl_first_dz,  "bed_top_outer_bl_first_dz",  0.0))
    bed_top_outer_bl_growth    = float(_pick(bed_top_outer_bl_growth,    "bed_top_outer_bl_growth",    1.3))
    atm_growth                 = float(_pick(atm_growth,                 "atm_growth",                 1.3))
    if atm_max_dz is None:
        atm_max_dz = _ovr.get("atm_max_dz", None)
    if atm_max_dz is not None:
        atm_max_dz = float(atm_max_dz)
    if atm_uniform_dz is None:
        atm_uniform_dz = _ovr.get("atm_uniform_dz", None)
    if atm_uniform_dz is not None:
        atm_uniform_dz = float(atm_uniform_dz)
    # Bed x-extent: default 0.0 → Lx (full bed for backward compat)
    bed_x_start = float(_pick(bed_x_start, "bed_x_start", 0.0))
    bed_x_end   = float(_pick(bed_x_end,   "bed_x_end",   Lx))

    # Phase 17e (2026-06-25): make case-spec params deck-drivable so that
    # workers stop hardcoding case-specific physics choices in Python kwargs.
    # The case spec (mesh, turbulence model, ignition setup, closures) is a
    # PROPERTY of the validation case and belongs in the deck file, not in
    # the worker script.  Deck values WIN over kwarg defaults — _pick can't
    # be used because boolean kwargs default to False (not None), and we
    # want deck values to override that.
    def _deck_first(deck_key, kw_val):
        """Prefer deck value over kwarg (deck-spec is canonical)."""
        for key in (deck_key, deck_key.lower()):
            if key in _ovr and _ovr[key] is not None:
                return _ovr[key]
        return kw_val
    dy            = float(_deck_first("dy",                       dy if dy is not None else dx))
    n_z_bed       = int  (_deck_first("n_z_bed",                  n_z_bed))
    turbulence_model      = str  (_deck_first("turbulence_model",         turbulence_model))
    combustion_closure    = str  (_deck_first("combustion_closure",       combustion_closure))
    chemistry_family      = str  (_deck_first("chemistry_family",         chemistry_family))
    boundary_condition_kind = str(_deck_first("boundary_condition_kind",  boundary_condition_kind))
    if boundary_condition_kind not in _bc_available():
        raise ValueError(
            f"boundary_condition_kind={boundary_condition_kind!r} not in "
            f"{_bc_available()!r}"
        )
    # Phase 23: resolve chemistry-family preset to the 5-scalar kwarg
    # dict.  Injected into every chemistry_closures.run() call below.
    # For family="biomass" this returns exactly the module-constant
    # values, so pre-Phase-23 behaviour is bit-exact preserved (Rule #17).
    _chem_family_kwargs = _resolve_chemistry_family(chemistry_family)
    projection_method     = str  (_deck_first("projection_method",        projection_method))
    radiation_solver      = str  (_deck_first("radiation_solver",         radiation_solver))
    dom_subcycle_every    = int  (_deck_first("dom_subcycle_every",       dom_subcycle_every))
    # Lagrangian bed
    lagrangian_bed_enable = bool (_deck_first("lagrangian_bed_enable",    lagrangian_bed_enable))
    lagrangian_bed_N_per_cell = int(_deck_first("lagrangian_bed_N_per_cell", lagrangian_bed_N_per_cell))
    lagrangian_bed_drying_mode = str(_deck_first("lagrangian_bed_drying_mode", lagrangian_bed_drying_mode))
    lagrangian_bed_h_conv  = float(_deck_first("lagrangian_bed_h_conv",   lagrangian_bed_h_conv))
    # Ignition setup
    ignition_T_pin_enable   = bool (_deck_first("ignition_T_pin_enable",  ignition_T_pin_enable))
    ignition_T_pin_K        = float(_deck_first("ignition_T_pin_K",       ignition_T_pin_K))
    ignition_T_pin_height_m = float(_deck_first("ignition_T_pin_height_m", ignition_T_pin_height_m))
    ignition_T_pin_ramp_s   = float(_deck_first("ignition_T_pin_ramp_s",  ignition_T_pin_ramp_s))
    ignition_T_pin_in_bed   = bool (_deck_first("ignition_T_pin_in_bed",  ignition_T_pin_in_bed))
    ignition_q_mult         = float(_deck_first("ignition_q_mult",        ignition_q_mult))
    ignition_width_mult     = float(_deck_first("ignition_width_mult",    ignition_width_mult))
    solid_phase_ignition_enable = bool (_deck_first("solid_phase_ignition_enable", solid_phase_ignition_enable))
    solid_phase_ignition_T_s_K  = float(_deck_first("solid_phase_ignition_T_s_K",  solid_phase_ignition_T_s_K))
    # Phase 17c level-set forcing mode
    level_set_passive       = bool (_deck_first("level_set_passive",       level_set_passive))
    # Phase 16 EDC extinction-threshold (A+B+C)
    edc_extinction_enable   = bool (_deck_first("edc_extinction_enable",   edc_extinction_enable))
    # Finney burst-convective preheat
    finney_tendril_enable   = bool (_deck_first("finney_tendril_enable",   finney_tendril_enable))
    nu_t_multiplier         = float(_deck_first("nu_t_multiplier",         nu_t_multiplier))
    empirical_ros_enable    = bool (_deck_first("empirical_ros_enable",    empirical_ros_enable))
    empirical_ros_model     = str  (_deck_first("empirical_ros_model",     empirical_ros_model))
    empirical_ros_a_ch      = float(_deck_first("empirical_ros_a_ch",      empirical_ros_a_ch))
    empirical_ros_u_threshold_m_s = float(_deck_first("empirical_ros_u_threshold_m_s", empirical_ros_u_threshold_m_s))
    empirical_ros_blend_width_m_s = float(_deck_first("empirical_ros_blend_width_m_s", empirical_ros_blend_width_m_s))
    empirical_ros_age_yr    = float(_deck_first("empirical_ros_age_yr",    empirical_ros_age_yr))
    char_ox_flux_cap_W_m2   = float(_deck_first("char_ox_flux_cap_W_m2",   char_ox_flux_cap_W_m2))
    char_ox_ash_exp         = float(_deck_first("char_ox_ash_exp",         char_ox_ash_exp))
    # Phase 24 moisture-jump BC — six flags.  None defaults resolved after
    # h_bed / Lx are known (see block near main-loop entry).
    moisture_jump_enable       = bool(_deck_first("moisture_jump_enable",       moisture_jump_enable))
    moisture_jump_t_s          = float(_deck_first("moisture_jump_t_s",          moisture_jump_t_s))
    moisture_jump_delta_frac   = float(_deck_first("moisture_jump_delta_frac",   moisture_jump_delta_frac))
    moisture_jump_x_lo_m       = _deck_first("moisture_jump_x_lo_m",             moisture_jump_x_lo_m)
    moisture_jump_x_hi_m       = _deck_first("moisture_jump_x_hi_m",             moisture_jump_x_hi_m)
    moisture_jump_z_lo_m       = _deck_first("moisture_jump_z_lo_m",             moisture_jump_z_lo_m)
    moisture_jump_z_hi_m       = _deck_first("moisture_jump_z_hi_m",             moisture_jump_z_hi_m)
    # Run controls
    min_dt_s          = float(_deck_first("min_dt_s",                 min_dt_s))
    max_wall_time_s   = float(_deck_first("max_wall_time_s",          max_wall_time_s))

    h_bed   = outdoor_cfg.fuel_depth_m
    rho_b   = outdoor_cfg.bulk_density_kg_m3
    T_amb   = outdoor_cfg.ambient_T_K if hasattr(outdoor_cfg, "ambient_T_K") else _T_AMB_DEFAULT

    # ── Build grid + allocate state ──────────────────────────────────────
    grid = Grid3D.build(Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dy,
                        h_bed=h_bed, n_z_bed=n_z_bed,
                        dz_expansion=dz_expansion,
                        dz_first=dz_first, bl_growth=bl_growth,
                        dz_first_above=dz_first_above,
                        bl_growth_above=bl_growth_above,
                        bed_refine_top=bed_refine_top,
                        wall_bl_N=wall_bl_N,
                        wall_bl_first_dz=wall_bl_first_dz,
                        wall_bl_growth=wall_bl_growth,
                        bed_top_inner_bl_N=bed_top_inner_bl_N,
                        bed_top_inner_bl_first_dz=bed_top_inner_bl_first_dz,
                        bed_top_inner_bl_growth=bed_top_inner_bl_growth,
                        bed_top_outer_bl_N=bed_top_outer_bl_N,
                        bed_top_outer_bl_first_dz=bed_top_outer_bl_first_dz,
                        bed_top_outer_bl_growth=bed_top_outer_bl_growth,
                        atm_max_dz=atm_max_dz,
                        atm_growth=atm_growth,
                        atm_uniform_dz=atm_uniform_dz)
    state = CellState3D.allocate(Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx, T_amb=T_amb)

    # Initialize bed cells: alpha_s + bulk fuel mass.
    # Per Phase 3 (Morvan & Dupuy 2004 single-pool grass kinetics), the
    # entire bulk fuel mass goes into m_hemi (which step_pyrolysis_md2004
    # treats as m_solid).  m_cell and m_lign are unused at field scale —
    # cured grass already has hemicellulose pre-degraded and char fraction
    # is captured via residual mass at end of pyrolysis.
    # Phase 14u: spatial bed mask in x.  Bed occupies cells whose x_mid
    # falls in [bed_x_start, bed_x_end].  Defaults to full domain (i=0..Nx)
    # when bed_x_start=0 and bed_x_end=Lx.  Setting bed_x_start > 0 leaves
    # bare ground upstream — wind develops naturally before the bed leading
    # edge (matches Cheney 1993 setup, Pimont 2009 FIRETEC validation).
    i_bed_start = int(np.searchsorted(grid.x_mid, bed_x_start, side='left'))
    i_bed_end   = int(np.searchsorted(grid.x_mid, bed_x_end,   side='right'))
    if i_bed_start >= i_bed_end:
        raise ValueError(
            f"bed_x_start={bed_x_start} >= bed_x_end={bed_x_end} after grid "
            f"snapping (i_bed_start={i_bed_start}, i_bed_end={i_bed_end})")
    alpha_s_avg = rho_b / _RHO_PARTICLE   # bed-volume-averaged solid fraction
    bed_slice = slice(0, grid.n_z_bed)
    # Per-cell α_s profile.  'uniform' = constant; 'exponential' = bottom-
    # loaded (Massman 1997 BLM 83:407, Lalic 2004 BLM 113:99 canopy
    # surveys document non-uniform leaf-area-density profiles for grass).
    # Exponential form: α(z) = α_max · exp(-K · z/h), normalized so that
    # the cell-averaged integral over the bed depth equals α_s_avg ×
    # h_bed.  K = alpha_s_decay_k (1.5 default = moderate stratification
    # matching cured pasture grass thatch+stem profile).  Conserves
    # total fuel mass per area.
    cell_factors = np.ones(grid.n_z_bed)
    if alpha_s_profile_type == "exponential":
        # Integrate exp(-K z/h) over each cell [z_k_bot, z_k_top].
        K = float(alpha_s_decay_k)
        if K > 0.0:
            # Normalization: α_max such that ∫α(z)dz = α_avg · h_bed.
            #   α_max = α_avg · K / (1 - exp(-K))
            # Cell factor = (h_bed/dz_k) · (α_max/α_avg) · 1/K ·
            #               [exp(-K·z_k_bot/h) − exp(-K·z_k_top/h)]
            #             = (h_bed/dz_k) / (1 − exp(−K)) ·
            #               [exp(-K·z_k_bot/h) − exp(-K·z_k_top/h)]
            denom = 1.0 - math.exp(-K)
            z_bot = 0.0
            for k in range(grid.n_z_bed):
                dz_k = grid.dz_arr[k]
                z_top = z_bot + dz_k
                cell_factors[k] = (
                    (h_bed / dz_k) *
                    (math.exp(-K * z_bot / h_bed) - math.exp(-K * z_top / h_bed)) /
                    denom
                )
                z_bot = z_top
            # Sanity-check: weighted sum = n_z_bed (since each cell's
            # contribution to ∫α(z)dz is α_avg · dz_k · cell_factor[k],
            # and the integral should be α_avg · h_bed = α_avg · Σdz_k).
            weighted_sum = sum(grid.dz_arr[k] * cell_factors[k]
                                for k in range(grid.n_z_bed))
            expected = h_bed
            if abs(weighted_sum - expected) > 1e-6 * expected:
                # Renormalize to be safe.
                cell_factors *= expected / weighted_sum
    elif alpha_s_profile_type != "uniform":
        raise ValueError(
            f"alpha_s_profile_type={alpha_s_profile_type!r} not in "
            f"{{'uniform', 'exponential'}}"
        )
    for k in range(grid.n_z_bed):
        state.alpha_s[k, :, i_bed_start:i_bed_end] = alpha_s_avg * cell_factors[k]
        state.m_hemi [k, :, i_bed_start:i_bed_end] = rho_b * cell_factors[k]
    state.m_cell [bed_slice, :, i_bed_start:i_bed_end] = 0.0
    state.m_lign [bed_slice, :, i_bed_start:i_bed_end] = 0.0
    print(f"  [bed-init] alpha_s_profile={alpha_s_profile_type}  "
          f"cell_factors={cell_factors.tolist()}",
          flush=True)

    # Phase 14aq (re-added on Phase 14ax base): y-periodic Fourier
    # perturbation on bed α_s + m_hemi to introduce fuel-bed heterogeneity
    # from t=0 (FIRETEC pattern, Linn 2002).  Y-periodic by construction (sum
    # of cosines with k·j/Ny phases) so coexists with periodic-y BC.
    _fpert_enable = bool(getattr(outdoor_cfg, "fuel_pert_enable", False))
    if _fpert_enable and grid.n_z_bed > 0 and i_bed_end > i_bed_start:
        _fpert_seed = int(getattr(outdoor_cfg, "fuel_pert_seed", 13))
        _fpert_amp  = float(getattr(outdoor_cfg, "fuel_pert_amp", 0.05))
        _fpert_kmax = int(getattr(outdoor_cfg, "fuel_pert_kmax", 2))
        _fp_rng = np.random.Generator(np.random.PCG64(seed=_fpert_seed))
        _n_x_bed = i_bed_end - i_bed_start
        _j_arr = np.arange(grid.Ny, dtype=np.float64)
        _i_arr = np.arange(_n_x_bed, dtype=np.float64)
        _pert_2d = np.zeros((grid.Ny, _n_x_bed), dtype=np.float64)
        for _ky in range(1, _fpert_kmax + 1):
            for _kx in range(0, _fpert_kmax + 1):
                _A = float(_fp_rng.standard_normal())
                _phi_y = float(_fp_rng.uniform(0.0, 2 * math.pi))
                _phi_x = float(_fp_rng.uniform(0.0, 2 * math.pi))
                _pert_2d += _A * (
                    np.cos(2 * math.pi * _ky * _j_arr / grid.Ny + _phi_y)[:, None]
                  * np.cos(2 * math.pi * _kx * _i_arr / max(_n_x_bed, 1) + _phi_x)[None, :]
                )
        _pert_2d *= _fpert_amp / max(np.abs(_pert_2d).max(), 1.0e-12)
        for k in range(grid.n_z_bed):
            state.alpha_s[k, :, i_bed_start:i_bed_end] *= (1.0 + _pert_2d)
            state.m_hemi [k, :, i_bed_start:i_bed_end] *= (1.0 + _pert_2d)
        _sigma_pert = float(_pert_2d.std())
        _abs_max = float(np.abs(_pert_2d).max())
        print(f"  [fuel-pert] amp={_fpert_amp:.3f}  kmax={_fpert_kmax}  "
              f"σ(δα_s/α_s)={_sigma_pert:.4f}  |max|={_abs_max:.4f}  "
              f"seed={_fpert_seed}", flush=True)

    # Initial fuel mass per cell — needed for MD2004 char-fraction limit
    # (Shafizadeh 1968).  Stored separately so pyrolysis kernel knows the
    # starting mass for the char floor.
    m_initial = state.m_hemi.copy()

    # Moisture state per cell (Grishin 1984; Margerit & Séro-Guillaume 2002).
    # Water mass per cell volume [kg/m³] = ρ_bulk × M_f.  Pyrolysis is
    # blocked while local water > 1% of initial; positive q_in is consumed
    # by evaporation at L_v before heating the solid.  Ported from 2D
    # spread.py (commit 68f109f).  Without this, 3D over-predicts HRRPUA
    # because dry pyrolysis fires before water has been removed.
    M_f = float(getattr(outdoor_cfg, 'initial_moisture_frac', 0.0) or 0.0)

    # ── Phase 19 empirical-ROS hybrid (low-U) ────────────────────────────────
    # Precompute the empirical ROS and blend weight once.  These are scalars
    # because U_inflow is constant in our cases; for a dynamic-wind extension,
    # move this inside the loop and recompute per step.
    _empirical_seed_step_counter = 0
    if empirical_ros_enable:
        _empirical_ros_m_s = _empirical_ros.evaluate_empirical_ros(
            empirical_ros_model, wind_speed_m_s, M_f, empirical_ros_a_ch,
            age_yr=empirical_ros_age_yr,
        )
        _empirical_blend_w = _empirical_ros.blend_resolved_empirical(
            wind_speed_m_s, empirical_ros_u_threshold_m_s,
            empirical_ros_blend_width_m_s,
        )
        _coeff_str = (f"a_ch={empirical_ros_a_ch}"
                      if empirical_ros_model == "cheney_eq6"
                      else f"age_yr={empirical_ros_age_yr}")
        print(f"  [empirical-ros] model={empirical_ros_model}  "
              f"{_coeff_str}  "
              f"ROS_empirical={_empirical_ros_m_s * 60.0:.3f} m/min  "
              f"blend_weight_empirical={_empirical_blend_w:.3f}  "
              f"(U={wind_speed_m_s} vs U_threshold={empirical_ros_u_threshold_m_s}"
              f"±{empirical_ros_blend_width_m_s})",
              flush=True)
    else:
        _empirical_ros_m_s = 0.0
        _empirical_blend_w = 0.0

    L_v = 2_257_000.0   # [J/kg] latent heat of water
    m_water = np.zeros_like(state.m_hemi)
    m_water[bed_slice, :, i_bed_start:i_bed_end] = rho_b * M_f
    m_water_init = rho_b * M_f             # scalar initial water density
    _has_moisture = m_water_init > 0

    # ── Phase 16 — Lagrangian sub-grid bed particles ────────────────────
    # When enabled, allocate one buffer of N_per_cell particles per
    # α_s > 0 cell, initialize from current m_hemi/m_water, then zero
    # the Eulerian m_hemi / m_water (particles ARE the bed now).
    # Per main-loop step we call step_bed_particles instead of the four
    # Eulerian pyrolysis kernels, and skip step_gas_solid_coupling
    # (particles handle their own T_s + emit Q_g_conv directly).
    _bed_buf = None
    _bed_N_alloc = 0
    _bed_Sp = _bed_Sd = _bed_Qp = _bed_Qd = None
    _bed_YFs = _bed_Qch = _bed_Qsm = _bed_Qgc = None
    _bed_M_local = None
    _bed_n_alive_diag = _bed_n_burned_diag = None
    if lagrangian_bed_enable:
        _bed_sav = (float(lagrangian_bed_sav_1_m)
                    if lagrangian_bed_sav_1_m is not None
                    else float(getattr(outdoor_cfg, "sav_ratio_1_m", 2000.0)))
        _drying_mode_map = {"arrhenius": 0, "equilibrium": 1, "combined": 2}
        if lagrangian_bed_drying_mode not in _drying_mode_map:
            raise ValueError(
                f"lagrangian_bed_drying_mode must be one of "
                f"{sorted(_drying_mode_map)} (got {lagrangian_bed_drying_mode!r})")
        _bed_drying_mode_int = _drying_mode_map[lagrangian_bed_drying_mode]
        _n_per_cell = int(lagrangian_bed_N_per_cell)
        # Count bed cells (only α_s > 0 inside fuel-bed extent)
        _alpha_bed = state.alpha_s
        _n_bed_cells = int((_alpha_bed[:grid.n_z_bed,
                                       :,
                                       i_bed_start:i_bed_end] > 0).sum())
        _N_max_bed = _n_bed_cells * _n_per_cell
        if _N_max_bed > 0:
            _bed_buf = lagrangian_bed_3d.allocate_bed_particle_buffers(_N_max_bed)
            _bed_N_alloc = lagrangian_bed_3d.initialize_bed_particles_from_alpha_s(
                _bed_buf, _alpha_bed,
                rho_b_dry=rho_b, moisture_frac=M_f,
                T_amb=T_amb,
                dx=grid.dx, dy=grid.dy, dz_arr=grid.dz_arr,
                n_z_bed=grid.n_z_bed, n_per_cell=_n_per_cell,
                sav=_bed_sav,
                i_lo=i_bed_start, i_hi=i_bed_end,
            )
            # Phase 17d solid-phase ignition is wired below after
            # i_src_start/i_src_end are defined (search for
            # solid_phase_ignition_enable).
            # Zero Eulerian inventory — particles own the bed.
            state.m_hemi.fill(0.0)
            m_water.fill(0.0)
            m_initial.fill(0.0)
            # Per-step source arrays (overwritten each call by kernel).
            # Allocate with explicit grid shape — `shape` local isn't bound
            # until later state init below.
            _bp_shape = (grid.Nz, grid.Ny, grid.Nx)
            _bed_Sp  = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Sd  = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Qp  = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Qd  = np.zeros(_bp_shape, dtype=np.float64)
            _bed_YFs = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Qch = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Qsm = np.zeros(_bp_shape, dtype=np.float64)
            _bed_Qgc = np.zeros(_bp_shape, dtype=np.float64)
            # Per-cell bed moisture (m_water/m_solid) for DOM κ_solid
            # wet-bed scaling (Phase 17a — Mell 2007 / Linn 2002).
            _bed_M_local = np.zeros(_bp_shape, dtype=np.float64)
            _bed_n_alive_diag  = np.zeros(1, dtype=np.int64)
            _bed_n_burned_diag = np.zeros(1, dtype=np.int64)
            _bed_diag_max      = np.zeros(16, dtype=np.float64)
            print(f"  [lagrangian-bed] ENABLED  N_per_cell={_n_per_cell}  "
                  f"bed_cells={_n_bed_cells}  N_alloc={_bed_N_alloc}  "
                  f"sav={_bed_sav:.0f}  h_conv={lagrangian_bed_h_conv:.0f}",
                  flush=True)

    # ── Derived parameters ───────────────────────────────────────────────
    sigma_sav = float(getattr(outdoor_cfg, "sav_ratio_1_m", 6562.0))
    chi_rad   = 0.34   # NIST TN 2314 (Sung 2025) for grass; matches existing 2D
    if chi_rad_override is not None:
        chi_rad = float(chi_rad_override)
        print(f"  [chi_rad] OVERRIDE: chi_rad={chi_rad} "
              f"(diagnostic; default 0.34)", flush=True)
    T_ign     = T_amb + 300.0   # standard piloted ignition for cellulosic
    ignition_duration_s = float(getattr(outdoor_cfg, "ignition_duration_s", 30.0))
    terrain   = getattr(outdoor_cfg, "terrain", "open")
    U_mf      = midflame_wind_speed(wind_speed_m_s, terrain)

    # ── Inlet wind profile: atmospheric log-law BL over bare ground ─────
    # The inflow at x=0 lies UPSTREAM of the fuel bed (bed_x_start ≥ 2m
    # in production runs).  The wind profile there is a developed
    # atmospheric boundary layer over BARE GROUND, not an in-canopy
    # profile.  Apply Monin-Obukhov log-law pinned by:
    #   u(z=0) = 0       (no-slip at ground)
    #   u(z=10m) = U_10  (reference wind input parameter)
    # with surface roughness z_0 = 0.01 m (short stubble / bare soil,
    # Monteith & Unsworth 2013 Table 4.1).
    #
    # Inside the bed (downstream of bed_x_start), the Ergun + Pimont
    # porous-bed drag handles canopy attenuation organically — no
    # prescribed Cionco profile needed.  Previously the inlet imposed
    # a Cionco bed-equilibrium profile (with u(z=0) = 0.37·U_mf, far
    # from zero), which artificially raised the in-bed wind at the bed
    # leading edge and contributed to advective washout failures in
    # sparse Nat-pasture cases.  The legacy wind_profile_in_bed()
    # remains available in boundary.py for backward compatibility but
    # Phase 23 Refactor 2A: inflow-profile computation now delegated
    # to the registered BoundaryCondition class.  Default
    # "outdoor_wind" preserves the exact pre-refactor behaviour
    # (bit-exact-invariant per Rule #17).  Future cup-burner /
    # suppressant-injection cases plug in as additional kinds; see
    # model_outdoor/boundary_conditions/.
    # Assemble BC kwargs.  Outdoor kwargs go to every BC (unused ones
    # absorbed by **_ignored_outdoor_kwargs).  Cup-burner-specific
    # kwargs (wick_enable, Y_agent_coflow, fuel_jet_radius, etc.) come
    # from the deck via outdoor_cfg attributes, defaulted if unset.
    _bc_kwargs = dict(
        wind_speed_m_s=wind_speed_m_s,
        wind_profile_type=wind_profile_type,
        h_bed=h_bed,
        sigma_sav=sigma_sav,
        alpha_s_avg=alpha_s_avg,
    )
    if boundary_condition_kind == "cup_burner":
        # Read cup-burner deck flags; use CupBurnerBC.__init__ defaults
        # for anything not in the deck.
        for _key in ("fuel_jet_radius_m", "chimney_radius_m",
                     "fuel_jet_velocity_m_s", "coflow_velocity_m_s",
                     "Y_F_fuel", "Y_O2_coflow", "Y_agent_coflow",
                     "T_inlet_K",
                     "wick_enable", "wick_Y_F", "wick_z_lo", "wick_z_hi"):
            _v = getattr(outdoor_cfg, _key, None)
            if _v is not None:
                _bc_kwargs[_key] = _v
    _bc = _get_bc_class(boundary_condition_kind)(**_bc_kwargs)
    # Outdoor cases initialize the interior u-field from the wind profile.
    # Cup burner has no x-inlet wind — start quiescent (u=0 everywhere);
    # CupBurnerBC.configure() will set the z-min inlet later and
    # buoyancy/coflow will spin up the flow naturally.
    if boundary_condition_kind == "outdoor_wind":
        u_inlet = _bc.build_u_inlet(grid)
        state.u[:, :, 0] = u_inlet
        state.u[:, :, :] = u_inlet[:, :, np.newaxis]
    else:
        # placeholder for proj_solver.set_inlet_BC below (no x-inlet)
        u_inlet = np.zeros((grid.Nz, grid.Ny))

    # Phase 14ap (re-added on Phase 14ax base): 3D Synthetic Eddy Method for
    # inlet turbulence — breaks y-translation symmetry that otherwise
    # prevents resolved flame-finger structure (phase14ao_1cm_les_no_fingers).
    # Eddies placed randomly in (-σ, +σ)×(0, Ly)×(0, Lz+σ), each carrying a
    # signed u/v/w perturbation via tent shape function (Jarrin 2006).
    # Phase 14at re-added 2026-05-30: solid-phase form-drag coefficient as
    # deck input.  Defaults to 0.30 (Wilson-Shaw) per drag_3d.C_D_DEFAULT.
    _canopy_C_d = float(getattr(outdoor_cfg, "canopy_C_d",
                                 drag_3d.C_D_DEFAULT))
    print(f"  [drag] canopy_C_d={_canopy_C_d:.3f}  "
          f"(Wilson-Shaw default 0.30; Mueller 2021 recommends 0.15 for pasture)",
          flush=True)
    # Phase 14at re-added 2026-05-30: Sanz 2003 canopy β_p/β_d as deck inputs.
    # Defaults per turbulence_3d module: 1.0 / 4.0 (Sanz 2003 dense canopy).
    _canopy_beta_p = float(getattr(outdoor_cfg, "canopy_beta_p",
                                    turbulence_3d.BETA_P_CANOPY_DEFAULT))
    _canopy_beta_d = float(getattr(outdoor_cfg, "canopy_beta_d",
                                    turbulence_3d.BETA_D_CANOPY_DEFAULT))
    print(f"  [canopy] β_p={_canopy_beta_p:.2f}  β_d={_canopy_beta_d:.2f}  "
          f"(Sanz 2003 defaults 1.0/4.0; Brunet 1994/Massman 1997 suggest "
          f"β_d ~1.5-2.0 for pasture)", flush=True)

    _sem_enable = bool(getattr(outdoor_cfg, "sem_enable", False))
    if _sem_enable:
        from model_outdoor.physics_3d.sem_3d import sem_tent as _sem_tent
        _sem_seed   = int(getattr(outdoor_cfg, "sem_seed", 42))
        _sem_I_t    = float(getattr(outdoor_cfg, "sem_I_t", 0.20))
        _sem_N      = int(getattr(outdoor_cfg, "sem_N", 200))
        _sem_sigma  = max(h_bed, 2.0 * grid.dx)
        _sem_rng    = np.random.Generator(np.random.PCG64(seed=_sem_seed))
        _sem_xk     = _sem_rng.uniform(-_sem_sigma, _sem_sigma, size=_sem_N)
        _sem_yk     = _sem_rng.uniform(0.0, grid.Ly, size=_sem_N)
        _Lz_total   = float(np.sum(grid.dz_arr))
        _sem_zk     = _sem_rng.uniform(-_sem_sigma, _Lz_total + _sem_sigma,
                                       size=_sem_N)
        _sem_eps    = _sem_rng.choice([-1.0, 1.0], size=(_sem_N, 3))  # u, v, w signs
        _sem_Uc     = max(U_mf, 0.1)
        _sem_amp    = _sem_I_t * U_mf
        _sem_inv_sqN = 1.0 / math.sqrt(_sem_N)
        _u_inlet_base = u_inlet.copy()       # unperturbed log-law baseline
        print(f"  [SEM] N={_sem_N} σ={_sem_sigma:.3f}m  "
              f"amp={_sem_amp:.4f}m/s  I_t={_sem_I_t}  seed={_sem_seed}",
              flush=True)
    else:
        _u_inlet_base = u_inlet
    # Always-allocated v_inlet, w_inlet ghost arrays (zero when SEM off).
    v_inlet = np.zeros((grid.Nz, grid.Ny), dtype=np.float64)
    w_inlet = np.zeros((grid.Nz, grid.Ny), dtype=np.float64)

    # Phase 14y: outflow-sponge sigma profile.  Quadratic ramp over the last
    # N_SPONGE cells; σ_max set so e-folding timescale is ~5 dt at the
    # outlet.  Israeli & Orszag 1981 quadratic ramp avoids reflection.
    _N_SPONGE = max(3, min(int(round(0.5 / grid.dx)), grid.Nx // 4))
    # last 0.5 m of domain, but capped at Nx/4 for small domains (cup burner
    # only has ~50 cells across a 100 mm chimney — a 250-cell sponge is
    # nonsensical there; Nx/4 keeps at least a usable interior).
    _SIGMA_SPONGE_MAX = 5.0   # [1/s] ~0.2 s e-folding at outlet
    _Y_F_SPONGE_SKIP  = 1.0e-3   # skip sponge in cells with Y_F above this
                                  # (active flame plume — don't damp it)
    _sigma_x_sponge = np.zeros(grid.Nx, dtype=np.float64)
    for _i in range(grid.Nx - _N_SPONGE, grid.Nx):
        _frac = (_i - (grid.Nx - _N_SPONGE)) / _N_SPONGE
        _sigma_x_sponge[_i] = _SIGMA_SPONGE_MAX * _frac * _frac

    # ── Ignition source: small x-strip starting at bed leading edge ─────
    # n_src cells matches the deck's notional 0.5m source width (ignition_q
    # zone) — physically a drip-torch / match starter.  Only bed cells are
    # clamped (k < n_z_bed).  Phase 14u: source begins at i_bed_start so the
    # drip-torch sits on the bed, not on bare ground upstream of the bed.
    _src_width_m = 0.5 * float(ignition_width_mult)   # Phase 15L kick width
    n_src = max(1, int(round(_src_width_m / grid.dx)))
    i_src_start = i_bed_start
    i_src_end = min(i_bed_end, i_src_start + n_src)
    n_src = i_src_end - i_src_start  # truncate if bed is too small

    # Phase 17d (2026-06-25): solid-phase ignition.  Skip the gas T_g pin
    # entirely; pre-heat source-patch bed particles to well above T_ign so
    # they self-ignite at t=0.  Fire then spreads via natural combustion-
    # driven buoyancy and DOM radiation only — no artificial sustained gas
    # heating.  This is the architectural fix for the low-U plume-reversal
    # artifact: the 5s × 1500K gas pin creates ~250 kJ/m² of artificial
    # gas-side energy that drives a coherent buoyant plume strong enough to
    # reverse the bed wind at low U.  Solid-phase ignition replaces that
    # with a one-shot particle-T_s assignment — the gas warms via natural
    # particle convection (h_conv) and pyrolysis volatile injection,
    # matching how a real torch-flash igniter works.
    if solid_phase_ignition_enable and lagrangian_bed_enable \
            and _bed_buf is not None:
        _t_s_seed_K = float(solid_phase_ignition_T_s_K)
        _x_buf = _bed_buf["x"]
        _alive = _bed_buf["alive"]
        _T_s_arr = _bed_buf["T_s"]
        _x_lo = grid.x_mid[i_src_start] - 0.5 * grid.dx
        _x_hi = grid.x_mid[i_src_end - 1] + 0.5 * grid.dx
        _n_seeded = 0
        for _p in range(_bed_N_alloc):
            if _alive[_p] == 0:
                continue
            if _x_lo <= _x_buf[_p] < _x_hi:
                _T_s_arr[_p] = _t_s_seed_K
                _n_seeded += 1
        print(f"  [solid-ignition] seeded {_n_seeded} particles at "
              f"T_s={_t_s_seed_K:.0f}K  in source patch "
              f"x∈[{_x_lo:.2f}, {_x_hi:.2f}]m",
              flush=True)

    # ── Pre-allocate work arrays (avoid per-step allocation) ─────────────
    shape = (grid.Nz, grid.Ny, grid.Nx)
    S_pyro  = np.zeros(shape, dtype=np.float64)
    Q_pyro  = np.zeros(shape, dtype=np.float64)
    _Q_char = np.zeros(shape, dtype=np.float64)   # Phase 14y-char workspace
    _Q_smold = np.zeros(shape, dtype=np.float64)  # Phase 14z-A1 smoldering workspace
    _Q_dry  = np.zeros(shape, dtype=np.float64)   # Phase 14 Try 7 R_d workspace
    omega   = np.zeros(shape, dtype=np.float64)
    omega_O2 = np.zeros(shape, dtype=np.float64)   # O₂-supply rate limit
    Q_comb  = np.zeros(shape, dtype=np.float64)
    # Phase 14s: zero-source array for Y_O2 transport (chemistry sink
    # already applied in operator A / step_chemistry_ode).  Pre-allocated
    # to avoid per-substep np.zeros_like allocation in the hot loop.
    _S_zero_o2 = np.zeros(shape, dtype=np.float64)
    Fx      = np.zeros(shape, dtype=np.float64)
    Fy      = np.zeros(shape, dtype=np.float64)
    Fz      = np.zeros(shape, dtype=np.float64)
    q_rad   = np.zeros(shape, dtype=np.float64)
    q_rad_gas = np.zeros(shape, dtype=np.float64)   # Phase 14a: P1 gas-phase net rad
    # Phase 14v-bc-soil: 1D vertical soil sub-grid for ground BC heat reservoir.
    # Geometric stretch from 1 mm at surface; total ~30 mm covers thermal
    # penetration depth at fire timescales.  Pimont 2006 / FIRESTAR pattern.
    from model_outdoor.physics_3d import soil_3d
    _N_SOIL = 6
    soil_dz, soil_d_above, soil_d_below, soil_depth = soil_3d.build_soil_grid(
        n_soil=_N_SOIL, dz_first=0.001, growth=1.5,
    )
    T_soil = np.full((_N_SOIL, grid.Ny, grid.Nx), T_amb, dtype=np.float64)
    q_in_soil = np.zeros((grid.Ny, grid.Nx), dtype=np.float64)
    burning_mask = np.zeros(shape, dtype=np.float64)
    # Mixing time scale.  Phase 14b: τ_mix = k/ε from k-ε turbulence (replaces
    # the FDS buoyancy-regime fallback τ_mix = sqrt(2·Δ/g) when k-ε is active).
    _delta = max(grid.dx, grid.dy, grid.dz)
    _tau_mix_buoy = math.sqrt(2.0 * _delta / _G)
    tau_mix = np.full(shape, _tau_mix_buoy, dtype=np.float64)

    # Phase 14b: 3D k-ε state arrays + workspace
    # Initial values: ν_t ≈ 0.01 m²/s (mild turbulence baseline), k & ε
    # back-calculated from C_μ k²/ε = ν_t with k ≈ (3/2)·(I_t·U)² where
    # I_t ≈ 0.10 (atmospheric turbulence intensity, Garratt 1992).
    _k_init = max(1.5 * (0.10 * max(U_mf, 0.5)) ** 2, 1.0e-4)
    _eps_init = turbulence_3d.C_MU * _k_init * _k_init / 0.01   # ν_t = 0.01
    k_turb = np.full(shape, _k_init, dtype=np.float64)
    eps_turb = np.full(shape, _eps_init, dtype=np.float64)
    nu_t = np.full(shape, 0.01, dtype=np.float64)
    _S_mag2_work = np.zeros(shape, dtype=np.float64)
    _Omega_mag2_work = np.zeros(shape, dtype=np.float64)   # Phase 14c.1: vorticity invariant
    # Phase 14v-bc Way B: wall-equilibrium ghost arrays for k and ε at the
    # wall face k=-0.5.  Filled by apply_wall_function each step (when
    # wall_function=True), used by step_k_epsilon as the kzL/ezL ghost.
    # Default K_MIN/EPS_MIN reproduces the no-WF behavior.
    k_wall_ghost   = np.full((grid.Ny, grid.Nx), turbulence_3d.K_MIN,   dtype=np.float64)
    eps_wall_ghost = np.full((grid.Ny, grid.Nx), turbulence_3d.EPS_MIN, dtype=np.float64)

    # ── Phase 14x: level-set front + flame-body bootstrap ────────────────
    # 3D signed-distance field tracks burning front position kinematically;
    # v_n driven by CFD heat fluxes (Frankman + DOM forward) integrated
    # over a constant-physical-size band (mesh-convergent).  Behind the
    # front, resolved EBU + chemistry runs as before (B2 architecture).
    # Newly-burning cells receive Q_bootstrap to bridge chem-bootstrap gap.
    # Reference: docs/phase14x_levelset_flame_body_plan.md;
    # Mell 2007 IJWF 16:1 §3.4 WFDS level-set option.
    lset = flame_front_3d.LevelSetFront3D(
        Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
        dx=grid.dx, dy=grid.dy, dz_arr=grid.dz_arr,
        L_burnout=flame_front_3d.L_BURNOUT_M,
    )
    # Ahead-band width for level-set Frankman/DOM heat-flux integral.
    # Fixed at DX_VN_BAND_M (Mell 2007 WFDS §3.4 preheating band length).
    # Natural plume tilt with wind is captured by the gas-phase phi_flame
    # level-set (state-derived from advected T_g/Y_F/omega), not by an
    # external Albini-1981 geometric tilt formula.  Floor at dx so the
    # mask retains ≥ 1 cell of width.
    _band_m_tilt = max(grid.dx, flame_front_3d.DX_VN_BAND_M)
    print(f"  [14x] U_10={wind_speed_m_s:.2f} m/s  "
          f"ahead_band={_band_m_tilt:.3f} m  (fixed; no Albini tilt)",
          flush=True)
    # Pre-burn the source patch (cells in source patch start as flame body).
    # Need k_top_bed = grid.n_z_bed - 1 (last bed cell index).
    # If n_z_bed=0 (no bed; cold-flow setup), skip — no fire to track.
    if grid.n_z_bed > 0 and i_src_end > i_src_start:
        lset.initialize_source_patch(
            i_start=i_src_start, i_end=i_src_end,
            k_top_bed=grid.n_z_bed - 1,
            x_mid=grid.x_mid,
        )
    # Per-cell time-since-ignition for bootstrap window.
    # Source patch cells start at age=0 so they ALSO get bootstrap during
    # the first t_bootstrap seconds — drip torch alone is insufficient at
    # the lit-correct in-bed wind to reach chem-bootstrap temperature.
    # Net source-cell heating during first 2s = drip(190) + bootstrap(500) =
    # 690 kW/m³ → gas spikes to ~1300K transient → chemistry self-sustains.
    cell_age = np.full(shape, np.inf, dtype=np.float64)
    cell_age[lset.flame_body_mask()] = 0.0
    # Phase 15O — per-cell time-of-last-tendril-spawn (frequency cap).
    # Initialised to a large negative sentinel so the first eligible
    # spawn at t≈0 always passes the cap.
    last_spawn_time_3d = np.full(shape, finney_tendril_3d._NEVER_SPAWNED,
                                  dtype=np.float64)
    finney_tendril_count = np.zeros(1, dtype=np.int64)
    # Phase 15O.1 — persistent per-cell remaining inventory for the
    # time-spread Eulerian release.  10 fields (5 for source-side sink,
    # 5 for target-side deposit), all zero-initialised.  Only used when
    # finney_tendril_t_contact_s > 0.
    _ft_sink_M     = np.zeros(shape, dtype=np.float64)
    _ft_sink_E     = np.zeros(shape, dtype=np.float64)
    _ft_sink_Yf    = np.zeros(shape, dtype=np.float64)
    _ft_sink_Px    = np.zeros(shape, dtype=np.float64)
    _ft_sink_t_rem = np.zeros(shape, dtype=np.float64)
    _ft_dep_M     = np.zeros(shape, dtype=np.float64)
    _ft_dep_E     = np.zeros(shape, dtype=np.float64)
    _ft_dep_Yf    = np.zeros(shape, dtype=np.float64)
    _ft_dep_Px    = np.zeros(shape, dtype=np.float64)
    _ft_dep_t_rem = np.zeros(shape, dtype=np.float64)
    # Phase 15P — Lagrangian Finney particle state (allocated only if
    # finney_lagrangian_enable; otherwise length-0 buffers).
    if finney_lagrangian_enable:
        _N_max_p = int(finney_lagrangian_N_max)
    else:
        _N_max_p = 0
    _fl_x     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_y     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_z     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_u     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_v     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_w     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_m     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_E     = np.zeros(_N_max_p, dtype=np.float64)
    _fl_Yf    = np.zeros(_N_max_p, dtype=np.float64)
    _fl_t_rem = np.zeros(_N_max_p, dtype=np.float64)
    _fl_alive = np.zeros(_N_max_p, dtype=np.int8)
    _fl_n_alive    = np.zeros(1, dtype=np.int64)
    _fl_n_exit     = np.zeros(1, dtype=np.int64)
    _fl_n_spawn    = np.zeros(1, dtype=np.int64)
    _fl_n_overflow = np.zeros(1, dtype=np.int64)
    # Workspace for Stage 2 coupling
    q_frankman_3d = np.zeros(shape, dtype=np.float64)
    q_dom_fwd_3d  = np.zeros(shape, dtype=np.float64)
    # Phase 15F snapshot diagnostics — initialised None so the snapshot
    # writer can reference even before the cold-bed case branch.
    q_in_at_front_2d: np.ndarray | None = None
    v_n_2d:           np.ndarray | None = None
    _q_in_to_solid = np.zeros(shape, dtype=np.float64)   # q_rad + q_frankman for coupling
    v_n_field     = np.zeros(shape, dtype=np.float64)
    _v_n_extinct_count = 0   # consecutive steps with v_n.max() < threshold
    _last_14x_print_t = -10.0   # separate from _last_print_t
    # Phase 14y: pin closure dropped (scaffolding); buffer no longer needed.

    # ── Pressure projection solver (LU-cached) ───────────────────────────
    proj_solver = projection_3d.ProjectionSolver3D(
        Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
        dy=grid.dy, dx=grid.dx,
        dz_arr=grid.dz_arr,
        d_face_above=grid.d_face_above,
        d_face_below=grid.d_face_below,
        y_bc=y_bc,
        method=projection_method,
        cg_rtol=projection_cg_rtol,
        amg_rebuild_every=projection_amg_rebuild_every,
    )
    # Phase 14v-bc: tell the projection the inlet face velocity (u_inlet
    # at face -0.5, x=0).  Used by div_compat as the mirror ghost value:
    # u_ghost = u_inlet ⇒ div_x[0] = (u[0] − u_inlet)/dx, capturing
    # deviations the projection corrects via pressure gradient at face 0.5.
    proj_solver.set_inlet_BC(u_inlet)
    # Phase 23 Refactor 2D: cup burner (or any future z-min-inlet BC) has
    # its own setup that populates the projection solver's z-min ghost,
    # per-species inlet ghost arrays on the BC object, T_g interior IC,
    # and ignition hot spot.  For outdoor_wind the u_inlet code above is
    # sufficient; skip the BC-side configure() to avoid duplicating it.
    _is_cup_burner = (boundary_condition_kind == "cup_burner")
    if _is_cup_burner:
        _bc.configure(proj_solver=proj_solver, grid=grid, state=state)
    # Phase 14a→14m: radiation solver — P1 (Eddington) or DOM (S4 ordinates).
    # DOM (Phase 14m) is the default: captures angular distribution of flux
    # in opaque dense beds where P1's isotropic averaging underpredicts
    # direct downward emission from above-bed flame plume.
    # Reference: Modest 2003 §16; FIRESTAR (Morvan 2009).
    if radiation_solver == "dom":
        _dom_kwargs = dict(
            Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
            dy=grid.dy, dx=grid.dx,
            dz_arr=grid.dz_arr,
            d_face_above=grid.d_face_above,
            d_face_below=grid.d_face_below,
            y_bc=y_bc,
            N_quadrature=4,
        )
        if dom_kappa_gas_max_override is not None:
            _dom_kwargs["kappa_gas_max"] = float(dom_kappa_gas_max_override)
        rad_solver = dom_3d.DOMRadiationSolver(**_dom_kwargs)
    elif radiation_solver == "p1":
        rad_solver = radiation_3d.P1RadiationSolver(
            Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx,
            dy=grid.dy, dx=grid.dx,
            dz_arr=grid.dz_arr,
            d_face_above=grid.d_face_above,
            d_face_below=grid.d_face_below,
            y_bc=y_bc,
        )
    else:
        raise ValueError(f"radiation_solver must be 'dom' or 'p1'; got {radiation_solver!r}")

    # ── Time-step from CFL ───────────────────────────────────────────────
    # Advective:    dt < cfl_factor · min(dx, dy, dz_arr) / U_max
    # Diffusion:    dt < 0.25 · dz_min² / ν_t_max  (turbulent diffusion is
    #               the binding constraint when dz is small relative to the
    #               turbulent length scale; molecular ν_air ~ 1.5e-5 is loose)
    # Buoyancy:     w_buoy ~ sqrt(g · h_bed) ~ 1.9 m/s for h_bed=0.37
    # Phase 14t-A: use dz_arr.min() (not the legacy scalar grid.dz which
    # equals dz_bed) so BL refinement at z=0 properly drives dt down.
    _U_char = max(U_mf, math.sqrt(_G * h_bed * 0.5), 0.1)
    _dz_min = float(grid.dz_arr.min())
    _dt_adv = cfl_factor * min(grid.dx, grid.dy, _dz_min) / _U_char
    # Estimate ν_t_max from initial k-ε (will be refreshed each step inside
    # turbulence kernel; this is just for the initial dt cap).
    _nu_t_max = max(float(nu_t.max()), 1.0e-3)
    _dt_diff = 0.25 * _dz_min * _dz_min / _nu_t_max
    dt = min(_dt_adv, _dt_diff, 0.05)
    if dt <= 0.0:
        raise ValueError(f"computed dt={dt} ≤ 0 (check CFL params)")

    # ── Time loop (operator-split, fractional step Chorin 1967) ──────────
    # Phase 14u Option 3a: track ρ across steps for variable-density Chorin's
    # ∂ρ/∂t term in div_target.  Initial ρ_prev = current ρ (uniform at t=0).
    rho_prev = state.rho.copy()
    # Phase 14aw-2 (2026-05-27): track α_s across steps for the
    # volume-weighted divergence target ∂α_g/∂t = -∂α_s/∂t.  Only
    # consumed when outdoor.volume_weighted_projection=True; cheap so
    # we always cache.  Initial = current bed-init α_s ⇒ dα_s/dt = 0
    # at step 1, matching rho_prev convention.
    alpha_s_prev = state.alpha_s.copy()
    # Phase 14aw-2: deck flag.  Default False — opt-in.  See
    # OutdoorEnvConfig.volume_weighted_projection docstring for the
    # AMG-CG pathology this gating avoids by default.
    _volume_weighted_projection = bool(getattr(
        outdoor_cfg, "volume_weighted_projection", False))
    if _volume_weighted_projection:
        print(f"  [projection] Phase 14aw: volume-weighted ON "
              f"(α_g = 1 - α_s in matrix coeffs and divergence)", flush=True)

    # Phase 17b ROS-fix (2026-06-22): initialize the front history to the
    # actual source-patch front position, not (t=0, x=0).  Previously
    # _compute_steady_ros computed slope from (0, 0) to (t_end, x_end)
    # which inflated reported ROS by including the source-patch jump on
    # step 1 (e.g., x goes 0 → 1.5 m on init, falsely treated as
    # 1.5 m of fire spread).  With proper initial position, slope reflects
    # only actual front advance.
    if grid.n_z_bed > 0 and i_src_end > i_src_start:
        _initial_front_x = float(grid.x_mid[i_src_end - 1] + 0.5 * grid.dx)
    else:
        _initial_front_x = 0.0
    front_t: list = [0.0]
    front_x: list = [_initial_front_x]
    # Time-series diagnostics (captured every step for plotting).
    diag_t: list = [0.0]
    diag_Tg_max: list = [float(state.T_g.max())]
    diag_Ts_max: list = [float(state.T_s.max())]
    diag_Sp_max: list = [0.0]
    diag_Qc_max: list = [0.0]
    diag_omega_max: list = [0.0]
    diag_Y_max: list = [float(state.Y_fuel.max())]
    diag_n_ign: list = [0]
    # Projection convergence tracking: how often the safety cap binds
    # vs the threshold (proj_max_iter == proj_n_iter ⇒ cap binding).
    diag_proj_div_max: list = [0.0]
    diag_proj_n_iter: list = [0]
    t = 0.0
    _last_print_t = -10.0
    _last_advance_t = 0.0
    _last_front_x = 0.0
    _print_dt = 0.5
    _stall_window_s = 3.0   # s — exit if no front advance in this window
                             # (post-ignition phase only).  Tightened from 30s
                             # in Phase 14y: 30s was effectively never reached
                             # within typical sim_t budgets.
    _step = 0
    # Phase 14y-snap: optional state-snapshot capture for animations.
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    _last_snapshot_t = -1.0e30
    _snapshot_idx = 0

    # Per-kernel wall-time accumulator for profiling.  Prints summary at
    # end of run.  Each entry: (cumulative_seconds, call_count).
    import time as _time
    _timings = {}
    def _tick(label, t0):
        dt_tick = _time.perf_counter() - t0
        if label in _timings:
            _timings[label] = (_timings[label][0] + dt_tick, _timings[label][1] + 1)
        else:
            _timings[label] = (dt_tick, 1)
    _t_loop_start = _time.perf_counter()

    # Phase 15D-SS — steady-state-detector state (used iff
    # steady_state_detect=True; otherwise inert).
    _ss_last_check_t: float | None = None
    _ss_last_slope:   float | None = None
    _ss_consec:       int          = 0

    # Phase 15E — per-k counter for ε realizability cap diagnostic.
    # Reset each outer step; summed and logged when nonzero.
    _eps_cap_count = np.zeros(grid.Nz, dtype=np.int64)
    _eps_durbin_count = np.zeros(grid.Nz, dtype=np.int64)
    _eps_cap_last_log_t: float = -1.0

    # Phase 14ap-2 spin-up gate (re-added on Phase 14ax base 2026-05-30).
    # During t < _t_combustion_start the loop runs ONLY:
    #   momentum, k-ε turbulence, SEM inlet, drag, projection, BCs.
    # Combustion, pyrolysis, char_ox, smolder, DOM radiation, gas-solid
    # coupling, ignition pulse, T_g advection are all skipped — gives
    # a clean cold-flow window for BL diagnostics.  Default 0.0
    # preserves Rule #11 (production runs always have combustion).
    _t_combustion_start = float(getattr(outdoor_cfg, "spin_up_s", 0.0))

    # Phase 15I — resolve MD2004 R_p Arrhenius overrides (Rule #1: must be set
    # together to avoid breaking kinetic compensation, Antal-Varhegyi 1998).
    _pyro_A_p = (float(pyrolysis_A_p_override)
                 if pyrolysis_A_p_override is not None
                 else pyrolysis_3d.A_MD2004)
    _pyro_E_p = (float(pyrolysis_E_p_override)
                 if pyrolysis_E_p_override is not None
                 else pyrolysis_3d.E_MD2004)
    if (pyrolysis_A_p_override is None) != (pyrolysis_E_p_override is None):
        raise ValueError(
            "Phase 15I: pyrolysis_A_p_override and pyrolysis_E_p_override "
            "must be set TOGETHER (kinetic-compensation effect — changing A "
            "without E violates the compensation line per Antal-Varhegyi "
            "1998 IECR 37:1267)."
        )

    # Adaptive-dt log throttle.
    _last_adt_log_t = -1.0

    # ── Phase 24 — sprinkler-activation moisture-jump BC ────────────────
    # Resolve None-defaults now that h_bed / Lx / grid are known.  The
    # jump is one-shot: `_moisture_jump_applied` flips True on first hit
    # and the block is skipped forever after.  When disabled the entire
    # block is skipped (bit-exact preservation of every prior case).
    _moisture_jump_applied = False
    if moisture_jump_enable:
        _mj_x_lo = 0.0    if moisture_jump_x_lo_m is None else float(moisture_jump_x_lo_m)
        _mj_x_hi = Lx     if moisture_jump_x_hi_m is None else float(moisture_jump_x_hi_m)
        _mj_z_lo = 0.0    if moisture_jump_z_lo_m is None else float(moisture_jump_z_lo_m)
        _mj_z_hi = h_bed  if moisture_jump_z_hi_m is None else float(moisture_jump_z_hi_m)
        if _mj_x_hi <= _mj_x_lo or _mj_z_hi <= _mj_z_lo:
            raise ValueError(
                f"Phase 24 moisture-jump zone empty or inverted: "
                f"x∈[{_mj_x_lo}, {_mj_x_hi}], z∈[{_mj_z_lo}, {_mj_z_hi}]")
        if moisture_jump_delta_frac <= 0.0:
            raise ValueError(
                f"Phase 24 moisture_jump_delta_frac must be > 0 (got "
                f"{moisture_jump_delta_frac}); a jump BC that removes or "
                f"leaves water alone has no suppression meaning.")
        # Cell-x centres are computed once from grid.dx.  Bed cells only.
        _mj_i_lo = max(i_bed_start, int(np.ceil (_mj_x_lo / grid.dx)))
        _mj_i_hi = min(i_bed_end,   int(np.floor(_mj_x_hi / grid.dx)))
        # Cell-z centres from grid.dz_arr cumulative.
        _mj_z_cell = np.cumsum(grid.dz_arr[:grid.n_z_bed]) - 0.5 * grid.dz_arr[:grid.n_z_bed]
        _mj_kz_mask = (_mj_z_cell >= _mj_z_lo) & (_mj_z_cell <= _mj_z_hi)
        print(f"  [moisture-jump] armed: t_s={moisture_jump_t_s:.3f}s  "
              f"ΔM={moisture_jump_delta_frac:.3f}  "
              f"zone x∈[{_mj_x_lo:.3f}, {_mj_x_hi:.3f}]m "
              f"(cells i={_mj_i_lo}..{_mj_i_hi-1})  "
              f"z∈[{_mj_z_lo:.4f}, {_mj_z_hi:.4f}]m "
              f"(bed layers {int(_mj_kz_mask.sum())}/{grid.n_z_bed})", flush=True)

    while t < max_wall_time_s:
        _step += 1
        # ── Adaptive dt: recompute each step from CURRENT state ──────────
        # Previously dt was computed ONCE at init from cold-flow state,
        # which under-counts CFL once combustion fires (plume buoyancy
        # spikes |u| 3-5×; nu_t spikes 10×+).  Recompute every step:
        #   advective: dt < cfl · min(dx, dy, dz_min) / max(|u|, |v|, |w|, U_floor)
        #   diffusion: dt < 0.25 · dz_min² / nu_t.max()
        # Costs ~3 numpy.abs().max() calls per step — negligible.
        _u_now = max(float(np.abs(state.u).max()),
                     float(np.abs(state.v).max()),
                     float(np.abs(state.w).max()),
                     U_mf, math.sqrt(_G * h_bed * 0.5), 0.1)
        _nu_now = max(float(nu_t.max()), 1.0e-3)
        _dt_adv_now  = cfl_factor * min(grid.dx, grid.dy, _dz_min) / _u_now
        _dt_diff_now = 0.25 * _dz_min * _dz_min / _nu_now
        _dt_new = min(_dt_adv_now, _dt_diff_now, 0.05)
        # Throttle changes — never grow by >1.5× to avoid oscillation,
        # but allow rapid shrinkage during ignition.
        _dt_new = min(_dt_new, dt * 1.5)
        # ── Lower floor: clamp at min_dt_s and continue.  CFL gets
        # violated when this happens, but it's better than the run
        # taking 10× wall time chasing pathological transients at the
        # microsecond scale.  Halt only on actual explosion (NaN/Inf
        # in T_g — checked below).
        _floor_hit = (_dt_new < float(min_dt_s))
        if _floor_hit:
            _dt_new = float(min_dt_s)
        # Halt on actual explosion (NaN/Inf in primary state).  Cheaper
        # than letting the projection wander.
        if not np.isfinite(float(state.T_g.max())):
            raise RuntimeError(
                f"State NaN/Inf at t={t:.3f}s — simulation exploded.  "
                f"(adaptive dt: {_dt_new*1e6:.2f} µs, u_max={_u_now:.2f} "
                f"m/s, nu_t_max={_nu_now:.3e} m²/s)"
            )
        # Throttled log if dt changes meaningfully OR floor hit.
        if (t - _last_adt_log_t >= 0.5 and
                (_dt_new < 0.5 * dt or _dt_new > 1.4 * dt or _floor_hit)):
            _tag = "FLOOR" if _floor_hit else ""
            print(f"  [adaptive-dt t={t:.2f}s {_tag}] dt {dt*1e6:.0f}μs → "
                  f"{_dt_new*1e6:.0f}μs  (u={_u_now:.2f} ν_t={_nu_now:.3e})",
                  flush=True)
            _last_adt_log_t = t
        dt = _dt_new
        # Phase 14ap-2: combustion-gate flag, evaluated each iteration.
        _combustion_active = (t >= _t_combustion_start)
        # ── Phase 24: sprinkler-activation moisture jump (one-shot) ─────
        if (moisture_jump_enable
                and not _moisture_jump_applied
                and t >= moisture_jump_t_s):
            _delta_w = rho_b * moisture_jump_delta_frac       # [kg/m³]
            # Eulerian branch: raise m_water in the (i, k) zone across ALL y.
            # Bed slice is k ∈ [0, n_z_bed); zone i is [_mj_i_lo, _mj_i_hi).
            if not lagrangian_bed_enable and _mj_i_hi > _mj_i_lo:
                for _kk in range(grid.n_z_bed):
                    if not _mj_kz_mask[_kk]:
                        continue
                    m_water[_kk, :, _mj_i_lo:_mj_i_hi] += _delta_w
                # Re-enable moisture bookkeeping if we just added the first
                # water into a dry-bed case (Rule #17-safe: only lifts a
                # False → True flag that gates a strictly-more-work branch).
                if m_water.max() > 0:
                    _has_moisture = True
            # Lagrangian branch: bump water per particle by ρ_b·ΔM ·
            # (per-particle cell-volume share).  Filter by cell (i, k).
            if lagrangian_bed_enable and _bed_buf is not None and _bed_N_alloc > 0:
                lagrangian_bed_3d.apply_moisture_jump_zone(
                    _bed_buf, N=_bed_N_alloc,
                    dx=grid.dx, dy=grid.dy, dz_arr=grid.dz_arr,
                    n_z_bed=grid.n_z_bed,
                    i_lo=_mj_i_lo, i_hi=_mj_i_hi,
                    kz_mask=_mj_kz_mask.astype(np.uint8),
                    delta_water_kg_m3=_delta_w,
                )
            _moisture_jump_applied = True
            _bed_layers_hit = int(_mj_kz_mask.sum())
            _cells_hit = max(0, _mj_i_hi - _mj_i_lo) * grid.Ny * _bed_layers_hit
            print(f"  [moisture-jump] APPLIED at t={t:.3f}s  "
                  f"ΔM={moisture_jump_delta_frac:.3f}  "
                  f"cells={_cells_hit}  Δwater={_delta_w:.3f} kg/m³",
                  flush=True)
        # ── 0. Save ρ at start of step (for ∂ρ/∂t in div_target) ─────────
        # Phase 14u Option 3a: variable-density Chorin needs (ρ_new−ρ_old)/dt
        # in the projection's mass source.  We save rho_prev here BEFORE any
        # T_g/ρ updates this step, so drho_dt captures everything that
        # changes in this step (pilot ignition + chemistry + coupling).
        np.copyto(rho_prev, state.rho)
        # Phase 14aw-2: snapshot α_s alongside ρ_prev so dα_s/dt covers
        # the same window (pyrolysis depletion happens BEFORE projection).
        np.copyto(alpha_s_prev, state.alpha_s)
        # ── Diagnostic AFT pin (opt-in, OFF by default) ─────────────────
        # Hold a hot-gas column above the source patch at kerosene AFT
        # for the ignition window.  Uses np.maximum (no cooling) and a
        # ramp from T_amb to T_pin over ignition_T_pin_ramp_s seconds to
        # avoid the density shock that would arise from an instantaneous
        # T jump (cold→1500K gives 5× density ratio at one cell).
        if ignition_T_pin_enable and t < ignition_duration_s:
            # Phase 18 scout: in_bed mode places pin column INSIDE the bed
            # (k=0..n_z_bed-1) instead of above it.  Default behaviour is
            # unchanged (above-bed pin in z ∈ [h_bed, h_bed + height_m]).
            if ignition_T_pin_in_bed:
                _k_pin_lo = 0
                _k_pin_hi = grid.n_z_bed
            else:
                _k_pin_lo = grid.n_z_bed
                _z_pin_top = h_bed + float(ignition_T_pin_height_m)
                _k_pin_hi = _k_pin_lo
                for _k_p in range(_k_pin_lo, grid.Nz):
                    if grid.z_face[_k_p + 1] > _z_pin_top:
                        break
                    _k_pin_hi = _k_p + 1
            if _k_pin_hi > _k_pin_lo:
                _ramp = float(ignition_T_pin_ramp_s)
                _f_ramp = min(1.0, t / max(_ramp, 1e-6))
                _T_pin_now = T_amb + _f_ramp * (float(ignition_T_pin_K) - T_amb)
                # np.maximum: only RAISE T_g, never cool.  Lets actual
                # combustion exceed the pin without being clamped down.
                _src_slice = state.T_g[_k_pin_lo:_k_pin_hi,
                                        :, i_src_start:i_src_end]
                np.maximum(_src_slice, _T_pin_now, out=_src_slice)
        # ── 1. Pilot ignition REMOVED in Phase 14v ───────────────────────
        # The legacy pilot T_g pin (np.maximum to 700K in source cells for
        # the entire ignition_duration_s = 10s) was an unphysical INFINITE
        # heat reservoir.  Per-cell energy needed to maintain T_g=700K
        # against wind advection: ~440 W; over 10s × 100 source cells:
        # ~440 kJ — about 35× the calibrated drip-torch energy (12.5 kJ).
        # The pin was masking real ignition energy budget — once Phase 14u
        # cleaned the projection, that artificial energy drove ROS over by
        # 30-100×.
        #
        # Replacement: rely on Q_drip (calibrated to Pyne 1993 hardware,
        # 5 kW/m² over 10s) injected via Q_comb in coupling kernel.  This
        # heats T_g via gas energy equation; q_conv heats T_s; once T_s
        # crosses pyrolysis threshold (~600K), pyrolysis releases Y_F;
        # combustion fires when T_g + Y_F + Y_O2 conditions met.  Estimate
        # for Cut bed at U=4: bed reaches 600K in ~1s under Q_drip.
        # ── 1b. EoS update (NEW: before momentum/projection) ─────────────
        # Phase 14u Option 3a: ρ must reflect new T_g (from pilot ignition,
        # and any other rapid T changes at start of step) BEFORE the
        # momentum/projection see it.  Otherwise the projection enforces
        # ∇·u=0 against stale ρ — mass conservation breaks for big T jumps.
        # The chemistry/coupling T_g changes happen LATER in the step;
        # they get captured by NEXT step's drho_dt term (one-step lag,
        # acceptable for slow combustion changes).
        np.copyto(state.rho, _P0 / (_R_AIR * np.maximum(state.T_g, T_amb)))

        # Inlet BC for Y_O2: fresh air flowing in at x=0.  Without this
        # the upwind transport scheme would let combustion-vitiated air
        # backflow into the inlet column, breaking mass conservation of
        # the open boundary.  Cup burner supplies inlet composition via
        # the z-min inlet ghost instead — skip the x-inlet hardcode.
        if not _is_cup_burner:
            state.Y_O2[:, :, 0] = 0.232

        # Phase 23 Refactor 2D: per-step BC hook.  Outdoor BC is a no-op
        # (bit-exact-invariant); cup burner with wick_enable applies a
        # dirichlet fuel source in the wick region each step.
        if _is_cup_burner:
            _bc.apply_per_step(state, grid, t)

        # ── 2/3/4. Pyrolysis + combustion + species (PRE-substepped) ─────
        # Phase 14ap-2 spin-up: pyrolysis chain is the upstream gate for
        # combustion (no S_pyro → no Y_fuel → no chemistry → no Q_comb).
        # Gating it here cleanly disables the full combustion cascade
        # without touching downstream kernels (they no-op on zero source).
        if _combustion_active:
            if lagrangian_bed_enable and _bed_buf is not None:
                # ── Phase 16 — Lagrangian sub-grid bed particles ──
                # Replaces drying + pyrolysis + char_ox + smolder Eulerian
                # kernels with one per-particle pass that emits sources
                # to gas cells.  Same sign convention for Q_pyro as the
                # Eulerian path (positive = solid heat sink).
                # Build Q_solid_ext = drip-torch + radiation absorbed by
                # bed cells (q_rad + q_frankman).  During ignition window,
                # the 80% drip-torch-to-solid split feeds particles
                # directly so they can heat to pyrolysis onset.  Radiation
                # is the forward-spread mechanism: DOM emits from hot
                # bed (T_s mirror) and bed cells ahead of the front
                # absorb q_rad → particles heat → pyrolysis → Y_F →
                # combustion → level-set advance.
                _bed_Qsx = np.zeros(_bp_shape, dtype=np.float64)
                if t < ignition_duration_s:
                    Q_DRIP_PER_AREA_BED = 30_000.0  # W/m² — same as Eulerian
                    _Q_drip_bed_vol = Q_DRIP_PER_AREA_BED / max(h_bed, 1e-3)
                    F_DRIP_TO_SOLID_BED = 0.80
                    _bed_Qsx[:grid.n_z_bed, :, i_src_start:i_src_end] += (
                        F_DRIP_TO_SOLID_BED * _Q_drip_bed_vol)
                # Add radiation + Frankman forward-convective flux from
                # the previous step (q_rad and q_frankman_3d were
                # computed late in the prior outer step but live in
                # arrays we can read at the top of this step).
                # NOTE: q_rad and q_frankman_3d are W/m² (per cell
                # SURFACE area); _bed_Qsx is W/m³ (volumetric).  Convert
                # per-cell via division by dz_arr[k] — matches the
                # coupling kernel's q_rad_volumetric = q_rad/dz_arr[k]
                # at coupling_3d.py line 154.
                # v38: q_rad routing with aggressive clamp matching drip-
                # torch magnitude (~6e4 W/m³).  This is the rate the
                # particles can absorb without coupling-rate instability;
                # capping prevents the step-2 spike when q_rad reaches
                # 2.4e7 W/m³ at the bed top.
                Q_RAD_MAX = 1.0e5    # W/m³
                for _k_bed in range(grid.n_z_bed):
                    _v = q_rad[_k_bed] / grid.dz_arr[_k_bed]
                    _v = np.clip(_v, -Q_RAD_MAX, Q_RAD_MAX)
                    _bed_Qsx[_k_bed] += _v
                _t0 = _time.perf_counter()
                lagrangian_bed_3d.step_bed_particles(
                    _bed_buf["x"], _bed_buf["y"], _bed_buf["z"],
                    _bed_buf["alive"],
                    _bed_buf["m_solid"], _bed_buf["m_water"],
                    _bed_buf["m_char"], _bed_buf["T_s"],
                    _bed_buf["m_water_0"], _bed_buf["sav"],
                    state.T_g, state.Y_O2,
                    _bed_Qsx, int(lagrangian_bed_N_per_cell),
                    _bed_Sp, _bed_Sd, _bed_Qp, _bed_Qd, _bed_YFs,
                    _bed_Qch, _bed_Qsm, _bed_Qgc,
                    grid.dx, grid.dy, grid.dz_arr, grid.z_face,
                    float(lagrangian_bed_h_conv),
                    float(lagrangian_bed_rho_solid_true),
                    float(lagrangian_bed_cp_solid),
                    float(lagrangian_bed_eps_solid),
                    float(T_amb),
                    float(lagrangian_bed_view_factor),
                    bool(lagrangian_bed_view_factor_geometric),
                    float(h_bed),                          # bed top z (m)
                    # Effective bed absorption coefficient κ ≈ sav · α_s
                    # where α_s = ρ_b / ρ_solid_true (volume fraction of solid
                    # in cell-average).  For grass: 1.07 / 380 ≈ 0.0028,
                    # sav=2000/m → κ ≈ 5.6 1/m; κ·h_bed ≈ 2.1 (optically
                    # thick), so deepest particles emit ~exp(-2.1) ≈ 12%
                    # of surface particles.
                    float(_bed_sav * rho_b
                          / float(lagrangian_bed_rho_solid_true)),
                    dt,
                    bool(lagrangian_bed_do_drying),
                    bool(lagrangian_bed_do_pyrolysis),
                    bool(lagrangian_bed_do_char_ox),
                    bool(lagrangian_bed_do_smolder),
                    _bed_drying_mode_int,
                    # Phase 20 char-ox knobs
                    float(char_ox_flux_cap_W_m2),
                    float(char_ox_ash_exp),
                    _bed_buf["m_char_max"],
                    _bed_n_alive_diag, _bed_n_burned_diag,
                    _bed_diag_max,
                )
                _tick("lagrangian_bed", _t0)
                # S_pyro = pyrolysis volatile + drying water vapor
                np.copyto(S_pyro, _bed_Sp)
                S_pyro += _bed_Sd
                # Q_pyro net solid heat sink — match Eulerian sign convention:
                # endo pyrolysis (+) + endo drying (+) − exo char_ox (−) − exo smolder (−)
                np.copyto(Q_pyro, _bed_Qp)
                Q_pyro += _bed_Qd
                Q_pyro -= _bed_Qch
                Q_pyro -= _bed_Qsm
                # T_s mirror — fast @njit kernel scatters particle T_s
                # into per-cell mass-weighted average.  Downstream DOM
                # radiation kernel uses state.T_s for K_emit·σ·T_s⁴; we
                # MUST keep it in sync or the bed never radiates forward
                # and the fire stalls at the ignition patch.
                lagrangian_bed_3d.aggregate_particles_to_T_s_grid(
                    _bed_buf["x"], _bed_buf["y"], _bed_buf["z"],
                    _bed_buf["alive"],
                    _bed_buf["m_solid"], _bed_buf["m_water"],
                    _bed_buf["m_char"], _bed_buf["T_s"],
                    grid.dx, grid.dy, grid.z_face,
                    state.T_s, T_amb,
                )
                # Horizontal solid conduction is RE-enabled.  Works in
                # isolation (v9); only the q_rad routing crashes.
                lagrangian_bed_3d.step_horizontal_solid_conduction_scatter(
                    _bed_buf["x"], _bed_buf["y"], _bed_buf["z"],
                    _bed_buf["alive"],
                    _bed_buf["m_solid"], _bed_buf["m_water"],
                    _bed_buf["m_char"], _bed_buf["T_s"],
                    state.T_s, state.alpha_s,
                    grid.dx, grid.dy, grid.z_face,
                    solid_conduction_3d.K_SOLID_GRASS,
                    float(lagrangian_bed_rho_solid_true),
                    float(lagrangian_bed_cp_solid),
                    grid.n_z_bed, dt,
                )
            else:
                _t0 = _time.perf_counter()
                if _has_moisture:
                    pyrolysis_3d.step_drying(
                        state.T_s, m_water, state.alpha_s, dt, _Q_dry,
                    )
                else:
                    _Q_dry.fill(0.0)
                _tick("pyrolysis:drying", _t0)
                _t0 = _time.perf_counter()
                pyrolysis_3d.step_pyrolysis_md2004(
                    state.T_s, state.m_hemi, m_initial, state.m_char,
                    state.Y_O2, state.alpha_s,
                    m_water, m_water_init if _has_moisture else 0.0,
                    dt,
                    S_pyro, Q_pyro,
                    _pyro_A_p, _pyro_E_p,
                )
                _tick("pyrolysis:md2004", _t0)
                Q_pyro += _Q_dry
                _t0 = _time.perf_counter()
                pyrolysis_3d.step_char_oxidation(
                    state.T_s, state.m_char, state.Y_O2, state.alpha_s,
                    dt, _Q_char,
                )
                _tick("pyrolysis:char_ox", _t0)
                Q_pyro -= _Q_char
                _t0 = _time.perf_counter()
                pyrolysis_3d.step_smoldering_oxidation(
                    state.T_s, state.m_char, state.Y_O2, state.alpha_s,
                    dt, _Q_smold,
                )
                _tick("pyrolysis:smolder", _t0)
                Q_pyro -= _Q_smold
        else:
            # Cold-flow spin-up: zero out sources so downstream kernels
            # see no fuel and no heating.
            S_pyro.fill(0.0)
            Q_pyro.fill(0.0)
        # ── 5. Drag (porous-media, fuel bed only) ────────────────────────
        _t0 = _time.perf_counter()
        drag_3d.step_drag_force(
            state.u, state.v, state.w, state.rho, state.alpha_s,
            sigma_sav, Fx, Fy, Fz,
            _canopy_C_d,
        )
        _tick("drag", _t0)
        # ── 5b. Phase 14ap (re-added): SEM inlet turbulence ──────────────
        # Builds (Nz, Ny) y-asymmetric perturbation by summing N tent-shaped
        # eddies, then writes u_inlet = u_inlet_base + du.  Momentum kernel
        # below picks up the perturbed ghost value at i=0 via Way B.  Eddies
        # advect at U_c each step and recycle when xk > σ.
        if _sem_enable:
            _t0 = _time.perf_counter()
            _du_sem = np.zeros((grid.Nz, grid.Ny), dtype=np.float64)
            _dv_sem = np.zeros((grid.Nz, grid.Ny), dtype=np.float64)
            _dw_sem = np.zeros((grid.Nz, grid.Ny), dtype=np.float64)
            for _ek in range(_sem_N):
                _fx = _sem_tent(_sem_xk[_ek] / _sem_sigma)
                if _fx == 0.0:
                    continue
                _dy = grid.y_mid - _sem_yk[_ek]
                _dy_p = _dy - grid.Ly * np.round(_dy / grid.Ly)
                _fy = _sem_tent(_dy_p / _sem_sigma)
                _fz = _sem_tent((grid.z_mid - _sem_zk[_ek]) / _sem_sigma)
                _shape = _fz[:, None] * _fy[None, :] * _fx       # (Nz, Ny)
                _du_sem += _sem_eps[_ek, 0] * _shape
                _dv_sem += _sem_eps[_ek, 1] * _shape
                _dw_sem += _sem_eps[_ek, 2] * _shape
            _du_sem *= _sem_inv_sqN * _sem_amp
            _dv_sem *= _sem_inv_sqN * _sem_amp
            _dw_sem *= _sem_inv_sqN * _sem_amp
            np.add(_u_inlet_base, _du_sem, out=u_inlet)
            np.copyto(v_inlet, _dv_sem)
            np.copyto(w_inlet, _dw_sem)
            # Advect eddies + recycle those that exited the σ box
            _sem_xk += _sem_Uc * dt
            _ex = _sem_xk > _sem_sigma
            if np.any(_ex):
                _n_ex = int(_ex.sum())
                _sem_xk[_ex] = -_sem_sigma
                _sem_yk[_ex] = _sem_rng.uniform(0.0, grid.Ly, size=_n_ex)
                _sem_zk[_ex] = _sem_rng.uniform(
                    -_sem_sigma, _Lz_total + _sem_sigma, size=_n_ex)
                _sem_eps[_ex] = _sem_rng.choice([-1.0, 1.0], size=(_n_ex, 3))
            _tick("sem:inlet_perturb", _t0)
        # ── 6. Tentative momentum (advect + diffuse + buoyancy + drag) ───
        _t0 = _time.perf_counter()
        momentum_3d.step_tentative_velocity(
            state.u, state.v, state.w, state.rho, state.T_g,
            Fx, Fy, Fz, dt, grid.dx, grid.dy,
            grid.dz_arr, grid.d_face_above, grid.d_face_below,
            T_amb, u_inlet, v_inlet, w_inlet,
        )
        _tick("momentum:tentative", _t0)
        # ── 7. Boundary conditions (before projection) ───────────────────
        _t0 = _time.perf_counter()
        _apply_velocity_bcs(state, u_inlet, y_bc, wall_function=wall_function)
        _tick("bcs:velocity", _t0)
        # ── 8. Pressure projection ───────────────────────────────────────
        # Low-Mach mass-source target: ∇·u = S_pyro / ρ.  Without this,
        # the strict-incompressible projection (∇·u = 0) silently
        # discards the gas mass added by pyrolysis — fuel accumulates
        # locally instead of expanding outward.  See projection_3d.py
        # docstring for derivation; FDS Tech Ref §3.2 (McDermott 2011).
        # S_pyro is integrated over dt in the species transport, so
        # the rate-form S_pyro/ρ here is the instantaneous divergence
        # source.  ρ is bounded below by ρ at T_amb (rho_amb ~ 1.2)
        # so no divide-by-zero risk.
        # Phase 14u Option 3a: variable-density continuity.
        # ∂ρ/∂t + ∇·(ρu) = S_mass  ⇒  ∇·u ≈ (S_mass − ∂ρ/∂t)/ρ
        # The ∂ρ/∂t term captures gas expansion from heating (e.g. pilot
        # ignition raising T_g 303→700K → ρ drops 1.17→0.50 → expansion
        # of ~+200 1/s in the source cells, which the projection now
        # correctly drives via Dirichlet pressure outflow at top/outlet).
        # Without this term, sudden T jumps caused NaN in cold-bed-fire
        # cases after Phase 14u clean projection (which had previously been
        # masked by div_residual ~0.016 acting as numerical damping).
        drho_dt = (state.rho - rho_prev) / dt
        if _volume_weighted_projection:
            # Phase 14aw-2: volume-weighted target for the ∇·(α_g·u)
            # divergence operator.  Derivation from mass conservation
            # ∂(α_g·ρ)/∂t + ∇·(α_g·ρ·u) = S_mass under low-Mach
            # (drop u·∇ρ):
            #   ρ·∂α_g/∂t + α_g·∂ρ/∂t + ρ·∇·(α_g·u) = S_mass
            #   ⟹ ∇·(α_g·u) = S_mass/ρ - ∂α_g/∂t - (α_g/ρ)·∂ρ/∂t
            # ∂α_g/∂t = -∂α_s/∂t (since α_g = 1 - α_s).
            alpha_g = 1.0 - state.alpha_s
            dalpha_g_dt = -(state.alpha_s - alpha_s_prev) / dt
            rho_safe = np.maximum(state.rho, 0.1)
            div_target = (S_pyro / rho_safe
                          - dalpha_g_dt
                          - (alpha_g / rho_safe) * drho_dt)
        else:
            # Pre-14aw target (matches the unweighted ∇·u operator).
            div_target = (S_pyro - drho_dt) / np.maximum(state.rho, 0.1)
        # Phase 14u: FDS-style "pressure iteration" (FDS Tech Ref §6.3).
        # Loop runs until residual max|∇·u − div_target| < proj_div_tol
        # (deck-configurable threshold; this is the breakout) or until
        # proj_max_iter is reached (deck-configurable safety cap).  The
        # BCs (zero-grad outflow, no-slip wall, periodic-y, Dirichlet
        # inlet) are NOT consistent with the projection's discrete
        # operator, so a single projection leaves boundary divergence
        # that propagates inward via advection.  Iterating drives ‖div‖_∞
        # below tolerance regardless of which specific BC is breaking it.
        # FDS default tolerance: 1e-3.
        proj_n_iter = 0
        proj_div_max = 0.0
        # Phase 14u-opt: rebuild matrix ONCE per outer step (ρ doesn't
        # change between iters).
        _t0 = _time.perf_counter()
        # Phase 14aw-2: set α_g on the solver each step.  When the flag
        # is off, set_alpha_g(None) keeps the solver in its baseline
        # ∇·((1/ρ)·∇p) operator + plain ∇·u divergence (pre-14aw).
        if _volume_weighted_projection:
            proj_solver.set_alpha_g(1.0 - state.alpha_s)
        else:
            proj_solver.set_alpha_g(None)
        proj_solver.rebuild_for_rho(state.rho)
        _tick("projection:rebuild", _t0)
        for _proj_iter in range(proj_max_iter):
            _t0 = _time.perf_counter()
            proj_solver.project(state.u, state.v, state.w, state.rho, dt,
                                div_target=div_target)
            _tick("projection:solve", _t0)
            # ── 9. BCs (after projection) ────────────────────────────
            _t0 = _time.perf_counter()
            _apply_velocity_bcs(state, u_inlet, y_bc, wall_function=wall_function)
            _tick("bcs:velocity", _t0)
            # Convergence check: ‖∇·u − div_target‖_∞ < tol
            _t0 = _time.perf_counter()
            div_now = proj_solver.divergence(state.u, state.v, state.w)
            _tick("projection:divergence", _t0)
            if div_target is not None:
                div_now = div_now - div_target
            proj_div_max = float(np.max(np.abs(div_now)))
            proj_n_iter = _proj_iter + 1
            if proj_div_max < proj_div_tol:
                break
        # Phase 14y: outflow sponge — damp u_x toward inlet log-law profile
        # in the last N_SPONGE cells near x=Lx.  Suppresses backflow at the
        # open outflow boundary that the Dirichlet-pressure (p=0) BC otherwise
        # admits via buoyancy-driven entrainment.  Flame-aware: skips cells
        # with Y_F > Y_F_SPONGE_SKIP so it doesn't damp the active flame
        # plume when the front reaches the outlet zone (Cheney sweep at
        # Lx=10: front routinely advances into the sponge region).
        _t0 = _time.perf_counter()
        momentum_3d.apply_outflow_sponge(
            state.u, u_inlet, _sigma_x_sponge,
            state.Y_fuel, _Y_F_SPONGE_SKIP, dt,
        )
        _tick("momentum:sponge", _t0)
        # ── 10. Burning mask (T_s ≥ T_ign AND has fuel) ──────────────────
        np.copyto(burning_mask,
                  (state.T_s >= T_ign).astype(np.float64) *
                  (state.alpha_s > 0.0).astype(np.float64))
        # ── 10b. k-ε turbulence (Phase 14b) ──────────────────────────────
        # Update k, ε, ν_t fields based on current u, v, w, T_g.  ν_t feeds
        # τ_mix = k/ε in combustion (replacing the buoyancy-fallback τ_mix)
        # and adds turbulent diffusion to species (lifts pyrolysis volatiles
        # to buffer cells, enabling proper flame-body development).
        # Phase 14v-bc Way B: log-law wall function fills GHOST arrays
        # (k_wall_ghost, eps_wall_ghost) that step_k_epsilon reads at its
        # k=0 stencil ghost slot.  No real cells are written.  Reads cell-1
        # u, v from PREVIOUS step's projection (acceptable explicit lag).
        # Reference: Launder & Spalding (1974) Comp. Methods Appl. Mech. Eng. 3:269.
        if wall_function:
            turbulence_3d.apply_wall_function(
                state.u, state.v, state.rho, state.alpha_s, grid.dz_arr,
                k_wall_ghost, eps_wall_ghost,
            )
        _t0 = _time.perf_counter()
        if turbulence_model == "smagorinsky":
            turbulence_3d.step_smagorinsky_les(
                state.u, state.v, state.w,
                grid.dx, grid.dy, grid.dz_arr,
                k_turb, eps_turb, nu_t, _S_mag2_work,
            )
            if nu_t_multiplier != 1.0:
                nu_t *= nu_t_multiplier
            _tick("turbulence:smagorinsky", _t0)
        else:
            # Phase 15E: realizability cap on ε (Durbin 1996 / Shih 1995).
            # Caps ε at k^1.5/L_canopy so the implied length scale doesn't
            # drop below the canopy depth (mesh-induced shear-layer spike
            # diagnosed at v5 E_fsd).  Caller controls via eps_realiz_L_min_m;
            # 0.0 keeps Phase 14 behaviour unchanged.
            turbulence_3d.step_k_epsilon(
                k_turb, eps_turb, nu_t,
                state.u, state.v, state.w, state.T_g,
                state.rho,         # Phase 14ai: rho for BVG buoyancy term
                state.alpha_s,
                sigma_sav, dt, grid.dx, grid.dy,
                grid.dz_arr, grid.d_face_above, grid.d_face_below,
                T_amb,
                _S_mag2_work, _Omega_mag2_work,
                u_inlet,
                k_wall_ghost, eps_wall_ghost,
                _canopy_beta_p, _canopy_beta_d,
                0.0,                                 # bvg_factor (unchanged)
                eps_realiz_L_min_m,                  # Phase 15E L-cap
                eps_realiz_durbin_alpha,             # Phase 15E-B Durbin α
                _eps_cap_count,                      # per-k L-cap counter
                _eps_durbin_count,                   # per-k Durbin counter
            )
            if nu_t_multiplier != 1.0:
                nu_t *= nu_t_multiplier
            _tick("turbulence:k_epsilon", _t0)
            # Phase 15E: log when either cap was active.  Per-outer-step
            # rollup; avoids per-substep spam.
            if (eps_realiz_L_min_m > 0.0 or
                    eps_realiz_durbin_alpha > 0.0):
                _n_l = int(_eps_cap_count.sum())
                _n_d = int(_eps_durbin_count.sum())
                _ntot = grid.Nz * grid.Ny * grid.Nx
                if (_n_l + _n_d > 0
                        and (t - _eps_cap_last_log_t) >= 1.0):
                    print(f"  [eps-cap] t={t:.2f}s  L-cap "
                          f"{_n_l} ({_n_l/_ntot*100:.2f}%)  "
                          f"Durbin {_n_d} ({_n_d/_ntot*100:.2f}%)",
                          flush=True)
                    _eps_cap_last_log_t = t
                _eps_cap_count[:] = 0
                _eps_durbin_count[:] = 0
        # τ_mix = k/ε (turbulent eddy timescale).  Phase 14w-H: removed
        # upper bound (was 1.0 s) — non-standard vs M&H 1977 / FDS /
        # FireFOAM, which let τ_mix grow unbounded in laminar zones so
        # ω_EBU naturally → 0 and chemistry/Arrhenius takes over there
        # (chemistry rate is the right limiter when turbulence is absent).
        # Lower clamp at the buoyancy fallback τ = √(2δ/g) so ω_EBU has
        # a physical ceiling = ρ·Y_lim/τ_buoy ≈ 7·ρ·Y_lim — prevents
        # unphysically large 1/τ when k-ε is still spinning up.
        np.copyto(tau_mix, np.maximum(k_turb / np.maximum(eps_turb, 1e-12),
                                       _tau_mix_buoy))
        # ── 11. P1 (Eddington) radiation (Phase 14a) ─────────────────────
        # Replaces Albini 1985 phenomenological slab + Frankman 2013 contact.
        # Each cell radiates from its own state (T_s + T_g blended by κ),
        # solved via elliptic ∇·(D∇G) − κG = −4πκB with sparse-LU.
        # References: Modest (2003) §15; Morvan-Dupuy (2004); Larini-Porterie (1998).
        # Phase 14v-bc-soil: pass T_soil[0] as ground BC; capture absorbed
        # incident at z=0 to drive the soil 1D conduction sub-step below.
        # Only DOM supports T_soil for now; P1 path uses default T_amb BC.
        if radiation_solver == "dom":
            # Phase 14ah-3: DOM sub-cycling.  Radiation field changes
            # slowly compared to advection — calling DOM every K hydro
            # steps and reusing the cached q_rad / q_rad_gas / q_in_soil
            # arrays between calls is standard (FDS Tech Ref Vol.1 §6.2;
            # Howell et al. 2010 "Thermal Radiation Heat Transfer" §17).
            # At K=5 the radiation share of loop drops 25% → ~5%.  Set
            # dom_subcycle_every=1 to disable (every-step behavior).
            if (_step % max(dom_subcycle_every, 1)) == 0:
                # Phase 17a — wet-bed κ_solid scaling (Mell 2007 WFDS /
                # Linn 2002 FIRETEC): aggregate per-particle m_water and
                # m_solid into per-cell M_local for DOM.  Bed-particle
                # path only; Eulerian path stays at M=0 (legacy).
                if lagrangian_bed_enable and _bed_buf is not None \
                        and _bed_M_local is not None:
                    lagrangian_bed_3d.aggregate_particles_to_M_local_grid(
                        _bed_buf["x"], _bed_buf["y"], _bed_buf["z"],
                        _bed_buf["alive"],
                        _bed_buf["m_solid"], _bed_buf["m_water"],
                        grid.dx, grid.dy, grid.z_face,
                        _bed_M_local,
                    )
                _t0 = _time.perf_counter()
                rad_solver.solve(
                    state.T_s, state.T_g, state.alpha_s, omega,
                    sigma_sav, T_amb,
                    q_rad_solid_out=q_rad, q_rad_gas_out=q_rad_gas,
                    T_soil_surface=T_soil[0],
                    q_in_soil_out=q_in_soil,
                    Y_H2O=state.Y_H2O, rho=state.rho,
                    bed_moisture_per_cell=_bed_M_local,
                )
                _tick("radiation:dom", _t0)
            # Soil 1D conduction step (Pimont 2006 / FIRESTAR pattern).
            # Surface BC: q_in_soil − ε σ T_soil[0]⁴ ; bottom: T = T_amb.
            _t0 = _time.perf_counter()
            soil_3d.step_soil_conduction(
                T_soil, q_in_soil, dt,
                soil_dz, soil_d_above, soil_d_below,
                alpha_s=soil_3d.K_SOIL_DEFAULT
                        / (soil_3d.RHO_SOIL_DEFAULT * soil_3d.CP_SOIL_DEFAULT),
                k_s=soil_3d.K_SOIL_DEFAULT,
                rho_s=soil_3d.RHO_SOIL_DEFAULT,
                cp_s=soil_3d.CP_SOIL_DEFAULT,
                eps_s=soil_3d.EPS_SOIL_DEFAULT,
                T_amb=T_amb,
            )
            _tick("soil_conduction", _t0)
        else:
            _t0 = _time.perf_counter()
            rad_solver.solve(
                state.T_s, state.T_g, state.alpha_s, omega,
                sigma_sav, T_amb,
                q_rad_solid_out=q_rad, q_rad_gas_out=q_rad_gas,
            )
            _tick("radiation:p1", _t0)
        # ── 11.5. External ignition-pulse flux on source cells ───────────
        # Quintiere (2006) §7.4: piloted ignition uses a fixed external
        # heat flux on the surface, not a T_s clamp.  Apply 40 kW/m² to
        # the top bed layer of source cells during the ignition pulse;
        # the coupling step then computes dT_s from the full energy
        # balance (q_rad + q_conv - q_loss - Q_pyro).  This matches the
        # 1D ROM's deck.q_in_constant approach and avoids the artificial
        # T_s = T_ign clamping that drove pyrolysis runaway in 13.G.
        if t < ignition_duration_s:
            k_top_bed = grid.n_z_bed - 1
            q_rad[k_top_bed, :, i_src_start:i_src_end] += (
                240_000.0 * float(ignition_q_mult)
            )  # [W/m²]  — Phase 15L kick intensity
            # 240 kW/m² for 5s = 1200 kJ/m² total (same integrated energy as the
            # prior 40 kW/m² × 30s setting, but with 6× peak intensity to
            # overcome advective washout in sparse beds; Pyne 1993 hardware-
            # consistent peak ~119 kW/m²).
        # ── 11.6. Phase 14y: dual level-set masks ────────────────────────
        # phi_bed (lset): kinematic level-set tracking bed-pyrolysis front.
        #     ahead_bed_band: 0 < phi_bed ≤ _band_m_tilt — heat-flux DEST
        #         (cells just ahead of bed-front, receive Frankman + DOM fwd).
        #
        # phi_flame: state-derived signed distance to active gas-phase flame.
        #     flame_body_mask = phi_flame ≤ 0 — Frankman SOURCE
        #         (where active flame T_g lives, regardless of phi_bed).
        #
        # The two are decoupled: bed can pyrolyze without flame; flame can
        # exist over already-pyrolyzed bed (advected plume).  Pre-Phase 14y
        # the SAME mask served both — broke at high wind where plume tilts.
        # Phase 15C: initialise phi_flame so cold-flow path (n_z_bed=0) can
        # still pass it through chemistry_closures.run without NameError.
        # The level_set_fsd closure asserts on None; other closures ignore.
        phi_flame = None
        if grid.n_z_bed > 0:
            _t0 = _time.perf_counter()
            ahead_band_mask = lset.ahead_band_mask(band_m=_band_m_tilt)
            # Phase 15F bugfix: phi_bed is z-uniform (enforce_z_uniformity
            # is a numerical fix for Sussman drift on a conceptually-2D
            # quantity).  The 3D ahead_band_mask is therefore True in
            # atmospheric cells above the bed too.  Subsequent column
            # sums in compute_q_in_at_front double-count atmosphere
            # absorption as if it heated the bed, giving mesh-runaway
            # at fine meshes (more atm cells per column = more sum
            # contributions).  Mask out everything above bed-top here
            # so q_in is a bed-only integral, mesh-independent.
            ahead_band_mask[grid.n_z_bed:] = False
            phi_flame = flame_front_3d.compute_phi_flame_from_state(
                omega, state.T_g, state.Y_fuel,
                grid.dx, grid.dy, grid.dz_arr,
            )
            flame_body_mask = flame_front_3d.flame_body_mask_from_phi_flame(
                phi_flame, band_m=0.0,
            )
            _tick("level_set:flame_mask", _t0)
            # Frankman 2013 flame-tip convection — DISABLED 2026-05-12.
            # Phenomenological non-local heat shortcut (per-y-strip max T_g
            # applied to ahead_band cells) that double-counts the natural
            # gas-solid coupling.  Kernel kept in flame_front_3d.py for
            # reference / future re-enable; here we just zero the output
            # so v_n is driven only by DOM forward intensity at the band.
            # flame_front_3d.step_frankman_flame_tip(
            #     state.T_g, state.T_s, flame_body_mask, ahead_band_mask,
            #     state.alpha_s, sigma_sav, grid.dz_arr,
            #     h_flame=flame_front_3d.H_FLAME_FRANKMAN,
            #     q_frankman_out=q_frankman_3d,
            # )
            q_frankman_3d.fill(0.0)
            # DOM forward-pointing intensity flux at ahead-band bed cells.
            _t0 = _time.perf_counter()
            if radiation_solver == "dom":
                flame_front_3d.compute_q_dom_fwd_at_band(
                    rad_solver, ahead_band_mask, q_dom_fwd_3d,
                )
            else:
                q_dom_fwd_3d.fill(0.0)
            _tick("level_set:q_dom_fwd", _t0)
            # Phase 15N — Finney burst-convective preheat.  Compute the (Ny,Nx)
            # surface flux here so we can ALSO add it to the bed-top q_rad
            # before the gas-solid coupling step.  Without that, the level-set
            # v_n driver got the burst flux but T_s did not, so the level-set
            # ran ahead of the chemistry it was supposed to be tracking
            # (verified Phase 15N Step 3, 2026-06-08: front degraded -50%
            # post-pulse because the kinematic-only routing produced a ghost
            # front).  Now: q_burst_conv_2d is added to q_rad at top-bed
            # ahead-band cells, so it physically heats T_s and drives
            # pyrolysis there, matching Finney 2015's measured
            # solid-phase heating mechanism.
            q_burst_conv_2d = None
            if finney_burst_enable and phi_flame is not None:
                I_fire_per_y = finney_burst_3d.compute_I_fire_per_y(
                    omega, grid.dx, grid.dz_arr,
                )
                q_burst_conv_2d = finney_burst_3d.compute_finney_burst_q_at_band(
                    phi_flame, ahead_band_mask, grid.dx, grid.x_mid,
                    I_fire_per_y=I_fire_per_y,
                    q_0=(finney_q_0 if finney_q_0 is not None
                         else finney_burst_3d.Q_0_DEFAULT),
                    L_burst=(finney_L_burst if finney_L_burst is not None
                             else finney_burst_3d.L_BURST_DEFAULT),
                    I_thresh=(finney_I_thresh if finney_I_thresh is not None
                              else finney_burst_3d.I_FIRE_THRESH),
                )
                # Add to q_rad at the top-bed cell of every (j,i) so the
                # gas-solid coupling step heats T_s there.  q_rad's z-index
                # for the bed top is n_z_bed - 1.
                k_top_bed = grid.n_z_bed - 1
                q_rad[k_top_bed, :, :] += q_burst_conv_2d
        else:
            # Cold-flow case (no bed): skip level-set physics
            flame_body_mask = None
            ahead_band_mask = None
            q_frankman_3d.fill(0.0)
            q_dom_fwd_3d.fill(0.0)
            q_burst_conv_2d = None
        # ── 11.7. Drip-torch heat injection on source bed cells ──────────
        # Cheney (1993) §2 used continuous drip-torch line ignition
        # (~138 kW/torch from kerosene, Drysdale 2011 Table 11.5).  Inject
        # this as a heat source in Q_comb (added to bed cells of the
        # source strip during ignition_duration).  Without this real
        # flame body P1 has no luminous source to broadcast from.
        # Scaled by ρ_b/ρ_b_ref so heating per unit fuel mass is constant
        # across bed types (Nat: ~190 kW/m³, Cut: ~520 kW/m³).
        # See memory/phase13w_finding.md for derivation.
        # NOTE: applied below in §12 sub-step loop where Q_comb lives.
        # ── 12a. Gas-temperature advection (upwind) ──────────────────────
        # Coupling kernel only applies point-wise Q_comb / q_conv to T_g;
        # gas-phase heat must convect downstream separately.
        _t0 = _time.perf_counter()
        _adv_gas_energy(state.T_g, state.u, state.v, state.w, dt,
                        grid.dx, grid.dy,
                        grid.dz_arr, grid.d_face_above, grid.d_face_below,
                        alpha_th=2.0e-5,
                        T_amb=T_amb,
                        # Phase 23 Refactor 2D: route the BC's z-min T ghost.
                        T_inlet_zmin=(_bc.T_inlet_zmin if _is_cup_burner else None),
                        z_min_inlet_active=_is_cup_burner)
        _tick("gas_energy:advection", _t0)
        # ── 12b. Sub-stepped chemistry+coupling (Strang split) ──────────
        # The combustion source term is stiff: at hot T_g + non-zero
        # Y_fuel, omega = ρ·k_chem·Y·Y_O2 produces Q_comb in 100s MW/m³,
        # which would drive T_g 100s K per main-step.  Sub-step the
        # combustion ↔ Y_fuel ↔ T_g ↔ T_s coupling at dt_sub = dt/N_sub
        # to bound per-iter T changes.  Pyrolysis (step 2) is held
        # constant during the sub-loop (S_pyro is rate, integrated over dt).
        # Reference: Strang (1968) SIAM J. Numer. Anal. 5:506 (operator
        # splitting); standard for stiff reaction-diffusion coupling.
        # Hard numerical safety cap at 10,000 K — matches the bed-particle
        # T_s cap convention.  Old value of 1900K (Drysdale 2011 grass
        # adiabatic) was acting as a PHYSICAL CLIP that erased moisture-
        # sensitive flame T_g differences: at peak burn, M=4% reaches
        # higher T_g than M=8%, but both got pinned at 1900K → identical
        # forward radiation → identical ROS.  Raising to 10K K lets the
        # chemistry/coupling/drying energy balance set the physical T_g
        # naturally; the cap only catches runaway from numerical bugs.
        T_FLAME_AD = 10_000.0   # numerical safety only
        T_SURF_MAX = 10_000.0   # numerical safety only (was 900K Frandsen
                                # 1971 grass char plateau — also a physical
                                # clip, erasing T_s response)
        N_SUB = 10
        dt_sub = dt / N_SUB
        # Phase 14ah-4 REVERTED 2026-05-17: substep early-exit removed.
        # Original optimization broke chemistry ignition: the
        # Y_F-convergence check fired DURING the slow pre-ignition phase
        # where Y_F accumulates quietly before runaway.  Breaking early
        # there prevents the runaway from ever happening — T_g_max stays
        # at ~330 K (vs Drysdale 1500–1800 K when the loop runs all 10
        # substeps) and chemistry self-extinguishes.  Diagnostic Phase
        # 14ak (2026-05-17): Nat 4% U=4 ratio recovered 0.36 → 0.45
        # when this exit was disabled.  Net perf cost: ~3× more inner-
        # loop work, but correctness is non-negotiable per Rule #1.
        # Phase 14h: moisture evaporation moved into step_gas_solid_coupling
        # so it competes with q_conv (not just q_rad) for the energy budget.
        # See coupling_3d.py docstring (Frandsen 1971; Albini 1985).
        # Phase 15D-F: when the FSD closure is active, compute the
        # smoothed-c gradient norm ONCE per outer step and reuse it
        # across all N_SUB chemistry sub-steps.  phi_flame doesn't
        # change inside the sub-step loop — see flame_front_3d update
        # site above — so the smoothing + gradient pass would be
        # identical on every call; precomputing saves ~N_SUB× work.
        _c_grad_norm_outer = None
        if (combustion_closure == "level_set_fsd"
                and phi_flame is not None):
            _t0 = _time.perf_counter()
            _c_grad_norm_outer = (
                level_set_fsd_3d.compute_c_grad_norm_from_phi_flame(
                    phi_flame, grid.dx, grid.dy, grid.dz_arr,
                )
            )
            _tick("combustion:fsd_c_grad", _t0)

        for _sub in range(N_SUB):
            # ── Phase 14s: operator-split chemistry (Lie split, A→B) ────
            # Operator A — local stiff ODE per cell.  Updates Y_F, Y_O2,
            # T_g in place via Rosenbrock-1 (linearly-implicit Euler).
            # Replaces former forward-Euler + Damköhler-cap + mass-cap
            # band-aids.  ω = min(ω_chem, ω_EBU, ω_O2_supply); the EBU
            # closure handles sub-grid mixing limit, ω_O2_supply handles
            # advective O₂ delivery (frozen during the ODE step).
            #
            # Drip-torch heat injection happens in the transport operator
            # below (it's an external energy source, not chemistry).
            _t0 = _time.perf_counter()
            omega_O2.fill(1.0e30)
            combustion_3d.step_o2_supply_rate(
                state.rho, state.u, state.v, state.w, state.Y_O2,
                grid.dx, grid.dy, grid.dz_arr, omega_O2,
            )
            _tick("combustion:o2_supply", _t0)
            # Phase 14u-cap: Damköhler turbulent-flame-speed cap on ω.
            # ω_max_T = ρ·(S_L+u')/dx — sub-grid model for our coarse dx.
            # Without this, EBU rate ρ·Y/τ_mix can give 10× higher ω in
            # mixing-layer cells where k-ε produces small τ_mix; the
            # explicit time integration can't handle the resulting
            # ∂ρ/∂t spikes (T_g jumps thousands of K per step).  This is
            # the Phase 14n cap moved INSIDE the chemistry ODE kernel.
            S_L_GRASS = 0.4   # [m/s] laminar flame speed (Williams 1985)
            _u_prime = np.sqrt(2.0 * k_turb / 3.0)
            _omega_max_T = state.rho * (S_L_GRASS + _u_prime) / grid.dx
            # Phase 15-0: dispatch via chemistry_closures registry.  Each
            # closure pulls what it needs from kwargs and ignores the rest
            # — see model_outdoor.physics_3d.chemistry_closures._interface.
            # Phase 15C: also passes phi_flame + grid metrics for FSD
            # (smoothing + |∇c| are computed inside the closure).
            _t0 = _time.perf_counter()
            _closure_kwargs = dict(
                rho=state.rho, T_g=state.T_g,
                Y_fuel=state.Y_fuel, Y_O2=state.Y_O2,
                tau_mix=tau_mix,
                omega_O2=omega_O2,
                omega_max_T=_omega_max_T,
                k_turb=k_turb,
                eps_turb=eps_turb,
                phi_flame=phi_flame,
                # Phase 15D-F precomputed per-outer-step grad — see above
                c_grad_norm=_c_grad_norm_outer,
                dx=grid.dx, dy=grid.dy, dz_arr=grid.dz_arr,
                chi_rad=chi_rad,
                cp_g=1100.0,
                dt=dt_sub,
                n_substeps=1,
                omega_out=omega,
            )
            # Phase 15D sensitivity-sweep override: pipe s_L through to FSD
            if s_L_fsd_override is not None:
                _closure_kwargs["s_L"] = float(s_L_fsd_override)
            # Phase 15G — Damköhler 1 turbulent s_T in FSD branch
            if turbulent_s_T_fsd:
                _closure_kwargs["use_turbulent_s_T"] = True
                _closure_kwargs["s_T_cap_factor"] = float(s_T_cap_factor)
            # Phase 15H — Charlette wrinkling factor on FSD chemistry
            if tfm_xi != 1.0:
                _closure_kwargs["tfm_xi"] = float(tfm_xi)
            # Phase 15J — Linn 2002 / Mell 2007 mixing-limited inner-body
            if inner_body_edc:
                _closure_kwargs["inner_body_edc"] = True
            # Phase 16 — opt-in extinction-threshold physics for EDC
            if edc_extinction_enable:
                _closure_kwargs["extinction_enable"] = True
            # Phase 16 — composition-dependent cp_mix for the T_g update
            # inside the chemistry kernel (water vapor cp ≈ 2× dry air).
            _closure_kwargs["Y_H2O"] = state.Y_H2O
            # Phase 23 — chemistry-family scalars.  Every closure's run()
            # accepts these; each picks up only the ones it needs.
            _closure_kwargs.update(_chem_family_kwargs)
            # Phase 23 2-step: pass Y_CO for the Westbrook-Dryer closure
            # (silently ignored by other closures via **_unused).
            _closure_kwargs["Y_CO"] = state.Y_CO
            chemistry_closures.run(combustion_closure, **_closure_kwargs)
            _tick(f"combustion:{combustion_closure}", _t0)
            # Q_comb stays zero — chemistry ODE already updated T_g.
            Q_comb.fill(0.0)
            # Phase 15O — Eulerian Finney-tendril spawn-and-deposit on the
            # flame leading-edge surface, conservation-preserving.  Operates
            # after the chemistry kernel so rho/T_g/Y_F reflect this step's
            # combustion, and before the next outer step's projection.
            if finney_tendril_enable and phi_flame is not None:
                _t0 = _time.perf_counter()
                _L_F_field = finney_tendril_3d._compute_L_F_per_column(
                    state.T_g,
                    T_thresh=finney_tendril_3d.T_GAS_FLAME,
                    dz_arr=grid.dz_arr,
                    n_z_bed=grid.n_z_bed,
                )
                _sr_use = (finney_tendril_sr if finney_tendril_sr is not None
                           else finney_tendril_3d.SR_DEFAULT)
                _duty_use = (finney_tendril_duty
                             if finney_tendril_duty is not None
                             else finney_tendril_3d.DUTY_CYCLE)
                _fmass_use = (finney_tendril_f_mass
                              if finney_tendril_f_mass is not None
                              else finney_tendril_3d.F_MASS_DEFAULT)
                _frmin_use = (finney_tendril_fr_min
                              if finney_tendril_fr_min is not None
                              else finney_tendril_3d.FR_MIN_DEFAULT)
                if finney_tendril_t_contact_s > 0.0:
                    # Phase 15O.1 — two-phase time-spread.
                    # Phase A: apply existing pending sink/deposit rates
                    finney_tendril_3d.step_finney_tendril_apply_pending(
                        state.rho, state.T_g, state.Y_fuel, state.u,
                        _ft_sink_M, _ft_sink_E, _ft_sink_Yf, _ft_sink_Px,
                        _ft_sink_t_rem,
                        _ft_dep_M, _ft_dep_E, _ft_dep_Yf, _ft_dep_Px,
                        _ft_dep_t_rem,
                        grid.dx, grid.dy, grid.dz_arr, dt_sub,
                    )
                    # Phase B: evaluate new spawns and queue inventory
                    finney_tendril_3d.step_finney_tendril_queue_spawns(
                        state.rho, state.T_g, state.Y_fuel, state.u,
                        phi_flame, _L_F_field, last_spawn_time_3d,
                        _ft_sink_M, _ft_sink_E, _ft_sink_Yf, _ft_sink_Px,
                        _ft_sink_t_rem,
                        _ft_dep_M, _ft_dep_E, _ft_dep_Yf, _ft_dep_Px,
                        _ft_dep_t_rem,
                        grid.dx, grid.dy, grid.dz_arr, t,
                        sr=_sr_use, duty_cycle=_duty_use,
                        f_mass=_fmass_use, fr_min=_frmin_use,
                        T_amb=T_amb,
                        t_contact_s=float(finney_tendril_t_contact_s),
                        n_spawn_events_out=finney_tendril_count,
                        box_dk_up_radius=int(finney_tendril_box_dk_up),
                        box_dk_down_radius=int(finney_tendril_box_dk_down),
                        box_dj_radius=int(finney_tendril_box_dj),
                        box_di_back_radius=int(finney_tendril_box_di_back),
                    )
                else:
                    # Phase 15O instantaneous (default, back-compat)
                    finney_tendril_3d.step_finney_tendril_spawn_deposit(
                        state.rho, state.T_g, state.Y_fuel,
                        state.u, state.v, state.w,
                        phi_flame, _L_F_field, last_spawn_time_3d,
                        grid.dx, grid.dy, grid.dz_arr, t,
                        sr=_sr_use, duty_cycle=_duty_use,
                        f_mass=_fmass_use, fr_min=_frmin_use,
                        T_amb=T_amb,
                        n_spawn_events_out=finney_tendril_count,
                    )
                _tick("finney_tendril", _t0)
            # Phase 15P — Lagrangian Finney particle closure.  Mutually
            # exclusive with the Eulerian path above; if both flags are
            # set, the Eulerian path wins (above ran already; skip below).
            if (finney_lagrangian_enable and phi_flame is not None
                    and not finney_tendril_enable):
                _t0 = _time.perf_counter()
                _L_F_field = finney_tendril_3d._compute_L_F_per_column(
                    state.T_g,
                    T_thresh=finney_tendril_3d.T_GAS_FLAME,
                    dz_arr=grid.dz_arr,
                    n_z_bed=grid.n_z_bed,
                )
                _sr_use = (finney_lagrangian_sr if finney_lagrangian_sr is not None
                           else finney_tendril_3d.SR_DEFAULT)
                _duty_use = (finney_lagrangian_duty
                             if finney_lagrangian_duty is not None
                             else finney_tendril_3d.DUTY_CYCLE)
                _fmass_use = (finney_lagrangian_f_mass
                              if finney_lagrangian_f_mass is not None
                              else finney_tendril_3d.F_MASS_DEFAULT)
                _frmin_use = (finney_lagrangian_fr_min
                              if finney_lagrangian_fr_min is not None
                              else finney_tendril_3d.FR_MIN_DEFAULT)
                # Phase A: advect live particles, deposit fractional inventory
                finney_lagrangian_3d.step_finney_lagrangian_advect(
                    state.rho, state.T_g, state.Y_fuel,
                    state.u, state.v, state.w,
                    _fl_x, _fl_y, _fl_z,
                    _fl_u, _fl_v, _fl_w,
                    _fl_m, _fl_E, _fl_Yf,
                    _fl_t_rem, _fl_alive,
                    grid.dx, grid.dy, grid.dz_arr, grid.z_face,
                    float(finney_lagrangian_d_p_m),
                    float(finney_lagrangian_C_D),
                    dt_sub,
                    _fl_n_alive, _fl_n_exit,
                )
                # Phase B: detect new spawns, debit source-cell, allocate particle
                finney_lagrangian_3d.step_finney_lagrangian_spawn(
                    state.rho, state.T_g, state.Y_fuel, state.u,
                    phi_flame, _L_F_field, last_spawn_time_3d,
                    _fl_x, _fl_y, _fl_z,
                    _fl_u, _fl_v, _fl_w,
                    _fl_m, _fl_E, _fl_Yf,
                    _fl_t_rem, _fl_alive,
                    grid.dx, grid.dy, grid.dz_arr, grid.z_mid,
                    t,
                    sr=_sr_use, duty_cycle=_duty_use,
                    f_mass=_fmass_use, fr_min=_frmin_use,
                    t_contact_s=float(finney_lagrangian_t_contact_s),
                    n_spawn_events_out=_fl_n_spawn,
                    n_spawn_overflow_out=_fl_n_overflow,
                )
                _tick("finney_lagrangian", _t0)
            # Phase 14ab: drip-torch external heat injection in source bed
            # cells during ignition_duration.  Volumetric W/m³ source;
            # treated as part of operator B (transport+coupling), fed via
            # Q_comb into step_gas_solid_coupling below.
            # Phase 14ap-2: drip torch gated off during spin-up.
            if _combustion_active and t < ignition_duration_s:
                # Phase 14q-2: drip-torch as per-area surface flux,
                # spread volumetrically over bed depth.  Pyne (1993)
                # Wildland Fire §11.3 hardware spec: ~0.2 L/min diesel/
                # gasoline → ~10–20 kW/m² fuel-chemical density × ~5%
                # bed-heating efficiency after plume losses → ~5 kW/m²
                # average bed-heating.  Independent of bed density
                # (ρ_b): drip torch is external fuel applied to bed
                # surface, not a bulk-density-dependent volumetric
                # source.  Replaces the prior `Q_DRIP_REF * (ρ_b/1.07)`
                # which over-energized Cut beds (~16× actual hardware).
                Q_DRIP_PER_AREA = 30_000.0  # [W/m²]  6× the prior 5 kW/m² setting,
                                            # applied for 1/6 the duration (5s vs 30s),
                                            # preserving total energy 150 kJ/m² but with
                                            # peak intensity that overcomes advective
                                            # washout in sparse beds.  Real drip-torch
                                            # hardware (Pyne 1993 §11.3) delivers ~119
                                            # kW/m² peak per swath for ~0.5s — this 30
                                            # kW/m² over 5s is the same TOTAL energy as
                                            # before, intermediate peak.
                _Q_drip = Q_DRIP_PER_AREA / max(h_bed, 1e-3)  # [W/m³]
                # Physically, a drip torch drops burning kerosene ONTO the
                # solid fuel.  Most of the heat goes directly to the bed
                # via conduction + flame impingement on grass blades; a
                # smaller fraction goes to the gas (combustion products).
                # WFDS (Mell 2007 §3.4) and FIRESTAR (Morvan & Dupuy 2004
                # §3) both apply ignition fluxes as surface heat on the
                # solid, not as a gas-phase volumetric source.  Split:
                #   80% → solid (via Q_pyro, signed: subtract = exothermic
                #         heat added to T_s in the coupling kernel).
                #   20% → gas  (via Q_comb, hot combustion products).
                # Prior to 2026-05-12 the full Q_drip went to Q_comb only,
                # so in sparse Nat beds (high in-bed wind, low gas-solid
                # coupling) the bed solid was barely heated by the drip
                # torch and pyrolysis couldn't bootstrap.
                F_DRIP_TO_SOLID = 0.80
                # Phase 16: when bed particles own the solid, the 80%
                # solid portion has ALREADY been distributed to particles
                # via _bed_Qsx above; only the 20% gas portion remains here.
                if not (lagrangian_bed_enable and _bed_buf is not None):
                    Q_pyro[:grid.n_z_bed, :, i_src_start:i_src_end] -= F_DRIP_TO_SOLID * _Q_drip
                Q_comb[:grid.n_z_bed, :, i_src_start:i_src_end] += (1.0 - F_DRIP_TO_SOLID) * _Q_drip
            # Phase 14x: bootstrap newly-burning cells via Q_bootstrap.
            # Fires only in flame_body cells with cell_age < t_bootstrap.
            # Source patch cells were initialized with cell_age past the
            # bootstrap window (drip torch handles them).
            # Phase 14y: bootstrap is GATED by combustion_closure: only the
            # legacy 'ebu_bootstrap' closure uses it.  EDC/PaSR don't need
            # it (subgrid closures self-ignite); 'pin' replaces it with a
            # post-coupling T_g pin (more energy-conserving).
            if (flame_body_mask is not None
                    and combustion_closure == "ebu_bootstrap"):
                flame_front_3d.apply_bootstrap_heat(
                    Q_comb, flame_body_mask, cell_age,
                    Q_bootstrap=flame_front_3d.Q_BOOTSTRAP_W_M3,
                    t_bootstrap=flame_front_3d.T_BOOTSTRAP_S,
                )
            # ── Operator B — transport (no chemistry source) ────────────
            # Y_fuel: source is +S_pyro only (chemistry sink already
            # applied in operator A).  Y_O2: source is 0 (chemistry already
            # consumed it).  Y_O2: source is 0.  Y_H2O: source is the
            # drying-vapor mass injection only.  All transport advect/
            # diffuse normally.
            #
            # Phase 16 (2026-06-16): split Y_fuel from Y_H2O.  Prior to
            # this change S_pyro = _bed_Sp + _bed_Sd was used as the
            # Y_fuel source, which counted drying water vapor AS fuel —
            # producing fake extra combustion at higher MC that masked
            # the real Cheney moisture penalty.  Now drying vapor goes
            # to Y_H2O (separate species); only true pyrolysis volatile
            # (_bed_Sp) drives Y_fuel.  When Lagrangian bed is disabled
            # the legacy combined S_pyro is reused (Eulerian path
            # backwards compatibility).
            _t0 = _time.perf_counter()
            if lagrangian_bed_enable and _bed_buf is not None:
                _S_F_only = _bed_Sp                  # true pyrolysis volatile
                _S_H2O    = _bed_Sd                  # drying water vapor only
            else:
                _S_F_only = S_pyro                   # Eulerian legacy
                _S_H2O    = np.zeros_like(S_pyro)
            # Mass-conservative source with dilution (Phase 16):
            # ∂Y_i/∂t = (S_i − Y_i·S_total) / ρ − u·∇Y_i
            # The Y_i·S_total term dilutes a species when OTHER species
            # are injected.  S_total here is the sum of all gas-phase
            # mass injections (pyrolysate volatile + drying vapor).
            _S_total = _S_F_only + _S_H2O
            _S_F_eff   = _S_F_only - state.Y_fuel * _S_total
            _S_O2_eff  =            - state.Y_O2  * _S_total
            _S_H2O_eff = _S_H2O    - state.Y_H2O  * _S_total
            # Phase 23 Refactor 2D: for cup burner (or any z-min-inlet BC),
            # pass the per-cell z-min ghost arrays that the BC populated
            # in configure() into each species transport call.  For
            # outdoor cases the helper returns dummy zeros + inactive
            # flag, which is bit-exact-invariant with the pre-Phase-23
            # zero-flux wall behaviour.
            def _species_zmin_kwargs(name):
                if _is_cup_burner:
                    return _bc.species_inlet_kwargs(name)
                return {}
            species_3d.step_species_transport(
                state.Y_fuel, state.rho, state.u, state.v, state.w,
                _S_F_eff, dt_sub, grid.dx, grid.dy,
                grid.dz_arr, grid.d_face_above, grid.d_face_below,
                D=1.0e-5,
                Y_inlet=0.0,   # clean wind: no fuel
                **_species_zmin_kwargs("Y_fuel"),
            )
            species_3d.step_species_transport(
                state.Y_O2, state.rho, state.u, state.v, state.w,
                _S_O2_eff, dt_sub, grid.dx, grid.dy,
                grid.dz_arr, grid.d_face_above, grid.d_face_below,
                D=1.0e-5,
                Y_inlet=0.232,   # ambient O2 mass fraction
                **_species_zmin_kwargs("Y_O2"),
            )
            species_3d.step_species_transport(
                state.Y_H2O, state.rho, state.u, state.v, state.w,
                _S_H2O_eff, dt_sub, grid.dx, grid.dy,
                grid.dz_arr, grid.d_face_above, grid.d_face_below,
                D=1.0e-5,
                Y_inlet=0.0,   # clean wind: no humidity in
                **_species_zmin_kwargs("Y_H2O"),
            )
            # Phase 23 2-step: Y_CO transported only when the 2-step
            # Westbrook-Dryer methane closure is active.  Source comes
            # from the chemistry ODE (R1 produces CO, R2 consumes it),
            # so transport source is zero here.  Gate on closure name
            # so outdoor cases skip the extra advection call entirely
            # (compute + bit-exact preserving).
            if combustion_closure == "edc_2step_methane":
                _S_CO_zero = np.zeros_like(_S_H2O_eff)
                species_3d.step_species_transport(
                    state.Y_CO, state.rho, state.u, state.v, state.w,
                    _S_CO_zero, dt_sub, grid.dx, grid.dy,
                    grid.dz_arr, grid.d_face_above, grid.d_face_below,
                    D=1.0e-5,
                    Y_inlet=0.0,   # clean wind: no CO in
                    **_species_zmin_kwargs("Y_CO"),
                )
            # Y_O2 stays bounded [0, 0.232] (fresh-air ceiling); Y_H2O is
            # bounded [0, 1] by clipping.  Mass-fraction completeness
            # (1 − Y_F − Y_O2 − Y_H2O ≥ 0) enforced by clipping each
            # individually; minor numerical leakage is acceptable.
            np.clip(state.Y_O2, 0.0, 0.232, out=state.Y_O2)
            np.clip(state.Y_H2O, 0.0, 1.0, out=state.Y_H2O)
            _tick("species:transport", _t0)
            # Phase 14b: turbulent diffusion of all gas species.
            _t0 = _time.perf_counter()
            turbulence_3d.apply_turbulent_diffusion(
                state.Y_fuel, nu_t, sc_t=turbulence_3d.SC_T,
                dt=dt_sub, dx=grid.dx, dy=grid.dy,
                dz_arr=grid.dz_arr, d_face_above=grid.d_face_above,
                d_face_below=grid.d_face_below,
            )
            turbulence_3d.apply_turbulent_diffusion(
                state.Y_O2, nu_t, sc_t=turbulence_3d.SC_T,
                dt=dt_sub, dx=grid.dx, dy=grid.dy,
                dz_arr=grid.dz_arr, d_face_above=grid.d_face_above,
                d_face_below=grid.d_face_below,
            )
            turbulence_3d.apply_turbulent_diffusion(
                state.Y_H2O, nu_t, sc_t=turbulence_3d.SC_T,
                dt=dt_sub, dx=grid.dx, dy=grid.dy,
                dz_arr=grid.dz_arr, d_face_above=grid.d_face_above,
                d_face_below=grid.d_face_below,
            )
            np.clip(state.Y_O2, 0.0, 0.232, out=state.Y_O2)
            np.clip(state.Y_fuel, 0.0, 1.0, out=state.Y_fuel)
            np.clip(state.Y_H2O, 0.0, 1.0, out=state.Y_H2O)
            _tick("species:turb_diff", _t0)
            # Coupling: T_g, T_s, m_water update at dt_sub.  Phase 14h:
            # moisture evap is computed inside this kernel using rad+conv
            # heat, so q_rad_in is the raw P1 net flux (no pre-subtraction).
            # Phase 14x: include Frankman flame-tip convective flux as a
            # separate contribution to the bed solid (kept distinguishable
            # via q_frankman_3d for diagnostics; combined into q_in_to_solid
            # for the coupling kernel which sees it as additional q_rad-equiv
            # flux to the bed solid).
            np.add(q_rad, q_frankman_3d, out=_q_in_to_solid)
            # ── q_rad_gas → Q_comb (gas-phase radiation absorption) ──
            # DOM computes the gas-share absorbed radiation (κ_g·α_g
            # fraction of total cell absorption) and writes it to
            # q_rad_gas in W/m².  This was added in Phase 14a but the
            # caller never read the array — gas absorption was being
            # silently discarded.  Plume gases (smoke, soot, H2O, CO2)
            # CAN absorb a meaningful fraction of bed radiation; at
            # hot plumes the gas-share can be 30-50% of the cell total.
            # Conversion: q_rad_gas [W/m²] / dz_arr[k] = volumetric W/m³.
            for _k_qrg in range(grid.Nz):
                Q_comb[_k_qrg] += q_rad_gas[_k_qrg] / grid.dz_arr[_k_qrg]
            _t0 = _time.perf_counter()
            if lagrangian_bed_enable and _bed_buf is not None:
                # Phase 16 — particle path: bed kernel has already updated
                # particle T_s + emitted Q_g_conv to extract heat from gas.
                # Apply Q_g_conv to T_g directly (per-cell energy balance:
                #   ρ·cp·dT_g/dt = -Q_g_conv + Q_comb)
                # Particle T_s is NOT in state.T_s grid — the Eulerian
                # state.T_s remains as a (now non-load-bearing) diagnostic
                # array; we set it to mean particle T_s in the snapshot
                # writer below.  The Q_pyro array (built from particle
                # sources above) acts as a normal volumetric heat source
                # in this branch only via the gas energy equation; we
                # apply that here directly since coupling is skipped.
                # Composition-dependent gas-mixture specific heat
                # (Phase 16, 2026-06-18).  Water vapor has cp ≈ 2000 J/kg/K
                # at flame T while dry air is ≈ 1100 J/kg/K (NIST tables,
                # 1000-2000 K).  Pre-Phase 16 the gas energy equation used
                # the dry-air constant; in moisture-laden burn cells with
                # Y_H2O up to 0.3, this UNDER-estimated thermal inertia by
                # up to ~25%, making the gas warm faster than it should
                # and erasing part of the Cheney moisture-coefficient
                # feedback.  Linear binary mixture:
                #   cp_mix = (1 − Y_H2O) × cp_air + Y_H2O × cp_H2O
                _CP_GAS_DRY = 1100.0   # dry air at flame T
                _CP_VAPOR   = 2000.0   # water vapor (NIST)
                _cp_mix = (1.0 - state.Y_H2O) * _CP_GAS_DRY + state.Y_H2O * _CP_VAPOR
                # Vapor sensible-heat debit at INJECTION (independent of
                # the cp_mix term above): each kg of pyrolysate / drying
                # vapor injected at T_inject must be warmed to T_g.
                _T_INJECT_DRY = lagrangian_bed_3d.T_BOIL_WATER
                _Q_vapor_debit = ((
                    _bed_Sp * np.maximum(state.T_g - state.T_s, 0.0)
                    + _bed_Sd * np.maximum(state.T_g - _T_INJECT_DRY, 0.0)
                ) * _CP_VAPOR)
                # ρ·cp_mix·dT_g = (Q_comb − Q_g_conv − Q_vapor_debit) · dt
                _gas_inv = dt_sub / (np.maximum(state.rho, 1.0e-3) * _cp_mix)
                state.T_g += (Q_comb - _bed_Qgc - _Q_vapor_debit) * _gas_inv
                # Cap downward at T_amb so transient over-cooling doesn't
                # tip cells into nonphysical T_g < T_amb (matches the
                # coupling kernel's effective floor).
                np.maximum(state.T_g, T_amb, out=state.T_g)
            else:
                coupling_3d.step_gas_solid_coupling(
                    state.T_g, state.T_s, state.rho,
                    state.u, state.v, state.w, state.alpha_s,
                    sigma_sav, _q_in_to_solid, Q_pyro, Q_comb,
                    m_water, L_v if _has_moisture else 0.0,
                    dt_sub, grid.dz_arr, T_amb,
                    q_loss_enable=False,   # Phase 14a: P1 already includes self-emission
                    h_conv_mult=h_conv_mult,
                )
            _tick("coupling:gas_solid", _t0)
            # ── Phase 14ac: vertical solid-side conduction ──────────────
            # Grass blade as continuous solid spanning bed cells.  Heat
            # absorbed at the top of the bed (DOM downward radiation +
            # gas-solid convective coupling from the overhead plume)
            # conducts down the blade and warms the base.  Without this
            # term, T_s remained strictly cell-local — only the topmost
            # bed cell heated, and pyrolysis there could not propagate
            # down through the bed depth.  See solid_conduction_3d.py
            # docstring (Fons 1946, Spalding 1963, Petrich 2008).
            _t0 = _time.perf_counter()
            solid_conduction_3d.step_solid_conduction_vertical(
                state.T_s, state.alpha_s,
                grid.dz_arr, grid.d_face_above, grid.d_face_below,
                k_solid=solid_conduction_3d.K_SOLID_GRASS,
                rho_solid=coupling_3d._RHO_SOLID,
                cp_solid=coupling_3d._CP_SOLID,
                dt=dt_sub,
            )
            _tick("solid_conduction", _t0)
            # Phase 14b: turbulent diffusion of T_g (lifts hot gas).
            # Use Pr_t = 0.85 (Kays & Crawford 1993).
            # ABLATION result: only -37K effect; ν_t in this regime is small
            # (~1e-3 m²/s, not the 0.1 my budget guess assumed).  Restored.
            _t0 = _time.perf_counter()
            turbulence_3d.apply_turbulent_diffusion(
                state.T_g, nu_t, sc_t=turbulence_3d.PR_T,
                dt=dt_sub, dx=grid.dx, dy=grid.dy,
                dz_arr=grid.dz_arr, d_face_above=grid.d_face_above,
                d_face_below=grid.d_face_below,
            )
            _tick("gas_energy:turb_diff", _t0)
            # Phase 14y: pin closure dropped — was a scaffolding shortcut
            # like bootstrap.  Rely on EDC/EBU+bootstrap/PaSR closures
            # for legitimate physics-based gas-phase combustion.
            # Apply caps each sub-step (prevent overshoot mid-loop)
            np.minimum(state.T_g, T_FLAME_AD, out=state.T_g)
            np.maximum(state.T_g, T_amb,      out=state.T_g)
            np.minimum(state.T_s, T_SURF_MAX, out=state.T_s)
            np.maximum(state.T_s, T_amb,      out=state.T_s)
            # Phase 14ah-4 REVERTED 2026-05-17: the previous early-exit
            # silently broke ignition (see comment at _y_fuel_prev_sub init).
            # Always run all N_SUB substeps; chemistry needs the full
            # integration window to bootstrap into self-sustaining mode.
        # ── 12c. Phase 14x: level-set evolution ─────────────────────────
        # Update cell_age, compute v_n from CFD heat flux integral
        # (mesh-convergent), evolve level-set front position.
        if grid.n_z_bed > 0 and flame_body_mask is not None:
            _t0 = _time.perf_counter()
            # Update cell_age for bootstrap window tracking
            flame_front_3d.update_cell_age(cell_age, flame_body_mask, dt)
            # Compute v_n from heat flux integral (mesh-convergent).
            # Phase 15N: q_burst_conv_2d was already computed during the
            # radiation section and added to q_rad there.  Pass the same
            # value to compute_q_in_at_front so the v_n driver also sees
            # the burst contribution; the chemistry now follows because
            # T_s rises from the q_rad addition.
            # Phase 17b — true 3D forcing.  q_in is now per-cell; v_n is
            # per-cell with moisture-aware E_ign (Drysdale §3.5 + Mell 2007
            # §3.4 latent-heat-of-evaporation term).  The level-set is
            # genuinely 3D — top-of-bed cells advance faster than bottom
            # because forward IR reaches them first; wet ahead-of-front
            # cells advance slower because some incoming heat goes to
            # drying.  ``enforce_z_uniformity`` removed: z-variation is now
            # physics, not numerical drift.
            #
            # Legacy 2D (compute_q_in_at_front + compute_v_n) is kept in
            # flame_front_3d.py for backward compatibility with cases that
            # don't pass M_local — they fall through compute_v_n_3d with
            # M_local=None and recover the dry-only sensible-heat formula.
            q_in_at_front_3d_full = flame_front_3d.compute_q_in_at_front_3d(
                q_frankman_3d, q_dom_fwd_3d, ahead_band_mask,
                q_burst_conv_2d=q_burst_conv_2d,
            )
            # Surface-flux equivalent for backward-compat snapshot output:
            # column-sum still useful as a diagnostic.
            q_in_at_front_2d = q_in_at_front_3d_full.sum(axis=0)
            if level_set_passive:
                # Phase 17c test (2026-06-23): zero v_n forcing.  Level-set
                # stays at source-patch position; bed must self-ignite
                # ahead via CFD advection + DOM radiation + bed coupling.
                # Tests whether the kinematic v_n was masking working
                # CFD physics or providing essential closure.
                v_n_field.fill(0.0)
            else:
                v_n_field[:] = flame_front_3d.compute_v_n_3d(
                    q_in_at_front_3d_full,
                    rho_b, _CP_SOLID, h_bed, T_ign, T_amb,
                    M_local=(_bed_M_local if _bed_M_local is not None else None),
                )
            # Phase 19: blend with empirical-ROS at low U.  WRF-Fire pattern.
            # v_n_blended = (1 - w) * v_n_resolved + w * v_n_empirical
            # w = 1 below U_threshold - blend_width (empirical dominates)
            # w = 0 at-or-above U_threshold        (resolved dominates)
            # See model_outdoor/empirical_ros.py + Phase 18 bug-sweep memo.
            if empirical_ros_enable and _empirical_blend_w > 0.0:
                _w = _empirical_blend_w
                v_n_field *= (1.0 - _w)
                v_n_field += _w * _empirical_ros_m_s
            # Evolve level-set + reinit run in BOTH modes (the prior fix
            # to skip reinit in passive turned out to be a regression —
            # without periodic reinit the float precision drifts even
            # with v_n=0, causing other downstream issues).
            lset.evolve(dt, v_n_field)
            lset.maybe_reinitialize()
            # Phase 19: dynamic bed seeding to follow empirical-ROS level-set.
            # WRF-Fire-style "burn at the front": as the level-set advances
            # at the prescribed empirical ROS, force bed particles BEHIND
            # the front to ignite at solid_phase_ignition_T_s_K.  Without
            # this the level-set "ghost-advances" without consuming fuel
            # (front-only marker, no resolved bed propagation).  Throttled
            # to every 50 outer steps (~50ms sim time at typical dt=1ms);
            # at empirical ROS = 0.1 m/s this is 5 mm of front advance per
            # check, well below dx=0.05m cell scale.
            if (empirical_ros_enable and _empirical_blend_w > 0.0
                    and lagrangian_bed_enable and _bed_buf is not None):
                _empirical_seed_step_counter += 1
                if _empirical_seed_step_counter >= 50:
                    _empirical_seed_step_counter = 0
                    _lset_front_x = lset.front_x(k=1, j=grid.Ny // 2)
                    _x_buf = _bed_buf["x"]
                    _alive = _bed_buf["alive"]
                    _T_s_arr = _bed_buf["T_s"]
                    _m_solid_arr = _bed_buf["m_solid"]
                    _x_lo_bed = grid.x_mid[i_bed_start] - 0.5 * grid.dx
                    _T_seed = float(solid_phase_ignition_T_s_K)
                    # As the empirical-ROS level-set advances, apply the same
                    # T_s kick that solid_phase_ignition applies at t=0 — but
                    # ONLY to particles that still have fuel to burn.  The
                    # m_solid > 1e-9 gate is the "particle isn't burnt out"
                    # check: it lets a particle go through its full
                    # combustion cycle, then stops re-heating it once dead.
                    # Without this gate, burnt-out source-patch particles
                    # get their T_s pinned at seed value forever = zombie
                    # hot block visible in early Phase 19 GIFs.
                    _mask = (
                        (_alive == 1)
                        & (_x_buf >= _x_lo_bed)
                        & (_x_buf <= _lset_front_x)
                        & (_T_s_arr < T_ign)
                        & (_m_solid_arr > 1.0e-9)
                    )
                    if _mask.any():
                        _T_s_arr[_mask] = _T_seed
            _tick("level_set:evolve", _t0)
            # Phase 14x diagnostics (every 2s)
            if t - _last_14x_print_t >= 2.0:
                _last_14x_print_t = t
                _front_x_lset = lset.front_x(k=1, j=grid.Ny // 2)
                # In-bed wind diagnostics: sample u at the front column
                # (j=Ny/2) at each bed-cell z, just ahead of the front.
                _i_front = int(_front_x_lset / grid.dx)
                _i_diag  = min(_i_front + 1, grid.Nx - 1)
                _j_diag  = grid.Ny // 2
                _u_bed_profile = state.u[:grid.n_z_bed, _j_diag, _i_diag]
                _u_bed_str = ",".join(f"{u:.2f}" for u in _u_bed_profile)
                # Chemistry diagnostics inside flame_body
                if flame_body_mask.any():
                    _T_g_fb = state.T_g[flame_body_mask]
                    _T_g_max_fb  = float(_T_g_fb.max())
                    _T_g_p99_fb  = float(np.percentile(_T_g_fb, 99))
                    _T_g_med_fb  = float(np.median(_T_g_fb))
                    _Y_F_max_fb  = float(state.Y_fuel[flame_body_mask].max())
                    _Y_O2_med_fb = float(np.median(state.Y_O2[flame_body_mask]))
                    _omega_max_fb = float(omega[flame_body_mask].max())
                    # Cap-activation: how many cells are pinned at the
                    # numerical safety cap (was 1850K = 1900 cap − 50K
                    # tolerance; now 9950K = 10000 cap − 50K).  Should
                    # never bind in normal physics.
                    _n_cap_active = int(((state.T_g >= 9950.0) & flame_body_mask).sum())
                    # omega_O2 supply limit: small value = O2-supply-limited
                    _omega_O2_fb = omega_O2[flame_body_mask]
                    _omega_O2_med_fb = float(np.median(_omega_O2_fb))
                    # Turbulence diagnostics — EDC closure rate
                    # γ*/τ* depends on k, ε.  If both are at floors
                    # (k=1e-4, ε=1e-6), EDC is essentially dead.
                    _k_turb_med_fb = float(np.median(k_turb[flame_body_mask]))
                    _eps_turb_med_fb = float(np.median(eps_turb[flame_body_mask]))
                else:
                    _T_g_max_fb = float(state.T_g.max())
                    _T_g_p99_fb = _T_g_max_fb
                    _T_g_med_fb = _T_g_max_fb
                    _Y_F_max_fb = float(state.Y_fuel.max())
                    _Y_O2_med_fb = float(np.median(state.Y_O2))
                    _omega_max_fb = float(omega.max())
                    _n_cap_active = 0
                    _omega_O2_med_fb = 0.0
                    _k_turb_med_fb = 0.0
                    _eps_turb_med_fb = 0.0
                # Phase 14ax/aq diagnose: full turbulence state for
                # natural-capture audit.
                _k_max = float(k_turb.max())
                _eps_min = float(eps_turb[eps_turb > 0].min()) if (eps_turb > 0).any() else 0.0
                _nu_t_max = float(nu_t.max())
                _sigma_y_u = float(state.u.std(axis=1).max())
                _sigma_y_w = float(state.w.std(axis=1).max())
                _sigma_y_Tg = float(state.T_g.std(axis=1).max())
                _sigma_y_alpha_s = float(state.alpha_s.std(axis=1).max())
                # Spatial probes of σ_y(u) at 3 x stations: inlet face,
                # bed-leading-edge, bed-middle.  Tells us if inflow
                # perturbations propagate or get diffused into 2D.
                _i_inlet = 0
                # Phase 15F: clip probe indices to grid bounds so small
                # mickey-mouse domains (Lx ≪ 40 m) don't IndexError.
                _i_lead  = min(int(2.0 / grid.dx), grid.Nx - 1)
                _i_mid   = min(int(20.0 / grid.dx), grid.Nx - 1)
                _sigma_y_u_inlet = float(state.u[:, :, _i_inlet].std(axis=1).max())
                _sigma_y_u_lead  = float(state.u[:, :, _i_lead].std(axis=1).max())
                _sigma_y_u_mid   = float(state.u[:, :, _i_mid].std(axis=1).max())
                # Solid-field probe: σ_y of S_pyro within burning bed cells.
                # If fuel-pert α_s asymmetry doesn't translate to pyrolysis
                # asymmetry, the bed is effectively y-uniform for combustion.
                _bed = (slice(0, grid.n_z_bed), slice(None), slice(None))
                # Turbulence levels in BED cells (where the surface layer
                # sits).  With wall_function=False these should be tiny
                # unless SEM or buoyancy pump TKE there.
                _k_bed_max = float(k_turb[_bed].max())
                _k_bed_med = float(np.median(k_turb[_bed]))
                _nu_t_bed_max = float(nu_t[_bed].max())
                # Above-bed gas (free atmosphere): expected to dominate.
                _abv = (slice(grid.n_z_bed, None), slice(None), slice(None))
                _k_abv_max = float(k_turb[_abv].max())
                _nu_t_abv_max = float(nu_t[_abv].max())
                _burn_mask = (state.T_s[_bed] > 600.0)   # ~ignition threshold
                if _burn_mask.any():
                    _sp_burn = S_pyro[_bed][_burn_mask]
                    _sigma_y_sp = float(_sp_burn.std()) if _sp_burn.size > 5 else 0.0
                else:
                    _sigma_y_sp = 0.0
                _sigma_y_alpha_s_bed = float(state.alpha_s[_bed].std(axis=1).max())
                print(f"  [14x] t={t:5.2f}s  front_lset={_front_x_lset:.3f}m  "
                      f"v_n_max={v_n_field.max():.4f} m/s  "
                      f"q_dom_fwd_max={q_dom_fwd_3d.max():.2e}  "
                      f"#flame_body={flame_body_mask.sum()}  "
                      f"#bootstrap={(cell_age < flame_front_3d.T_BOOTSTRAP_S).sum()}  "
                      f"u_bed[{_u_bed_str}]m/s  "
                      f"chem[T_g={_T_g_max_fb:.0f}K p99={_T_g_p99_fb:.0f} "
                      f"med={_T_g_med_fb:.0f}, Y_F={_Y_F_max_fb:.3f}, "
                      f"Y_O2_med={_Y_O2_med_fb:.3f}, "
                      f"omega={_omega_max_fb:.2e}, "
                      f"omega_O2_med={_omega_O2_med_fb:.2e}, "
                      f"#cap@1900K={_n_cap_active}, "
                      f"k={_k_turb_med_fb:.2e}, eps={_eps_turb_med_fb:.2e}]  "
                      f"turb[k_max={_k_max:.2e}, nu_t_max={_nu_t_max:.2e}, "
                      f"eps_min={_eps_min:.2e}]  "
                      f"σ_y[u={_sigma_y_u:.3f}, w={_sigma_y_w:.3f}, "
                      f"Tg={_sigma_y_Tg:.1f}K, α_s={_sigma_y_alpha_s:.2e}]  "
                      f"σ_y(u)_x[inlet={_sigma_y_u_inlet:.3f}, "
                      f"lead={_sigma_y_u_lead:.3f}, mid={_sigma_y_u_mid:.3f}]  "
                      f"σ_y[α_s_bed={_sigma_y_alpha_s_bed:.2e}, "
                      f"S_pyro_burn={_sigma_y_sp:.2e}]  "
                      f"bed[k_max={_k_bed_max:.2e}, "
                      f"k_med={_k_bed_med:.2e}, "
                      f"nu_t_max={_nu_t_bed_max:.2e}]  "
                      f"abv[k_max={_k_abv_max:.2e}, nu_t_max={_nu_t_abv_max:.2e}]",
                      flush=True)
            # Termination criteria (2026-05-13 update):
            # (a) Fire extinct: flame_body_mask empty for N consecutive
            #     steps AFTER ignition_duration_s + bootstrap window.
            #     Old criterion (v_n < 1e-3) didn't fire because residual
            #     DOM-only propagation keeps v_n ≈ 0.05 m/s well above
            #     threshold even with zero flame body.
            # (b) Front exit-near: front_lset within 1 cell of bed_x_end →
            #     stop to avoid outlet-BC contamination of the ROS measure.
            _front_x_lset = lset.front_x(k=1, j=grid.Ny // 2)
            _front_exit_thresh = bed_x_end - grid.dx
            if (flame_body_mask is not None
                    and flame_body_mask.sum() == 0
                    and t > ignition_duration_s + flame_front_3d.T_BOOTSTRAP_S):
                _v_n_extinct_count += 1
                if _v_n_extinct_count > 50:
                    print(f"  [Phase 14x] flame_body=0 for 50 steps after "
                          f"ignition+bootstrap (t={t:.2f}s); fire extinct, "
                          f"exiting.", flush=True)
                    break
            else:
                _v_n_extinct_count = 0
            if _front_x_lset > _front_exit_thresh:
                print(f"  [Phase 14x] front_lset={_front_x_lset:.2f}m within "
                      f"1 cell of bed_x_end={bed_x_end:.2f}m at t={t:.2f}s; "
                      f"exiting before outlet contamination.", flush=True)
                break
        # ── 13. EoS update (low-Mach: ρ = P0 / (R_AIR · T_g)) ────────────
        np.copyto(state.rho, _P0 / (_R_AIR * np.maximum(state.T_g, T_amb)))
        # ── 14. Front tracking ───────────────────────────────────────────
        # Phase 14x: when level-set is active and advancing, append its
        # front position to front_t/front_x for ROS computation.
        # Phase 15F: T_s-based tracker is now diagnostic-only when the
        # level-set is active (n_z_bed > 0).  The "any cell ≥ T_ign"
        # heuristic is mesh-runaway by construction (fine meshes get
        # cells past T_ign sooner since each cell has less thermal mass
        # per horizontal area), and was previously silently overriding
        # the level-set via ``max()`` — see ``phase15f_front_tracking_*``
        # memory note for the diagnosis.  Cold-flow runs (n_z_bed=0)
        # keep ``append=True`` so the T_s tracker remains the sole
        # source of front_x when there is no level-set.
        _front_ts = _update_front_tracking(
            state, grid, T_ign, t, front_t, front_x,
            append=(grid.n_z_bed == 0),   # only T_s-driven in cold-flow
        )
        if grid.n_z_bed > 0:
            _front_lset = lset.front_x(k=grid.n_z_bed // 2, j=grid.Ny // 2)
            if math.isfinite(_front_lset):
                _last_appended = front_x[-1] if front_x else 0.0
                if _front_lset > _last_appended:
                    front_t.append(t)
                    front_x.append(_front_lset)
                _new_front = _front_lset    # NOT max() — level-set wins
            else:
                _new_front = _front_ts      # fallback when level-set undefined
        else:
            _new_front = _front_ts
        if _new_front > _last_front_x:
            _last_front_x = _new_front
            _last_advance_t = t

        # Phase 15D-SS — steady-state-driven termination.
        if steady_state_detect and t >= steady_state_warmup_s:
            if (_ss_last_check_t is None
                    or (t - _ss_last_check_t) >= steady_state_check_interval_s):
                # Slice the front history to the rolling window
                _ft_arr = np.asarray(front_t, dtype=np.float64)
                _fx_arr = np.asarray(front_x, dtype=np.float64)
                _mask = _ft_arr >= (t - steady_state_window_s)
                if int(_mask.sum()) >= 5:
                    _tt = _ft_arr[_mask]
                    _xx = _fx_arr[_mask]
                    _slope = float(np.polyfit(_tt, _xx, 1)[0])
                    if _ss_last_slope is not None and _ss_last_slope > 0.0:
                        _rel = abs(_slope - _ss_last_slope) / abs(_ss_last_slope)
                        if _rel < steady_state_tolerance:
                            _ss_consec += 1
                            if _ss_consec >= 2:
                                print(f"  [steady-state] converged at t={t:.2f}s  "
                                      f"slope={_slope:.4f} m/s = {_slope*60:.2f} m/min  "
                                      f"(rel change {_rel*100:.2f}% < {steady_state_tolerance*100:.1f}% "
                                      f"for 2 consecutive {steady_state_check_interval_s:.1f}s checks)",
                                      flush=True)
                                break
                        else:
                            _ss_consec = 0
                    _ss_last_slope = _slope
                _ss_last_check_t = t

        t += dt
        # Append diagnostics each step.
        diag_t.append(t)
        diag_Tg_max.append(float(state.T_g.max()))
        diag_Ts_max.append(float(state.T_s.max()))
        diag_Sp_max.append(float(S_pyro.max()))
        diag_Qc_max.append(float(Q_comb.max()))
        diag_omega_max.append(float(omega.max()))
        diag_Y_max.append(float(state.Y_fuel.max()))
        diag_n_ign.append(int(np.sum(state.T_s >= T_ign)))
        diag_proj_div_max.append(proj_div_max)
        diag_proj_n_iter.append(proj_n_iter)

        # ── Diagnostic print ─────────────────────────────────────────────
        if t - _last_print_t >= _print_dt:
            _last_print_t = t
            _n_ign = int(np.sum(state.T_s >= T_ign))
            _Y_max = float(state.Y_fuel.max())
            _Q_comb_max = float(Q_comb.max())
            _omega_max = float(omega.max())
            _S_pyro_max = float(S_pyro.max())
            # Where is Y_max?
            _idx_y = np.unravel_index(np.argmax(state.Y_fuel), state.Y_fuel.shape)
            _Tg_at_Ymax = float(state.T_g[_idx_y])
            # Where is Tg_max (excl pilot)?
            _idx_tg = np.unravel_index(np.argmax(state.T_g), state.T_g.shape)
            _Y_at_Tgmax = float(state.Y_fuel[_idx_tg])
            print(f"  t={t:6.2f}s  front={front_x[-1]:5.2f}m  "
                  f"n_ign={_n_ign:5d}  Tg_max={float(state.T_g.max()):.0f}K  "
                  f"Y_max={_Y_max:.3e}@Tg{_Tg_at_Ymax:.0f}K  "
                  f"Tg_max@Y{_Y_at_Tgmax:.3e}  "
                  f"Sp={_S_pyro_max:.2e} om={_omega_max:.2e}  "
                  f"dt={dt:.4f}s  proj_iter={proj_n_iter} divmax={proj_div_max:.2e}",
                  flush=True)

        # ── Phase 14y-snap: snapshot state for animation rendering ──────
        if snapshot_dir is not None and t - _last_snapshot_t >= snapshot_interval_s:
            _last_snapshot_t = t
            _snap_path = snapshot_dir / f"snap_{_snapshot_idx:04d}.npz"
            # ── Phase 16 bed-particle + projection convergence diagnostic ──
            # Print per-step budget for the hottest particle and projection
            # residual to localize the Q_in source driving T_s overshoot.
            if lagrangian_bed_enable and _bed_diag_max[0] > 0.0:
                _d = _bed_diag_max
                print(f"  [bp-diag t={t:6.3f}] T_s_max={_d[0]:7.0f}K  "
                      f"T_g={_d[9]:5.0f}K  "
                      f"Q_conv={_d[1]:+8.2f}  Q_char={_d[4]:+7.2f}  "
                      f"Q_smold={_d[5]:+6.2f}  Q_ext={_d[6]:+7.2f}  "
                      f"Q_rxn={_d[2]:+7.2f}  Q_dry={_d[3]:+6.2f}  "
                      f"|Q_in={_d[8]:+9.2f}W  Q_rad={_d[7]:+9.2f}W  "
                      f"mc/dt={_d[10]:.1f}  C_rad={_d[11]:.2e}  "
                      f"A_p={_d[14]:.2e}  m_p={_d[15]:.2e}  "
                      f"newt_F_local={_d[12]:.2e}  "
                      f"newt_F_global={_d[13]:.2e}",
                      flush=True)
                print(f"  [proj  t={t:6.3f}] div_max={proj_div_max:.2e}  "
                      f"n_iter={proj_n_iter}  dt={dt:.2e}",
                      flush=True)
            # Compute phi_flame for visualization (matches what consumers use)
            try:
                _phi_flame_snap = flame_front_3d.compute_phi_flame_from_state(
                    omega, state.T_g, state.Y_fuel,
                    grid.dx, grid.dy, grid.dz_arr,
                )
            except Exception:
                _phi_flame_snap = np.full(shape, np.nan, dtype=np.float64)
            # ── Phase 16 — per-cell bed-particle aggregation (snapshot) ──
            if lagrangian_bed_enable and _bed_buf is not None:
                _bp_count_per_cell  = np.zeros(shape, dtype=np.float32)
                _bp_m_solid_per_cell = np.zeros(shape, dtype=np.float32)
                _bp_m_water_per_cell = np.zeros(shape, dtype=np.float32)
                _bp_m_char_per_cell  = np.zeros(shape, dtype=np.float32)
                _bp_T_s_weighted = np.zeros(shape, dtype=np.float64)
                _bp_mass_weighted = np.zeros(shape, dtype=np.float64)
                _alive_arr = _bed_buf["alive"]
                _N = _alive_arr.shape[0]
                for _p in range(_N):
                    if _alive_arr[_p] == 0:
                        continue
                    _xi = int(_bed_buf["x"][_p] / grid.dx)
                    _yj = int(_bed_buf["y"][_p] / grid.dy)
                    if not (0 <= _xi < grid.Nx and 0 <= _yj < grid.Ny):
                        continue
                    # z-locate via cumulative z_face
                    _zz = _bed_buf["z"][_p]
                    _zk = -1
                    for _ki in range(grid.Nz):
                        if _zz < grid.z_face[_ki + 1]:
                            _zk = _ki
                            break
                    if _zk < 0:
                        continue
                    _m_s = _bed_buf["m_solid"][_p]
                    _m_w = _bed_buf["m_water"][_p]
                    _m_c = _bed_buf["m_char"][_p]
                    _T_s = _bed_buf["T_s"][_p]
                    _m_t = _m_s + _m_w + _m_c
                    _bp_count_per_cell[_zk, _yj, _xi]    += 1.0
                    _bp_m_solid_per_cell[_zk, _yj, _xi]  += _m_s
                    _bp_m_water_per_cell[_zk, _yj, _xi]  += _m_w
                    _bp_m_char_per_cell[_zk, _yj, _xi]   += _m_c
                    _bp_T_s_weighted[_zk, _yj, _xi]      += _T_s * _m_t
                    _bp_mass_weighted[_zk, _yj, _xi]     += _m_t
                _bp_T_s_avg = np.where(_bp_mass_weighted > 0.0,
                                        _bp_T_s_weighted / np.maximum(_bp_mass_weighted, 1e-30),
                                        0.0).astype(np.float32)
            else:
                _bp_count_per_cell    = np.zeros(shape, dtype=np.float32)
                _bp_m_solid_per_cell  = np.zeros(shape, dtype=np.float32)
                _bp_m_water_per_cell  = np.zeros(shape, dtype=np.float32)
                _bp_m_char_per_cell   = np.zeros(shape, dtype=np.float32)
                _bp_T_s_avg           = np.zeros(shape, dtype=np.float32)
            np.savez_compressed(
                _snap_path,
                t=np.float64(t), step=np.int64(_step),
                T_g=state.T_g.astype(np.float32),
                T_s=state.T_s.astype(np.float32),
                Y_fuel=state.Y_fuel.astype(np.float32),
                Y_O2=state.Y_O2.astype(np.float32),
                Y_H2O=state.Y_H2O.astype(np.float32),
                u=state.u.astype(np.float32),
                v=state.v.astype(np.float32),
                w=state.w.astype(np.float32),
                rho=state.rho.astype(np.float32),
                omega=omega.astype(np.float32),
                S_pyro=S_pyro.astype(np.float32),
                k_turb=k_turb.astype(np.float32),
                eps_turb=eps_turb.astype(np.float32),
                nu_t=nu_t.astype(np.float32),
                tau_mix=tau_mix.astype(np.float32),
                phi_bed=lset.phi.astype(np.float32) if grid.n_z_bed > 0 else np.zeros(shape, np.float32),
                phi_flame=_phi_flame_snap.astype(np.float32),
                front_x=np.float64(front_x[-1] if front_x else 0.0),
                # Phase 15F diagnostic — radiation forward flux + level-set
                # v_n drive, to localize the mesh-runaway in front advance.
                q_dom_fwd_3d=q_dom_fwd_3d.astype(np.float32),
                q_frankman_3d=q_frankman_3d.astype(np.float32),
                # Per-(y,x) integrated band heat flux and predicted v_n
                # (the field that drives level-set advance):
                q_in_at_front_2d=q_in_at_front_2d.astype(np.float32)
                    if q_in_at_front_2d is not None
                    else np.zeros((grid.Ny, grid.Nx), np.float32),
                # Phase 17b: v_n is now 3D; snapshot the z-mean as 2D for
                # backward compatibility with existing diagnostic scripts.
                v_n_2d=(v_n_field.mean(axis=0).astype(np.float32)
                        if v_n_field is not None
                        else np.zeros((grid.Ny, grid.Nx), np.float32)),
                # Phase 15O.1 — persistent per-cell tendril inventory
                # (zero everywhere when finney_tendril_t_contact_s = 0).
                ft_sink_M=_ft_sink_M.astype(np.float32),
                ft_sink_E=_ft_sink_E.astype(np.float32),
                ft_dep_M=_ft_dep_M.astype(np.float32),
                ft_dep_E=_ft_dep_E.astype(np.float32),
                ft_sink_t_rem=_ft_sink_t_rem.astype(np.float32),
                ft_dep_t_rem=_ft_dep_t_rem.astype(np.float32),
                # Phase 16 — per-cell aggregated bed-particle state (all
                # zero when lagrangian_bed_enable=False).
                bp_count=_bp_count_per_cell,
                bp_m_solid=_bp_m_solid_per_cell,
                bp_m_water=_bp_m_water_per_cell,
                bp_m_char=_bp_m_char_per_cell,
                bp_T_s_avg=_bp_T_s_avg,
                x_mid=grid.x_mid.astype(np.float32),
                z_mid=grid.z_mid.astype(np.float32),
                dx=np.float64(grid.dx), dy=np.float64(grid.dy),
                dz_arr=grid.dz_arr.astype(np.float32),
                wind_U=np.float64(wind_speed_m_s),
                closure=np.array([combustion_closure], dtype=object),
            )
            _snapshot_idx += 1

        # ── Exit conditions ──────────────────────────────────────────────
        if front_x[-1] > grid.Lx * 0.85:
            break  # fire reached domain end
        if (t - _last_advance_t > _stall_window_s
                and t > ignition_duration_s
                and not level_set_passive):
            # In passive-lset mode, front_x is intentionally locked at
            # the source patch — the level-set doesn't propagate
            # kinematically.  The CFD/bed-state front (T_s ≥ T_ign)
            # is the real signal but isn't tracked in front_x.  Skip
            # the stall break in passive mode.
            break  # fire stalled (no advance in 30s, post-ignition phase)

    # ── Compute ROS from front history (last 30s window, like 2D) ────────
    ros_m_s = _compute_steady_ros(front_t, front_x, t,
                                  source_x=n_src * grid.dx,
                                  domain_m=grid.Lx)
    n_cells_ignited = int(np.sum(state.T_s >= T_ign))

    # ── Kernel timing summary (Phase 14z) ────────────────────────────────
    _loop_total = _time.perf_counter() - _t_loop_start
    print(f"\n[timing] total loop wall: {_loop_total:.1f}s "
          f"({_step} outer steps)", flush=True)
    print(f"[timing] {'kernel':<28s}  {'cum_s':>9s}  {'calls':>7s}  "
          f"{'%loop':>6s}  {'µs/call':>9s}", flush=True)
    items = sorted(_timings.items(), key=lambda x: -x[1][0])
    accounted = 0.0
    for label, (cum, n_calls) in items:
        pct = 100.0 * cum / max(_loop_total, 1e-9)
        accounted += cum
        us_per_call = 1e6 * cum / max(n_calls, 1)
        print(f"[timing] {label:<28s}  {cum:>9.2f}  {n_calls:>7d}  "
              f"{pct:>5.1f}%  {us_per_call:>9.1f}", flush=True)
    other = _loop_total - accounted
    print(f"[timing] {'other (unprofiled)':<28s}  {other:>9.2f}  "
          f"{'-':>7s}  {100*other/_loop_total:>5.1f}%", flush=True)

    return Spread3DResult(
        ros_m_s=ros_m_s,
        n_cells_ignited=n_cells_ignited,
        front_t=front_t,
        front_x=front_x,
        grid=grid,
        state_final=state,
        k_turb_final=k_turb.copy(),
        eps_turb_final=eps_turb.copy(),
        nu_t_final=nu_t.copy(),
        diag_t=diag_t,
        diag_Tg_max=diag_Tg_max,
        diag_Ts_max=diag_Ts_max,
        diag_Sp_max=diag_Sp_max,
        diag_Qc_max=diag_Qc_max,
        diag_omega_max=diag_omega_max,
        diag_Y_max=diag_Y_max,
        diag_n_ign=diag_n_ign,
        diag_proj_div_max=diag_proj_div_max,
        diag_proj_n_iter=diag_proj_n_iter,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _apply_velocity_bcs(state: "CellState3D", u_inlet: np.ndarray,
                         y_bc: str, wall_function: bool = False) -> None:
    """No-op — all BCs enforced via on-the-fly ghost computation in operators
    (Way B per project_parallelism_architecture.md).  See
    feedback_no_cell_pinning.md: real cells must not be overwritten during
    a run.  Periodic-y handled via modular indexing inside each kernel.
    """
    pass


def _update_front_tracking(state: "CellState3D", grid: "Grid3D",
                            T_ign: float, t: float,
                            front_t: list, front_x: list,
                            append: bool = True) -> float:
    """Return the maximum x-position of any ignited bed cell.

    Phase 15F: ``append`` flag — when False, this routine becomes a
    pure diagnostic (no side effects on ``front_t`` / ``front_x``).
    With ``n_z_bed > 0`` the main loop drives ``front_x`` from the
    level-set only (mesh-stable per the Phase 15F bed-only ahead-band
    mask patch); the T_s-based "any cell ≥ T_ign" heuristic is left as
    a diagnostic value because it is mesh-runaway by construction
    (fine meshes get cells past T_ign sooner — each cell has less
    thermal mass per unit horizontal area).  Cold-flow runs
    (``n_z_bed == 0``) still need the T_s tracker as the only source
    so ``append=True`` is preserved there.
    """
    # Burning mask in bed cells only (k < n_z_bed).
    bed_burning = (state.T_s[:grid.n_z_bed] >= T_ign) & \
                  (state.alpha_s[:grid.n_z_bed] > 0.0)
    if not bed_burning.any():
        return front_x[-1] if front_x else 0.0
    # Project to (Nx,) array: any (k,j) burning at this i?
    # bed_burning shape: (n_z_bed, Ny, Nx) → any over (k,j) → (Nx,)
    col_burning = bed_burning.any(axis=(0, 1))
    # Highest i with col_burning True.
    burning_indices = np.where(col_burning)[0]
    if len(burning_indices) == 0:
        return front_x[-1] if front_x else 0.0
    i_max = int(burning_indices[-1])
    x_front = float(grid.x_mid[i_max])
    if append and x_front > (front_x[-1] if front_x else 0.0):
        front_t.append(t)
        front_x.append(x_front)
    return x_front


def _estimate_ros(front_t: list, front_x: list) -> float:
    """Estimate current ROS from front history (last 5 points or all)."""
    if len(front_t) < 2:
        return 0.0
    n = min(len(front_t), 5)
    dt_window = front_t[-1] - front_t[-n]
    dx_window = front_x[-1] - front_x[-n]
    if dt_window <= 0:
        return 0.0
    return dx_window / dt_window


def _compute_steady_ros(front_t: list, front_x: list, t_end: float,
                         source_x: float, domain_m: float) -> float:
    """Compute steady-state ROS using sliding 30s window with stall reject.

    Same logic as 2D run_2d_momentum_spread post-loop ROS calc:
    - Reject if advance from source < 5 cells
    - Reject if last advance was >30s ago AND fire didn't reach domain end
    - Otherwise use last 30s of history slope
    """
    if len(front_t) < 2:
        return 0.0
    advance_m = front_x[-1] - source_x
    if advance_m < 0.1:   # < 0.1m past source
        return 0.0
    # Stall check: last advance recent enough OR fire reached far?
    t_since_advance = t_end - front_t[-1]
    reached_far = front_x[-1] > domain_m * 0.7
    if t_since_advance > 30.0 and not reached_far:
        return 0.0
    # Phase 17b ROS fix (2026-06-22): use full-sim slope from initial
    # source position to current front position over total elapsed time
    # (t_end - front_t[0]).  Previously this used (ft[-1] - ft[idx]),
    # which gave a SHORT window when the front stalled early — e.g., for
    # a wet bed that stalled at t=1.7s of a 3s sim, the denominator was
    # 1.7s instead of 3s, over-estimating ROS by ~75%.  The full-sim
    # slope matches the snapshot-based ROS_lset_snap metric used by the
    # validation workers.
    ft = np.asarray(front_t)
    fx = np.asarray(front_x)
    t0 = float(ft[0])
    if t_end > t0:
        return float((fx[-1] - fx[0]) / (t_end - t0))
    return 0.0


def _adv_gas_energy(T_g: np.ndarray, u: np.ndarray, v: np.ndarray,
                     w: np.ndarray, dt: float,
                     dx: float, dy: float,
                     dz_arr: np.ndarray,
                     d_face_above: np.ndarray,
                     d_face_below: np.ndarray,
                     alpha_th: float, T_amb: float,
                     # Phase 23 Refactor 2C: z-min inlet ghost for cup burner.
                     # Default None → pre-Phase-23 zero-flux wall.
                     T_inlet_zmin: np.ndarray = None,
                     z_min_inlet_active: bool = False) -> None:
    """Advect T_g (gas temperature) in place via MUSCL flux differencing
    (Phase 14k — minmod-limited 2nd-order, replacing 1st-order upwind)
    + central diffusion (FV form for non-uniform dz).

    ∂T/∂t + u·∇T = α_th ∇²T

    No source terms; sources (Q_comb, q_conv) handled by coupling_3d.
    Boundary cells (i,j,k = 0 or N-1) are not updated.
    """
    from model_outdoor.physics_3d.muscl_3d import advect_3d_scalar_muscl

    # z-direction reshapes for broadcasting (diffusion only)
    inv_d_below = (1.0 / d_face_below[1:-1]).reshape(-1, 1, 1)
    inv_d_above = (1.0 / d_face_above[1:-1]).reshape(-1, 1, 1)
    inv_dz_int  = (1.0 / dz_arr[1:-1]).reshape(-1, 1, 1)

    dT = np.zeros_like(T_g)
    # MUSCL advection (replaces 1st-order upwind in x, y, z).  Phase 14v-bc:
    # phi_inlet=T_amb (cold air at inlet face).  Phase 23 Refactor 2C: z-min
    # inlet ghost activated for cup burner via T_inlet_zmin / z_min_inlet_active.
    _phi_inlet_zmin = T_inlet_zmin if T_inlet_zmin is not None \
                                   else np.zeros((1, 1))
    advect_3d_scalar_muscl(T_g, u, v, w, dx, dy, d_face_above, d_face_below, dT,
                           phi_inlet=T_amb,
                           phi_inlet_zmin=_phi_inlet_zmin,
                           z_min_inlet_active=z_min_inlet_active)
    # Diffusion (central, FV form).  Phase 14v-bc Way B: x-inlet ghost=T_amb,
    # x-outlet zero-grad, y periodic, z wall/top zero-flux Neumann (ghost=self).
    inv_dx2 = 1.0 / (dx * dx)
    # x interior i=1..Nx-2
    dT[:, :, 1:-1] += alpha_th * (T_g[:, :, 2:] - 2.0 * T_g[:, :, 1:-1] + T_g[:, :, :-2]) * inv_dx2
    # x inlet i=0: ghost = T_amb at x=-0.5dx
    dT[:, :, 0] += alpha_th * (T_g[:, :, 1] - 2.0 * T_g[:, :, 0] + T_amb) * inv_dx2
    # x outlet i=Nx-1: zero-grad ghost = self  ⇒  (T[Nx-2] - T[Nx-1]) only
    dT[:, :, -1] += alpha_th * (T_g[:, :, -2] - T_g[:, :, -1]) * inv_dx2
    # y (periodic via np.roll)
    dT += alpha_th * (np.roll(T_g, -1, axis=1) - 2.0 * T_g + np.roll(T_g, 1, axis=1)) / (dy * dy)
    # z interior k=1..Nz-2 (FV form)
    dT[1:-1, :, :] += alpha_th * (
        (T_g[2:, :, :] - T_g[1:-1, :, :]) * inv_d_above
        - (T_g[1:-1, :, :] - T_g[:-2, :, :]) * inv_d_below
    ) * inv_dz_int
    # z k=0: zero-flux wall by default (pre-Phase-23); cup-burner z-min
    # inlet supplies T_inlet_zmin ghost when active.
    if z_min_inlet_active and T_inlet_zmin is not None:
        dT[0, :, :] += alpha_th * (
            (T_g[1, :, :] - T_g[0, :, :]) / d_face_above[0]
            - (T_g[0, :, :] - T_inlet_zmin) / d_face_below[0]
        ) / dz_arr[0]
    else:
        dT[0, :, :] += alpha_th * ((T_g[1, :, :] - T_g[0, :, :]) / d_face_above[0]) / dz_arr[0]
    # z top k=Nz-1: zero-grad ghost ⇒ only below-flux contributes (negated)
    dT[-1, :, :] += alpha_th * (-(T_g[-1, :, :] - T_g[-2, :, :]) / d_face_below[-1]) / dz_arr[-1]

    T_g += dT * dt
    # Floor at ambient and numerical safety cap (10K K — matches the
    # main-loop T_FLAME_AD numerical safety; only catches runaway,
    # never binds in real physics).
    np.maximum(T_g, T_amb, out=T_g)
    np.minimum(T_g, 10_000.0, out=T_g)
