"""Unit tests for Phase 23 Refactor 2B: CupBurnerBC geometry skeleton.

Rule #18 (unit tests required for every new module): CupBurnerBC's
geometry math (fuel-jet/coflow/wall masks) + coflow-composition
computation + registry entry all need tests.

configure() intentionally raises NotImplementedError (Refactor 2C
will land the projection-solver z-min inlet plumbing).  This test
file validates every piece that IS ready.
"""
import numpy as np
import pytest

from model_outdoor.boundary_conditions import (
    available, get_bc_class, CupBurnerBC, BoundaryCondition,
)


class _MiniGrid:
    """Grid stub with Nx, Ny + evenly spaced x_mid centered on Lx/2."""
    def __init__(self, Lx=0.10, Nx=50, Ly=0.005, Ny=1):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        dx = Lx / Nx
        dy = Ly / Ny
        self.x_mid = np.arange(Nx) * dx + 0.5 * dx
        self.y_mid = np.arange(Ny) * dy + 0.5 * dy


# ── Registry ─────────────────────────────────────────────────────────
def test_registry_contains_cup_burner():
    assert "cup_burner" in available()


def test_get_bc_class_returns_cup_burner():
    cls = get_bc_class("cup_burner")
    assert cls is CupBurnerBC
    assert issubclass(cls, BoundaryCondition)
    assert cls.kind == "cup_burner"


# ── Instantiation + kwarg absorbing ───────────────────────────────────
def test_default_construction():
    bc = CupBurnerBC()
    assert bc.fuel_jet_radius_m == 0.014
    assert bc.chimney_radius_m == 0.0425
    assert bc.Y_agent_coflow == 0.0


def test_absorbs_outdoor_kwargs():
    """spread_3d.py passes outdoor-style kwargs at BC construction; the
    cup_burner class must silently absorb them so registry-based
    dispatch stays uniform."""
    bc = CupBurnerBC(
        wind_speed_m_s=4.0,       # outdoor kwarg
        wind_profile_type="log_law",
        h_bed=0.5,
        sigma_sav=2000.0,
        alpha_s_avg=0.001,
    )
    assert bc.fuel_jet_radius_m == 0.014


# ── Geometry masks ────────────────────────────────────────────────────
def test_fuel_mask_covers_expected_radius():
    """Fuel mask should cover cells within R_fuel of the centerline."""
    grid = _MiniGrid(Lx=0.10, Nx=100)   # dx = 1 mm
    bc = CupBurnerBC(fuel_jet_radius_m=0.014)
    bc._build_masks(grid)
    x_c = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
    r_x = np.abs(grid.x_mid - x_c)
    expected = r_x <= 0.014
    # Compare against the y=0 row of the (Ny, Nx) fuel_mask
    assert np.array_equal(bc.fuel_mask[0, :], expected)


def test_coflow_mask_covers_annulus():
    grid = _MiniGrid(Lx=0.10, Nx=100)
    bc = CupBurnerBC(fuel_jet_radius_m=0.014, chimney_radius_m=0.0425)
    bc._build_masks(grid)
    x_c = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
    r_x = np.abs(grid.x_mid - x_c)
    expected = (r_x > 0.014) & (r_x <= 0.0425)
    assert np.array_equal(bc.coflow_mask[0, :], expected)


def test_wall_mask_covers_outside_chimney():
    grid = _MiniGrid(Lx=0.20, Nx=100)   # Lx > 2*chimney_radius
    bc = CupBurnerBC(chimney_radius_m=0.0425)
    bc._build_masks(grid)
    x_c = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
    r_x = np.abs(grid.x_mid - x_c)
    expected = r_x > 0.0425
    assert np.array_equal(bc.wall_mask[0, :], expected)


def test_masks_are_disjoint_and_complete():
    """Every cell is in exactly one of fuel, coflow, wall (partition)."""
    grid = _MiniGrid(Lx=0.20, Nx=100, Ny=5)
    bc = CupBurnerBC()
    bc._build_masks(grid)
    total = (bc.fuel_mask.astype(int) +
             bc.coflow_mask.astype(int) +
             bc.wall_mask.astype(int))
    assert np.all(total == 1), "masks must partition the z=0 face"


# ── Axisymmetric-analog (Ny > 1) mask geometry (Refactor 2D-3D) ─────
def test_axisymmetric_masks_use_2d_radial_distance():
    """In 3D (Ny > 1), fuel_mask must be a DISC centered at (Lx/2, Ly/2)."""
    grid = _MiniGrid(Lx=0.100, Nx=50, Ly=0.100, Ny=50)   # square 3D
    bc = CupBurnerBC(fuel_jet_radius_m=0.014,
                     chimney_radius_m=0.0425)
    bc._build_masks(grid)
    x_c = 0.5 * (grid.x_mid[0] + grid.x_mid[-1])
    y_c = 0.5 * (grid.y_mid[0] + grid.y_mid[-1])
    xx, yy = np.meshgrid(grid.x_mid, grid.y_mid, indexing="xy")
    r = np.sqrt((xx - x_c)**2 + (yy - y_c)**2)
    assert np.array_equal(bc.fuel_mask,   r <= 0.014)
    assert np.array_equal(bc.coflow_mask, (r > 0.014) & (r <= 0.0425))
    assert np.array_equal(bc.wall_mask,   r > 0.0425)


def test_axisymmetric_partition_holds():
    grid = _MiniGrid(Lx=0.10, Nx=25, Ly=0.10, Ny=25)
    bc = CupBurnerBC()
    bc._build_masks(grid)
    total = bc.fuel_mask.astype(int) + bc.coflow_mask.astype(int) + \
            bc.wall_mask.astype(int)
    assert np.all(total == 1)


# ── w_inlet at z=0 ────────────────────────────────────────────────────
def test_w_inlet_matches_masks():
    grid = _MiniGrid(Lx=0.20, Nx=100, Ny=3)
    bc = CupBurnerBC(fuel_jet_velocity_m_s=0.06, coflow_velocity_m_s=0.10)
    w = bc._build_w_inlet_zmin(grid)
    assert w.shape == (grid.Ny, grid.Nx)
    assert np.all(w[bc.fuel_mask] == 0.06)
    assert np.all(w[bc.coflow_mask] == 0.10)
    assert np.all(w[bc.wall_mask] == 0.0)


def test_w_inlet_cached_on_bc():
    grid = _MiniGrid()
    bc = CupBurnerBC()
    assert bc.w_inlet_zmin is None
    w = bc._build_w_inlet_zmin(grid)
    assert bc.w_inlet_zmin is w


# ── Coflow O2/agent composition ──────────────────────────────────────
def test_effective_Y_O2_pure_air():
    """No agent → coflow O2 unchanged from baseline."""
    bc = CupBurnerBC(Y_agent_coflow=0.0)
    assert bc.effective_Y_O2_coflow() == pytest.approx(0.232)


def test_effective_Y_O2_full_dilution():
    """Y_agent = Y_N2_baseline → coflow O2 drops to zero."""
    bc = CupBurnerBC(Y_O2_coflow=0.232, Y_agent_coflow=1.0 - 0.232)
    assert bc.effective_Y_O2_coflow() == pytest.approx(0.0)


def test_effective_Y_O2_partial_dilution():
    """Y_agent halfway → coflow O2 halved."""
    Y_N2_baseline = 1.0 - 0.232
    bc = CupBurnerBC(Y_O2_coflow=0.232,
                     Y_agent_coflow=0.5 * Y_N2_baseline)
    assert bc.effective_Y_O2_coflow() == pytest.approx(0.116)


# ── configure() active (Refactor 2C landed) ─────────────────────────
class _MiniProjSolver:
    """Duck-typed projection solver stub — records the w_inlet_zmin
    that was installed so we can inspect it."""
    def __init__(self):
        self.installed_w = None
    def set_bottom_inlet_BC(self, w):
        self.installed_w = np.asarray(w).copy()


class _MiniState:
    def __init__(self, Nz, Ny, Nx):
        self.T_g    = np.full((Nz, Ny, Nx), 300.0)
        self.rho    = np.full((Nz, Ny, Nx), 1.19)   # ambient air
        self.Y_fuel = np.zeros((Nz, Ny, Nx))
        self.Y_O2   = np.full((Nz, Ny, Nx), 0.232)


def _mk_full_grid(Lx=0.20, Nx=40, Ly=0.005, Ny=1, Lz=0.30, Nz=30):
    """Grid stub with dz_arr + z_mid + x_mid populated."""
    grid = _MiniGrid(Lx=Lx, Nx=Nx, Ly=Ly, Ny=Ny)
    dz = Lz / Nz
    grid.Nz = Nz
    grid.dz_arr = np.full(Nz, dz)
    grid.z_mid = np.arange(Nz) * dz + 0.5 * dz
    return grid


def test_configure_installs_w_inlet_into_projection():
    grid = _mk_full_grid()
    ps = _MiniProjSolver()
    state = _MiniState(grid.Nz, grid.Ny, grid.Nx)
    bc = CupBurnerBC()
    bc.configure(proj_solver=ps, grid=grid, state=state)
    assert ps.installed_w is not None
    assert ps.installed_w.shape == (grid.Ny, grid.Nx)
    # fuel cells should have U_fuel, coflow cells U_coflow
    assert np.all(ps.installed_w[bc.fuel_mask] == 0.06)
    assert np.all(ps.installed_w[bc.coflow_mask] == 0.10)
    assert np.all(ps.installed_w[bc.wall_mask] == 0.0)


def test_configure_populates_species_and_temp_ghosts():
    grid = _mk_full_grid()
    ps = _MiniProjSolver()
    state = _MiniState(grid.Nz, grid.Ny, grid.Nx)
    bc = CupBurnerBC(Y_agent_coflow=0.10)
    bc.configure(proj_solver=ps, grid=grid, state=state)

    # Y_F: 1 in fuel cells, 0 elsewhere
    assert np.all(bc.Y_F_inlet_zmin[bc.fuel_mask] == 1.0)
    assert np.all(bc.Y_F_inlet_zmin[bc.coflow_mask] == 0.0)
    # Y_O2: 0 in fuel cells, effective (< baseline 0.232) in coflow
    assert np.all(bc.Y_O2_inlet_zmin[bc.fuel_mask] == 0.0)
    expected_Y_O2 = bc.effective_Y_O2_coflow()
    assert np.all(bc.Y_O2_inlet_zmin[bc.coflow_mask] == expected_Y_O2)
    # T_inlet uniform 298 K
    assert np.all(bc.T_inlet_zmin == 298.0)


def test_configure_seeds_hot_spot():
    """Ignition kernel: pre-mixed stoichiometric CH4-air pocket at
    1500 K spanning z ~ 3-9 mm above the cup rim, full chimney x."""
    grid = _mk_full_grid(Lz=0.30, Nz=100)   # dz = 3 mm
    ps = _MiniProjSolver()
    state = _MiniState(grid.Nz, grid.Ny, grid.Nx)
    bc = CupBurnerBC()
    bc.configure(proj_solver=ps, grid=grid, state=state)
    # Some cells at 1500 K
    hot_cells = np.sum(state.T_g == 1500.0)
    assert hot_cells > 0, "no ignition hot spot seeded"
    # Same cells should have stoichiometric premix (Y_F=0.055, Y_O2=0.220)
    hot_mask = state.T_g == 1500.0
    assert np.all(state.Y_fuel[hot_mask] == 0.055)
    assert np.all(state.Y_O2[hot_mask] == 0.220)
    # Density should be consistent (ρ = P/(R·T) ≈ 0.235 kg/m³ at 1500 K)
    assert np.all(np.isclose(state.rho[hot_mask], 101325.0/(287.0*1500.0),
                              rtol=1e-6))


def test_species_inlet_kwargs_for_fuel():
    grid = _mk_full_grid()
    ps = _MiniProjSolver()
    state = _MiniState(grid.Nz, grid.Ny, grid.Nx)
    bc = CupBurnerBC()
    bc.configure(proj_solver=ps, grid=grid, state=state)
    kw = bc.species_inlet_kwargs("Y_fuel")
    assert kw["z_min_inlet_active"] is True
    assert kw["Y_inlet_zmin"].shape == (grid.Ny, grid.Nx)


def test_species_inlet_kwargs_unknown_species_returns_wall():
    grid = _mk_full_grid()
    ps = _MiniProjSolver()
    state = _MiniState(grid.Nz, grid.Ny, grid.Nx)
    bc = CupBurnerBC()
    bc.configure(proj_solver=ps, grid=grid, state=state)
    kw = bc.species_inlet_kwargs("no_such_species")
    assert kw["z_min_inlet_active"] is False
