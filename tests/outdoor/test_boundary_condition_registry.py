"""Unit tests for Phase 23 Refactor 2A: boundary_conditions registry.

Rule #18 (unit tests required for every new module): the new
boundary_conditions subpackage requires tests covering the registry
mechanics + bit-exact preservation of the OutdoorWindBC wrap around
the pre-refactor inline u_inlet computation.

Bit-exact preservation of the full outdoor simulation is verified
separately by re-running Cheney Nat4_U4 and comparing ROS to the
pre-refactor number (31.900523522689642 m/min all 18 sig figs).
Here we test just the u_inlet-array level.
"""
import numpy as np
import pytest

from model_outdoor.boundary_conditions import (
    available, get_bc_class, OutdoorWindBC, BoundaryCondition,
)
from model_outdoor.boundary import (
    wind_profile_log_law, wind_profile_canopy_bl, raupach_d_z0,
)


class _MiniGrid:
    """Duck-typed grid stub used by build_u_inlet."""
    def __init__(self, Nz=20, Ny=5):
        self.Nz = Nz
        self.Ny = Ny
        # Simple uniform vertical spacing 0.05 m up to 1 m
        self.z_mid = np.linspace(0.025, 0.975, Nz)


# ── Registry ─────────────────────────────────────────────────────────
def test_registry_contains_outdoor_wind():
    assert "outdoor_wind" in available()


def test_get_bc_class_returns_outdoor_wind():
    cls = get_bc_class("outdoor_wind")
    assert cls is OutdoorWindBC
    assert issubclass(cls, BoundaryCondition)
    assert cls.kind == "outdoor_wind"


def test_get_bc_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown boundary_condition_kind"):
        get_bc_class("no_such_bc")


# ── OutdoorWindBC — bit-exact reproduction of pre-refactor code ─────
def test_outdoor_wind_log_law_matches_inline():
    """OutdoorWindBC.build_u_inlet() with log_law must exactly reproduce
    the inline computation that used to live in spread_3d.py lines
    1337-1344 (pre-Refactor 2A)."""
    grid = _MiniGrid(Nz=20, Ny=3)
    U_ref = 4.0

    bc = OutdoorWindBC(wind_speed_m_s=U_ref, wind_profile_type="log_law")
    u_class = bc.build_u_inlet(grid)

    # Reproduce the inline block verbatim
    Z_0_INLET = 0.01
    u_inline = np.zeros((grid.Nz, grid.Ny))
    for k in range(grid.Nz):
        z = grid.z_mid[k]
        u_inline[k, :] = wind_profile_log_law(
            z, U_ref, z_ref=10.0, z_0=Z_0_INLET,
        )
    assert np.array_equal(u_class, u_inline)


def test_outdoor_wind_canopy_bl_matches_inline():
    """Same bit-exact-invariant for the canopy_bl branch."""
    grid = _MiniGrid(Nz=15, Ny=4)
    U_ref = 4.0
    h_bed = 0.37
    sigma_sav = 2000.0
    alpha_s_avg = 0.001

    bc = OutdoorWindBC(
        wind_speed_m_s=U_ref, wind_profile_type="canopy_bl",
        h_bed=h_bed, sigma_sav=sigma_sav, alpha_s_avg=alpha_s_avg,
    )
    u_class = bc.build_u_inlet(grid)

    lambda_F_bed = 0.5 * sigma_sav * alpha_s_avg * h_bed
    d_canopy, z0_canopy = raupach_d_z0(h_bed, lambda_F_bed)
    u_inline = np.zeros((grid.Nz, grid.Ny))
    for k in range(grid.Nz):
        z = grid.z_mid[k]
        u_inline[k, :] = wind_profile_canopy_bl(
            z, U_ref, z_ref=10.0, h_canopy=h_bed,
            alpha_cionco=1.0,
            d_canopy=d_canopy, z_0_canopy=z0_canopy,
        )
    assert np.array_equal(u_class, u_inline)


def test_outdoor_wind_unknown_profile_raises():
    with pytest.raises(ValueError, match="wind_profile_type"):
        OutdoorWindBC(wind_speed_m_s=4.0, wind_profile_type="bogus")


def test_outdoor_wind_cache_populated_after_build():
    """After .build_u_inlet(), .u_inlet attribute should hold the array."""
    grid = _MiniGrid()
    bc = OutdoorWindBC(wind_speed_m_s=4.0)
    assert bc.u_inlet is None
    result = bc.build_u_inlet(grid)
    assert bc.u_inlet is result


def test_outdoor_wind_log_law_zero_at_z0():
    """Physical sanity: log-law goes to zero at z=0 (no-slip anchor)."""
    grid = _MiniGrid(Nz=20)
    grid.z_mid[0] = 0.0  # simulate a ground-hugging first cell
    bc = OutdoorWindBC(wind_speed_m_s=5.0)
    u = bc.build_u_inlet(grid)
    assert u[0, 0] == 0.0


def test_outdoor_wind_log_law_monotonic_in_z():
    """Physical sanity: log-law wind increases monotonically with z."""
    grid = _MiniGrid(Nz=20)
    bc = OutdoorWindBC(wind_speed_m_s=4.0)
    u = bc.build_u_inlet(grid)
    # u[k, 0] should be non-decreasing in k
    assert np.all(np.diff(u[:, 0]) >= 0.0)


# ── Determinism (Rule #17) ────────────────────────────────────────────
def test_build_u_inlet_bit_exact_across_calls():
    """Two back-to-back .build_u_inlet() calls with identical inputs
    must produce bit-exact identical arrays (deterministic)."""
    grid = _MiniGrid(Nz=25)
    bc = OutdoorWindBC(wind_speed_m_s=4.0, wind_profile_type="log_law")
    a = bc.build_u_inlet(grid)
    b = bc.build_u_inlet(grid)
    assert np.array_equal(a, b)
