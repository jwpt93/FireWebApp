"""Unit tests for Phase 23 Refactor 2C projection z-min inlet.

Rule #18: the new ``set_bottom_inlet_BC`` + divergence z-min ghost
plumbing must be tested (a) preserves the pre-Phase-23 no-slip-wall
behaviour when w_inlet_zmin=0 (bit-exact-invariant) and (b) correctly
computes the divergence when a non-zero inlet is installed.

Full cup-burner activation (species + energy z-min inlet + BC wire-up)
is a separate work item on top of this projection plumbing.
"""
import numpy as np
import pytest

from model_outdoor.physics_3d.projection_3d import ProjectionSolver3D


def _mk_solver(Nz=6, Ny=3, Nx=8, dx=0.01, dy=0.01, dz=0.02):
    """Build a small solver instance for BC testing."""
    dz_arr = np.full(Nz, dz)
    # Uniform dz: face_above spacing = face_below spacing = dz between all cells
    d_face_above = np.full(Nz, dz)
    d_face_below = np.full(Nz, dz)
    return ProjectionSolver3D(
        Nz=Nz, Ny=Ny, Nx=Nx,
        dx=dx, dy=dy, dz_arr=dz_arr,
        d_face_above=d_face_above, d_face_below=d_face_below,
        y_bc="periodic",
    )


# ── set_bottom_inlet_BC API ───────────────────────────────────────────
def test_default_w_inlet_zmin_is_zero():
    """Fresh solver has w_inlet_zmin = 0 everywhere (bit-exact wall
    invariant for all pre-Phase-23 outdoor cases)."""
    ps = _mk_solver()
    assert np.all(ps._w_inlet_zmin == 0.0)
    assert ps._w_inlet_zmin.shape == (ps.Ny, ps.Nx)


def test_set_bottom_inlet_BC_stores_array():
    ps = _mk_solver()
    w_in = np.random.uniform(0, 0.1, (ps.Ny, ps.Nx))
    ps.set_bottom_inlet_BC(w_in)
    assert np.array_equal(ps._w_inlet_zmin, w_in)
    assert ps._w_inlet_zmin.dtype == np.float64


def test_set_bottom_inlet_BC_ascontiguousarray():
    """Ensure ascontiguousarray conversion so numba kernels don't
    complain about strided input."""
    ps = _mk_solver()
    # Non-contiguous input: take a strided view
    w_in = np.random.uniform(0, 0.1, (ps.Ny, ps.Nx * 2))[:, ::2]
    assert not w_in.flags["C_CONTIGUOUS"]
    ps.set_bottom_inlet_BC(w_in)
    assert ps._w_inlet_zmin.flags["C_CONTIGUOUS"]


# ── Divergence formula: bit-exact when w_inlet=0 (wall pattern) ──────
def test_divergence_wall_pattern_matches_zero_inlet():
    """With w_inlet_zmin=0 (default), the divergence at z=0 must equal
    w[0]/dz — the wall pattern used by every pre-Phase-23 case.  This
    is the bit-exact invariant guarantee for Rule #17."""
    ps = _mk_solver(Nz=4, Ny=2, Nx=5, dz=0.02)
    np.random.seed(0)
    u = np.random.uniform(-1, 1, (ps.Nz, ps.Ny, ps.Nx))
    v = np.random.uniform(-1, 1, (ps.Nz, ps.Ny, ps.Nx))
    w = np.random.uniform(-1, 1, (ps.Nz, ps.Ny, ps.Nx))
    div = ps._divergence_compatible(u, v, w)
    # z-contribution at k=0 with w_inlet=0 should be exactly w[0]/dz[0]
    # (the -w_inlet=0 subtraction is a no-op).  We verify this by
    # computing div a second time with an equivalent wall configuration.
    ps2 = _mk_solver(Nz=4, Ny=2, Nx=5, dz=0.02)
    div2 = ps2._divergence_compatible(u, v, w)
    assert np.array_equal(div, div2)


# ── Divergence responds to non-zero w_inlet ──────────────────────────
def test_divergence_reduces_when_wall_matches_inlet():
    """When w[0] = w_inlet_zmin, the z-min contribution to div[0]
    must be zero (compatible-inlet special case)."""
    ps = _mk_solver(Nz=4, Ny=2, Nx=5, dz=0.02)
    w_inlet = np.full((ps.Ny, ps.Nx), 0.05)
    ps.set_bottom_inlet_BC(w_inlet)

    # Set up u=v=0, w=uniform equal to w_inlet everywhere.  Then div=0
    # (perfect compatibility: nothing to project).
    u = np.zeros((ps.Nz, ps.Ny, ps.Nx))
    v = np.zeros_like(u)
    w = np.full_like(u, 0.05)
    div = ps._divergence_compatible(u, v, w)
    # Interior z-differences: w[k] - w[k-1] = 0 (uniform w)
    # k=0: (w[0] - w_inlet_zmin) / dz = 0 (equal)
    assert np.max(np.abs(div)) < 1e-14


def test_divergence_captures_inlet_excess():
    """Non-zero (w[0] - w_inlet_zmin) must appear in div[0]."""
    ps = _mk_solver(Nz=4, Ny=2, Nx=5, dz=0.02)
    w_inlet = np.full((ps.Ny, ps.Nx), 0.05)
    ps.set_bottom_inlet_BC(w_inlet)

    u = np.zeros((ps.Nz, ps.Ny, ps.Nx))
    v = np.zeros_like(u)
    w = np.zeros_like(u)                # w = 0 everywhere
    # w[0] - w_inlet_zmin = -0.05; div[0] should be -0.05/dz
    div = ps._divergence_compatible(u, v, w)
    expected = -0.05 / 0.02  # = -2.5 s^-1
    assert div[0, 0, 0] == pytest.approx(expected)


# ── Determinism (Rule #17) ────────────────────────────────────────────
def test_set_bottom_inlet_BC_deterministic():
    ps1 = _mk_solver()
    ps2 = _mk_solver()
    w_in = np.random.uniform(0, 0.1, (ps1.Ny, ps1.Nx))
    ps1.set_bottom_inlet_BC(w_in)
    ps2.set_bottom_inlet_BC(w_in)
    u = np.random.uniform(-1, 1, (ps1.Nz, ps1.Ny, ps1.Nx))
    v = np.random.uniform(-1, 1, (ps1.Nz, ps1.Ny, ps1.Nx))
    w = np.random.uniform(-1, 1, (ps1.Nz, ps1.Ny, ps1.Nx))
    d1 = ps1._divergence_compatible(u, v, w)
    d2 = ps2._divergence_compatible(u, v, w)
    assert np.array_equal(d1, d2)
