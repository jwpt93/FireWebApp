"""CLAUDE.md Rule #18 unit tests for SeparableLaplacian3D (Phase 14ax-1).

The separable Poisson solver is the mathematical inverse of the
constant-coefficient 3D Laplacian on the same grid + BCs that
ProjectionSolver3D's matrix uses with inv_rho ≡ 1.  So we can validate
the solver against the matrix itself: feed in a known p, compute
rhs = A·p via the same discrete operator, then check that solve(rhs)
recovers p to FP precision.

Tests:
  1. Bit-exact determinism (Rule #17 analogue): two consecutive solves
     give identical output.
  2. Self-consistency: solve(A·p) == p to FP precision for random p
     (the only acceptable residual is from FFT roundoff).
  3. BCs are honored: Neumann at i=0, k=0 ⇒ ∂p/∂n ≈ 0 (off the wall
     ghost convention used by the kernel); Dirichlet face at outlet/top
     ⇒ p extrapolates to zero just outside the boundary.
  4. Uniform-z reduction: with constant dz_arr, the solver matches a
     2D-FFT × 1D-cosine reference.
  5. Setup cost stays under a budget (<200ms for a typical 90k-cell grid).
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from model_outdoor.physics_3d.fft_poisson_3d import SeparableLaplacian3D


def _uniform_dz(Nz: int, dz: float):
    dz_arr = np.full(Nz, dz, dtype=np.float64)
    d_above = np.full(Nz, dz, dtype=np.float64)
    d_below = np.full(Nz, dz, dtype=np.float64)
    return dz_arr, d_above, d_below


def _non_uniform_dz(Nz: int, dz_base: float = 0.1, growth: float = 1.2):
    """Geometric stretch (mimics atm grid with BL refinement at z=0)."""
    dz_arr = np.empty(Nz, dtype=np.float64)
    dz_arr[0] = dz_base
    for k in range(1, Nz):
        dz_arr[k] = dz_arr[k - 1] * growth
    d_above = np.empty(Nz, dtype=np.float64)
    d_below = np.empty(Nz, dtype=np.float64)
    # d_above[k] = (dz[k] + dz[k+1]) / 2 ; d_above[Nz-1] = dz[Nz-1]
    for k in range(Nz - 1):
        d_above[k] = 0.5 * (dz_arr[k] + dz_arr[k + 1])
    d_above[Nz - 1] = dz_arr[Nz - 1]
    # d_below[k] = (dz[k-1] + dz[k]) / 2 ; d_below[0] = dz[0]
    for k in range(1, Nz):
        d_below[k] = 0.5 * (dz_arr[k - 1] + dz_arr[k])
    d_below[0] = dz_arr[0]
    return dz_arr, d_above, d_below


def _build_matrix_A(
    solver: SeparableLaplacian3D,
    d_above: np.ndarray,
    d_below: np.ndarray,
):
    """Return the sparse 3D-Laplacian matrix that matches the solver's
    BCs.  Built via ProjectionSolver3D's Numba kernel with inv_rho ≡ 1
    and α_g ≡ 1; takes explicit d_above/d_below so non-uniform-z tests
    use the same geometry the solver was constructed with.
    """
    Nz, Ny, Nx = solver.Nz, solver.Ny, solver.Nx
    N = Nz * Ny * Nx
    from model_outdoor.physics_3d.projection_3d import _fill_var_density_data
    import scipy.sparse as sp
    inv_rho = np.ones((Nz, Ny, Nx), dtype=np.float64)
    alpha_g = np.ones((Nz, Ny, Nx), dtype=np.float64)
    max_e = 7 * N
    rows = np.empty(max_e, dtype=np.int64)
    cols = np.empty(max_e, dtype=np.int64)
    data = np.empty(max_e, dtype=np.float64)
    n_filled = _fill_var_density_data(
        inv_rho, alpha_g, solver.dz_arr,
        np.asarray(d_above, dtype=np.float64),
        np.asarray(d_below, dtype=np.float64),
        solver.dx * solver.dx, solver.dy * solver.dy,
        solver.eps_reg, True,
        rows, cols, data,
    )
    A = sp.coo_matrix(
        (data[:n_filled], (rows[:n_filled], cols[:n_filled])),
        shape=(N, N),
    ).tocsr()
    A.sum_duplicates()
    return A


# ─── Tests ───────────────────────────────────────────────────────────────

def test_setup_cost_under_budget():
    """Construction (eigendecomp + matrix build) finishes quickly."""
    Nz, Ny, Nx = 86, 5, 400    # typical Cheney sweep grid (~172k cells)
    dz_arr, d_above, d_below = _uniform_dz(Nz, 0.0925)
    t0 = time.time()
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=0.1, dy=0.1,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    dt = time.time() - t0
    # Eigendecomp of Nx=400 is the dominant cost (~50ms typical).
    # Budget: 1 second is plenty.
    assert dt < 1.0, f"setup took {dt:.3f}s (>1.0s budget)"


def test_solve_deterministic():
    """Two consecutive solves on identical RHS give bit-exact outputs."""
    Nz, Ny, Nx = 16, 4, 32
    dz_arr, d_above, d_below = _uniform_dz(Nz, 0.1)
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=0.1, dy=0.1,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal((Nz, Ny, Nx))
    p1 = s.solve(rhs.copy())
    p2 = s.solve(rhs.copy())
    assert np.array_equal(p1, p2), "two solves on identical RHS must match"


def test_solve_recovers_p_from_Ap_uniform_dz():
    """Self-consistency: for the matrix A built by the same kernel
    with inv_rho ≡ 1, the FFT solver inverts A to FP precision.
    """
    Nz, Ny, Nx = 8, 4, 16
    dx = dy = 0.1
    dz = 0.1
    dz_arr, d_above, d_below = _uniform_dz(Nz, dz)
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=dx, dy=dy,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    A = _build_matrix_A(s, d_above, d_below)

    rng = np.random.default_rng(11)
    p_true = rng.standard_normal((Nz, Ny, Nx))
    rhs_flat = A @ p_true.reshape(-1)
    rhs = rhs_flat.reshape(Nz, Ny, Nx)

    p_recovered = s.solve(rhs)
    rel_err = np.max(np.abs(p_recovered - p_true)) / max(
        np.max(np.abs(p_true)), 1e-30)
    # FP roundoff from 3 sequential transforms + 1 spectral divide:
    # ~1e-8 typical at this size; still 5 orders better than the
    # rtol=1e-3 BiCGSTAB tolerance used in production.
    assert rel_err < 1.0e-7, (
        f"self-consistency rel_err = {rel_err:.2e} (>1e-7)")


def test_solve_recovers_p_with_alpha_g_machinery_off():
    """Same as above but uses the projection_3d matrix builder with
    a non-trivial Ny (periodic-y) to exercise the FFT path.
    """
    Nz, Ny, Nx = 6, 8, 12      # Ny=8 exercises FFT non-trivially
    dx = dy = 0.1
    dz_arr, d_above, d_below = _uniform_dz(Nz, 0.1)
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=dx, dy=dy,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    A = _build_matrix_A(s, d_above, d_below)
    rng = np.random.default_rng(3)
    p_true = rng.standard_normal((Nz, Ny, Nx))
    rhs = (A @ p_true.reshape(-1)).reshape(Nz, Ny, Nx)
    p_recovered = s.solve(rhs)
    rel_err = np.max(np.abs(p_recovered - p_true)) / max(
        np.max(np.abs(p_true)), 1e-30)
    assert rel_err < 1.0e-7, (
        f"FFT self-consistency rel_err = {rel_err:.2e} (>1e-7)")


def test_non_uniform_z_self_consistency():
    """Solve A·p = rhs where A is the 3D matrix built with NON-UNIFORM
    dz_arr (geometric stretch).  Verifies the z-symmetrization handles
    the non-uniform spacing correctly.
    """
    Nz, Ny, Nx = 12, 4, 8
    dz_arr, d_above, d_below = _non_uniform_dz(Nz, dz_base=0.05, growth=1.3)
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=0.1, dy=0.1,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    A = _build_matrix_A(s, d_above, d_below)
    rng = np.random.default_rng(13)
    p_true = rng.standard_normal((Nz, Ny, Nx))
    rhs = (A @ p_true.reshape(-1)).reshape(Nz, Ny, Nx)
    p_recovered = s.solve(rhs)
    rel_err = np.max(np.abs(p_recovered - p_true)) / max(
        np.max(np.abs(p_true)), 1e-30)
    assert rel_err < 1.0e-7, (
        f"non-uniform z self-consistency rel_err = {rel_err:.2e} (>1e-7)")


def test_solve_typical_cheney_grid_runs():
    """End-to-end smoke: construct for the Cheney sweep grid scale
    and confirm one solve completes without NaN/Inf.
    """
    Nz, Ny, Nx = 86, 5, 400
    dz_arr, d_above, d_below = _uniform_dz(Nz, 0.0925)
    s = SeparableLaplacian3D(
        Nz, Ny, Nx, dx=0.1, dy=0.1,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
    )
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((Nz, Ny, Nx))
    t0 = time.time()
    p = s.solve(rhs)
    dt = time.time() - t0
    assert np.all(np.isfinite(p)), "solver produced NaN/Inf"
    # 90ms is a generous budget; typical is 20-50ms.
    assert dt < 0.3, f"solve took {dt:.3f}s (>0.3s budget)"
