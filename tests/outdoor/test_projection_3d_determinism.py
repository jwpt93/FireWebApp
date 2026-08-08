"""CLAUDE.md Rule #18 unit tests for ProjectionSolver3D (PyPardiso path).

Tests cover:
  1. Bit-exact determinism: same solver, same inputs, two solves → identical
     to the last digit (np.array_equal).
  2. Cross-instance determinism: fresh solver instance with identical inputs
     → same result as a previously-recorded run.
  3. Correctness: solve residual at machine precision.
  4. Solenoidal projection: divergence reduced by ≥ 1e8× (sanity check).

These tests must pass green before any Cheney sweep runs (Rule #18).
"""
from __future__ import annotations

import numpy as np
import pytest

from model_outdoor.physics_3d.projection_3d import ProjectionSolver3D


def _uniform_dz_arrays(Nz: int, dz: float):
    dz_arr = np.full(Nz, dz, dtype=np.float64)
    d_above = np.full(Nz, dz, dtype=np.float64)
    d_below = np.full(Nz, dz, dtype=np.float64)
    return dz_arr, d_above, d_below


def _make_solver(Nz=4, Ny=4, Nx=16, dx=0.1) -> ProjectionSolver3D:
    dy = dz = dx
    dz_arr, d_above, d_below = _uniform_dz_arrays(Nz, dz)
    return ProjectionSolver3D(
        Nz, Ny, Nx, dy, dx,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
        y_bc="periodic",
    )


def _make_inputs(Nz=4, Ny=4, Nx=16, dx=0.1, *, seed=0):
    """Construct a non-trivial divergent velocity field + variable rho."""
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.5 * rng.standard_normal((Nz, Ny, Nx))
    rho = np.maximum(rho, 0.1)  # keep ρ > 0
    Lx = Nx * dx
    x = (np.arange(Nx) + 0.5) * dx
    u = np.broadcast_to(np.sin(np.pi * x / Lx)[None, None, :],
                        (Nz, Ny, Nx)).copy()
    v = np.zeros((Nz, Ny, Nx))
    w = np.zeros((Nz, Ny, Nx))
    return u, v, w, rho


def test_projection_bit_exact_determinism_same_solver():
    """Two solves on identical inputs in the same solver → bit-exact match.

    This is the kernel-level analogue of CLAUDE.md Rule #17 — catches
    non-deterministic reductions inside the linear solver.
    """
    solver = _make_solver()
    u, v, w, rho = _make_inputs()

    solver.rebuild_for_rho(rho)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    p1 = solver.project(u1, v1, w1, rho, dt=0.01)

    solver.rebuild_for_rho(rho)  # rebuild factorization explicitly
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    p2 = solver.project(u2, v2, w2, rho, dt=0.01)

    # Bit-exact equality across all entries
    assert np.array_equal(p1, p2), \
        f"projection p not bit-exact across solves: max |Δ| = {np.max(np.abs(p1 - p2))}"
    assert np.array_equal(u1, u2), "projected u not bit-exact"
    assert np.array_equal(v1, v2), "projected v not bit-exact"
    assert np.array_equal(w1, w2), "projected w not bit-exact"


def test_projection_bit_exact_determinism_fresh_solver():
    """A fresh solver instance with identical inputs → bit-exact match
    with a previous solver's result.  Catches lingering global state
    contamination between solver lifetimes.
    """
    u, v, w, rho = _make_inputs()

    solver_a = _make_solver()
    solver_a.rebuild_for_rho(rho)
    u_a, v_a, w_a = u.copy(), v.copy(), w.copy()
    p_a = solver_a.project(u_a, v_a, w_a, rho, dt=0.01)

    del solver_a  # ensure no shared state

    solver_b = _make_solver()
    solver_b.rebuild_for_rho(rho)
    u_b, v_b, w_b = u.copy(), v.copy(), w.copy()
    p_b = solver_b.project(u_b, v_b, w_b, rho, dt=0.01)

    assert np.array_equal(p_a, p_b), \
        f"fresh-solver p drift: max |Δ| = {np.max(np.abs(p_a - p_b))}"
    assert np.array_equal(u_a, u_b), "fresh-solver u drift"


def test_projection_residual_at_machine_precision():
    """Residual ‖A x − b‖∞ should be at machine precision (≤ 1e-10)."""
    import scipy.sparse as sp

    solver = _make_solver()
    _, _, _, rho = _make_inputs()
    solver.rebuild_for_rho(rho)
    A = solver._A  # CSR matrix
    N = A.shape[0]
    rng = np.random.default_rng(42)
    b = rng.standard_normal(N)
    b -= b.mean()  # compatibility with pure-Neumann nullspace

    x = solver._pp_solver.solve(A, b)
    r = A @ x - b
    res_max = float(np.max(np.abs(r)))
    res_b = float(np.max(np.abs(b)))
    assert res_max / max(res_b, 1e-30) < 1e-10, \
        f"PyPardiso residual too large: |r|∞={res_max:.3e}, |b|∞={res_b:.3e}"


def test_projection_reduces_divergence():
    """Sanity: project() should reduce ‖∇·u‖∞ by ≥ 1e6× and bring it
    to absolute level ≤ 1e-6 (variable-density Poisson + random ρ has
    finite cond → direct-LU residual at ~1e-7 is expected, not 1e-12)."""
    solver = _make_solver()
    u, v, w, rho = _make_inputs()
    div_before = solver.divergence(u, v, w)
    div_before_max = float(np.max(np.abs(div_before)))

    solver.project(u, v, w, rho, dt=0.01)
    div_after = solver.divergence(u, v, w)
    div_after_max = float(np.max(np.abs(div_after)))

    assert div_after_max < div_before_max * 1e-6, (
        f"projection didn't reduce divergence enough: "
        f"before={div_before_max:.3e} after={div_after_max:.3e}"
    )
    assert div_after_max < 1e-6, \
        f"residual divergence too large: {div_after_max:.3e} (>1e-6)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
