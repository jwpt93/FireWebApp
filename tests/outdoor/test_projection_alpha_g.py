"""CLAUDE.md Rule #18 unit tests for Phase 14aw-1 volume-weighted projection.

Phase 14aw-1 adds α_g (gas volume fraction) weighting to the variable-
density Poisson operator in ProjectionSolver3D.  When ``set_alpha_g`` is
called with a non-None array, the operator becomes ∇·((α_g/ρ)·∇p) and
the divergence operator becomes ∇·(α_g·u).  When ``set_alpha_g`` is not
called (or called with None), behavior is the pre-14aw baseline:
operator ∇·((1/ρ)·∇p), divergence ∇·u.

This module verifies:
  1. Default behavior (no ``set_alpha_g`` call) is bit-exact identical
     to the pre-14aw baseline.
  2. Calling ``set_alpha_g`` with α_g ≡ 1 everywhere reproduces the
     pre-14aw baseline (the natural α_g=1 limit).
  3. Calling ``set_alpha_g`` with a non-trivial α_g profile changes the
     projection result (sanity that the volume-weighted path is
     actually active).
  4. After projection, the volume-weighted divergence ∇·(α_g·u) is
     approximately zero (discrete identity).
  5. Bit-exact determinism on two back-to-back solves with the same
     α_g (Rule #17 kernel-level analogue).
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


def _make_solver(method: str = "pardiso", Nz=8, Ny=4, Nx=16, dx=0.1, **kwargs):
    dy = dz = dx
    dz_arr, d_above, d_below = _uniform_dz_arrays(Nz, dz)
    return ProjectionSolver3D(
        Nz, Ny, Nx, dy, dx,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
        y_bc="periodic", method=method, **kwargs,
    )


def _make_inputs(Nz=8, Ny=4, Nx=16, dx=0.1, *, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.5 * rng.standard_normal((Nz, Ny, Nx))
    rho = np.maximum(rho, 0.1)
    Lx = Nx * dx
    x = (np.arange(Nx) + 0.5) * dx
    u = np.broadcast_to(np.sin(np.pi * x / Lx)[None, None, :],
                        (Nz, Ny, Nx)).copy().astype(np.float64)
    v = np.zeros((Nz, Ny, Nx))
    w = np.zeros((Nz, Ny, Nx))
    return u, v, w, rho


def test_default_alpha_g_none_is_baseline_bit_exact():
    """Without ``set_alpha_g`` call (or alpha_g=None), output matches
    pre-14aw baseline bit-exactly.  Reference baseline computed by the
    same solver instance (the None branch *is* the baseline path)."""
    u, v, w, rho = _make_inputs()
    s = _make_solver("pardiso")
    s.rebuild_for_rho(rho)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    p1 = s.project(u1, v1, w1, rho, dt=0.01)

    # Second instance: explicitly call set_alpha_g(None)
    s2 = _make_solver("pardiso")
    s2.set_alpha_g(None)
    s2.rebuild_for_rho(rho)
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    p2 = s2.project(u2, v2, w2, rho, dt=0.01)

    assert np.array_equal(p1, p2), "set_alpha_g(None) must match default behavior"
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(w1, w2)


def test_alpha_g_all_ones_matches_baseline_to_FP_precision():
    """α_g ≡ 1 should match the pre-14aw baseline to FP precision.

    NOT bit-exact (the volume-weighted divergence does extra
    multiplications by 1.0 — different code path), but the result
    differs by at most ~1e-14 relative on the projected velocity.
    """
    u, v, w, rho = _make_inputs()
    Nz, Ny, Nx = rho.shape

    s_base = _make_solver("pardiso")
    s_base.rebuild_for_rho(rho)
    ub, vb, wb = u.copy(), v.copy(), w.copy()
    s_base.project(ub, vb, wb, rho, dt=0.01)

    s_aw = _make_solver("pardiso")
    s_aw.set_alpha_g(np.ones((Nz, Ny, Nx), dtype=np.float64))
    s_aw.rebuild_for_rho(rho)
    uw, vw, ww = u.copy(), v.copy(), w.copy()
    s_aw.project(uw, vw, ww, rho, dt=0.01)

    rel_u = np.max(np.abs(uw - ub)) / max(np.max(np.abs(ub)), 1e-30)
    assert rel_u < 1.0e-10, f"α_g≡1 should reproduce baseline; rel_u={rel_u:.2e}"


def test_alpha_g_non_trivial_changes_projection():
    """A non-trivial α_g profile produces a *different* projected
    velocity from the baseline.  Sanity: confirms volume-weighted path
    is wired in (not a no-op due to a stale buffer or missed branch).
    """
    u, v, w, rho = _make_inputs()
    Nz, Ny, Nx = rho.shape

    s_base = _make_solver("pardiso")
    s_base.rebuild_for_rho(rho)
    ub, vb, wb = u.copy(), v.copy(), w.copy()
    s_base.project(ub, vb, wb, rho, dt=0.01)

    s_aw = _make_solver("pardiso")
    # Non-trivial: low α_g (lots of solid) in lower-left quadrant
    alpha_g = np.ones((Nz, Ny, Nx), dtype=np.float64)
    alpha_g[:Nz // 2, :, :Nx // 2] = 0.3   # bed-like region
    s_aw.set_alpha_g(alpha_g)
    s_aw.rebuild_for_rho(rho)
    uw, vw, ww = u.copy(), v.copy(), w.copy()
    s_aw.project(uw, vw, ww, rho, dt=0.01)

    rel_u = np.max(np.abs(uw - ub)) / max(np.max(np.abs(ub)), 1e-30)
    assert rel_u > 1.0e-3, (
        f"Non-trivial α_g should change projection; rel_u={rel_u:.2e}"
    )


def test_alpha_g_volume_weighted_divergence_is_zero_after_project():
    """After projection with α_g set and div_target=None, the volume-
    weighted divergence ∇·(α_g u^{n+1}) should be ≈ 0 to solver tol.
    """
    u, v, w, rho = _make_inputs()
    Nz, Ny, Nx = rho.shape

    s = _make_solver("pardiso")
    alpha_g = np.ones((Nz, Ny, Nx), dtype=np.float64)
    # Smooth α_g profile (no abrupt jump)
    z = (np.arange(Nz) + 0.5) / Nz
    alpha_g[:, :, :] = (0.5 + 0.5 * z)[:, None, None]
    s.set_alpha_g(alpha_g)
    s.rebuild_for_rho(rho)
    s.project(u, v, w, rho, dt=0.01)

    div = s.divergence(u, v, w)   # volume-weighted ∇·(α_g u) by Phase 14aw-1
    # ε-regularized matrix means there's a tiny constant-mode residual.
    # Should still be tiny — < 1e-6 in max norm.
    max_div = np.max(np.abs(div))
    assert max_div < 1.0e-6, f"post-projection ∇·(α_g u) max = {max_div:.2e}"


def test_alpha_g_bit_exact_determinism():
    """Two back-to-back projections with the same α_g and inputs give
    bit-exact identical results (Rule #17 kernel-level analogue)."""
    u, v, w, rho = _make_inputs()
    Nz, Ny, Nx = rho.shape
    alpha_g = 1.0 - 0.3 * np.random.default_rng(7).random((Nz, Ny, Nx))

    s = _make_solver("pardiso")
    s.set_alpha_g(alpha_g)
    s.rebuild_for_rho(rho)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    p1 = s.project(u1, v1, w1, rho, dt=0.01)

    s.rebuild_for_rho(rho)
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    p2 = s.project(u2, v2, w2, rho, dt=0.01)

    assert np.array_equal(p1, p2), "two solves with same input must be bit-exact"
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(w1, w2)


def test_alpha_g_set_then_unset_returns_to_baseline():
    """Setting α_g, then calling set_alpha_g(None) reverts to baseline.

    Tolerance is FP-precision (not bit-exact): the reused-solver path
    triggers PARDISO phase-22 numerical refactor, which can pick a
    slightly different pivoting order than the fresh-solver phase-12
    path used by the baseline reference.  Bit-exactness of the
    default-None branch is covered by test_default_alpha_g_none_is_baseline_bit_exact.
    """
    u, v, w, rho = _make_inputs()
    Nz, Ny, Nx = rho.shape

    s_base = _make_solver("pardiso")
    s_base.rebuild_for_rho(rho)
    ub, vb, wb = u.copy(), v.copy(), w.copy()
    s_base.project(ub, vb, wb, rho, dt=0.01)

    s = _make_solver("pardiso")
    s.set_alpha_g(0.5 * np.ones((Nz, Ny, Nx)))
    s.rebuild_for_rho(rho)
    _ = s.project(u.copy(), v.copy(), w.copy(), rho, dt=0.01)
    # Reset to baseline mode
    s.set_alpha_g(None)
    s.rebuild_for_rho(rho)
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    s.project(u2, v2, w2, rho, dt=0.01)

    rel_u = np.max(np.abs(u2 - ub)) / max(np.max(np.abs(ub)), 1e-30)
    assert rel_u < 1.0e-10, (
        f"after set_alpha_g(None), should match baseline; rel_u={rel_u:.2e}"
    )
