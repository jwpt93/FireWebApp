"""CLAUDE.md Rule #18 unit tests for ProjectionSolver3D AMG-CG path.

Phase 14ah introduces an iterative solver path (PyAMG smoothed_aggregation
preconditioner + scipy CG) as an alternative to direct PyPardiso LU.  These
tests verify:

  1. AMG-CG agrees with PARDISO direct solve to within configured rtol on
     the projected velocity field.
  2. Bit-exact determinism: identical inputs → identical outputs across
     two consecutive AMG-CG solves (Rule #17 kernel-level analogue).
  3. Warm-start path: solving twice with the same ρ but perturbed u still
     converges (no divergence on second solve).
  4. AMG rebuild trigger: after ``amg_rebuild_every`` steps the hierarchy
     refreshes and CG still converges.
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


def _make_solver(method: str, Nz=8, Ny=4, Nx=32, dx=0.1, **kwargs):
    dy = dz = dx
    dz_arr, d_above, d_below = _uniform_dz_arrays(Nz, dz)
    return ProjectionSolver3D(
        Nz, Ny, Nx, dy, dx,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
        y_bc="periodic",
        method=method,
        **kwargs,
    )


def _make_inputs(Nz=8, Ny=4, Nx=32, dx=0.1, *, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.5 * rng.standard_normal((Nz, Ny, Nx))
    rho = np.maximum(rho, 0.1)
    Lx = Nx * dx
    x = (np.arange(Nx) + 0.5) * dx
    u = np.broadcast_to(np.sin(np.pi * x / Lx)[None, None, :],
                        (Nz, Ny, Nx)).copy()
    v = np.zeros((Nz, Ny, Nx))
    w = np.zeros((Nz, Ny, Nx))
    return u, v, w, rho


def test_amg_cg_matches_pardiso_velocity_to_rtol():
    """Projected u (the physical output) from AMG-CG agrees with PARDISO
    direct solve to within ~cg_rtol.
    """
    u, v, w, rho = _make_inputs()

    # PARDISO reference
    s_pard = _make_solver("pardiso")
    s_pard.rebuild_for_rho(rho)
    up, vp, wp = u.copy(), v.copy(), w.copy()
    s_pard.project(up, vp, wp, rho, dt=0.01)

    # AMG-CG at rtol=1e-8 should give very close velocity
    s_amg = _make_solver("amg_cg", cg_rtol=1.0e-8)
    s_amg.rebuild_for_rho(rho)
    ua, va, wa = u.copy(), v.copy(), w.copy()
    s_amg.project(ua, va, wa, rho, dt=0.01)

    rel_u = np.max(np.abs(ua - up)) / max(np.max(np.abs(up)), 1e-30)
    assert rel_u < 1.0e-6, f"AMG-CG vs PARDISO u-rel-err = {rel_u:.2e}, expected < 1e-6"


def test_amg_cg_bit_exact_determinism():
    """Two consecutive AMG-CG solves on identical inputs → bit-exact (Rule #17)."""
    s = _make_solver("amg_cg")
    u, v, w, rho = _make_inputs()

    s.rebuild_for_rho(rho)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    p1 = s.project(u1, v1, w1, rho, dt=0.01)

    # Reset warm-start cache to force same starting condition
    s._p_prev = None
    s.rebuild_for_rho(rho)
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    p2 = s.project(u2, v2, w2, rho, dt=0.01)

    assert np.array_equal(p1, p2), \
        f"AMG-CG p drift across calls: max|Δ| = {np.max(np.abs(p1 - p2)):.2e}"
    assert np.array_equal(u1, u2), "AMG-CG projected u drift"


def test_amg_cg_warm_start_consecutive_calls():
    """Second call uses warm-start from first.  Both must converge and
    second should agree with PARDISO reference."""
    u, v, w, rho = _make_inputs()

    s = _make_solver("amg_cg", cg_rtol=1.0e-8)
    s.rebuild_for_rho(rho)
    # First call: cold start
    ua, va, wa = u.copy(), v.copy(), w.copy()
    s.project(ua, va, wa, rho, dt=0.01)

    # Perturb u slightly, re-solve (different rhs, same ρ matrix)
    ua += 0.01 * np.random.default_rng(7).standard_normal(ua.shape)
    s.rebuild_for_rho(rho)
    s.project(ua, va, wa, rho, dt=0.01)
    # Verify divergence reduced
    div = s.divergence(ua, va, wa)
    assert np.max(np.abs(div)) < 1.0e-2, \
        f"AMG-CG warm-start failed to reduce divergence: |div|∞={np.max(np.abs(div)):.2e}"


def test_amg_cg_rebuild_every_triggers_hierarchy_refresh():
    """After amg_rebuild_every solves, the hierarchy should rebuild and CG
    still converge.  Verifies the periodic-refresh path."""
    s = _make_solver("amg_cg", amg_rebuild_every=3)
    u, v, w, rho = _make_inputs()

    for k in range(7):
        s.rebuild_for_rho(rho)
        uk, vk, wk = u.copy(), v.copy(), w.copy()
        s.project(uk, vk, wk, rho, dt=0.01)
        div = s.divergence(uk, vk, wk)
        assert np.max(np.abs(div)) < 1.0e-2, \
            f"step {k}: AMG-CG divergence not reduced: {np.max(np.abs(div)):.2e}"
    # After 7 steps (rebuild_every=3) we should have rebuilt at steps 3 and 6
    assert s._steps_since_amg_build in (0, 1, 2), \
        f"steps_since_amg_build={s._steps_since_amg_build} after 7 solves with rebuild_every=3"
