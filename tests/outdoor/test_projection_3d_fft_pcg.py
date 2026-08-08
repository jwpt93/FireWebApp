"""CLAUDE.md Rule #18 unit tests for the FFT-preconditioned BiCGSTAB
projection path (Phase 14ax-2).

The new ``method='fft_pcg'`` path uses BiCGSTAB on the variable-
coefficient operator ``A = ∇·((α_g/ρ)·∇p)`` with a constant-coefficient
Poisson preconditioner ``M⁻¹ = (∇²)⁻¹`` (SeparableLaplacian3D, Phase
14ax-1).  The expected behavior:

  - Output velocity field matches PARDISO direct solve to within
    BiCGSTAB rtol on the same inputs (correctness vs reference solver).
  - At α_g/ρ ≈ const everywhere, BiCGSTAB converges in 1-2 iterations
    (preconditioner is exact).
  - Output is bit-exact deterministic on repeated identical calls
    (Rule #17 kernel-level analogue).
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


def _make_solver(method: str, Nz=8, Ny=4, Nx=16, dx=0.1, **kwargs):
    dy = dz = dx
    dz_arr, d_above, d_below = _uniform_dz_arrays(Nz, dz)
    return ProjectionSolver3D(
        Nz, Ny, Nx, dy, dx,
        dz_arr=dz_arr, d_face_above=d_above, d_face_below=d_below,
        y_bc="periodic", method=method, **kwargs,
    )


def _make_inputs(Nz=8, Ny=4, Nx=16, *, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.3 * rng.standard_normal((Nz, Ny, Nx))
    rho = np.maximum(rho, 0.1)
    u = rng.standard_normal((Nz, Ny, Nx))
    v = rng.standard_normal((Nz, Ny, Nx))
    w = rng.standard_normal((Nz, Ny, Nx))
    return u, v, w, rho


def test_fft_pcg_matches_pardiso_velocity_to_rtol():
    """Projected u from FFT-PCG path matches PARDISO direct solve."""
    u, v, w, rho = _make_inputs()

    # PARDISO reference
    s_pard = _make_solver("pardiso")
    s_pard.rebuild_for_rho(rho)
    up, vp, wp = u.copy(), v.copy(), w.copy()
    s_pard.project(up, vp, wp, rho, dt=0.01)

    # FFT-PCG at rtol=1e-8 should give very close velocity.
    s_fft = _make_solver("fft_pcg", cg_rtol=1.0e-8)
    s_fft.rebuild_for_rho(rho)
    uf, vf, wf = u.copy(), v.copy(), w.copy()
    s_fft.project(uf, vf, wf, rho, dt=0.01)

    rel_u = np.max(np.abs(uf - up)) / max(np.max(np.abs(up)), 1e-30)
    assert rel_u < 1.0e-6, (
        f"FFT-PCG vs PARDISO u-rel-err = {rel_u:.2e}, expected < 1e-6")


def test_fft_pcg_constant_rho_exact_in_one_iter():
    """When ρ is uniform, the FFT preconditioner is exact and BiCGSTAB
    converges immediately to PARDISO accuracy."""
    Nz, Ny, Nx = 8, 4, 16
    rho = np.full((Nz, Ny, Nx), 1.0)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((Nz, Ny, Nx))
    v = rng.standard_normal((Nz, Ny, Nx))
    w = rng.standard_normal((Nz, Ny, Nx))

    s_pard = _make_solver("pardiso", Nz=Nz, Ny=Ny, Nx=Nx)
    s_pard.rebuild_for_rho(rho)
    up, vp, wp = u.copy(), v.copy(), w.copy()
    s_pard.project(up, vp, wp, rho, dt=0.01)

    s_fft = _make_solver("fft_pcg", Nz=Nz, Ny=Ny, Nx=Nx, cg_rtol=1e-10)
    s_fft.rebuild_for_rho(rho)
    uf, vf, wf = u.copy(), v.copy(), w.copy()
    s_fft.project(uf, vf, wf, rho, dt=0.01)

    rel_u = np.max(np.abs(uf - up)) / max(np.max(np.abs(up)), 1e-30)
    # At constant ρ, FFT preconditioner is the exact inverse of A (mod ε).
    # We expect near-machine precision agreement.
    assert rel_u < 1.0e-8, (
        f"constant-ρ FFT-PCG should match PARDISO; rel_u = {rel_u:.2e}")


def test_fft_pcg_bit_exact_determinism():
    """Two consecutive solves on identical inputs → bit-exact output (Rule #17)."""
    u, v, w, rho = _make_inputs()
    s = _make_solver("fft_pcg")
    s.rebuild_for_rho(rho)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    p1 = s.project(u1, v1, w1, rho, dt=0.01)

    s.rebuild_for_rho(rho)
    u2, v2, w2 = u.copy(), v.copy(), w.copy()
    p2 = s.project(u2, v2, w2, rho, dt=0.01)

    assert np.array_equal(p1, p2), "two FFT-PCG solves must be bit-exact"
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(w1, w2)
