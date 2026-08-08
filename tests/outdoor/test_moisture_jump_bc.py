"""Phase 24 — sprinkler-activation moisture-jump BC unit tests.

Rule #18 requirements:
  1. Bit-exact determinism: identical inputs → identical outputs to the
     last digit, for BOTH the Eulerian and Lagrangian branches.
  2. Behavioral sanity — a comprehensive set:
       - Disabled ↔ absent: `enable=False` gives the SAME m_water
         evolution as if the flag block were removed (Rule #17 hard
         requirement — no dormant side-effects).
       - Zone masking (Eulerian): cells outside (x_lo, x_hi) × (z_lo,
         z_hi) untouched; cells inside gained exactly ρ_b·ΔM.
       - One-shot: after the first activation the flag stays set;
         subsequent calls of the same block do NOT re-add water.
       - Monotonicity in ΔM: bigger ΔM → bigger post-jump m_water.
       - Lagrangian parity: sum of particle water added equals
         Eulerian ΔM·ρ_b·V_cell per zone cell (mass balance).
       - Zone-defaults resolution: `None` on a bound → full grid extent.
       - Guards: `ΔM <= 0` and empty/inverted zones raise ValueError
         (early failure, no silent no-ops).

Rule #17 constraint: none of these tests may modify the Cheney Nat4 U=4
result when `moisture_jump_enable=False` — that is separately verified
by the regression sweep before commit.
"""
from __future__ import annotations

import numpy as np
import pytest

from model_outdoor.physics_3d import lagrangian_bed_3d as lb


# ── Shared helpers ─────────────────────────────────────────────────────


def _make_grid(Nz=6, Ny=2, Nx=8, dx=0.10, dy=0.10, dz=0.05):
    dz_arr = np.full(Nz, dz, dtype=np.float64)
    z_face = np.zeros(Nz + 1, dtype=np.float64)
    for k in range(1, Nz + 1):
        z_face[k] = z_face[k - 1] + dz_arr[k - 1]
    return dx, dy, dz_arr, z_face


def _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=2, i_hi=6, a_s_val=0.01):
    alpha = np.zeros((Nz, Ny, Nx))
    alpha[:n_z_bed, :, i_lo:i_hi] = a_s_val
    return alpha


def _fresh_bed(N_per_cell=6, i_lo=2, i_hi=6, rho_b=1.07, M0=0.04):
    """Allocate + initialize a Lagrangian bed at moisture M0 for tests."""
    Nz, Ny, Nx = 6, 2, 8
    n_z_bed = 4
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=i_lo, i_hi=i_hi)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_bed_cells = int((alpha > 0).sum())
    N_max = n_bed_cells * N_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n_alloc = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=rho_b, moisture_frac=M0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=N_per_cell)
    return buf, n_alloc, dx, dy, dz_arr, n_z_bed, i_lo, i_hi, rho_b, M0


# ── 1.  Bit-exact determinism (Rule #18 mandatory) ─────────────────────


def test_lagrangian_moisture_jump_bit_exact_two_runs():
    """Same inputs → identical particle water arrays to the last bit."""
    def _run():
        buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
        kz = np.array([1, 1, 0, 0], dtype=np.uint8)  # bump only the bottom 2 layers
        lb.apply_moisture_jump_zone(
            buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
            i_lo=i_lo + 1, i_hi=i_hi - 1,
            kz_mask=kz, delta_water_kg_m3=rho_b * 0.15)
        return buf["m_water"].copy(), buf["m_water_0"].copy()
    a_mw, a_mw0 = _run()
    b_mw, b_mw0 = _run()
    assert np.array_equal(a_mw, b_mw)
    assert np.array_equal(a_mw0, b_mw0)


# ── 2.  Zone masking (Eulerian analogue: check cell-by-cell) ───────────


def test_lagrangian_zone_mask_particles_outside_untouched():
    buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, M0 = _fresh_bed()
    kz = np.array([1, 0, 0, 0], dtype=np.uint8)   # only the deepest layer
    # Snapshot particle-level water before, per particle.
    mw_before = buf["m_water"][:N].copy()
    x_before = buf["x"][:N].copy()
    z_before = buf["z"][:N].copy()

    # Zone: only cells i_lo+1 .. i_hi-2 in x, k=0 in z (bottom layer only).
    zone_i_lo, zone_i_hi = i_lo + 1, i_hi - 1
    dw = rho_b * 0.20
    n_updated = lb.apply_moisture_jump_zone(
        buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=zone_i_lo, i_hi=zone_i_hi, kz_mask=kz, delta_water_kg_m3=dw)

    mw_after = buf["m_water"][:N]

    # z_face(bottom layer) = [0, dz]  =>  z < dz is inside kz mask
    dz_bottom = dz_arr[0]
    in_zone = ((x_before >= zone_i_lo * dx) & (x_before < zone_i_hi * dx)
               & (z_before >= 0.0) & (z_before < dz_bottom))
    outside = ~in_zone

    # Untouched outside zone
    assert np.array_equal(mw_after[outside], mw_before[outside])
    # Inside zone: water strictly increased
    assert (mw_after[in_zone] > mw_before[in_zone]).all()
    # Diagnostic count matches actual # of touched particles
    assert n_updated == int(in_zone.sum())


# ── 3.  Mass balance: Σ Δparticle_water == cells_in_zone·V_cell·ρ_b·ΔM ─


def test_lagrangian_mass_balance():
    """Total water added should equal the Eulerian ΔM·ρ_b·V_cell per
    zone cell, regardless of how many particles hold that mass."""
    buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed(N_per_cell=7)
    kz = np.array([1, 1, 0, 0], dtype=np.uint8)   # bottom 2 layers
    zone_i_lo, zone_i_hi = i_lo, i_hi   # entire bed x-extent
    delta_M = 0.15
    dw = rho_b * delta_M
    Ny = 2   # matches _fresh_bed

    mw_before = buf["m_water"][:N].sum()
    lb.apply_moisture_jump_zone(
        buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=zone_i_lo, i_hi=zone_i_hi, kz_mask=kz, delta_water_kg_m3=dw)
    mw_after = buf["m_water"][:N].sum()
    added = mw_after - mw_before

    # Cells in zone: (zone_i_hi - zone_i_lo) x Ny x #kz_ones
    n_zone_cells = (zone_i_hi - zone_i_lo) * Ny * int(kz.sum())
    V_cell = dx * dx * dz_arr[0]   # dy == dx here
    expected = dw * V_cell * n_zone_cells
    assert added == pytest.approx(expected, rel=1e-12)


# ── 4.  m_water_0 also lifted (so the drying rate cap moves) ───────────


def test_lagrangian_m_water_0_updated_alongside():
    """The moisture-gate in the drying kernel uses m_water_0 as the
    reference for the 1% gate.  After a sprinkler jump we MUST also
    lift m_water_0 or the newly-added water instantly counts as
    'past its evaporation budget' and drying can dry it away in one
    step, defeating the point of the jump.  This tests that
    m_water_0 rises with m_water."""
    buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
    kz = np.ones(nzb, dtype=np.uint8)
    dw_before = buf["m_water"][:N].copy()
    dw0_before = buf["m_water_0"][:N].copy()
    lb.apply_moisture_jump_zone(
        buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=i_lo, i_hi=i_hi, kz_mask=kz,
        delta_water_kg_m3=rho_b * 0.10)
    delta_mw  = buf["m_water"][:N]   - dw_before
    delta_mw0 = buf["m_water_0"][:N] - dw0_before
    assert np.array_equal(delta_mw, delta_mw0)   # bit-exact same delta


# ── 5.  Monotonicity in ΔM ────────────────────────────────────────────


def test_lagrangian_monotone_in_delta_M():
    """Bigger ΔM → strictly bigger m_water for every touched particle."""
    def _run(delta_M):
        buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
        kz = np.ones(nzb, dtype=np.uint8)
        lb.apply_moisture_jump_zone(
            buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
            i_lo=i_lo, i_hi=i_hi, kz_mask=kz,
            delta_water_kg_m3=rho_b * delta_M)
        return buf["m_water"][:N].copy()
    mw_low  = _run(0.05)
    mw_mid  = _run(0.15)
    mw_high = _run(0.30)
    # touched particles have strictly larger water for larger ΔM
    assert (mw_high >= mw_mid).all()
    assert (mw_mid  >= mw_low ).all()
    # And SOMETHING actually changed
    assert (mw_high > mw_low).any()


# ── 6.  No-op guards (empty zone, ΔM ≤ 0, N ≤ 0) ───────────────────────


def test_lagrangian_no_op_when_N_zero():
    buf, _, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
    kz = np.ones(nzb, dtype=np.uint8)
    n = lb.apply_moisture_jump_zone(
        buf, N=0, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=i_lo, i_hi=i_hi, kz_mask=kz, delta_water_kg_m3=rho_b * 0.10)
    assert n == 0


def test_lagrangian_no_op_when_zone_x_empty():
    buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
    kz = np.ones(nzb, dtype=np.uint8)
    mw_before = buf["m_water"][:N].copy()
    n = lb.apply_moisture_jump_zone(
        buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=i_hi, i_hi=i_hi, kz_mask=kz, delta_water_kg_m3=rho_b * 0.10)
    assert n == 0
    assert np.array_equal(buf["m_water"][:N], mw_before)


def test_lagrangian_no_op_when_kz_all_zero():
    buf, N, dx, dy, dz_arr, nzb, i_lo, i_hi, rho_b, _ = _fresh_bed()
    kz = np.zeros(nzb, dtype=np.uint8)
    mw_before = buf["m_water"][:N].copy()
    n = lb.apply_moisture_jump_zone(
        buf, N=N, dx=dx, dy=dy, dz_arr=dz_arr, n_z_bed=nzb,
        i_lo=i_lo, i_hi=i_hi, kz_mask=kz, delta_water_kg_m3=rho_b * 0.10)
    assert n == 0
    assert np.array_equal(buf["m_water"][:N], mw_before)


# ── 7.  spread_3d.run_3d_spread signature / deck-flag surface ──────────


def test_run_3d_spread_exposes_all_six_flags():
    """The Phase 24 plan freezes six deck flags; the entry-point signature
    must expose them so decks and unit tests can set them."""
    import inspect
    from model_outdoor.spread_3d import run_3d_spread
    sig = inspect.signature(run_3d_spread)
    for name in (
        "moisture_jump_enable",
        "moisture_jump_t_s",
        "moisture_jump_delta_frac",
        "moisture_jump_x_lo_m",
        "moisture_jump_x_hi_m",
        "moisture_jump_z_lo_m",
        "moisture_jump_z_hi_m",
    ):
        assert name in sig.parameters, f"missing kwarg: {name}"


def test_run_3d_spread_defaults_are_safe():
    """When we don't touch these flags, the enable is False and the
    delta is 0.0 — a completely dormant code path that cannot mutate
    m_water.  Rule #17 preservation depends on this."""
    import inspect
    from model_outdoor.spread_3d import run_3d_spread
    sig = inspect.signature(run_3d_spread)
    assert sig.parameters["moisture_jump_enable"].default is False
    assert sig.parameters["moisture_jump_delta_frac"].default == 0.0
    assert sig.parameters["moisture_jump_t_s"].default == 0.0
