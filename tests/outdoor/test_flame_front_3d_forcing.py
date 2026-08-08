"""Phase 17b — 3D level-set forcing tests.

Verifies:
 1. compute_v_n_3d with M_local=None reduces exactly to compute_v_n dry
    (regression preservation; no behavior change for dry-only cases).
 2. compute_v_n_3d with M_local > 0 reduces v_n by the moisture latent-heat
    factor (Drysdale §3.5 + Mell 2007 §3.4).
 3. compute_q_in_at_front_3d respects ahead_band_mask (zero outside band).
 4. q_in_3d + v_n_3d is bit-exact deterministic under repeat (Rule #18).
 5. Cells with M_local=0.30 advance ~36% as fast as M_local=0 dry, given
    the same q_in (this is the key Phase 17b moisture-sensitivity gain).
"""
from __future__ import annotations

import numpy as np
import pytest

from model_outdoor.physics_3d import flame_front_3d as ff


# Standard grass-bed thermophysics used across tests.
RHO_B   = 1.07
CP_S    = 1300.0
H_BED   = 0.10
T_IGN   = 600.0
T_AMB   = 300.0
L_VAP   = ff.L_VAP_WATER


def _make_ahead_band_mask(Nz=4, Ny=2, Nx=8):
    """Mock ahead-band mask: cells 2-5 in x are the ahead band."""
    m = np.zeros((Nz, Ny, Nx), dtype=bool)
    m[:, :, 2:6] = True
    return m


def _make_q_uniform(Nz=4, Ny=2, Nx=8, q=1.0e4):
    """Uniform q_dom_fwd and q_frankman fields."""
    q_dom = np.full((Nz, Ny, Nx), q, dtype=np.float64)
    q_fra = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    return q_dom, q_fra


def test_v_n_3d_dry_matches_legacy_v_n_2d():
    """With M_local=None and uniform q_in, v_n_3d should give the SAME
    scalar as the legacy compute_v_n at every cell (regression: dry
    case behavior preserved exactly)."""
    Nz, Ny, Nx = 4, 2, 8
    q_in_3d = np.full((Nz, Ny, Nx), 5.0e4, dtype=np.float64)
    v3d = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                             M_local=None)
    # Legacy 2D for comparison
    q_2d = np.full((Ny, Nx), 5.0e4, dtype=np.float64)
    v2d = ff.compute_v_n(q_2d, RHO_B, CP_S, H_BED, T_IGN, T_AMB)
    # Every cell of v3d should equal the scalar v2d value
    assert np.allclose(v3d, v2d[0, 0]), \
        f"v_n_3d mean {v3d.mean()} != v_n_2d scalar {v2d[0,0]}"


def test_v_n_3d_moisture_reduces_propagation():
    """M_local=0.30 should cut v_n by (1 + L_vap·M/(cp·ΔT))^-1.

    Per Drysdale §3.5:
      E_ign_dry = ρ·cp·h·ΔT
      E_ign_wet = E_ign_dry + ρ·M·h·L_vap
      ratio    = E_ign_dry / E_ign_wet
              = 1 / (1 + M·L_vap/(cp·ΔT))

    At M=0.30: ratio = 1/(1 + 0.30·2.26e6/(1300·300)) = 1/(1+1.74) = 0.366.
    """
    Nz, Ny, Nx = 4, 2, 8
    q_in = 5.0e4
    q_in_3d = np.full((Nz, Ny, Nx), q_in, dtype=np.float64)
    v_dry = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                               M_local=None)
    M_wet = np.full((Nz, Ny, Nx), 0.30, dtype=np.float64)
    v_wet = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                               M_local=M_wet)
    # Expected ratio
    expected = 1.0 / (1.0 + 0.30 * L_VAP / (CP_S * (T_IGN - T_AMB)))
    actual = float(v_wet[0, 0, 0] / v_dry[0, 0, 0])
    assert abs(actual - expected) < 1e-6, \
        f"v_wet/v_dry = {actual:.4f}, expected {expected:.4f}"


def test_v_n_3d_zero_moisture_equals_dry():
    """M_local=0 (not None) should give identical v_n to the dry path."""
    Nz, Ny, Nx = 4, 2, 8
    q_in_3d = np.full((Nz, Ny, Nx), 3.0e4, dtype=np.float64)
    v_none = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                                 M_local=None)
    M_zero = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    v_zero = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                                 M_local=M_zero)
    assert np.array_equal(v_none, v_zero), \
        "M_local=0 array should match M_local=None"


def test_v_n_3d_z_varying_q_produces_z_varying_v():
    """Top-of-bed cell has higher q_in than bottom → higher v_n at top.

    This is the key Phase 17b physics — z-variation in φ is no longer
    masked by enforce_z_uniformity()."""
    Nz, Ny, Nx = 4, 2, 8
    # Top (k=3) gets full forward IR, bottom (k=0) attenuated to 10%
    q_in_3d = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    q_in_3d[3, :, :] = 5.0e4
    q_in_3d[2, :, :] = 4.0e4
    q_in_3d[1, :, :] = 2.0e4
    q_in_3d[0, :, :] = 0.5e4
    v3d = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                             M_local=None)
    # v_n at top should be 10× v_n at bottom (matching q_in ratio)
    assert v3d[3, 0, 0] > v3d[2, 0, 0] > v3d[1, 0, 0] > v3d[0, 0, 0], \
        "v_n_3d failed to capture z-variation in q_in"
    ratio = v3d[3, 0, 0] / v3d[0, 0, 0]
    assert abs(ratio - 10.0) < 0.01, \
        f"v_n ratio top/bottom = {ratio:.2f}, expected 10.0"


def test_q_in_at_front_3d_respects_ahead_band_mask():
    """Cells outside the ahead-band mask must have q_in_3d = 0."""
    Nz, Ny, Nx = 4, 2, 8
    q_dom, q_fra = _make_q_uniform(Nz, Ny, Nx, q=1.0e4)
    mask = _make_ahead_band_mask(Nz, Ny, Nx)
    q_in = ff.compute_q_in_at_front_3d(q_fra, q_dom, mask)
    # Inside band: q_in == q_dom + q_fra = 1e4
    assert np.allclose(q_in[mask], 1.0e4), \
        f"in-band q_in {q_in[mask].mean()} != 1e4"
    # Outside band: q_in == 0
    assert np.all(q_in[~mask] == 0.0), \
        "out-of-band cells should be zero"


def test_q_in_at_front_3d_with_burst_adds_to_top_of_band():
    """q_burst_conv_2d should be added to the top-of-band cell only."""
    Nz, Ny, Nx = 4, 2, 8
    q_dom, q_fra = _make_q_uniform(Nz, Ny, Nx, q=1.0e4)
    mask = _make_ahead_band_mask(Nz, Ny, Nx)
    q_burst = np.full((Ny, Nx), 5.0e3, dtype=np.float64)
    q_in = ff.compute_q_in_at_front_3d(q_fra, q_dom, mask,
                                          q_burst_conv_2d=q_burst)
    # Top-of-band: k=3 inside band cells = 1e4 + 5e3 = 1.5e4
    for j in range(Ny):
        for i in range(Nx):
            if mask[:, j, i].any():
                assert abs(q_in[3, j, i] - 1.5e4) < 1.0, \
                    f"top-of-band[{j},{i}] = {q_in[3,j,i]:.0f}, expected 1.5e4"
                assert abs(q_in[2, j, i] - 1.0e4) < 1.0, \
                    f"non-top[{j},{i}] = {q_in[2,j,i]:.0f}, expected 1e4"


def test_v_n_3d_bit_exact_determinism():
    """Rule #18: bit-exact under repeat."""
    Nz, Ny, Nx = 6, 4, 16
    rng = np.random.default_rng(42)
    q_in_3d = (rng.standard_normal((Nz, Ny, Nx)) * 1e4 + 5e4).astype(np.float64)
    M = (rng.standard_normal((Nz, Ny, Nx)) * 0.05 + 0.15).clip(0, 0.5)
    v1 = ff.compute_v_n_3d(q_in_3d.copy(), RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                            M_local=M.copy())
    v2 = ff.compute_v_n_3d(q_in_3d.copy(), RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                            M_local=M.copy())
    assert np.array_equal(v1, v2), "Rule #18: v_n_3d bit-exact under repeat"


def test_v_n_3d_clamps_forward_only():
    """Negative q_in (would imply heat flowing away from cell) must
    produce v_n ≥ 0 (front advances forward only — Sethian convention)."""
    Nz, Ny, Nx = 4, 2, 8
    q_in_3d = np.full((Nz, Ny, Nx), -1.0e4, dtype=np.float64)
    v3d = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                             M_local=None)
    assert np.all(v3d >= 0.0), "v_n must be ≥ 0 (forward-only)"


def test_v_n_3d_moisture_sensitivity_table():
    """Spot-check the moisture sensitivity at multiple M values matches
    the Drysdale §3.5 analytical formula (sanity check on the physics
    that drives the Phase 17b expected ROS improvement)."""
    Nz, Ny, Nx = 2, 1, 1
    q_in_3d = np.full((Nz, Ny, Nx), 1.0e4, dtype=np.float64)
    v_dry = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                               M_local=None)[0, 0, 0]
    expected_ratios = {}
    for M in [0.05, 0.10, 0.20, 0.30]:
        M_arr = np.full((Nz, Ny, Nx), M, dtype=np.float64)
        v_wet = ff.compute_v_n_3d(q_in_3d, RHO_B, CP_S, H_BED, T_IGN, T_AMB,
                                    M_local=M_arr)[0, 0, 0]
        ratio = v_wet / v_dry
        expected = 1.0 / (1.0 + M * L_VAP / (CP_S * (T_IGN - T_AMB)))
        expected_ratios[M] = (ratio, expected)
        assert abs(ratio - expected) < 1e-6, \
            f"M={M}: v_wet/v_dry={ratio:.4f}, expected {expected:.4f}"
    # Sanity check the absolute drops expected to drive the empirical
    # moisture sensitivity in the next sweep:
    drop_5_to_30 = 1.0 - expected_ratios[0.30][0] / expected_ratios[0.05][0]
    # Expected: ~53% drop from this mechanism alone (M=5→M=30)
    assert drop_5_to_30 > 0.45, \
        f"Expected ROS drop M=5→30 ≥ 45%, got {drop_5_to_30:.2%}"
