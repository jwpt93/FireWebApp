"""Phase 15N unit tests — Finney 2015 parameterized burst-convective closure.

Rule #18: every new module ships with unit tests in the same commit.
Covers: dimensions, magnitude at lit values, decay shape, I_fire gate,
bit-exact determinism (Rule #17).
"""
from __future__ import annotations

import numpy as np
import pytest

from model_outdoor.physics_3d import finney_burst_3d as fb


def _make_band_at_band_index(Nz: int, Ny: int, Nx: int, i_band_start: int,
                              i_band_end: int) -> np.ndarray:
    """Build a (Nz, Ny, Nx) ahead-band mask True for i in [i_band_start, i_band_end)."""
    m = np.zeros((Nz, Ny, Nx), dtype=bool)
    m[:, :, i_band_start:i_band_end] = True
    return m


def _make_flame_body_at(Nz: int, Ny: int, Nx: int, i_flame: int) -> np.ndarray:
    """phi_flame array with phi_flame ≤ 0 at column i = i_flame only.
    All other columns set to large positive (outside flame body)."""
    phi = np.full((Nz, Ny, Nx), +1e6, dtype=np.float64)
    phi[:, :, i_flame] = -0.1
    return phi


def test_dimensions_W_per_m2():
    """Output is per-cell W/m² surface flux, shape (Ny, Nx)."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=20)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    assert q.shape == (Ny, Nx)
    assert q.dtype == np.float64
    assert q[0, 11] > 0.0, "expected non-zero burst flux just ahead of flame"
    assert q[0, 5] == 0.0, "no contribution behind flame body"


def test_value_at_zero_distance_equals_q0():
    """Cell immediately ahead of the flame edge gets q ≈ q_0."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=20)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    # Cell i=11 center is one dx ahead of flame center at i=10
    d_expected = x_mid[11] - x_mid[10]   # = dx
    q_expected = fb.Q_0_DEFAULT * np.exp(-d_expected / fb.L_BURST_DEFAULT)
    assert abs(q[0, 11] - q_expected) < 1e-9, (
        f"q[0, 11] = {q[0, 11]:.1f} vs expected {q_expected:.1f}"
    )


def test_exponential_decay_shape():
    """q[j, i] / q[j, i+1] = exp(dx / L_burst) — exponential decay."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=20)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    # Check decay between successive band cells
    for i in range(11, 18):
        if q[0, i] == 0 or q[0, i+1] == 0:
            continue
        ratio = q[0, i] / q[0, i+1]
        expected = np.exp(dx / fb.L_BURST_DEFAULT)
        assert abs(ratio - expected) < 1e-9, (
            f"decay ratio at i={i}: {ratio:.4f} vs expected {expected:.4f}"
        )


def test_zero_when_no_flame_body():
    """No flame body → q = 0 everywhere."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = np.full((Nz, Ny, Nx), +1e6, dtype=np.float64)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=10, i_band_end=20)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    assert np.all(q == 0.0), "expected zero flux when no flame body"


def test_zero_outside_ahead_band():
    """Cells outside ahead_band_mask get q = 0 even if flame body upstream."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    # Band covers only i=11..14, but flame is at i=10
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=15)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    # i=15..19 should be 0 (outside band) even though d_ahead is small
    for i in (15, 16, 17, 18, 19):
        assert q[0, i] == 0.0, f"i={i} outside band but q={q[0, i]}"


def test_distance_cutoff_d_max():
    """q = 0 when d_ahead > d_max."""
    Nz, Ny, Nx = 18, 5, 200
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=Nx)
    q = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    # At i=120, d = 11.0 m >> 1.0 m cutoff → q = 0
    assert q[0, 120] == 0.0, "expected zero beyond d_max cutoff"


def test_I_fire_soft_gate_below_threshold():
    """When I_fire_per_y < I_thresh, gate scales linearly, flux reduced."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=20)

    # Without I_fire input → unmasked
    q_full = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    # With I_fire = 0.5 × I_thresh → gate ≈ 0.5
    I_half = 0.5 * fb.I_FIRE_THRESH
    q_half = fb.compute_finney_burst_q_at_band(
        phi, band, dx, x_mid,
        I_fire_per_y=np.full(Ny, I_half),
    )
    ratio = q_half[0, 11] / q_full[0, 11]
    assert abs(ratio - 0.5) < 1e-9, (
        f"gate ratio at 0.5×I_thresh: {ratio:.4f} vs expected 0.5"
    )


def test_I_fire_gate_saturated_above_threshold():
    """When I_fire_per_y > I_thresh, gate = 1.0 (saturated)."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    phi = _make_flame_body_at(Nz, Ny, Nx, i_flame=10)
    band = _make_band_at_band_index(Nz, Ny, Nx, i_band_start=11, i_band_end=20)
    I_large = 5.0 * fb.I_FIRE_THRESH
    q_lit = fb.compute_finney_burst_q_at_band(
        phi, band, dx, x_mid,
        I_fire_per_y=np.full(Ny, I_large),
    )
    q_full = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    assert abs(q_lit[0, 11] - q_full[0, 11]) < 1e-12


def test_deterministic_under_repeat():
    """Rule #17: bit-exact deterministic output on identical inputs."""
    Nz, Ny, Nx = 18, 5, 60
    dx = 0.1
    x_mid = np.arange(Nx) * dx + dx / 2
    rng = np.random.default_rng(15)
    # Mix of flame cells and non-flame cells (random)
    phi = np.where(rng.random((Nz, Ny, Nx)) < 0.1, -0.1, +1e6)
    band = (rng.random((Nz, Ny, Nx)) < 0.3)

    q1 = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    q2 = fb.compute_finney_burst_q_at_band(phi, band, dx, x_mid)
    assert np.array_equal(q1, q2), "non-deterministic output"


def test_I_fire_helper_dimensions_and_sign():
    """compute_I_fire_per_y returns (Ny,) non-negative array."""
    Nz, Ny, Nx = 18, 5, 60
    omega = np.zeros((Nz, Ny, Nx))
    omega[5, 2, 30] = 1.0   # 1 kg/m³/s in one cell, only in y-row j=2
    dz_arr = np.full(Nz, 0.02)
    I = fb.compute_I_fire_per_y(omega, dx=0.1, dz_arr=dz_arr)
    assert I.shape == (Ny,)
    assert np.all(I >= 0.0)
    # Only j=2 should have non-zero intensity
    assert I[2] > 0.0
    assert I[0] == 0.0


def test_conservative_lit_values_are_committed():
    """Documents the committed Phase 15N values; if they change, the
    full verification sequence must be re-run."""
    assert fb.Q_0_DEFAULT == 100_000.0, (
        "Phase 15N committed value: q_0 = 100 kW/m² (Finney 2015 conservative). "
        "Changing requires re-running mickey + Cheney verification per the plan."
    )
    assert fb.L_BURST_DEFAULT == 0.30, (
        "Phase 15N committed value: L_burst = 0.30 m "
        "(Strouhal-Froude scaling for U=4 m/s, L_F≈1 m)."
    )
    assert fb.D_MAX_CUTOFF == 1.0
    assert fb.I_FIRE_THRESH == 100_000.0
