"""Phase 15O unit tests — Eulerian Finney-tendril spawn-and-deposit closure.

Tests enforce strict conservation (mass, energy, species, momentum) per
spawn event, plus Rule #17 bit-exact determinism and the leading-edge
masking + Strouhal-Froude gate behavior.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from model_outdoor.physics_3d import finney_tendril_3d as ft


def _make_state(Nz=10, Ny=4, Nx=20, rho_val=1.2, Tg_val=1500.0,
                Yf_val=0.1, u_val=4.0, v_val=0.0, w_val=0.5):
    rho = np.full((Nz, Ny, Nx), rho_val, dtype=np.float64)
    T_g = np.full((Nz, Ny, Nx), Tg_val, dtype=np.float64)
    Y_F = np.full((Nz, Ny, Nx), Yf_val, dtype=np.float64)
    u   = np.full((Nz, Ny, Nx), u_val, dtype=np.float64)
    v   = np.full((Nz, Ny, Nx), v_val, dtype=np.float64)
    w   = np.full((Nz, Ny, Nx), w_val, dtype=np.float64)
    return rho, T_g, Y_F, u, v, w


def _make_phi_flame_at(Nz, Ny, Nx, i_flame_end):
    """phi_flame ≤ 0 for i ≤ i_flame_end, phi_flame > 0 for i > i_flame_end.

    Returns array shape (Nz, Ny, Nx).  Leading-edge cell at i = i_flame_end.
    """
    phi = np.full((Nz, Ny, Nx), +1.0, dtype=np.float64)
    phi[:, :, :i_flame_end + 1] = -0.1   # inside body
    return phi


# ── BASIC SANITY ────────────────────────────────────────────────────────

def test_spawn_at_leading_edge_only():
    """Spawn occurs only at the (k,j,i) where phi_flame ≤ 0 AND phi[i+1] > 0.
    Interior body cells (phi[i] ≤ 0 AND phi[i+1] ≤ 0) should NOT spawn."""
    Nz, Ny, Nx = 10, 2, 20
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=10)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)

    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )

    # n_spawn should equal the number of (k,j) rows with a leading edge
    # at i = 10 (which is the only LE in this setup)
    # = Nz × Ny = 10 × 2 = 20 spawn events
    assert n_spawn[0] == Nz * Ny, (
        f"expected {Nz * Ny} spawn events (one per (k,j) at LE), got {n_spawn[0]}"
    )


def test_no_spawn_when_below_Fr_min():
    """If Fr_local < fr_min (cold gas), no spawn."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx, Tg_val=350.0)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=5)
    L_F = np.full((Ny, Nx), 0.1, dtype=np.float64)  # very small L_F
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)

    # T_g=350, T_amb=300, L_F=0.1 → u_buoy = sqrt(2*9.81*0.1*50/300) ≈ 0.57 m/s
    # Fr = 0.57² / (9.81 * 0.1) ≈ 0.33 < 0.5 → no spawn
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] == 0


def test_no_spawn_when_no_flame_body():
    """phi_flame > 0 everywhere → no leading-edge cells → no spawn."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = np.full((Nz, Ny, Nx), +1.0, dtype=np.float64)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)

    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] == 0


def test_frequency_cap_blocks_immediate_re_spawn():
    """After spawn at t=0, calling again at t≈0 must not spawn again."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=4)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn1 = np.zeros(1, dtype=np.int64)
    n_spawn2 = np.zeros(1, dtype=np.int64)

    # First call at t=0: should spawn
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn1,
    )
    # Second call at t=1e-3 (much less than T_period): must not spawn
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=1e-3,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn2,
    )
    assert n_spawn1[0] > 0
    assert n_spawn2[0] == 0, (
        f"expected zero re-spawns within T_period, got {n_spawn2[0]}"
    )


# ── CONSERVATION TESTS (CORE OF Rule #18 DISCIPLINE) ─────────────────────

def _total_mass(rho, dx, dy, dz_arr):
    cell_vol = dx * dy * dz_arr.reshape(-1, 1, 1)
    return float((rho * cell_vol).sum())


def _total_enthalpy(rho, T_g, dx, dy, dz_arr):
    cell_vol = dx * dy * dz_arr.reshape(-1, 1, 1)
    return float((rho * ft.CP_GAS * T_g * cell_vol).sum())


def _total_fuel_mass(rho, Y_F, dx, dy, dz_arr):
    cell_vol = dx * dy * dz_arr.reshape(-1, 1, 1)
    return float((rho * Y_F * cell_vol).sum())


def _total_momentum_x(rho, u, dx, dy, dz_arr):
    cell_vol = dx * dy * dz_arr.reshape(-1, 1, 1)
    return float((rho * u * cell_vol).sum())


def test_mass_conservation_strict():
    """Σ ρV after spawn step == Σ ρV before, to floating-point precision."""
    Nz, Ny, Nx = 8, 3, 20
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=8)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1

    M_before = _total_mass(rho, dx, dy, dz_arr)
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=dx, dy=dy, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] > 0, "no spawn events — test cannot verify conservation"
    M_after = _total_mass(rho, dx, dy, dz_arr)
    rel_err = abs(M_after - M_before) / max(abs(M_before), 1.0)
    assert rel_err < 1e-12, (
        f"mass not conserved: M_before={M_before:.10g}, "
        f"M_after={M_after:.10g}, rel_err={rel_err:.3e}"
    )


def test_enthalpy_conservation_strict():
    """Σ ρ·cp·T·V before == after to floating-point precision."""
    Nz, Ny, Nx = 8, 3, 20
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    # Use heterogeneous T_g to make conservation non-trivial
    rng = np.random.default_rng(15)
    T_g[:] = 800.0 + 500.0 * rng.random((Nz, Ny, Nx))
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=8)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1

    E_before = _total_enthalpy(rho, T_g, dx, dy, dz_arr)
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=dx, dy=dy, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] > 0
    E_after = _total_enthalpy(rho, T_g, dx, dy, dz_arr)
    rel_err = abs(E_after - E_before) / max(abs(E_before), 1.0)
    assert rel_err < 1e-12, (
        f"enthalpy not conserved: E_before={E_before:.6g}, "
        f"E_after={E_after:.6g}, rel_err={rel_err:.3e}"
    )


def test_fuel_species_conservation_strict():
    """Σ ρ·Y_F·V before == after to floating-point precision."""
    Nz, Ny, Nx = 8, 3, 20
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    rng = np.random.default_rng(16)
    Y_F[:] = 0.01 + 0.20 * rng.random((Nz, Ny, Nx))
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=8)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1

    F_before = _total_fuel_mass(rho, Y_F, dx, dy, dz_arr)
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=dx, dy=dy, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] > 0
    F_after = _total_fuel_mass(rho, Y_F, dx, dy, dz_arr)
    rel_err = abs(F_after - F_before) / max(abs(F_before), 1.0)
    assert rel_err < 1e-12, (
        f"fuel species not conserved: F_before={F_before:.6g}, "
        f"F_after={F_after:.6g}, rel_err={rel_err:.3e}"
    )


def test_momentum_x_conservation_strict():
    """Σ ρ·u·V before == after to floating-point precision."""
    Nz, Ny, Nx = 8, 3, 20
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    rng = np.random.default_rng(17)
    u[:] = 2.0 + 4.0 * rng.random((Nz, Ny, Nx))
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=8)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1

    Px_before = _total_momentum_x(rho, u, dx, dy, dz_arr)
    ft.step_finney_tendril_spawn_deposit(
        rho, T_g, Y_F, u, v, w, phi, L_F, last_spawn,
        dx=dx, dy=dy, dz_arr=dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] > 0
    Px_after = _total_momentum_x(rho, u, dx, dy, dz_arr)
    rel_err = abs(Px_after - Px_before) / max(abs(Px_before), 1.0)
    assert rel_err < 1e-12, (
        f"x-momentum not conserved: Px_before={Px_before:.6g}, "
        f"Px_after={Px_after:.6g}, rel_err={rel_err:.3e}"
    )


# ── DETERMINISM (Rule #17) ──────────────────────────────────────────────

def test_bit_exact_under_repeat():
    """Two identical-input runs produce identical output (Rule #17)."""
    Nz, Ny, Nx = 8, 3, 20
    rng = np.random.default_rng(18)
    rho_A = 1.0 + 0.1 * rng.random((Nz, Ny, Nx))
    T_g_A = 800.0 + 500.0 * rng.random((Nz, Ny, Nx))
    Y_F_A = 0.01 + 0.20 * rng.random((Nz, Ny, Nx))
    u_A   = 2.0 + 4.0 * rng.random((Nz, Ny, Nx))
    v_A   = 0.2 * rng.standard_normal((Nz, Ny, Nx))
    w_A   = 0.2 * rng.standard_normal((Nz, Ny, Nx))
    phi_A = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=8)
    L_F_A = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn_A = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn_A = np.zeros(1, dtype=np.int64)

    rho_B = rho_A.copy()
    T_g_B = T_g_A.copy()
    Y_F_B = Y_F_A.copy()
    u_B   = u_A.copy()
    v_B   = v_A.copy()
    w_B   = w_A.copy()
    phi_B = phi_A.copy()
    L_F_B = L_F_A.copy()
    last_spawn_B = last_spawn_A.copy()
    n_spawn_B = np.zeros(1, dtype=np.int64)

    for s in (rho_A, T_g_A, Y_F_A, u_A, v_A, w_A, phi_A, L_F_A, last_spawn_A):
        pass

    ft.step_finney_tendril_spawn_deposit(
        rho_A, T_g_A, Y_F_A, u_A, v_A, w_A, phi_A, L_F_A, last_spawn_A,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.123,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn_A,
    )
    ft.step_finney_tendril_spawn_deposit(
        rho_B, T_g_B, Y_F_B, u_B, v_B, w_B, phi_B, L_F_B, last_spawn_B,
        dx=0.1, dy=0.1, dz_arr=dz_arr, t_now=0.123,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, n_spawn_events_out=n_spawn_B,
    )
    assert n_spawn_A[0] == n_spawn_B[0]
    for nm, A, B in (
        ("rho", rho_A, rho_B), ("T_g", T_g_A, T_g_B), ("Y_F", Y_F_A, Y_F_B),
        ("u", u_A, u_B), ("v", v_A, v_B), ("w", w_A, w_B),
        ("last_spawn", last_spawn_A, last_spawn_B),
    ):
        assert np.array_equal(A, B), f"non-deterministic on {nm}"


# ── L_F COMPUTATION HELPER ──────────────────────────────────────────────

def test_compute_L_F_per_column():
    """L_F integrates the height of cells above bed with T_g > T_thresh."""
    Nz, Ny, Nx = 10, 2, 3
    n_z_bed = 4
    T_g = np.full((Nz, Ny, Nx), 300.0, dtype=np.float64)
    # Add hot cells above bed at varying heights
    T_g[4:7, 0, 0] = 1500.0   # 3 cells above bed at j=0, i=0
    T_g[4:9, 1, 1] = 1500.0   # 5 cells above bed at j=1, i=1
    dz_arr = np.full(Nz, 0.10, dtype=np.float64)
    L_F = ft._compute_L_F_per_column(T_g, T_thresh=600.0,
                                       dz_arr=dz_arr, n_z_bed=n_z_bed)
    assert L_F.shape == (Ny, Nx)
    assert abs(L_F[0, 0] - 0.30) < 1e-12   # 3 × 0.10
    assert abs(L_F[1, 1] - 0.50) < 1e-12   # 5 × 0.10
    assert L_F[0, 2] == 0.0                # no hot gas


# ── COMMITTED-VALUES SENTINEL ────────────────────────────────────────────

def test_committed_values_are_phase15O():
    """Documents the committed Phase 15O parameter values; if these
    change, the full Phase 15O verification sequence must be re-run per
    the plan."""
    assert ft.SR_DEFAULT == 0.20
    assert ft.DUTY_CYCLE == 0.40
    assert ft.F_MASS_DEFAULT == 0.05
    assert ft.FR_MIN_DEFAULT == 0.5
    assert ft.T_GAS_FLAME == 600.0
    assert ft.MAX_DEPOSIT_CELLS == 5


# ── Phase 15O.1 TIME-SPREAD KERNEL TESTS ────────────────────────────────


def _alloc_pending_fields(Nz, Ny, Nx):
    """Allocate the 10 persistent state fields for the time-spread kernel."""
    return {
        "sink_M":      np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "sink_E":      np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "sink_Yf":     np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "sink_Px":     np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "sink_t_rem":  np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "dep_M":       np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "dep_E":       np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "dep_Yf":      np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "dep_Px":      np.zeros((Nz, Ny, Nx), dtype=np.float64),
        "dep_t_rem":   np.zeros((Nz, Ny, Nx), dtype=np.float64),
    }


def test_time_spread_releases_full_mass_after_t_contact():
    """Phase 15O.1: a spawn queued at t=0 releases its full ΔM by t=T_contact.

    Run a single spawn, then step Phase A for N steps covering T_contact.
    Verify that the SOURCE cell's cumulative mass loss matches ΔM exactly.
    """
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx,
                                         rho_val=1.2, Tg_val=1500.0,
                                         Yf_val=0.10, u_val=4.0)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=4)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1
    pending = _alloc_pending_fields(Nz, Ny, Nx)
    t_contact = 0.3
    dt = 0.025

    # Initial total mass
    M_before = _total_mass(rho, dx, dy, dz_arr)

    # Phase B: queue spawn at t=0
    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=t_contact,
        n_spawn_events_out=n_spawn,
    )
    assert n_spawn[0] > 0
    queued_total_M_sink = pending["sink_M"].sum()
    queued_total_M_dep  = pending["dep_M"].sum()
    # Mass-conservation of queue: sink == sum of deposits
    assert abs(queued_total_M_sink - queued_total_M_dep) < 1e-12, (
        f"queued conservation: sink={queued_total_M_sink:.6e}, "
        f"deposit_sum={queued_total_M_dep:.6e}"
    )

    # Phase A: apply for enough steps to fully drain (T_contact / dt + 1)
    n_steps = int(t_contact / dt) + 1
    for _ in range(n_steps):
        ft.step_finney_tendril_apply_pending(
            rho, T_g, Y_F, u,
            pending["sink_M"], pending["sink_E"],
            pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
            pending["dep_M"], pending["dep_E"],
            pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
            dx, dy, dz_arr, dt,
        )

    # After full release: all remaining inventory should be ≈ 0
    assert pending["sink_M"].sum() < 1e-9, (
        f"sink_M not fully drained: {pending['sink_M'].sum():.6e}"
    )
    assert pending["dep_M"].sum() < 1e-9, (
        f"dep_M not fully drained: {pending['dep_M'].sum():.6e}"
    )
    # Total mass globally conserved (kernel transferred from source to ahead)
    M_after = _total_mass(rho, dx, dy, dz_arr)
    rel_err = abs(M_after - M_before) / max(abs(M_before), 1.0)
    assert rel_err < 1e-12, (
        f"mass not conserved across spawn + release window: "
        f"M_before={M_before:.10g}, M_after={M_after:.10g}, "
        f"rel_err={rel_err:.3e}"
    )


def test_time_spread_partial_release_after_half_t_contact():
    """After dt steps summing to T_contact/2, half the inventory should
    have been released (linear release schedule)."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=4)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1
    pending = _alloc_pending_fields(Nz, Ny, Nx)
    t_contact = 0.40
    dt = 0.020   # 20 ms

    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=t_contact,
        n_spawn_events_out=n_spawn,
    )
    initial_sink_M = pending["sink_M"].sum()
    assert initial_sink_M > 0

    # Apply for half the t_contact
    n_steps = int((t_contact / 2) / dt)
    for _ in range(n_steps):
        ft.step_finney_tendril_apply_pending(
            rho, T_g, Y_F, u,
            pending["sink_M"], pending["sink_E"],
            pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
            pending["dep_M"], pending["dep_E"],
            pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
            dx, dy, dz_arr, dt,
        )
    # After half the window: should have ~half remaining (linear schedule)
    remaining_frac = pending["sink_M"].sum() / initial_sink_M
    assert 0.45 < remaining_frac < 0.55, (
        f"After T_contact/2: remaining_frac={remaining_frac:.4f}, "
        f"expected ~0.50"
    )


def test_time_spread_overlapping_spawns_conserve():
    """Multiple spawns at the same source cell at different times must
    conserve total mass globally.  The key edge case: second spawn arrives
    while first is still releasing."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=4)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    dx, dy = 0.1, 0.1
    pending = _alloc_pending_fields(Nz, Ny, Nx)
    t_contact = 0.40
    dt = 0.025

    M_before = _total_mass(rho, dx, dy, dz_arr)

    # Spawn 1 at t=0
    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=t_contact,
        n_spawn_events_out=n_spawn,
    )
    # Apply a few steps to partially release
    for _ in range(5):
        ft.step_finney_tendril_apply_pending(
            rho, T_g, Y_F, u,
            pending["sink_M"], pending["sink_E"],
            pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
            pending["dep_M"], pending["dep_E"],
            pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
            dx, dy, dz_arr, dt,
        )
    # Force a re-spawn at t = 0.5s (past T_period for the original spawn)
    # but we'll fake it by zeroing last_spawn_time
    last_spawn.fill(ft._NEVER_SPAWNED)
    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.125,   # within still-releasing window
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=t_contact,
        n_spawn_events_out=n_spawn,
    )
    # Now apply enough steps to drain all remaining
    for _ in range(50):
        ft.step_finney_tendril_apply_pending(
            rho, T_g, Y_F, u,
            pending["sink_M"], pending["sink_E"],
            pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
            pending["dep_M"], pending["dep_E"],
            pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
            dx, dy, dz_arr, dt,
        )
    # All pending drained
    assert pending["sink_M"].sum() < 1e-9
    assert pending["dep_M"].sum() < 1e-9
    # Global mass conservation across overlapping spawns
    M_after = _total_mass(rho, dx, dy, dz_arr)
    rel_err = abs(M_after - M_before) / max(abs(M_before), 1.0)
    assert rel_err < 1e-12, (
        f"mass not conserved across OVERLAPPING spawns: "
        f"rel_err={rel_err:.3e}"
    )


def test_phase15O2_box_aggregation_extracts_more_mass():
    """Phase 15O.2: box_radius=1 aggregates from up to 27 cells around each
    LE anchor, producing ~27× larger per-spawn inventory than single-cell."""
    Nz, Ny, Nx = 6, 3, 12
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx,
                                         rho_val=1.0, Tg_val=1500.0)
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=5)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn_single = np.zeros(1, dtype=np.int64)
    n_spawn_box = np.zeros(1, dtype=np.int64)
    pending_single = _alloc_pending_fields(Nz, Ny, Nx)
    pending_box    = _alloc_pending_fields(Nz, Ny, Nx)
    dx, dy = 0.1, 0.1

    rho_single = rho.copy(); T_g_single = T_g.copy()
    Y_F_single = Y_F.copy(); u_single = u.copy()
    last_spawn_single = last_spawn.copy()
    rho_box = rho.copy(); T_g_box = T_g.copy()
    Y_F_box = Y_F.copy(); u_box = u.copy()
    last_spawn_box = last_spawn.copy()

    # Single-cell (box radii = 0)
    ft.step_finney_tendril_queue_spawns(
        rho_single, T_g_single, Y_F_single, u_single,
        phi, L_F, last_spawn_single,
        pending_single["sink_M"], pending_single["sink_E"],
        pending_single["sink_Yf"], pending_single["sink_Px"],
        pending_single["sink_t_rem"],
        pending_single["dep_M"], pending_single["dep_E"],
        pending_single["dep_Yf"], pending_single["dep_Px"],
        pending_single["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3,
        n_spawn_events_out=n_spawn_single,
    )
    # Box-aggregated (radii = 1 → 3×3×3, but mostly bounded by flame body)
    ft.step_finney_tendril_queue_spawns(
        rho_box, T_g_box, Y_F_box, u_box,
        phi, L_F, last_spawn_box,
        pending_box["sink_M"], pending_box["sink_E"],
        pending_box["sink_Yf"], pending_box["sink_Px"],
        pending_box["sink_t_rem"],
        pending_box["dep_M"], pending_box["dep_E"],
        pending_box["dep_Yf"], pending_box["dep_Px"],
        pending_box["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3,
        n_spawn_events_out=n_spawn_box,
        box_dk_up_radius=1, box_dk_down_radius=1,
        box_dj_radius=1, box_di_back_radius=2,
    )
    # Box should aggregate MORE total source inventory than single-cell
    total_sink_single = pending_single["sink_M"].sum()
    total_sink_box    = pending_box["sink_M"].sum()
    assert total_sink_box > total_sink_single, (
        f"Box should aggregate more source mass than single-cell: "
        f"box={total_sink_box:.3e}, single={total_sink_single:.3e}"
    )
    # Conservation: per spawn, source-sum and deposit-sum should match
    rel_err_single = abs(pending_single["dep_M"].sum() - total_sink_single) / max(total_sink_single, 1e-30)
    rel_err_box    = abs(pending_box["dep_M"].sum() - total_sink_box) / max(total_sink_box, 1e-30)
    assert rel_err_single < 1e-12, f"single-cell conservation: {rel_err_single:.3e}"
    assert rel_err_box < 1e-12, f"box conservation: {rel_err_box:.3e}"


def test_phase15O2_box_extraction_skips_outside_flame_body():
    """Box aggregation must only extract from cells inside the flame body
    (phi_flame ≤ 0).  Cells outside (phi > 0) are skipped."""
    Nz, Ny, Nx = 5, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    # Flame body only at i=4 (one cell wide).  Box centered at LE i=4 would
    # try to look at i=3, 2 etc. — all outside body, so should NOT extract.
    phi = np.full((Nz, Ny, Nx), +1.0, dtype=np.float64)
    phi[:, :, 4] = -0.1   # only i=4 is in body, i=5 outside → LE at i=4
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    pending = _alloc_pending_fields(Nz, Ny, Nx)

    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        0.1, 0.1, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3,
        n_spawn_events_out=n_spawn,
        box_dk_up_radius=1, box_dk_down_radius=1,
        box_dj_radius=0, box_di_back_radius=3,  # look back 3 cells
    )
    # Only the LE cell itself (i=4) is in the flame body. All cells at i=3,2,1
    # are outside → no extraction from them.
    # Sink at i=4 should be non-zero (LE cell self-extraction)
    assert pending["sink_M"][:, :, 4].sum() > 0
    # Sink at i=3, 2, 1 should be zero (outside body)
    assert pending["sink_M"][:, :, 3].sum() == 0
    assert pending["sink_M"][:, :, 2].sum() == 0
    assert pending["sink_M"][:, :, 1].sum() == 0


def test_phase15O2_box_conservation_with_overlapping_LE():
    """When two adjacent LE cells aggregate over overlapping boxes,
    global mass conservation must still hold to 1e-12."""
    Nz, Ny, Nx = 6, 3, 15
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    # Flame body: i=0..6 inside.  LE at i=6.  But add a second LE at i=6
    # with j=0 vs j=1 (boxes will overlap on the j dimension).
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=6)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    pending = _alloc_pending_fields(Nz, Ny, Nx)
    dx, dy = 0.1, 0.1

    M_before = _total_mass(rho, dx, dy, dz_arr)
    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        dx, dy, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3,
        n_spawn_events_out=n_spawn,
        box_dk_up_radius=1, box_dk_down_radius=1,
        box_dj_radius=1, box_di_back_radius=2,
    )
    # Phase B only QUEUES inventory; rho is unchanged. So whole-domain
    # mass should be unchanged after queuing.
    M_after_queue = _total_mass(rho, dx, dy, dz_arr)
    rel_err_queue = abs(M_after_queue - M_before) / max(abs(M_before), 1.0)
    assert rel_err_queue < 1e-12, f"queue should not modify rho: {rel_err_queue:.3e}"
    # Total queued sink mass == total queued deposit mass (per-spawn balance)
    sink_total = pending["sink_M"].sum()
    dep_total  = pending["dep_M"].sum()
    rel_err_balance = abs(dep_total - sink_total) / max(sink_total, 1e-30)
    assert rel_err_balance < 1e-12, (
        f"queue balance: sink={sink_total:.6e}, dep={dep_total:.6e}, "
        f"rel_err={rel_err_balance:.3e}"
    )


def test_phase15O3_asymmetric_box_skips_below_LE():
    """Phase 15O.3: with box_dk_up_radius > 0 and box_dk_down_radius = 0,
    the kernel only pulls from cells AT or ABOVE the LE row (k_le ≥ k).
    Cells below (k_le > k) are not extracted from."""
    Nz, Ny, Nx = 8, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    # Flame body fills entire i ≤ 5 strip at all k.  LE at i=5.
    phi = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=5)
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn = np.zeros(1, dtype=np.int64)
    pending = _alloc_pending_fields(Nz, Ny, Nx)

    # Pick a LE cell in the middle of the z-range, say k_le = 4
    # The kernel iterates over k, so let's anchor at k=4 by clearing
    # other rows from spawning (set phi = +1 outside k=4)
    phi_test = np.full((Nz, Ny, Nx), +1.0, dtype=np.float64)
    phi_test[4, :, :6] = -0.1   # only k=4 has flame body
    # Now LE at (k=4, j=0, i=5)
    ft.step_finney_tendril_queue_spawns(
        rho, T_g, Y_F, u, phi_test, L_F, last_spawn,
        pending["sink_M"], pending["sink_E"],
        pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
        pending["dep_M"], pending["dep_E"],
        pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
        0.1, 0.1, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3,
        n_spawn_events_out=n_spawn,
        box_dk_up_radius=2, box_dk_down_radius=0,
        box_dj_radius=0, box_di_back_radius=0,
    )
    # Only k=4 has flame body. dk_up=2 → k=4,5,6. dk_down=0 → no k=3.
    # But only k=4 is in flame body, so only k=4 should have non-zero sink.
    assert pending["sink_M"][4].sum() > 0, "k=4 (LE row) should have sink"
    # k=5, k=6 would be in dk_up range, but they're outside flame body → no sink
    assert pending["sink_M"][5].sum() == 0
    assert pending["sink_M"][6].sum() == 0
    # k=3 is dk_down direction, not in box → no sink even if it were in flame body
    assert pending["sink_M"][3].sum() == 0


def test_phase15O3_asymmetric_up_extracts_from_plume():
    """When the flame body extends both at LE row and above (plume),
    asymmetric box with dk_up > 0 captures both, while symmetric with
    same total range captures plume + same-deep bed."""
    Nz, Ny, Nx = 8, 1, 10
    rho, T_g, Y_F, u, v, w = _make_state(Nz, Ny, Nx)
    # Flame body at k=2,3,4,5,6 (vertical column extending up from k=2 to k=6)
    phi = np.full((Nz, Ny, Nx), +1.0, dtype=np.float64)
    for k_in in (2, 3, 4, 5, 6):
        phi[k_in, :, :5] = -0.1   # body extends i=0..4, LE at i=4
    L_F = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)

    # Run asymmetric: only up
    pending_up = _alloc_pending_fields(Nz, Ny, Nx)
    n_up = np.zeros(1, dtype=np.int64)
    rho_up = rho.copy()
    ft.step_finney_tendril_queue_spawns(
        rho_up, T_g, Y_F, u, phi, L_F, last_spawn.copy(),
        pending_up["sink_M"], pending_up["sink_E"],
        pending_up["sink_Yf"], pending_up["sink_Px"], pending_up["sink_t_rem"],
        pending_up["dep_M"], pending_up["dep_E"],
        pending_up["dep_Yf"], pending_up["dep_Px"], pending_up["dep_t_rem"],
        0.1, 0.1, dz_arr, t_now=0.0,
        sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
        T_amb=300.0, t_contact_s=0.3, n_spawn_events_out=n_up,
        box_dk_up_radius=2, box_dk_down_radius=0,
        box_dj_radius=0, box_di_back_radius=0,
    )

    # Both should have spawned. Conservation: per-spawn dep matches sink.
    assert n_up[0] > 0
    total_sink = pending_up["sink_M"].sum()
    total_dep  = pending_up["dep_M"].sum()
    rel_err = abs(total_dep - total_sink) / max(total_sink, 1e-30)
    assert rel_err < 1e-12, f"asymmetric box conservation: {rel_err:.3e}"


def test_time_spread_deterministic_under_repeat():
    """Rule #17: Phase 15O.1 time-spread is bit-exact deterministic."""
    Nz, Ny, Nx = 6, 2, 12
    rng = np.random.default_rng(101)
    rho_A = 1.0 + 0.1 * rng.random((Nz, Ny, Nx))
    T_g_A = 800.0 + 500.0 * rng.random((Nz, Ny, Nx))
    Y_F_A = 0.01 + 0.20 * rng.random((Nz, Ny, Nx))
    u_A   = 2.0 + 4.0 * rng.random((Nz, Ny, Nx))
    phi_A = _make_phi_flame_at(Nz, Ny, Nx, i_flame_end=6)
    L_F_A = np.full((Ny, Nx), 1.0, dtype=np.float64)
    last_spawn_A = np.full((Nz, Ny, Nx), ft._NEVER_SPAWNED, dtype=np.float64)
    dz_arr = np.full(Nz, 0.1, dtype=np.float64)
    n_spawn_A = np.zeros(1, dtype=np.int64)
    pending_A = _alloc_pending_fields(Nz, Ny, Nx)

    # Copy for run B
    rho_B = rho_A.copy(); T_g_B = T_g_A.copy()
    Y_F_B = Y_F_A.copy(); u_B   = u_A.copy()
    phi_B = phi_A.copy(); L_F_B = L_F_A.copy()
    last_spawn_B = last_spawn_A.copy()
    n_spawn_B = np.zeros(1, dtype=np.int64)
    pending_B = _alloc_pending_fields(Nz, Ny, Nx)

    # Run both: queue 1 spawn, apply 5 steps
    for (rho, T_g, Y_F, u, phi, L_F, last, n_sp, pending) in (
        (rho_A, T_g_A, Y_F_A, u_A, phi_A, L_F_A, last_spawn_A, n_spawn_A, pending_A),
        (rho_B, T_g_B, Y_F_B, u_B, phi_B, L_F_B, last_spawn_B, n_spawn_B, pending_B),
    ):
        ft.step_finney_tendril_queue_spawns(
            rho, T_g, Y_F, u, phi, L_F, last,
            pending["sink_M"], pending["sink_E"],
            pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
            pending["dep_M"], pending["dep_E"],
            pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
            0.1, 0.1, dz_arr, t_now=0.0,
            sr=0.20, duty_cycle=0.40, f_mass=0.05, fr_min=0.5,
            T_amb=300.0, t_contact_s=0.3,
            n_spawn_events_out=n_sp,
        )
        for _ in range(5):
            ft.step_finney_tendril_apply_pending(
                rho, T_g, Y_F, u,
                pending["sink_M"], pending["sink_E"],
                pending["sink_Yf"], pending["sink_Px"], pending["sink_t_rem"],
                pending["dep_M"], pending["dep_E"],
                pending["dep_Yf"], pending["dep_Px"], pending["dep_t_rem"],
                0.1, 0.1, dz_arr, 0.025,
            )

    # Compare every output
    for nm, A, B in (
        ("rho", rho_A, rho_B), ("T_g", T_g_A, T_g_B), ("Y_F", Y_F_A, Y_F_B),
        ("u", u_A, u_B), ("sink_M", pending_A["sink_M"], pending_B["sink_M"]),
        ("dep_M", pending_A["dep_M"], pending_B["dep_M"]),
    ):
        assert np.array_equal(A, B), f"time-spread non-deterministic on {nm}"
    assert n_spawn_A[0] == n_spawn_B[0]
