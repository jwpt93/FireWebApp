"""Phase 16-0 unit tests — shared Lagrangian particle infrastructure.

Tests the reusable primitives consumed by finney_lagrangian_3d (today)
and by lagrangian_bed_3d / lagrangian_firebrand_3d (future).

Coverage:
  - Locator correctness on non-uniform z meshes
  - Slot allocator (first-free + full-buffer)
  - Buffer factory (shape + dtype + zero-init)
  - Kinematic step:
      * Bit-exact determinism (Rule #17)
      * Pure-drag: particle velocity converges to gas velocity
      * Pure-buoyancy: light particle rises, heavy particle sinks
      * Pure-gravity: particle accelerates downward
      * No forces (all flags off): particle moves at constant velocity
      * Periodic-y BC: wraps in-domain
      * Non-periodic-y BC: retires on exit
      * x and z exits always retire
      * Age increments only for retained particles
  - rho_p sphere helper: density = m / (π/6 · d³), zero for dead slots
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from model_outdoor.physics_3d import lagrangian_particles_3d as lp


# ── Helpers ────────────────────────────────────────────────────────────


def _make_grid(Nz=10, Ny=4, Nx=20, dx=0.1, dy=0.1, dz=0.1):
    dz_arr = np.full(Nz, dz, dtype=np.float64)
    z_face = np.zeros(Nz + 1, dtype=np.float64)
    for k in range(1, Nz + 1):
        z_face[k] = z_face[k - 1] + dz_arr[k - 1]
    return dx, dy, dz_arr, z_face


def _make_uniform_gas(Nz, Ny, Nx, rho=1.2, u=0.0, v=0.0, w=0.0):
    return (
        np.full((Nz, Ny, Nx), rho, dtype=np.float64),
        np.full((Nz, Ny, Nx), u,   dtype=np.float64),
        np.full((Nz, Ny, Nx), v,   dtype=np.float64),
        np.full((Nz, Ny, Nx), w,   dtype=np.float64),
    )


def _seed_particle(buf, rho_p_arr, idx, x, y, z, u=0.0, v=0.0, w=0.0, rho_p=1.2):
    buf["x"][idx]     = x
    buf["y"][idx]     = y
    buf["z"][idx]     = z
    buf["u"][idx]     = u
    buf["v"][idx]     = v
    buf["w"][idx]     = w
    buf["alive"][idx] = lp.ALIVE_TRUE
    buf["age"][idx]   = 0.0
    rho_p_arr[idx]    = rho_p


# ── Locator ────────────────────────────────────────────────────────────


def test_locate_k_from_z_uniform():
    z_face = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    Nz = 5
    assert lp.locate_k_from_z(0.05, z_face, Nz) == 0
    assert lp.locate_k_from_z(0.15, z_face, Nz) == 1
    assert lp.locate_k_from_z(0.25, z_face, Nz) == 2
    assert lp.locate_k_from_z(0.45, z_face, Nz) == 4
    # On boundary: belongs to upper cell (z >= face[k] AND z < face[k+1])
    assert lp.locate_k_from_z(0.0, z_face, Nz) == 0
    assert lp.locate_k_from_z(0.1, z_face, Nz) == 1
    # Out of domain
    assert lp.locate_k_from_z(-0.001, z_face, Nz) == -1
    assert lp.locate_k_from_z(0.5,   z_face, Nz) == -1
    assert lp.locate_k_from_z(0.6,   z_face, Nz) == -1


def test_locate_k_from_z_nonuniform():
    """Non-uniform z (BL-style): first few cells small, then growing."""
    z_face = np.array([0.0, 0.005, 0.012, 0.025, 0.050, 0.10, 0.20], dtype=np.float64)
    Nz = 6
    assert lp.locate_k_from_z(0.001, z_face, Nz) == 0     # wall cell
    assert lp.locate_k_from_z(0.008, z_face, Nz) == 1
    assert lp.locate_k_from_z(0.030, z_face, Nz) == 3
    assert lp.locate_k_from_z(0.099, z_face, Nz) == 4
    assert lp.locate_k_from_z(0.199, z_face, Nz) == 5
    assert lp.locate_k_from_z(0.20,  z_face, Nz) == -1    # exactly at top: out


def test_locate_cell_3d():
    z_face = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    i, j, k = lp.locate_cell(0.25, 0.15, 0.18, dx=0.10, dy=0.10,
                             z_face=z_face, Nz=3, Nx=4, Ny=4)
    assert (i, j, k) == (2, 1, 1)
    # Negative x → exit
    i, j, k = lp.locate_cell(-0.05, 0.15, 0.18, dx=0.10, dy=0.10,
                             z_face=z_face, Nz=3, Nx=4, Ny=4)
    assert i == -1
    # x exactly at Lx → exit (open right edge)
    i, j, k = lp.locate_cell(0.40, 0.15, 0.18, dx=0.10, dy=0.10,
                             z_face=z_face, Nz=3, Nx=4, Ny=4)
    assert i == -1


# ── Slot allocator ─────────────────────────────────────────────────────


def test_alloc_dead_slot_empty_buffer():
    alive = np.zeros(8, dtype=np.int8)
    assert lp.alloc_dead_slot(alive) == 0


def test_alloc_dead_slot_partially_full():
    alive = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    assert lp.alloc_dead_slot(alive) == 2


def test_alloc_dead_slot_full_returns_minus_one():
    alive = np.ones(4, dtype=np.int8)
    assert lp.alloc_dead_slot(alive) == -1


# ── Buffer factory ─────────────────────────────────────────────────────


def test_allocate_kinematic_buffers_shape_and_dtype():
    buf = lp.allocate_kinematic_buffers(16)
    assert set(buf.keys()) == {"x", "y", "z", "u", "v", "w", "alive", "age"}
    for k in ("x", "y", "z", "u", "v", "w", "age"):
        assert buf[k].dtype == np.float64
        assert buf[k].shape == (16,)
        assert (buf[k] == 0.0).all()
    assert buf["alive"].dtype == np.int8
    assert buf["alive"].shape == (16,)
    assert (buf["alive"] == 0).all()


def test_allocate_kinematic_buffers_rejects_negative():
    with pytest.raises(ValueError):
        lp.allocate_kinematic_buffers(-1)


def test_allocate_kinematic_buffers_zero_is_ok():
    buf = lp.allocate_kinematic_buffers(0)
    assert buf["x"].shape == (0,)


# ── Kinematic step ─────────────────────────────────────────────────────


def _step(buf, rho_p, rho, ug, vg, wg, grid, dt, **kw):
    n_alive = np.zeros(1, dtype=np.int64)
    n_exit  = np.zeros(1, dtype=np.int64)
    dx, dy, dz_arr, z_face = grid
    lp.step_kinematics(
        buf["x"], buf["y"], buf["z"],
        buf["u"], buf["v"], buf["w"],
        buf["alive"], buf["age"],
        rho_p,
        rho, ug, vg, wg,
        dx, dy, dz_arr, z_face,
        kw.get("d_p", 0.075),
        kw.get("C_D", 1.0),
        kw.get("use_drag", False),
        kw.get("use_buoyancy", False),
        kw.get("use_gravity", False),
        kw.get("y_periodic", True),
        dt,
        n_alive, n_exit,
    )
    return int(n_alive[0]), int(n_exit[0])


def test_no_forces_moves_constant_velocity():
    Nz, Ny, Nx = 10, 4, 20
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.zeros(4)
    _seed_particle(buf, rho_p, 0, 0.5, 0.15, 0.45, u=2.0, v=0.0, w=0.0)
    dt = 0.01
    for _ in range(10):
        _step(buf, rho_p, rho, ug, vg, wg, grid, dt)
    # x advanced by u·n·dt = 2.0 · 0.1 = 0.2
    assert buf["x"][0] == pytest.approx(0.5 + 2.0 * 10 * dt, rel=1e-12)
    assert buf["u"][0] == pytest.approx(2.0)
    # Age = 10·dt
    assert buf["age"][0] == pytest.approx(10 * dt)


def test_pure_drag_converges_to_gas_velocity():
    """Stationary particle in moving gas, drag only → tracks gas velocity."""
    Nz, Ny, Nx = 10, 4, 60
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, u=5.0)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)   # match gas → no buoyancy if it were on
    _seed_particle(buf, rho_p, 0, 0.5, 0.15, 0.45, u=0.0, rho_p=1.2)
    dt = 0.001
    for _ in range(500):
        if buf["alive"][0] == 0:
            break
        _step(buf, rho_p, rho, ug, vg, wg, grid, dt,
              use_drag=True, d_p=0.10, C_D=2.0)
    # Particle should match gas u closely (within 5%) if still alive
    if buf["alive"][0]:
        assert abs(buf["u"][0] - 5.0) < 0.25


def test_pure_buoyancy_light_particle_rises():
    """ρ_p < ρ_g → upward buoyancy → positive w gain."""
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, rho=1.2)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.zeros(4)
    _seed_particle(buf, rho_p, 0, 0.25, 0.15, 0.45, w=0.0, rho_p=0.2)
    w0 = float(buf["w"][0])
    _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.001,
          use_buoyancy=True)
    assert buf["w"][0] > w0
    # Expected a_buoy = g · (1.2 - 0.2)/0.2 = 49 m/s² → Δw=49·0.001=0.049
    assert buf["w"][0] == pytest.approx(0.0 + 9.81 * (1.2 - 0.2) / 0.2 * 0.001,
                                        rel=1e-10)


def test_pure_buoyancy_heavy_particle_sinks():
    """ρ_p > ρ_g → downward buoyancy → negative w gain."""
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, rho=1.2)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.zeros(4)
    _seed_particle(buf, rho_p, 0, 0.25, 0.15, 0.55, w=0.0, rho_p=10.0)
    _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.001, use_buoyancy=True)
    assert buf["w"][0] < 0.0


def test_pure_gravity_accelerates_downward():
    Nz, Ny, Nx = 20, 4, 5
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.0)
    _seed_particle(buf, rho_p, 0, 0.25, 0.15, 1.95, w=0.0)
    _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.001, use_gravity=True)
    assert buf["w"][0] == pytest.approx(-9.81 * 0.001, rel=1e-12)


def test_periodic_y_wraps_in_domain():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx, dy=0.1)   # Ly = 4·0.1 = 0.4
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, v=10.0)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    _seed_particle(buf, rho_p, 0, 0.25, 0.35, 0.45, v=10.0, rho_p=1.2)
    # Particle at y=0.35, dt=0.01 → next position y=0.35+0.1=0.45 → out
    # With periodic-y: wraps to 0.05
    _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01, y_periodic=True)
    assert buf["alive"][0] == lp.ALIVE_TRUE
    assert 0.0 <= buf["y"][0] < 0.4


def test_non_periodic_y_retires_on_exit():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx, dy=0.1)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, v=10.0)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    _seed_particle(buf, rho_p, 0, 0.25, 0.35, 0.45, v=10.0, rho_p=1.2)
    n_alive, n_exit = _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01,
                            y_periodic=False)
    assert n_exit == 1
    assert buf["alive"][0] == lp.ALIVE_FALSE


def test_x_exit_always_retires():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)  # Lx = 5·0.1 = 0.5
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    _seed_particle(buf, rho_p, 0, 0.45, 0.15, 0.45, u=20.0, rho_p=1.2)
    n_alive, n_exit = _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01)
    assert n_exit == 1
    assert buf["alive"][0] == lp.ALIVE_FALSE


def test_z_exit_always_retires():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)  # Lz = 10·0.1 = 1.0
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    _seed_particle(buf, rho_p, 0, 0.25, 0.15, 0.95, w=20.0, rho_p=1.2)
    n_alive, n_exit = _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01)
    assert n_exit == 1
    assert buf["alive"][0] == lp.ALIVE_FALSE


def test_age_only_advances_for_retained():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    # P0 stays in-domain; P1 exits
    _seed_particle(buf, rho_p, 0, 0.25, 0.15, 0.45, u=0.0, rho_p=1.2)
    _seed_particle(buf, rho_p, 1, 0.45, 0.15, 0.45, u=20.0, rho_p=1.2)
    n_alive, n_exit = _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01)
    assert n_alive == 1 and n_exit == 1
    assert buf["age"][0] == pytest.approx(0.01)
    # Exited slot age is whatever it was at exit (we don't reset; it's dead state)
    # Just verify it didn't advance past the exit step
    assert buf["age"][1] == 0.0


def test_dead_particles_skipped():
    Nz, Ny, Nx = 10, 4, 5
    grid = _make_grid(Nz, Ny, Nx)
    rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx)
    buf = lp.allocate_kinematic_buffers(4)
    rho_p = np.full(4, 1.2)
    # Place dead particles with bogus position; verify no crash
    buf["x"][:] = -999.0
    buf["alive"][:] = lp.ALIVE_FALSE
    n_alive, n_exit = _step(buf, rho_p, rho, ug, vg, wg, grid, dt=0.01)
    assert n_alive == 0 and n_exit == 0


# ── Determinism (Rule #17) ─────────────────────────────────────────────


def test_kinematic_step_bit_exact_under_repeat():
    Nz, Ny, Nx = 10, 4, 20
    grid = _make_grid(Nz, Ny, Nx)

    def fresh():
        rho, ug, vg, wg = _make_uniform_gas(Nz, Ny, Nx, u=4.0, rho=1.2)
        buf = lp.allocate_kinematic_buffers(16)
        rho_p = np.zeros(16)
        for p in range(5):
            _seed_particle(buf, rho_p, p,
                           x=0.3 + 0.1 * p, y=0.05 + 0.05 * p, z=0.4 + 0.02 * p,
                           u=1.0 + 0.1 * p, v=0.0, w=0.05,
                           rho_p=0.4 + 0.1 * p)
        return rho, ug, vg, wg, buf, rho_p

    rho_a, ug_a, vg_a, wg_a, buf_a, rho_p_a = fresh()
    rho_b, ug_b, vg_b, wg_b, buf_b, rho_p_b = fresh()

    for _ in range(50):
        _step(buf_a, rho_p_a, rho_a, ug_a, vg_a, wg_a, grid, dt=0.001,
              use_drag=True, use_buoyancy=True)
        _step(buf_b, rho_p_b, rho_b, ug_b, vg_b, wg_b, grid, dt=0.001,
              use_drag=True, use_buoyancy=True)

    for key in ("x", "y", "z", "u", "v", "w", "age"):
        assert np.array_equal(buf_a[key], buf_b[key]), (
            f"buf.{key} not bit-exact under repeat")
    assert np.array_equal(buf_a["alive"], buf_b["alive"])


# ── rho_p sphere helper ────────────────────────────────────────────────


def test_compute_rho_p_sphere_uniform():
    N_max = 8
    d_p = 0.10
    V_p = (math.pi / 6.0) * d_p**3
    m = np.zeros(N_max)
    alive = np.zeros(N_max, dtype=np.int8)
    rho_p_out = np.zeros(N_max)
    # 4 alive particles with m=1e-3 → rho = 1e-3 / V_p
    for p in range(4):
        m[p] = 1.0e-3
        alive[p] = lp.ALIVE_TRUE
    lp.compute_rho_p_sphere(m, d_p, alive, rho_p_out)
    expected = 1.0e-3 / V_p
    for p in range(4):
        assert rho_p_out[p] == pytest.approx(expected, rel=1e-12)
    for p in range(4, N_max):
        # dead slots → rho_p = 0
        assert rho_p_out[p] == 0.0


def test_compute_rho_p_sphere_zero_d_p():
    """d_p=0 → all densities = 0 (no division-by-zero crash)."""
    N_max = 4
    m = np.full(N_max, 1.0e-3)
    alive = np.full(N_max, lp.ALIVE_TRUE, dtype=np.int8)
    rho_p_out = np.zeros(N_max)
    lp.compute_rho_p_sphere(m, 0.0, alive, rho_p_out)
    assert (rho_p_out == 0.0).all()
