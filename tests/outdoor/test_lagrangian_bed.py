"""Phase 16 unit tests — Lagrangian bed-particle module.

Tests enforce:
  - Init mass-balance: Σ particles = Eulerian-equivalent inventory
  - Sub-cell position determinism (Rule #17)
  - Drying mass conservation: Δm_water_particle = Δm_water_gas-aggregate
  - Pyrolysis mass conservation: Δm_solid_particle = (Δm_volatile_gas +
    Δm_char_particle), with eta + char_yield = 1
  - Moisture gate: when m_water/m_water_0 > 1%, pyrolysis blocked
  - T_s convection: T_g > T_s → T_s rises; T_g = T_s → no T_s change
  - T_s endothermic drying cools particle
  - Burnout retirement: particle dies when total mass ≤ threshold
  - Bit-exact determinism (Rule #17)
  - Empty buffer / no-bed-cell init → 0 particles allocated
  - Buffer overflow detection raises ValueError
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from model_outdoor.physics_3d import lagrangian_bed_3d as lb
from model_outdoor.physics_3d import lagrangian_particles_3d as lp
from model_outdoor.physics_3d.pyrolysis_3d import (
    L_VAP_WATER, ETA_MD2004, CHAR_YIELD_MD2004,
)


# ── Helpers ────────────────────────────────────────────────────────────


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


def _make_uniform_gas(Nz, Ny, Nx, T_g=300.0, Y_O2=0.23):
    return (np.full((Nz, Ny, Nx), T_g, dtype=np.float64),
            np.full((Nz, Ny, Nx), Y_O2, dtype=np.float64))


def _make_source_arrays(Nz, Ny, Nx):
    return (
        np.zeros((Nz, Ny, Nx)),  # S_pyro
        np.zeros((Nz, Ny, Nx)),  # S_drying
        np.zeros((Nz, Ny, Nx)),  # Q_pyro
        np.zeros((Nz, Ny, Nx)),  # Q_drying
        np.zeros((Nz, Ny, Nx)),  # Y_F_source
        np.zeros((Nz, Ny, Nx)),  # Q_char
        np.zeros((Nz, Ny, Nx)),  # Q_smold
        np.zeros((Nz, Ny, Nx)),  # Q_g_conv
    )


def _run_step(buf, T_g, Y_O2, grid, dt=0.001,
              h_conv=25.0, rho_solid_true=380.0, cp_solid=1500.0,
              eps_solid=0.9, T_amb=300.0,
              view_factor=1.0,
              view_factor_geometric=False, h_bed=1.0, kappa_bed_eff=0.0,
              do_drying=True, do_pyrolysis=True,
              do_char_ox=True, do_smolder=True,
              drying_mode=0,
              Q_solid_ext=None, n_per_cell_for_split=1):
    dx, dy, dz_arr, z_face = grid
    Nz, Ny, Nx = T_g.shape
    Sp, Sd, Qp, Qd, YFs, Qch, Qsm, Qgc = _make_source_arrays(Nz, Ny, Nx)
    if Q_solid_ext is None:
        Q_solid_ext = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    n_alive = np.zeros(1, dtype=np.int64)
    n_burned = np.zeros(1, dtype=np.int64)
    diag_max = np.zeros(16, dtype=np.float64)
    lb.step_bed_particles(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"], buf["m_char"], buf["T_s"],
        buf["m_water_0"], buf["sav"],
        T_g, Y_O2,
        Q_solid_ext, int(n_per_cell_for_split),
        Sp, Sd, Qp, Qd, YFs, Qch, Qsm, Qgc,
        dx, dy, dz_arr, z_face,
        h_conv, rho_solid_true, cp_solid,
        eps_solid, T_amb,
        view_factor, view_factor_geometric, h_bed, kappa_bed_eff, dt,
        do_drying, do_pyrolysis, do_char_ox, do_smolder,
        int(drying_mode),
        n_alive, n_burned, diag_max,
    )
    return Sp, Sd, Qp, Qd, YFs, Qch, Qsm, Qgc, int(n_alive[0]), int(n_burned[0])


# ── Buffer allocation ──────────────────────────────────────────────────


def test_allocate_bed_buffers_includes_bed_state():
    buf = lb.allocate_bed_particle_buffers(8)
    for key in ("x", "y", "z", "u", "v", "w", "alive", "age",
                "m_solid", "m_water", "m_char", "T_s",
                "m_solid_0", "m_water_0", "sav"):
        assert key in buf, f"missing key {key}"
    for key in ("m_solid", "m_water", "m_char", "T_s", "m_solid_0",
                "m_water_0", "sav"):
        assert buf[key].dtype == np.float64
        assert buf[key].shape == (8,)
        assert (buf[key] == 0.0).all()


def test_allocate_bed_buffers_rejects_negative():
    with pytest.raises(ValueError):
        lb.allocate_bed_particle_buffers(-1)


# ── Initialisation ─────────────────────────────────────────────────────


def test_init_total_mass_matches_eulerian():
    """Σ_p m_solid_p = ρ_b · Σ_cell V_cell (sum over α_s > 0 cells).

    ρ_b is the BULK density (kg/m³ averaged over cell volume); the α_s
    > 0 mask just selects which cells get particles.  Mass per cell
    is ρ_b × V_cell, NOT multiplied by α_s — matches Eulerian
    state.m_hemi = ρ_b convention in spread_3d.
    """
    Nz, Ny, Nx = 6, 2, 8
    n_z_bed = 4
    rho_b = 1.07
    M = 0.04
    a_s_val = 0.01
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=2, i_hi=6,
                                       a_s_val=a_s_val)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 8
    n_bed_cells = int((alpha > 0).sum())
    N_max = n_bed_cells * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=rho_b, moisture_frac=M,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n == n_bed_cells * n_per_cell
    # Expected total m_solid = ρ_b · V_cell · n_bed_cells (mask, NOT α_s)
    V_per_cell = dx * dy * dz_arr[0]   # uniform dz here
    expected_m_solid = rho_b * V_per_cell * n_bed_cells
    assert buf["m_solid"].sum() == pytest.approx(expected_m_solid, rel=1e-12)
    # Expected total m_water = M * m_solid
    assert buf["m_water"].sum() == pytest.approx(M * expected_m_solid, rel=1e-12)
    # All particles at T_amb
    assert (buf["T_s"][:n] == 300.0).all()
    # SAV default
    assert (buf["sav"][:n] == lb.SAV_GRASS_DEFAULT).all()


def test_init_positions_inside_bed_cells():
    """All particles fall inside their assigned cell's geometry."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 4
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=0, i_hi=Nx)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 12
    buf = lb.allocate_bed_particle_buffers(int((alpha > 0).sum()) * n_per_cell)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.04,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    Lx, Ly = dx * Nx, dy * Ny
    Lz_bed = float(dz_arr[:n_z_bed].sum())
    for p in range(n):
        assert 0.0 <= buf["x"][p] < Lx
        assert 0.0 <= buf["y"][p] < Ly
        assert 0.0 <= buf["z"][p] < Lz_bed


def test_init_deterministic_under_repeat():
    Nz, Ny, Nx = 5, 2, 4
    n_z_bed = 3
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, _ = _make_grid(Nz, Ny, Nx)
    n_per_cell = 10
    N_max = int((alpha > 0).sum()) * n_per_cell

    buf_a = lb.allocate_bed_particle_buffers(N_max)
    buf_b = lb.allocate_bed_particle_buffers(N_max)
    n_a = lb.initialize_bed_particles_from_alpha_s(
        buf_a, alpha, rho_b_dry=1.07, moisture_frac=0.04,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    n_b = lb.initialize_bed_particles_from_alpha_s(
        buf_b, alpha, rho_b_dry=1.07, moisture_frac=0.04,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n_a == n_b
    for key in ("x", "y", "z", "m_solid", "m_water", "T_s"):
        assert np.array_equal(buf_a[key], buf_b[key]), f"{key} not bit-exact"


def test_init_empty_alpha_allocates_zero():
    Nz, Ny, Nx = 4, 2, 4
    alpha = np.zeros((Nz, Ny, Nx))
    dx, dy, dz_arr, _ = _make_grid(Nz, Ny, Nx)
    buf = lb.allocate_bed_particle_buffers(8)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.04,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=4, n_per_cell=2)
    assert n == 0
    assert int(buf["alive"].sum()) == 0


def test_init_buffer_overflow_raises():
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 4
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, _ = _make_grid(Nz, Ny, Nx)
    # Need 4*2*4=32 cells * 5 particles = 160 slots; only allocate 10.
    buf = lb.allocate_bed_particle_buffers(10)
    with pytest.raises(ValueError, match="Buffer too small"):
        lb.initialize_bed_particles_from_alpha_s(
            buf, alpha, rho_b_dry=1.07, moisture_frac=0.04,
            T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
            n_z_bed=n_z_bed, n_per_cell=5)


# ── Drying conservation ───────────────────────────────────────────────


def test_drying_mass_conserves_particle_to_gas():
    """Δm_water_particle = Δm_water_gas-aggregate · V_total over a step."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.10,
        T_amb=400.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=400.0)
    mw_before = buf["m_water"].sum()
    dt = 0.05
    Sp, Sd, Qp, Qd, YFs, _, _, _, _, _ = _run_step(buf, T_g, Y_O2,
                                          (dx, dy, dz_arr, z_face), dt=dt)
    mw_after = buf["m_water"].sum()
    dm_p = mw_before - mw_after
    # Volumetric drying source: dm_gas = Σ_cell S_drying[k,j,i] * V_cell * dt
    V_cell = dx * dy * dz_arr[0]
    dm_g = float(Sd.sum() * V_cell * dt)
    assert dm_g == pytest.approx(dm_p, rel=1e-10)


def test_drying_endothermic_cools_particle():
    """When T_s ≥ ambient and there's water, T_s should decrease from evap cooling."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.20,
        T_amb=400.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g = np.full((Nz, Ny, Nx), 400.0)
    Y_O2 = np.full((Nz, Ny, Nx), 0.23)
    T_s_before = float(buf["T_s"][0])
    # Single step with T_g = T_s → no convection, but drying should still cool
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01)
    T_s_after = float(buf["T_s"][0])
    assert T_s_after < T_s_before


# ── Pyrolysis conservation ────────────────────────────────────────────


def test_pyrolysis_mass_conserves_dry_to_volatile_plus_char():
    """Δm_solid_particle = (volatile_emitted_via_S_pyro) + Δm_char_particle.

    Set moisture_frac=0 to disable the moisture gate so pyrolysis fires.
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=700.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g = np.full((Nz, Ny, Nx), 700.0)
    Y_O2 = np.full((Nz, Ny, Nx), 0.0)   # Disable R_op to test thermal-only path
    ms_before = buf["m_solid"].sum()
    mc_before = buf["m_char"].sum()
    dt = 0.01
    Sp, Sd, Qp, Qd, YFs, _, _, _, _, _ = _run_step(buf, T_g, Y_O2,
                                          (dx, dy, dz_arr, z_face), dt=dt)
    ms_after = buf["m_solid"].sum()
    mc_after = buf["m_char"].sum()
    dm_solid = ms_before - ms_after        # mass lost from particles
    dm_char  = mc_after  - mc_before       # char gained by particles
    V_cell = dx * dy * dz_arr[0]
    dm_volatile_gas = float(Sp.sum() * V_cell * dt)  # to gas
    # ETA + CHAR_YIELD = 1.  dm_solid = dm_volatile + dm_char.
    assert dm_volatile_gas + dm_char == pytest.approx(dm_solid, rel=1e-10)
    # Per ETA_MD2004 split:
    assert dm_volatile_gas == pytest.approx(ETA_MD2004 * dm_solid, rel=1e-10)
    assert dm_char         == pytest.approx(CHAR_YIELD_MD2004 * dm_solid, rel=1e-10)


def test_moisture_gate_blocks_pyrolysis_when_wet():
    """High moisture → pyrolysis HEAVILY suppressed (linear soft gate).

    Phase 16 (2026-06-18): changed from hard-cutoff (1 - 100·wet) to
    linear (1 - wet) to allow modest pyrolysis at moderate M.  At fully-
    saturated (wet=1.0), gate=0 → pyrolysis still blocked initially.
    Tiny amount can fire as wet drops slightly via drying — the test
    now checks for ≤ 1% mass loss vs the dry-case full burn.
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.5,   # very wet
        T_amb=700.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=700.0)
    ms_before = float(buf["m_solid"].sum())
    Sp, *_ = _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001,
                       do_char_ox=False, do_smolder=False)
    ms_after = float(buf["m_solid"].sum())
    # Wet fuel: less than 1% pyrolysis after one step
    frac_lost = (ms_before - ms_after) / ms_before
    assert frac_lost < 0.01, f"wet pyrolysis ran too much: {frac_lost*100:.2f}%"


def test_cold_particle_negligible_pyrolysis():
    """At T_s=300 K the Arrhenius rate is < 1 ppm of m_solid per step.

    Not strictly zero (Arrhenius never is), but functionally negligible
    relative to bed inventory.  Matches Eulerian step_pyrolysis_md2004
    behaviour at the same T.
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=300.0)
    ms_before = buf["m_solid"].sum()
    Sp, *_ = _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01)
    ms_after = buf["m_solid"].sum()
    # Δm should be < 1e-6 × initial mass (vanishingly small)
    assert (ms_before - ms_after) / ms_before < 1e-6


# ── T_s convective heating ────────────────────────────────────────────


def test_convection_heats_particle_when_gas_hot():
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,    # no drying cooling
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g = np.full((Nz, Ny, Nx), 800.0)
    Y_O2 = np.full((Nz, Ny, Nx), 0.0)  # disable pyrolysis exo/endo
    T_s_before = float(buf["T_s"][0])
    # Run several short steps so T_s rises without Arrhenius firing too much
    # at the very start (k_pyro at 300 K is negligible)
    for _ in range(50):
        _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001)
    T_s_after = float(buf["T_s"][0])
    assert T_s_after > T_s_before


def test_no_convection_when_T_g_equals_T_s_and_dry():
    """T_g = T_s, no water, no pyrolysis → T_s unchanged."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g = np.full((Nz, Ny, Nx), 300.0)   # match T_s
    Y_O2 = np.full((Nz, Ny, Nx), 0.0)    # no R_op
    T_s_before = buf["T_s"].copy()
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01)
    # At T=300K, k_pyro and k_dry are ~0; T_s shouldn't move appreciably
    assert np.allclose(buf["T_s"], T_s_before, atol=1e-6)


# ── Burnout retirement ───────────────────────────────────────────────


def test_burnout_retires_when_mass_below_threshold():
    """Particle drained to <M_BURNOUT total mass → alive=0, n_burned bumped."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 2
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=900.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # Manually drain particles to near-zero mass
    for p in range(n):
        buf["m_solid"][p] = 1e-10
        buf["m_water"][p] = 0.0
        buf["m_char"][p]  = 0.0
    T_g = np.full((Nz, Ny, Nx), 900.0)
    Y_O2 = np.full((Nz, Ny, Nx), 0.23)
    Sp, Sd, Qp, Qd, YFs, _, _, _, n_alive, n_burned = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001)
    assert n_burned >= n  # all should retire (1e-10 < M_PARTICLE_BURNOUT=1e-8)
    assert int(buf["alive"].sum()) == 0


# ── Bit-exact determinism (Rule #17) ──────────────────────────────────


def test_step_bit_exact_under_repeat():
    Nz, Ny, Nx = 5, 2, 4
    n_z_bed = 3
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 6
    N_max = int((alpha > 0).sum()) * n_per_cell

    def fresh():
        buf = lb.allocate_bed_particle_buffers(N_max)
        lb.initialize_bed_particles_from_alpha_s(
            buf, alpha, rho_b_dry=1.07, moisture_frac=0.05,
            T_amb=400.0, dx=dx, dy=dy, dz_arr=dz_arr,
            n_z_bed=n_z_bed, n_per_cell=n_per_cell)
        T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=500.0)
        return buf, T_g, Y_O2

    buf_a, T_g_a, Y_O2_a = fresh()
    buf_b, T_g_b, Y_O2_b = fresh()
    for _ in range(20):
        _run_step(buf_a, T_g_a, Y_O2_a, (dx, dy, dz_arr, z_face), dt=0.001)
        _run_step(buf_b, T_g_b, Y_O2_b, (dx, dy, dz_arr, z_face), dt=0.001)

    for key in ("x", "y", "z", "m_solid", "m_water", "m_char", "T_s", "alive"):
        assert np.array_equal(buf_a[key], buf_b[key]), f"{key} not bit-exact"


# ── Total-mass conservation across drying + pyrolysis ─────────────────


def test_total_mass_conservation_in_domain():
    """In-domain: Δ(particle solid + water + char) = -Δ(gas-emitted) exactly.

    Σ_p [Δm_solid_p + Δm_water_p + Δm_char_p] = -Σ_cell [S_pyro+S_drying]·V_cell·dt
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.05,
        T_amb=550.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=700.0)
    dt = 0.005

    m_part_before = (buf["m_solid"].sum() + buf["m_water"].sum()
                     + buf["m_char"].sum())
    # Isolate drying + pyrolysis: char_ox and smolder consume mass without
    # emitting tracked Y_F species (CO2 not in our gas balance), breaking
    # the strict Σ(particle) = Σ(gas) invariant in the full physics.
    Sp, Sd, Qp, Qd, YFs, _, _, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=dt,
        do_char_ox=False, do_smolder=False)
    m_part_after = (buf["m_solid"].sum() + buf["m_water"].sum()
                    + buf["m_char"].sum())
    V_cell = dx * dy * dz_arr[0]
    m_gas_emitted = float((Sp.sum() + Sd.sum()) * V_cell * dt)
    # Particle loss = gas gain (m_char stays on particle; S_pyro and
    # S_drying are the only gas-emission paths)
    assert (m_part_before - m_part_after) == pytest.approx(m_gas_emitted, rel=1e-10)


# ── Char oxidation (Phase 3a) ─────────────────────────────────────────


def test_char_ox_consumes_m_char_when_hot_and_O2():
    """T_s > T_CHAR_ONSET=600 K AND Y_O2 > min → m_char drops, Q_char > 0."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=900.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # Seed m_char (would normally come from pyrolysis)
    for p in range(n):
        buf["m_char"][p] = buf["m_solid"][p] * 0.5
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=900.0)
    mc_before = buf["m_char"].sum()
    Sp, Sd, Qp, Qd, YFs, Qch, Qsm, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        do_drying=False, do_pyrolysis=False, do_smolder=False)
    mc_after = buf["m_char"].sum()
    assert mc_after < mc_before
    # Q_char should equal (dm_char × HOC_CHAR) / dt aggregated to cells
    from model_outdoor.physics_3d.pyrolysis_3d import HOC_CHAR
    V_cell = dx * dy * dz_arr[0]
    dm_char_total = mc_before - mc_after
    Q_char_total = float(Qch.sum()) * V_cell
    assert Q_char_total == pytest.approx(dm_char_total * HOC_CHAR / 0.01, rel=1e-10)


def test_char_ox_blocked_below_onset_temp():
    """T_s < T_CHAR_ONSET=600 K → no char_ox even with O2 + m_char."""
    from model_outdoor.physics_3d.pyrolysis_3d import T_CHAR_ONSET
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=T_CHAR_ONSET - 50.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    for p in range(n):
        buf["m_char"][p] = buf["m_solid"][p] * 0.5
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=T_CHAR_ONSET - 50.0)
    mc_before = buf["m_char"].sum()
    _, _, _, _, _, Qch, _, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        do_drying=False, do_pyrolysis=False, do_smolder=False)
    mc_after = buf["m_char"].sum()
    assert mc_after == pytest.approx(mc_before, rel=1e-12)
    assert float(Qch.sum()) == 0.0


def test_char_ox_blocked_when_no_O2():
    """Y_O2 ≈ 0 → no char_ox even at high T_s with m_char available."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=900.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    for p in range(n):
        buf["m_char"][p] = buf["m_solid"][p] * 0.5
    T_g = np.full((Nz, Ny, Nx), 900.0)
    Y_O2 = np.zeros((Nz, Ny, Nx))   # no O2
    mc_before = buf["m_char"].sum()
    _, _, _, _, _, Qch, _, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        do_drying=False, do_pyrolysis=False, do_smolder=False)
    assert buf["m_char"].sum() == pytest.approx(mc_before, rel=1e-12)
    assert float(Qch.sum()) == 0.0


# ── Smoldering oxidation (Phase 3a) ───────────────────────────────────


def test_smolder_fires_in_low_T_regime():
    """T_SMOLD_ONSET=473 K ≤ T_s < T_CHAR_ONSET=600 K → smolder fires, char_ox doesn't."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=550.0, dx=dx, dy=dy, dz_arr=dz_arr,   # between SMOLD and CHAR onsets
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=550.0)
    ms_before = buf["m_solid"].sum()
    _, _, _, _, _, Qch, Qsm, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        do_drying=False, do_pyrolysis=False)
    ms_after = buf["m_solid"].sum()
    # Smolder consumed some m_solid
    assert ms_after < ms_before
    # Q_smold > 0; Q_char = 0 (below char onset)
    assert float(Qsm.sum()) > 0.0
    assert float(Qch.sum()) == 0.0


def test_smolder_blocked_below_onset():
    """T_s < T_SMOLD_ONSET=473 K → no smolder."""
    from model_outdoor.physics_3d.pyrolysis_3d import T_SMOLD_ONSET
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=T_SMOLD_ONSET - 50.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=T_SMOLD_ONSET - 50.0)
    ms_before = buf["m_solid"].sum()
    _, _, _, _, _, _, Qsm, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        do_drying=False, do_pyrolysis=False)
    assert buf["m_solid"].sum() == pytest.approx(ms_before, rel=1e-12)
    assert float(Qsm.sum()) == 0.0


# ── Process-toggle independence ───────────────────────────────────────


def test_all_processes_off_no_change():
    """All physics toggles off + T_s = T_g = T_amb → particles unchanged.

    Note: radiation loss ε·σ·A_p·(T_s⁴-T_amb⁴) is ALWAYS on (not toggle-
    gated) because it's a passive cooling mechanism, not a physics
    process.  To get a true no-change baseline, set T_s = T_amb so the
    rad-loss term is zero too.
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    # T_s = T_amb = 300 K so Q_rad_loss = 0.
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.05,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=300.0)
    ms_before = buf["m_solid"].copy()
    mw_before = buf["m_water"].copy()
    mc_before = buf["m_char"].copy()
    Ts_before = buf["T_s"].copy()
    Sp, Sd, Qp, Qd, YFs, Qch, Qsm, _, _, _ = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
        T_amb=300.0,
        do_drying=False, do_pyrolysis=False,
        do_char_ox=False, do_smolder=False)
    assert np.array_equal(buf["m_solid"], ms_before)
    assert np.array_equal(buf["m_water"], mw_before)
    assert np.array_equal(buf["m_char"], mc_before)
    # T_s unchanged: T_g=T_s (no convection); T_s=T_amb (no rad loss).
    assert np.allclose(buf["T_s"], Ts_before)
    for arr in (Sp, Sd, Qp, Qd, YFs, Qch, Qsm):
        assert float(arr.sum()) == 0.0


def test_rad_loss_cools_particle_above_T_amb():
    """T_s > T_amb with all reactions off → T_s drops via Stefan-Boltzmann loss."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=1500.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # T_g = T_s = 1500K so convection is zero; rad-loss is the only term.
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=1500.0)
    Ts_before = buf["T_s"][0]
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
              T_amb=300.0,
              do_drying=False, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False)
    Ts_after = buf["T_s"][0]
    assert Ts_after < Ts_before, "rad loss should cool particle"


# ── External Q to solid (Phase 3c.7 — drip torch / bootstrap) ─────────


def test_Q_solid_ext_heats_particles():
    """Q_solid_ext > 0 should raise T_s independently of gas convection.

    Verifies the energy split:
       dT_s = Q_ext_per_cell · V_cell · dt / (n_per_cell · m_p · cp_solid)
    """
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # Constant Q_solid_ext = 50_000 W/m³ everywhere in the bed
    Q_ext = np.zeros((Nz, Ny, Nx))
    Q_ext[:n_z_bed, :, :] = 50_000.0
    # T_g = T_s = 300 → no convection
    T_g = np.full((Nz, Ny, Nx), 300.0)
    Y_O2 = np.full((Nz, Ny, Nx), 0.0)  # no R_op
    T_s_before = float(buf["T_s"][0])
    # eps_solid=0 disables the linearized-implicit Stefan-Boltzmann
    # correction, so the test sees a pure explicit-Euler dT_s.
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
              eps_solid=0.0,
              Q_solid_ext=Q_ext, n_per_cell_for_split=n_per_cell,
              do_drying=False, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False)
    T_s_after = float(buf["T_s"][0])
    assert T_s_after > T_s_before
    # Expected dT_s = Q_ext · V_cell · dt / (n_per_cell · m_p · cp)
    V_cell = dx * dy * dz_arr[0]
    m_p = float(buf["m_solid"][0])   # m_water=0, m_char=0
    expected_dT = 50_000.0 * V_cell * 0.01 / (n_per_cell * m_p * 1500.0)
    assert (T_s_after - T_s_before) == pytest.approx(expected_dT, rel=1e-10)


def test_Q_solid_ext_zero_leaves_T_s_unchanged():
    """Q_solid_ext = 0 → no T_s effect from this term."""
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=300.0)
    T_s_before = buf["T_s"].copy()
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01,
              Q_solid_ext=None,   # → zero array
              do_drying=False, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False)
    assert np.allclose(buf["T_s"], T_s_before)


# ── Dead particles skipped ────────────────────────────────────────────


def test_dead_particles_skipped_no_crash():
    Nz, Ny, Nx = 4, 2, 4
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 2
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,
        T_amb=400.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # Kill half manually with bogus positions
    for p in range(0, n, 2):
        buf["alive"][p] = lp.ALIVE_FALSE
        buf["x"][p] = -999.0
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=400.0)
    Sp, Sd, Qp, Qd, YFs, _, _, _, n_alive, n_burned = _run_step(
        buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.01)
    # Only the live half should count
    assert n_alive == n // 2


# ── Drying-mode equilibrium (FIRETEC heat-rate-limited) ──────────────────


def test_drying_mode_equilibrium_pins_T_s_at_boil_with_water():
    """A particle at T_s = 400 K with m_water > 0 in equilibrium mode
    should be pinned back to T_BOIL = 373.15 K after one step, with
    the excess thermal energy converted to evaporation."""
    Nz, Ny, Nx = 4, 2, 8
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 1
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=1.0,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n > 0
    # Force T_s slightly above boil with abundant water.  At M=1.0 a
    # small over-pin excess (Δ=10K) is far less than the latent heat
    # of available water, so the override must pin T_s and leave water.
    buf["T_s"][:n] = T_OVER = lb.T_BOIL_WATER + 10.0
    m_water_before = buf["m_water"][:n].copy()
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=T_OVER)
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001,
              do_drying=True, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False,
              drying_mode=1)
    # T_s pinned at T_BOIL after step (water still remains).
    assert np.allclose(buf["T_s"][:n], lb.T_BOIL_WATER, atol=0.01), \
        f"T_s {buf['T_s'][:n]} not pinned at {lb.T_BOIL_WATER}"
    # Water decreased but not exhausted.
    assert np.all(buf["m_water"][:n] < m_water_before)
    assert np.all(buf["m_water"][:n] > 0.0)


def test_drying_mode_equilibrium_no_water_no_pin():
    """With m_water = 0 (M_f=0), equilibrium mode must NOT pin T_s —
    particle should heat normally."""
    Nz, Ny, Nx = 4, 2, 8
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 1
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.0,  # NO water
        T_amb=400.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n > 0
    buf["T_s"][:n] = 500.0
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=500.0)
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001,
              do_drying=True, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False,
              drying_mode=1)
    # Dry particle stays at its temperature (no pin override).
    assert np.all(buf["T_s"][:n] > lb.T_BOIL_WATER + 50)


def test_drying_mode_equilibrium_burns_water_faster_at_higher_M():
    """At equilibrium drying, the same Q_in dries fixed mass per second
    regardless of M_f.  Higher-M particles take proportionally longer
    to finish drying — the property Cheney's exp(-0.0707·M) reflects."""
    Nz, Ny, Nx = 4, 2, 8
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 1
    N_max = int((alpha > 0).sum()) * n_per_cell

    def _drying_time_to_99(m_f):
        buf = lb.allocate_bed_particle_buffers(N_max)
        n = lb.initialize_bed_particles_from_alpha_s(
            buf, alpha, rho_b_dry=1.07, moisture_frac=m_f,
            T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
            n_z_bed=n_z_bed, n_per_cell=n_per_cell)
        # Pin T_s just above boil so excess heat per step is small and
        # exhaustion takes many steps.  10K excess + dt=1ms → ~few dozen
        # evaporation steps for typical M_f, enough to measure the ratio.
        T_PINNED = lb.T_BOIL_WATER + 10.0
        buf["T_s"][:n] = T_PINNED
        T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=T_PINNED)
        m0 = buf["m_water"][:n].sum()
        for step in range(2000):
            buf["T_s"][:n] = T_PINNED  # re-set each step (excess heat steady)
            _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=0.001,
                      do_drying=True, do_pyrolysis=False,
                      do_char_ox=False, do_smolder=False,
                      drying_mode=1)
            if buf["m_water"][:n].sum() < 0.01 * m0:
                return step + 1
        return 2000

    n_steps_04 = _drying_time_to_99(0.04)
    n_steps_08 = _drying_time_to_99(0.08)
    # With heat-rate-limited drying and fixed heat input, time scales
    # linearly with initial water mass (= moisture content).
    # M=8% should take ~2× the steps of M=4%.
    ratio = n_steps_08 / max(n_steps_04, 1)
    assert 1.5 < ratio < 2.5, (
        f"M=0.08 took {n_steps_08} steps vs M=0.04 {n_steps_04} steps "
        f"(ratio {ratio:.2f}); expected ~2× for equilibrium drying")


def test_drying_mode_combined_arrhenius_below_boil_equilibrium_above():
    """Combined mode: grass Arrhenius fires below T_BOIL, equilibrium
    fires above.  Verify a particle at T_s=350 K with water present
    loses water via the slow Arrhenius (grass values are ~100× faster
    than Lautenberger at preheat T, but still measurable per step)."""
    Nz, Ny, Nx = 4, 2, 8
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 1
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.08,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n > 0
    # Particle at preheat T (below boil, but warm enough for grass
    # Arrhenius to fire at ~0.15 /s).  Single step at dt=1s removes
    # ~14% of water — measurable.
    buf["T_s"][:n] = 350.0
    m_water_before = buf["m_water"][:n].copy()
    T_g, Y_O2 = _make_uniform_gas(Nz, Ny, Nx, T_g=350.0)
    _run_step(buf, T_g, Y_O2, (dx, dy, dz_arr, z_face), dt=1.0,
              do_drying=True, do_pyrolysis=False,
              do_char_ox=False, do_smolder=False,
              drying_mode=2)  # combined
    # Water decreased but not gone (gradual Arrhenius decay).
    dropped = (m_water_before - buf["m_water"][:n]) / m_water_before
    assert np.all(dropped > 0.05), \
        f"combined-mode Arrhenius below T_BOIL did not remove water (dropped {dropped})"
    assert np.all(buf["m_water"][:n] > 0.0)


# ── Phase 17a: per-cell M_local aggregator for DOM κ_solid scaling ──

def test_M_local_aggregator_empty_bed_is_zero():
    """Cells with no live particles produce M_local = 0."""
    Nz, Ny, Nx = 4, 2, 6
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    buf = lb.allocate_bed_particle_buffers(8)
    # No alive particles
    M_local = np.full((Nz, Ny, Nx), 99.0, dtype=np.float64)
    lb.aggregate_particles_to_M_local_grid(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"],
        dx, dy, z_face, M_local,
    )
    assert np.array_equal(M_local, np.zeros((Nz, Ny, Nx)))


def test_M_local_aggregator_uniform_moisture_recovers_input():
    """A bed initialized at M=0.16 → aggregator returns 0.16 in all
    bed cells (and 0.0 in non-bed cells)."""
    Nz, Ny, Nx = 4, 2, 6
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=1, i_hi=5)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    M_INIT = 0.16
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=M_INIT,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n > 0
    M_local = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    lb.aggregate_particles_to_M_local_grid(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"],
        dx, dy, z_face, M_local,
    )
    # Bed cells: M_local ≈ 0.16 (exact since each particle has m_w = M·m_s).
    bed_mask = alpha > 0
    assert np.allclose(M_local[bed_mask], M_INIT, atol=1e-12), \
        f"bed cells should have M_local={M_INIT}, got {M_local[bed_mask].min()}-{M_local[bed_mask].max()}"
    # Non-bed cells: M_local = 0.
    non_bed = ~bed_mask
    assert np.array_equal(M_local[non_bed], np.zeros_like(M_local[non_bed]))


def test_M_local_aggregator_drying_reduces_M():
    """Particles dried partially → M_local decreases proportionally."""
    Nz, Ny, Nx = 4, 2, 6
    n_z_bed = 2
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=1, i_hi=5)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 4
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.30,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    # Halve water on first half of particles
    buf["m_water"][:n // 2] *= 0.5
    M_local = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    lb.aggregate_particles_to_M_local_grid(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"],
        dx, dy, z_face, M_local,
    )
    bed_mask = alpha > 0
    # Mean of bed-cell M_local must be < 0.30 (some dried)
    assert M_local[bed_mask].mean() < 0.30
    # No cell should exceed input M (water can only decrease)
    assert M_local[bed_mask].max() <= 0.30 + 1e-12


def test_M_local_aggregator_bit_exact_determinism():
    """Rule #18: aggregator output must match bit-exactly across
    two calls on the same input."""
    Nz, Ny, Nx = 6, 4, 10
    n_z_bed = 3
    alpha = _make_alpha_s_uniform_bed(Nz, Ny, Nx, n_z_bed, i_lo=2, i_hi=8)
    dx, dy, dz_arr, z_face = _make_grid(Nz, Ny, Nx)
    n_per_cell = 8
    N_max = int((alpha > 0).sum()) * n_per_cell
    buf = lb.allocate_bed_particle_buffers(N_max)
    n = lb.initialize_bed_particles_from_alpha_s(
        buf, alpha, rho_b_dry=1.07, moisture_frac=0.18,
        T_amb=300.0, dx=dx, dy=dy, dz_arr=dz_arr,
        n_z_bed=n_z_bed, n_per_cell=n_per_cell)
    assert n > 0
    # Add some noise to particle positions / masses for variety
    rng = np.random.default_rng(0)
    buf["m_water"][:n] *= (1.0 + 0.1 * rng.standard_normal(n))
    buf["m_solid"][:n] *= (1.0 + 0.05 * rng.standard_normal(n))
    M1 = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    M2 = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    lb.aggregate_particles_to_M_local_grid(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"],
        dx, dy, z_face, M1,
    )
    lb.aggregate_particles_to_M_local_grid(
        buf["x"], buf["y"], buf["z"], buf["alive"],
        buf["m_solid"], buf["m_water"],
        dx, dy, z_face, M2,
    )
    assert np.array_equal(M1, M2), \
        "Rule #18 violation: aggregator gave different output on repeat call"
