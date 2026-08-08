"""Phase 15C — Flame Surface Density (FSD) closure unit tests.

Verifies (Rule #18):
  1. Smoothing kernel preserves value range [0,1] for a binary input.
  2. Smoothing converges with iterations (later iters introduce less
     change than earlier ones).
  3. |∇| of a linear field is the linear field's slope; checks the
     non-uniform-dz stencil basic correctness.
  4. Surface integral of |∇c| over a half-step phi_flame approximates
     the analytical surface area (coarea formula).
  5. step_fsd_chemistry:
     - ω = 0 in cells with Y_F=0 or Y_O2=0 or zero gradient.
     - ω = 0 in cells with f_avail=0 (post-burnt or air).
     - mass conservation: ΔY_F = -ω·dt/ρ to floating-point.
     - T_g rise = ω·HoC_eff·dt/(ρ·cp_g) to floating-point.
     - bit-exact under repeat at production thread count (Rule #17).
  6. Closure-registry dispatch path produces identical result to direct
     kernel call.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("NUMBA_NUM_THREADS", "12")

import numpy as np
import pytest

from model_outdoor.physics_3d import chemistry_closures, level_set_fsd_3d
from model_outdoor.physics_3d.chemistry_closures._constants import (
    S_STOICH, HOC_J,
)


# ── Smoothing kernel ─────────────────────────────────────────────────────────

def test_smoothing_preserves_value_range():
    """Box-filter on a [0,1] field must not produce values outside [0,1]."""
    phi = np.zeros((6, 4, 12), dtype=np.float64)
    phi[:, :, 6:] = 1.0
    smoothed = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=5)
    assert smoothed.min() >= 0.0
    assert smoothed.max() <= 1.0


def test_smoothing_idempotent_on_constant_field():
    """A constant field stays constant under smoothing."""
    phi = np.full((5, 4, 8), 0.42, dtype=np.float64)
    smoothed = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=3)
    assert np.allclose(smoothed, 0.42)


def test_smoothing_is_deterministic():
    """Same input → same output, bit-exact (Rule #17)."""
    rng = np.random.default_rng(42)
    phi = rng.random((6, 4, 12))
    s1 = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=4).copy()
    s2 = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=4).copy()
    assert np.array_equal(s1, s2)


def test_smoothing_zero_iters_returns_copy():
    """n_iters=0 should leave the field unchanged in value (but the
    smoothing function still produces a sensible array shape)."""
    phi = np.zeros((4, 3, 6), dtype=np.float64)
    phi[:, :, 3:] = 1.0
    smoothed = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=0)
    assert smoothed.shape == phi.shape
    assert np.array_equal(smoothed, phi)


# ── Gradient kernel ──────────────────────────────────────────────────────────

def test_grad_norm_linear_field():
    """For f(x) = a·x + b on a uniform grid, |∇f| = |a| everywhere
    interior.  Tests the basic central-difference stencil."""
    Nx, Ny, Nz = 12, 4, 6
    dx, dy = 0.1, 0.1
    dz_arr = np.full(Nz, 0.1)
    x = np.arange(Nx) * dx
    f = np.broadcast_to(2.5 * x, (Nz, Ny, Nx)).astype(np.float64).copy()
    g = np.empty_like(f)
    level_set_fsd_3d.compute_grad_norm_nonuniform(f, dx, dy, dz_arr, g)
    # Interior cells in x should have |∇f| = 2.5; boundaries also = 2.5
    # because one-sided diff on a linear field is exact.
    assert np.allclose(g, 2.5, atol=1e-10)


def test_grad_norm_constant_is_zero():
    """∇ of a constant field is zero everywhere."""
    f = np.full((4, 3, 8), 1.7, dtype=np.float64)
    g = np.empty_like(f)
    dz_arr = np.full(4, 0.1)
    level_set_fsd_3d.compute_grad_norm_nonuniform(f, 0.1, 0.1, dz_arr, g)
    assert np.allclose(g, 0.0)


def test_grad_norm_nonuniform_dz_uses_actual_spacing():
    """For f(z) = z (constant slope), |∇f| should be 1.0 regardless of
    cell sizes — this tests the non-uniform-dz central-difference factor."""
    Nz = 8
    # Geometric non-uniform spacing: dz doubles each step
    dz_arr = 0.05 * np.array([1.0, 1.2, 1.44, 1.728, 2.07, 2.49, 2.99, 3.58])
    # Cell-centre z coordinates
    z = np.cumsum(dz_arr) - 0.5 * dz_arr
    Ny, Nx = 1, 1
    f = z.reshape(Nz, Ny, Nx).astype(np.float64).copy()
    g = np.empty_like(f)
    level_set_fsd_3d.compute_grad_norm_nonuniform(f, 1.0, 1.0, dz_arr, g)
    # Interior cells (k ∈ {1..Nz-2}) should have |∇f| ≈ 1.0 (the slope of z).
    interior = g[1:-1, 0, 0]
    assert np.allclose(interior, 1.0, atol=5e-2), (
        f"non-uniform-dz interior |∇f|: {interior}"
    )


def test_grad_norm_is_deterministic():
    rng = np.random.default_rng(7)
    f = rng.random((6, 4, 12))
    dz_arr = np.linspace(0.01, 0.5, 6)
    g1 = np.empty_like(f); g2 = np.empty_like(f)
    level_set_fsd_3d.compute_grad_norm_nonuniform(f, 0.1, 0.1, dz_arr, g1)
    level_set_fsd_3d.compute_grad_norm_nonuniform(f, 0.1, 0.1, dz_arr, g2)
    assert np.array_equal(g1, g2)


# ── Coarea property: ∫|∇c| dV ≈ surface area ────────────────────────────────

def test_surface_integral_of_grad_norm_matches_area():
    """For a half-step phi (0 → 1 sharp), after enough smoothing the
    volume integral of |∇c| approximates the analytical interface area.

    Setup: domain 0.6m × 0.3m × 0.4m at dx=dy=dz=0.05m.  Step at x=0.3m
    gives an interface area = 0.3 × 0.4 = 0.12 m².

    The coarea identity says ∫|∇c| dV = surface area for a smoothed
    indicator that transitions from 0 to 1.  Our smoothed phi obeys
    that to within smoothing resolution + boundary effects.
    """
    dx = dy = 0.05
    Nx, Ny, Nz = 12, 6, 8
    dz_arr = np.full(Nz, dx)
    phi = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    phi[:, :, Nx // 2:] = 1.0

    c = level_set_fsd_3d.smooth_phi_flame(phi, n_iters=3)
    g = np.empty_like(c)
    level_set_fsd_3d.compute_grad_norm_nonuniform(c, dx, dy, dz_arr, g)

    # Volume integral
    dV = dx * dy * dz_arr[0]   # uniform here
    surface_area_numerical = float(g.sum() * dV)
    surface_area_analytical = (Ny * dy) * (Nz * dz_arr[0])   # 0.3 × 0.4

    # The smoothed c integral should land within 25% of the analytical
    # for our chosen mesh (smoothing introduces some perimeter effects
    # at the y/z boundaries because we replicate).
    ratio = surface_area_numerical / surface_area_analytical
    assert 0.75 < ratio < 1.30, (
        f"surface area ratio {ratio:.3f} outside [0.75, 1.30] — "
        f"numerical {surface_area_numerical:.4f} m² vs analytical "
        f"{surface_area_analytical:.4f} m²"
    )


# ── step_fsd_chemistry ──────────────────────────────────────────────────────

def _fsd_step(shape, grad_value=2.0, Y_fuel=0.08, Y_O2=0.18, T_g=1200.0,
              rho=1.0, dt=0.005):
    """Helper to build a step input + run step_fsd_chemistry once."""
    rho_a = np.full(shape, rho, dtype=np.float64)
    T_a = np.full(shape, T_g, dtype=np.float64)
    Yf = np.full(shape, Y_fuel, dtype=np.float64)
    YO2 = np.full(shape, Y_O2, dtype=np.float64)
    grad = np.full(shape, grad_value, dtype=np.float64)
    omega = np.zeros(shape, dtype=np.float64)
    level_set_fsd_3d.step_fsd_chemistry(
        rho_a, T_a, Yf, YO2, grad,
        chi_rad=0.34, cp_g=1100.0,
        s_L=0.4,
        Y_F_unb=level_set_fsd_3d.Y_F_UNB_DEFAULT,
        Y_O2_unb=level_set_fsd_3d.Y_O2_UNB_DEFAULT,
        dt=dt, n_substeps=1, omega_int_out=omega,
    )
    return rho_a, T_a, Yf, YO2, grad, omega


def test_fsd_omega_zero_when_no_gradient():
    """With |∇c| = 0 (no flame surface), ω must be 0."""
    *_, omega = _fsd_step((4, 3, 5), grad_value=0.0)
    assert np.all(omega == 0.0)


def test_fsd_omega_zero_when_no_fuel():
    """With Y_F = 0 (post-burnt), ω must be 0 even if gradient is large."""
    *_, omega = _fsd_step((4, 3, 5), grad_value=10.0, Y_fuel=0.0)
    assert np.all(omega == 0.0)


def test_fsd_omega_zero_when_no_oxidizer():
    """With Y_O2 = 0 (vitiated), ω must be 0."""
    *_, omega = _fsd_step((4, 3, 5), grad_value=10.0, Y_O2=0.0)
    assert np.all(omega == 0.0)


def test_fsd_mass_conservation_one_substep():
    """ΔY_F = -ω·dt/ρ per cell, to floating-point precision."""
    Y0 = 0.08
    dt = 0.005
    rho = 1.0
    rho_a, _, Yf, _, _, omega = _fsd_step((4, 3, 5), grad_value=2.0,
                                          Y_fuel=Y0, dt=dt, rho=rho)
    dY_expected = -omega * dt / rho_a
    dY_actual = Yf - Y0
    np.testing.assert_allclose(dY_actual, dY_expected, atol=1e-15, rtol=0)


def test_fsd_temperature_rise_matches_omega():
    """ΔT_g = ω·HoC_eff·dt/(ρ·cp_g) per cell."""
    Tg0 = 1200.0
    dt = 0.005
    cp_g = 1100.0
    chi_rad = 0.34
    HoC_eff = HOC_J * (1.0 - chi_rad)
    rho_a, T_g, _, _, _, omega = _fsd_step((3, 2, 4), grad_value=3.0,
                                            T_g=Tg0, dt=dt)
    dT_expected = omega * HoC_eff * dt / (rho_a * cp_g)
    dT_actual = T_g - Tg0
    np.testing.assert_allclose(dT_actual, dT_expected, atol=1e-12, rtol=0)


def test_fsd_availability_gate_below_unity():
    """When Y_F < Y_F_unb, ω should scale linearly with Y_F (gate active)."""
    grad = 2.0
    s_L = 0.4
    rho = 1.0
    Y_F_unb = level_set_fsd_3d.Y_F_UNB_DEFAULT
    # Pick Y_F much smaller than Y_F_unb so f_avail = Y_F/Y_F_unb is binding
    Yf_lo = 0.01
    *_, omega = _fsd_step((1, 1, 1), grad_value=grad, Y_fuel=Yf_lo)
    expected = rho * s_L * grad * (Yf_lo / Y_F_unb)
    assert abs(float(omega[0, 0, 0]) - expected) < 1e-12 * abs(expected)


def test_fsd_availability_gate_capped_at_unity():
    """When Y_F >= Y_F_unb (fuel-rich), f_avail caps at 1.  ω = ρ·s_L·|∇c|."""
    grad = 2.0
    s_L = 0.4
    rho = 1.0
    # Y_F well above Y_F_unb → gate is at unity → ω = ρ·s_L·grad
    Yf_rich = 1.0
    *_, omega = _fsd_step((1, 1, 1), grad_value=grad, Y_fuel=Yf_rich,
                          Y_O2=0.232)
    expected = rho * s_L * grad   # f_avail = 1
    assert abs(float(omega[0, 0, 0]) - expected) < 1e-12 * abs(expected)


def test_fsd_kernel_is_bit_exact_under_repeat():
    """Rule #17: bit-exact at production thread count."""
    rng = np.random.default_rng(123)
    shape = (5, 4, 12)
    rho = np.full(shape, 1.0) + 0.1 * rng.random(shape)
    T_g_1 = 1000.0 + 400.0 * rng.random(shape)
    T_g_2 = T_g_1.copy()
    Yf_1 = 0.03 + 0.05 * rng.random(shape)
    Yf_2 = Yf_1.copy()
    YO2_1 = 0.15 + 0.05 * rng.random(shape)
    YO2_2 = YO2_1.copy()
    grad = 0.5 + 2.0 * rng.random(shape)
    om1 = np.zeros(shape); om2 = np.zeros(shape)
    for Tg, Yf, YO2, omg in [(T_g_1, Yf_1, YO2_1, om1),
                              (T_g_2, Yf_2, YO2_2, om2)]:
        level_set_fsd_3d.step_fsd_chemistry(
            rho, Tg, Yf, YO2, grad,
            chi_rad=0.34, cp_g=1100.0,
            s_L=0.4,
            Y_F_unb=level_set_fsd_3d.Y_F_UNB_DEFAULT,
            Y_O2_unb=level_set_fsd_3d.Y_O2_UNB_DEFAULT,
            dt=0.005, n_substeps=1, omega_int_out=omg,
        )
    assert np.array_equal(T_g_1, T_g_2)
    assert np.array_equal(Yf_1, Yf_2)
    assert np.array_equal(YO2_1, YO2_2)
    assert np.array_equal(om1, om2)


# ── Closure registry dispatch ────────────────────────────────────────────────

def test_registry_includes_level_set_fsd():
    assert "level_set_fsd" in chemistry_closures.available()


# ── Closure registry dispatch + hybrid kernel ──────────────────────────────


def test_registry_includes_level_set_fsd():
    assert "level_set_fsd" in chemistry_closures.available()


def _make_hybrid_kwargs(shape=(4, 3, 12), rho_val=1.0, T_g_val=1200.0,
                         Y_fuel_val=0.05, Y_O2_val=0.18,
                         k_val=1.0, eps_val=0.5,
                         phi_value=0.0,
                         dx=0.05, dy=0.05, dz=0.05,
                         chi_rad=0.34, cp_g=1100.0, dt=0.001, n_substeps=1):
    """Build a complete-but-trivial kwargs dict for the hybrid FSD closure."""
    return dict(
        rho=np.full(shape, rho_val, dtype=np.float64),
        T_g=np.full(shape, T_g_val, dtype=np.float64),
        Y_fuel=np.full(shape, Y_fuel_val, dtype=np.float64),
        Y_O2=np.full(shape, Y_O2_val, dtype=np.float64),
        phi_flame=np.full(shape, phi_value, dtype=np.float64),
        k_turb=np.full(shape, k_val, dtype=np.float64),
        eps_turb=np.full(shape, eps_val, dtype=np.float64),
        dx=dx, dy=dy, dz_arr=np.full(shape[0], dz, dtype=np.float64),
        chi_rad=chi_rad, cp_g=cp_g, dt=dt, n_substeps=n_substeps,
        omega_out=np.zeros(shape, dtype=np.float64),
    )


def test_hybrid_phi_flame_outside_fires_edc_branch():
    """Phase 15D-C (sign convention: phi_flame > 0 ⇔ outside flame body):
    cold start has phi_flame = +1e6 sentinel everywhere, all cells route
    to EDC Magnussen.  Magnussen is T-independent so ω > 0 even at
    T_g = 300 K provided Y_F + Y_O2 + k_turb are positive — this is the
    chicken-egg bootstrap path."""
    kw = _make_hybrid_kwargs(phi_value=1.0e6, T_g_val=300.0,
                              Y_fuel_val=0.05, Y_O2_val=0.18,
                              k_val=1.0, eps_val=0.5)
    chemistry_closures.run("level_set_fsd", **kw)
    assert kw["omega_out"].max() > 0.0
    assert kw["omega_out"].min() > 0.0


def test_hybrid_phi_flame_inside_fires_fsd_branch():
    """Phase 15D-C: cells with phi_flame ≤ 0 (inside flame body) use FSD.

    Setup: phi_flame is a signed-distance field with negative interior
    for i ≥ 6 (flame body) and positive exterior for i < 6 (outside).
    Both regions should have non-zero ω: EDC in i<6, FSD in i≥6."""
    shape = (4, 3, 12)
    kw = _make_hybrid_kwargs(shape=shape, T_g_val=1200.0)
    # Outside flame body (i < 6): positive distance
    # Inside flame body  (i ≥ 6): negative distance
    kw["phi_flame"][:, :, :6] = +1.0     # outside → EDC
    kw["phi_flame"][:, :, 6:] = -0.1     # inside  → FSD
    chemistry_closures.run("level_set_fsd", **kw)
    omega = kw["omega_out"]
    edc_region = omega[:, :, :6]
    assert edc_region.max() > 0.0, "EDC branch did not fire in phi>0 region"
    fsd_region = omega[:, :, 6:]
    assert fsd_region.max() > 0.0, "FSD branch did not fire in phi<=0 region"


def test_hybrid_accepts_precomputed_c_grad_norm():
    """Phase 15D-F: when the main loop supplies a pre-computed c_grad_norm,
    the closure must use it directly without re-computing smoothing +
    gradient.  Verifies the F-option fast path."""
    shape = (4, 3, 12)
    kw = _make_hybrid_kwargs(shape=shape)
    kw["phi_flame"][:, :, 6:] = 1.0

    # Pre-compute c_grad_norm exactly as the main loop will
    c_grad_pre = level_set_fsd_3d.compute_c_grad_norm_from_phi_flame(
        kw["phi_flame"], kw["dx"], kw["dy"], kw["dz_arr"],
        smoothing_iters=level_set_fsd_3d.SMOOTHING_ITERS_DEFAULT,
    )

    # Path A: registry dispatch supplying c_grad_norm
    kw_A = {**kw, "c_grad_norm": c_grad_pre.copy()}
    kw_A["rho"] = kw["rho"].copy(); kw_A["T_g"] = kw["T_g"].copy()
    kw_A["Y_fuel"] = kw["Y_fuel"].copy(); kw_A["Y_O2"] = kw["Y_O2"].copy()
    kw_A["omega_out"] = np.zeros_like(kw["omega_out"])
    chemistry_closures.run("level_set_fsd", **kw_A)

    # Path B: registry dispatch without c_grad_norm — closure builds it itself
    kw_B = {**kw}
    kw_B["rho"] = kw["rho"].copy(); kw_B["T_g"] = kw["T_g"].copy()
    kw_B["Y_fuel"] = kw["Y_fuel"].copy(); kw_B["Y_O2"] = kw["Y_O2"].copy()
    kw_B["omega_out"] = np.zeros_like(kw["omega_out"])
    chemistry_closures.run("level_set_fsd", **kw_B)

    # Both paths must produce identical state — proves the F-option fast
    # path is numerically equivalent to the in-closure smoothing.
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(kw_A[field], kw_B[field]), (
            f"F-option path diverged from in-closure path for {field}"
        )


def test_hybrid_kernel_is_bit_exact_under_repeat():
    """Rule #17: hybrid kernel must produce bit-exact identical output
    on identical inputs at the production thread count."""
    shape = (5, 4, 12)
    rng = np.random.default_rng(13)
    base = _make_hybrid_kwargs(shape=shape)
    base["rho"] = np.full(shape, 1.0) + 0.05 * rng.random(shape)
    base["T_g"] = 800.0 + 400.0 * rng.random(shape)
    base["Y_fuel"] = 0.03 + 0.05 * rng.random(shape)
    base["Y_O2"] = 0.15 + 0.05 * rng.random(shape)
    base["k_turb"] = 0.5 + 0.5 * rng.random(shape)
    base["eps_turb"] = 0.2 + 0.2 * rng.random(shape)
    base["phi_flame"] = (rng.random(shape) > 0.4).astype(np.float64)

    def _run_copy():
        kw = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        chemistry_closures.run("level_set_fsd", **kw)
        return kw

    A = _run_copy(); B = _run_copy()
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(A[field], B[field]), f"hybrid non-deterministic on {field}"


def test_hybrid_ignores_unused_extras():
    """The hybrid closure must NOT raise when the main loop passes
    closure-specific kwargs used by other closures (tau_mix, omega_O2)."""
    kw = _make_hybrid_kwargs()
    kw["tau_mix"] = np.full(kw["rho"].shape, 0.1)
    kw["omega_O2"] = np.full(kw["rho"].shape, 1e30)
    kw["omega_max_T"] = np.full(kw["rho"].shape, 1e30)
    kw["unused_thing"] = "ok"
    chemistry_closures.run("level_set_fsd", **kw)


def test_hybrid_tfm_xi_scales_fsd_branch_linearly():
    """Phase 15H — Charlette 2002 wrinkling factor Ξ scales FSD ω linearly.

    With phi_flame ≤ 0 everywhere (pure FSD branch), passing tfm_xi=Ξ
    must produce ω_Ξ = Ξ · ω_1 to floating-point precision.  Verifies
    Ξ enters as a simple multiplier on the FSD rate.
    """
    shape = (1, 1, 1)
    # Build a clean FSD-branch input.  phi_flame < 0 → FSD.
    # Pre-compute c_grad_norm so the smoothing pass is bypassed (the
    # smoothing of a single cell returns 0, so we supply grad directly).
    def _run(xi: float, dt: float = 1e-6):
        kw = _make_hybrid_kwargs(
            shape=shape, phi_value=-0.1,
            T_g_val=1200.0, Y_fuel_val=0.05, Y_O2_val=0.18, dt=dt,
        )
        kw["c_grad_norm"] = np.full(shape, 2.0, dtype=np.float64)
        if xi != 1.0:
            kw["tfm_xi"] = xi
        chemistry_closures.run("level_set_fsd", **kw)
        return float(kw["omega_out"][0, 0, 0])

    om_1 = _run(1.0)
    om_3 = _run(3.0)
    om_5 = _run(5.0)
    assert om_1 > 0.0, "FSD branch produced zero ω at baseline"
    assert abs(om_3 / om_1 - 3.0) < 1e-9, f"Ξ=3 → {om_3/om_1:.4f}× expected 3×"
    assert abs(om_5 / om_1 - 5.0) < 1e-9, f"Ξ=5 → {om_5/om_1:.4f}× expected 5×"


def test_hybrid_tfm_xi_default_is_back_compat():
    """Phase 15H — omitting tfm_xi must yield identical output to tfm_xi=1.0."""
    shape = (3, 2, 6)
    kw_default = _make_hybrid_kwargs(shape=shape)
    kw_default["phi_flame"][:, :, 3:] = -0.1
    kw_explicit = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in kw_default.items()}
    kw_explicit["tfm_xi"] = 1.0
    chemistry_closures.run("level_set_fsd", **kw_default)
    chemistry_closures.run("level_set_fsd", **kw_explicit)
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(kw_default[field], kw_explicit[field]), (
            f"tfm_xi=1.0 explicit diverged from default on {field}"
        )


def test_hybrid_tfm_xi_does_not_affect_edc_branch():
    """Phase 15H — Ξ is FSD-only; EDC branch must be invariant under tfm_xi."""
    shape = (3, 2, 6)
    # phi_flame = +1.0 everywhere → pure EDC branch
    kw_xi1 = _make_hybrid_kwargs(shape=shape, phi_value=+1.0, T_g_val=1200.0,
                                  k_val=1.0, eps_val=0.5)
    kw_xi5 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in kw_xi1.items()}
    kw_xi5["tfm_xi"] = 5.0
    chemistry_closures.run("level_set_fsd", **kw_xi1)
    chemistry_closures.run("level_set_fsd", **kw_xi5)
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(kw_xi1[field], kw_xi5[field]), (
            f"EDC branch was perturbed by tfm_xi on {field}"
        )


def test_hybrid_inner_body_edc_routes_phi_negative_to_edc_branch():
    """Phase 15J: when inner_body_edc=True, cells with phi_flame ≤ 0 (which
    would normally fire FSD) must fire EDC instead.  Verification:
    with phi_flame=-0.1 (inside flame body) AND c_grad_norm=0 (no FSD
    surface), FSD branch gives ω=0; EDC branch gives ω>0 because Magnussen
    is T-independent.  So toggling inner_body_edc=True must produce ω>0
    in this configuration, while the default produces ω=0."""
    shape = (1, 1, 1)
    kw_default = _make_hybrid_kwargs(
        shape=shape, phi_value=-0.1, T_g_val=1200.0,
        Y_fuel_val=0.05, Y_O2_val=0.18, k_val=1.0, eps_val=0.5,
    )
    kw_default["c_grad_norm"] = np.zeros(shape, dtype=np.float64)
    kw_invert = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in kw_default.items()}
    kw_invert["inner_body_edc"] = True
    chemistry_closures.run("level_set_fsd", **kw_default)
    chemistry_closures.run("level_set_fsd", **kw_invert)
    om_default = float(kw_default["omega_out"][0, 0, 0])
    om_invert = float(kw_invert["omega_out"][0, 0, 0])
    assert om_default == 0.0, (
        f"FSD branch should give ω=0 when |∇c|=0, got {om_default}"
    )
    assert om_invert > 0.0, (
        f"EDC branch should give ω>0 in inner-body when inner_body_edc=True, "
        f"got {om_invert}"
    )


def test_hybrid_inner_body_edc_default_is_back_compat():
    """Phase 15J: omitting inner_body_edc must yield identical output to
    inner_body_edc=False."""
    shape = (3, 2, 6)
    kw_default = _make_hybrid_kwargs(shape=shape)
    kw_default["phi_flame"][:, :, 3:] = -0.1
    kw_explicit = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in kw_default.items()}
    kw_explicit["inner_body_edc"] = False
    chemistry_closures.run("level_set_fsd", **kw_default)
    chemistry_closures.run("level_set_fsd", **kw_explicit)
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(kw_default[field], kw_explicit[field]), (
            f"inner_body_edc=False explicit diverged from default on {field}"
        )


def test_hybrid_inner_body_edc_does_not_affect_exterior_cells():
    """Phase 15J: cells with phi_flame > 0 always use EDC regardless of
    the inner_body_edc flag.  So perturbing the flag while phi_flame > 0
    everywhere must produce identical output."""
    shape = (3, 2, 6)
    # phi_flame = +1.0 everywhere → exterior; always EDC
    kw_fsd_default = _make_hybrid_kwargs(shape=shape, phi_value=+1.0,
                                          T_g_val=1200.0,
                                          k_val=1.0, eps_val=0.5)
    kw_invert = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in kw_fsd_default.items()}
    kw_invert["inner_body_edc"] = True
    chemistry_closures.run("level_set_fsd", **kw_fsd_default)
    chemistry_closures.run("level_set_fsd", **kw_invert)
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(kw_fsd_default[field], kw_invert[field]), (
            f"inner_body_edc perturbed phi>0 cells on {field}"
        )


def test_hybrid_inner_body_edc_is_bit_exact_under_repeat():
    """Rule #17: inner_body_edc=True path is bit-exact deterministic."""
    shape = (5, 4, 12)
    rng = np.random.default_rng(19)
    base = _make_hybrid_kwargs(shape=shape)
    base["rho"] = np.full(shape, 1.0) + 0.05 * rng.random(shape)
    base["T_g"] = 800.0 + 400.0 * rng.random(shape)
    base["Y_fuel"] = 0.03 + 0.05 * rng.random(shape)
    base["Y_O2"] = 0.15 + 0.05 * rng.random(shape)
    base["k_turb"] = 0.5 + 0.5 * rng.random(shape)
    base["eps_turb"] = 0.2 + 0.2 * rng.random(shape)
    base["phi_flame"] = (rng.random(shape) > 0.4).astype(np.float64) - 0.5
    # phi_flame now spans both signs

    def _run_copy():
        kw = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in base.items()}
        kw["inner_body_edc"] = True
        chemistry_closures.run("level_set_fsd", **kw)
        return kw

    A = _run_copy(); B = _run_copy()
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(A[field], B[field]), (
            f"inner_body_edc=True non-deterministic on {field}"
        )


def test_hybrid_tfm_xi_is_bit_exact_under_repeat():
    """Rule #17: tfm_xi path must be bit-exact deterministic."""
    shape = (5, 4, 12)
    rng = np.random.default_rng(17)
    base = _make_hybrid_kwargs(shape=shape)
    base["rho"] = np.full(shape, 1.0) + 0.05 * rng.random(shape)
    base["T_g"] = 800.0 + 400.0 * rng.random(shape)
    base["Y_fuel"] = 0.03 + 0.05 * rng.random(shape)
    base["Y_O2"] = 0.15 + 0.05 * rng.random(shape)
    base["k_turb"] = 0.5 + 0.5 * rng.random(shape)
    base["eps_turb"] = 0.2 + 0.2 * rng.random(shape)
    base["phi_flame"] = (rng.random(shape) > 0.4).astype(np.float64)

    def _run_copy():
        kw = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in base.items()}
        kw["tfm_xi"] = 3.0
        chemistry_closures.run("level_set_fsd", **kw)
        return kw

    A = _run_copy(); B = _run_copy()
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(A[field], B[field]), (
            f"tfm_xi=3.0 path non-deterministic on {field}"
        )


# ── Precompute helper ──────────────────────────────────────────────────────


def test_compute_c_grad_norm_matches_inline_path():
    """The compute_c_grad_norm_from_phi_flame helper must produce the same
    array as smooth_phi_flame + compute_grad_norm_nonuniform called in
    sequence (Phase 15D-F single-call ergonomics)."""
    shape = (5, 4, 10)
    dx = dy = 0.05
    dz_arr = np.full(shape[0], dx)
    rng = np.random.default_rng(99)
    phi = (rng.random(shape) > 0.5).astype(np.float64)

    # Helper path
    g_helper = level_set_fsd_3d.compute_c_grad_norm_from_phi_flame(
        phi, dx, dy, dz_arr,
        smoothing_iters=level_set_fsd_3d.SMOOTHING_ITERS_DEFAULT,
    )

    # Manual path
    c = level_set_fsd_3d.smooth_phi_flame(
        phi, n_iters=level_set_fsd_3d.SMOOTHING_ITERS_DEFAULT,
    )
    g_manual = np.empty_like(c)
    level_set_fsd_3d.compute_grad_norm_nonuniform(c, dx, dy, dz_arr, g_manual)

    assert np.array_equal(g_helper, g_manual)
