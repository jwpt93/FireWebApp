"""Ranz-Marshall correlation validation for the gas-solid coupling kernel.

The convective heat-transfer coefficient h_p in step_gas_solid_coupling
uses the Ranz-Marshall (1952) particle Nusselt correlation:

    Nu = 2 + 0.6 · Re^0.5 · Pr^(1/3)

with Re = ρ·u·d_p/μ, d_p = 4/σ_SAV, and h_p = Nu·k_gas/d_p.

These tests verify the kernel's heat-flux output matches the analytic
Ranz-Marshall prediction across a range of Re, α_s, and ΔT.  Gas/solid
properties (μ, k, Pr, ρ_s, cp_s) are baked into the kernel as module
constants from Drysdale 2011 Tab 2.4 + Janssens 1993.

References:
  - Ranz, W.E. & Marshall, W.R. (1952) Chem. Eng. Prog. 48:141 — original
    correlation for evaporation from droplets, widely used in particle-
    laden flows for forced convection.
  - Drysdale (2011) Fire Dynamics 3rd ed. §2.4 — gas property values and
    range of applicability.
  - Whitaker, S. (1972) AIChE J. 18:361 — improved correlation valid up to
    Re ~ 7e4 (Ranz-Marshall is technically valid for Re ≤ ~200; FDS/WFDS
    use R-M everywhere for fire-relevant cells without correction).

Acceptance band: ±1% absolute on (q_conv, ΔT) compared to hand-computed
analytic values — this is a deterministic correlation; only floating-
point error should be present.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from model_outdoor.physics_3d import coupling_3d
from model_outdoor.physics_3d.coupling_3d import (
    _MU_GAS, _K_GAS, _PR_GAS,
    _RHO_SOLID, _CP_SOLID, _EPS_SOLID, _SIGMA_SB,
)


def _ranz_marshall_h(rho_g: float, u: float, sigma_sav: float) -> tuple[float, float, float]:
    """Hand-compute (d_p, h_p, a_v_factor) from Ranz-Marshall + project constants.
    Returns (d_p, h_p, sigma_sav).  a_v = sigma_sav * α_s in caller."""
    d_p = 4.0 / sigma_sav
    Re = max(rho_g * u * d_p / _MU_GAS, 0.1)
    Nu = 2.0 + 0.6 * (Re ** 0.5) * (_PR_GAS ** (1.0 / 3.0))
    h_p = Nu * _K_GAS / d_p
    return d_p, h_p, sigma_sav


def _run_single_cell_coupling(
    T_g_init: float,
    T_s_init: float,
    rho_g: float,
    u: float,
    alpha_s: float,
    sigma_sav: float,
    dt: float,
    T_amb: float = 300.0,
    q_loss_enable: bool = False,    # disable for clean R-M test
    q_rad_in: float = 0.0,
):
    """Drive a single (1,1,1) cell through one coupling step and return
    (T_g_new, T_s_new, q_conv_analytic)."""
    shape = (1, 1, 1)
    T_g = np.full(shape, T_g_init)
    T_s = np.full(shape, T_s_init)
    rho = np.full(shape, rho_g)
    u_arr = np.full(shape, u);  v_arr = np.zeros(shape);  w_arr = np.zeros(shape)
    alpha = np.full(shape, alpha_s)
    q_rad = np.full(shape, q_rad_in)
    Q_pyro = np.zeros(shape)
    Q_comb = np.zeros(shape)
    m_water = np.zeros(shape)
    dz_arr = np.array([0.05])
    coupling_3d.step_gas_solid_coupling(
        T_g, T_s, rho, u_arr, v_arr, w_arr, alpha,
        sigma_sav, q_rad, Q_pyro, Q_comb, m_water,
        L_v=2.26e6, dt=dt, dz_arr=dz_arr,
        T_amb=T_amb, q_loss_enable=q_loss_enable,
    )
    return float(T_g[0, 0, 0]), float(T_s[0, 0, 0])


def _expected_q_conv(rho_g, u, sigma_sav, alpha_s, T_g, T_s):
    """Analytic q_conv [W/m³] from Ranz-Marshall."""
    _, h_p, _ = _ranz_marshall_h(rho_g, u, sigma_sav)
    a_v = sigma_sav * alpha_s
    return h_p * a_v * (T_g - T_s)


# ─── Test 1: Ranz-Marshall at multiple Re values ──────────────────────────────

@pytest.mark.parametrize("u", [0.0, 0.5, 2.0, 5.0])
def test_ranz_marshall_q_conv_matches_analytic(u: float):
    """At several wind speeds (Re = 0.1 floor, low, moderate, high), the
    kernel's gas-to-solid heat transfer should match analytic
    q_conv = h·a_v·(T_g - T_s) to <1% (pure correlation algebra, no chemistry)."""
    rho_g = 1.2
    sigma_sav = 2000.0    # Nat 4% grass
    alpha_s = 7.5e-4
    T_g_init = 1200.0
    T_s_init = 300.0
    dt = 1.0e-3
    cp_g = 1100.0
    T_g_new, T_s_new = _run_single_cell_coupling(
        T_g_init, T_s_init, rho_g, u, alpha_s, sigma_sav, dt,
    )
    # Expected ΔT_g from R-M: -q_conv·dt/(ρ_g·cp_g)
    q_conv = _expected_q_conv(rho_g, u, sigma_sav, alpha_s, T_g_init, T_s_init)
    dT_g_expected = -q_conv * dt / (rho_g * cp_g)
    dT_g_actual   = T_g_new - T_g_init
    rel_err = abs(dT_g_actual - dT_g_expected) / max(abs(dT_g_expected), 1e-12)
    assert rel_err < 0.01, (
        f"u={u}: kernel ΔT_g={dT_g_actual:.6f} K, analytic R-M={dT_g_expected:.6f} K, "
        f"rel_err={rel_err:.2e}"
    )


# ─── Test 2: ΔT_s scales linearly with α_s (a_v ∝ α_s) ────────────────────────

def test_dT_s_scales_with_alpha_s():
    """With other inputs fixed and small dt, ΔT_s ∝ α_s up to C_s = ρ_s·cp_s·α_s
    in the denominator.  Actually ΔT_s / α_s = constant in the limit
    (q_conv ∝ α_s, C_s ∝ α_s → cancels).  So ΔT_s itself should be ~constant
    across α_s for this regime."""
    rho_g = 1.2
    u = 2.0
    sigma_sav = 2000.0
    T_g_init = 1500.0
    T_s_init = 300.0
    dt = 1.0e-3
    dT_s_values = []
    for alpha_s in [1.0e-4, 5.0e-4, 2.5e-3, 1.0e-2]:
        _, T_s_new = _run_single_cell_coupling(
            T_g_init, T_s_init, rho_g, u, alpha_s, sigma_sav, dt,
        )
        dT_s_values.append(T_s_new - T_s_init)
    # Verify ΔT_s is approximately constant across α_s (cancellation property).
    rel_spread = (max(dT_s_values) - min(dT_s_values)) / max(np.mean(dT_s_values), 1e-12)
    assert rel_spread < 0.01, (
        f"ΔT_s should be ~constant in α_s (q_conv and C_s both ∝ α_s), "
        f"got values {dT_s_values}"
    )


# ─── Test 3: sign and direction of heat transfer ─────────────────────────────

def test_hot_gas_warms_cold_solid():
    """T_g > T_s ⇒ T_g decreases, T_s increases."""
    T_g_new, T_s_new = _run_single_cell_coupling(
        T_g_init=1500.0, T_s_init=300.0, rho_g=1.2, u=2.0,
        alpha_s=2.5e-3, sigma_sav=2000.0, dt=1.0e-3,
    )
    assert T_g_new < 1500.0, f"T_g should decrease (got {T_g_new})"
    assert T_s_new > 300.0, f"T_s should increase (got {T_s_new})"


def test_hot_solid_warms_cold_gas():
    """T_s > T_g ⇒ T_s decreases, T_g increases.  This is the pathway
    by which a smoldering bed pre-heats incoming cold gas (relevant for
    high-wind Cheney cases where solid may stay hot longer than gas)."""
    T_g_new, T_s_new = _run_single_cell_coupling(
        T_g_init=300.0, T_s_init=900.0, rho_g=1.2, u=2.0,
        alpha_s=2.5e-3, sigma_sav=2000.0, dt=1.0e-3,
    )
    assert T_g_new > 300.0, f"T_g should increase from hot solid (got {T_g_new})"
    assert T_s_new < 900.0, f"T_s should decrease (got {T_s_new})"


# ─── Test 4: zero-α_s short-circuits coupling ─────────────────────────────────

def test_zero_alpha_s_decouples_phases():
    """No solid present (α_s = 0): no coupling, T_g and T_s unchanged."""
    T_g_new, T_s_new = _run_single_cell_coupling(
        T_g_init=1500.0, T_s_init=300.0, rho_g=1.2, u=2.0,
        alpha_s=0.0, sigma_sav=2000.0, dt=1.0e-3,
    )
    assert T_g_new == 1500.0, f"T_g changed with α_s=0: {T_g_new}"
    assert T_s_new == 300.0, f"T_s changed with α_s=0: {T_s_new}"


# ─── Test 5: bit-exact determinism (Rule #18) ─────────────────────────────────

def test_coupling_bit_exact_determinism():
    """Same inputs ⇒ same outputs to the last digit (kernel-level Rule #17)."""
    T_g_1, T_s_1 = _run_single_cell_coupling(
        T_g_init=1500.0, T_s_init=400.0, rho_g=1.2, u=2.0,
        alpha_s=2.5e-3, sigma_sav=2000.0, dt=1.0e-3,
    )
    T_g_2, T_s_2 = _run_single_cell_coupling(
        T_g_init=1500.0, T_s_init=400.0, rho_g=1.2, u=2.0,
        alpha_s=2.5e-3, sigma_sav=2000.0, dt=1.0e-3,
    )
    assert T_g_1 == T_g_2, f"T_g drift: {T_g_1} vs {T_g_2}"
    assert T_s_1 == T_s_2, f"T_s drift: {T_s_1} vs {T_s_2}"


# ─── Test 6: Re=0 returns natural-convection floor ────────────────────────────

def test_natural_convection_floor_at_zero_wind():
    """At u=0, kernel floors Re=0.1 → Nu = 2 + 0.6·(0.1)^0.5·Pr^(1/3) ≈ 2.168.
    Verify hand-computed h_p matches kernel output (zero-wind limit)."""
    rho_g = 1.2
    sigma_sav = 2000.0
    alpha_s = 2.5e-3
    T_g_init = 1500.0
    T_s_init = 300.0
    dt = 1.0e-3
    cp_g = 1100.0
    T_g_new, _ = _run_single_cell_coupling(
        T_g_init, T_s_init, rho_g, u=0.0,
        alpha_s=alpha_s, sigma_sav=sigma_sav, dt=dt,
    )
    # Analytic at Re=0.1 (the floor)
    d_p = 4.0 / sigma_sav
    Nu_floor = 2.0 + 0.6 * (0.1 ** 0.5) * (_PR_GAS ** (1.0 / 3.0))
    h_p_floor = Nu_floor * _K_GAS / d_p
    a_v = sigma_sav * alpha_s
    q_conv_floor = h_p_floor * a_v * (T_g_init - T_s_init)
    dT_g_analytic = -q_conv_floor * dt / (rho_g * cp_g)
    rel_err = abs((T_g_new - T_g_init) - dT_g_analytic) / max(abs(dT_g_analytic), 1e-12)
    assert rel_err < 0.01, (
        f"Floor-Re mismatch: kernel ΔT_g={T_g_new - T_g_init:.6f}, "
        f"analytic={dT_g_analytic:.6f}, rel_err={rel_err:.2e}"
    )
