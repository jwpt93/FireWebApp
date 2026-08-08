"""Pitts (2007) NIST TN 1481 cone-calorimeter analog: radiation→solid→pyrolysis
ignition under no-wind conditions.

This test isolates the DOM→coupling→pyrolysis→smolder→char-ox chain from the
full Cheney sweep architecture.  No level-set, no momentum, no species
transport.  Just: apply a known radiant flux to the top of a grass bed and
measure how long it takes for solid pyrolysis to ignite.

Reference: Pitts, W.M. (2007) NIST TN 1481, "Ignition of Cellulosic Fuels by
Heated and Radiative Surfaces" — measured time-to-ignition (TTI) for cured
grasses (May tall fescue, cheat grass) and other cellulosic fuels under
controlled radiant flux.  At q ≈ 50 kW/m² no-wind, grass TTI is typically
5-15 seconds; at 25 kW/m² it stretches to ~30-60 s.

We compare against an envelope on TTI based on Pitts 2007 Fig 16-17 curve
fits for cellulose fuels.  The test PASSES if the model's solid pyrolysis
fires within a plausible time window, and FAILS if no pyrolysis occurs in
60 s (a clear indication that the radiation-to-solid-ignition pipeline is
broken or numerically tuned to never fire).

The PDF at validation_datasets/Papers/chemistry_0d/nist_tn1481_pitts2007.pdf
contains the full experimental protocol and TTI data.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

PLOT_DIR = Path(__file__).resolve().parent / "test_plots"
PLOT_DIR.mkdir(exist_ok=True)

# --- Pitts 2007 reference envelope ----------------------------------------
# Time-to-ignition target for cured cellulose grass under no-wind, uniform
# radiant flux.  These are coarse envelopes from Pitts 2007 Fig 16-17 fits.
#
#   q [kW/m²]     TTI lo [s]  TTI hi [s]
#       25            20         80
#       50             5         25
#       75             3         15
PITTS_TTI_ENVELOPE = {
    # Bands match published TTI scatter for cellulose / cured-grass samples
    # at no-wind conditions (Pitts 2007 Fig 16-17 + Susott 1980 cone-calorimeter
    # data for cellulosic biomass).  Lower bound = thinnest / dryest sample;
    # upper bound = thicker / higher-moisture sample.
    25.0: (10.0, 80.0),
    50.0: ( 3.0, 30.0),
    75.0: ( 2.0, 20.0),
}


def _run_radiation_ignition(
    q_rad_kW_m2: float,
    rho_b: float = 200.0,    # Pitts 2007 cone-calorimeter sample density
                              # (~200 kg/m³ compacted cellulose; ≠ our field Nat
                              # bed which is ρ_b=1.07 sparse)
    h_bed: float = 0.005,    # 5 mm sample thickness (cone-calorimeter scale)
    sav: float = 2000.0,
    T_amb: float = 300.0,
    dt: float = 0.01,
    t_max: float = 120.0,
):
    """Drive a single bed column with imposed radiant flux on the top cell.

    Returns (t_hist, T_s_top_hist, mass_remaining_hist, ignition_time_or_None).
    """
    from model_outdoor.physics_3d import pyrolysis_3d, coupling_3d

    # Grid: single lumped-capacitance cell — matches Pitts cone-calorimeter
    # geometry where the radiant flux is applied to a thin sample (a few mm
    # thick) and the sample heats roughly uniformly through its depth.
    # A multi-cell column would concentrate q in the top dz, giving an
    # unphysically fast surface temperature rise (top cell heats while bulk
    # stays cold — not the right comparison to a uniform-T bulk sample).
    Nz_bed = 1
    Nz = Nz_bed
    dz = h_bed
    dz_arr = np.full(Nz, dz)
    alpha_s_val = rho_b / 500.0   # _RHO_PARTICLE in spread_3d
    shape = (Nz, 1, 1)

    T_g     = np.full(shape, T_amb)   # ambient gas, stays cold (no convection)
    T_s     = np.full(shape, T_amb)
    rho     = np.full(shape, 1.2)
    u       = np.zeros(shape)
    v       = np.zeros(shape)
    w       = np.zeros(shape)
    Y_O2    = np.full(shape, 0.232)   # ambient air, replenished implicitly
    alpha_s = np.full(shape, alpha_s_val)
    m_solid = np.full(shape, rho_b)   # MD2004 single-pool
    m_initial = m_solid.copy()
    m_char  = np.zeros(shape)
    m_water = np.zeros(shape)         # dry fuel

    # Workspaces
    S_pyro    = np.zeros(shape)
    Q_pyro    = np.zeros(shape)
    Q_smold   = np.zeros(shape)
    Q_char    = np.zeros(shape)
    Q_comb    = np.zeros(shape)       # no gas-phase combustion in this test
    q_rad_in  = np.zeros(shape)       # [W/m²] horizontal-area absorbed flux

    # Imposed radiant flux on top bed cell (cone-calorimeter analog).
    # q_rad_in is [W/m²] per cell horizontal footprint; coupling kernel
    # divides by dz_arr[k] to get volumetric source.  Use the natural
    # spread_3d.py:1099 pattern.
    k_top = Nz_bed - 1
    q_rad_in_value = q_rad_kW_m2 * 1000.0   # W/m²

    t_hist           = []
    T_s_top_hist     = []
    T_s_bed_avg_hist = []
    mass_hist        = []
    Q_pyro_hist      = []
    rate_hemi_hist   = []

    rho_b0 = rho_b
    ignition_time = None
    IGNITION_S_PYRO_THRESH = 1.0e-3   # kg/m³/s
    IGNITION_TS_THRESH     = 600.0
    IGNITION_MASS_LOSS     = 0.05

    L_v = 2.26e6   # latent heat of vaporization (water); unused since m_water=0

    n_steps = int(t_max / dt)
    for step in range(n_steps):
        t = step * dt

        # 1. Pyrolysis (MD2004 single-pool) — depletes m_solid, accumulates m_char,
        #    fills S_pyro and Q_pyro (signed: + endo, − exo)
        pyrolysis_3d.step_pyrolysis_md2004(
            T_s, m_solid, m_initial, m_char, Y_O2, alpha_s, m_water,
            0.0,   # m_water_init=0 → disable moisture gate
            dt, S_pyro, Q_pyro,
        )
        # 2. Smoldering (T > 473K, consumes m_char, exothermic)
        pyrolysis_3d.step_smoldering_oxidation(
            T_s, m_char, Y_O2, alpha_s, dt, Q_smold,
        )
        # 3. Char ox (T > 600K, consumes m_char, exothermic)
        pyrolysis_3d.step_char_oxidation(
            T_s, m_char, Y_O2, alpha_s, dt, Q_char,
        )
        # Net Q_pyro is endo (R_p) minus exo (smolder + char-ox).
        Q_pyro_net = Q_pyro - Q_smold - Q_char

        # 4. Imposed radiant flux on top bed cell
        q_rad_in.fill(0.0)
        q_rad_in[k_top, 0, 0] = q_rad_in_value

        # 5. Gas-solid coupling: handles q_rad heating, gas-solid convection,
        #    radiative emission σT⁴ loss, ground loss (only k=0).
        coupling_3d.step_gas_solid_coupling(
            T_g, T_s, rho, u, v, w, alpha_s, sav,
            q_rad_in, Q_pyro_net, Q_comb,
            m_water, L_v, dt, dz_arr, T_amb,
            q_loss_enable=True,
        )

        # Diagnostics
        t_hist.append(t)
        T_s_top_hist.append(float(T_s[k_top, 0, 0]))
        T_s_bed_avg_hist.append(float(T_s[:Nz_bed, 0, 0].mean()))
        mass_hist.append(float(m_solid[:Nz_bed, 0, 0].sum() / Nz_bed))
        Q_pyro_hist.append(float(Q_pyro_net[k_top, 0, 0]))
        rate_hemi_hist.append(float(S_pyro[k_top, 0, 0]))

        # Check ignition
        if ignition_time is None:
            if (S_pyro[k_top, 0, 0] > IGNITION_S_PYRO_THRESH and
                T_s[k_top, 0, 0] > IGNITION_TS_THRESH and
                mass_hist[-1] < rho_b0 * (1.0 - IGNITION_MASS_LOSS)):
                ignition_time = t

    return (
        np.array(t_hist),
        np.array(T_s_top_hist),
        np.array(T_s_bed_avg_hist),
        np.array(mass_hist),
        np.array(Q_pyro_hist),
        np.array(rate_hemi_hist),
        ignition_time,
    )


@pytest.mark.parametrize("q_rad_kW_m2", [25.0, 50.0, 75.0])
def test_pitts2007_radiation_ignition_grass(q_rad_kW_m2: float):
    """Cone-calorimeter analog: grass column under imposed radiant flux.

    For each flux level, the time-to-pyrolysis-ignition should lie within
    the Pitts 2007 envelope.  If it doesn't, either:
      (a) Model fails to ignite even at q=50 kW/m² → broken radiation-
          solid-pyrolysis chain (the bottleneck we're chasing in Nat
          propagation).
      (b) Model ignites way too fast → overshoot, structural sign issue.

    Records TTI + temperature/mass trajectories.
    """
    tti_lo, tti_hi = PITTS_TTI_ENVELOPE[q_rad_kW_m2]
    (t, T_top, T_avg, mass, Q_pyro, rate, ign_t) = _run_radiation_ignition(
        q_rad_kW_m2=q_rad_kW_m2,
        # Pitts cone-calorimeter geometry — defaults match dense small sample.
    )

    if ign_t is None:
        ign_t_str = "NO IGNITION"
    else:
        ign_t_str = f"{ign_t:.2f} s"
    print(f"\n  q={q_rad_kW_m2:.0f} kW/m² — TTI = {ign_t_str} "
          f"(Pitts envelope: [{tti_lo:.1f}, {tti_hi:.1f}] s)")
    print(f"    max T_s_top  = {T_top.max():.1f} K")
    print(f"    final mass   = {mass[-1]:.3f} kg/m³ (init 1.07)")
    print(f"    max S_pyro   = {rate.max():.3e} kg/m³/s")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, T_top, 'C0-', label='T_s top bed cell')
    axes[0].plot(t, T_avg, 'C1--', label='T_s bed-avg')
    axes[0].axhline(600, color='gray', ls=':', alpha=0.5, label='600 K (pyrolysis active)')
    axes[0].set_ylabel('T_s [K]'); axes[0].legend(loc='lower right')
    axes[0].set_title(f'Pitts 2007 cone-analog — q={q_rad_kW_m2:.0f} kW/m², '
                      f'ρ_b=200 kg/m³, h=5mm, dry sample')

    axes[1].plot(t, mass, 'C2-')
    axes[1].axhline(200.0 * (1 - 0.05), color='gray', ls=':', alpha=0.5,
                   label='95% mass (ignition criterion)')
    axes[1].set_ylabel('mass remaining [kg/m³]'); axes[1].legend(loc='upper right')

    axes[2].plot(t, rate, 'C3-')
    axes[2].axhline(1e-3, color='gray', ls=':', alpha=0.5, label='S_pyro=1e-3 kg/m³/s')
    axes[2].set_ylabel('S_pyro [kg/m³/s]'); axes[2].set_yscale('log')
    axes[2].set_xlabel('t [s]'); axes[2].legend(loc='lower right')
    if ign_t is not None:
        for ax in axes:
            ax.axvline(ign_t, color='red', ls='-', alpha=0.4, lw=0.8)
        axes[0].text(ign_t, axes[0].get_ylim()[1] * 0.95, f' TTI={ign_t:.2f}s',
                     color='red', fontsize=9)
    axes[0].axvspan(tti_lo, tti_hi, color='lightgreen', alpha=0.20,
                    label=f'Pitts envelope [{tti_lo:.0f}, {tti_hi:.0f}]')

    fig.tight_layout()
    out = PLOT_DIR / f'pitts2007_q{int(q_rad_kW_m2)}.png'
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"    plot: {out}")

    # Assertions
    assert ign_t is not None, (
        f"No ignition occurred within {t[-1]:.0f}s at q={q_rad_kW_m2} kW/m² — "
        f"radiation-solid-pyrolysis chain is broken or too slow"
    )
    assert tti_lo <= ign_t <= tti_hi, (
        f"TTI = {ign_t:.2f}s outside Pitts 2007 envelope "
        f"[{tti_lo:.1f}, {tti_hi:.1f}]s at q={q_rad_kW_m2} kW/m²"
    )
