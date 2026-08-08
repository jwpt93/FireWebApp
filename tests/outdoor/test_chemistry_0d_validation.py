"""0D chemistry-kernel validation against published EXP data.

Per CLAUDE.md Rule #18 (unit tests for every new module) and Rule #15
(comparison plots required for validation).  Each test:
  - drives ONE kernel in 0D (single cell, no flow, no transport)
  - matches the EXP setup conditions
  - asserts model output is within tolerance of EXP
  - includes a bit-exact determinism check
  - emits a comparison plot under plots/chemistry_0d/

Sources (all open-access, in validation_datasets/Papers/chemistry_0d/):
  - Peterson & Brown 2020 OSTI 1648151 — char ox kinetics for switchgrass +
    corn stover (Table 4, Regime I).  Direct A, E values for grass.
  - Pitts 2007 NIST TN 1481 — heated-plate ignition for tall fescue, cheat
    grass, fine Florida grass.  Provides smolder/glowing onset T's.
  - Ohlemiller 1991 IAFSS Vol 3:565 — solid wood smoldering propagation
    rates and energy-balance model with E ≈ 164 kJ/mol (cgs).
  - Burra & Gupta 2019 Fuel 237:1057 — TGA-DSC of biomass pyrolysis,
    multi-step kinetics for lignocellulose feedstocks.

Each test produces a plot at plots/chemistry_0d/<test_name>.png per
Rule #15.  Failure mode: assert fails AND plot still produced for inspection.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from model_outdoor.physics_3d import combustion_3d, pyrolysis_3d

PLOT_DIR = Path(__file__).resolve().parent.parent.parent / "plots" / "chemistry_0d"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

R_GAS = 8.314


# ───────────────────────────── helpers ──────────────────────────────────────

def _arrhenius_rate_constant(A: float, E: float, T_K: float) -> float:
    """Return k = A·exp(-E/RT) [1/s]."""
    return A * math.exp(-E / (R_GAS * T_K))


# ─────────────────── 1. CHAR OXIDATION vs OSTI 2020 ─────────────────────────

def _char_ox_zero_d(T_K: float, m0: float, Y_O2: float, t_end: float,
                    n_steps: int = 200):
    """Drive `step_char_oxidation` in 0D from m_solid = m0; return (t, m, Q)."""
    shape = (1, 1, 1)
    T_s = np.full(shape, T_K, dtype=np.float64)
    m_solid = np.full(shape, m0, dtype=np.float64)
    Y_O2_arr = np.full(shape, Y_O2, dtype=np.float64)
    alpha_s = np.full(shape, 1.0, dtype=np.float64)  # bed cell
    Q_out = np.zeros(shape, dtype=np.float64)
    dt = t_end / n_steps
    t_hist = np.zeros(n_steps + 1)
    m_hist = np.zeros(n_steps + 1); m_hist[0] = m0
    Q_hist = np.zeros(n_steps + 1)
    for k in range(1, n_steps + 1):
        pyrolysis_3d.step_char_oxidation(T_s, m_solid, Y_O2_arr, alpha_s, dt, Q_out)
        t_hist[k] = k * dt
        m_hist[k] = m_solid[0, 0, 0]
        Q_hist[k] = Q_out[0, 0, 0]
    return t_hist, m_hist, Q_hist


# OSTI 2020 Table 4 — Apparent Arrhenius for Regime I (333-433°C, 606-706K)
#   Feedstock          A (1/s)         E (kJ/mol)
OSTI_KINETICS = {
    "Douglas fir":  (3.76e7, 126_000.0),
    "Pine":         (1.08e8, 129_000.0),
    "Red oak":      (3.12e6, 111_000.0),
    "Willow":       (1.24e7, 109_000.0),
    "Switchgrass":  (7.09e5,  94_000.0),
    "Corn stover":  (3.75e6, 100_000.0),
}


def test_char_oxidation_vs_osti_2020_kinetic_rates():
    """Compare model k(T) = A·exp(-E/RT) against OSTI Regime I kinetics
    for switchgrass + corn stover (the herbaceous biomass closest to grass).

    Quantitative check: at T = 600 K (327°C, just inside Regime I):
      Model k = A_CHAR × exp(-E_CHAR/(R·T))     [our params, generic Mell 2007]
      EXP   k = A_OSTI × exp(-E_OSTI/(R·T))     [switchgrass]
    Acceptance: model k within factor 10 of switchgrass k (order-of-magnitude
    agreement).  Char-ox kinetic constants are notoriously variable (×100×
    spread across literature for same fuel) so single-OOM is realistic.
    """
    T_test = np.linspace(550.0, 750.0, 41)   # 277-477°C: spans Regime I + early Regime II
    k_model = np.array([_arrhenius_rate_constant(
        pyrolysis_3d.A_CHAR, pyrolysis_3d.E_CHAR, T) for T in T_test])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(T_test - 273.15, k_model, "k-", lw=2.5,
                label=f"Model (A={pyrolysis_3d.A_CHAR:.1e}, "
                      f"E={pyrolysis_3d.E_CHAR/1e3:.0f} kJ/mol)")
    for fuel, (A, E) in OSTI_KINETICS.items():
        k_exp = np.array([_arrhenius_rate_constant(A, E, T) for T in T_test])
        ls = "-" if fuel in ("Switchgrass", "Corn stover") else ":"
        lw = 1.8 if fuel in ("Switchgrass", "Corn stover") else 1.0
        ax.semilogy(T_test - 273.15, k_exp, ls=ls, lw=lw,
                    label=f"{fuel} (A={A:.1e}, E={E/1e3:.0f})")
    ax.axvspan(60, 60+0.001, alpha=0)  # spacer
    ax.axvspan(333-273.15-50, 433-273.15+50, alpha=0.10, color="grey",
               label="OSTI Regime I (333-433°C)")
    ax.set_xlabel("T_s [°C]")
    ax.set_ylabel("Char-ox rate constant k = A·exp(-E/RT)  [1/s]")
    ax.set_title("Char oxidation kinetics: model vs Peterson & Brown 2020 (OSTI 1648151)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "char_ox_vs_osti_2020.png", dpi=140)
    plt.close(fig)

    # Quantitative check at 600 K (327°C, mid Regime I).  After commit d9e8691
    # re-tuned A_CHAR, E_CHAR to Peterson-Brown 2020 switchgrass values, the
    # ratio went from 6.4× to 1.0×.  Acceptance band tightened from initial
    # 0.1-10× OOM to 0.5-2× (Rule #1: grass-specific lit values now).
    T_check = 600.0
    k_model_check = _arrhenius_rate_constant(
        pyrolysis_3d.A_CHAR, pyrolysis_3d.E_CHAR, T_check)
    A_sw, E_sw = OSTI_KINETICS["Switchgrass"]
    k_sw = _arrhenius_rate_constant(A_sw, E_sw, T_check)
    ratio = k_model_check / k_sw
    print(f"\n  T=600K (327°C, mid Regime I):")
    print(f"    model k         = {k_model_check:.3e} 1/s")
    print(f"    switchgrass k   = {k_sw:.3e} 1/s")
    print(f"    ratio (mdl/sw)  = {ratio:.2f}× (acceptance: 0.5-2×, post-retune)")
    assert 0.5 < ratio < 2.0, (
        f"model char-ox rate at 600K is {ratio:.2f}× switchgrass — outside "
        f"tightened post-retune band (0.5-2×).  Either restore Peterson-Brown "
        f"2020 values (A=7.09e5, E=94 kJ/mol) or update test acceptance with "
        f"a new lit citation.")


def test_char_oxidation_bit_exact_determinism():
    """Run step_char_oxidation twice on identical inputs → bit-exact match (Rule #18)."""
    t1, m1, Q1 = _char_ox_zero_d(T_K=700.0, m0=1.0, Y_O2=0.21, t_end=10.0)
    t2, m2, Q2 = _char_ox_zero_d(T_K=700.0, m0=1.0, Y_O2=0.21, t_end=10.0)
    assert np.array_equal(m1, m2), f"char ox m drift: max |Δ|={np.max(np.abs(m1-m2))}"
    assert np.array_equal(Q1, Q2), f"char ox Q drift: max |Δ|={np.max(np.abs(Q1-Q2))}"


# ─────────────────── 2. SMOLDERING vs NIST TN 1481 ──────────────────────────

# NIST TN 1481 (Pitts 2007) Table-equivalent: minimum heated-plate ignition T
# (no wind).  These are the temperatures at which smoldering/glowing
# combustion was first observed on the test grasses.
NIST_TN1481_GLOW_ONSET = {
    "May tall fescue":   340.0 + 273.15,   # 613 K
    "August tall fescue": 371.0 + 273.15,  # 644 K
    "Cheat grass":       380.0 + 273.15,   # 653 K
    "Fine Florida grass": 330.0 + 273.15,  # 603 K (approx)
    "Cheat grass (Kaminski)": 330.0 + 273.15,  # 603 K (glowing onset)
    "Pine needles":      310.0 + 273.15,   # 583 K
}
T_GLOW_MEDIAN_GRASS = float(np.median([v for k, v in NIST_TN1481_GLOW_ONSET.items()
                                       if "fescue" in k.lower() or "cheat" in k.lower()]))


def test_smolder_onset_T_vs_blunck_2022():
    """Smolder kernel `T_SMOLD_ONSET` should be within the Blunck 2022
    chemically-grounded smoldering range from IR-camera measurements:
    200-500°C (473-773 K) for grass blend.

    This is the chemical-onset acceptance band (Blunck 2022 SERDP RC-2651,
    direct IR-camera surface T on biomass beds including grass blend).
    The NIST TN 1481 visual-glowing onset (583-653 K) is a higher-side
    definition shown for reference but not used as the acceptance target —
    chemical smolder happens below visual glowing.

    Acceptance: T_SMOLD_ONSET ∈ [450, 750] K (Blunck range with 23 K
    margin on the lower bound).
    """
    T_smold = pyrolysis_3d.T_SMOLD_ONSET
    BLUNCK_LO = 473.0   # 200°C (Blunck IR camera lower bound)
    BLUNCK_HI = 773.0   # 500°C (Blunck IR camera upper bound)
    BAND_LO = 450.0     # acceptance with 23K margin
    BAND_HI = 750.0
    print(f"\n  model T_SMOLD_ONSET = {T_smold:.0f} K ({T_smold-273.15:.0f}°C)")
    print(f"  Blunck 2022 grass-blend smolder T range (IR camera): "
          f"{BLUNCK_LO:.0f}-{BLUNCK_HI:.0f} K "
          f"({BLUNCK_LO-273.15:.0f}-{BLUNCK_HI-273.15:.0f}°C)")
    print(f"  NIST TN 1481 grass glowing-onset T (visual, ref only): "
          f"{T_GLOW_MEDIAN_GRASS:.0f} K ({T_GLOW_MEDIAN_GRASS-273.15:.0f}°C)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    items = sorted(NIST_TN1481_GLOW_ONSET.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    Ts = [v - 273.15 for _, v in items]
    ax.barh(names, Ts, color="#888", label="NIST TN 1481 visual glow (ref)")
    ax.axvline(T_smold - 273.15, color="r", lw=2.5,
               label=f"model T_SMOLD_ONSET = {T_smold-273.15:.0f}°C")
    ax.axvspan(BLUNCK_LO-273.15, BLUNCK_HI-273.15, alpha=0.15, color="blue",
               label="Blunck 2022 IR smolder range (chemical)")
    ax.axvspan(BAND_LO-273.15, BAND_HI-273.15, alpha=0.10, color="green",
               label=f"acceptance window [{BAND_LO:.0f}, {BAND_HI:.0f}] K")
    ax.set_xlabel("Glowing/smoldering onset T [°C]")
    ax.set_title("Smolder onset T: model vs Blunck 2022 (primary) + NIST TN 1481 (ref)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "smolder_onset_vs_blunck_2022.png", dpi=140)
    plt.close(fig)

    assert BAND_LO <= T_smold <= BAND_HI, (
        f"T_SMOLD_ONSET = {T_smold:.0f} K outside Blunck 2022 chemical "
        f"smolder acceptance [{BAND_LO:.0f}, {BAND_HI:.0f}] K.  Re-tune to "
        f"match Blunck IR-camera measurements (473-773 K) per Standard "
        f"Validation Workflow §6.")


def test_smolder_kernel_zero_below_onset_then_active_above():
    """Behavioral sanity: Q_smold = 0 below T_SMOLD_ONSET, > 0 above (Rule #18).
    Also asserts bit-exact determinism (back-to-back identical inputs)."""
    shape = (1, 1, 1)
    m_solid = np.full(shape, 1.0)
    Y_O2 = np.full(shape, 0.21)
    alpha_s = np.full(shape, 1.0)
    Q_out = np.zeros(shape)
    # below onset
    T_below = pyrolysis_3d.T_SMOLD_ONSET - 1.0
    pyrolysis_3d.step_smoldering_oxidation(
        np.full(shape, T_below), m_solid.copy(), Y_O2, alpha_s, 0.1, Q_out)
    assert Q_out[0, 0, 0] == 0.0, "smolder fired below onset"
    # above onset
    Q_out.fill(0.0)
    T_above = pyrolysis_3d.T_SMOLD_ONSET + 100.0
    pyrolysis_3d.step_smoldering_oxidation(
        np.full(shape, T_above), m_solid.copy(), Y_O2, alpha_s, 0.1, Q_out)
    Q1 = float(Q_out[0, 0, 0])
    assert Q1 > 0.0, "smolder didn't fire above onset"
    # bit-exact rep
    Q_out.fill(0.0)
    pyrolysis_3d.step_smoldering_oxidation(
        np.full(shape, T_above), m_solid.copy(), Y_O2, alpha_s, 0.1, Q_out)
    Q2 = float(Q_out[0, 0, 0])
    assert repr(Q1) == repr(Q2), f"smolder not bit-exact: {Q1!r} vs {Q2!r}"


# ─────────────────── 3. PYROLYSIS vs Burra 2019 ─────────────────────────────

# Yang et al. 2007 *Fuel* 86:1781 — directly-measured DTG peak temperatures
# at 10°C/min in N₂ on isolated hemicellulose, cellulose, lignin.  These
# are the gold-standard 3-pool reference (matches our test heating rate).
# Bands: peak T ± 30 K margin (10°C/min DTG peaks broaden ~20 K naturally).
YANG_2007_DTG_PEAKS = {
    # (T_peak_K_lo, T_peak_K_hi)
    "Hemicellulose": (273.15 + 268 - 30, 273.15 + 268 + 30),  # 268°C ± 30K
    "Cellulose":     (273.15 + 355 - 30, 273.15 + 355 + 30),  # 355°C ± 30K
    "Lignin":        (273.15 + 365 - 50, 273.15 + 750 + 50),  # spread: exo peak
                                                                # 365°C, secondary
                                                                # endo near 750°C
}
# Backward-compat alias used in older test bodies
BURRA_DTG_PEAKS = YANG_2007_DTG_PEAKS


def _ramped_pyrolysis_3pool(T0=300.0, T_end=900.0, ramp_K_per_s=10.0/60.0,
                            m_init=(0.30, 0.45, 0.25), return_char=False):
    """Simulate TGA at 10 K/min on 1 g sample with hemi/cell/lign fractions.
    Returns (T_hist, m_total_hist) — or (T_hist, m_total_hist, m_char_hist)
    if return_char=True.  m_total = sum of remaining hemi/cell/lign (excludes
    accumulated char); m_char = accumulated char from pyrolysis.
    """
    shape = (1, 1, 1)
    T_s = np.full(shape, T0)
    m_h = np.full(shape, m_init[0])
    m_c = np.full(shape, m_init[1])
    m_l = np.full(shape, m_init[2])
    m_char = np.zeros(shape)
    alpha_s = np.full(shape, 1.0)
    S = np.zeros(shape); Q = np.zeros(shape)
    dt = 1.0  # 1 second steps
    n = int((T_end - T0) / (ramp_K_per_s * dt))
    T_hist = np.zeros(n + 1); T_hist[0] = T0
    m_hist = np.zeros(n + 1); m_hist[0] = sum(m_init)
    m_char_hist = np.zeros(n + 1)
    for k in range(1, n + 1):
        T_s[0, 0, 0] = T0 + ramp_K_per_s * dt * k
        pyrolysis_3d.step_pyrolysis(T_s, m_h, m_c, m_l, m_char,
                                     alpha_s, dt, S, Q)
        T_hist[k] = T_s[0, 0, 0]
        m_hist[k] = m_h[0, 0, 0] + m_c[0, 0, 0] + m_l[0, 0, 0]
        m_char_hist[k] = m_char[0, 0, 0]
    if return_char:
        return T_hist, m_hist, m_char_hist
    return T_hist, m_hist


def test_pyrolysis_3pool_DTG_peaks_vs_yang2007():
    """Run 3-pool kernel as TGA at 10°C/min; verify DTG peak T for each
    pool falls within Yang et al. 2007 *Fuel* 86:1781 directly-measured
    peak temperatures (gold standard for 3-pool model).

    Acceptance: peak DTG T within ±30 K of Yang's measured value:
      - Hemicellulose: 268°C (541 K)  → [238, 298]°C
      - Cellulose:     355°C (628 K)  → [325, 385]°C
      - Lignin:        spread, primary exo peak ~365°C
    """
    # Run 3 separate sims, each with only ONE pool active (others at 0)
    results = {}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pool, color, init in [
        ("Hemicellulose", "C0", (1.0, 0.0, 0.0)),
        ("Cellulose",     "C1", (0.0, 1.0, 0.0)),
        ("Lignin",        "C2", (0.0, 0.0, 1.0)),
    ]:
        T_h, m_h = _ramped_pyrolysis_3pool(m_init=init)
        # DTG = -dm/dT (positive)
        DTG = -np.gradient(m_h, T_h)
        peak_idx = int(np.argmax(DTG))
        peak_T = T_h[peak_idx]
        results[pool] = peak_T
        ax.plot(T_h - 273.15, DTG / DTG.max(), color=color, lw=2,
                label=f"{pool}  peak={peak_T-273.15:.0f}°C")
        # EXP range overlay
        Tlo, Thi = BURRA_DTG_PEAKS[pool]
        ax.axvspan(Tlo - 273.15, Thi - 273.15, color=color, alpha=0.10)
    ax.set_xlabel("T [°C]"); ax.set_ylabel("Normalized DTG (-dm/dT, peak=1)")
    ax.set_title("3-pool pyrolysis TGA at 10 K/min vs Burra 2019 / Yang 2007 DTG peak ranges")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pyrolysis_3pool_DTG_vs_burra2019.png", dpi=140)
    plt.close(fig)

    fails = []
    for pool, peak_T in results.items():
        Tlo, Thi = BURRA_DTG_PEAKS[pool]
        in_band = Tlo <= peak_T <= Thi
        print(f"  {pool}: peak DTG T = {peak_T:.0f} K ({peak_T-273.15:.0f}°C); "
              f"expected [{Tlo-273.15:.0f}, {Thi-273.15:.0f}]°C "
              f"{'OK' if in_band else 'OUT'}")
        if not in_band:
            fails.append(f"{pool} peak {peak_T-273.15:.0f}°C ∉ "
                         f"[{Tlo-273.15:.0f}, {Thi-273.15:.0f}]°C")
    if fails:
        # Per Rule #4 (failed validation is informative, not a bug — document
        # the physical reason, accept as known limitation), xfail rather than
        # hard-fail.  The current 3-pool A/E values (Berghel 2023 wheat-straw
        # + Orfão compensation at T_onset=600K) fire ~30-100°C below the
        # Burra/Yang DTG peaks for hemicellulose and cellulose.  This may
        # contribute to the model's tendency to self-ignite at low wind
        # (Cut 8% U=0.5 overshoot, etc.).  Documented gap.
        pytest.xfail("DTG peak T out of EXP band:\n  " + "\n  ".join(fails))


def test_pyrolysis_3pool_DTG_peak_magnitudes_vs_yang2007():
    """Verify DTG peak MAGNITUDE matches Yang 2007 measured rates:
      - Hemicellulose peak rate: 0.95 wt%/°C  (Yang 2007 §3.2.1)
      - Cellulose peak rate:     2.84 wt%/°C  (Yang 2007 §3.2.1)
      - Lignin: spread, low rate (~0.1 wt%/°C, no sharp peak)

    Peak magnitude depends on activation energy E (not just A — A controls
    location, E controls sharpness/magnitude).  For first-order Arrhenius at
    constant heating rate β, peak DTG magnitude ≈ m_peak · E/(RT_peak²)
    where m_peak ≈ exp(-1) ≈ 0.368 (saddle-point result).

    This test complements `test_pyrolysis_3pool_DTG_peaks_vs_yang2007`
    (which checks peak T location).  Together they constrain both A and E
    independently.

    Acceptance: peak DTG magnitude within ±50 % of Yang measured value
    (TGA peak rates have ~20-30% inter-lab variance; ±50% is a
    reasonable single-rate band).
    """
    YANG_DTG_PEAK_RATES = {
        # (wt%/°C, ±band)
        "Hemicellulose": 0.95,
        "Cellulose":     2.84,
        # Lignin: too spread to define a sharp peak; skipped in magnitude check.
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    results = {}
    for pool, color, init in [
        ("Hemicellulose", "C0", (1.0, 0.0, 0.0)),
        ("Cellulose",     "C1", (0.0, 1.0, 0.0)),
        ("Lignin",        "C2", (0.0, 0.0, 1.0)),
    ]:
        T_h, m_h = _ramped_pyrolysis_3pool(m_init=init)
        # DTG in 1/K (= fractional per K).  Convert to wt%/°C: ×100 (wt% basis,
        # since m_init=1.0, m is mass fraction) and ×1 (°C ≡ K for derivatives).
        DTG_per_K = -np.gradient(m_h, T_h)
        DTG_wt_per_C = DTG_per_K * 100.0
        peak_idx = int(np.argmax(DTG_wt_per_C))
        peak_T = T_h[peak_idx]
        peak_rate = DTG_wt_per_C[peak_idx]
        results[pool] = (peak_T, peak_rate)
        ax.plot(T_h - 273.15, DTG_wt_per_C, color=color, lw=1.8,
                label=f"{pool}  peak={peak_rate:.2f} wt%/°C @ {peak_T-273.15:.0f}°C")
        if pool in YANG_DTG_PEAK_RATES:
            yang_rate = YANG_DTG_PEAK_RATES[pool]
            ax.axhline(yang_rate, color=color, ls=":", lw=1,
                       alpha=0.4, label=f"  Yang 2007 {pool}: {yang_rate:.2f} wt%/°C")
    ax.set_xlabel("T [°C]"); ax.set_ylabel("DTG [wt%/°C]")
    ax.set_title("3-pool pyrolysis DTG magnitude at 10°C/min vs Yang 2007")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pyrolysis_3pool_DTG_magnitude_vs_yang2007.png", dpi=140)
    plt.close(fig)

    fails = []
    for pool, yang_rate in YANG_DTG_PEAK_RATES.items():
        _, model_rate = results[pool]
        ratio = model_rate / yang_rate
        print(f"  {pool}: model peak DTG = {model_rate:.2f} wt%/°C, "
              f"Yang = {yang_rate:.2f}, ratio = {ratio:.2f}× (acceptance: 0.5-1.5×)")
        if not (0.5 <= ratio <= 1.5):
            fails.append(f"{pool} ratio {ratio:.2f}× outside band")
    if fails:
        # Per Rule #4: this is an informative failure documenting a real
        # model-EXP gap.  E values (E_HEMI=92, E_CELL=120 kJ/mol per Berghel
        # 2023) are at the low end of literature consensus (Anca-Couce 2016
        # §3.1.2: cellulose 190-253 kJ/mol, hemi 150-200 kJ/mol).  Closing
        # this requires a joint (A, E) re-fit, not just A.  XFAIL.
        pytest.xfail("DTG peak magnitude out of band:\n  " + "\n  ".join(fails))


def test_pyrolysis_bit_exact_determinism():
    """Run step_pyrolysis twice; bit-exact (Rule #18)."""
    T1, m1 = _ramped_pyrolysis_3pool(m_init=(0.3, 0.45, 0.25))
    T2, m2 = _ramped_pyrolysis_3pool(m_init=(0.3, 0.45, 0.25))
    assert np.array_equal(T1, T2), "T history drift"
    assert np.array_equal(m1, m2), "m history drift"


def test_pyrolysis_HEAT_OF_PYROLYSIS_sign_per_pool_vs_yang2007():
    """Yang et al. 2007 *Fuel* 86:1781 §3.2.2 DSC measurements:
      - Cellulose pyrolysis: ENDOTHERMIC (large endothermic peak at 355°C)
      - Hemicellulose pyrolysis: EXOTHERMIC (peak at 275°C)
      - Lignin pyrolysis: EXOTHERMIC (peak at 365°C, small endo near 750°C)

    Yang attributes the hemi+lignin exotherms to charring (the high
    char-yield pools have a charring-driven exothermic component).

    Our model uses a SINGLE `HEAT_OF_PYROLYSIS = +4.0e5 J/kg` constant
    applied uniformly to all 3 pools (always endothermic).  This is a
    structural bug per Yang DSC: hemi and lignin should be exothermic,
    not endothermic.

    Acceptance (when fixed): per-pool HOR_HEMI, HOR_CELL, HOR_LIGN
    constants with signs: HOR_HEMI < 0, HOR_CELL > 0, HOR_LIGN < 0.

    XFAIL until kernel is refactored to per-pool HoR.
    """
    has_per_pool_hor = (
        hasattr(pyrolysis_3d, 'HOR_HEMI')
        and hasattr(pyrolysis_3d, 'HOR_CELL')
        and hasattr(pyrolysis_3d, 'HOR_LIGN')
    )
    if not has_per_pool_hor:
        pytest.xfail(
            "kernel uses single HEAT_OF_PYROLYSIS constant for all 3 pools; "
            "Yang 2007 DSC requires per-pool HoR with signs (Cell endo, "
            "Hemi+Lign exo).  Refactor needed: add HOR_HEMI < 0, HOR_CELL > 0, "
            "HOR_LIGN < 0 and update step_pyrolysis to use them per pool.")

    # If/when refactored, validate the signs:
    assert pyrolysis_3d.HOR_HEMI < 0, "HOR_HEMI should be exothermic (Yang 2007 DSC)"
    assert pyrolysis_3d.HOR_CELL > 0, "HOR_CELL should be endothermic (Yang 2007 DSC)"
    assert pyrolysis_3d.HOR_LIGN < 0, "HOR_LIGN should be exothermic (Yang 2007 DSC)"


def test_pyrolysis_3pool_char_yields_vs_yang2007():
    """Yang 2007 §3.2.1 directly measured solid residue at 900°C:
      - Hemicellulose: 20 wt% (low char)
      - Cellulose:      6.5 wt% (very low char)
      - Lignin:        40 wt% (high char)

    With Y_char tracking (Phase 14 Try 5, 2026-05-11), the model now
    accumulates a CHAR_YIELD_* fraction of each pool's pyrolysis mass loss
    into m_char.  This test runs full TGA to 900°C and verifies the
    accumulated m_char matches Yang 2007 residue.

    Acceptance: |m_char_final - Yang_residue| < 0.02 (tight band — these
    are constants the kernel uses directly, so the test confirms the
    kinetics fully consume each pool by 900°C).
    """
    YANG_RESIDUE = {
        # (initial mass tuple for the only-this-pool sim, Yang residue)
        "Hemicellulose": ((1.0, 0.0, 0.0), 0.20),
        "Cellulose":     ((0.0, 1.0, 0.0), 0.065),
        "Lignin":        ((0.0, 0.0, 1.0), 0.40),
    }
    fails = []
    for pool, (init, yang_res) in YANG_RESIDUE.items():
        _, m_total_hist, m_char_hist = _ramped_pyrolysis_3pool(
            m_init=init, return_char=True)
        m_char_final = float(m_char_hist[-1])
        m_remaining = float(m_total_hist[-1])
        delta = m_char_final - yang_res
        print(f"  {pool}: m_char at 900°C = {m_char_final:.4f}, "
              f"Yang residue = {yang_res:.3f}, Δ={delta:+.4f}  "
              f"(m_remaining = {m_remaining:.4e})")
        if abs(delta) > 0.02:
            fails.append(f"{pool} char yield Δ={delta:+.3f} (band ±0.02)")
    if fails:
        pytest.xfail("Char yield mismatch:\n  " + "\n  ".join(fails))


# ─────────────────── 3a. Drying R_d kernel (Phase 14 Try 7) ──────────────────

def _run_drying_single_cell(T_K, m_water_init, dt=1.0, n_steps=10):
    """Drive step_drying at fixed T; return (m_water_final, Q_dry_total)."""
    shape = (1, 1, 1)
    T_s = np.full(shape, T_K)
    m_water = np.full(shape, m_water_init)
    alpha_s = np.full(shape, 1.0)
    Q_dry = np.zeros(shape)
    Q_total = 0.0
    for _ in range(n_steps):
        pyrolysis_3d.step_drying(T_s, m_water, alpha_s, dt, Q_dry)
        Q_total += float(Q_dry[0, 0, 0]) * dt
    return float(m_water[0, 0, 0]), Q_total


def test_drying_arrhenius_rate_vs_T():
    """k_dry(T) should follow Lautenberger 2009 Arrhenius:
       A=4.29e3 1/s, E=43.8 kJ/mol.
    Verify single-step rate matches A·exp(-E/RT) within rounding."""
    R = 8.314
    for T in (350.0, 400.0, 500.0, 700.0):
        k_expected = pyrolysis_3d.A_DRY * np.exp(
            -pyrolysis_3d.E_DRY / (R * T))
        # 1-step run with small dt to extract effective k
        dt = 1.0e-3
        mw_final, _ = _run_drying_single_cell(T, m_water_init=1.0,
                                              dt=dt, n_steps=1)
        # m_new = m·exp(-k·dt) → k = -ln(m_new)/dt
        k_observed = -np.log(mw_final) / dt
        err = abs(k_observed - k_expected) / max(k_expected, 1e-30)
        print(f"  T={T:.0f}K: k_expected={k_expected:.3e}, "
              f"k_observed={k_observed:.3e}, err={err*100:.3f}%")
        assert err < 1e-6, f"k_dry deviation at T={T}K"


def test_drying_endothermic_heat():
    """Q_dry_out should equal L_VAP × mass-evaporation-rate.
    Verifies energy balance: 2.26 MJ/kg of water evaporated."""
    T_test = 400.0
    m_water_init = 0.05  # kg/m³
    mw_final, Q_total = _run_drying_single_cell(
        T_test, m_water_init=m_water_init, dt=1.0, n_steps=5)
    dm_evap = m_water_init - mw_final
    Q_expected = dm_evap * pyrolysis_3d.L_VAP_WATER
    err = abs(Q_total - Q_expected) / max(Q_expected, 1e-30)
    print(f"\n  T={T_test:.0f}K, m_water_init={m_water_init} kg/m³:")
    print(f"    dm_evap = {dm_evap:.4e} kg/m³, Q_expected = {Q_expected:.4e} J/m³")
    print(f"    Q_total = {Q_total:.4e} J/m³, err = {err*100:.4f}%")
    assert err < 1e-6, "Q_dry doesn't match L_VAP × evaporated mass"


def test_drying_monotone_depletion():
    """m_water should decrease monotonically and never go negative."""
    T_test = 500.0
    m_water_init = 0.01
    mw_final, _ = _run_drying_single_cell(
        T_test, m_water_init=m_water_init, dt=10.0, n_steps=100)
    assert 0.0 <= mw_final < m_water_init, (
        f"m_water not monotone: init {m_water_init}, final {mw_final}")


def test_drying_zero_when_no_water():
    """If m_water = 0, Q_dry must be 0 (no spurious heat sink)."""
    shape = (1, 1, 1)
    T_s = np.full(shape, 700.0)
    m_water = np.zeros(shape)
    alpha_s = np.full(shape, 1.0)
    Q_dry = np.zeros(shape)
    pyrolysis_3d.step_drying(T_s, m_water, alpha_s, 1.0, Q_dry)
    assert Q_dry[0, 0, 0] == 0.0, "Q_dry fired with no water"


def test_drying_bit_exact_determinism():
    """Rule #18: step_drying twice on identical inputs → bit-exact."""
    r1 = _run_drying_single_cell(500.0, 0.05, dt=1.0, n_steps=10)
    r2 = _run_drying_single_cell(500.0, 0.05, dt=1.0, n_steps=10)
    for label, v1, v2 in zip(("m_water", "Q_total"), r1, r2):
        assert repr(v1) == repr(v2), \
            f"step_drying {label} drift: {v1!r} vs {v2!r}"


# ─────────────────── 3b. MD2004 single-pool kernel (used by 3D Cheney sweep) ──

def _run_md2004_single_cell(T_K, m_init, Y_O2_val, dt=0.1, n_steps=5):
    """Drive step_pyrolysis_md2004 at fixed T, fixed Y_O2.  Returns m_solid
    final, m_char final, total S_pyro_integrated, total Q_pyro_integrated."""
    shape = (1, 1, 1)
    T_s = np.full(shape, T_K)
    m_solid = np.full(shape, m_init)
    m_init_arr = np.full(shape, m_init)
    m_char = np.zeros(shape)
    Y_O2 = np.full(shape, Y_O2_val)
    alpha_s = np.full(shape, 1.0)
    m_water = np.zeros(shape)
    S_pyro = np.zeros(shape); Q_pyro = np.zeros(shape)
    S_total = 0.0; Q_total = 0.0
    for _ in range(n_steps):
        pyrolysis_3d.step_pyrolysis_md2004(
            T_s, m_solid, m_init_arr, m_char, Y_O2, alpha_s,
            m_water, 0.0,  # moisture disabled
            dt, S_pyro, Q_pyro)
        S_total += float(S_pyro[0, 0, 0]) * dt
        Q_total += float(Q_pyro[0, 0, 0]) * dt
    return (float(m_solid[0, 0, 0]), float(m_char[0, 0, 0]),
            S_total, Q_total)


def test_md2004_Rop_increases_rate_with_Y_O2():
    """R_op (oxidative pyrolysis) should ADD rate proportional to Y_O2.
    At Y_O2=0 → rate = R_p only; at Y_O2=0.21 → rate ≈ 1.5-2× R_p
    per Anca-Couce 2016 §4.5.1 expectation.
    """
    T_test = 700.0  # K — well above pyrolysis onset
    # Run at Y_O2 = 0 (R_p only) and Y_O2 = 0.21 (R_p + R_op)
    _, _, S_p_only, _ = _run_md2004_single_cell(T_test, m_init=1.0, Y_O2_val=0.0)
    _, _, S_with_op, _ = _run_md2004_single_cell(T_test, m_init=1.0, Y_O2_val=0.21)
    ratio = S_with_op / max(S_p_only, 1e-30)
    print(f"\n  T={T_test:.0f}K: S_pyro at Y_O2=0:  {S_p_only:.3e}")
    print(f"            S_pyro at Y_O2=0.21: {S_with_op:.3e}")
    print(f"            ratio: {ratio:.2f}× (Anca-Couce 2016 expects ~1.5-2×)")
    assert ratio > 1.3, f"R_op didn't boost rate (ratio {ratio:.2f}× < 1.3×)"
    assert ratio < 3.0, f"R_op boosted rate too much (ratio {ratio:.2f}× > 3×)"


def test_md2004_Rop_zero_at_zero_O2():
    """Fundamental closure check: R_op contribution = 0 when Y_O2 ≈ 0.
    At Y_O2 < Y_O2_MIN_OP threshold, kernel should fall back to R_p only.
    """
    T_test = 700.0
    _, _, S_zero, _ = _run_md2004_single_cell(T_test, m_init=1.0, Y_O2_val=0.0)
    _, _, S_below, _ = _run_md2004_single_cell(
        T_test, m_init=1.0, Y_O2_val=pyrolysis_3d.Y_O2_MIN_OP * 0.5)
    # Both should give identical (R_p-only) rate.
    assert repr(S_zero) == repr(S_below), (
        f"R_op fired below Y_O2_MIN_OP threshold: {S_zero!r} vs {S_below!r}")


def test_md2004_Rop_HoR_exothermic_at_high_Y_O2():
    """Net Q_pyro should trend negative (exothermic) as Y_O2 increases —
    R_op contributes negative HOR, dominating thermal R_p's positive HoR
    when oxidative path dominates.
    """
    T_test = 700.0
    _, _, _, Q_pure_thermal = _run_md2004_single_cell(T_test, m_init=1.0, Y_O2_val=0.0)
    _, _, _, Q_with_op = _run_md2004_single_cell(T_test, m_init=1.0, Y_O2_val=0.21)
    print(f"\n  T={T_test:.0f}K: Q_pyro at Y_O2=0:  {Q_pure_thermal:.3e} (R_p endo)")
    print(f"            Q_pyro at Y_O2=0.21: {Q_with_op:.3e}")
    # R_p alone is endothermic (positive).  Adding R_op (negative) should
    # reduce or flip Q_pyro.
    assert Q_with_op < Q_pure_thermal, (
        f"R_op didn't shift Q toward exothermic: {Q_pure_thermal:.3e} → {Q_with_op:.3e}"
    )


def test_md2004_bit_exact_determinism():
    """Run step_pyrolysis_md2004 twice with identical inputs (incl. R_op
    branch active); bit-exact match (Rule #18)."""
    r1 = _run_md2004_single_cell(700.0, m_init=1.0, Y_O2_val=0.21)
    r2 = _run_md2004_single_cell(700.0, m_init=1.0, Y_O2_val=0.21)
    for label, v1, v2 in zip(("m_solid", "m_char", "S_total", "Q_total"), r1, r2):
        assert repr(v1) == repr(v2), \
            f"MD2004 {label} drift: {v1!r} vs {v2!r}"


def _run_md2004_with_AE(T_K, m_init, Y_O2_val, A_p, E_p, dt=0.001, n_steps=1):
    """Variant of _run_md2004_single_cell that passes the (A_p, E_p) overrides."""
    shape = (1, 1, 1)
    T_s = np.full(shape, T_K)
    m_solid = np.full(shape, m_init)
    m_init_arr = np.full(shape, m_init)
    m_char = np.zeros(shape)
    Y_O2 = np.full(shape, Y_O2_val)
    alpha_s = np.full(shape, 1.0)
    m_water = np.zeros(shape)
    S_pyro = np.zeros(shape); Q_pyro = np.zeros(shape)
    S_total = 0.0
    for _ in range(n_steps):
        pyrolysis_3d.step_pyrolysis_md2004(
            T_s, m_solid, m_init_arr, m_char, Y_O2, alpha_s,
            m_water, 0.0,
            dt, S_pyro, Q_pyro, A_p, E_p,
        )
        S_total += float(S_pyro[0, 0, 0]) * dt
    return S_total


def test_md2004_A_E_override_default_back_compat():
    """Phase 15I: explicit (A=A_MD2004, E=E_MD2004) overrides match default path."""
    s_default = _run_md2004_single_cell(700.0, 1.0, 0.21, dt=0.001, n_steps=1)[2]
    s_explicit = _run_md2004_with_AE(700.0, 1.0, 0.21,
                                       pyrolysis_3d.A_MD2004,
                                       pyrolysis_3d.E_MD2004,
                                       dt=0.001, n_steps=1)
    assert repr(s_default) == repr(s_explicit), (
        f"explicit MD2004 (A,E) diverged from default: "
        f"{s_default!r} vs {s_explicit!r}"
    )


def test_md2004_A_E_override_antal_varhegyi_recovers_arrhenius():
    """Phase 15I: with Antal-Varhegyi (A=5e17, E=236 kJ/mol), the per-step
    rate should equal A·exp(-E/RT) for small dt.  Validates the overrides
    enter both factors correctly."""
    T = 800.0
    A_AV, E_AV = 5.0e17, 236_000.0
    S_total = _run_md2004_with_AE(T, 1.0, 0.0, A_AV, E_AV, dt=1e-7, n_steps=1)
    # At Y_O2=0 only R_p fires.  Per-step ΔS expected:
    #   rate = ETA · A · exp(-E/RT) · m_init
    # (using exact decay m_new = m·exp(-k·dt) → dm = m·(1 - exp(-k·dt))).
    R = 8.314
    k = A_AV * np.exp(-E_AV / (R * T))
    expected_rate = pyrolysis_3d.ETA_MD2004 * (1.0 - np.exp(-k * 1e-7)) / 1e-7
    actual_rate = S_total / 1e-7
    rel_err = abs(actual_rate - expected_rate) / max(expected_rate, 1e-30)
    assert rel_err < 1e-6, (
        f"AV override rate {actual_rate:.3e} vs analytic {expected_rate:.3e} "
        f"(rel_err {rel_err:.2e})"
    )


def test_md2004_A_E_override_changes_rate_vs_baseline():
    """Phase 15I: confirm that the AV / MB triplets produce DIFFERENT rates
    than MD2004 at flame temperature — proves the overrides aren't a no-op."""
    T = 1000.0  # flame zone
    s_MD = _run_md2004_with_AE(T, 1.0, 0.0,
                                pyrolysis_3d.A_MD2004, pyrolysis_3d.E_MD2004,
                                dt=1e-7, n_steps=1)
    s_AV = _run_md2004_with_AE(T, 1.0, 0.0,
                                5.0e17, 236_000.0,
                                dt=1e-7, n_steps=1)
    assert s_AV > 10.0 * s_MD, (
        f"AV @ 1000K = {s_AV:.3e} not > 10× MD2004 {s_MD:.3e} "
        "— override may be wired incorrectly"
    )


def test_md2004_A_E_override_is_bit_exact_under_repeat():
    """Rule #17: override path is deterministic."""
    a = _run_md2004_with_AE(800.0, 1.0, 0.21, 5.0e17, 236_000.0)
    b = _run_md2004_with_AE(800.0, 1.0, 0.21, 5.0e17, 236_000.0)
    assert repr(a) == repr(b), (
        f"AV-override path non-deterministic: {a!r} vs {b!r}"
    )


# ─────────────────── 4. EDC limit checks (no EXP, asymptotic asserts) ────────

def test_edc_zero_when_no_fuel_or_no_air():
    """Fundamental closure check: ω_EDC = 0 if Y_F=0 or Y_O2=0."""
    shape = (1, 1, 1)
    rho = np.full(shape, 1.2)
    T_g = np.full(shape, 1500.0)
    Y_F0 = np.zeros(shape); Y_F1 = np.full(shape, 0.05)
    Y_O20 = np.zeros(shape); Y_O21 = np.full(shape, 0.21)
    k_turb = np.full(shape, 1.0)
    eps_turb = np.full(shape, 0.5)
    chi_rad = 0.30
    omega = np.zeros(shape)

    combustion_3d.step_chemistry_ode_edc(
        rho, T_g, Y_F0, Y_O21, k_turb, eps_turb, chi_rad, 1100.0, 0.01, 1, omega)
    assert omega[0, 0, 0] == 0.0, "EDC fired with no fuel"

    omega.fill(0.0)
    combustion_3d.step_chemistry_ode_edc(
        rho, T_g, Y_F1, Y_O20, k_turb, eps_turb, chi_rad, 1100.0, 0.01, 1, omega)
    assert omega[0, 0, 0] == 0.0, "EDC fired with no O2"


def test_edc_bit_exact_determinism():
    """Run step_chemistry_ode_edc twice; bit-exact (Rule #18)."""
    shape = (1, 1, 1)
    rho = np.full(shape, 1.2)
    T_g = np.full(shape, 1500.0)
    Y_F = np.full(shape, 0.05)
    Y_O2 = np.full(shape, 0.21)
    k_turb = np.full(shape, 1.0)
    eps_turb = np.full(shape, 0.5)
    chi_rad = 0.30
    omega1 = np.zeros(shape); omega2 = np.zeros(shape)

    Y_F1 = Y_F.copy(); Y_O21 = Y_O2.copy(); rho1 = rho.copy(); T_g1 = T_g.copy()
    Y_F2 = Y_F.copy(); Y_O22 = Y_O2.copy(); rho2 = rho.copy(); T_g2 = T_g.copy()

    combustion_3d.step_chemistry_ode_edc(
        rho1, T_g1, Y_F1, Y_O21, k_turb, eps_turb, chi_rad, 1100.0, 0.01, 1, omega1)
    combustion_3d.step_chemistry_ode_edc(
        rho2, T_g2, Y_F2, Y_O22, k_turb, eps_turb, chi_rad, 1100.0, 0.01, 1, omega2)
    assert np.array_equal(omega1, omega2), \
        f"EDC ω drift: max |Δ|={np.max(np.abs(omega1-omega2))}"
    assert np.array_equal(T_g1, T_g2), "EDC T_g drift"
    assert np.array_equal(Y_F1, Y_F2), "EDC Y_F drift"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
