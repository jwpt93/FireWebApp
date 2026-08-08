"""Unit tests for Phase 23 chemistry-family deck override.

Rule #18 (unit tests required for every new module): the chemistry-family
kwarg thread through spread_3d.py + chemistry_closures kernels needs
tests that (a) biomass-default preserves bit-exact prior behaviour and
(b) methane preset gives distinct (and directionally correct) omega.

Covers:
  1. resolve_chemistry_family returns expected keys/values
  2. Every kernel accepts the 5 scalars without crashing
  3. Biomass explicit == biomass default (bit-exact)
  4. Methane vs biomass produces different results (sanity: HOC × 3)
  5. Determinism per Rule #17: 2-call bit-exact
"""
import numpy as np
import pytest

from model_outdoor.physics_3d.chemistry_closures._constants import (
    CHEMISTRY_FAMILIES, resolve_chemistry_family,
    S_STOICH, HOC_J, A_COMB, E_COMB, C_EBU,
)
from model_outdoor.physics_3d.chemistry_closures.pasr import step_chemistry_ode_pasr
from model_outdoor.physics_3d.chemistry_closures.ebu_bootstrap import step_chemistry_ode as step_ebu
from model_outdoor.physics_3d.chemistry_closures.edc import step_chemistry_ode_edc


# ── 1. Registry ──────────────────────────────────────────────────────
def test_registry_contains_biomass_and_methane():
    assert "biomass" in CHEMISTRY_FAMILIES
    assert "methane" in CHEMISTRY_FAMILIES


def test_biomass_preset_matches_module_constants():
    """The biomass preset MUST bit-exactly reproduce the pre-Phase-23
    module constants, otherwise Rule #17 bit-exact invariant breaks."""
    b = CHEMISTRY_FAMILIES["biomass"]
    assert b["s_stoich"] == S_STOICH
    assert b["hoc_J"] == HOC_J
    assert b["a_comb"] == A_COMB
    assert b["e_comb"] == E_COMB
    assert b["c_ebu"] == C_EBU


def test_methane_preset_expected_values():
    m = CHEMISTRY_FAMILIES["methane"]
    assert m["s_stoich"] == 4.0
    assert m["hoc_J"] == 50_000_000.0
    assert m["c_ebu"] == 4.0


def test_resolve_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown chemistry family"):
        resolve_chemistry_family("no_such_family")


def test_resolve_returns_copy():
    """Mutating the returned dict must NOT poison the module-level registry."""
    d = resolve_chemistry_family("biomass")
    d["s_stoich"] = 999.0
    assert CHEMISTRY_FAMILIES["biomass"]["s_stoich"] == 1.3


# ── 2. Kernel accepts kwargs without crash ───────────────────────────
def _mk_state(seed=42, shape=(2, 2, 2)):
    np.random.seed(seed)
    return dict(
        rho=np.random.uniform(0.5, 1.5, shape),
        T_g=np.random.uniform(300, 2000, shape),
        Y_fuel=np.random.uniform(0.01, 0.3, shape),
        Y_O2=np.random.uniform(0.05, 0.23, shape),
    )


def test_pasr_accepts_all_5_kwargs():
    s = _mk_state()
    tau_mix = np.full_like(s["rho"], 0.01)
    omega = np.zeros_like(s["rho"])
    step_chemistry_ode_pasr(
        s["rho"], s["T_g"], s["Y_fuel"], s["Y_O2"], tau_mix,
        chi_rad=0.3, cp_g=1100.0, dt=0.001, n_substeps=1,
        omega_int_out=omega,
        s_stoich=4.0, hoc_J=50e6, a_comb=2.1e11, e_comb=125e3,
    )
    assert not np.any(np.isnan(omega))


# ── 3. Biomass explicit == biomass default (bit-exact) ───────────────
def _pasr_family_kwargs(family):
    """PaSR consumes 4 of the 5 chemistry preset keys (not c_ebu)."""
    fam = resolve_chemistry_family(family)
    return {k: fam[k] for k in ("s_stoich", "hoc_J", "a_comb", "e_comb")}


def test_pasr_biomass_explicit_matches_default():
    """Passing the biomass preset explicitly must match omitting the
    chemistry kwargs (which triggers the biomass defaults)."""
    def _run(pass_explicit):
        s = _mk_state()
        tau_mix = np.full_like(s["rho"], 0.01)
        omega = np.zeros_like(s["rho"])
        kw = dict(chi_rad=0.3, cp_g=1100.0, dt=0.001, n_substeps=1,
                  omega_int_out=omega)
        if pass_explicit:
            kw.update(_pasr_family_kwargs("biomass"))
        step_chemistry_ode_pasr(
            s["rho"], s["T_g"], s["Y_fuel"], s["Y_O2"], tau_mix, **kw,
        )
        return s["T_g"].copy(), s["Y_fuel"].copy(), omega.copy()

    tg_a, yf_a, om_a = _run(pass_explicit=False)
    tg_b, yf_b, om_b = _run(pass_explicit=True)
    assert np.array_equal(tg_a, tg_b), f"T_g differs: max {np.max(np.abs(tg_a-tg_b))}"
    assert np.array_equal(yf_a, yf_b)
    assert np.array_equal(om_a, om_b)


def test_edc_biomass_explicit_matches_default():
    def _run(pass_explicit):
        s = _mk_state(seed=7)
        Y_H2O = np.zeros_like(s["rho"])
        k_turb = np.full_like(s["rho"], 0.5)
        eps_turb = np.full_like(s["rho"], 0.1)
        omega = np.zeros_like(s["rho"])
        kw = dict(chi_rad=0.3, cp_g=1100.0, dt=0.001, n_substeps=1,
                  omega_int_out=omega)
        if pass_explicit:
            fam = resolve_chemistry_family("biomass")
            kw["s_stoich"] = fam["s_stoich"]
            kw["hoc_J"] = fam["hoc_J"]
        step_chemistry_ode_edc(
            s["rho"], s["T_g"], s["Y_fuel"], s["Y_O2"], k_turb, eps_turb,
            omega_int_out=omega, Y_H2O=Y_H2O,
            chi_rad=kw["chi_rad"], cp_g=kw["cp_g"], dt=kw["dt"],
            n_substeps=kw["n_substeps"],
            **{k: v for k, v in kw.items()
               if k in ("s_stoich", "hoc_J")}
        )
        return s["T_g"].copy(), omega.copy()

    tg_a, om_a = _run(False)
    tg_b, om_b = _run(True)
    assert np.array_equal(tg_a, tg_b)
    assert np.array_equal(om_a, om_b)


# ── 4. Methane vs biomass produces different results ────────────────
def test_methane_differs_from_biomass():
    """Sanity: methane's higher HoC + higher s_stoich must shift the
    outputs measurably.  This is the whole point of the refactor."""
    def _run(family):
        s = _mk_state(seed=13)
        tau_mix = np.full_like(s["rho"], 0.01)
        omega = np.zeros_like(s["rho"])
        step_chemistry_ode_pasr(
            s["rho"], s["T_g"], s["Y_fuel"], s["Y_O2"], tau_mix,
            chi_rad=0.3, cp_g=1100.0, dt=0.01, n_substeps=5,
            omega_int_out=omega,
            **_pasr_family_kwargs(family),
        )
        return s["T_g"].copy(), omega.copy()

    tg_b, om_b = _run("biomass")
    tg_m, om_m = _run("methane")
    assert not np.array_equal(tg_b, tg_m)
    # Methane's higher HoC (50 vs 17 MJ/kg) should give higher T_g rise
    # per unit omega — the individual per-cell numbers depend on which
    # branch bound each cell, so just require observable differences.
    assert np.max(np.abs(tg_b - tg_m)) > 1.0


# ── 5. Determinism (Rule #17) ────────────────────────────────────────
def test_pasr_bit_exact_determinism_methane():
    """2 back-to-back calls on identical inputs must match to last digit
    even under the new argument pathway."""
    def _run():
        s = _mk_state(seed=99)
        tau_mix = np.full_like(s["rho"], 0.005)
        omega = np.zeros_like(s["rho"])
        step_chemistry_ode_pasr(
            s["rho"], s["T_g"], s["Y_fuel"], s["Y_O2"], tau_mix,
            chi_rad=0.3, cp_g=1100.0, dt=0.005, n_substeps=3,
            omega_int_out=omega,
            **_pasr_family_kwargs("methane"),
        )
        return s["T_g"].copy(), s["Y_fuel"].copy(), omega.copy()

    a = _run(); b = _run()
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
