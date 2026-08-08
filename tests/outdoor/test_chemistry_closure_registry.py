"""Phase 15-0 — chemistry-closure registry framework tests.

Verifies:
  1. Registry exposes the three current closures (edc, ebu_bootstrap, pasr).
  2. Dispatch by name routes to the correct underlying kernel.
  3. Each closure silently ignores kwargs it does not consume — required
     so the main loop can pack one kwarg bag without per-closure logic.
  4. Unknown closure name raises ValueError.
  5. Bit-exact determinism (Rule #17): each closure called twice on
     identical inputs at the production thread count produces identical
     outputs to the last bit.
  6. Backward-compat: combustion_3d still re-exports the moved kernels
     and the shared constants so existing tests do not break.

Rule #17 + #18 — these tests freeze the bit-exact behaviour of each
closure at Phase 15-0 landing; any future regression is detected at
commit time.

Run:
    /home/jw/.venvs/unitiedmodel2/bin/python -m pytest \
        tests/outdoor/test_chemistry_closure_registry.py -v
"""
from __future__ import annotations

import os

# Pin thread count for determinism asserts (Rule #17 — same as
# 2-case bit-exact rule used by the production sweep).
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("NUMBA_NUM_THREADS", "12")

import numpy as np
import pytest

from model_outdoor.physics_3d import chemistry_closures
from model_outdoor.physics_3d import combustion_3d


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_state(shape=(4, 4, 4), seed=0):
    """Build a small but non-trivial gas state for closure dispatch.

    Spatially-varying T, Y_F, Y_O2 to make the kernels actually do work
    (constant fields can pass spuriously when a kernel short-circuits).
    """
    rng = np.random.default_rng(seed)
    rho = np.full(shape, 1.0) + 0.1 * rng.random(shape)
    T_g = 800.0 + 400.0 * rng.random(shape)             # 800..1200 K
    Y_fuel = 0.02 + 0.05 * rng.random(shape)            # 0.02..0.07
    Y_O2 = 0.15 + 0.05 * rng.random(shape)              # 0.15..0.20
    k_turb = 0.5 + 0.5 * rng.random(shape)
    eps_turb = 0.2 + 0.2 * rng.random(shape)
    tau_mix = 0.05 + 0.05 * rng.random(shape)
    omega_O2 = np.full(shape, 1.0e30)                   # "infinite supply"
    omega_max_T = np.full(shape, 1.0e30)                # no Damköhler cap
    omega_out = np.zeros(shape)
    return dict(
        rho=rho, T_g=T_g, Y_fuel=Y_fuel, Y_O2=Y_O2,
        k_turb=k_turb, eps_turb=eps_turb, tau_mix=tau_mix,
        omega_O2=omega_O2, omega_max_T=omega_max_T,
        chi_rad=0.34, cp_g=1100.0, dt=0.005, n_substeps=1,
        omega_out=omega_out,
    )


def _copy_state(state):
    """Deep-copy ndarray fields so a closure doesn't pollute a peer call."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


# ── Test 1: registry contents and availability ─────────────────────────────

def test_registry_advertises_current_closures():
    """Phase 15-0 framework + 15C closure: four closures registered.
    Update this test in the same commit as any future closure addition
    so the registry contract stays explicit (Rule #18 contract-test
    principle)."""
    assert chemistry_closures.available() == (
        "ebu_bootstrap", "edc", "level_set_fsd", "pasr",
    )


def test_unknown_closure_name_raises_with_actionable_message():
    """The error must list available choices so a typo is fixed in one
    iteration, not after a debugging spelunk."""
    state = _make_state()
    with pytest.raises(ValueError) as exc:
        chemistry_closures.run("not-a-closure", **state)
    msg = str(exc.value)
    assert "not-a-closure" in msg
    assert "edc" in msg and "pasr" in msg and "ebu_bootstrap" in msg


# ── Test 2: dispatch correctness — each closure routes to its kernel ───────

@pytest.mark.parametrize(
    "closure_name, kernel_attr",
    [
        ("edc",           "step_chemistry_ode_edc"),
        ("ebu_bootstrap", "step_chemistry_ode"),
        ("pasr",          "step_chemistry_ode_pasr"),
    ],
)
def test_dispatch_routes_to_expected_kernel(closure_name, kernel_attr):
    """Running via the registry and via direct kernel call produce the
    same updated state.  Confirms the registry doesn't introduce a
    subtle wrapper bug (e.g., swapped arguments).

    Strategy: call the kernel directly, save outputs; reset inputs;
    call via registry; compare bit-exactly.
    """
    state_a = _make_state(seed=42)
    state_b = _copy_state(state_a)

    # Direct kernel call path A — use kernel re-export on combustion_3d
    # to match what the existing tests do.
    kernel = getattr(combustion_3d, kernel_attr)
    if closure_name == "edc":
        kernel(
            state_a["rho"], state_a["T_g"], state_a["Y_fuel"], state_a["Y_O2"],
            state_a["k_turb"], state_a["eps_turb"], state_a["chi_rad"],
            state_a["cp_g"], state_a["dt"], state_a["n_substeps"],
            state_a["omega_out"],
        )
    elif closure_name == "ebu_bootstrap":
        kernel(
            state_a["rho"], state_a["T_g"], state_a["Y_fuel"], state_a["Y_O2"],
            state_a["tau_mix"], state_a["omega_O2"], state_a["omega_max_T"],
            state_a["chi_rad"], state_a["cp_g"], state_a["dt"],
            state_a["n_substeps"], state_a["omega_out"],
        )
    else:  # pasr
        kernel(
            state_a["rho"], state_a["T_g"], state_a["Y_fuel"], state_a["Y_O2"],
            state_a["tau_mix"], state_a["chi_rad"], state_a["cp_g"],
            state_a["dt"], state_a["n_substeps"], state_a["omega_out"],
        )

    # Registry path B
    chemistry_closures.run(closure_name, **state_b)

    # Identical post-state
    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(state_a[field], state_b[field]), (
            f"registry dispatch diverged from direct kernel call for {closure_name}.{field}"
        )


# ── Test 3: closures silently ignore extra kwargs they don't use ───────────

@pytest.mark.parametrize("closure_name", ["edc", "ebu_bootstrap", "pasr"])
def test_closures_accept_extra_kwargs_silently(closure_name):
    """The main loop passes one kwarg bag containing everything; each
    closure picks what it needs.  Extras must NOT raise."""
    state = _make_state(seed=99)
    state["unused_extra_field"] = "anything-here"
    state["another_unused"] = np.zeros((1,))
    # Should not raise.
    chemistry_closures.run(closure_name, **state)


# ── Test 4: bit-exact determinism (Rule #17, kernel level) ─────────────────

@pytest.mark.parametrize("closure_name", ["edc", "ebu_bootstrap", "pasr"])
def test_closure_is_bit_exact_under_repeat(closure_name):
    """Per Rule #17 and Rule #18: each closure must produce bit-exact
    identical output on identical input at the production thread count.

    Detects non-deterministic reductions BEFORE they contaminate sweep
    results.  Cf. Phase 14ak silent ignition bug — kernel-level test
    catches this in seconds, not in an 80-minute sweep."""
    state_1 = _make_state(seed=7)
    state_2 = _copy_state(state_1)

    chemistry_closures.run(closure_name, **state_1)
    chemistry_closures.run(closure_name, **state_2)

    for field in ("T_g", "Y_fuel", "Y_O2", "omega_out"):
        assert np.array_equal(state_1[field], state_2[field]), (
            f"{closure_name} non-deterministic on field {field}"
        )


# ── Test 5: backward-compat re-exports on combustion_3d ────────────────────

def test_combustion_3d_reexports_constants():
    """tests/outdoor/test_3d_components.py imports these from combustion_3d
    directly.  The Phase 15-0 refactor must preserve those import sites."""
    assert combustion_3d.S_STOICH == 1.3
    assert combustion_3d.Y_O2_AIR == 0.232
    assert combustion_3d.A_COMB == 1.0e9
    assert combustion_3d.E_COMB == 84_000.0
    assert combustion_3d._R_GAS == 8.314
    assert combustion_3d.C_EBU == 1.0
    assert combustion_3d.HOC_J == 17_000_000.0


def test_combustion_3d_reexports_moved_kernels():
    """Existing chemistry-validation tests import these from combustion_3d.
    They must remain available as attributes on that module (same Python
    object as in chemistry_closures, not a wrapper)."""
    assert combustion_3d.step_chemistry_ode_edc is chemistry_closures.edc.step_chemistry_ode_edc
    assert combustion_3d.step_chemistry_ode    is chemistry_closures.ebu_bootstrap.step_chemistry_ode
    assert combustion_3d.step_chemistry_ode_pasr is chemistry_closures.pasr.step_chemistry_ode_pasr


def test_combustion_3d_retains_shared_utility_kernels():
    """step_combustion, step_o2_supply_rate, apply_t_g_pin are NOT
    closure-specific — they remain in combustion_3d."""
    assert hasattr(combustion_3d, "step_combustion")
    assert hasattr(combustion_3d, "step_o2_supply_rate")
    assert hasattr(combustion_3d, "apply_t_g_pin")
    assert combustion_3d.T_FLAME_PIN == 1100.0
