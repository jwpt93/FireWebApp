from __future__ import annotations

import logging

from ..config.schemas import FuelConfig, Thresholds
from .properties import clamp01, safe_exp


_LOG = logging.getLogger(__name__)
_TINY = 1.0e-12


# ── Mass-source resolution helpers ───────────────────────────────────────────
# Determine how M1 is interpreted (fuel-fraction vs moisture) and compute
# the remaining fuel mass [kg/m²] used as the kinetic driving quantity.

def _kinetics_mode_token(fuel_cfg: FuelConfig) -> str:
    return str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()


def normalize_pyrolysis_mass_source(source: str | None) -> str:
    token = str(source or "legacy_M1").strip().lower().replace("-", "_")
    if token in {"legacy_m1", "legacym1", "legacy", "m1"}:
        return "legacy_M1"
    if token in {"fuel_state", "fuelstate", "fuel_remaining", "fuel"}:
        return "fuel_state"
    return "legacy_M1"


def resolve_pyrolysis_mass_source(fuel_cfg: FuelConfig) -> str:
    raw = getattr(fuel_cfg, "pyrolysis_mass_source", "legacy_M1")
    return normalize_pyrolysis_mass_source(str(raw))


def moisture_factor(M1: float, fuel_cfg: FuelConfig) -> float:
    """Moisture suppression factor [-]."""

    m1 = max(M1, 0.0)
    f = safe_exp(-fuel_cfg.alpha_moist * m1)
    return clamp01(f)


def _interp_schedule(schedule: list[tuple[float, float]], t: float, hold_last: bool = True) -> float:
    if not schedule:
        return 0.0
    schedule = sorted(schedule, key=lambda x: x[0])
    t0, v0 = schedule[0]
    if t <= t0:
        return float(v0)
    for i in range(1, len(schedule)):
        ti, vi = schedule[i]
        if t <= ti:
            t_prev, v_prev = schedule[i - 1]
            if ti == t_prev:
                return float(vi)
            frac = (t - t_prev) / (ti - t_prev)
            return float(v_prev + frac * (vi - v_prev))
    return float(schedule[-1][1] if hold_last else 0.0)


def _arrhenius_branch(T1: float, A: float, E: float, fuel_cfg: FuelConfig) -> float:
    if T1 <= 1.0:
        return 0.0
    arr = safe_exp(-E / (fuel_cfg.R * T1))
    return float(max(A, 0.0) * arr)


def _mass_action_rate(k_1_s: float, m_kg_m2: float, n_order: float) -> float:
    m = max(float(m_kg_m2), 0.0)
    n = max(float(n_order), _TINY)
    if m <= 0.0 or k_1_s <= 0.0:
        return 0.0
    return float(max(k_1_s, 0.0) * (m ** n))


def _resolve_a_basis(fuel_cfg: FuelConfig) -> str:
    """Resolve preferred Arrhenius prefactor basis with legacy fallback."""

    basis = str(getattr(fuel_cfg, "A_basis", "") or "").strip().lower()
    legacy = str(getattr(fuel_cfg, "pyrolysis_rate_basis", "") or "").strip().lower()
    valid = {"mass", "flux"}

    # Backward-compatibility: legacy field edits should still work when A_basis remains default.
    if basis in valid and legacy in valid and basis != legacy and basis == "mass":
        return legacy
    if basis in valid:
        return basis
    if legacy in valid:
        return legacy
    raise ValueError(f"Unknown A basis: A_basis={basis!r}, pyrolysis_rate_basis={legacy!r}")


def _resolve_m1_represents(fuel_cfg: FuelConfig) -> str:
    mode = str(getattr(fuel_cfg, "M1_represents", "fraction") or "fraction").strip().lower()
    if mode not in {"kg_m2", "fraction"}:
        raise ValueError(f"Unknown M1_represents: {mode!r}")
    return mode


def _resolve_m_tot_kg_m2(fuel_cfg: FuelConfig) -> float:
    m_tot_cfg = getattr(fuel_cfg, "m_fuel_total_kg_m2", None)
    if m_tot_cfg is not None:
        return max(float(m_tot_cfg), 0.0)

    rho = getattr(fuel_cfg, "rho", None)
    if rho is None:
        rho = getattr(fuel_cfg, "density_kg_m3", None)
    if rho is None:
        rho = getattr(fuel_cfg, "rho_solid", None)

    thickness_m = getattr(fuel_cfg, "thickness_m", None)
    if thickness_m is None:
        thickness_m = getattr(fuel_cfg, "regression_L0_m", None)

    if rho is not None and thickness_m is not None and float(rho) > 0.0 and float(thickness_m) > 0.0:
        return float(rho) * float(thickness_m)

    return max(float(getattr(fuel_cfg, "m_fuel_kg_m2", 0.0)), 0.0)


def resolve_total_fuel_mass_kg_m2(fuel_cfg: FuelConfig) -> float:
    return _resolve_m_tot_kg_m2(fuel_cfg)


def _to_remaining_mass_kg_m2(M1: float, fuel_cfg: FuelConfig) -> tuple[float, float, str]:
    m1_mode = _resolve_m1_represents(fuel_cfg)
    m_tot = _resolve_m_tot_kg_m2(fuel_cfg)

    if m1_mode == "kg_m2":
        m_remain = max(float(M1), 0.0)
    elif m1_mode == "fraction":
        m_remain = max(float(M1), 0.0) * max(float(m_tot), 0.0)
    else:
        raise ValueError(f"Unknown M1_represents: {m1_mode!r}")

    return float(m_remain), float(m_tot), m1_mode


# ── Single-pool kinetics (arrhenius / sigmoid / two_step) ────────────────────
# compute_m_dot_kinetics:  m_dot = A*exp(-E/RT) * m_remain  (or variants)
# compute_pyrolysis_kinetics_terms: diagnostics wrapper that returns breakdown

def compute_pyrolysis_kinetics_terms(
    T1: float,
    M1: float,
    fuel_cfg: FuelConfig,
    m_remain_kg_m2: float | None = None,
) -> dict[str, float]:
    """Return kinetic-only pyrolysis terms using the same internal formulas."""

    if m_remain_kg_m2 is not None:
        m_remain = max(float(m_remain_kg_m2), 0.0)
    else:
        m_remain, _m_tot, _m1_mode = _to_remaining_mass_kg_m2(M1, fuel_cfg)
    m_dot_kin, diag = compute_m_dot_kinetics(T1, m_remain, fuel_cfg)
    gate_factor = float(diag.get("S", 1.0))
    kinetics_gate_active = 1 if "S" in diag else 0
    return {
        "m_remaining_kg_m2": float(max(m_remain, 0.0)),
        "mdot_kin_kg_m2_s": float(max(m_dot_kin, 0.0)),
        "kinetics_gate_active": float(kinetics_gate_active),
        "gate_factor": float(gate_factor),
    }


def _sequential_yield_value(value: float, name: str, fuel_cfg: FuelConfig) -> float:
    y = float(value)
    if bool(getattr(fuel_cfg, "seq_clamp_yields", True)):
        return float(clamp01(y))
    if not (0.0 <= y <= 1.0):
        raise ValueError(f"two_step_sequential yield out of bounds for {name}: {y}")
    return y


# ── Two-step sequential kinetics ─────────────────────────────────────────────
# Staged condensed mass pools: m1 → (volatiles + m2) → (volatiles + residue).
# Each pool has its own Arrhenius rate.  Requires fuel_state tracking (M1/M2/M3
# integrated in ODE state vector).

def compute_two_step_sequential_rates(
    T1: float,
    m1_kg_m2: float,
    m2_kg_m2: float,
    fuel_cfg: FuelConfig,
    T_pool2: float | None = None,
) -> dict[str, float]:
    """Compute true sequential two-step kinetic rates on staged condensed-mass pools.

    States:
      m1: stage-1 reactive condensed mass [kg/m^2]
      m2: stage-2 reactive condensed mass [kg/m^2]
    Optional extension (Phase 2d): a competing stage-2 secondary charring sink
    can be enabled via ``seq_secondary_char_enable`` with branch parameters
    ``A3_py``/``E3_py``.

    Returns continuous-time rates and a conservation residual:
      d(m1+m2+mr)/dt + mdot_vol = 0
    """

    if _kinetics_mode_token(fuel_cfg) != "two_step_sequential":
        raise ValueError("compute_two_step_sequential_rates requires kinetics_mode='two_step_sequential'")

    for attr in ("A1_py", "E1_py", "A2_py", "E2_py"):
        if getattr(fuel_cfg, attr, None) is None:
            raise ValueError(f"two_step_sequential requires explicit branch parameter {attr}")

    m1 = max(float(m1_kg_m2), 0.0)
    m2 = max(float(m2_kg_m2), 0.0)
    A1 = float(fuel_cfg.A1_py)
    E1 = float(fuel_cfg.E1_py)
    A2 = float(fuel_cfg.A2_py)
    E2 = float(fuel_cfg.E2_py)

    k1 = _arrhenius_branch(float(T1), A1, E1, fuel_cfg)
    _T2_eff = float(T_pool2) if T_pool2 is not None else float(T1)
    k2 = _arrhenius_branch(_T2_eff, A2, E2, fuel_cfg)
    r1 = max(k1, 0.0) * m1
    r2 = max(k2, 0.0) * m2
    secondary_char_enable = bool(getattr(fuel_cfg, "seq_secondary_char_enable", False))
    if secondary_char_enable:
        if getattr(fuel_cfg, "A3_py", None) is None or getattr(fuel_cfg, "E3_py", None) is None:
            raise ValueError("two_step_sequential secondary charring requires A3_py and E3_py")
        k3 = _arrhenius_branch(float(T1), float(fuel_cfg.A3_py), float(fuel_cfg.E3_py), fuel_cfg)
        r3_char = max(k3, 0.0) * m2
    else:
        k3 = 0.0
        r3_char = 0.0

    y1 = _sequential_yield_value(float(getattr(fuel_cfg, "seq_y1_vol", 0.0)), "seq_y1_vol", fuel_cfg)
    y2 = _sequential_yield_value(float(getattr(fuel_cfg, "seq_y2_vol", 0.7)), "seq_y2_vol", fuel_cfg)
    f12_to_m2 = _sequential_yield_value(float(getattr(fuel_cfg, "seq_f12_to_m2", 1.0)), "seq_f12_to_m2", fuel_cfg)

    r1_nonvolatile = (1.0 - y1) * r1
    r1_to_m2 = f12_to_m2 * r1_nonvolatile
    r1_to_residue = (1.0 - f12_to_m2) * r1_nonvolatile

    dm1_dt = -r1
    dm2_dt = r1_to_m2 - r2 - r3_char
    dmr_dt = r1_to_residue + (1.0 - y2) * r2 + r3_char
    mdot_vol = y1 * r1 + y2 * r2
    mass_balance_residual = (dm1_dt + dm2_dt + dmr_dt) + mdot_vol

    return {
        "k1_1_s": float(max(k1, 0.0)),
        "k2_1_s": float(max(k2, 0.0)),
        "r1_kg_m2_s": float(max(r1, 0.0)),
        "r2_kg_m2_s": float(max(r2, 0.0)),
        "k3_char_1_s": float(max(k3, 0.0)),
        "r3_char_kg_m2_s": float(max(r3_char, 0.0)),
        "y1_vol": float(y1),
        "y2_vol": float(y2),
        "f12_to_m2": float(f12_to_m2),
        "secondary_char_enable": 1.0 if secondary_char_enable else 0.0,
        "r1_to_m2_kg_m2_s": float(max(r1_to_m2, 0.0)),
        "r1_to_residue_kg_m2_s": float(max(r1_to_residue, 0.0)),
        "mdot_vol_kg_m2_s": float(max(mdot_vol, 0.0)),
        "dm1_dt_kg_m2_s": float(dm1_dt),
        "dm2_dt_kg_m2_s": float(dm2_dt),
        "dmr_dt_kg_m2_s": float(dmr_dt),
        "mass_balance_residual_kg_m2_s": float(mass_balance_residual),
        "m1_kg_m2": float(m1),
        "m2_kg_m2": float(m2),
    }


# ── Semi-global product-yield staged kinetics ─────────────────────────────────
# Three condensed pools (gas/intermediate/char) with fractional yields per
# stage.  Designed for materials like particle board where charring strongly
# redistributes mass between volatile and residue fractions.

def compute_semi_global_seq_yield_rates(
    T1: float,
    m1_kg_m2: float,
    m2_kg_m2: float,
    fuel_cfg: FuelConfig,
) -> dict[str, float]:
    """Compute a semi-global sequential product-yield model on staged condensed pools.

    Stage 1 (virgin): m1 --r1--> gas + intermediate + residue
    Stage 2 (intermediate): m2 --r2--> gas + residue
    Optional secondary char sink (Phase 2d carryover): m2 --r3--> residue only
    """

    if _kinetics_mode_token(fuel_cfg) != "semi_global_seq_yield":
        raise ValueError(
            "compute_semi_global_seq_yield_rates requires kinetics_mode='semi_global_seq_yield'"
        )

    for attr in ("A1_py", "E1_py", "A2_py", "E2_py"):
        if getattr(fuel_cfg, attr, None) is None:
            raise ValueError(f"semi_global_seq_yield requires explicit branch parameter {attr}")

    m1 = max(float(m1_kg_m2), 0.0)
    m2 = max(float(m2_kg_m2), 0.0)
    A1 = float(fuel_cfg.A1_py)
    E1 = float(fuel_cfg.E1_py)
    A2 = float(fuel_cfg.A2_py)
    E2 = float(fuel_cfg.E2_py)

    k1 = _arrhenius_branch(float(T1), A1, E1, fuel_cfg)
    k2 = _arrhenius_branch(float(T1), A2, E2, fuel_cfg)
    n1 = max(float(getattr(fuel_cfg, "sg_n1", 1.0)), _TINY)
    n2 = max(float(getattr(fuel_cfg, "sg_n2", 1.0)), _TINY)
    r1 = _mass_action_rate(k1, m1, n1)
    r2 = _mass_action_rate(k2, m2, n2)

    secondary_char_enable = bool(getattr(fuel_cfg, "seq_secondary_char_enable", False))
    if secondary_char_enable:
        if getattr(fuel_cfg, "A3_py", None) is None or getattr(fuel_cfg, "E3_py", None) is None:
            raise ValueError("semi_global_seq_yield secondary charring requires A3_py and E3_py")
        k3 = _arrhenius_branch(float(T1), float(fuel_cfg.A3_py), float(fuel_cfg.E3_py), fuel_cfg)
        n3 = max(float(getattr(fuel_cfg, "sg_n3", 1.0)), _TINY)
        r3_char = _mass_action_rate(k3, m2, n3)
    else:
        k3 = 0.0
        n3 = max(float(getattr(fuel_cfg, "sg_n3", 1.0)), _TINY)
        r3_char = 0.0

    clamp_sg = bool(getattr(fuel_cfg, "sg_clamp_yields", True))

    def _sg_yield_value(value: float, name: str) -> float:
        y = float(value)
        if clamp_sg:
            return float(clamp01(y))
        if not (0.0 <= y <= 1.0):
            raise ValueError(f"semi_global_seq_yield yield out of bounds for {name}: {y}")
        return y

    y_g1 = _sg_yield_value(float(getattr(fuel_cfg, "sg_y_g1", 0.15)), "sg_y_g1")
    y_i1 = _sg_yield_value(float(getattr(fuel_cfg, "sg_y_i1", 0.55)), "sg_y_i1")
    y_c1 = _sg_yield_value(float(getattr(fuel_cfg, "sg_y_c1", 0.30)), "sg_y_c1")
    y_g2 = _sg_yield_value(float(getattr(fuel_cfg, "sg_y_g2", 0.70)), "sg_y_g2")
    y_c2 = _sg_yield_value(float(getattr(fuel_cfg, "sg_y_c2", 0.30)), "sg_y_c2")

    dm1_dt = -r1
    dm2_dt = y_i1 * r1 - r2 - r3_char
    dmr_dt = y_c1 * r1 + y_c2 * r2 + r3_char
    mdot_vol = y_g1 * r1 + y_g2 * r2
    mass_balance_residual = (dm1_dt + dm2_dt + dmr_dt) + mdot_vol

    return {
        "k1_1_s": float(max(k1, 0.0)),
        "k2_1_s": float(max(k2, 0.0)),
        "k3_char_1_s": float(max(k3, 0.0)),
        "n1": float(n1),
        "n2": float(n2),
        "n3": float(n3),
        "r1_kg_m2_s": float(max(r1, 0.0)),
        "r2_kg_m2_s": float(max(r2, 0.0)),
        "r3_char_kg_m2_s": float(max(r3_char, 0.0)),
        "sg_y_g1": float(y_g1),
        "sg_y_i1": float(y_i1),
        "sg_y_c1": float(y_c1),
        "sg_y_g2": float(y_g2),
        "sg_y_c2": float(y_c2),
        "secondary_char_enable": 1.0 if secondary_char_enable else 0.0,
        "mdot_vol_kg_m2_s": float(max(mdot_vol, 0.0)),
        "dm1_dt_kg_m2_s": float(dm1_dt),
        "dm2_dt_kg_m2_s": float(dm2_dt),
        "dmr_dt_kg_m2_s": float(dmr_dt),
        "mass_balance_residual_kg_m2_s": float(mass_balance_residual),
        "m1_kg_m2": float(m1),
        "m2_kg_m2": float(m2),
    }


# ── Main single-pool kinetics dispatcher ─────────────────────────────────────

def compute_m_dot_kinetics(
    T1: float,
    m_remain_kg_m2: float,
    fuel_cfg: FuelConfig,
) -> tuple[float, dict[str, float]]:
    """Compute kinetic pyrolysis rate [kg/m^2/s] and diagnostics.

    `m_remain_kg_m2` must be the remaining areal mass basis passed by the caller.
    Supports:
    - arrhenius: A_py/E_py
    - sigmoid: Arrhenius multiplied by logistic turn-on
    - two_step: branch sum (A1/E1 + A2/E2), defaulting missing branch params to A_py/E_py
    """

    mode = _kinetics_mode_token(fuel_cfg)
    basis = _resolve_a_basis(fuel_cfg)
    m_remain = max(float(m_remain_kg_m2), 0.0)
    diag: dict[str, float] = {"m_remain_kg_m2": float(m_remain)}

    if mode in {"two_step_sequential", "semi_global_seq_yield"}:
        raise ValueError(
            f"compute_m_dot_kinetics does not support kinetics_mode={mode!r} with a single "
            "remaining-mass pool; use the staged-mass fuel_state recurrence helpers."
        )

    if mode == "two_step":
        A1 = float(fuel_cfg.A1_py) if fuel_cfg.A1_py is not None else float(fuel_cfg.A_py)
        E1 = float(fuel_cfg.E1_py) if fuel_cfg.E1_py is not None else float(fuel_cfg.E_py)
        A2 = float(fuel_cfg.A2_py) if fuel_cfg.A2_py is not None else float(fuel_cfg.A_py)
        E2 = float(fuel_cfg.E2_py) if fuel_cfg.E2_py is not None else float(fuel_cfg.E_py)
        k1 = _arrhenius_branch(T1, A1, E1, fuel_cfg)
        k2 = _arrhenius_branch(T1, A2, E2, fuel_cfg)
        k_eff = k1 + k2
        diag["k_branch1"] = float(max(k1, 0.0))
        diag["k_branch2"] = float(max(k2, 0.0))
    else:
        k_arr = _arrhenius_branch(T1, float(fuel_cfg.A_py), float(fuel_cfg.E_py), fuel_cfg)
        diag["k_arrhenius"] = float(max(k_arr, 0.0))
        if mode == "sigmoid":
            T0 = float(getattr(fuel_cfg, "sigmoid_T0_K", 650.0))
            dT = max(float(getattr(fuel_cfg, "sigmoid_dT_K", 25.0)), 1.0e-6)
            S = 1.0 / (1.0 + safe_exp(-(T1 - T0) / dT))
            k_eff = S * k_arr
            diag["S"] = float(clamp01(S))
        else:
            k_eff = k_arr

    if basis == "mass":
        m_dot = max(k_eff, 0.0) * m_remain
    elif basis == "flux":
        m_dot = max(k_eff, 0.0)
        if m_remain <= 0.0:
            m_dot = 0.0
    else:
        raise ValueError(f"Unknown A basis: {basis!r}")

    m_dot = float(max(m_dot, 0.0))
    diag["A_basis"] = 0.0 if basis == "mass" else 1.0
    diag["m_dot_kin"] = m_dot
    diag["implied_k_1_s"] = float(m_dot / max(m_remain, _TINY)) if basis == "mass" else float("nan")
    return m_dot, diag


# ── Public entry points ───────────────────────────────────────────────────────
# pyrolysis_flux: called by the ODE RHS at every time step.
# pyrolysis_margin: returns how far m_dot is from the kinetic maximum (for
#   diagnostics / rate-cap blending).

def pyrolysis_flux(
    T1: float,
    M1: float,
    fuel_cfg: FuelConfig,
    t: float | None = None,
    m_remain_kg_m2: float | None = None,
) -> float:
    """Pyrolysis mass flux [kg/m^2/s]."""

    if fuel_cfg.pyrolysis_mode.lower().startswith("presc"):
        if t is None:
            return 0.0
        return max(_interp_schedule(fuel_cfg.m_py_schedule, t, hold_last=True), 0.0)

    mode_token = _kinetics_mode_token(fuel_cfg)
    if mode_token in {"two_step_sequential", "semi_global_seq_yield"}:
        raise ValueError(
            f"pyrolysis_flux() cannot evaluate kinetics_mode={mode_token!r} from a single M1/m_remaining state; "
            "use the fuel_state staged recurrence path in tests.fds_aligned.rom_adapter."
        )

    if m_remain_kg_m2 is not None:
        m_remain = max(float(m_remain_kg_m2), 0.0)
        m1_mode = "override_kg_m2"
        moist_mult = 1.0
    else:
        mass_source = resolve_pyrolysis_mass_source(fuel_cfg)
        if mass_source == "legacy_M1":
            # Original semantics: M1 is moisture content (not fuel fraction).
            # Fuel-mass basis is constant; moisture suppresses via moisture_factor.
            m_remain = _resolve_m_tot_kg_m2(fuel_cfg)
            m1_mode = "legacy_M1_constant"
            moist_mult = moisture_factor(M1, fuel_cfg)
        else:
            # fuel_state: M1 is explicit fuel-remaining fraction.
            m_remain, _m_tot, m1_mode = _to_remaining_mass_kg_m2(M1, fuel_cfg)
            moist_mult = 1.0
    m_dot_kin, diag = compute_m_dot_kinetics(T1, m_remain, fuel_cfg)
    m_dot_kin = float(max(m_dot_kin * moist_mult, 0.0))
    basis = _resolve_a_basis(fuel_cfg)
    _LOG.debug(
        "pyrolysis_flux: mass_source=%s A_basis=%s M1_raw=%.6g m_remain_kg_m2=%.6g moist_mult=%.6g implied_k_1_s=%.6g",
        m1_mode,
        basis,
        float(M1),
        m_remain,
        moist_mult,
        float(diag.get("implied_k_1_s", float("nan"))),
    )
    return m_dot_kin


def pyrolysis_margin(
    T1: float,
    M1: float,
    thresholds: Thresholds,
    fuel_cfg: FuelConfig,
    t: float | None = None,
) -> float:
    """Return pyrolysis margin [kg/m^2/s]."""

    m_py = pyrolysis_flux(T1, M1, fuel_cfg, t=t)
    return m_py - thresholds.m_py_crit
