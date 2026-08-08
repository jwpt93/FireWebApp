from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..config.schemas import FuelConfig, EnvConfig, SimConfig
from .moisture import evap_heat_sink, moisture_loss_rate
from .heat_transfer import heat_losses, open_face_loss_flux
from .pyrolysis import compute_m_dot_kinetics, compute_pyrolysis_kinetics_terms, compute_two_step_sequential_rates


# ── Forcing protocol ──────────────────────────────────────────────────────────
# Callable interface for time-varying boundary conditions passed to the ODE.

class Forcing(Protocol):
    def q_in(self, t: float) -> float: ...

    def rewet_rate(self, t: float) -> float: ...

    def M1_eq(self, t: float) -> float: ...


# ── Utility helpers ───────────────────────────────────────────────────────────

def _call_or_default(
    forcing: Mapping[str, Callable[[float], float]] | Forcing,
    name: str,
    t: float,
    default: float,
) -> float:
    if isinstance(forcing, Mapping):
        func = forcing.get(name, None)
        return float(func(t)) if func is not None else default
    func = getattr(forcing, name, None)
    if func is None:
        return default
    return float(func(t))


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _interp_schedule_hold_last(schedule: list[tuple[float, float]], t: float) -> float:
    if not schedule:
        return 0.0
    s = [(float(tt), float(qq)) for tt, qq in schedule]
    s.sort(key=lambda p: p[0])
    times = np.asarray([p[0] for p in s], dtype=float)
    vals = np.asarray([p[1] for p in s], dtype=float)
    if times.size == 0:
        return 0.0
    if times.size == 1:
        return float(vals[0])
    t_clamped = float(np.clip(float(t), float(times[0]), float(times[-1])))
    return float(np.interp(t_clamped, times, vals))


# ── Incident flux evaluation ──────────────────────────────────────────────────

def eval_q_in_incident_W_m2(t: float, cfg: Any) -> tuple[float, str]:
    """Evaluate applied incident heat flux and source label used by the RHS."""

    schedule = _cfg_get(cfg, "q_in_schedule", None)
    if schedule is None:
        schedule = []
    try:
        schedule_len = len(schedule)
    except Exception:
        schedule_len = 0
    if schedule_len > 0:
        q_sched = _interp_schedule_hold_last(list(schedule), float(t))
        return max(float(q_sched), 0.0), "schedule"

    q_const = _cfg_get(cfg, "q_in_constant_W_m2", None)
    if q_const is not None:
        try:
            q_const_f = float(q_const)
        except Exception:
            q_const_f = 0.0
        if q_const_f > 0.0:
            return q_const_f, "constant"
    return 0.0, "none"


# ── Thermal property helpers ──────────────────────────────────────────────────
# k(T) functions and char/virgin blending used by the multi-node heat balance.

def _front_limit_enabled(fuel_cfg: FuelConfig) -> bool:
    if bool(getattr(fuel_cfg, "front_limit_enable", False)):
        return True
    mode = str(getattr(fuel_cfg, "front_model_mode", "") or "").strip().lower()
    return mode in {"on", "enabled", "enable", "true", "1", "yes", "y"}


def _k_pmma_piecewise(T: float) -> float:
    """PMMA conductivity law [W/m/K] for T in K."""
    if T < 378.0:
        k = 0.45 - 0.00038 * T
    else:
        k = 0.27 - 0.00024 * T
    return max(k, 0.01)


def _k12_dynamic(T1: float, T2: float, fuel_cfg: FuelConfig) -> float:
    if fuel_cfg.k_temp_mode != "pmma_piecewise":
        return fuel_cfg.K12
    if fuel_cfg.K12_ref is None or fuel_cfg.k_ref is None or fuel_cfg.k_ref <= 0.0:
        return fuel_cfg.K12
    Tm = 0.5 * (T1 + T2)
    kT = _k_pmma_piecewise(Tm)
    ratio = kT / max(fuel_cfg.k_ref, 1.0e-9)
    return fuel_cfg.K12_ref * ratio


def _blend_prop(alpha: float, virgin: float, char: float | None) -> float:
    """Linear blend between virgin and char property values. alpha=0 → virgin, alpha=1 → char."""
    if char is None:
        return virgin
    a = float(np.clip(alpha, 0.0, 1.0))
    return virgin * (1.0 - a) + char * a


def _char_kinetics_rate(T_K: float, fuel_cfg: FuelConfig) -> float:
    """First-order char formation rate constant at temperature T_K [1/s].
    Uses A_char/E_char if set, otherwise falls back to primary pyrolysis A/E."""
    A = fuel_cfg.A_char if fuel_cfg.A_char is not None else (
        fuel_cfg.A1_py if fuel_cfg.A1_py is not None else fuel_cfg.A_py)
    E = fuel_cfg.E_char if fuel_cfg.E_char is not None else (
        fuel_cfg.E1_py if fuel_cfg.E1_py is not None else fuel_cfg.E_py)
    return float(A) * np.exp(-float(E) / (fuel_cfg.R * max(float(T_K), 1.0)))


def _smoothstep01(x: float) -> float:
    x_c = float(np.clip(x, 0.0, 1.0))
    return x_c * x_c * (3.0 - 2.0 * x_c)


def _softmin(a: float, b: float, beta: float) -> float:
    if beta <= 0.0:
        return min(a, b)
    xa = -beta * a
    xb = -beta * b
    m = max(xa, xb)
    return -(m + np.log(np.exp(xa - m) + np.exp(xb - m))) / beta


def _rho_for_front_limit(L: float, fuel_cfg: FuelConfig) -> float:
    if fuel_cfg.rho_solid is not None and fuel_cfg.rho_solid > 0.0:
        return float(fuel_cfg.rho_solid)
    if L > 1.0e-12 and fuel_cfg.m_fuel_kg_m2 > 0.0:
        return float(fuel_cfg.m_fuel_kg_m2 / L)
    return 0.0


# ── Stefan char-front regression (rate-cap mechanism) ─────────────────────────
# Limits pyrolysis rate to what the thermal front can supply, using the Stefan
# condition: alpha = k_char*(T1-T_py)/(rho*dH_py).  Active when
# front_limit_enable=True.  The k_crack_frac field enhances k_char_eff to
# account for crack-network heat transport through deep char layers.

def compute_front_limit_terms(
    T1: float,
    M1: float,
    delta_py: float,
    m_c: float,
    L: float,
    fuel_cfg: FuelConfig,
    alpha_bar: float = 0.0,
) -> Dict[str, float]:
    """Compute PMMA front-limit/regression terms for diagnostics and RHS.

    k_char_eff = k_char * (1 + k_crack_frac * delta_capped / L)
    d_delta_dt = alpha / delta_capped   where delta_capped = min(delta_py, regression_delta_cap_m)

    Crack enhancement and Stefan denominator both use delta_capped (= min(delta_py, delta_cap)).
    When delta_py < delta_cap: normal 1/delta_py decline. When delta_py ≥ delta_cap: quasi-steady
    rate = k_char*(1+k_crack_frac*delta_cap/L)*(T1-T_py)/(dH_py*delta_cap) — models char spalling.
    Literature: Moghtaderi 2006, Di Blasi 2009 — char layer crumbles/spalls, bounding thermal path.
    Default delta_cap = L (no cap) → backward compatible.

    alpha_bar: retained in signature for backward compatibility; not used internally.
    """

    _ = M1  # Reserved for compatibility; PMMA front-limit kinetics now uses areal mass basis.

    delta_min = max(float(fuel_cfg.regression_delta_min_m), 1.0e-12)
    delta_eff = max(float(delta_py), delta_min)
    _delta_cap_raw = None   # initialise before branching so line 245 check is always defined

    # Dynamic Stefan alpha: alpha = k_char * max(T_surf - T_py, 0) / (rho * dH_py)
    # Activated when regression_T_py_K is set; otherwise use fixed regression_alpha.
    _T_py_K = getattr(fuel_cfg, "regression_T_py_K", None)
    if _T_py_K is not None and fuel_cfg.k_char is not None:
        # Crack-enhanced effective k_char: grows with char-front penetration depth.
        # k_char_eff = k_char * (1 + k_crack_frac * delta_py / L)
        # Near zero at peak (delta_py ≈ delta_min); meaningful only when char front
        # has penetrated, i.e., during the sustained-burn phase.
        # Physical basis: Di Blasi 2002, Moghtaderi 2006, Shi & Chew 2023 — thermal
        # stress cracking proportional to char-layer depth, not surface char fraction.
        _k_crack = float(getattr(fuel_cfg, "k_crack_frac", 0.0))
        _L_ref = max(float(L), delta_min)
        # Char spalling cap: bound the effective thermal path length used in the Stefan rate.
        # The ODE state delta_py advances freely (tracks cumulative front depth for energy
        # accounting); only the rate formula uses the capped value.
        # Literature basis: Moghtaderi 2006, Di Blasi 2009 — char spalls/crumbles periodically,
        # limiting effective thermal resistance and creating quasi-steady pyrolysis.
        _delta_cap_raw = getattr(fuel_cfg, "regression_delta_cap_m", None)
        _delta_cap = float(_delta_cap_raw) if _delta_cap_raw is not None else float(L)
        _delta_capped = max(min(delta_eff, _delta_cap), delta_min)
        # Char fall-off: progressive reduction of effective delta_capped as pyrolysis front
        # approaches back face (Moghtaderi 2006; Di Blasi 2009).  As delta_py/L → 1, char
        # segments lose structural support and detach, exposing fresh pyrolyzing surface and
        # causing a transient Stefan rate increase (secondary HRRPUA peak).
        _spall_onset = float(getattr(fuel_cfg, "regression_spall_onset_frac", 0.0))
        _spall_red   = float(getattr(fuel_cfg, "regression_spall_reduction_frac", 0.0))
        if _spall_onset > 0.0 and _spall_red > 0.0:
            _depth_frac = float(delta_py) / max(float(L), delta_min, 1e-6)
            _f_spall = _smoothstep01((_depth_frac - _spall_onset) / max(1.0 - _spall_onset, 1e-6))
            _delta_capped = max(_delta_capped * (1.0 - _spall_red * _f_spall), float(delta_min))
        _k_c = float(fuel_cfg.k_char) * (1.0 + _k_crack * _delta_capped / _L_ref)
        _H_py = float(getattr(fuel_cfg, "dH_py", 0.0))
        _rho_d = max(float(fuel_cfg.rho_solid) if fuel_cfg.rho_solid is not None else 0.0, 1.0)
        if _H_py > 0.0:
            alpha = _k_c * max(float(T1) - float(_T_py_K), 0.0) / (_rho_d * _H_py)
        else:
            alpha = float(fuel_cfg.regression_alpha)
        d_delta_dt = alpha / _delta_capped
    else:
        alpha = float(fuel_cfg.regression_alpha)
        d_delta_dt = alpha / delta_eff

    L_eff = max(float(L), delta_min)
    rho = _rho_for_front_limit(L_eff, fuel_cfg)
    m_dot_cap = rho * max(d_delta_dt, 0.0)
    m_avail = rho * min(max(float(delta_py), 0.0), L_eff) - max(float(m_c), 0.0)

    _km_flt = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
    if _km_flt in {"two_step_sequential", "semi_global_seq_yield"}:
        # Staged kinetics: Stefan cap alone drives the front; staged pools computed in ODE/post-proc.
        m_dot_kin = m_dot_cap
        kin_diag = {}
    else:
        m_dot_kin, kin_diag = compute_m_dot_kinetics(T1, max(m_avail, 0.0), fuel_cfg)

    beta = float(fuel_cfg.softmin_beta)
    m_dot_pp = max(_softmin(max(m_dot_kin, 0.0), max(m_dot_cap, 0.0), beta), 0.0)
    if m_avail <= 0.0:
        m_dot_pp = 0.0

    start = float(fuel_cfg.handoff_start_frac)
    end = float(fuel_cfg.handoff_end_frac)
    if end < start:
        start, end = end, start
    # Suppress handoff-to-kinetics while delta_cap is active and front has reached the cap.
    # In quasi-steady spalling mode the ODE state delta_py keeps advancing even though the
    # effective thermal path is bounded.  If blend were allowed to fire at delta_py/L > 0.9,
    # Arrhenius kinetics at T1≈800 K would produce an enormous runaway spike.
    _in_spalling_mode = (_delta_cap_raw is not None and float(delta_py) > _delta_cap)
    # Suppress handoff entirely when Stefan dynamic formula is active (regression_T_py_K set).
    # The Stefan cap IS the physically correct rate for charring materials all the way to the
    # back face.  Blending in unconstrained Arrhenius kinetics (T1 >> T_py → huge rate)
    # as delta_py/L → 1 creates an unphysical HRRPUA spike near end-of-burn.
    _stefan_active = (_T_py_K is not None and fuel_cfg.k_char is not None)
    if _in_spalling_mode or _stefan_active:
        blend = 0.0
    elif end <= start + 1.0e-12:
        blend = 1.0 if (delta_py / L_eff) >= end else 0.0
    else:
        blend = _smoothstep01(((delta_py / L_eff) - start) / (end - start))
    m_dot_used = (1.0 - blend) * m_dot_pp + blend * max(m_dot_kin, 0.0)
    if m_avail <= 0.0:
        m_dot_used = 0.0
    # Panel fully penetrated — freeze ODE state and stop Stefan front pyrolysis.
    # The blend < 1.0 guard was wrong: when delta_py first reaches L, blend == exactly 1.0
    # (handoff_end_frac=1.0 default) so the guard was False, letting m_dot_kin (unconstrained
    # Arrhenius at T1~800K) run free on any m_avail residual → late-time HRRPUA spike.
    if float(delta_py) >= float(L_eff):
        m_dot_used = 0.0
        d_delta_dt = 0.0

    terms: Dict[str, float] = {
        "m_dot_kin": float(max(m_dot_kin, 0.0)),
        "m_dot_cap": float(max(m_dot_cap, 0.0)),
        "m_dot_pp": float(max(m_dot_pp, 0.0)),
        "m_dot_used": float(max(m_dot_used, 0.0)),
        "m_avail": float(m_avail),
        "d_delta_dt": float(d_delta_dt),
        "delta_ratio": float(delta_py / L_eff),
        "handoff_blend": float(np.clip(blend, 0.0, 1.0)),
        "rho_solid": float(rho),
    }
    for key, val in kin_diag.items():
        terms[key] = float(val)
    return terms


def compute_front_limited_terms(
    T1: float,
    M1: float,
    delta_py: float,
    m_c: float,
    L: float,
    fuel_cfg: FuelConfig,
) -> Dict[str, float]:
    """Compatibility wrapper for legacy test/API name."""

    terms = compute_front_limit_terms(T1, M1, delta_py, m_c, L, fuel_cfg, alpha_bar=0.0)
    compat = dict(terms)
    compat["blend"] = float(terms["handoff_blend"])
    # Expose the blended output (m_dot_used) as m_dot_pp so callers see the effective rate
    # after the handoff blend, not the raw Stefan-cap softmin value.
    compat["m_dot_pp"] = float(terms["m_dot_used"])
    compat["m_dot_pp_raw"] = float(terms["m_dot_pp"])
    compat["m_dot_pp"] = float(terms["m_dot_used"])
    return compat


# ── Heat-of-gasification (Tewarson) rate cap ──────────────────────────────────
# Alternative rate cap: m_dot ≤ max(q_in - q_crit, 0) / L_eff.
# Disabled in current wood decks (replaced by Stefan); kept for PMMA.

def compute_hog_terms(
    T1: float,
    M1: float,
    q_in: float,
    fuel_cfg: FuelConfig,
    t: float = 0.0,
    alpha_bar: float = 0.0,          # mean char fraction [0, 1]; reduces HoG cap via (1 - alpha_bar)
) -> Dict[str, float]:
    """Heat-of-gasification and thermal-penetration rate caps.

    q_in is the INCIDENT (external) heat flux [W/m²] — the Tewarson HoG formulation uses
    the external cone flux, not the net flux. L_eff implicitly accounts for steady-state
    surface losses; q_crit is the critical flux below which ignition does not occur.

    HoG:      m_dot = max(q_in - q_crit, 0) / L_eff * max(1 - alpha_bar, 0)
    ThermPen: m_dot = rho * sqrt(alpha_th / (4 * max(t, t_floor)))
    Both compete against kinetics via hard min.

    Applicability domain (HoG):
    - Thermally THICK specimens required: Fourier number Fo = α·t / L² < ~0.10 at test end.
    - For wood (α ≈ 1.7×10⁻⁷ m²/s): need thickness > ~26 mm at 400 s test duration.
    - Thin panels (<~25 mm): thermal wave reaches back face → double-hump HRRPUA shape
      not reproducible by this model. Validated for 38 mm SPF lumber (FSRI Wood Stud).

    TODO (thin-panel extension): Add two-face burning with back-face boundary condition,
    or a burnthrough transition when the char front approaches the back face. This would
    extend the model to thin panels (Basswood 19 mm, Aalto Spruce 20 mm, etc.).
    """
    use_hog = bool(getattr(fuel_cfg, "hog_enable", False))
    use_tp  = bool(getattr(fuel_cfg, "therm_pen_enable", False))

    m_dot_cap_hog = 0.0
    if use_hog:
        L_eff  = max(float(getattr(fuel_cfg, "hog_L_eff_J_kg",  None) or 5.0e6), 1.0)
        q_crit = max(float(getattr(fuel_cfg, "hog_q_crit_W_m2", None) or 0.0),  0.0)
        m_dot_cap_hog = max(float(q_in) - q_crit, 0.0) / L_eff
        m_dot_cap_hog *= max(1.0 - float(alpha_bar), 0.0)

    m_dot_cap_tp = float("inf")
    if use_tp:
        k_s   = max(float(getattr(fuel_cfg, "k",       None) or 0.1),   1e-9)
        rho_s = max(float(getattr(fuel_cfg, "density", None) or 400.0),  1.0)
        cp_s  = max(float(getattr(fuel_cfg, "cp",      None) or 1000.0), 1.0)
        alpha_th = k_s / (rho_s * cp_s)
        dmin  = max(float(getattr(fuel_cfg, "regression_delta_min_m", 1.0e-4)), 1e-9)
        t_floor = dmin ** 2 / max(alpha_th, 1e-12)
        t_eff = max(float(t), t_floor)
        m_dot_cap_tp = rho_s * math.sqrt(alpha_th / (4.0 * t_eff))

    if use_hog and use_tp:
        m_dot_cap = min(m_dot_cap_hog, m_dot_cap_tp)
    elif use_hog:
        m_dot_cap = m_dot_cap_hog
    elif use_tp:
        m_dot_cap = m_dot_cap_tp
    else:
        m_dot_cap = float("inf")

    rho_s = max(float(getattr(fuel_cfg, "density", None) or 400.0), 1.0)
    L0    = max(float(getattr(fuel_cfg, "regression_L0_m", 0.038)), 1e-9)
    m_fuel_total = (
        float(fuel_cfg.m_fuel_total_kg_m2)
        if getattr(fuel_cfg, "m_fuel_total_kg_m2", None) is not None
        else rho_s * L0
    )
    m_remain = max(float(M1), 0.0) * m_fuel_total
    _km_hog = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
    if _km_hog in {"two_step_sequential", "semi_global_seq_yield"}:
        # HoG cap alone applies; staged ODE rates computed separately in fuel_rhs/post-processing.
        m_dot_kin = float("inf")
    else:
        m_dot_kin, _ = compute_m_dot_kinetics(float(T1), m_remain, fuel_cfg)

    if m_dot_cap == float("inf"):
        m_dot_used = max(m_dot_kin, 0.0)
    else:
        m_dot_used = min(max(m_dot_kin, 0.0), max(m_dot_cap, 0.0))
    if m_remain <= 0.0:
        m_dot_used = 0.0
    elif m_remain < 0.05 * m_fuel_total:
        # Soft ramp: scale m_dot linearly to zero over the last 5% of fuel so the
        # end-of-burn transition is gradual rather than a hard 1-step cliff.
        m_dot_used *= m_remain / (0.05 * m_fuel_total)

    return {
        "m_dot_kin":     float(max(m_dot_kin, 0.0)),
        "m_dot_cap_hog": float(max(m_dot_cap_hog, 0.0)),
        "m_dot_cap_tp":  float(m_dot_cap_tp if m_dot_cap_tp < float("inf") else 0.0),
        "m_dot_cap":     float(max(m_dot_cap, 0.0) if m_dot_cap < float("inf") else 0.0),
        "m_dot_used":    float(max(m_dot_used, 0.0)),
        "m_fuel_total":  float(m_fuel_total),
    }


# ── Pyrolysis attribution diagnostics ────────────────────────────────────────
# Decomposes total m_dot into kinetic / Stefan-cap / HoG-cap contributions
# for each output timestep.  Used by rom_adapter for diagnostic CSV export.

def compute_pyrolysis_attribution_terms(
    T1: float,
    M1: float,
    fuel_cfg: FuelConfig,
    delta_py: float | None = None,
    m_c: float | None = None,
    L: float | None = None,
    m_fuel_remaining_kg_m2: float | None = None,
) -> Dict[str, float]:
    """Return attribution terms for mdot used by the RHS without changing physics."""

    use_front_limit = _front_limit_enabled(fuel_cfg) and (delta_py is not None and m_c is not None and L is not None)
    if use_front_limit:
        terms = compute_front_limit_terms(float(T1), float(M1), float(delta_py), float(m_c), float(L), fuel_cfg)
        mdot_kin = float(max(terms.get("m_dot_kin", 0.0), 0.0))
        mdot_cap = float(max(terms.get("m_dot_cap", 0.0), 0.0))
        mdot_limit = float(max(terms.get("m_dot_pp", 0.0), 0.0))
        mdot_final = float(max(terms.get("m_dot_used", 0.0), 0.0))
        m_remaining = float(max(terms.get("m_avail", 0.0), 0.0))
        cap_active = 1 if (mdot_limit + 1.0e-12 < mdot_kin) else 0
        limiter_active = 1 if (mdot_final + 1.0e-12 < mdot_kin) else 0
        gate_factor = float(terms.get("S", 1.0))
        kinetics_gate_active = 1 if "S" in terms else 0
        return {
            "m_remaining_kg_m2": m_remaining,
            "mdot_kin_kg_m2_s": mdot_kin,
            "mdot_cap_kg_m2_s": mdot_cap,
            "mdot_limit_kg_m2_s": mdot_limit,
            "mdot_final_kg_m2_s": mdot_final,
            "limiter_active": float(limiter_active),
            "cap_active": float(cap_active),
            "kinetics_gate_active": float(kinetics_gate_active),
            "gate_factor": gate_factor,
        }

    _km_attr = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
    _staged_attr = _km_attr in {"two_step_sequential", "semi_global_seq_yield"}
    if _staged_attr:
        # Staged kinetics: attribution not available via single-pool helpers; return zeros.
        kin = {"m_remaining_kg_m2": 0.0, "mdot_kin_kg_m2_s": 0.0, "kinetics_gate_active": 0.0, "gate_factor": 1.0}
    elif m_fuel_remaining_kg_m2 is not None and np.isfinite(float(m_fuel_remaining_kg_m2)):
        m_remaining = float(max(m_fuel_remaining_kg_m2, 0.0))
        mdot_kin, kin_diag = compute_m_dot_kinetics(float(T1), m_remaining, fuel_cfg)
        gate_factor = float(kin_diag.get("S", 1.0))
        kinetics_gate_active = 1 if "S" in kin_diag else 0
        kin = {
            "m_remaining_kg_m2": m_remaining,
            "mdot_kin_kg_m2_s": float(max(mdot_kin, 0.0)),
            "kinetics_gate_active": float(kinetics_gate_active),
            "gate_factor": gate_factor,
        }
    else:
        kin = compute_pyrolysis_kinetics_terms(float(T1), float(M1), fuel_cfg)
    return {
        "m_remaining_kg_m2": float(kin["m_remaining_kg_m2"]),
        "mdot_kin_kg_m2_s": float(kin["mdot_kin_kg_m2_s"]),
        "mdot_cap_kg_m2_s": float("nan"),
        "mdot_limit_kg_m2_s": float("nan"),
        "mdot_final_kg_m2_s": float(kin["mdot_kin_kg_m2_s"]),
        "limiter_active": 0.0,
        "cap_active": 0.0,
        "kinetics_gate_active": float(kin["kinetics_gate_active"]),
        "gate_factor": float(kin["gate_factor"]),
    }


# ── Surface energy balance ────────────────────────────────────────────────────
# Computes q_conv and q_rad loss at the exposed surface for each output step.
# Used for diagnostic export; not called during ODE integration.

def compute_surface_heat_terms(
    T1: float,
    q_in: float,
    fuel_cfg: FuelConfig,
    env_cfg: EnvConfig,
    sim_cfg: SimConfig | None = None,
) -> Dict[str, float]:
    """Surface boundary heat terms with the exact branches/signs used by the RHS."""

    T_sur = env_cfg.T_sur if env_cfg.T_sur is not None else env_cfg.Tamb
    losses = heat_losses(
        T_s=float(T1),
        T_inf=float(env_cfg.Tamb),
        T_sur=float(T_sur),
        eps=float(fuel_cfg.eps),
        L=float(fuel_cfg.L_m),
        u_inf=float(fuel_cfg.u_inf_m_s),
        mode=str(fuel_cfg.convection_mode),
        orientation=str(fuel_cfg.orientation),
        C_h_conv=float(fuel_cfg.C_h_conv),
        C_eps=float(fuel_cfg.C_eps),
    )

    q_conv_loss = float(losses["q_conv"]) + float(fuel_cfg.h_amb) * (float(T1) - float(env_cfg.Tamb))
    q_rad_loss = float(losses["q_rad"])
    q_mode = str(getattr(sim_cfg, "q_in_mode", "incident") if sim_cfg is not None else "incident").strip().lower()
    if q_mode not in {"incident", "net"}:
        q_mode = "incident"

    q_in_term = float(q_in)
    if q_mode == "net":
        q_conv_term = 0.0
        q_rad_term = 0.0
    else:
        q_conv_term = -q_conv_loss
        q_rad_term = -q_rad_loss

    q_net_surface = q_in_term + q_conv_term + q_rad_term
    return {
        "q_in": q_in_term,
        "q_conv": q_conv_term,
        "q_rad": q_rad_term,
        "q_net_surface": q_net_surface,
        "q_conv_loss": q_conv_loss,
        "q_rad_loss": q_rad_loss,
        "h_conv_total": float(losses["h_conv"]) + float(fuel_cfg.h_amb),
        "eps_eff": float(losses["eps_eff"]),
        "q_mode": q_mode,
    }


# ── ODE right-hand side ───────────────────────────────────────────────────────
# fuel_rhs() is the core ODE evaluated by scipy solve_ivp.
#
# State vector layout (N thermal nodes):
#   [T1..TN, M1..MN, alpha1..alphaN, delta_py, m_c, L]  (N=2: 6 states; N=3: 12 states)
#   - Ti     : node temperatures [K]
#   - Mi     : per-node moisture/fuel state (semantics depend on pyrolysis_mass_source)
#   - alphai : per-node char fraction (only active when char_state_mode='kinetic')
#   - delta_py : char-front depth [m]  (front_limit_enable=True)
#   - m_c    : cumulative char mass [kg/m²] (front_limit path)
#   - L      : current slab thickness [m]   (front_limit path)
#
# Branching:
#   - 2-node vs 3/N-node: different conductance expressions
#   - front_limit: Stefan rate-cap replaces kinetic m_dot when active
#   - kinetic char: alpha_i driven by Arrhenius char kinetics
#   - sequential/semi-global kinetics: staged mass pools (see pyrolysis.py)

def fuel_rhs(
    t: float,
    y: NDArray[np.float64],
    fuel_cfg: FuelConfig,
    env_cfg: EnvConfig,
    forcing: Mapping[str, Callable[[float], float]] | Forcing,
    sim_cfg: SimConfig | None = None,
    prop_interp: dict | None = None,
) -> NDArray[np.float64]:
    """Lumped fuel ODE RHS (2-node default, optional 3-node thermal prototype).

    2-node states:
      y = [T1, T2, M1]                              (3 states)
      y = [T1, T2, M1, delta_py, m_c, L]            (6 states, front-limit path)
    3-node thermal prototype states:
      y = [T1, T2, T3, M1]                          (4 states, staggered prop_interp)
      y = [T1, T2, T3, M1, a1, a2, a3]              (7 states, char_state_mode=kinetic)
      y = [T1, T2, T3, M1, d, m_c, L]               (7 states, front-limit only)
      y = [T1, T2, T3, M1, a1, a2, a3, d, m_c, L]  (10 states, kinetic + front-limit)
    """

    thermal_order = int(getattr(fuel_cfg, "thermal_model_order", 2) or 2)
    if thermal_order < 2:
        raise ValueError(f"thermal_model_order must be >= 2 (got {thermal_order})")

    _N = thermal_order  # number of thermal nodes
    use_kinetic_char = (
        _N >= 3
        and str(getattr(fuel_cfg, "char_state_mode", "none")).lower() == "kinetic"
        and len(y) >= 3 * _N  # per-node M_i states present
    )
    use_3node_front_limit = _N >= 3 and _front_limit_enabled(fuel_cfg)

    if _N >= 3:
        if len(y) < _N + 1:
            raise ValueError(f"{_N}-node thermal model requires y of length >= {_N + 1}")
        T_nodes = [float(y[i]) for i in range(_N)]
        T1 = T_nodes[0]  # surface node alias used by surface/evap/HoG functions
        T2 = T_nodes[1]
        T3 = T_nodes[2] if _N >= 3 else None
        if use_kinetic_char:
            # Per-node fuel fractions: y[N..2N-1]; α: y[2N..3N-1]
            M_nodes = [max(float(y[_N + i]), 0.0) for i in range(_N)]
            M1 = M_nodes[0]  # surface-node fuel alias for evap/legacy functions
        else:
            M_nodes = None
            M1 = float(y[_N])  # single global M1
        use_front_limit = False  # 2-node front-limit flag unused for N-node path
        if use_3node_front_limit:
            _fl_offset = 3 * _N if use_kinetic_char else _N + 1
            delta_py = float(y[_fl_offset]) if len(y) > _fl_offset else 0.0
            m_c      = float(y[_fl_offset + 1]) if len(y) > _fl_offset + 1 else 0.0
            L        = float(y[_fl_offset + 2]) if len(y) > _fl_offset + 2 else 1.0
        else:
            _fl_offset = 3 * _N if use_kinetic_char else _N + 1
            delta_py = 0.0
            m_c = 0.0
            L = 1.0
        _km_rhs = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
        _use_staged = _km_rhs in {"two_step_sequential", "semi_global_seq_yield"}
        _staged_offset = _fl_offset + (3 if use_3node_front_limit else 0)
        if _use_staged:
            m1_global = float(y[_staged_offset])     if len(y) > _staged_offset     else 0.0
            m2_global = float(y[_staged_offset + 1]) if len(y) > _staged_offset + 1 else 0.0
        else:
            m1_global = 0.0
            m2_global = 0.0
    else:
        T1, T2, M1 = float(y[0]), float(y[1]), float(y[2])
        T3 = None
        T_nodes = [T1, T2]
        use_front_limit = _front_limit_enabled(fuel_cfg) or len(y) >= 6
    if thermal_order == 2 and len(y) >= 6:
        delta_py = float(y[3])
        m_c = float(y[4])
        L = float(y[5])
    elif thermal_order == 2:
        delta_py = float(getattr(fuel_cfg, "delta_py0_m", 0.0))
        m_c = float(getattr(fuel_cfg, "m_char0_kg_m2", 0.0))
        L = float(max(getattr(fuel_cfg, "regression_L0_m", 1.0), 1.0e-9))

    q_in_cfg = forcing.get("q_in_cfg", None) if isinstance(forcing, Mapping) else getattr(forcing, "q_in_cfg", None)
    if q_in_cfg is not None:
        q_in, _q_in_source = eval_q_in_incident_W_m2(t, q_in_cfg)
    else:
        q_in = _call_or_default(forcing, "q_in", t, 0.0)
        _q_in_source = "callable"
    # Flame radiation feedback: augments incident flux when flame coupling is active.
    # Supports float (constant) or callable q_fb(t) [W/m²] via _call_or_default.
    # Default 0.0 — zero change to existing behavior when flame coupling is disabled.
    q_fb = _call_or_default(forcing, "q_fb", t, 0.0)
    q_in_cone = q_in  # cone-only flux; preserved for HoG ceiling in split-flux mode
    q_in = q_in + q_fb  # total flux used for surface heat terms and temperature ODE
    # Char oxidation ODE thermal feedback: sustains node-1 temperature after volatile
    # depletion so char glow is self-sustaining (Frandsen 1991 smoldering criterion).
    # Populated by runner.py when char_ox_enable=true; zero otherwise (no-op).
    q_in = q_in + _call_or_default(forcing, "q_char_ox", t, 0.0)
    rewet_rate = _call_or_default(forcing, "rewet_rate", t, 0.0)
    M1_eq = _call_or_default(forcing, "M1_eq", t, M1)

    surf_terms = compute_surface_heat_terms(T1=T1, q_in=q_in, fuel_cfg=fuel_cfg, env_cfg=env_cfg, sim_cfg=sim_cfg)
    q_evap = evap_heat_sink(T1, M1, fuel_cfg)

    # Resolve per-node char fractions: ODE state (kinetic) > prop_interp (staggered) > zero
    if _N >= 3:
        if use_kinetic_char:
            # α_i at indices 2N..3N-1 (M_i now occupies N..2N-1)
            alpha = [float(np.clip(y[2 * _N + i], 0.0, 1.0)) for i in range(_N)]
        elif prop_interp is not None:
            _t_arr = prop_interp["t"]
            alpha = []
            for _i in range(_N):
                _key = f"alpha{_i + 1}"
                if _key in prop_interp:
                    alpha.append(float(np.interp(t, _t_arr, prop_interp[_key])))
                else:
                    alpha.append(0.0)
        else:
            alpha = [0.0] * _N
        # Keep legacy aliases for compatibility with code below
        alpha1 = alpha[0]
        alpha2 = alpha[1]
        alpha3 = alpha[2]
    else:
        alpha1 = alpha2 = alpha3 = 0.0
        alpha = [alpha1, alpha2]

    # Per-node sequential burnthrough (enabled only when alpha_burnthrough < 1.0).
    # Mechanism: when node i chars through (alpha_i >= threshold), it is excluded from
    # the alpha_bar denominator used in the HoG cap, so (1-alpha_bar) resets toward 1.
    # This allows the next node's fresh fuel to fire at full HoG rate — producing a
    # second (or third) HRRPUA peak without any explicit q_in routing.
    # Temperatures are still governed by K_char-blended conduction between nodes.
    _alpha_bt = float(getattr(fuel_cfg, "alpha_burnthrough", 1.1))
    _bt_active = _alpha_bt < 1.0 and use_kinetic_char and _N >= 3
    node_bt = [_bt_active and alpha[i] >= _alpha_bt for i in range(_N)]
    node1_bt = node_bt[0] if _N >= 1 else False
    node2_bt = node_bt[1] if _N >= 2 else False
    node3_bt = node_bt[2] if _N >= 3 else False

    # Effective per-node capacitances (blend virgin → char)
    C1_eff = _blend_prop(alpha1, fuel_cfg.C1, fuel_cfg.C1_char)
    C2_eff = _blend_prop(alpha2, fuel_cfg.C2, fuel_cfg.C2_char)

    # Effective inter-node conductances (blend based on mean char fraction of adjacent nodes)
    K12_base = _k12_dynamic(T1, T2, fuel_cfg)
    K12_eff = _blend_prop(0.5 * (alpha1 + alpha2), K12_base, fuel_cfg.K12_char)

    # N-node: collect all C_eff and K_eff as arrays (used by the N-node ODE loop below)
    if _N >= 3:
        _C_nodes = [float(getattr(fuel_cfg, f"C{i + 1}", None) or (fuel_cfg.C1 if i == 0 else fuel_cfg.C2))
                    for i in range(_N)]
        _C_char_nodes = [getattr(fuel_cfg, f"C{i + 1}_char", None) for i in range(_N)]
        _K_links_base = []
        for _i in range(_N - 1):
            if _i == 0:
                _K_links_base.append(_k12_dynamic(T_nodes[0], T_nodes[1], fuel_cfg))
            else:
                _attr = f"K{_i + 1}{_i + 2}"
                _K_links_base.append(float(getattr(fuel_cfg, _attr, None) or 0.0))
        _K_char_links = [getattr(fuel_cfg, f"K{i + 1}{i + 2}_char", None) for i in range(_N - 1)]
        # Crack-enhanced ODE inter-node conductances: when enabled, interfaces within the char region
        # (delta_py has passed them) use k_char_eff = k_char*(1+k_crack*delta_py/L) — same crack factor
        # as compute_front_limit_terms(). Breaks the circular dependency: low T_N→low alpha→no enhancement.
        # Gate: k_crack_ode_enable=False by default (protects Wood Stud and other calibrations).
        if bool(getattr(fuel_cfg, "k_crack_ode_enable", False)) and use_3node_front_limit:
            _k_crack_ode = float(getattr(fuel_cfg, "k_crack_frac", 0.0) or 0.0)
            _L_ode = max(float(L), 1e-9)
            _crack_mult = 1.0 + _k_crack_ode * max(float(delta_py), 0.0) / _L_ode
            _delta_frac = max(float(delta_py), 0.0) / _L_ode
            _cum_frac = 0.0
            _K_char_links_eff = []
            for _li in range(_N - 1):
                _k_base = _K_char_links[_li]
                _cum_frac += float(getattr(fuel_cfg, f"node{_li + 1}_frac", 1.0 / _N))
                if _k_base is not None and _k_base > 0.0 and _delta_frac >= _cum_frac:
                    _K_char_links_eff.append(_k_base * _crack_mult)
                else:
                    _K_char_links_eff.append(_k_base)
            _K_char_links = _K_char_links_eff
        _C_eff_N = [_blend_prop(alpha[i], _C_nodes[i], _C_char_nodes[i]) for i in range(_N)]
        _K_eff_N = [_blend_prop(0.5 * (alpha[i] + alpha[i + 1]), _K_links_base[i], _K_char_links[i])
                    for i in range(_N - 1)]

    # ── Bed collapse: scale inter-node conductances by 1/m_frac ──────────────
    # Physical: h(t) = h₀ × m_frac(t) → inter-node distance d12(t) = d12₀ × m_frac
    # → K12(t) = k/d12(t) = K12₀/m_frac. Heat capacity per unit area is unchanged.
    # m_frac ≡ M1 (fuel remaining fraction in fuel_state mode, range 0→1).
    if bool(getattr(fuel_cfg, "bed_collapse_enable", False)):
        _m_frac_coll = max(float(M1), 0.05)
        K12_eff = K12_eff / _m_frac_coll
        if _N >= 3:
            _K_eff_N = [k / _m_frac_coll for k in _K_eff_N]

    q_open = 0.0
    T_sur = env_cfg.T_sur if env_cfg.T_sur is not None else env_cfg.Tamb
    if str(getattr(fuel_cfg, "back_bc_mode", "adiabatic")).strip().lower() == "open":
        eps_open = fuel_cfg.eps if fuel_cfg.eps_open is None else fuel_cfg.eps_open
        # Use back-face temperature: last T_node for N≥3, T2 (back node) for 2-node
        _T_back_for_loss = T_nodes[_N - 1] if _N >= 3 else T2
        q_open = open_face_loss_flux(
            T2=_T_back_for_loss,
            h_open=fuel_cfg.h_open,
            eps_open=float(eps_open),
            T_inf=env_cfg.Tamb,
            T_sur=T_sur,
            sigma=env_cfg.sigma,
        )

    q_py = 0.0
    d_delta_dt = 0.0
    dm_c_dt = 0.0
    dL_dt = 0.0
    if use_front_limit or use_3node_front_limit:
        # Compute alpha_bar for crack-enhanced k_char_eff (Stefan path only)
        _stefan_alpha_bar = 0.0
        if use_3node_front_limit and use_kinetic_char and _N >= 3 and alpha is not None:
            _C_s = [float(getattr(fuel_cfg, f"C{i + 1}", None) or fuel_cfg.C1) for i in range(_N)]
            _C_s_tot = max(sum(_C_s), 1e-12)
            _stefan_alpha_bar = sum(_C_s[i] * alpha[i] for i in range(_N)) / _C_s_tot
        terms = compute_front_limit_terms(T1, M1, delta_py, m_c, L, fuel_cfg, alpha_bar=_stefan_alpha_bar)
        d_delta_dt = terms["d_delta_dt"]
        dm_c_dt = terms["m_dot_used"]
        q_py = max(float(getattr(fuel_cfg, "dH_py", 0.0)), 0.0) * max(dm_c_dt, 0.0)

    use_hog       = bool(getattr(fuel_cfg, "hog_enable",       False))
    use_therm_pen = bool(getattr(fuel_cfg, "therm_pen_enable", False))
    if (use_hog or use_therm_pen) and not (use_front_limit or use_3node_front_limit):
        if use_kinetic_char and _N >= 3:
            # Mass-fraction weighted alpha: frac_i = C_i / C_total (= node_i_frac for uniform rho/cp)
            _C_raw = [float(getattr(fuel_cfg, f"C{i + 1}", None) or (fuel_cfg.C1 if i == 0 else fuel_cfg.C2))
                      for i in range(_N)]
            # Soft burnthrough weight: node contribution to alpha_bar ramps linearly from 1→0
            # as alpha rises from (alpha_bt - 0.15) to alpha_bt.  Smooths HRRPUA spikes vs
            # the old hard-threshold approach (instantaneous exclusion at alpha_bt).
            # Fuel drain (_is_active_drain) is M-based only — no fuel abandoned after BT.
            _M_eps = 1e-9
            _M_soft_thr = 0.01
            _bt_ramp_hi = _alpha_bt
            _bt_ramp_lo = max(_bt_ramp_hi - 0.15, 0.0) if _bt_active else 0.0
            _bt_ramp_w = max(_bt_ramp_hi - _bt_ramp_lo, 0.01)
            def _soft_bt(a):
                if not _bt_active: return 1.0
                return max(0.0, min(1.0, (_bt_ramp_hi - a) / _bt_ramp_w))
            def _m_ramp(m):
                # Soft M-depletion ramp: weight 0→1 as M rises from 0 to _M_soft_thr.
                # Replaces hard boolean to avoid discrete alpha_bar jumps at node exhaustion.
                return min(1.0, m / _M_soft_thr) if m > 0.0 else 0.0
            _is_active_drain = [M_nodes[i] > _M_eps for i in range(_N)]
            _weights = [_C_raw[i] * _soft_bt(alpha[i]) * _m_ramp(M_nodes[i])
                        for i in range(_N)]
            _C_tot = max(sum(_weights), 1e-12)
            _alpha_bar = sum(_weights[i] * alpha[i] for i in range(_N)) / _C_tot
            # Effective M for kinetics: total fuel in non-exhausted nodes (includes BT'd)
            _M_active = sum(M_nodes[i] for i in range(_N) if _is_active_drain[i])
            # Sequential front: shallowest non-exhausted node drains its fuel
            _front_idx = next((i for i in range(_N) if _is_active_drain[i]), _N - 1)
        else:
            _alpha_bar = 0.0
            _M_active = M1
            _is_active_drain = None
            _front_idx = 0
        _q_for_hog = q_in_cone if (q_fb != 0.0 and bool(getattr(fuel_cfg, "flame_hog_split_flux", False))) else q_in
        hog_terms = compute_hog_terms(T1, _M_active, _q_for_hog, fuel_cfg, t=t, alpha_bar=_alpha_bar)
        dm_c_dt = hog_terms["m_dot_used"]
        q_py = max(float(getattr(fuel_cfg, "dH_py", 0.0)), 0.0) * max(dm_c_dt, 0.0)

    dT1 = (surf_terms["q_net_surface"] - q_evap - q_py + K12_eff * (T2 - T1)) / C1_eff
    if _N >= 3:
        # Validate that all inter-node conductances and capacitances are positive
        for _i in range(_N - 1):
            if _K_eff_N[_i] <= 0.0:
                _kname = f"K{_i + 1}{_i + 2}"
                _cname = f"C{_i + 2}"
                raise ValueError(
                    f"{_N}-node thermal model requires positive {_kname} and {_cname}. "
                    f"Check that geometry.node{_i + 1}_frac/node{_i + 2}_frac are set and non-zero."
                )
            if _C_eff_N[_i + 1] <= 0.0:
                raise ValueError(f"{_N}-node thermal model requires positive C{_i + 2}")
        if _C_eff_N[0] <= 0.0:
            raise ValueError(f"{_N}-node thermal model requires positive C1")
        _q_loss_back = float(getattr(fuel_cfg, "q_loss3", None) or fuel_cfg.q_loss2)
        _q_back_in = float(getattr(fuel_cfg, "back_face_q_in_W_m2", 0.0))
        # Build dT array for all N nodes using the pre-computed _C_eff_N / _K_eff_N arrays
        _dT = [0.0] * _N
        for _i in range(_N):
            _q_left  = _K_eff_N[_i - 1] * (T_nodes[_i - 1] - T_nodes[_i]) if _i > 0 else 0.0
            _q_right = _K_eff_N[_i]     * (T_nodes[_i + 1] - T_nodes[_i]) if _i < _N - 1 else 0.0
            if _i == 0:
                _src = surf_terms["q_net_surface"] - q_evap - q_py + _q_left + _q_right
            elif _i == _N - 1:
                _src = _q_left + _q_right - _q_loss_back - q_open + _q_back_in
            else:
                _src = _q_left + _q_right
            _dT[_i] = _src / _C_eff_N[_i]
        dT2 = _dT[1]  # keep alias for 2-node path below (not used in N-node return)
    else:
        dT2 = (-K12_eff * (T2 - T1) - fuel_cfg.q_loss2 - q_open) / C2_eff

    k_ev = moisture_loss_rate(T1, M1, fuel_cfg)
    if (use_hog or use_therm_pen) and not (use_front_limit or use_3node_front_limit):
        if use_kinetic_char and _N >= 3:
            _dM = [0.0] * _N
            _m_fuel_tot = max(hog_terms["m_fuel_total"], 1e-12)
            if _is_active_drain is not None and any(_is_active_drain):
                if _bt_active:
                    # Sequential front (BT enabled): shallowest non-exhausted node drains.
                    # BT'd nodes continue draining until fuel exhausted — no fuel abandoned.
                    _dM[_front_idx] = -dm_c_dt / _m_fuel_tot
                else:
                    # Proportional distribution (BT disabled): equivalent to single-M1 model.
                    # All non-exhausted nodes drain proportionally — backward compatible.
                    _M_sum = sum(M_nodes[i] for i in range(_N) if _is_active_drain[i])
                    if _M_sum > 1e-12:
                        for _i in range(_N):
                            if _is_active_drain[_i]:
                                _dM[_i] = (-dm_c_dt / _m_fuel_tot) * (M_nodes[_i] / _M_sum)
            dM1 = _dM[0]  # alias for compatibility
        else:
            # Single global M1 (non-kinetic-char N-node or non-HoG path)
            dM1 = -dm_c_dt / max(hog_terms["m_fuel_total"], 1e-12)
            _dM = None
    else:
        if use_kinetic_char and _N >= 3 and M_nodes is not None:
            if use_3node_front_limit and dm_c_dt >= 0.0:
                # Stefan front controls total rate.
                # Two distribution modes:
                #
                # front_limit_surface_only=False (default): proportional to all node fuel fractions.
                #   Appropriate for thick panels (e.g. Wood Stud 38mm) where the pyrolysis zone is
                #   genuinely a thin front — sub-front nodes are too cold for significant Arrhenius.
                #
                # front_limit_surface_only=True: Stefan cap on surface node only; Arrhenius for deeper.
                #   Appropriate for thin panels (e.g. Basswood 19mm) where sub-front nodes heat to
                #   pyrolysis temperatures during the burn — they contribute distributed Arrhenius
                #   pyrolysis simultaneously with the Stefan front advance. Surface node still needs
                #   the Stefan cap to prevent Arrhenius runaway at T1 ≈ 800-1000K.
                _rho_s = max(float(getattr(fuel_cfg, "density", None) or 400.0), 1.0)
                _L0 = max(float(getattr(fuel_cfg, "regression_L0_m", 0.038)), 1e-9)
                _m_fuel_tot = max(_rho_s * _L0, 1e-12)
                _surface_only = bool(getattr(fuel_cfg, "front_limit_surface_only", False))
                if _surface_only:
                    # Stefan cap on node 1 only; Arrhenius for nodes 2-N
                    _dM = [0.0] * _N
                    _dM[0] = -(dm_c_dt / _m_fuel_tot)
                    for _si in range(1, _N):
                        _dM[_si] = -_char_kinetics_rate(T_nodes[_si], fuel_cfg) * M_nodes[_si]
                else:
                    _m_sum = sum(M_nodes)
                    if _m_sum > 1e-12:
                        _dM = [-(dm_c_dt / _m_fuel_tot) * (M_nodes[i] / _m_sum) for i in range(_N)]
                    else:
                        _dM = [0.0] * _N
                dM1 = _dM[0]
            else:
                # No rate cap: drain each node via Arrhenius kinetics directly.
                _dM = [-_char_kinetics_rate(T_nodes[i], fuel_cfg) * M_nodes[i] for i in range(_N)]
                dM1 = _dM[0]
        else:
            dM1 = -k_ev * M1 + rewet_rate * (M1_eq - M1)
            _dM = None

    if use_front_limit:
        return np.array([dT1, dT2, dM1, d_delta_dt, dm_c_dt, dL_dt], dtype=float)
    if _N >= 3:
        if use_kinetic_char and _dM is not None:
            base = _dT + _dM  # per-node fuel derivatives (3N elements total with da_dt)
        else:
            base = _dT + [dM1]  # single M1
        if use_kinetic_char:
            da_dt = [_char_kinetics_rate(T_nodes[i], fuel_cfg) * (1.0 - alpha[i]) for i in range(_N)]
            # Freeze char kinetics for nodes that have burned through OR exhausted
            for _i in range(_N):
                if node_bt[_i] or (M_nodes is not None and M_nodes[_i] <= 1e-9):
                    da_dt[_i] = 0.0
            base += da_dt
        if use_3node_front_limit:
            base += [d_delta_dt, dm_c_dt, dL_dt]
        if _use_staged:
            _T_pool2 = (
                T_nodes[_N - 1]
                if getattr(fuel_cfg, "seq_pool2_use_back_node", False) and _N >= 3
                else None
            )
            _sr = compute_two_step_sequential_rates(T1, m1_global, m2_global, fuel_cfg, T_pool2=_T_pool2)
            base += [float(_sr["dm1_dt_kg_m2_s"]), float(_sr["dm2_dt_kg_m2_s"])]
        return np.array(base, dtype=float)
    return np.array([dT1, dT2, dM1], dtype=float)


# ── ODE integration entry point ───────────────────────────────────────────────
# integrate_fuel() is the public API: builds the initial state vector, calls
# solve_ivp, and returns the raw integration result.

def integrate_fuel(
    y0: NDArray[np.float64],
    t_span: Tuple[float, float],
    fuel_cfg: FuelConfig,
    env_cfg: EnvConfig,
    forcing: Mapping[str, Callable[[float], float]] | Forcing,
    sim_cfg: SimConfig,
    prop_interp: dict | None = None,
) -> "FuelIntegrationResult":
    """Integrate the fuel model.

    Returns time history sampled at internal solver steps.
    """

    thermal_order = int(getattr(fuel_cfg, "thermal_model_order", 2) or 2)
    if thermal_order < 2:
        raise ValueError(f"thermal_model_order must be >= 2 (got {thermal_order})")
    _N = thermal_order
    use_kinetic_char = (
        _N >= 3
        and str(getattr(fuel_cfg, "char_state_mode", "none")).lower() == "kinetic"
    )
    y0_use = np.asarray(y0, dtype=float)
    use_3node_front_limit = _N >= 3 and _front_limit_enabled(fuel_cfg)
    _use_staged_init = False  # set to True below if _N >= 3 and staged kinetics active
    if _N >= 3:
        if y0_use.size == 3:
            # Backward-compatible expansion from [T1, T2, M1] -> [T1, T2, ..., TN, M1].
            y0_use = np.concatenate([[y0_use[0]] + [y0_use[1]] * (_N - 1), [y0_use[2]]])
        elif y0_use.size == _N + 1:
            pass  # Already [T1, ..., TN, M1]
        if use_kinetic_char and y0_use.size == _N + 1:
            # Expand [T1..TN, M1] -> [T1..TN, M1..MN, alpha1=0..alphaN=0]
            # Distribute M1_init proportionally to each node's thermal mass fraction
            _C_raw = [float(getattr(fuel_cfg, f"C{i+1}", None) or fuel_cfg.C1) for i in range(_N)]
            _C_tot = max(sum(_C_raw), 1e-12)
            _fracs = [c / _C_tot for c in _C_raw]
            _M1_init = float(y0_use[_N])
            _M_init = [_M1_init * f for f in _fracs]
            y0_use = np.concatenate([y0_use[:_N], _M_init, [0.0] * _N])
        if use_3node_front_limit and y0_use.size in (_N + 1, 3 * _N):
            # Expand to include [delta_py, m_c, L] front-limit states
            L0 = float(max(getattr(fuel_cfg, "regression_L0_m", 0.038), 1.0e-9))
            delta0 = float(max(getattr(fuel_cfg, "delta_py0_m", 0.0), 0.0))
            m_c0 = float(max(getattr(fuel_cfg, "m_char0_kg_m2", 0.0), 0.0))
            y0_use = np.concatenate([y0_use, [delta0, m_c0, L0]])
        _km_init = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
        _use_staged_init = _km_init in {"two_step_sequential", "semi_global_seq_yield"}
        if _use_staged_init:
            _rho_s_init = max(float(getattr(fuel_cfg, "density", None) or 400.0), 1.0)
            _L0_init = max(float(getattr(fuel_cfg, "regression_L0_m", 0.038) or 0.038), 1e-9)
            _m0_total = float(
                getattr(fuel_cfg, "m_fuel_total_kg_m2", None)
                or (_rho_s_init * _L0_init)
            )
            _m1_init = max(float(getattr(fuel_cfg, "seq_m1_frac", 1.0) or 1.0) * _m0_total, 0.0)
            _m2_init = max(float(getattr(fuel_cfg, "seq_m2_frac0", 0.0) or 0.0) * _m0_total, 0.0)
            y0_use = np.concatenate([y0_use, [_m1_init, _m2_init]])
        if use_3node_front_limit:
            # Include _N+4 = [T1..TN, M1, delta, m_c, L] for front-limit without kinetic-char.
            valid_sizes = {_N + 1, _N + 4, 3 * _N, 3 * _N + 3}
        else:
            valid_sizes = {_N + 1, 3 * _N}
        if _use_staged_init:
            valid_sizes = {v + 2 for v in valid_sizes}
        if y0_use.size not in valid_sizes:
            raise ValueError(
                f"{_N}-node thermal model y0 size mismatch: got {y0_use.size}, "
                f"expected one of {sorted(valid_sizes)}"
            )
    elif y0_use.size == 3 and _front_limit_enabled(fuel_cfg):
        L0 = float(max(getattr(fuel_cfg, "regression_L0_m", 1.0), 1.0e-9))
        delta0 = float(max(getattr(fuel_cfg, "delta_py0_m", 0.0), 0.0))
        m_c0 = float(max(getattr(fuel_cfg, "m_char0_kg_m2", 0.0), 0.0))
        y0_use = np.array([y0_use[0], y0_use[1], y0_use[2], delta0, m_c0, L0], dtype=float)

    def rhs(t: float, y_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        return fuel_rhs(t, y_vec, fuel_cfg, env_cfg, forcing, sim_cfg=sim_cfg, prop_interp=prop_interp)

    sol = solve_ivp(
        rhs,
        t_span,
        y0_use,
        method=sim_cfg.method,
        rtol=sim_cfg.rtol,
        atol=sim_cfg.atol,
        max_step=sim_cfg.max_step if sim_cfg.max_step is not None else np.inf,
    )
    if sol.status < 0:
        raise RuntimeError("Integration failed")

    m_py_pp: NDArray[np.float64] | None = None
    m_dot_cap: NDArray[np.float64] | None = None
    m_dot_kin: NDArray[np.float64] | None = None
    handoff_blend: NDArray[np.float64] | None = None
    _use_char_ox: bool = False
    char_ox_hrrpua_pp: NDArray[np.float64] | None = None
    _fl_offset_3n = (3 * _N) if use_kinetic_char else (_N + 1)
    if thermal_order == 2 and sol.y.shape[0] >= 6:
        n = sol.t.size
        m_py_pp = np.zeros(n, dtype=float)
        m_dot_cap = np.zeros(n, dtype=float)
        m_dot_kin = np.zeros(n, dtype=float)
        handoff_blend = np.zeros(n, dtype=float)
        for i in range(n):
            terms = compute_front_limit_terms(
                T1=float(sol.y[0, i]),
                M1=float(sol.y[2, i]),
                delta_py=float(sol.y[3, i]),
                m_c=float(sol.y[4, i]),
                L=float(sol.y[5, i]),
                fuel_cfg=fuel_cfg,
            )
            m_py_pp[i] = float(terms["m_dot_used"])
            m_dot_cap[i] = float(terms["m_dot_cap"])
            m_dot_kin[i] = float(terms["m_dot_kin"])
            handoff_blend[i] = float(terms["handoff_blend"])
    elif use_3node_front_limit and sol.y.shape[0] > _fl_offset_3n + 2:
        n = sol.t.size
        m_py_pp = np.zeros(n, dtype=float)
        m_dot_cap = np.zeros(n, dtype=float)
        m_dot_kin = np.zeros(n, dtype=float)
        handoff_blend = np.zeros(n, dtype=float)
        _kc_pp = str(getattr(fuel_cfg, "char_state_mode", "")).strip().lower() == "kinetic"
        _C_pp3 = [float(getattr(fuel_cfg, f"C{j + 1}", None) or fuel_cfg.C1) for j in range(_N)]
        _C_pp3_tot = max(sum(_C_pp3), 1e-12)
        # Sub-front Arrhenius: add per-node Arrhenius contribution to m_py_pp for nodes 2..N.
        # Active when front_limit_surface_only=True — Stefan cap stays on surface (node 1) only;
        # deeper nodes pyrolyze via Arrhenius as the thermal wave heats them (distributed process).
        _subfront_arrh = bool(getattr(fuel_cfg, "front_limit_surface_only", False))
        _rho_sf = max(float(getattr(fuel_cfg, "density", None) or 400.0), 1.0)
        _L0_sf  = max(float(getattr(fuel_cfg, "regression_L0_m", 0.038)), 1e-9)
        _m_fuel_tot_sf = max(_rho_sf * _L0_sf, 1e-12)
        # Char oxidation — same physics as HoG branch; active when char_ox_enable=true
        _use_char_ox = bool(getattr(fuel_cfg, "char_ox_enable", False))
        _q_char_ref_cx   = float(getattr(fuel_cfg, "char_ox_q_ref_W_m2", None) or 150000.0)
        _q_stefan0_cx    = max(float(getattr(fuel_cfg, "char_ox_q_stefan0_W_m2", None) or 80000.0), 1.0)
        _char_yield_cx   = float(getattr(fuel_cfg, "char_ox_char_yield", 0.25))
        _char_hoc_cx     = max(float(getattr(fuel_cfg, "char_ox_char_hoc_J_kg", 32.7e6)), 1.0)
        _q_in_src_cx = None
        _char_pool_cx = 0.0
        if _use_char_ox:
            _q_in_cfg_pp = (
                forcing.get("q_in_cfg", None) if isinstance(forcing, Mapping)
                else getattr(forcing, "q_in_cfg", None)
            )
            _q_in_src_cx = _q_in_cfg_pp if _q_in_cfg_pp is not None else fuel_cfg
            char_ox_hrrpua_pp = np.zeros(n, dtype=float)
        for i in range(n):
            # Compute alpha_bar for crack-enhanced k_char in post-processing
            _ab_pp = 0.0
            if _kc_pp and sol.y.shape[0] >= 3 * _N:
                _alphas_pp = [float(np.clip(sol.y[2 * _N + j, i], 0.0, 1.0)) for j in range(_N)]
                _ab_pp = sum(_C_pp3[j] * _alphas_pp[j] for j in range(_N)) / _C_pp3_tot
            terms = compute_front_limit_terms(
                T1=float(sol.y[0, i]),
                M1=float(sol.y[3, i]),
                delta_py=float(sol.y[_fl_offset_3n, i]),
                m_c=float(sol.y[_fl_offset_3n + 1, i]),
                L=float(sol.y[_fl_offset_3n + 2, i]),
                fuel_cfg=fuel_cfg,
                alpha_bar=_ab_pp,
            )
            m_py_pp[i] = float(terms["m_dot_used"])
            if _subfront_arrh and sol.y.shape[0] >= 2 * _N:
                # Add sub-front Arrhenius contribution from nodes 2..N (below Stefan front).
                # Physical: cellulose pyrolysis at the sharp front (Stefan, node 1) + distributed
                # hemicellulose/extractive volatilization at T=400-550K in deeper virgin wood.
                # These are additive physical processes; Stefan does not cap sub-front pyrolysis.
                for _sf_i in range(1, _N):
                    _T_sf = float(sol.y[_sf_i, i])
                    _M_sf = max(float(sol.y[_N + _sf_i, i]), 0.0)
                    m_py_pp[i] += _char_kinetics_rate(_T_sf, fuel_cfg) * _M_sf * _m_fuel_tot_sf
            m_dot_cap[i] = float(terms["m_dot_cap"])
            m_dot_kin[i] = float(terms["m_dot_kin"])
            handoff_blend[i] = float(terms["handoff_blend"])
            if _use_char_ox and _q_in_src_cx is not None:
                _dt_i = float(sol.t[i] - sol.t[i - 1]) if i > 0 else 0.0
                if _dt_i > 0:
                    _char_pool_cx += _char_yield_cx * m_py_pp[i] * _dt_i
                    _m_py_s0_cx = float(getattr(fuel_cfg, "char_ox_m_py_stefan0_kg_m2_s", None) or 0.0)
                    if _m_py_s0_cx > 0.0:
                        # Pyrolysis-rate-based Stefan blow suppression: f_stefan → 1 as m_py → 0.
                        # Physical: volatile blow suppresses O2 to char surface proportional to m_py.
                        # When pyrolysis drops (end-of-burn), blow decreases → char oxidation surges.
                        _f_stefan_cx = max(1.0 - m_py_pp[i] / _m_py_s0_cx, 0.0)
                    else:
                        _q_in_i_cx, _ = eval_q_in_incident_W_m2(float(sol.t[i]), _q_in_src_cx)
                        _f_stefan_cx = max(1.0 - _q_in_i_cx / _q_stefan0_cx, 0.0)
                    _raw_cx = float(_ab_pp) * _q_char_ref_cx * _f_stefan_cx
                    _delta_demanded = _raw_cx * _dt_i / _char_hoc_cx
                    _delta_actual = min(_delta_demanded, _char_pool_cx)
                    _char_pool_cx = max(_char_pool_cx - _delta_actual, 0.0)
                    char_ox_hrrpua_pp[i] = _delta_actual * _char_hoc_cx / _dt_i
    elif bool(getattr(fuel_cfg, "hog_enable", False)) or bool(getattr(fuel_cfg, "therm_pen_enable", False)):
        _q_in_cfg_pp = (
            forcing.get("q_in_cfg", None) if isinstance(forcing, Mapping)
            else getattr(forcing, "q_in_cfg", None)
        )
        _q_in_src = _q_in_cfg_pp if _q_in_cfg_pp is not None else fuel_cfg
        _M1_idx = _N if _N >= 3 else 2
        n = sol.t.size
        m_py_pp   = np.zeros(n, dtype=float)
        m_dot_cap = np.zeros(n, dtype=float)
        m_dot_kin = np.zeros(n, dtype=float)
        handoff_blend = np.zeros(n, dtype=float)
        _use_kc_pp = str(getattr(fuel_cfg, "char_state_mode", "")).strip().lower() == "kinetic"
        # Collect virgin capacitances for alpha_bar weighting (all N nodes)
        _C_pp = [float(getattr(fuel_cfg, f"C{i + 1}", None) or (fuel_cfg.C1 if i == 0 else fuel_cfg.C2))
                 for i in range(_N)]
        _C_tot_pp = max(sum(_C_pp), 1e-12)
        # Burnthrough threshold for post-processing alpha_bar (matches fuel_rhs)
        _alpha_bt_pp = float(getattr(fuel_cfg, "alpha_burnthrough", 1.1))
        _bt_active_pp = _alpha_bt_pp < 1.0 and _use_kc_pp and _N >= 3
        # Per-node M states present when kinetic char is active (3N state vector)
        _per_node_M_pp = _use_kc_pp and _N >= 3 and sol.y.shape[0] >= 3 * _N
        _use_char_ox = bool(getattr(fuel_cfg, "char_ox_enable", False))
        _q_char_ref  = float(getattr(fuel_cfg, "char_ox_q_ref_W_m2", None) or 150000.0)
        _q_stefan0   = max(float(getattr(fuel_cfg, "char_ox_q_stefan0_W_m2", None) or 80000.0), 1.0)
        _char_yield  = float(getattr(fuel_cfg, "char_ox_char_yield", 0.25))
        _char_hoc    = max(float(getattr(fuel_cfg, "char_ox_char_hoc_J_kg", 32.7e6)), 1.0)
        char_ox_hrrpua_pp = np.zeros(n, dtype=float)
        _char_pool   = 0.0   # [kg/m²] running char pool (production − consumption)
        for i in range(n):
            _q_in_i, _ = eval_q_in_incident_W_m2(float(sol.t[i]), _q_in_src)
            if _use_kc_pp and _N >= 3:
                # alpha states at indices 2N..3N-1 (per-node M at N..2N-1)
                _alphas_i = [float(sol.y[2 * _N + j, i]) for j in range(_N)]
                if _per_node_M_pp:
                    _M_nodes_i = [max(float(sol.y[_N + j, i]), 0.0) for j in range(_N)]
                    _is_active_drain_i = [_M_nodes_i[j] > 1e-9 for j in range(_N)]
                    # Soft BT weight (same ramp as fuel_rhs): 1→0 over alpha in [bt-0.15, bt]
                    _pp_ramp_hi = _alpha_bt_pp
                    _pp_ramp_lo = max(_pp_ramp_hi - 0.15, 0.0) if _bt_active_pp else 0.0
                    _pp_ramp_w  = max(_pp_ramp_hi - _pp_ramp_lo, 0.01)
                    def _soft_bt_pp(a):
                        if not _bt_active_pp: return 1.0
                        return max(0.0, min(1.0, (_pp_ramp_hi - a) / _pp_ramp_w))
                    _M_soft_thr_pp = 0.01
                    def _m_ramp_pp(m):
                        return min(1.0, m / _M_soft_thr_pp) if m > 0.0 else 0.0
                    _wts = [_C_pp[j] * _soft_bt_pp(_alphas_i[j]) * _m_ramp_pp(_M_nodes_i[j])
                            for j in range(_N)]
                    _c_tot_i = max(sum(_wts), 1e-12)
                    _alpha_bar_i = sum(_wts[j] * _alphas_i[j] for j in range(_N)) / _c_tot_i
                    _M_eff_i = sum(_M_nodes_i[j] for j in range(_N) if _is_active_drain_i[j])
                    # Unfiltered alpha_bar for char_ox: uses all nodes (BT'd and exhausted included)
                    # so char oxidation continues correctly after fuel burnout
                    _alpha_bar_all_i = sum(_C_pp[j] * _alphas_i[j] for j in range(_N)) / _C_tot_pp
                elif _bt_active_pp:
                    _node_bt_i = [_alphas_i[j] >= _alpha_bt_pp for j in range(_N)]
                    _wts = [0.0 if _node_bt_i[j] else _C_pp[j] for j in range(_N)]
                    _c_tot_i = max(sum(_wts), 1e-12)
                    _alpha_bar_i = sum(_wts[j] * _alphas_i[j] for j in range(_N)) / _c_tot_i
                    _M_eff_i = float(sol.y[_M1_idx, i])
                    _alpha_bar_all_i = sum(_C_pp[j] * _alphas_i[j] for j in range(_N)) / _C_tot_pp
                else:
                    _alpha_bar_i = sum(_C_pp[j] * _alphas_i[j] for j in range(_N)) / _C_tot_pp
                    _M_eff_i = float(sol.y[_M1_idx, i])
                    _alpha_bar_all_i = _alpha_bar_i
            elif _use_kc_pp:
                # 2-node: alpha states at y[3] and y[4]
                _c_tot2 = max(_C_pp[0] + _C_pp[1], 1e-12)
                _alpha_bar_i = (_C_pp[0] * float(sol.y[3, i]) + _C_pp[1] * float(sol.y[4, i])) / _c_tot2
                _M_eff_i = float(sol.y[_M1_idx, i])
                _alpha_bar_all_i = _alpha_bar_i
            else:
                _alpha_bar_i = 0.0
                _M_eff_i = float(sol.y[_M1_idx, i])
                _alpha_bar_all_i = _alpha_bar_i
            _terms = compute_hog_terms(
                T1=float(sol.y[0, i]),
                M1=_M_eff_i,
                q_in=_q_in_i,
                fuel_cfg=fuel_cfg,
                t=float(sol.t[i]),
                alpha_bar=_alpha_bar_i,
            )
            m_py_pp[i]   = float(_terms["m_dot_used"])
            m_dot_cap[i] = float(_terms["m_dot_cap"])
            m_dot_kin[i] = float(_terms["m_dot_kin"])
            if _use_char_ox:
                _dt_i = float(sol.t[i] - sol.t[i - 1]) if i > 0 else 0.0
                if _dt_i > 0:
                    # Add char produced from volatile pyrolysis this step [kg/m²]
                    _char_pool += _char_yield * m_py_pp[i] * _dt_i
                    # Raw char oxidation capacity [W/m²]: uses unfiltered all-nodes alpha so
                    # char_ox continues correctly after BT and after fuel exhaustion.
                    _m_py_s0 = float(getattr(fuel_cfg, "char_ox_m_py_stefan0_kg_m2_s", None) or 0.0)
                    if _m_py_s0 > 0.0:
                        _f_stefan = max(1.0 - m_py_pp[i] / _m_py_s0, 0.0)
                    else:
                        _f_stefan = max(1.0 - _q_in_i / _q_stefan0, 0.0)
                    _raw_char_ox = float(_alpha_bar_all_i) * _q_char_ref * _f_stefan
                    # Char demanded this step [kg/m²]; limited to available pool
                    _delta_demanded = _raw_char_ox * _dt_i / _char_hoc
                    _delta_actual = min(_delta_demanded, _char_pool)
                    _char_pool = max(_char_pool - _delta_actual, 0.0)
                    # Actual char_ox [W/m²]
                    char_ox_hrrpua_pp[i] = _delta_actual * _char_hoc / _dt_i

    # Scale cap-rate m_py_pp by staged volatile fraction (post-HoG or post-Stefan).
    # m_py_pp from rate-cap block = total solid consumption rate.
    # Staged volatile fraction = (y1*r1 + y2*r2)/(r1+r2).
    # HRRPUA = cap_rate × vol_frac × hoc_eff.
    # seq_T_ign_K gate: below this surface temperature, HRRPUA is forced to zero
    # (pre-ignition gate — prevents false signal before the material is burning).
    _staged_pp_offset = _fl_offset_3n + (3 if use_3node_front_limit else 0)
    if _use_staged_init and m_py_pp is not None and sol.y.shape[0] > _staged_pp_offset + 1:
        _y1_pp = float(getattr(fuel_cfg, "seq_y1_vol", 1.0))
        _y2_pp = float(getattr(fuel_cfg, "seq_y2_vol", 0.0))
        _A1_pp = float(fuel_cfg.A1_py)
        _E1_pp = float(fuel_cfg.E1_py)
        _A2_pp = float(fuel_cfg.A2_py)
        _E2_pp = float(fuel_cfg.E2_py)
        _R_pp = 8.314
        _T_ign_pp = float(getattr(fuel_cfg, "seq_T_ign_K", 560.0) or 560.0)
        _n_interp_pp = float(getattr(fuel_cfg, "seq_vol_interp_n", 0.0) or 0.0)
        # m1 at ignition onset (first T1 >= T_ign_K) as power-law denominator.
        # Using this instead of t=0 ensures vol_frac=y1 exactly at the flash peak,
        # regardless of pre-ignition Arrhenius depletion while T1 < T_ign.
        _m1_0_pp = max(float(sol.y[_staged_pp_offset, 0]), 1e-15)  # fallback: t=0
        for _j_pp in range(sol.t.size):
            if max(float(sol.y[0, _j_pp]), 300.0) >= _T_ign_pp:
                _m1_0_pp = max(float(sol.y[_staged_pp_offset, _j_pp]), 1e-15)
                break
        for _i_pp in range(sol.t.size):
            _T1_pp = max(float(sol.y[0, _i_pp]), 300.0)
            if _T1_pp < _T_ign_pp:
                # Pre-ignition gate: surface not yet hot enough for volatile combustion.
                m_py_pp[_i_pp] = 0.0
                continue
            _m1_pp = max(float(sol.y[_staged_pp_offset, _i_pp]), 0.0)
            _m2_pp = max(float(sol.y[_staged_pp_offset + 1, _i_pp]), 0.0)
            if _n_interp_pp > 0.0:
                # Power-law interpolation: vol_frac = y2 + (y1-y2)*(m1/m1_0)^n.
                # n<1 gives concave decay (fast initial drop, slow tail) matching FR EXP shape.
                # Invariant: vol_frac = y1 at t=ignition (m1=m1_0), → y2 as m1→0.
                _frac_pp = min(_m1_pp / _m1_0_pp, 1.0)
                _vol_frac_pp = _y2_pp + (_y1_pp - _y2_pp) * (_frac_pp ** _n_interp_pp)
            else:
                # Rate-weighted (default): vol_frac = (y1*r1 + y2*r2)/(r1+r2).
                _k1_pp = _A1_pp * np.exp(-_E1_pp / (_R_pp * _T1_pp))
                _k2_pp = _A2_pp * np.exp(-_E2_pp / (_R_pp * _T1_pp))
                _r1_pp = _k1_pp * _m1_pp
                _r2_pp = _k2_pp * _m2_pp
                _r_tot_pp = _r1_pp + _r2_pp
                if _r_tot_pp > 1e-15:
                    _vol_frac_pp = (_y1_pp * _r1_pp + _y2_pp * _r2_pp) / _r_tot_pp
                elif _m1_pp + _m2_pp <= 0.0:
                    _vol_frac_pp = 0.0
                else:
                    # Rates negligible → zero volatile flux regardless of remaining mass.
                    # Physical: zero decomposition rate = zero volatile release.
                    # (Prior mass-weighted fallback produced false HRRPUA at ambient T.)
                    _vol_frac_pp = 0.0
            m_py_pp[_i_pp] *= _vol_frac_pp

    # --- Back-face pyrolysis mass flux [kg/m²/s] ---
    back_py_mdot_pp: NDArray[np.float64] | None = None
    _back_py_en = (
        _N >= 3
        and bool(getattr(fuel_cfg, "back_face_pyrolysis_enable", False))
        and m_py_pp is not None
    )
    if _back_py_en:
        n_bp = sol.t.size
        _M1_idx_bp = _N  # M1 is at index N in N-node state vector
        _T_back_idx = _N - 1  # back-face temperature is the last temperature state
        _back_frac = max(float(getattr(fuel_cfg, "back_face_node_frac", 0.333)), 1e-9)
        _rho_bp = max(float(getattr(fuel_cfg, "density", None) or 400.0), 1.0)
        _L0_bp  = max(float(getattr(fuel_cfg, "regression_L0_m", 0.038)), 1e-9)
        _m_total_bp = (
            float(fuel_cfg.m_fuel_total_kg_m2)
            if getattr(fuel_cfg, "m_fuel_total_kg_m2", None) is not None
            else _rho_bp * _L0_bp
        )
        # Use α_N (back-face node char fraction) when available — kinetically driven by T_back,
        # self-limits: as T_back rises, α_N → 1 → remaining fuel → 0.
        _use_alpha_back = str(getattr(fuel_cfg, "char_state_mode", "")).strip().lower() == "kinetic"
        # Per-node M layout: α at 2N..3N-1; back-face α_N is at index 3N-1.
        _alpha_back_idx = 3 * _N - 1  # y[2N + (N-1)] = y[3N-1] in per-node-M kinetic state
        _has_alpha_back = _use_alpha_back and sol.y.shape[0] > _alpha_back_idx
        _back_face_T_py = getattr(fuel_cfg, "back_face_T_py_K", -1.0)
        if _back_face_T_py is not None and float(_back_face_T_py) > 0.0:
            _T_py_bp = float(_back_face_T_py)
        else:
            _T_py_bp = float(getattr(fuel_cfg, "regression_T_py_K", 600.0) or 600.0)
        back_py_mdot_pp = np.zeros(n_bp, dtype=float)
        # HoG fuel budget: back-face node mass = rho × L0 × back_frac [kg/m²].
        # This is the fuel pool accessible to back-face gasification (node 3 only).
        # A finite budget creates the falling limb of the secondary HRRPUA hump
        # when the back-face fuel is exhausted.
        _m_burned_back = 0.0  # cumulative back-face mass consumed [kg/m²]
        # Fixed back-face fuel budget = rho × L0 × node_frac [kg/m²].
        # Budget is the only depletion gate — does NOT depend on char-front position.
        # Rationale: the Stefan front nominally "reaching" the back face does not mean the
        # volatile pool is exhausted.  The volatile pool burns until budget is consumed,
        # producing the correct exponential-like declining tail in the HRRPUA secondary peak.
        _m_back_budget_fixed = _rho_bp * _L0_bp * _back_frac
        _hog_min_char_frac = float(getattr(fuel_cfg, "back_face_hog_min_char_frac", 0.0) or 0.0)
        for _i in range(n_bp):
            _T_back_i = float(sol.y[_T_back_idx, _i])
            if _has_alpha_back:
                _alpha_back_i = min(max(float(sol.y[_alpha_back_idx, _i]), 0.0), 1.0)
                _m_remain_back = (1.0 - _alpha_back_i) * _m_total_bp * _back_frac
            else:
                _M1_i = max(float(sol.y[_M1_idx_bp, _i]), 0.0)
                _m_remain_back = _M1_i * _m_total_bp * _back_frac
            if bool(getattr(fuel_cfg, "back_face_hog_enable", False)):
                # Energy-balance (HoG) back-face rate:
                #   m_dot = (q_net/dH_py) × (m_avail / m_budget) — smooth depletion factor
                # Budget = remaining virgin wood = rho*(L0-delta_py), less what back-face
                # has already consumed.  The depletion factor (m_avail/m_budget) creates a
                # smooth exponential-like decline rather than a hard cliff when fuel runs out.
                # K23 is enhanced by k_crack (not in ODE) to capture crack-driven conductance.
                # Gate: delta_py/L >= min_char_frac AND q_net > 0 AND budget not exhausted.
                # min_char_frac delays back-face activation until crack-enhanced conductance
                # is meaningful — preserves the primary/secondary valley in thin-panel EXP.
                _delta_py_i = max(float(sol.y[_fl_offset_3n, _i]), 0.0) if sol.y.shape[0] > _fl_offset_3n else 0.0
                # Activation ramp for back-face HoG: two modes.
                # T3-based (physics): triggers when back face heats to extractive volatilization
                #   onset temperature — directly encodes the thermal wave breakthrough event.
                #   back_face_hog_T_min_K > 0 selects this path.
                # Legacy (char-front depth): delays activation until front char depth fraction
                #   exceeds min_char_frac — a proxy trigger unrelated to back-face temperature.
                #   Used for backward compatibility when T_min_K = 0 (default).
                _T_min_hog = float(getattr(fuel_cfg, "back_face_hog_T_min_K", 0.0) or 0.0)
                if _T_min_hog > 0.0:
                    # Physics-based: T3 (back-face) reaches extractive volatilization onset.
                    # Ramp 0→1 over T_min_K → T_min_K + T_ramp_dT_K.
                    _T_ramp_dT = float(getattr(fuel_cfg, "back_face_hog_T_ramp_dT_K", 20.0) or 20.0)
                    _ramp_factor = max(0.0, min((_T_back_i - _T_min_hog) / max(_T_ramp_dT, 1e-3), 1.0))
                else:
                    # Legacy: char-front depth trigger (backward compatible with Wood Stud decks).
                    _char_frac_i = (_delta_py_i / _L0_bp) if _L0_bp > 0 else 0.0
                    _ramp_width = float(getattr(fuel_cfg, "back_face_hog_ramp_width", 0.20) or 0.20)
                    _ramp_factor = max(0.0, min((_char_frac_i - _hog_min_char_frac) / _ramp_width, 1.0))
                if _ramp_factor <= 0.0 or _m_burned_back >= _m_back_budget_fixed:
                    # HoG inactive: back-face not yet warm enough, or fuel budget exhausted.
                    # Note: does NOT gate on char-front position (_m_virgin_remain).
                    # The back-face fuel pool is independent of the Stefan front — even after the
                    # front nominally reaches the back face, the volatile pool continues to burn
                    # until the fixed budget (rho × L0 × node_frac) is consumed.
                    _m_dot_back = 0.0
                else:
                    _k_char_bp = float(getattr(fuel_cfg, "k_char", 0.06) or 0.06)
                    # Use dedicated back-face k_crack if set; fall back to ODE k_crack_frac.
                    # This decouples the Stefan front advance (ODE) from the K23 conductance
                    # (HoG post-processing), enabling deep valleys AND large secondary humps.
                    _k_crack_bp_hog = float(getattr(fuel_cfg, "back_face_hog_k_crack_frac", 0.0) or 0.0)
                    _k_crack_bp = _k_crack_bp_hog if _k_crack_bp_hog > 0.0 else float(getattr(fuel_cfg, "k_crack_frac", 0.0) or 0.0)
                    _dH_py_bp = float(getattr(fuel_cfg, "dH_py", 1.8e6) or 1.8e6)
                    _h_open_bp = float(getattr(fuel_cfg, "h_open", 10.0) or 10.0)
                    _eps_open_bp = float(getattr(fuel_cfg, "eps_open", 0.9) or 0.9)
                    _Tamb_bp = float(env_cfg.Tamb)
                    # K23_eff computed from geometry (centroid-to-centroid, nodes 2→3).
                    # Cannot read fuel_cfg.K23 — that field does not exist in FuelConfig.
                    # dx23 = (f2/2 + f3/2) * L0 — distance between node centroids.
                    _f2_bp = float(getattr(fuel_cfg, "node2_frac", 0.80) or 0.80)
                    _f3_bp = float(getattr(fuel_cfg, "node3_frac", 0.10) or 0.10)
                    _dx23_bp = (_f2_bp / 2.0 + _f3_bp / 2.0) * _L0_bp
                    # Char conductivity enhanced by crack factor at char-front depth
                    # (_delta_py_i already read above for threshold check)
                    _k_eff_bp = _k_char_bp * (1.0 + _k_crack_bp * _delta_py_i / _L0_bp)
                    _K23_eff_bp = _k_eff_bp / _dx23_bp if _dx23_bp > 0 else 0.0
                    # Use middle interior node as hot-side driver (N=3: T2 at 45%; N=5: T3 at 50%).
                    # For N=3: (_N-1)//2 = 1 → sol.y[1] = T2 (unchanged, backward-compatible).
                    # For N=5: (_N-1)//2 = 2 → sol.y[2] = T3 (correct bulk mid-panel temp).
                    # The default dx23=8.55mm coincidentally equals the T3→T5 centroid distance
                    # for the 5-node [0.10,0.20,0.40,0.20,0.10] configuration.
                    _T2_bp = float(sol.y[(_N - 1) // 2, _i])
                    _q_cond_bp = _K23_eff_bp * max(_T2_bp - _T_back_i, 0.0)
                    _q_loss_bp = (
                        _h_open_bp * max(_T_back_i - _Tamb_bp, 0.0)
                        + _eps_open_bp * 5.67e-8 * max(_T_back_i**4 - _Tamb_bp**4, 0.0)
                    )
                    _q_net_bp = max(_q_cond_bp - _q_loss_bp, 0.0)
                    # Back-face fuel budget = back-face node mass only (node3_frac of total panel).
                    # Physically: back-face HoG draws from the back-node fuel pool only,
                    # not the entire remaining panel.  This creates a finite budget that
                    # depletes within the simulation window, producing the falling limb of
                    # the secondary hump.
                    # Energy-limited rate [kg/m²/s], smoothly tapered by remaining fuel fraction.
                    # Smooth depletion: rate × (m_avail/m_budget_fixed) → exponential-like decay
                    # rather than a hard cliff.  Total energy delivered ≈ m_budget_fixed × hoc_eff.
                    _m_dot_energy = (_q_net_bp / _dH_py_bp) if _dH_py_bp > 0 else 0.0
                    _depletion_frac = max(1.0 - _m_burned_back / _m_back_budget_fixed, 0.0)
                    _m_dot_back = _m_dot_energy * _depletion_frac * _ramp_factor
                    _dt_i = sol.t[_i] - sol.t[max(_i - 1, 0)]
                    _m_burned_back += _m_dot_back * _dt_i
            else:
                # Legacy Arrhenius back-face path — appropriate for THICK panels only.
                #
                # Selection rule:
                #   back_face_hog_enable = False  (this branch):
                #     Use for panels where T_back eventually reaches T_py (≥ 600 K).
                #     Example: Wood Stud 38 mm — T_back reaches 600 K via conduction through
                #     the thick panel, so standard Arrhenius kinetics at T_back fires correctly.
                #     The temperature cap to T_py prevents a spurious spike at T_back > T_py.
                #
                #   back_face_hog_enable = True  (HoG branch above):
                #     Use for THIN panels (L < ~25 mm) where T_back never reaches T_py.
                #     Example: Basswood Panel 19 mm — T_back peaks at ~380 K, far below T_py.
                #     Arrhenius here gives negligible rates. HoG uses energy-balance
                #     (q_cond > q_loss) gated by back_face_hog_T_min_K (extractive volatilization
                #     onset, ~363–373 K per Shafizadeh 1982) to capture the secondary peak.
                #
                # Do NOT set back_face_hog_enable=True for thick panels — HoG will fire
                # prematurely at T_min_K (~363 K) rather than at T_py, producing a spurious
                # secondary peak that does not appear in EXP data.
                _T_back_eff = min(_T_back_i, _T_py_bp)
                _m_dot_back, _ = compute_m_dot_kinetics(_T_back_eff, _m_remain_back, fuel_cfg)
            back_py_mdot_pp[_i] = max(_m_dot_back, 0.0)

    return FuelIntegrationResult(
        t=sol.t,
        y=sol.y.T,
        m_py_pp=m_py_pp,
        m_dot_cap=m_dot_cap,
        m_dot_kin=m_dot_kin,
        handoff_blend=handoff_blend,
        thermal_node_order=thermal_order,
        char_ox_hrrpua_pp=char_ox_hrrpua_pp if _use_char_ox else None,
        back_py_mdot_pp=back_py_mdot_pp,
    )


@dataclass
class FuelIntegrationResult:
    """Time history of fuel state."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    m_py_pp: NDArray[np.float64] | None = None
    m_dot_cap: NDArray[np.float64] | None = None
    m_dot_kin: NDArray[np.float64] | None = None
    handoff_blend: NDArray[np.float64] | None = None
    thermal_node_order: int = 2
    char_ox_hrrpua_pp: NDArray[np.float64] | None = None  # [W/m²] char oxidation HRRPUA addend
    back_py_mdot_pp: NDArray[np.float64] | None = None    # [kg/m²/s] back-face pyrolysis mass flux
