from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import csv

import numpy as np

import os
from pathlib import Path
from model.config.defaults import default_env_config, default_fuel_config, default_sim_config
from model.io.text_input import (
    RomInputs,
    convert_q_in,
    convert_m_py,
    hoc_eff_to_j_per_kg,
    hoc_eff_to_kj_per_kg,
    load_text_input,
    normalize_hoc_units,
    q_in_callable,
    resolve_geometry,
    apply_material_geometry,
)
from model.fuel.pyrolysis import (
    compute_semi_global_seq_yield_rates,
    compute_two_step_sequential_rates,
    compute_pyrolysis_kinetics_terms,
    pyrolysis_flux,
    resolve_pyrolysis_mass_source,
    resolve_total_fuel_mass_kg_m2,
)
from model.fuel.depletion import apply_depletion
from model.fuel.heat_transfer import open_face_loss_flux
from model.fuel.two_node import (
    compute_front_limit_terms,
    compute_pyrolysis_attribution_terms,
    compute_surface_heat_terms,
    eval_q_in_incident_W_m2,
    integrate_fuel,
)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return "_".join([p for p in safe.split("_") if p])


def _q_inc_ramp_factor(t: float, mode: str, tau: float) -> float:
    mode_l = (mode or "none").strip().lower()
    if mode_l == "none":
        return 1.0
    t_eff = max(float(t), 0.0)
    tau_eff = max(float(tau), 1.0e-9)
    if mode_l == "exp":
        return float(1.0 - np.exp(-t_eff / tau_eff))
    if mode_l == "cosine":
        if t_eff >= tau_eff:
            return 1.0
        return float(0.5 * (1.0 - np.cos(np.pi * (t_eff / tau_eff))))
    return 1.0


# Provenance: SRC-2026-001 (review anchor), SRC-2026-003 (sequential model-form prototype anchor),
# SRC-2026-006 (secondary charring extension model-form anchor), SRC-2026-007 (3-node thermal reduction prototype),
# SRC-2026-010 (semi-global product-yield staged prototype)
SEQUENTIAL_KINETICS_SOURCE_ID = "SRC-2026-003"
SEMI_GLOBAL_YIELD_SOURCE_ID = "SRC-2026-010"
REACTIVE_ACCESS_SOURCE_ID = "SRC-2026-013"


# ── Output data structures ────────────────────────────────────────────────────
# RomSignals holds the full time-series output of run_rom() — temperatures,
# HRRPUA, pyrolysis rates, per-node char fractions, heat-term breakdowns, and
# diagnostic arrays.  All arrays are on a uniform dt_out grid.

@dataclass
class RomSignals:
    t: np.ndarray
    T_surf: np.ndarray
    T_inner: Optional[np.ndarray]
    M1_moisture: Optional[np.ndarray]
    hrrpua: np.ndarray
    hrrpua_diag: Optional[np.ndarray]
    m_py: np.ndarray
    mlr: np.ndarray
    mass_total: np.ndarray
    mass_total_units: str = "kg_total"
    m_fuel_remaining_kg_m2: Optional[np.ndarray] = None
    pyrolysis_mass_source: str = "legacy_M1"
    m_seq_stage1_kg_m2: Optional[np.ndarray] = None
    m_seq_stage2_kg_m2: Optional[np.ndarray] = None
    m_seq_residue_kg_m2: Optional[np.ndarray] = None
    mdot_seq_step1_kg_m2_s: Optional[np.ndarray] = None
    mdot_seq_step2_kg_m2_s: Optional[np.ndarray] = None
    mdot_seq_char_sink_kg_m2_s: Optional[np.ndarray] = None
    mdot_seq_vol_kg_m2_s: Optional[np.ndarray] = None
    m_seq_reactive_total_kg_m2: Optional[np.ndarray] = None
    m_smolder_pool_kg_m2: Optional[np.ndarray] = None  # [kg/m²] smoldering char pool (Frandsen 1991 glowing combustion)
    sequential_kinetics_enabled: bool = False
    sequential_kinetics_source_id: Optional[str] = None
    sequential_mass_balance_max_residual_kg_m2_s: Optional[float] = None
    reactive_access_mode: str = "none"
    reactive_access_source_id: Optional[str] = None
    access_factor_stage2: Optional[np.ndarray] = None
    time_grid_mode: str = "solver"
    dt_out: float = 1.0
    t_solver: Optional[np.ndarray] = None
    mass_total_solver: Optional[np.ndarray] = None
    m_py_solver: Optional[np.ndarray] = None
    hrrpua_diag_solver: Optional[np.ndarray] = None
    hoc_eff_raw: float = 1.0
    hoc_units: str = "kJ/kg"
    hoc_eff_J_kg: float = 1000.0
    hoc_eff: float = 1.0
    q_in_incident_W_m2: Optional[np.ndarray] = None
    q_net_into_surface_W_m2: Optional[np.ndarray] = None
    q_conv_loss_W_m2: Optional[np.ndarray] = None
    q_rad_loss_W_m2: Optional[np.ndarray] = None
    q_in_W_m2: Optional[np.ndarray] = None
    q_conv_W_m2: Optional[np.ndarray] = None
    q_rad_W_m2: Optional[np.ndarray] = None
    q_net_surface_W_m2: Optional[np.ndarray] = None
    h_amb_W_m2K: Optional[np.ndarray] = None
    eps_surface: Optional[np.ndarray] = None
    q_in_mode: str = "incident"
    q_in_value_W_m2: float = 0.0
    h_amb_model: str = "UNKNOWN"
    rom_eps: Optional[float] = None
    q_in_source: Optional[list[str]] = None
    deck_q_in_units_raw: str = "UNKNOWN"
    deck_q_in_constant_raw: Optional[float] = None
    cfg_q_in_mode: str = "incident"
    cfg_q_in_constant_W_m2: float = 0.0
    cfg_q_in_constant_source: str = "unknown"
    cfg_q_in_constant_altkey_raw: Optional[float] = None
    cfg_has_schedule: bool = False
    cfg_schedule_len: int = 0
    q_in_source_at_t0: str = "none"
    q_in_applied_at_t0_W_m2: float = 0.0
    qin_guardrail_error: Optional[str] = None
    registry_exposure_q_kW_m2: Optional[float] = None
    pyro_m_remaining_kg_m2: Optional[np.ndarray] = None
    pyro_mdot_kin_kg_m2_s: Optional[np.ndarray] = None
    pyro_mdot_cap_kg_m2_s: Optional[np.ndarray] = None
    pyro_mdot_limit_kg_m2_s: Optional[np.ndarray] = None
    pyro_mdot_final_kg_m2_s: Optional[np.ndarray] = None
    pyro_limiter_active: Optional[np.ndarray] = None
    pyro_cap_active: Optional[np.ndarray] = None
    pyro_kinetics_gate_active: Optional[np.ndarray] = None
    pyro_gate_factor: Optional[np.ndarray] = None
    area_m2_used: Optional[float] = None
    thickness_m_used: Optional[float] = None
    rho_kg_m3_used: Optional[float] = None
    T_mid: Optional[np.ndarray] = None
    fuel_cfg_used: Optional[object] = None
    T_nodes: Optional[list] = None       # [T1_arr, T2_arr, ..., TN_arr] on output grid
    alpha_nodes: Optional[list] = None   # [α1_arr, ..., αN_arr] on output grid (None if not kinetic)
    flame_height_m: Optional[np.ndarray] = None      # L_f(t) [m] — Heskestad (1983); active when flame_geometry_mode=heskestad
    flame_view_factor_t: Optional[np.ndarray] = None # F(t) [-] — geometry-derived pool-fire view factor
    plume_T_K: Optional[list] = None                  # list[ndarray] — McCaffrey centerline dT_K(z,t) per height
    plume_u_m_s: Optional[list] = None                # list[ndarray] — McCaffrey centerline u_m_s(z,t) per height


@dataclass
class _FuelStatePyrolysisTrace:
    m_dot_vol_kg_m2_s: np.ndarray
    m_remaining_total_kg_m2: np.ndarray
    m_stage1_kg_m2: Optional[np.ndarray] = None
    m_stage2_kg_m2: Optional[np.ndarray] = None
    m_residue_kg_m2: Optional[np.ndarray] = None
    mdot_step1_kg_m2_s: Optional[np.ndarray] = None
    mdot_step2_kg_m2_s: Optional[np.ndarray] = None
    mdot_char_sink_kg_m2_s: Optional[np.ndarray] = None
    mass_balance_max_residual_kg_m2_s: Optional[float] = None
    sequential_source_id: Optional[str] = None
    reactive_access_mode: Optional[str] = None
    reactive_access_source_id: Optional[str] = None
    access_factor_stage2: Optional[np.ndarray] = None
    # Per-zone char fractions for evolving-property staggered coupling
    alpha_zone1: Optional[np.ndarray] = None  # char fraction at thermal node 1 vs time
    alpha_zone2: Optional[np.ndarray] = None  # char fraction at thermal node 2 vs time
    alpha_zone3: Optional[np.ndarray] = None  # char fraction at thermal node 3 vs time


def _write_pmma_debug_csv(
    out_path: Path,
    t: np.ndarray,
    y: np.ndarray,
    fuel_cfg,
    env_cfg,
    q_raw,
    q_ramped,
    ramp_mode: str,
    ramp_tau: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T1 = y[:, 0]
    T2 = y[:, 1]
    M1 = y[:, 2]
    if y.shape[1] >= 6:
        delta_arr = y[:, 3]
        m_c_arr = y[:, 4]
        L_arr = y[:, 5]
    else:
        delta_arr = np.full_like(t, float(getattr(fuel_cfg, "delta_py0_m", 0.0)))
        m_c_arr = np.full_like(t, float(getattr(fuel_cfg, "m_char0_kg_m2", 0.0)))
        L_arr = np.full_like(t, float(max(getattr(fuel_cfg, "regression_L0_m", 1.0), 1.0e-9)))

    T_sur = env_cfg.T_sur if env_cfg.T_sur is not None else env_cfg.Tamb
    back_mode = str(getattr(fuel_cfg, "back_bc_mode", "adiabatic")).strip().lower()
    eps_open = float(fuel_cfg.eps if fuel_cfg.eps_open is None else fuel_cfg.eps_open)

    cols = [
        "t_s",
        "T1_K",
        "T2_K",
        "M1",
        "delta_py_m",
        "m_c_kg_m2",
        "L_m",
        "q_inc_raw_W_m2",
        "q_inc_ramped_W_m2",
        "ramp_factor",
        "m_dot_kin_kg_m2_s",
        "m_dot_cap_kg_m2_s",
        "m_dot_pp_kg_m2_s",
        "m_dot_used_kg_m2_s",
        "m_avail_kg_m2",
        "d_delta_dt_m_s",
        "delta_ratio",
        "handoff_blend",
        "q_open_W_m2",
        "cap_limited",
        "handoff_active",
        "avail_limited",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(len(t)):
            terms = compute_front_limit_terms(
                float(T1[i]),
                float(M1[i]),
                float(delta_arr[i]),
                float(m_c_arr[i]),
                float(L_arr[i]),
                fuel_cfg,
            )
            q_open = 0.0
            if back_mode == "open":
                q_open = open_face_loss_flux(
                    T2=float(T2[i]),
                    h_open=float(getattr(fuel_cfg, "h_open", 0.0)),
                    eps_open=eps_open,
                    T_inf=float(env_cfg.Tamb),
                    T_sur=float(T_sur),
                    sigma=float(env_cfg.sigma),
                )

            q_raw_i = float(q_raw(float(t[i])))
            q_ramped_i = float(q_ramped(float(t[i])))
            ramp_i = _q_inc_ramp_factor(float(t[i]), ramp_mode, ramp_tau)
            cap_limited = int(terms["m_dot_pp"] + 1.0e-12 < terms["m_dot_kin"])
            handoff_active = int(terms["handoff_blend"] > 1.0e-9)
            avail_limited = int(terms["m_avail"] <= 0.0)

            writer.writerow(
                [
                    f"{float(t[i]):.6f}",
                    f"{float(T1[i]):.6f}",
                    f"{float(T2[i]):.6f}",
                    f"{float(M1[i]):.6f}",
                    f"{float(delta_arr[i]):.9f}",
                    f"{float(m_c_arr[i]):.9f}",
                    f"{float(L_arr[i]):.9f}",
                    f"{q_raw_i:.6f}",
                    f"{q_ramped_i:.6f}",
                    f"{ramp_i:.6f}",
                    f"{terms['m_dot_kin']:.9e}",
                    f"{terms['m_dot_cap']:.9e}",
                    f"{terms['m_dot_pp']:.9e}",
                    f"{terms['m_dot_used']:.9e}",
                    f"{terms['m_avail']:.9e}",
                    f"{terms['d_delta_dt']:.9e}",
                    f"{terms['delta_ratio']:.9f}",
                    f"{terms['handoff_blend']:.9f}",
                    f"{q_open:.6f}",
                    cap_limited,
                    handoff_active,
                    avail_limited,
                ]
            )


# ── Internal helpers for run_rom ─────────────────────────────────────────────

def _parse_flux_from_token(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    digits = "".join(ch for ch in token if ch.isdigit())
    if len(digits) >= 2:
        val = float(digits[-3:]) if len(digits) >= 3 else float(digits)
        return val
    return None


def _apply_overrides(obj, overrides: dict) -> None:
    for key, val in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, val)


def _interp_series_to_grid(t_out: np.ndarray, t_src: np.ndarray, y_src: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y_src is None:
        return None
    t_o = np.asarray(t_out, dtype=float)
    t_s = np.asarray(t_src, dtype=float)
    y_s = np.asarray(y_src, dtype=float)
    if t_o.size == 0:
        return np.asarray([], dtype=float)
    if t_s.size == 0 or y_s.size == 0:
        return None
    mask = np.isfinite(t_s) & np.isfinite(y_s)
    if np.count_nonzero(mask) == 0:
        return None
    t_s = t_s[mask]
    y_s = y_s[mask]
    order = np.argsort(t_s)
    t_s = t_s[order]
    y_s = y_s[order]
    t_unique, idx = np.unique(t_s, return_index=True)
    y_unique = y_s[idx]
    if t_unique.size == 0:
        return None
    if t_unique.size == 1:
        return np.full_like(t_o, float(y_unique[0]), dtype=float)
    return np.interp(t_o, t_unique, y_unique)


def _remaining_mass_from_legacy_m1(T1: np.ndarray, M1: np.ndarray, fuel_cfg) -> np.ndarray:
    n = min(int(T1.size), int(M1.size))
    out = np.zeros(n, dtype=float)
    for i in range(n):
        terms = compute_pyrolysis_kinetics_terms(float(T1[i]), float(M1[i]), fuel_cfg)
        out[i] = float(max(terms.get("m_remaining_kg_m2", 0.0), 0.0))
    return out


def _chem_depth_partition_mode_token(fuel_cfg) -> str:
    return str(getattr(fuel_cfg, "chem_depth_partition_mode", "none") or "none").strip().lower()


def _reactive_access_mode_token(fuel_cfg) -> str:
    return str(getattr(fuel_cfg, "reactive_access_mode", "none") or "none").strip().lower()


def _compute_delayed_access_factor(
    fuel_cfg,
    *,
    residue_frac: float,
    depth_weight: float = 1.0,
) -> float:
    mode = _reactive_access_mode_token(fuel_cfg)
    if mode == "none":
        return 1.0
    if mode != "transport_reduced_wood_char":
        return 1.0
    beta = float(getattr(fuel_cfg, "access_reduction_beta", 0.0) or 0.0)
    amin = float(getattr(fuel_cfg, "access_min", 0.0) or 0.0)
    drv = str(getattr(fuel_cfg, "access_driver", "residue_fraction") or "residue_fraction").strip().lower()
    depth_pow = float(getattr(fuel_cfg, "access_depth_weight_pow", 1.0) or 1.0)
    rf = float(np.clip(residue_frac if np.isfinite(residue_frac) else 0.0, 0.0, 1.0))
    dw = float(np.clip(depth_weight if np.isfinite(depth_weight) else 1.0, 0.0, 1.0))
    if drv == "residue_and_depth":
        driver = rf * (dw ** max(depth_pow, 0.0))
    else:
        driver = rf
    raw = 1.0 / (1.0 + max(beta, 0.0) * max(driver, 0.0))
    return float(np.clip(raw, min(max(amin, 0.0), 1.0), 1.0))


def _apply_delayed_access_to_rates(
    fuel_cfg,
    *,
    r2_used: float,
    r3_used: float,
    residue_frac: float,
    depth_weight: float = 1.0,
) -> tuple[float, float, float]:
    mode = _reactive_access_mode_token(fuel_cfg)
    if mode == "none":
        return float(r2_used), float(r3_used), 1.0
    access = _compute_delayed_access_factor(
        fuel_cfg,
        residue_frac=residue_frac,
        depth_weight=depth_weight,
    )
    target = str(getattr(fuel_cfg, "access_target", "stage2_only") or "stage2_only").strip().lower()
    if target == "delayed_paths":
        return float(r2_used) * access, float(r3_used) * access, float(access)
    return float(r2_used) * access, float(r3_used), float(access)


def _thermal_capacity_fracs_three(fuel_cfg) -> tuple[float, float, float]:
    c1 = float(getattr(fuel_cfg, "C1", 0.0) or 0.0)
    c2 = float(getattr(fuel_cfg, "C2", 0.0) or 0.0)
    c3 = float(getattr(fuel_cfg, "C3", 0.0) or 0.0)
    vals = np.asarray([c1, c2, c3], dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0.0):
        raise ValueError(
            "chem_depth_partition_mode='thermal_nodes' requires positive finite C1/C2/C3 to derive chemistry-zone masses"
        )
    s = float(np.sum(vals))
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError("Invalid thermal capacities for chemistry depth partitioning")
    fr = vals / s
    return float(fr[0]), float(fr[1]), float(fr[2])


# ── Post-processing: pyrolysis from fuel state ────────────────────────────────
# These functions re-evaluate the pyrolysis flux on the output time grid from
# the ODE state vector.  The ODE integrates mass pools; post-processing converts
# them to m_dot [kg/m²/s] for HRRPUA calculation.
#
# Variants:
#   _singlepool         — single-pool arrhenius/sigmoid/two_step
#   _sequential         — two-step staged pools (m1→m2→residue)
#   _sequential_zonal   — staged with per-zone depth partitioning
#   _semi_global        — product-yield staged (gas/intermediate/char)
#   _semi_global_zonal  — semi-global with zone partitioning
#   _pyrolysis_from_fuel_state — dispatcher selecting the right variant

def _pyrolysis_from_fuel_state_singlepool(
    t: np.ndarray,
    T1: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
) -> _FuelStatePyrolysisTrace:
    n = min(int(t.size), int(T1.size))
    m_dot = np.zeros(n, dtype=float)
    m_rem = np.zeros(n, dtype=float)
    if n == 0:
        return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=m_dot, m_remaining_total_kg_m2=m_rem)
    m_rem[0] = max(float(m0_fuel_kg_m2), 0.0)
    for i in range(n):
        m_i = max(float(m_rem[i]), 0.0)
        terms = compute_pyrolysis_kinetics_terms(float(T1[i]), 0.0, fuel_cfg, m_remain_kg_m2=m_i)
        m_dot_i = float(max(terms.get("mdot_kin_kg_m2_s", 0.0), 0.0))
        m_dot[i] = m_dot_i
        if i + 1 < n:
            dt = float(t[i + 1] - t[i])
            if np.isfinite(dt) and dt > 0.0:
                m_rem[i + 1] = max(m_i - m_dot_i * dt, 0.0)
            else:
                m_rem[i + 1] = m_i
    return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=m_dot, m_remaining_total_kg_m2=m_rem)


def _pyrolysis_from_fuel_state_sequential(
    t: np.ndarray,
    T1: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
) -> _FuelStatePyrolysisTrace:
    n = min(int(t.size), int(T1.size))
    mdot_vol = np.zeros(n, dtype=float)
    m_total = np.zeros(n, dtype=float)
    m1_arr = np.zeros(n, dtype=float)
    m2_arr = np.zeros(n, dtype=float)
    mr_arr = np.zeros(n, dtype=float)
    r1_arr = np.zeros(n, dtype=float)
    r2_arr = np.zeros(n, dtype=float)
    r3c_arr = np.zeros(n, dtype=float)
    access_arr = np.ones(n, dtype=float)
    max_resid = 0.0
    if n == 0:
        return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=mdot_vol, m_remaining_total_kg_m2=m_total)

    m0 = max(float(m0_fuel_kg_m2), 0.0)
    m1 = max(float(getattr(fuel_cfg, "seq_m1_frac", 1.0)) * m0, 0.0)
    m2 = max(float(getattr(fuel_cfg, "seq_m2_frac0", 0.0)) * m0, 0.0)
    mr = max(float(getattr(fuel_cfg, "seq_mr_frac0", 0.0)) * m0, 0.0)

    for i in range(n):
        m1 = max(float(m1), 0.0)
        m2 = max(float(m2), 0.0)
        mr = max(float(mr), 0.0)
        m1_arr[i] = m1
        m2_arr[i] = m2
        mr_arr[i] = mr
        m_total[i] = m1 + m2 + mr

        terms = compute_two_step_sequential_rates(float(T1[i]), m1, m2, fuel_cfg)
        y1 = float(terms.get("y1_vol", 0.0))
        y2 = float(terms.get("y2_vol", 0.0))
        f12 = float(terms.get("f12_to_m2", 1.0))
        r1_raw = float(max(terms.get("r1_kg_m2_s", 0.0), 0.0))
        r2_raw = float(max(terms.get("r2_kg_m2_s", 0.0), 0.0))
        r3_raw = float(max(terms.get("r3_char_kg_m2_s", 0.0), 0.0))

        dt = float(t[i + 1] - t[i]) if (i + 1 < n) else float("nan")
        if np.isfinite(dt) and dt > 0.0:
            r1_cap = m1 / dt
            r1_used = min(r1_raw, max(r1_cap, 0.0))
            r23_cap = max(m2 / dt, 0.0)
            r23_raw = max(r2_raw + r3_raw, 0.0)
            if r23_raw > r23_cap and r23_raw > 0.0:
                scale23 = r23_cap / r23_raw
                r2_used = r2_raw * scale23
                r3_used = r3_raw * scale23
            else:
                r2_used = r2_raw
                r3_used = r3_raw
        else:
            r1_used = r1_raw
            r2_used = r2_raw
            r3_used = r3_raw

        total_condensed_i = max(m1 + m2 + mr, 0.0)
        residue_frac_i = (mr / total_condensed_i) if total_condensed_i > 0.0 else 0.0
        r2_used, r3_used, access_i = _apply_delayed_access_to_rates(
            fuel_cfg,
            r2_used=r2_used,
            r3_used=r3_used,
            residue_frac=residue_frac_i,
            depth_weight=1.0,
        )

        mdot_vol_i = y1 * r1_used + y2 * r2_used
        dm1_dt = -r1_used
        r1_nonvolatile = (1.0 - y1) * r1_used
        dm2_dt = f12 * r1_nonvolatile - r2_used - r3_used
        dmr_dt = (1.0 - f12) * r1_nonvolatile + (1.0 - y2) * r2_used + r3_used
        resid_i = abs((dm1_dt + dm2_dt + dmr_dt) + mdot_vol_i)

        mdot_vol[i] = float(max(mdot_vol_i, 0.0))
        r1_arr[i] = float(max(r1_used, 0.0))
        r2_arr[i] = float(max(r2_used, 0.0))
        r3c_arr[i] = float(max(r3_used, 0.0))
        access_arr[i] = float(np.clip(access_i, 0.0, 1.0))
        if np.isfinite(resid_i):
            max_resid = max(max_resid, resid_i)

        if i + 1 < n:
            if np.isfinite(dt) and dt > 0.0:
                m1 = max(m1 + dm1_dt * dt, 0.0)
                m2 = max(m2 + dm2_dt * dt, 0.0)
                mr = max(mr + dmr_dt * dt, 0.0)

    return _FuelStatePyrolysisTrace(
        m_dot_vol_kg_m2_s=mdot_vol,
        m_remaining_total_kg_m2=m_total,
        m_stage1_kg_m2=m1_arr,
        m_stage2_kg_m2=m2_arr,
        m_residue_kg_m2=mr_arr,
        mdot_step1_kg_m2_s=r1_arr,
        mdot_step2_kg_m2_s=r2_arr,
        mdot_char_sink_kg_m2_s=r3c_arr,
        mass_balance_max_residual_kg_m2_s=float(max_resid),
        sequential_source_id=SEQUENTIAL_KINETICS_SOURCE_ID,
        reactive_access_mode=_reactive_access_mode_token(fuel_cfg),
        reactive_access_source_id=(REACTIVE_ACCESS_SOURCE_ID if _reactive_access_mode_token(fuel_cfg) != "none" else None),
        access_factor_stage2=access_arr,
    )


def _pyrolysis_from_fuel_state_sequential_zonal(
    t: np.ndarray,
    T1: np.ndarray,
    T_mid: np.ndarray,
    T_deep: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
) -> _FuelStatePyrolysisTrace:
    n = min(int(t.size), int(T1.size), int(T_mid.size), int(T_deep.size))
    mdot_vol = np.zeros(n, dtype=float)
    m_total = np.zeros(n, dtype=float)
    m1_arr = np.zeros(n, dtype=float)
    m2_arr = np.zeros(n, dtype=float)
    mr_arr = np.zeros(n, dtype=float)
    r1_arr = np.zeros(n, dtype=float)
    r2_arr = np.zeros(n, dtype=float)
    r3c_arr = np.zeros(n, dtype=float)
    access_arr = np.ones(n, dtype=float)
    max_resid = 0.0
    if n == 0:
        return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=mdot_vol, m_remaining_total_kg_m2=m_total)

    f1, f2, f3 = _thermal_capacity_fracs_three(fuel_cfg)
    z_fr = np.asarray([f1, f2, f3], dtype=float)
    z_edges = np.concatenate(([0.0], np.cumsum(z_fr)))
    z_depth_weights = np.clip(0.5 * (z_edges[:-1] + z_edges[1:]), 0.0, 1.0)
    m0 = max(float(m0_fuel_kg_m2), 0.0)
    m1z = z_fr * (max(float(getattr(fuel_cfg, "seq_m1_frac", 1.0)), 0.0) * m0)
    m2z = z_fr * (max(float(getattr(fuel_cfg, "seq_m2_frac0", 0.0)), 0.0) * m0)
    mrz = z_fr * (max(float(getattr(fuel_cfg, "seq_mr_frac0", 0.0)), 0.0) * m0)
    m0z = np.maximum(z_fr * m0, 1.0e-12)  # initial mass per zone for alpha computation
    alpha1_arr = np.zeros(n, dtype=float)
    alpha2_arr = np.zeros(n, dtype=float)
    alpha3_arr = np.zeros(n, dtype=float)

    for i in range(n):
        m1z = np.maximum(np.asarray(m1z, dtype=float), 0.0)
        m2z = np.maximum(np.asarray(m2z, dtype=float), 0.0)
        mrz = np.maximum(np.asarray(mrz, dtype=float), 0.0)
        m1_arr[i] = float(np.sum(m1z))
        m2_arr[i] = float(np.sum(m2z))
        mr_arr[i] = float(np.sum(mrz))
        m_total[i] = m1_arr[i] + m2_arr[i] + mr_arr[i]
        # Per-zone char fractions: residue mass / initial zone mass
        alpha1_arr[i] = float(np.clip(mrz[0] / m0z[0], 0.0, 1.0))
        alpha2_arr[i] = float(np.clip(mrz[1] / m0z[1], 0.0, 1.0))
        alpha3_arr[i] = float(np.clip(mrz[2] / m0z[2], 0.0, 1.0))

        dt = float(t[i + 1] - t[i]) if (i + 1 < n) else float("nan")
        Tz = (float(T1[i]), float(T_mid[i]), float(T_deep[i]))
        r1_sum = 0.0
        r2_sum = 0.0
        r3_sum = 0.0
        mdot_sum = 0.0
        resid_sum = 0.0
        access_num = 0.0
        access_den = 0.0
        for z in range(3):
            m1_i = float(max(m1z[z], 0.0))
            m2_i = float(max(m2z[z], 0.0))
            mr_i = float(max(mrz[z], 0.0))
            terms = compute_two_step_sequential_rates(Tz[z], m1_i, m2_i, fuel_cfg)
            y1 = float(terms.get("y1_vol", 0.0))
            y2 = float(terms.get("y2_vol", 0.0))
            f12_to_m2 = float(terms.get("f12_to_m2", 1.0))
            r1_raw = float(max(terms.get("r1_kg_m2_s", 0.0), 0.0))
            r2_raw = float(max(terms.get("r2_kg_m2_s", 0.0), 0.0))
            r3_raw = float(max(terms.get("r3_char_kg_m2_s", 0.0), 0.0))

            if np.isfinite(dt) and dt > 0.0:
                r1_cap = m1_i / dt
                r1_used = min(r1_raw, max(r1_cap, 0.0))
                r23_cap = max(m2_i / dt, 0.0)
                r23_raw = max(r2_raw + r3_raw, 0.0)
                if r23_raw > r23_cap and r23_raw > 0.0:
                    scale23 = r23_cap / r23_raw
                    r2_used = r2_raw * scale23
                    r3_used = r3_raw * scale23
                else:
                    r2_used = r2_raw
                    r3_used = r3_raw
            else:
                r1_used = r1_raw
                r2_used = r2_raw
                r3_used = r3_raw

            total_condensed_z = max(m1_i + m2_i + mr_i, 0.0)
            residue_frac_z = (mr_i / total_condensed_z) if total_condensed_z > 0.0 else 0.0
            r2_used, r3_used, access_z = _apply_delayed_access_to_rates(
                fuel_cfg,
                r2_used=r2_used,
                r3_used=r3_used,
                residue_frac=residue_frac_z,
                depth_weight=float(z_depth_weights[z]),
            )

            mdot_vol_i = y1 * r1_used + y2 * r2_used
            dm1_dt = -r1_used
            r1_nonvolatile = (1.0 - y1) * r1_used
            dm2_dt = f12_to_m2 * r1_nonvolatile - r2_used - r3_used
            dmr_dt = (1.0 - f12_to_m2) * r1_nonvolatile + (1.0 - y2) * r2_used + r3_used
            resid_i = abs((dm1_dt + dm2_dt + dmr_dt) + mdot_vol_i)

            r1_sum += float(max(r1_used, 0.0))
            r2_sum += float(max(r2_used, 0.0))
            r3_sum += float(max(r3_used, 0.0))
            mdot_sum += float(max(mdot_vol_i, 0.0))
            resid_sum += float(resid_i if np.isfinite(resid_i) else 0.0)
            w_access = max(float(m2_i), 0.0)
            access_num += float(access_z) * w_access
            access_den += w_access

            if i + 1 < n and np.isfinite(dt) and dt > 0.0:
                m1z[z] = max(m1_i + dm1_dt * dt, 0.0)
                m2z[z] = max(m2_i + dm2_dt * dt, 0.0)
                mrz[z] = max(mr_i + dmr_dt * dt, 0.0)

        mdot_vol[i] = mdot_sum
        r1_arr[i] = r1_sum
        r2_arr[i] = r2_sum
        r3c_arr[i] = r3_sum
        if access_den > 0.0:
            access_arr[i] = float(np.clip(access_num / access_den, 0.0, 1.0))
        else:
            access_arr[i] = 1.0
        max_resid = max(max_resid, resid_sum)

    return _FuelStatePyrolysisTrace(
        m_dot_vol_kg_m2_s=mdot_vol,
        m_remaining_total_kg_m2=m_total,
        m_stage1_kg_m2=m1_arr,
        m_stage2_kg_m2=m2_arr,
        m_residue_kg_m2=mr_arr,
        mdot_step1_kg_m2_s=r1_arr,
        mdot_step2_kg_m2_s=r2_arr,
        mdot_char_sink_kg_m2_s=r3c_arr,
        mass_balance_max_residual_kg_m2_s=float(max_resid),
        sequential_source_id=SEQUENTIAL_KINETICS_SOURCE_ID,
        reactive_access_mode=_reactive_access_mode_token(fuel_cfg),
        reactive_access_source_id=(REACTIVE_ACCESS_SOURCE_ID if _reactive_access_mode_token(fuel_cfg) != "none" else None),
        access_factor_stage2=access_arr,
        alpha_zone1=alpha1_arr,
        alpha_zone2=alpha2_arr,
        alpha_zone3=alpha3_arr,
    )


def _pyrolysis_from_fuel_state_semi_global(
    t: np.ndarray,
    T1: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
) -> _FuelStatePyrolysisTrace:
    n = min(int(t.size), int(T1.size))
    mdot_vol = np.zeros(n, dtype=float)
    m_total = np.zeros(n, dtype=float)
    m1_arr = np.zeros(n, dtype=float)
    m2_arr = np.zeros(n, dtype=float)
    mr_arr = np.zeros(n, dtype=float)
    r1_arr = np.zeros(n, dtype=float)
    r2_arr = np.zeros(n, dtype=float)
    r3c_arr = np.zeros(n, dtype=float)
    access_arr = np.ones(n, dtype=float)
    max_resid = 0.0
    if n == 0:
        return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=mdot_vol, m_remaining_total_kg_m2=m_total)

    m0 = max(float(m0_fuel_kg_m2), 0.0)
    m1 = max(float(getattr(fuel_cfg, "seq_m1_frac", 1.0)) * m0, 0.0)
    m2 = max(float(getattr(fuel_cfg, "seq_m2_frac0", 0.0)) * m0, 0.0)
    mr = max(float(getattr(fuel_cfg, "seq_mr_frac0", 0.0)) * m0, 0.0)

    for i in range(n):
        m1 = max(float(m1), 0.0)
        m2 = max(float(m2), 0.0)
        mr = max(float(mr), 0.0)
        m1_arr[i] = m1
        m2_arr[i] = m2
        mr_arr[i] = mr
        m_total[i] = m1 + m2 + mr

        terms = compute_semi_global_seq_yield_rates(float(T1[i]), m1, m2, fuel_cfg)
        y_g1 = float(terms.get("sg_y_g1", 0.0))
        y_i1 = float(terms.get("sg_y_i1", 0.0))
        y_c1 = float(terms.get("sg_y_c1", 0.0))
        y_g2 = float(terms.get("sg_y_g2", 0.0))
        y_c2 = float(terms.get("sg_y_c2", 0.0))
        r1_raw = float(max(terms.get("r1_kg_m2_s", 0.0), 0.0))
        r2_raw = float(max(terms.get("r2_kg_m2_s", 0.0), 0.0))
        r3_raw = float(max(terms.get("r3_char_kg_m2_s", 0.0), 0.0))

        dt = float(t[i + 1] - t[i]) if (i + 1 < n) else float("nan")
        if np.isfinite(dt) and dt > 0.0:
            r1_cap = m1 / dt
            r1_used = min(r1_raw, max(r1_cap, 0.0))
            r23_cap = max(m2 / dt, 0.0)
            r23_raw = max(r2_raw + r3_raw, 0.0)
            if r23_raw > r23_cap and r23_raw > 0.0:
                scale23 = r23_cap / r23_raw
                r2_used = r2_raw * scale23
                r3_used = r3_raw * scale23
            else:
                r2_used = r2_raw
                r3_used = r3_raw
        else:
            r1_used = r1_raw
            r2_used = r2_raw
            r3_used = r3_raw

        total_condensed_i = max(m1 + m2 + mr, 0.0)
        residue_frac_i = (mr / total_condensed_i) if total_condensed_i > 0.0 else 0.0
        r2_used, r3_used, access_i = _apply_delayed_access_to_rates(
            fuel_cfg,
            r2_used=r2_used,
            r3_used=r3_used,
            residue_frac=residue_frac_i,
            depth_weight=1.0,
        )

        mdot_vol_i = y_g1 * r1_used + y_g2 * r2_used
        dm1_dt = -r1_used
        dm2_dt = y_i1 * r1_used - r2_used - r3_used
        dmr_dt = y_c1 * r1_used + y_c2 * r2_used + r3_used
        resid_i = abs((dm1_dt + dm2_dt + dmr_dt) + mdot_vol_i)

        mdot_vol[i] = float(max(mdot_vol_i, 0.0))
        r1_arr[i] = float(max(r1_used, 0.0))
        r2_arr[i] = float(max(r2_used, 0.0))
        r3c_arr[i] = float(max(r3_used, 0.0))
        access_arr[i] = float(np.clip(access_i, 0.0, 1.0))
        if np.isfinite(resid_i):
            max_resid = max(max_resid, resid_i)

        if i + 1 < n and np.isfinite(dt) and dt > 0.0:
            m1 = max(m1 + dm1_dt * dt, 0.0)
            m2 = max(m2 + dm2_dt * dt, 0.0)
            mr = max(mr + dmr_dt * dt, 0.0)

    return _FuelStatePyrolysisTrace(
        m_dot_vol_kg_m2_s=mdot_vol,
        m_remaining_total_kg_m2=m_total,
        m_stage1_kg_m2=m1_arr,
        m_stage2_kg_m2=m2_arr,
        m_residue_kg_m2=mr_arr,
        mdot_step1_kg_m2_s=r1_arr,
        mdot_step2_kg_m2_s=r2_arr,
        mdot_char_sink_kg_m2_s=r3c_arr,
        mass_balance_max_residual_kg_m2_s=float(max_resid),
        sequential_source_id=SEMI_GLOBAL_YIELD_SOURCE_ID,
        reactive_access_mode=_reactive_access_mode_token(fuel_cfg),
        reactive_access_source_id=(REACTIVE_ACCESS_SOURCE_ID if _reactive_access_mode_token(fuel_cfg) != "none" else None),
        access_factor_stage2=access_arr,
    )


def _pyrolysis_from_fuel_state_semi_global_zonal(
    t: np.ndarray,
    T1: np.ndarray,
    T_mid: np.ndarray,
    T_deep: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
) -> _FuelStatePyrolysisTrace:
    n = min(int(t.size), int(T1.size), int(T_mid.size), int(T_deep.size))
    mdot_vol = np.zeros(n, dtype=float)
    m_total = np.zeros(n, dtype=float)
    m1_arr = np.zeros(n, dtype=float)
    m2_arr = np.zeros(n, dtype=float)
    mr_arr = np.zeros(n, dtype=float)
    r1_arr = np.zeros(n, dtype=float)
    r2_arr = np.zeros(n, dtype=float)
    r3c_arr = np.zeros(n, dtype=float)
    access_arr = np.ones(n, dtype=float)
    max_resid = 0.0
    if n == 0:
        return _FuelStatePyrolysisTrace(m_dot_vol_kg_m2_s=mdot_vol, m_remaining_total_kg_m2=m_total)

    f1, f2, f3 = _thermal_capacity_fracs_three(fuel_cfg)
    z_fr = np.asarray([f1, f2, f3], dtype=float)
    z_edges = np.concatenate(([0.0], np.cumsum(z_fr)))
    z_depth_weights = np.clip(0.5 * (z_edges[:-1] + z_edges[1:]), 0.0, 1.0)
    m0 = max(float(m0_fuel_kg_m2), 0.0)
    m1z = z_fr * (max(float(getattr(fuel_cfg, "seq_m1_frac", 1.0)), 0.0) * m0)
    m2z = z_fr * (max(float(getattr(fuel_cfg, "seq_m2_frac0", 0.0)), 0.0) * m0)
    mrz = z_fr * (max(float(getattr(fuel_cfg, "seq_mr_frac0", 0.0)), 0.0) * m0)
    m0z = np.maximum(z_fr * m0, 1.0e-12)  # initial mass per zone for alpha computation
    alpha1_arr = np.zeros(n, dtype=float)
    alpha2_arr = np.zeros(n, dtype=float)
    alpha3_arr = np.zeros(n, dtype=float)

    for i in range(n):
        m1z = np.maximum(np.asarray(m1z, dtype=float), 0.0)
        m2z = np.maximum(np.asarray(m2z, dtype=float), 0.0)
        mrz = np.maximum(np.asarray(mrz, dtype=float), 0.0)
        m1_arr[i] = float(np.sum(m1z))
        m2_arr[i] = float(np.sum(m2z))
        mr_arr[i] = float(np.sum(mrz))
        m_total[i] = m1_arr[i] + m2_arr[i] + mr_arr[i]
        # Per-zone char fractions: residue mass / initial zone mass
        alpha1_arr[i] = float(np.clip(mrz[0] / m0z[0], 0.0, 1.0))
        alpha2_arr[i] = float(np.clip(mrz[1] / m0z[1], 0.0, 1.0))
        alpha3_arr[i] = float(np.clip(mrz[2] / m0z[2], 0.0, 1.0))

        dt = float(t[i + 1] - t[i]) if (i + 1 < n) else float("nan")
        Tz = (float(T1[i]), float(T_mid[i]), float(T_deep[i]))
        r1_sum = 0.0
        r2_sum = 0.0
        r3_sum = 0.0
        mdot_sum = 0.0
        resid_sum = 0.0
        access_num = 0.0
        access_den = 0.0
        for z in range(3):
            m1_i = float(max(m1z[z], 0.0))
            m2_i = float(max(m2z[z], 0.0))
            mr_i = float(max(mrz[z], 0.0))
            terms = compute_semi_global_seq_yield_rates(Tz[z], m1_i, m2_i, fuel_cfg)
            y_g1 = float(terms.get("sg_y_g1", 0.0))
            y_i1 = float(terms.get("sg_y_i1", 0.0))
            y_c1 = float(terms.get("sg_y_c1", 0.0))
            y_g2 = float(terms.get("sg_y_g2", 0.0))
            y_c2 = float(terms.get("sg_y_c2", 0.0))
            r1_raw = float(max(terms.get("r1_kg_m2_s", 0.0), 0.0))
            r2_raw = float(max(terms.get("r2_kg_m2_s", 0.0), 0.0))
            r3_raw = float(max(terms.get("r3_char_kg_m2_s", 0.0), 0.0))

            if np.isfinite(dt) and dt > 0.0:
                r1_cap = m1_i / dt
                r1_used = min(r1_raw, max(r1_cap, 0.0))
                r23_cap = max(m2_i / dt, 0.0)
                r23_raw = max(r2_raw + r3_raw, 0.0)
                if r23_raw > r23_cap and r23_raw > 0.0:
                    scale23 = r23_cap / r23_raw
                    r2_used = r2_raw * scale23
                    r3_used = r3_raw * scale23
                else:
                    r2_used = r2_raw
                    r3_used = r3_raw
            else:
                r1_used = r1_raw
                r2_used = r2_raw
                r3_used = r3_raw

            total_condensed_z = max(m1_i + m2_i + mr_i, 0.0)
            residue_frac_z = (mr_i / total_condensed_z) if total_condensed_z > 0.0 else 0.0
            r2_used, r3_used, access_z = _apply_delayed_access_to_rates(
                fuel_cfg,
                r2_used=r2_used,
                r3_used=r3_used,
                residue_frac=residue_frac_z,
                depth_weight=float(z_depth_weights[z]),
            )

            mdot_vol_i = y_g1 * r1_used + y_g2 * r2_used
            dm1_dt = -r1_used
            dm2_dt = y_i1 * r1_used - r2_used - r3_used
            dmr_dt = y_c1 * r1_used + y_c2 * r2_used + r3_used
            resid_i = abs((dm1_dt + dm2_dt + dmr_dt) + mdot_vol_i)

            r1_sum += float(max(r1_used, 0.0))
            r2_sum += float(max(r2_used, 0.0))
            r3_sum += float(max(r3_used, 0.0))
            mdot_sum += float(max(mdot_vol_i, 0.0))
            resid_sum += float(resid_i if np.isfinite(resid_i) else 0.0)
            w_access = max(float(m2_i), 0.0)
            access_num += float(access_z) * w_access
            access_den += w_access

            if i + 1 < n and np.isfinite(dt) and dt > 0.0:
                m1z[z] = max(m1_i + dm1_dt * dt, 0.0)
                m2z[z] = max(m2_i + dm2_dt * dt, 0.0)
                mrz[z] = max(mr_i + dmr_dt * dt, 0.0)

        mdot_vol[i] = mdot_sum
        r1_arr[i] = r1_sum
        r2_arr[i] = r2_sum
        r3c_arr[i] = r3_sum
        if access_den > 0.0:
            access_arr[i] = float(np.clip(access_num / access_den, 0.0, 1.0))
        else:
            access_arr[i] = 1.0
        max_resid = max(max_resid, resid_sum)

    return _FuelStatePyrolysisTrace(
        m_dot_vol_kg_m2_s=mdot_vol,
        m_remaining_total_kg_m2=m_total,
        m_stage1_kg_m2=m1_arr,
        m_stage2_kg_m2=m2_arr,
        m_residue_kg_m2=mr_arr,
        mdot_step1_kg_m2_s=r1_arr,
        mdot_step2_kg_m2_s=r2_arr,
        mdot_char_sink_kg_m2_s=r3c_arr,
        mass_balance_max_residual_kg_m2_s=float(max_resid),
        sequential_source_id=SEMI_GLOBAL_YIELD_SOURCE_ID,
        reactive_access_mode=_reactive_access_mode_token(fuel_cfg),
        reactive_access_source_id=(REACTIVE_ACCESS_SOURCE_ID if _reactive_access_mode_token(fuel_cfg) != "none" else None),
        access_factor_stage2=access_arr,
        alpha_zone1=alpha1_arr,
        alpha_zone2=alpha2_arr,
        alpha_zone3=alpha3_arr,
    )


def _pyrolysis_from_fuel_state(
    t: np.ndarray,
    T1: np.ndarray,
    m0_fuel_kg_m2: float,
    fuel_cfg,
    T_mid: Optional[np.ndarray] = None,
    T_deep: Optional[np.ndarray] = None,
    thermal_node_order: int = 2,
) -> _FuelStatePyrolysisTrace:
    mode = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
    chem_depth_mode = _chem_depth_partition_mode_token(fuel_cfg)
    use_zonal = (
        chem_depth_mode == "thermal_nodes"
        and int(thermal_node_order or 2) == 3
        and T_mid is not None
        and T_deep is not None
        and mode in {"two_step_sequential", "semi_global_seq_yield"}
    )
    if chem_depth_mode == "thermal_nodes" and int(thermal_node_order or 2) == 3 and (T_mid is None or T_deep is None):
        raise ValueError("chem_depth_partition_mode='thermal_nodes' requires T_mid and T_deep arrays in 3-node mode")
    if mode == "two_step_sequential":
        if use_zonal:
            return _pyrolysis_from_fuel_state_sequential_zonal(
                t=t,
                T1=T1,
                T_mid=np.asarray(T_mid, dtype=float),
                T_deep=np.asarray(T_deep, dtype=float),
                m0_fuel_kg_m2=m0_fuel_kg_m2,
                fuel_cfg=fuel_cfg,
            )
        return _pyrolysis_from_fuel_state_sequential(t=t, T1=T1, m0_fuel_kg_m2=m0_fuel_kg_m2, fuel_cfg=fuel_cfg)
    if mode == "semi_global_seq_yield":
        if use_zonal:
            return _pyrolysis_from_fuel_state_semi_global_zonal(
                t=t,
                T1=T1,
                T_mid=np.asarray(T_mid, dtype=float),
                T_deep=np.asarray(T_deep, dtype=float),
                m0_fuel_kg_m2=m0_fuel_kg_m2,
                fuel_cfg=fuel_cfg,
            )
        return _pyrolysis_from_fuel_state_semi_global(t=t, T1=T1, m0_fuel_kg_m2=m0_fuel_kg_m2, fuel_cfg=fuel_cfg)
    return _pyrolysis_from_fuel_state_singlepool(t=t, T1=T1, m0_fuel_kg_m2=m0_fuel_kg_m2, fuel_cfg=fuel_cfg)


def _convert_q_in_strict(value: float, units: str) -> float:
    u = (units or "").strip().lower().replace(" ", "")
    if "kw" in u and "/m2" in u:
        return float(value) * 1000.0
    if "w" in u and "/m2" in u and "kw" not in u:
        return float(value)
    raise ValueError(f"Unsupported q_in_units='{units}'. Use 'kW/m2' or 'W/m2'.")


def _debug_guardrail_enabled() -> bool:
    env = os.environ.get("ROM_DEBUG_QIN", "").strip().lower()
    return env in {"1", "true", "yes", "on"} or ("PYTEST_CURRENT_TEST" in os.environ)


def _write_qin_guardrail_error(case_id: Optional[str], message: str) -> None:
    root = Path("test_debug")
    case_tag = case_id or "run_rom"
    out_dir = root / case_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qin_guardrail_error.txt"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


# ── Diagnostic CSV writers ────────────────────────────────────────────────────
# Used by tests and compare_exp.py to export full time-series breakdowns.

def write_rom_heat_terms_csv(path: Path, rom: RomSignals) -> None:
    t = np.asarray(getattr(rom, "t", np.array([])), dtype=float)
    q_in = np.asarray(getattr(rom, "q_in_W_m2", np.array([])), dtype=float)
    q_conv = np.asarray(getattr(rom, "q_conv_W_m2", np.array([])), dtype=float)
    q_rad = np.asarray(getattr(rom, "q_rad_W_m2", np.array([])), dtype=float)
    q_net = np.asarray(getattr(rom, "q_net_surface_W_m2", np.array([])), dtype=float)
    T1 = np.asarray(getattr(rom, "T_surf", np.array([])), dtype=float)
    T2 = np.asarray(getattr(rom, "T_inner", np.array([])), dtype=float)
    mdot = np.asarray(getattr(rom, "m_py", np.array([])), dtype=float)
    hrr = np.asarray(
        getattr(rom, "hrrpua_diag", None) if getattr(rom, "hrrpua_diag", None) is not None else getattr(rom, "hrrpua", np.array([])),
        dtype=float,
    )
    q_source = getattr(rom, "q_in_source", None)
    if q_source is None:
        q_source = ["none"] * int(t.size)

    n = int(t.size)
    if not (
        n > 0
        and q_in.size == n
        and q_conv.size == n
        and q_rad.size == n
        and q_net.size == n
        and T1.size == n
        and T2.size == n
        and mdot.size == n
        and hrr.size == n
        and len(q_source) == n
    ):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Time",
                "q_in_W_m2",
                "q_in_source",
                "q_conv_W_m2",
                "q_rad_W_m2",
                "q_net_surface_W_m2",
                "T1_K",
                "T2_K",
                "mdot_py_kg_m2_s",
                "hrrpua_diag_kW_m2",
            ]
        )
        for i in range(n):
            writer.writerow(
                [
                    f"{float(t[i]):.12g}",
                    f"{float(q_in[i]):.12g}",
                    str(q_source[i]),
                    f"{float(q_conv[i]):.12g}",
                    f"{float(q_rad[i]):.12g}",
                    f"{float(q_net[i]):.12g}",
                    f"{float(T1[i]):.12g}",
                    f"{float(T2[i]):.12g}",
                    f"{float(mdot[i]):.12g}",
                    f"{float(hrr[i]):.12g}",
                ]
            )


def write_pyrolysis_terms_csv(path: Path, rom: RomSignals) -> None:
    t = np.asarray(getattr(rom, "t", np.array([])), dtype=float)
    T1 = np.asarray(getattr(rom, "T_surf", np.array([])), dtype=float)
    m_rem = np.asarray(getattr(rom, "pyro_m_remaining_kg_m2", np.array([])), dtype=float)
    mdot_kin = np.asarray(getattr(rom, "pyro_mdot_kin_kg_m2_s", np.array([])), dtype=float)
    mdot_cap = np.asarray(getattr(rom, "pyro_mdot_cap_kg_m2_s", np.array([])), dtype=float)
    mdot_limit = np.asarray(getattr(rom, "pyro_mdot_limit_kg_m2_s", np.array([])), dtype=float)
    mdot_final = np.asarray(getattr(rom, "pyro_mdot_final_kg_m2_s", np.array([])), dtype=float)
    limiter = np.asarray(getattr(rom, "pyro_limiter_active", np.array([])), dtype=float)
    cap = np.asarray(getattr(rom, "pyro_cap_active", np.array([])), dtype=float)
    gate = np.asarray(getattr(rom, "pyro_kinetics_gate_active", np.array([])), dtype=float)
    gate_factor = np.asarray(getattr(rom, "pyro_gate_factor", np.array([])), dtype=float)
    seq_m1 = getattr(rom, "m_seq_stage1_kg_m2", None)
    seq_m2 = getattr(rom, "m_seq_stage2_kg_m2", None)
    seq_mr = getattr(rom, "m_seq_residue_kg_m2", None)
    seq_mrct = getattr(rom, "m_seq_reactive_total_kg_m2", None)
    seq_r1 = getattr(rom, "mdot_seq_step1_kg_m2_s", None)
    seq_r2 = getattr(rom, "mdot_seq_step2_kg_m2_s", None)
    seq_r3c = getattr(rom, "mdot_seq_char_sink_kg_m2_s", None)
    seq_vol = getattr(rom, "mdot_seq_vol_kg_m2_s", None)
    seq_access = getattr(rom, "access_factor_stage2", None)

    n = int(t.size)
    if not (
        n > 0
        and T1.size == n
        and m_rem.size == n
        and mdot_kin.size == n
        and mdot_cap.size == n
        and mdot_limit.size == n
        and mdot_final.size == n
        and limiter.size == n
        and cap.size == n
        and gate.size == n
        and gate_factor.size == n
    ):
        return

    seq_arrays: list[tuple[str, np.ndarray]] = []
    for name, arr in (
        ("m_seq_stage1_kg_m2", seq_m1),
        ("m_seq_stage2_kg_m2", seq_m2),
        ("m_seq_residue_kg_m2", seq_mr),
        ("m_seq_reactive_total_kg_m2", seq_mrct),
        ("mdot_seq_step1_kg_m2_s", seq_r1),
        ("mdot_seq_step2_kg_m2_s", seq_r2),
        ("mdot_seq_char_sink_kg_m2_s", seq_r3c),
        ("mdot_seq_vol_kg_m2_s", seq_vol),
        ("access_factor_stage2", seq_access),
    ):
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size == n:
            seq_arrays.append((name, arr_np))

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Time",
        "T1_K",
        "m_remaining_kg_m2",
        "mdot_kin_kg_m2_s",
        "mdot_cap_kg_m2_s",
        "mdot_limit_kg_m2_s",
        "mdot_final_kg_m2_s",
        "limiter_active",
        "cap_active",
        "kinetics_gate_active",
        "gate_factor",
    ]
    cols = [t, T1, m_rem, mdot_kin, mdot_cap, mdot_limit, mdot_final, limiter, cap, gate, gate_factor]
    for name, arr in seq_arrays:
        headers.append(name)
        cols.append(arr)
    np.savetxt(
        path,
        np.column_stack(cols),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


def _load_rom_inputs(case_id: Optional[str] = None) -> Optional[RomInputs]:
    file_path = os.environ.get("ROM_INPUT_FILE", "").strip()
    if file_path:
        p = Path(file_path)
        if p.exists() and p.is_file():
            return load_text_input(p)

    dir_path = os.environ.get("ROM_INPUT_DIR", "").strip()
    if dir_path and case_id:
        d = Path(dir_path)
        if d.exists() and d.is_dir():
            ids = [case_id]
            if "__" in case_id:
                base = case_id.split("__", 1)[0]
                if base not in ids:
                    ids.append(base)
            for cid in ids:
                for suffix in (".txt", ".ini", ".cfg"):
                    candidate = d / f"{cid}{suffix}"
                    if candidate.exists():
                        return load_text_input(candidate)
    return None


# ── MoL solver dispatch ───────────────────────────────────────────────────────

def _run_mol_impl(
    rom_inputs: "RomInputs",
    q_in_kW_m2: float,
    t_end_s: float,
    area_m2: float,
    Tamb_K: float,
    hoc_eff_j_kg: float,
    hoc_eff_kj_kg: float,
    registry_exposure_q_kW_m2: Optional[float],
) -> RomSignals:
    """Dispatch to the method-of-lines 1D pyrolysis solver.

    Called from run_rom() when rom_inputs.mol_enable is True.
    Returns a RomSignals object compatible with all downstream analysis code.
    Non-applicable fields (staged kinetics, Stefan diagnostics, etc.) are None.
    """
    from model.fuel.mol_pyrolysis import (  # noqa: PLC0415
        MolParams, MolSolidSpecies, MolReaction, integrate_mol,
        build_single_species_params,
    )

    ri = rom_inputs
    N = int(ri.mol_n_cells or 20)
    n_species = int(ri.mol_n_species or 1)

    # ── Material properties ───────────────────────────────────────────────────
    L = float(ri.thickness_m or 0.038)
    rho = float(ri.density or 400.0)
    cp = float(ri.cp or 1500.0)
    k = float(ri.k or 0.12)
    eps = float(ri.eps or 0.87)
    rho_char = float(ri.rho_char) if ri.rho_char is not None else 130.0
    cp_char = float(ri.cp_char or 1100.0)
    k_char = float(ri.k_char or 0.08)
    dH_py = float(ri.dH_py) if ri.dH_py is not None else 1.8e6  # J/kg gas (Lautenberger dH_vol)
    h_conv = float(ri.mol_h_conv or 15.0)
    T0_K = float(ri.T1 or Tamb_K)
    T_sur = float(ri.T_sur or Tamb_K)

    # ── Back-face BC ─────────────────────────────────────────────────────────
    back_bc = "adiabatic"
    if ri.mol_back_bc in ("open",):
        back_bc = "open"
    elif ri.back_bc_mode in ("open",):
        back_bc = "open"
    h_back = float(ri.h_open or 10.0)
    eps_back = float(ri.eps_open or 0.87)

    # ── Flame coupling parameters ─────────────────────────────────────────────
    fov = ri.fuel_overrides
    flame_enable = bool(fov.get("flame_enable", False))
    n_passes = int(float(fov.get("flame_coupling_passes", 3)))
    tau_growth = float(fov.get("flame_tau_growth_s", 20.0))
    flame_cfg = None
    fuel_viability = None
    if flame_enable:
        try:
            from model.flame import FlameConfig, SolidFuelViability  # noqa: PLC0415
            flame_cfg = FlameConfig(
                chi_rad=float(fov.get("flame_chi_rad", 0.35)),
                view_factor=float(fov.get("flame_view_factor", 0.40)),
                persistence_s=float(fov.get("flame_persistence_s", 5.0)),
            )
            fuel_viability = SolidFuelViability(
                m_py_ignite=float(fov.get("flame_m_py_ignite", 0.005)),
                T_ignite=float(fov.get("flame_T_ignite", 600.0)),
                m_py_crit=float(fov.get("flame_m_py_crit", 0.001)),
                T_py=float(fov.get("flame_T_py", 500.0)),
            )
        except ImportError:
            flame_enable = False

    # ── Solver settings ───────────────────────────────────────────────────────
    method = str(ri.method or "Radau")
    max_step = float(ri.sim_overrides.get("max_step", 0.5))
    dt_eval = float(ri.sim_overrides.get("dt_eval", 0.5))

    # ── Build MolParams (Lautenberger multi-species framework) ───────────────
    # Map legacy deck format to MolSolidSpecies + MolReaction.
    # dH_py in decks = dH_vol (J per kg of gas produced) per Lautenberger conv.
    if n_species == 1:
        # Single charring reaction: virgin → char + gas
        # nu_gas derived from density ratio; char starts at rho0=0 and accumulates
        nu_gas_1 = (rho - rho_char) / max(rho, 1e-6)
        A_py = float(ri.A_py or 3.1e11)
        E_py = float(ri.E_py or 1.62e5)
        n_rxn = float(ri.sim_overrides.get("n_rxn", 1.0))
        virgin_sp = MolSolidSpecies(name="virgin", k0=k, rho0=rho, cp0=cp, eps=eps)
        char_sp   = MolSolidSpecies(name="char",   k0=k_char, rho0=0.0, cp0=cp_char)
        rxn0      = MolReaction(from_idx=0, to_idx=1, Z=A_py, E=E_py, n=n_rxn,
                                dH_vol=dH_py, nu_gas=nu_gas_1)
        species_list   = [virgin_sp, char_sp]
        reactions_list = [rxn0]

    elif n_species == 2:
        # Two independent reactions on separate mass fractions of the solid.
        # Species 0: resin/hemicellulose component (m_frac_1, nu_gas_1)
        # Species 1: cellulosic bulk component (m_frac_2, nu_gas_2)
        # Species 2: char (shared product, rho0=0)
        mf1  = float(ri.seq_m1_frac  or 0.10)
        mf2  = float(ri.seq_m2_frac0 or 0.90)
        ng1  = float(ri.seq_y1_vol   or 1.0)
        ng2  = float(ri.seq_y2_vol   or 0.15)
        A1   = float(ri.A1_py or 1.03e7)
        E1   = float(ri.E1_py or 9.0e4)
        A2   = float(ri.A2_py or 3.1e4)
        E2   = float(ri.E2_py or 1.62e5)
        n_rxn = float(ri.sim_overrides.get("n_rxn", 1.0))
        # Initial bulk densities: split total virgin density by mass fraction
        rho_s0 = rho * mf1   # species 0 initial density
        rho_s1 = rho * mf2   # species 1 initial density
        sp0  = MolSolidSpecies(name="resin_hemi", k0=k, rho0=rho_s0, cp0=cp, eps=eps)
        sp1  = MolSolidSpecies(name="cellulose",  k0=k, rho0=rho_s1, cp0=cp, eps=eps)
        sp2  = MolSolidSpecies(name="char",        k0=k_char, rho0=0.0, cp0=cp_char)
        rxn0 = MolReaction(from_idx=0, to_idx=2, Z=A1, E=E1, n=n_rxn,
                           dH_vol=dH_py, nu_gas=ng1)
        rxn1 = MolReaction(from_idx=1, to_idx=2, Z=A2, E=E2, n=n_rxn,
                           dH_vol=dH_py, nu_gas=ng2)
        species_list   = [sp0, sp1, sp2]
        reactions_list = [rxn0, rxn1]

    elif ri.mol_species_list:
        # Lautenberger-style: species and reactions fully specified in deck via
        # mol.species.N.{name,k0,nk,rho0,cp0,nc,eps,gamma}
        # mol.reaction.N.{from,to,Z,E,n,dH_vol,nu_gas,dH_sol,nO2}
        species_list = []
        for sp_d in ri.mol_species_list:
            # Keys are lowercase (deck keys lowercased during parsing)
            species_list.append(MolSolidSpecies(
                name=str(sp_d.get("name", "species")),
                k0=float(sp_d.get("k0", 0.12)),
                nk=float(sp_d.get("nk", 0.0)),
                rho0=float(sp_d.get("rho0", 0.0)),
                cp0=float(sp_d.get("cp0", 1500.0)),
                nc=float(sp_d.get("nc", 0.0)),
                eps=float(sp_d.get("eps", 0.87)),
                gamma=float(sp_d.get("gamma", 0.0)),
            ))
        reactions_list = []
        for rxn_d in ri.mol_reactions_list:
            # Keys are lowercase; dH_vol → "dh_vol", Z → "z", E → "e"
            reactions_list.append(MolReaction(
                from_idx=int(rxn_d.get("from", 0)),
                to_idx=int(rxn_d.get("to", 1)),
                Z=float(rxn_d.get("z", 1.0e10)),
                E=float(rxn_d.get("e", 1.62e5)),
                n=float(rxn_d.get("n", 1.0)),
                dH_vol=float(rxn_d.get("dh_vol", 0.0)),
                nu_gas=float(rxn_d.get("nu_gas", 0.0)),
                nO2=float(rxn_d.get("no2", 0.0)),
                dH_sol=float(rxn_d.get("dh_sol", 0.0)),
                rho_ref=float(rxn_d.get("rho_ref", 1.0)),
                T_gate_K=float(rxn_d.get("t_gate_k", 0.0)),
                dT_gate_K=float(rxn_d.get("dt_gate_k", 25.0)),
            ))
        # Use rho from first non-zero species as the "bulk density" for output diagnostics
        rho = float(next((sp.rho0 for sp in species_list if sp.rho0 > 0), rho))

    else:
        raise ValueError(f"mol_n_species={n_species} not supported; use 1 or 2")

    grid_stretch = float(ri.mol_grid_stretch or 1.0)
    k_crack_frac = float(ri.mol_k_crack_frac or 0.0)
    surface_recession_enable = bool(ri.mol_surface_recession_enable)
    surface_rho_floor = float(ri.mol_surface_rho_floor or 0.01)
    spall_enable = bool(ri.mol_spall_enable)
    spall_depth_m = float(ri.mol_spall_depth_m or 0.003)
    spall_rho_floor = float(ri.mol_spall_rho_floor or 0.05)
    cp_gas = float(ri.mol_cp_gas or 0.0)
    y_o2_surf = float(ri.mol_surface_y_o2 if ri.mol_surface_y_o2 is not None else 0.21)
    material_coords = bool(ri.mol_material_coords)
    lagrangian_mode = bool(ri.mol_lagrangian_mode)
    charring_front_bc = bool(ri.mol_charring_front_bc)
    charring_T_py = float(ri.mol_charring_T_py)
    surface_ablation_bc = bool(ri.mol_surface_ablation_bc)
    surface_ablation_L_py = float(ri.mol_surface_ablation_L_py)
    surface_ablation_T_min = float(ri.mol_surface_ablation_T_min)
    # Bed collapse: read from fuel_overrides (set via fuel.bed_collapse_enable = true in deck)
    bed_collapse_enable = bool(ri.fuel_overrides.get("bed_collapse_enable", False))
    in_depth_rad_kappa = float(ri.mol_in_depth_rad_kappa or 0.0)
    in_depth_rad_density_weighted = bool(ri.mol_in_depth_rad_density_weighted)

    params = MolParams(
        L=L, N=N,
        species=species_list,
        reactions=reactions_list,
        Tamb=Tamb_K, T_sur=T_sur, h_conv=h_conv,
        back_bc=back_bc, h_back=h_back, eps_back=eps_back,
        method=method, rtol=1e-4, atol=1e-6,
        max_step=max_step, dt_eval=dt_eval,
        grid_stretch=grid_stretch, k_crack_frac=k_crack_frac,
        surface_recession_enable=surface_recession_enable,
        surface_rho_floor=surface_rho_floor,
        spall_enable=spall_enable,
        spall_depth_m=spall_depth_m,
        spall_rho_floor=spall_rho_floor,
        cp_gas=cp_gas,
        y_o2_surf=y_o2_surf,
        material_coords=material_coords,
        lagrangian_mode=lagrangian_mode,
        charring_front_bc=charring_front_bc,
        charring_T_py=charring_T_py,
        surface_ablation_bc=surface_ablation_bc,
        surface_ablation_L_py=surface_ablation_L_py,
        surface_ablation_T_min=surface_ablation_T_min,
        bed_collapse_enable=bed_collapse_enable,
        in_depth_rad_kappa=in_depth_rad_kappa,
        in_depth_rad_density_weighted=in_depth_rad_density_weighted,
    )

    q_in_W_m2 = float(q_in_kW_m2) * 1000.0

    def q_incident_fn(t_: float) -> float:
        return q_in_W_m2

    # ── Run MoL integration ───────────────────────────────────────────────────
    result = integrate_mol(
        params=params,
        t_span=(0.0, t_end_s),
        T0_K=T0_K,
        q_incident_fn=q_incident_fn,
        hoc_eff_J_kg=hoc_eff_j_kg,
        flame_enable=flame_enable,
        flame_cfg=flame_cfg,
        fuel_viability=fuel_viability,
        n_passes=n_passes,
        tau_growth_s=tau_growth,
    )

    # ── Char oxidation post-processing (MoL) ─────────────────────────────────
    # Physics: char layer oxidizes when volatile blow-off is low (volatile
    # flow suppresses O2 access to char surface).
    # Formula (Di Blasi 2002; Tran & White 1992):
    #   q_char_ox = alpha_bar × q_char_ref × f_blow
    #   f_blow = max(1 - m_dot / m_py_stefan0, 0)
    # Applies only when flame is enabled (requires combustion atmosphere).
    char_ox_addend = np.zeros(len(result.t))
    if bool(ri.mol_char_ox_enable) and flame_enable:
        _q_char_ref = float(ri.mol_char_ox_q_ref_W_m2)    # [W/m²]
        _m_py_s0 = float(ri.mol_char_ox_m_py_stefan0)     # [kg/m²/s]
        # Total initial density [kg/m²]
        rho0_slab = sum(sp.rho0 * params.L for sp in params.species)
        # Total density at each solver step [kg/m²]
        dx_arr_p = params.dx_arr  # (N,)
        M_sp = len(params.species)
        n_t = len(result.t)
        rho_slab = np.zeros(n_t)
        for s in range(M_sp):
            rho_slab += np.einsum("it,i->t", result.rho[s], dx_arr_p)
        if rho0_slab > 1e-6:
            alpha_bar_arr = np.clip(1.0 - rho_slab / rho0_slab, 0.0, 1.0)
        else:
            alpha_bar_arr = np.zeros(n_t)
        if _m_py_s0 > 0.0:
            f_blow = np.maximum(1.0 - result.m_dot / _m_py_s0, 0.0)
        else:
            f_blow = np.ones(n_t)
        char_ox_addend = alpha_bar_arr * _q_char_ref * f_blow / 1000.0  # kW/m²

    # ── Interpolate to uniform 1s output grid ─────────────────────────────────
    dt_out = 1.0
    t_out = np.arange(0.0, t_end_s + dt_out * 0.5, dt_out)
    hrrpua_base = np.interp(t_out, result.t, result.hrrpua_kW)
    char_ox_out = np.interp(t_out, result.t, char_ox_addend)
    hrrpua_out = hrrpua_base + char_ox_out
    m_py_out = np.interp(t_out, result.t, result.m_dot)
    T_surf_out = np.interp(t_out, result.t, result.T_surf)
    mass_total_out = np.cumsum(m_py_out) * dt_out * area_m2

    # Build q_in diagnostic arrays (constant incident flux)
    q_in_applied = np.full(t_out.size, q_in_W_m2)

    return RomSignals(
        t=t_out,
        T_surf=T_surf_out,
        T_inner=None,
        M1_moisture=None,
        hrrpua=hrrpua_out,
        hrrpua_diag=None,
        m_py=m_py_out,
        mlr=m_py_out * area_m2,
        mass_total=mass_total_out,
        mass_total_units="kg_cumulative",
        m_fuel_remaining_kg_m2=None,
        pyrolysis_mass_source="mol_1d",
        sequential_kinetics_enabled=False,
        time_grid_mode="uniform",
        dt_out=dt_out,
        t_solver=result.t,
        m_py_solver=result.m_dot,
        hrrpua_diag_solver=None,
        hoc_eff_raw=float(hoc_eff_kj_kg),
        hoc_units="kJ/kg",
        hoc_eff_J_kg=float(hoc_eff_j_kg),
        hoc_eff=float(hoc_eff_kj_kg),
        q_in_incident_W_m2=q_in_applied,
        q_net_into_surface_W_m2=q_in_applied,
        q_in_W_m2=q_in_applied,
        q_net_surface_W_m2=q_in_applied,
        q_in_mode="incident",
        q_in_value_W_m2=float(q_in_W_m2),
        h_amb_model="mol_1d",
        registry_exposure_q_kW_m2=registry_exposure_q_kW_m2,
        area_m2_used=float(area_m2),
        thickness_m_used=float(L),
        rho_kg_m3_used=float(rho),
        T_nodes=[np.interp(t_out, result.t, result.T[i, :]) for i in range(N)],
        alpha_nodes=None,
    )


# ── Main integration function ─────────────────────────────────────────────────
# run_rom() is the single public entry point for running a ROM case.
#
# Internal phases:
#   1. Parse / validate inputs (q_in, hoc_eff, geometry, material properties)
#   2. Build FuelConfig, EnvConfig, SimConfig from RomInputs + overrides
#   3. Construct forcing callables (q_in schedule or constant, ramp envelope)
#   4. Initialize ODE state vector y0
#   5. Call integrate_fuel() → solve_ivp result
#   6. Post-process: re-evaluate m_dot on output grid
#   7. Compute HRRPUA (m_dot × hoc_eff), apply char-ox addend if enabled
#   8. Apply flame radiation feedback (2-pass if flame_enable=True)
#   9. Assemble and return RomSignals

def run_rom(
    q_in_kW_m2: Optional[float],
    t_end_s: float,
    area_m2: float,
    Tamb_K: float,
    M1_init: float,
    hoc_eff: float,
    subcase_token: Optional[str],
    m0_kg: Optional[float] = None,
    rom_inputs: Optional[RomInputs] = None,
    case_id: Optional[str] = None,
    start_delay_s: float = 0.0,
) -> RomSignals:
    registry_exposure_q_kW_m2 = float(q_in_kW_m2) if q_in_kW_m2 is not None else None
    flux = q_in_kW_m2
    if flux is None:
        flux = _parse_flux_from_token(subcase_token)
        if flux is None:
            flux = 50.0
    q_in_W_m2 = flux * 1.0e3

    if rom_inputs is None:
        rom_inputs = _load_rom_inputs(case_id)

    # ── MoL dispatch ─────────────────────────────────────────────────────────
    if rom_inputs is not None and getattr(rom_inputs, "mol_enable", False):
        hoc_units_local = normalize_hoc_units(getattr(rom_inputs, "hoc_units", "kJ/kg"))
        hoc_eff_raw_ = float(getattr(rom_inputs, "hoc_eff", None) or hoc_eff)
        hoc_eff_j_kg_ = hoc_eff_to_j_per_kg(hoc_eff_raw_, hoc_units_local) or max(float(hoc_eff), 1e-12) * 1000.0
        hoc_eff_kj_kg_ = max(hoc_eff_j_kg_ / 1000.0, 1e-12)
        _area = float(getattr(rom_inputs, "area_m2", None) or area_m2)
        _Tamb = float(getattr(rom_inputs, "Tamb", None) or Tamb_K)
        _q_kw = float(q_in_kW_m2) if q_in_kW_m2 is not None else float(flux)
        return _run_mol_impl(
            rom_inputs=rom_inputs,
            q_in_kW_m2=_q_kw,
            t_end_s=t_end_s,
            area_m2=_area,
            Tamb_K=_Tamb,
            hoc_eff_j_kg=hoc_eff_j_kg_,
            hoc_eff_kj_kg=hoc_eff_kj_kg_,
            registry_exposure_q_kW_m2=registry_exposure_q_kW_m2,
        )

    fuel_cfg = default_fuel_config()
    env_cfg = default_env_config()
    sim_cfg = default_sim_config()

    hoc_units_local = "kJ/kg"
    hoc_eff_raw = float(hoc_eff)
    if rom_inputs and rom_inputs.hoc_units:
        hoc_units_local = normalize_hoc_units(rom_inputs.hoc_units)
    if rom_inputs and rom_inputs.hoc_eff is not None:
        hoc_eff_raw = float(rom_inputs.hoc_eff)

    hoc_eff_j_kg = hoc_eff_to_j_per_kg(hoc_eff_raw, hoc_units_local)
    if hoc_eff_j_kg is None and rom_inputs and rom_inputs.hoc_eff_J_kg is not None:
        hoc_eff_j_kg = float(rom_inputs.hoc_eff_J_kg)
    if hoc_eff_j_kg is None:
        hoc_eff_j_kg = max(float(hoc_eff), 1.0e-12) * 1000.0
    hoc_eff_kj_kg = max(float(hoc_eff_to_kj_per_kg(hoc_eff_raw, hoc_units_local) or (hoc_eff_j_kg / 1000.0)), 1.0e-12)

    if rom_inputs:
        # Apply geometry/material-derived values first
        area_m2, L_m = resolve_geometry(rom_inputs, area_m2)
        if L_m is not None:
            fuel_cfg.L_m = L_m
        if rom_inputs.T_sur is not None:
            env_cfg.T_sur = rom_inputs.T_sur
        apply_material_geometry(rom_inputs, fuel_cfg)
        if rom_inputs.pyrolysis_mode:
            fuel_cfg.pyrolysis_mode = rom_inputs.pyrolysis_mode
        if rom_inputs.m_py_schedule:
            schedule = [
                (t, convert_m_py(m, rom_inputs.m_py_units, hoc_eff_raw, hoc_units_local))
                for t, m in rom_inputs.m_py_schedule
            ]
            fuel_cfg.pyrolysis_mode = "prescribed"
            fuel_cfg.m_py_schedule = schedule

        # Then apply explicit overrides
        _apply_overrides(fuel_cfg, rom_inputs.fuel_overrides)
        _apply_overrides(sim_cfg, rom_inputs.sim_overrides)
        if rom_inputs.force_htc_zero:
            fuel_cfg.h_amb = 0.0
            fuel_cfg.C_h_conv = 0.0

    env_cfg.Tamb = rom_inputs.Tamb if rom_inputs and rom_inputs.Tamb is not None else Tamb_K
    t_end_s = rom_inputs.t_end if rom_inputs and rom_inputs.t_end is not None else t_end_s

    T1_init = rom_inputs.T1 if rom_inputs and rom_inputs.T1 is not None else env_cfg.Tamb
    T2_init = rom_inputs.T2 if rom_inputs and rom_inputs.T2 is not None else env_cfg.Tamb
    T3_init = rom_inputs.T3 if rom_inputs and rom_inputs.T3 is not None else T2_init
    M1_init = rom_inputs.M1 if rom_inputs and rom_inputs.M1 is not None else M1_init
    thermal_order_cfg = int(getattr(fuel_cfg, "thermal_model_order", 2) or 2)
    if thermal_order_cfg >= 3:
        # Build [T1, T2, T3, ..., TN, M1]; T3..TN initialized to T3_init (back-face guess)
        _T_init_list = [T1_init, T2_init] + [T3_init] * (thermal_order_cfg - 2) + [M1_init]
        y0 = np.array(_T_init_list, dtype=float)
    else:
        y0 = np.array([T1_init, T2_init, M1_init], dtype=float)

    deck_q_in_units_raw = rom_inputs.q_in_units if rom_inputs is not None else "kW/m2(case_flux)"
    deck_q_in_constant_raw: Optional[float] = None
    cfg_q_in_constant_altkey_raw: Optional[float] = None
    cfg_q_in_constant_source = "unknown"

    q_in_schedule_w_m2: list[tuple[float, float]] = []
    if rom_inputs is not None and rom_inputs.q_in_schedule:
        q_in_schedule_w_m2 = [
            (float(t), max(_convert_q_in_strict(float(q), rom_inputs.q_in_units), 0.0))
            for t, q in rom_inputs.q_in_schedule
        ]

    q_in_constant_cfg_w_m2: Optional[float] = None
    if rom_inputs is not None and rom_inputs.q_in_constant is not None:
        q_in_constant_cfg_w_m2 = max(_convert_q_in_strict(float(rom_inputs.q_in_constant), rom_inputs.q_in_units), 0.0)
        q_const_key = str(getattr(rom_inputs, "q_in_constant_key", "") or "q_in_constant").strip().lower()
        if q_const_key == "q_in_constant":
            deck_q_in_constant_raw = float(rom_inputs.q_in_constant)
            cfg_q_in_constant_source = "deck:q_in_constant"
        else:
            cfg_q_in_constant_altkey_raw = float(rom_inputs.q_in_constant)
            cfg_q_in_constant_source = f"deck:{q_const_key}"
    elif not q_in_schedule_w_m2:
        q_in_constant_cfg_w_m2 = max(float(q_in_W_m2), 0.0)
        if registry_exposure_q_kW_m2 is not None:
            cfg_q_in_constant_source = "case_registry:exposure_id"
        else:
            cfg_q_in_constant_source = "adapter_default"

    q_in_mode_cfg = str(getattr(sim_cfg, "q_in_mode", "incident") or "incident").strip().lower()
    if q_in_mode_cfg not in {"incident", "net"}:
        q_in_mode_cfg = "incident"
    q_in_cfg = {
        "q_in_schedule": q_in_schedule_w_m2,
        "q_in_constant_W_m2": q_in_constant_cfg_w_m2,
        "q_in_mode": q_in_mode_cfg,
    }

    q_in_applied_at_t0, q_in_source_at_t0 = eval_q_in_incident_W_m2(0.0, q_in_cfg)
    qin_guardrail_error: Optional[str] = None
    has_schedule = len(q_in_schedule_w_m2) > 0
    if (
        q_in_constant_cfg_w_m2 is not None
        and q_in_constant_cfg_w_m2 > 0.0
        and not has_schedule
        and q_in_mode_cfg == "incident"
        and q_in_applied_at_t0 <= 0.5 * q_in_constant_cfg_w_m2
    ):
        qin_guardrail_error = (
            "q_in constant specified but applied q_in(t=0)=0; check parsing/branching; "
            f"q_in_source={q_in_source_at_t0}"
        )
        _write_qin_guardrail_error(case_id, f"ERROR: {qin_guardrail_error}")
        if _debug_guardrail_enabled():
            raise ValueError(f"ERROR: {qin_guardrail_error}")

    def q_raw_func(t: float, cfg=q_in_cfg) -> float:
        q_val, _ = eval_q_in_incident_W_m2(float(t), cfg)
        return float(q_val)

    q_func = q_raw_func
    ramp_mode = "none"
    ramp_tau = 1.0

    if sim_cfg.warn_on_initial_temp_offset:
        dT1 = abs(y0[0] - env_cfg.Tamb)
        dT2 = abs(y0[1] - env_cfg.Tamb)
        if dT1 > sim_cfg.initial_temp_warn_K or dT2 > sim_cfg.initial_temp_warn_K:
            print(
                "Warning: initial T1/T2 differ from Tamb by more than "
                f"{sim_cfg.initial_temp_warn_K:.1f} K "
                f"(dT1={dT1:.1f} K, dT2={dT2:.1f} K)."
            )

    forcing = {"q_in_cfg": q_in_cfg, "q_in": q_func, "rewet_rate": lambda t: 0.0, "M1_eq": lambda t: 0.0}

    pyro_mass_source = resolve_pyrolysis_mass_source(fuel_cfg)

    # Resolve total fuel mass (needed in pyrolysis calls inside and after the loop)
    m0_kg_m2_solver = resolve_total_fuel_mass_kg_m2(fuel_cfg)
    if m0_kg is not None and area_m2 > 0.0:
        m0_kg_m2_solver = max(float(m0_kg) / float(area_m2), 0.0)
    elif not np.isfinite(m0_kg_m2_solver) or m0_kg_m2_solver <= 0.0:
        m0_kg_m2_solver = max(float(getattr(fuel_cfg, "m_fuel_kg_m2", 0.0)), 0.0)

    # Staggered coupling loop: thermal ODE → post-hoc pyrolysis → blended properties → repeat
    # When char_state_mode="kinetic", α_i are ODE state variables — no staggered passes needed.
    _use_kinetic_char = str(getattr(fuel_cfg, "char_state_mode", "none")).lower() == "kinetic"
    _n_passes = 0 if _use_kinetic_char else int(getattr(fuel_cfg, "evolving_props_passes", 0))
    _prop_interp: Optional[dict] = None
    t_solver: np.ndarray = np.array([], dtype=float)
    T1_solver: np.ndarray = np.array([], dtype=float)
    T_mid_solver: Optional[np.ndarray] = None
    T2_solver: np.ndarray = np.array([], dtype=float)
    M1_solver: np.ndarray = np.array([], dtype=float)
    thermal_node_order: int = 2
    use_front_limit: bool = False
    _use_3node_front_limit: bool = False
    pyro_state_trace: Optional[_FuelStatePyrolysisTrace] = None

    for _pass in range(_n_passes + 1):
        result = integrate_fuel(y0, (0.0, t_end_s), fuel_cfg, env_cfg, forcing, sim_cfg, prop_interp=_prop_interp)
        t_solver = np.asarray(result.t, dtype=float)
        thermal_node_order = int(getattr(result, "thermal_node_order", 2) or 2)
        T1_solver = np.asarray(result.y[:, 0], dtype=float)
        T_mid_solver = None
        _N_therm = thermal_node_order
        if _N_therm >= 3:
            T_mid_solver = np.asarray(result.y[:, 1], dtype=float)
            T2_solver = np.asarray(result.y[:, _N_therm - 1], dtype=float)
            if _use_kinetic_char and result.y.shape[1] >= 3 * _N_therm:
                # Per-node M layout: M_i at columns N..2N-1; sum for total fuel remaining
                M1_solver = np.clip(
                    np.sum(result.y[:, _N_therm: 2 * _N_therm], axis=1), 0.0, None
                )
            else:
                M1_solver = np.asarray(result.y[:, _N_therm], dtype=float)
        else:
            T2_solver = np.asarray(result.y[:, 1], dtype=float)
            M1_solver = np.asarray(result.y[:, 2], dtype=float)
        # 3-node Stefan front: m_py_pp is populated whenever front_limit_enable=True and
        # N>=3 (regardless of char_state_mode).  Old guard required _use_kinetic_char which
        # excluded the arrhenius+Stefan case (no per-node char tracking).
        _use_3node_front_limit = (
            result.m_py_pp is not None
            and _N_therm >= 3
            and bool(getattr(fuel_cfg, "front_limit_enable", False))
        )
        # shape[1] >= 6 detects 2-node Stefan front (T1,T2,M1,delta_py,m_c,L).
        # For N>=3 + staged kinetics (two_step_sequential), there are also >=6 state vars
        # (T1..TN, M1, m1_global, m2_global) but NO Stefan front — check thermal order.
        use_front_limit = (
            _N_therm == 2 and result.y.shape[1] >= 6 and not _use_kinetic_char
        ) or _use_3node_front_limit

        # Extract kinetic char fractions from ODE state when char_state_mode=kinetic
        _ode_alpha1: Optional[np.ndarray] = None
        _ode_alpha2: Optional[np.ndarray] = None
        _ode_alpha3: Optional[np.ndarray] = None
        _ode_alpha_all: Optional[list] = None  # all N alpha arrays for N-node
        if _use_kinetic_char and result.y.shape[1] >= 3 * _N_therm:
            # Per-node M layout: α at columns 2N..3N-1
            _ode_alpha_all = [
                np.clip(np.asarray(result.y[:, 2 * _N_therm + j], dtype=float), 0.0, 1.0)
                for j in range(_N_therm)
            ]
            _ode_alpha1 = _ode_alpha_all[0]
            _ode_alpha2 = _ode_alpha_all[1]
            _ode_alpha3 = _ode_alpha_all[2] if _N_therm >= 3 else None

        if pyro_mass_source == "fuel_state" and not use_front_limit:
            pyro_state_trace = _pyrolysis_from_fuel_state(
                t=t_solver,
                T1=T1_solver,
                T_mid=T_mid_solver,
                T_deep=T2_solver if thermal_node_order == 3 else None,
                m0_fuel_kg_m2=m0_kg_m2_solver,
                fuel_cfg=fuel_cfg,
                thermal_node_order=thermal_node_order,
            )
        else:
            pyro_state_trace = None

        # Inject ODE-extracted char fractions into trace for diagnostics (kinetic char mode)
        if _use_kinetic_char and _ode_alpha1 is not None:
            if pyro_state_trace is not None:
                pyro_state_trace.alpha_zone1 = _ode_alpha1
                pyro_state_trace.alpha_zone2 = _ode_alpha2
                pyro_state_trace.alpha_zone3 = _ode_alpha3
            # (If pyro_state_trace is None, α data is available but not attached — acceptable)

        # Build property interpolators for next pass if alpha data is available
        if (
            _pass < _n_passes
            and pyro_state_trace is not None
            and pyro_state_trace.alpha_zone1 is not None
        ):
            _prop_interp = {
                "t": t_solver,
                "alpha1": pyro_state_trace.alpha_zone1,
                "alpha2": pyro_state_trace.alpha_zone2 if pyro_state_trace.alpha_zone2 is not None else np.zeros_like(t_solver),
                "alpha3": pyro_state_trace.alpha_zone3 if pyro_state_trace.alpha_zone3 is not None else np.zeros_like(t_solver),
            }
        else:
            break  # No alpha data or final pass complete

    if pyro_mass_source == "fuel_state" and not use_front_limit and pyro_state_trace is not None:
        m_py_raw_solver = pyro_state_trace.m_dot_vol_kg_m2_s
        m_fuel_remaining_solver = pyro_state_trace.m_remaining_total_kg_m2
    elif result.m_py_pp is not None:
        # Covers both Stefan front-limit and N>=3 HoG paths where m_py_pp was post-processed.
        m_py_raw_solver = np.asarray(result.m_py_pp, dtype=float)
        _km_runner = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
        if _km_runner in {"two_step_sequential", "semi_global_seq_yield"}:
            # Staged: remaining = m1_global + m2_global.
            # Pool state index = _fl_offset_3n + (3 if Stefan else 0).
            _use_kc_r = str(getattr(fuel_cfg, "char_state_mode", "none")).lower() == "kinetic"
            _use_fl_r = bool(getattr(fuel_cfg, "front_limit_enable", False))
            _N_r = int(getattr(fuel_cfg, "thermal_node_order", thermal_node_order) or thermal_node_order)
            _fl_3n_r = (3 * _N_r) if _use_kc_r else (_N_r + 1)
            _pool_r = _fl_3n_r + (3 if _use_fl_r else 0)
            if result.y.shape[1] > _pool_r + 1:
                _m1_rem = np.clip(result.y[:, _pool_r], 0.0, None)
                _m2_rem = np.clip(result.y[:, _pool_r + 1], 0.0, None)
                m_fuel_remaining_solver = _m1_rem + _m2_rem
            else:
                m_fuel_remaining_solver = _remaining_mass_from_legacy_m1(T1_solver, M1_solver, fuel_cfg)
        else:
            m_fuel_remaining_solver = _remaining_mass_from_legacy_m1(T1_solver, M1_solver, fuel_cfg)
    else:
        m_py_raw_solver = np.array(
            [pyrolysis_flux(t1, m1, fuel_cfg, t=t) for t1, m1, t in zip(T1_solver, M1_solver, t_solver)],
            dtype=float,
        )
        m_fuel_remaining_solver = _remaining_mass_from_legacy_m1(T1_solver, M1_solver, fuel_cfg)

    if fuel_cfg.enable_depletion and not (pyro_mass_source == "fuel_state" and not use_front_limit):
        m_py_solver, _ = apply_depletion(t_solver, m_py_raw_solver, m0_kg_m2_solver)
    else:
        m_py_solver = m_py_raw_solver
    hrrpua_solver = (m_py_solver * hoc_eff_j_kg) / 1000.0
    # Add char surface oxidation HRRPUA addend [W/m²] → [kW/m²] (Vermesi 2020 framework)
    if getattr(result, "char_ox_hrrpua_pp", None) is not None:
        _char_ox_kw = np.asarray(result.char_ox_hrrpua_pp, dtype=float) / 1000.0
        if _char_ox_kw.size == hrrpua_solver.size:
            hrrpua_solver = hrrpua_solver + _char_ox_kw
    elif (
        pyro_mass_source == "fuel_state"
        and not use_front_limit
        and bool(getattr(fuel_cfg, "char_ox_enable", False))
    ):
        # Char oxidation for fuel_state (two_step_sequential) path.
        # two_node.py only computes char_ox_hrrpua_pp for Stefan/HoG branches; this
        # replicates the same physics post-hoc from the pyrolysis rate time series.
        # Physical basis: residual lignin char accumulates as volatile pyrolysis proceeds;
        # it smolders once the volatile blow rate drops below the suppression threshold.
        _cx_cy     = float(getattr(fuel_cfg, "char_ox_char_yield",          0.0))
        _cx_qref   = float(getattr(fuel_cfg, "char_ox_q_ref_W_m2",    None) or 45000.0)
        _cx_hoc    = max(float(getattr(fuel_cfg, "char_ox_char_hoc_J_kg",  32.7e6)), 1.0)
        _cx_mps0   = float(getattr(fuel_cfg, "char_ox_m_py_stefan0_kg_m2_s", None) or 0.0)
        _cx_qs0    = max(float(getattr(fuel_cfg, "char_ox_q_stefan0_W_m2", None) or 80000.0), 1.0)
        _cx_m0     = max(float(m_fuel_remaining_solver[0]), 1e-9)
        _cx_pool   = 0.0
        _cx_arr    = np.zeros(len(t_solver), dtype=float)
        for _cxi in range(1, len(t_solver)):
            _dt_cx = t_solver[_cxi] - t_solver[_cxi - 1]
            if _dt_cx <= 0.0:
                continue
            # alpha proxy: fraction of initial fuel consumed (0 → 1 as bed burns)
            _alpha_cx = max(1.0 - float(m_fuel_remaining_solver[_cxi]) / _cx_m0, 0.0)
            _cx_pool += _cx_cy * float(m_py_solver[_cxi]) * _dt_cx
            if _cx_mps0 > 0.0:
                _f_cx = max(1.0 - float(m_py_solver[_cxi]) / _cx_mps0, 0.0)
            else:
                _q_cx_i, _ = eval_q_in_incident_W_m2(t_solver[_cxi], fuel_cfg)
                _f_cx = max(1.0 - _q_cx_i / _cx_qs0, 0.0)
            _raw_cx      = _alpha_cx * _cx_qref * _f_cx          # [W/m²]
            _demanded_cx = _raw_cx * _dt_cx / _cx_hoc
            _actual_cx   = min(_demanded_cx, _cx_pool)
            _cx_pool     = max(_cx_pool - _actual_cx, 0.0)
            _cx_arr[_cxi] = _actual_cx * _cx_hoc / _dt_cx        # [W/m²]
        hrrpua_solver = hrrpua_solver + _cx_arr / 1000.0         # W/m² → kW/m²
    # Add back-face pyrolysis HRRPUA [kg/m²/s × J/kg → kW/m²]
    if getattr(result, "back_py_mdot_pp", None) is not None:
        _back_py_kw = np.asarray(result.back_py_mdot_pp, dtype=float) * hoc_eff_j_kg / 1000.0
        if _back_py_kw.size == hrrpua_solver.size:
            hrrpua_solver = hrrpua_solver + _back_py_kw
    # Add lignin slow-release pool HRRPUA addend (Orfão 1999; E_lig~200 kJ/mol, slow burn-phase release)
    # Post-processing: integrate dm_lig/dt = -k_lig(T1)*m_lig from T1 solver history.
    # Does not modify ODE state; pure addend to reported HRRPUA. Disabled by default (lig_a_1_s=0).
    if bool(getattr(fuel_cfg, "lig_enable", False)) and float(getattr(fuel_cfg, "lig_a_1_s", 0.0)) > 0.0:
        _m_total_lig = float(getattr(fuel_cfg, "lig_m_frac", 0.27)) * float(
            getattr(fuel_cfg, "m_fuel_total_kg_m2", None) or (
                float(getattr(fuel_cfg, "rho_solid", None) or getattr(fuel_cfg, "rho", 379.0) or 379.0)
                * float(getattr(fuel_cfg, "thickness_m", None) or 0.038)
            )
        )
        _A_lig = float(getattr(fuel_cfg, "lig_a_1_s", 0.0))
        _E_lig = float(getattr(fuel_cfg, "lig_e_j_mol", 200000.0))
        _k_lig = _A_lig * np.exp(-_E_lig / (8.314 * np.maximum(T1_solver, 1.0)))
        _m_lig = np.zeros(t_solver.size, dtype=float)
        _m_lig[0] = _m_total_lig
        for _li in range(1, t_solver.size):
            _dt_li = float(t_solver[_li] - t_solver[_li - 1])
            _m_lig[_li] = max(_m_lig[_li - 1] * (1.0 - _k_lig[_li - 1] * _dt_li), 0.0)
        hrrpua_solver = hrrpua_solver + (_k_lig * _m_lig) * hoc_eff_j_kg / 1000.0

    # ── Flame radiation feedback (convergent multi-pass, disabled by default) ───────
    # Activated by: fuel.flame_enable = true  (plus optional fuel.flame_* params)
    # Physics: q_fb = chi_rad × view_factor × HRRPUA  (Tewarson / De Ris closure)
    # Iterates to self-consistent fixed point: HRRPUA* = HRRPUA_0/(1 - chi_rad×F).
    # Contraction ratio = chi_rad×view_factor ≈ 0.10 < 1 → guaranteed convergence
    # by Banach fixed-point theorem. Typically converges in 3-4 ODE solves.
    # fuel.flame_coupling_passes = max iterations (default 10; exits early via tol).
    # fuel.flame_coupling_passes = 1 reproduces legacy single-pass behavior exactly.
    _flame_enable = bool(getattr(fuel_cfg, "flame_enable", False))
    if _flame_enable:
        try:
            from model.flame import (  # noqa: PLC0415
                FlameConfig, FlameInternalState, flame_step, SolidFuelViability,
            )
            _fcfg = FlameConfig(
                chi_rad=float(getattr(fuel_cfg, "flame_chi_rad", 0.35)),
                view_factor=float(getattr(fuel_cfg, "flame_view_factor", 0.40)),
                persistence_s=float(getattr(fuel_cfg, "flame_persistence_s", 5.0)),
            )
            _fvib = SolidFuelViability(
                m_py_ignite=float(getattr(fuel_cfg, "flame_m_py_ignite", 0.005)),
                T_ignite=float(getattr(fuel_cfg, "flame_T_ignite", 600.0)),
                m_py_crit=float(getattr(fuel_cfg, "flame_m_py_crit", 0.001)),
                T_py=float(getattr(fuel_cfg, "flame_T_py", 500.0)),
            )
            _tau_fl_growth = float(getattr(fuel_cfg, "flame_tau_growth_s", -1.0))
            _max_fl_passes = int(float(getattr(fuel_cfg, "flame_coupling_passes", 10)))
            _fl_tol_w = float(getattr(fuel_cfg, "flame_coupling_tol_W_m2", 1.0))
            _hrrpua_prev_w: np.ndarray | None = None
            _t_prev_w: np.ndarray | None = None
            # Char oxidation ODE feedback carry-over: t and q arrays from previous pass.
            # Populated after each pass; fed into forcing_fl for the next ODE integration.
            # On pass 0, no char_ox feedback (carry is None) — same as no-flame-coupling baseline.
            _q_char_ox_t_carry: np.ndarray | None = None
            _q_char_ox_q_carry: np.ndarray | None = None
            # Heskestad geometry: compute F(t) from bed area + HRRPUA at each coupling step.
            # Active when flame_geometry_mode = "heskestad" AND flame_area_m2 is set.
            # On each pass, overrides _fcfg.view_factor per time-step with the geometry-derived F.
            _fl_geom_mode = str(getattr(fuel_cfg, "flame_geometry_mode", "deck")).lower()
            _fl_area_m2_fl = getattr(fuel_cfg, "flame_area_m2", None)
            _fl_geom_active = False
            _D_eq_fl = 0.0
            # Wind speed for flame-tilt enhancement (Albini 1981).
            # u_inf_m_s is the midflame wind speed set by fuel_element.py via fuel_overrides.
            # Tilt multiplier = 1 + 0.4 sin(θ); θ from Byram (1959) Froude correlation.
            # Active only when Heskestad geometry is used; backwards compatible (still air → mult=1.0).
            _fl_wind_m_s = float(getattr(fuel_cfg, "u_inf_m_s", 0.0))
            if _fl_geom_mode == "heskestad" and _fl_area_m2_fl is not None and float(_fl_area_m2_fl) > 0.0:
                try:
                    from model.flame.geometry import (  # noqa: PLC0415
                        bed_equivalent_diameter as _bed_eq_diam_fl,
                        heskestad_flame_height as _hsk_lf_fl,
                        pool_fire_view_factor as _pf_vf_fl,
                    )
                    from model_outdoor.boundary import (  # noqa: PLC0415
                        flame_tilt_angle as _fl_tilt_fn,
                        view_factor_tilt_enhancement as _fl_vf_tilt_fn,
                    )
                    _D_eq_fl = _bed_eq_diam_fl(float(_fl_area_m2_fl))
                    _fl_geom_active = True
                except ImportError:
                    pass
            for _fl_pass in range(_max_fl_passes):
                # Current HRRPUA in W/m² (includes all addends: char_ox, back_py, lignin)
                _hrrpua_w = hrrpua_solver * 1000.0
                # Convergence check: exit when HRRPUA change < tolerance (skip on first pass).
                # Interpolate previous pass onto current time grid before comparing — adaptive
                # ODE solver may return different grid lengths across passes.
                if _hrrpua_prev_w is not None and _t_prev_w is not None:
                    _hrrpua_prev_interp = np.interp(t_solver, _t_prev_w, _hrrpua_prev_w)
                    if float(np.max(np.abs(_hrrpua_w - _hrrpua_prev_interp))) < _fl_tol_w:
                        break
                _hrrpua_prev_w = _hrrpua_w.copy()
                _t_prev_w = t_solver.copy()
                # Drive flame state machine through current HRRPUA time series
                _q_fb_arr = np.zeros(t_solver.size, dtype=float)
                _internal_fl = FlameInternalState()
                _t_ign_fl: float | None = None  # raw time of first flame ignition
                for _fi in range(t_solver.size):
                    # Heskestad geometry: override view factor from current HRRPUA.
                    # Physical basis: taller flame (higher HRRPUA) subtends larger solid angle
                    # at the bed → higher view factor → stronger feedback. Replaces static deck F.
                    # Flame tilt (Albini 1981): wind tilts flame toward fuel, enhancing F by
                    #   (1 + 0.4 sin θ).  θ from Byram (1959): tan θ = 0.88 (U_mf²/gL_f)^0.5.
                    #   Applied per timestep using current L_f.  Still air → mult=1.0 (no change).
                    if _fl_geom_active:
                        _Q_kw_fi = max(float(_hrrpua_w[_fi]) / 1000.0 * float(_fl_area_m2_fl), 0.0)
                        _L_f_fi  = _hsk_lf_fl(_Q_kw_fi, _D_eq_fl)
                        _F_fi    = _pf_vf_fl(_L_f_fi, _D_eq_fl)
                        if _fl_wind_m_s > 0.0 and _L_f_fi > 0.0:
                            _F_fi *= _fl_vf_tilt_fn(_fl_tilt_fn(_fl_wind_m_s, _L_f_fi, "open"))
                        _fcfg.view_factor = _F_fi
                    _fo = {
                        "HRRPUA_W_m2": float(_hrrpua_w[_fi]),
                        "T_surf_K": float(T1_solver[_fi]),
                        "m_py": float(m_py_solver[_fi]),
                    }
                    _q_fi, _, _internal_fl = flame_step(float(t_solver[_fi]), _fo, _fcfg, _fvib, _internal_fl)
                    # Flame growth ramp: q_fb *= (1 - exp(-(t-t_ign)/tau_growth))
                    # Physical basis: flame grows from pilot spark to full coverage over ~tau_growth s.
                    # De Ris / Zukoski flame height correlations: H_fl ~ m_dot^0.4 → feedback scales
                    # with flame height → ramp timescale = time from piloted ignition to full flame.
                    if _tau_fl_growth > 0.0 and _q_fi > 0.0:
                        if _t_ign_fl is None:
                            _t_ign_fl = float(t_solver[_fi])
                        _t_rel_fl = float(t_solver[_fi]) - _t_ign_fl
                        _q_fi *= 1.0 - float(np.exp(-_t_rel_fl / _tau_fl_growth))
                    _q_fb_arr[_fi] = _q_fi
                # Build time-interpolating callable for use inside fuel_rhs()
                _t_fl = t_solver.copy()
                _q_fl = _q_fb_arr.copy()
                def _q_fb_callable(t_: float, _tv=_t_fl, _qv=_q_fl) -> float:
                    return float(np.interp(t_, _tv, _qv))
                forcing_fl = dict(forcing)
                forcing_fl["q_fb"] = _q_fb_callable
                # Char oxidation ODE thermal feedback: inject q_char_ox from previous pass.
                # Physical basis: char surface oxidation heats node 1, sustaining T1 above
                # T_py after volatile depletion (Frandsen 1991 smoldering criterion).
                # Without ODE feedback, char heat appears only in hrrpua_solver (post-processing)
                # and cannot sustain element temperature — element self-extinguishes despite
                # available char energy.  This carry-over approach mirrors how q_fb is applied.
                if _q_char_ox_t_carry is not None and _q_char_ox_q_carry is not None:
                    _t_cx_c = _q_char_ox_t_carry
                    _q_cx_c = _q_char_ox_q_carry
                    def _q_cx_callable_carry(t_: float, _tv=_t_cx_c, _qv=_q_cx_c) -> float:
                        return float(np.interp(t_, _tv, _qv))
                    forcing_fl["q_char_ox"] = _q_cx_callable_carry
                # Re-integrate fuel ODE with current flame feedback
                result = integrate_fuel(
                    y0, (0.0, t_end_s), fuel_cfg, env_cfg, forcing_fl, sim_cfg,
                    prop_interp=_prop_interp,
                )
                t_solver = np.asarray(result.t, dtype=float)
                thermal_node_order = int(getattr(result, "thermal_node_order", 2) or 2)
                _N_therm = thermal_node_order
                T1_solver = np.asarray(result.y[:, 0], dtype=float)
                if _N_therm >= 3:
                    T_mid_solver = np.asarray(result.y[:, 1], dtype=float)
                    T2_solver = np.asarray(result.y[:, _N_therm - 1], dtype=float)
                    if _use_kinetic_char and result.y.shape[1] >= 3 * _N_therm:
                        M1_solver = np.clip(
                            np.sum(result.y[:, _N_therm: 2 * _N_therm], axis=1), 0.0, None
                        )
                        _ode_alpha_all = [
                            np.clip(np.asarray(result.y[:, 2 * _N_therm + j], dtype=float), 0.0, 1.0)
                            for j in range(_N_therm)
                        ]
                        _ode_alpha1 = _ode_alpha_all[0]
                        _ode_alpha2 = _ode_alpha_all[1]
                        _ode_alpha3 = _ode_alpha_all[2] if _N_therm >= 3 else None
                    else:
                        M1_solver = np.asarray(result.y[:, _N_therm], dtype=float)
                else:
                    T_mid_solver = None
                    T2_solver = np.asarray(result.y[:, 1], dtype=float)
                    M1_solver = np.asarray(result.y[:, 2], dtype=float)
                # Recompute m_py and hrrpua for this pass
                if pyro_mass_source == "fuel_state" and not use_front_limit:
                    _fl_trace = _pyrolysis_from_fuel_state(
                        t=t_solver, T1=T1_solver, T_mid=T_mid_solver,
                        T_deep=T2_solver if thermal_node_order == 3 else None,
                        m0_fuel_kg_m2=m0_kg_m2_solver, fuel_cfg=fuel_cfg,
                        thermal_node_order=thermal_node_order,
                    )
                    m_py_solver = _fl_trace.m_dot_vol_kg_m2_s if _fl_trace is not None else m_py_solver
                    if _fl_trace is not None and getattr(_fl_trace, "m_remaining_total_kg_m2", None) is not None:
                        m_fuel_remaining_solver = _fl_trace.m_remaining_total_kg_m2
                    if _fl_trace is not None:
                        pyro_state_trace = _fl_trace
                elif result.m_py_pp is not None:
                    # Covers Stefan front-limit AND HoG-cap paths (hog_enable=true).
                    # Mirrors the initial m_py_raw_solver setup at lines 1812-1814.
                    m_py_solver = np.asarray(result.m_py_pp, dtype=float)
                else:
                    m_py_solver = np.array(
                        [pyrolysis_flux(t1, m1, fuel_cfg, t=t) for t1, m1, t in zip(T1_solver, M1_solver, t_solver)],
                        dtype=float,
                    )
                if fuel_cfg.enable_depletion and not (pyro_mass_source == "fuel_state" and not use_front_limit):
                    m_py_solver, _ = apply_depletion(t_solver, m_py_solver, m0_kg_m2_solver)
                hrrpua_solver = (m_py_solver * hoc_eff_j_kg) / 1000.0
                if getattr(result, "char_ox_hrrpua_pp", None) is not None:
                    _cok = np.asarray(result.char_ox_hrrpua_pp, dtype=float) / 1000.0
                    if _cok.size == hrrpua_solver.size:
                        hrrpua_solver = hrrpua_solver + _cok
                elif (
                    pyro_mass_source == "fuel_state"
                    and not use_front_limit
                    and bool(getattr(fuel_cfg, "char_ox_enable", False))
                ):
                    # Char oxidation for fuel_state path — mirrors block at ~line 1868.
                    # Must be repeated here because flame feedback re-integrates and
                    # rebuilds hrrpua_solver, discarding any previous char_ox addend.
                    _cx2_cy   = float(getattr(fuel_cfg, "char_ox_char_yield",          0.0))
                    _cx2_qref = float(getattr(fuel_cfg, "char_ox_q_ref_W_m2",    None) or 45000.0)
                    _cx2_hoc  = max(float(getattr(fuel_cfg, "char_ox_char_hoc_J_kg",  32.7e6)), 1.0)
                    _cx2_mps0 = float(getattr(fuel_cfg, "char_ox_m_py_stefan0_kg_m2_s", None) or 0.0)
                    _cx2_qs0  = max(float(getattr(fuel_cfg, "char_ox_q_stefan0_W_m2", None) or 80000.0), 1.0)
                    _cx2_m0   = max(float(m_fuel_remaining_solver[0]), 1e-9)
                    _cx2_pool = 0.0
                    _cx2_arr  = np.zeros(len(t_solver), dtype=float)
                    for _cx2i in range(1, len(t_solver)):
                        _dt2 = t_solver[_cx2i] - t_solver[_cx2i - 1]
                        if _dt2 <= 0.0:
                            continue
                        _alpha2 = max(1.0 - float(m_fuel_remaining_solver[_cx2i]) / _cx2_m0, 0.0)
                        _cx2_pool += _cx2_cy * float(m_py_solver[_cx2i]) * _dt2
                        if _cx2_mps0 > 0.0:
                            _f2 = max(1.0 - float(m_py_solver[_cx2i]) / _cx2_mps0, 0.0)
                        else:
                            _q2_i, _ = eval_q_in_incident_W_m2(t_solver[_cx2i], fuel_cfg)
                            _f2 = max(1.0 - _q2_i / _cx2_qs0, 0.0)
                        _raw2     = _alpha2 * _cx2_qref * _f2
                        _dem2     = _raw2 * _dt2 / _cx2_hoc
                        _act2     = min(_dem2, _cx2_pool)
                        _cx2_pool = max(_cx2_pool - _act2, 0.0)
                        _cx2_arr[_cx2i] = _act2 * _cx2_hoc / _dt2
                    hrrpua_solver = hrrpua_solver + _cx2_arr / 1000.0
                    # Save char_ox for next coupling pass (carry-over; injected into
                    # forcing_fl at the start of the next pass, before integrate_fuel).
                    _q_char_ox_t_carry = t_solver.copy()
                    _q_char_ox_q_carry = _cx2_arr.copy()
                    # Reset convergence state so the next pass always runs with the new
                    # char_ox carry applied to the ODE.  Without this, the convergence
                    # check at the top of the next iteration compares hrrpua values that
                    # do not yet reflect ODE-level char_ox thermal feedback (flame may not
                    # ignite at all, making pass-to-pass hrrpua look unchanged), causing
                    # early exit before char_ox coupling has any thermal effect.
                    _hrrpua_prev_w = None
                    _t_prev_w = None
                if getattr(result, "back_py_mdot_pp", None) is not None:
                    _bpk = np.asarray(result.back_py_mdot_pp, dtype=float) * hoc_eff_j_kg / 1000.0
                    if _bpk.size == hrrpua_solver.size:
                        hrrpua_solver = hrrpua_solver + _bpk
                # Lignin pool (re-applied after each flame-coupled T1 history)
                if bool(getattr(fuel_cfg, "lig_enable", False)) and float(getattr(fuel_cfg, "lig_a_1_s", 0.0)) > 0.0:
                    _m_total_lig2 = float(getattr(fuel_cfg, "lig_m_frac", 0.27)) * float(
                        getattr(fuel_cfg, "m_fuel_total_kg_m2", None) or (
                            float(getattr(fuel_cfg, "rho_solid", None) or getattr(fuel_cfg, "rho", 379.0) or 379.0)
                            * float(getattr(fuel_cfg, "thickness_m", None) or 0.038)
                        )
                    )
                    _A_lig2 = float(getattr(fuel_cfg, "lig_a_1_s", 0.0))
                    _E_lig2 = float(getattr(fuel_cfg, "lig_e_j_mol", 200000.0))
                    _k_lig2 = _A_lig2 * np.exp(-_E_lig2 / (8.314 * np.maximum(T1_solver, 1.0)))
                    _m_lig2 = np.zeros(t_solver.size, dtype=float)
                    _m_lig2[0] = _m_total_lig2
                    for _li2 in range(1, t_solver.size):
                        _dt_li2 = float(t_solver[_li2] - t_solver[_li2 - 1])
                        _m_lig2[_li2] = max(_m_lig2[_li2 - 1] * (1.0 - _k_lig2[_li2 - 1] * _dt_li2), 0.0)
                    hrrpua_solver = hrrpua_solver + (_k_lig2 * _m_lig2) * hoc_eff_j_kg / 1000.0
        except ImportError:
            pass  # model/flame not on path — skip flame coupling silently

    # ── Pre-ignition volatile pool burst (Sanned et al. 2023, Babrauskas 2023) ──────
    # Before flame establishes, Arrhenius kinetics produce m_dot_kin > 0. Volatiles
    # accumulate in the boundary layer and burn as a burst at ignition (T1 >= T_ignite).
    # This mechanism explains the ignition spike at low flux (e.g. 257 kW/m² at 25 kW/m²
    # for Wood Stud, from ~30s pre-ignition accumulation).
    if bool(getattr(fuel_cfg, "vol_pool_enable", False)) and getattr(result, "m_dot_kin", None) is not None:
        _T_ign = float(getattr(fuel_cfg, "flame_T_ignite", 600.0))
        _t_ign_mask = T1_solver >= _T_ign
        if _t_ign_mask.any():
            _ign_idx = int(np.argmax(_t_ign_mask))
            if _ign_idx > 0:
                _t_ign = float(t_solver[_ign_idx])
                _mk_interp = np.interp(t_solver, result.t, np.asarray(result.m_dot_kin, dtype=float))
                _m_vol_acc = float(np.trapezoid(_mk_interp[:_ign_idx], t_solver[:_ign_idx]))
                if bool(getattr(fuel_cfg, "vol_pool_tau_auto", False)):
                    # τ = 1/k(T_py + 80K): pool consumption timescale at piloted ignition temp.
                    # Physical basis: first-order decay dm/dt = -k*m → τ = 1/k(T_s_ignition).
                    # T_s_ignition ≈ T_py + 80K (material property: ~80K overshoot above pyrolysis
                    # onset at piloted ignition for wood; Drysdale 2011, SFPE 4th ed.).
                    _A_py_bp = float(getattr(fuel_cfg, "A_py", 0.0))
                    _E_py_bp = float(getattr(fuel_cfg, "E_py", 0.0))
                    _T_py_bp = float(getattr(fuel_cfg, "regression_T_py_K", 600.0) or 600.0)
                    _T_ign_surf = _T_py_bp + 80.0
                    if _A_py_bp > 0.0 and _E_py_bp > 0.0:
                        _k_ign = _A_py_bp * float(np.exp(-_E_py_bp / (8.314 * _T_ign_surf)))
                        _tau = max(1.0 / _k_ign, 0.5) if _k_ign > 0.0 else 8.0
                    else:
                        _tau = 8.0  # fallback if Arrhenius params missing
                else:
                    _tau = float(getattr(fuel_cfg, "vol_pool_tau_s", 8.0))
                _sig = max(_tau / 2.355, 0.5)
                # Burst center offset: shift peak to t_ign + t_peak_offset.
                # EXP often shows HRRPUA rising for ~10s after ignition before peaking
                # (pre-accumulated volatiles ignite as flame grows). Default 0 = peak at ignition.
                _t_peak_offset = float(getattr(fuel_cfg, "vol_pool_t_peak_s", 0.0))
                _tau_decay = float(getattr(fuel_cfg, "vol_pool_tau_decay_s", -1.0))
                _t_rel = t_solver - _t_ign - _t_peak_offset
                if _tau_decay > 0.0:
                    # Asymmetric burst: Gaussian rise (left), exponential decay (right).
                    # Physical basis: pool burn-down is first-order (dm/dt = -m/τ_decay),
                    # flame-limited, not surface-kinetics-limited. Bridges gap between
                    # burst decay and Stefan ramp-up. (Williams 1985; pool combustion kinetics)
                    _burst_raw = np.where(
                        _t_rel <= 0.0,
                        np.exp(-0.5 * (_t_rel / _sig) ** 2),
                        np.exp(-_t_rel / _tau_decay),
                    )
                else:
                    # Symmetric Gaussian (default, backward compatible)
                    _burst_raw = np.exp(-0.5 * (_t_rel / _sig) ** 2)
                # Mask burst to post-ignition times only (t >= t_ign).
                # Physical: pre-ignition volatiles can only combust after piloted ignition;
                # without a flame there is no oxidiser delivery to the surface layer.
                # Without masking, a wide-sigma burst (tau > ~8s) would add spurious
                # HRRPUA before t_ign, shift the ignition-reference time, and distort
                # the EXP time-alignment in the validation script.
                _burst_raw = np.where(t_solver >= _t_ign, _burst_raw, 0.0)
                # Normalise with trapezoid integration to handle non-uniform ODE time steps.
                # The adaptive solver uses very fine steps near ignition (dt≈0.01-0.025s)
                # and coarser steps elsewhere (up to max_step=1s). Using mean_dt × sum()
                # would under-count the Gaussian area by 10-30× for the fine-step region.
                _norm = float(np.trapezoid(_burst_raw, t_solver))
                if _norm > 1e-12:
                    _burst_norm = _burst_raw / _norm  # unit-area Gaussian impulse [1/s]
                    # Main volatile pool from Stefan Arrhenius m_dot_kin (already computed)
                    if _m_vol_acc > 0.0:
                        hrrpua_solver = hrrpua_solver + _m_vol_acc * hoc_eff_j_kg / 1000.0 * _burst_norm
                    # ── Pre-ignition low-T hemicellulose accumulation (Orfão 1999) ──────
                    # Hemicellulose dehydration/depolymerization: E≈80 kJ/mol, onset ~200°C.
                    # Below T_py=600K the main Stefan Arrhenius (E=162 kJ/mol) produces ≈0;
                    # this separate kinetic captures the 200–330°C accumulation window.
                    _A_pre = float(getattr(fuel_cfg, "vol_pool_a_preheat_1_s", 0.0))
                    _E_pre = float(getattr(fuel_cfg, "vol_pool_e_preheat_j_mol", 80000.0))
                    if _A_pre > 0.0:
                        _R_gas = 8.314
                        _rho0 = float(getattr(fuel_cfg, "rho_solid", 0.0) or getattr(fuel_cfg, "density", 0.0) or 379.0)
                        _L0   = float(getattr(fuel_cfg, "regression_L0_m", 0.038))
                        _m0   = _rho0 * _L0   # initial fuel surface density [kg/m²]
                        _T1_pre = T1_solver[:_ign_idx]
                        _m_dot_pre = _A_pre * np.exp(-_E_pre / (_R_gas * np.maximum(_T1_pre, 200.0))) * _m0
                        _m_vol_pre = float(np.trapezoid(_m_dot_pre, t_solver[:_ign_idx]))
                        if _m_vol_pre > 0.0:
                            hrrpua_solver = hrrpua_solver + _m_vol_pre * hoc_eff_j_kg / 1000.0 * _burst_norm

    mlr_solver = m_py_solver * area_m2

    # ── Front-face HoG floor (post-processing, Stefan thin-panel correction) ──────────────
    # Stefan rate ∝ 1/delta_py → declines steeply as char thickens. For thin panels the
    # distributed pyrolysis zone behind the char front sustains a higher HRRPUA than the
    # sharp-front model gives. Apply floor: hrrpua = max(hrrpua, q_in/L_eff × (1-α_bar)).
    # Only active when front_limit_enable=True AND front_hog_floor_enable=True in the deck.
    if use_front_limit and bool(getattr(fuel_cfg, "front_hog_floor_enable", False)):
        _fhf_L_eff = max(float(getattr(fuel_cfg, "front_hog_floor_L_eff_J_kg", 5.5e6) or 5.5e6), 1.0)
        _fhf_q_in  = float(q_in_kW_m2) * 1000.0  # [W/m²] cone irradiance (Tewarson calibration basis)
        # alpha_bar: volume-weighted char fraction. State layout for N-node Stefan:
        # [T1..TN, M1..MN, α1..αN, delta_py, m_c, L] = 3N+3 states → alpha start = 2N.
        # Detect N from state vector size so 3-node and 5-node both work correctly.
        _fhf_alpha_bar = np.zeros_like(hrrpua_solver)
        if result is not None and result.y is not None and result.y.shape[1] >= 12:
            _fhf_n_states = result.y.shape[1]
            _fhf_N = (_fhf_n_states - 3) // 3   # e.g. 12→3, 18→5
            _fhf_alpha_start = 2 * _fhf_N
            if _fhf_alpha_start + _fhf_N <= _fhf_n_states:
                _fhf_node_fracs = [
                    float(v) if (v := getattr(rom_inputs, f"node{i + 1}_frac", None) if rom_inputs else None) is not None
                    else (1.0 / _fhf_N)
                    for i in range(_fhf_N)
                ]
                _fhf_alpha_bar = np.clip(
                    sum(_fhf_node_fracs[i] * result.y[:, _fhf_alpha_start + i] for i in range(_fhf_N)),
                    0.0, 1.0,
                )
        # Gate: floor activates only when T1 ≥ T_py (pyrolysis thermally feasible; prevents pre-ignition artifact)
        _fhf_T_py = float(getattr(fuel_cfg, "regression_T_py_K", 548.0) or 548.0)
        if result is not None and result.y is not None and result.y.shape[0] == len(t_solver) and result.y.shape[1] >= 1:
            _fhf_active = np.where(result.y[:, 0] >= _fhf_T_py, 1.0, 0.0)
        else:
            _fhf_active = np.ones_like(hrrpua_solver)
        _fhf_mdot  = np.maximum(_fhf_q_in, 0.0) / _fhf_L_eff * np.maximum(1.0 - _fhf_alpha_bar, 0.0) * _fhf_active
        _fhf_kw    = _fhf_mdot * hoc_eff_j_kg / 1000.0
        hrrpua_solver = np.maximum(hrrpua_solver, _fhf_kw)

    # ── Smoldering char oxidation — O₂-diffusion-limited glowing combustion (Frandsen 1991) ──
    # Post-processing addend to hrrpua_solver; no ODE state changes (Rule #7: new physics only when
    # existing architecture demonstrably cannot capture the feature AND a suppression pathway exists;
    # here: re-ignition from residual glowing char after agent application is the suppression scenario).
    # Pool charged by pyrolysis × char_yield_smolder; discharged by blow-suppression-gated slow
    # oxidation (q_ref ~ 1–8 kW/m²). Gate: f_sm = max(1 - m_py/m_py_s0, 0) delays activation until
    # active flaming (high volatile flux) subsides — matching physical O₂ access to char bed.
    _sm_m_arr_solver = None
    if bool(getattr(fuel_cfg, "char_smolder_enable", False)):
        _sm_q_ref  = float(getattr(fuel_cfg, "char_smolder_q_ref_W_m2", None) or 5000.0)
        _sm_hoc    = float(getattr(fuel_cfg, "char_smolder_hoc_J_kg", 32.7e6))
        _sm_yield_v = getattr(fuel_cfg, "char_smolder_char_yield", None)
        if _sm_yield_v is None:
            # Fall back to seq_mr_frac0 (char residue fraction from two_step_sequential)
            _sm_yield_v = float(getattr(fuel_cfg, "seq_mr_frac0", 0.20) or 0.20)
        else:
            _sm_yield_v = float(_sm_yield_v)
        _sm_py_s0_v = getattr(fuel_cfg, "char_smolder_m_py_s0_kg_m2_s", None)
        _sm_py_s0_v = float(_sm_py_s0_v) if _sm_py_s0_v is not None else None

        _sm_pool       = 0.0  # [kg/m²] smoldering char pool current value
        _sm_m_arr_solver   = np.zeros(t_solver.size, dtype=float)
        _sm_hrrpua_arr = np.zeros(t_solver.size, dtype=float)
        _sm_consume_max = _sm_q_ref / _sm_hoc  # [kg/m²/s] peak consumption rate at q_ref

        for _si in range(1, t_solver.size):
            _dt_si = float(t_solver[_si] - t_solver[_si - 1])
            if _dt_si <= 0.0:
                _sm_m_arr_solver[_si] = _sm_pool
                continue
            # Volatile flux at previous time step [kg/m²/s]
            _m_py_i = float(m_py_solver[_si - 1])
            # Charge: char accumulated from pyrolysis over this step
            _sm_pool = _sm_pool + _sm_yield_v * _m_py_i * _dt_si
            # Blow-suppression gate: f_sm=0 while high volatile flux; f_sm→1 after flaming subsides
            if _sm_py_s0_v is not None and _sm_py_s0_v > 0.0:
                _f_sm = max(1.0 - _m_py_i / _sm_py_s0_v, 0.0)
            else:
                _f_sm = 1.0  # no gate — always active once pool has mass
            # Discharge: limited by pool availability
            _dm_consume = min(_sm_consume_max * _f_sm * _dt_si, _sm_pool)
            _sm_pool    = max(_sm_pool - _dm_consume, 0.0)
            _sm_hrrpua_arr[_si] = (_dm_consume / _dt_si) * _sm_hoc / 1000.0  # [kW/m²]
            _sm_m_arr_solver[_si] = _sm_pool

        if _sm_hrrpua_arr.size == hrrpua_solver.size:
            hrrpua_solver = hrrpua_solver + _sm_hrrpua_arr

    # After flame 2nd pass t_solver may have more points; re-align m_fuel_remaining if needed
    if m_fuel_remaining_solver is not None and m_fuel_remaining_solver.size != t_solver.size:
        _t_mfr_old = np.linspace(0.0, t_solver[-1], m_fuel_remaining_solver.size)
        m_fuel_remaining_solver = np.interp(t_solver, _t_mfr_old, m_fuel_remaining_solver)

    if pyro_mass_source == "fuel_state" and not use_front_limit and area_m2 > 0.0:
        mass_total_solver = np.maximum(m_fuel_remaining_solver * area_m2, 0.0)
    else:
        if m0_kg is None:
            m0_kg = 1.0
        cumulative_loss_solver = np.cumsum(
            np.concatenate([[0.0], 0.5 * (mlr_solver[1:] + mlr_solver[:-1]) * np.diff(t_solver)])
        )
        mass_total_solver = np.maximum(m0_kg - cumulative_loss_solver, 0.0)

    hrrpua_diag_solver = None
    if area_m2 > 0.0:
        mlr_diag_solver = -np.gradient(mass_total_solver, t_solver, edge_order=1)
        mlr_diag_solver = np.maximum(mlr_diag_solver, 0.0)
        hrrpua_diag_solver = ((mlr_diag_solver / area_m2) * hoc_eff_j_kg) / 1000.0
        if rom_inputs and rom_inputs.hrr_from_mlr:
            hrrpua_solver = hrrpua_diag_solver
            mlr_solver = mlr_diag_solver

    dt_out = float(getattr(sim_cfg, "dt_out", 1.0) or 1.0)
    if not np.isfinite(dt_out) or dt_out <= 0.0:
        dt_out = 1.0
    t_end_out = max(float(t_end_s), 0.0)
    t_out = np.arange(0.0, t_end_out + dt_out, dt_out, dtype=float)
    if t_out.size == 0:
        t_out = np.array([0.0], dtype=float)
    t_out = t_out[t_out <= t_end_out + 1.0e-9]
    if t_out.size == 0:
        t_out = np.array([0.0], dtype=float)
    if abs(float(t_out[-1]) - t_end_out) > 1.0e-9:
        t_out = np.append(t_out, t_end_out)

    T1 = _interp_series_to_grid(t_out, t_solver, T1_solver)
    T2 = _interp_series_to_grid(t_out, t_solver, T2_solver)
    T_mid_out = _interp_series_to_grid(t_out, t_solver, T_mid_solver)
    M1_out = _interp_series_to_grid(t_out, t_solver, M1_solver)
    m_fuel_remaining_out = _interp_series_to_grid(t_out, t_solver, m_fuel_remaining_solver)
    m_seq_stage1_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.m_stage1_kg_m2,
    )
    m_seq_stage2_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.m_stage2_kg_m2,
    )
    m_seq_residue_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.m_residue_kg_m2,
    )
    mdot_seq_step1_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.mdot_step1_kg_m2_s,
    )
    mdot_seq_step2_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.mdot_step2_kg_m2_s,
    )
    mdot_seq_char_sink_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.mdot_char_sink_kg_m2_s,
    )
    mdot_seq_vol_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.m_dot_vol_kg_m2_s,
    )
    access_factor_stage2_out = _interp_series_to_grid(
        t_out,
        t_solver,
        None if pyro_state_trace is None else pyro_state_trace.access_factor_stage2,
    )
    m_py = _interp_series_to_grid(t_out, t_solver, m_py_solver)
    hrrpua = _interp_series_to_grid(t_out, t_solver, hrrpua_solver)
    mlr = _interp_series_to_grid(t_out, t_solver, mlr_solver)
    mass_total = _interp_series_to_grid(t_out, t_solver, mass_total_solver)
    hrrpua_diag = _interp_series_to_grid(t_out, t_solver, hrrpua_diag_solver)
    if result.y.shape[1] >= 6:
        delta_out = _interp_series_to_grid(t_out, t_solver, np.asarray(result.y[:, 3], dtype=float))
        m_c_out = _interp_series_to_grid(t_out, t_solver, np.asarray(result.y[:, 4], dtype=float))
        L_out = _interp_series_to_grid(t_out, t_solver, np.asarray(result.y[:, 5], dtype=float))
    else:
        delta_out = None
        m_c_out = None
        L_out = None

    if T1 is None:
        T1 = np.full_like(t_out, float(T1_solver[-1]) if T1_solver.size else env_cfg.Tamb, dtype=float)
    if T2 is None:
        T2 = np.full_like(t_out, float(T2_solver[-1]) if T2_solver.size else env_cfg.Tamb, dtype=float)
    if T_mid_out is None and T_mid_solver is not None:
        T_mid_out = np.full_like(t_out, float(T_mid_solver[-1]) if T_mid_solver.size else float(T2_init), dtype=float)
    if M1_out is None:
        M1_out = np.full_like(t_out, float(M1_solver[-1]) if M1_solver.size else M1_init, dtype=float)
    if m_fuel_remaining_out is None:
        m_fuel_remaining_out = np.full_like(t_out, float(m0_kg_m2_solver), dtype=float)
    if m_py is None:
        m_py = np.zeros_like(t_out, dtype=float)
    if hrrpua is None:
        hrrpua = np.zeros_like(t_out, dtype=float)
    if mlr is None:
        mlr = np.zeros_like(t_out, dtype=float)
    if mass_total is None:
        mass_total = np.full_like(t_out, float(m0_kg), dtype=float)

    # ── Flame geometry post-processing: L_f(t), F(t), plume T(z,t), u(z,t) ────
    # Computed from the fully-converged hrrpua (on t_out) using Heskestad (1983)
    # and Drysdale (1999) view factor, plus McCaffrey (1979) centreline plume.
    # Active only when flame_geometry_mode = "heskestad" AND flame_area_m2 > 0.
    # Diagnostic output only — not coupled back into the ODE.
    _flame_height_arr: np.ndarray | None = None
    _flame_vf_arr: np.ndarray | None = None
    _plume_T_arrs: list | None = None
    _plume_u_arrs: list | None = None
    _fl_area_m2_g = getattr(fuel_cfg, "flame_area_m2", None)
    _fl_geom_mode_g = str(getattr(fuel_cfg, "flame_geometry_mode", "deck")).lower()
    if _fl_area_m2_g is not None and float(_fl_area_m2_g) > 0.0 and _fl_geom_mode_g == "heskestad":
        try:
            from model.flame.geometry import (  # noqa: PLC0415
                bed_equivalent_diameter as _bed_eq_diam_g,
                heskestad_flame_height as _hsk_lf_g,
                pool_fire_view_factor as _pf_vf_g,
            )
            from model.flame.plume import mccaffrey_plume as _mc_plume  # noqa: PLC0415
            _D_eq_g = _bed_eq_diam_g(float(_fl_area_m2_g))
            _n_g = t_out.size
            _lf_g = np.empty(_n_g, dtype=float)
            _vf_g = np.empty(_n_g, dtype=float)
            for _gi in range(_n_g):
                _Q_kw_g = max(float(hrrpua[_gi]) * float(_fl_area_m2_g), 0.0)
                _lf_g[_gi] = _hsk_lf_g(_Q_kw_g, _D_eq_g)
                _vf_g[_gi] = _pf_vf_g(_lf_g[_gi], _D_eq_g)
            _flame_height_arr = _lf_g
            _flame_vf_arr = _vf_g
            # Plume evaluation heights — parse semicolon-separated string from deck
            _fl_ph_raw = getattr(fuel_cfg, "flame_plume_heights_m", None)
            if isinstance(_fl_ph_raw, str) and _fl_ph_raw.strip():
                _pl_hts = [float(x.strip()) for x in _fl_ph_raw.replace(",", ";").split(";") if x.strip()]
            elif isinstance(_fl_ph_raw, (list, tuple)):
                _pl_hts = [float(x) for x in _fl_ph_raw]
            else:
                _pl_hts = []
            if _pl_hts:
                _T_amb_g = float(getattr(env_cfg, "Tamb", 293.0) or 293.0)
                _plume_T_arrs = []
                _plume_u_arrs = []
                for _ph_g in _pl_hts:
                    _pT_g = np.empty(_n_g, dtype=float)
                    _pu_g = np.empty(_n_g, dtype=float)
                    for _gi in range(_n_g):
                        _Q_kw_g2 = max(float(hrrpua[_gi]) * float(_fl_area_m2_g), 0.0)
                        _pT_g[_gi], _pu_g[_gi] = _mc_plume(_Q_kw_g2, float(_ph_g), T_amb_K=_T_amb_g)
                    _plume_T_arrs.append(_pT_g)
                    _plume_u_arrs.append(_pu_g)
        except ImportError:
            pass

    pyro_m_remaining = np.zeros_like(t_out, dtype=float)
    pyro_mdot_kin = np.zeros_like(t_out, dtype=float)
    pyro_mdot_cap = np.full_like(t_out, np.nan, dtype=float)
    pyro_mdot_limit = np.full_like(t_out, np.nan, dtype=float)
    pyro_mdot_final = np.zeros_like(t_out, dtype=float)
    pyro_limiter_active = np.zeros_like(t_out, dtype=float)
    pyro_cap_active = np.zeros_like(t_out, dtype=float)
    pyro_gate_active = np.zeros_like(t_out, dtype=float)
    pyro_gate_factor = np.ones_like(t_out, dtype=float)
    sequential_mode_active = (
        pyro_state_trace is not None
        and pyro_state_trace.m_stage1_kg_m2 is not None
        and pyro_state_trace.m_stage2_kg_m2 is not None
        and pyro_state_trace.m_residue_kg_m2 is not None
    )
    if sequential_mode_active:
        pyro_m_remaining[:] = np.asarray(m_fuel_remaining_out, dtype=float)
        if mdot_seq_vol_out is not None:
            pyro_mdot_kin[:] = np.asarray(mdot_seq_vol_out, dtype=float)
            pyro_mdot_final[:] = np.asarray(mdot_seq_vol_out, dtype=float)
        else:
            pyro_mdot_kin[:] = np.asarray(m_py, dtype=float)
            pyro_mdot_final[:] = np.asarray(m_py, dtype=float)
        pyro_mdot_cap[:] = np.nan
        pyro_mdot_limit[:] = np.nan
        pyro_limiter_active[:] = 0.0
        pyro_cap_active[:] = 0.0
        pyro_gate_active[:] = 0.0
        pyro_gate_factor[:] = 1.0
    else:
        for i in range(t_out.size):
            d_i = float(delta_out[i]) if delta_out is not None else None
            m_c_i = float(m_c_out[i]) if m_c_out is not None else None
            L_i = float(L_out[i]) if L_out is not None else None
            m_fuel_i = float(m_fuel_remaining_out[i]) if m_fuel_remaining_out is not None else None
            p = compute_pyrolysis_attribution_terms(
                T1=float(T1[i]),
                M1=float(M1_out[i]),
                fuel_cfg=fuel_cfg,
                delta_py=d_i,
                m_c=m_c_i,
                L=L_i,
                m_fuel_remaining_kg_m2=m_fuel_i if (pyro_mass_source == "fuel_state" and not use_front_limit) else None,
            )
            pyro_m_remaining[i] = float(p["m_remaining_kg_m2"])
            pyro_mdot_kin[i] = float(p["mdot_kin_kg_m2_s"])
            pyro_mdot_cap[i] = float(p["mdot_cap_kg_m2_s"])
            pyro_mdot_limit[i] = float(p["mdot_limit_kg_m2_s"])
            pyro_mdot_final[i] = float(p["mdot_final_kg_m2_s"])
            pyro_limiter_active[i] = float(p["limiter_active"])
            pyro_cap_active[i] = float(p["cap_active"])
            pyro_gate_active[i] = float(p["kinetics_gate_active"])
            pyro_gate_factor[i] = float(p["gate_factor"])

    q_in_mode = q_in_mode_cfg
    q_in_applied_out = np.zeros_like(t_out, dtype=float)
    q_conv_applied_out = np.zeros_like(t_out, dtype=float)
    q_rad_applied_out = np.zeros_like(t_out, dtype=float)
    q_net_surface_out = np.zeros_like(t_out, dtype=float)
    q_conv_loss_out = np.zeros_like(t_out, dtype=float)
    q_rad_loss_out = np.zeros_like(t_out, dtype=float)
    h_amb_out = np.zeros_like(t_out, dtype=float)
    eps_surface_out = np.zeros_like(t_out, dtype=float)
    q_in_source_out: list[str] = []
    for i, (ti, T1i) in enumerate(zip(t_out, T1)):
        q_in_i, q_in_src_i = eval_q_in_incident_W_m2(float(ti), q_in_cfg)
        surf_terms = compute_surface_heat_terms(
            T1=float(T1i),
            q_in=float(q_in_i),
            fuel_cfg=fuel_cfg,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
        )
        q_in_applied_out[i] = float(surf_terms["q_in"])
        q_conv_applied_out[i] = float(surf_terms["q_conv"])
        q_rad_applied_out[i] = float(surf_terms["q_rad"])
        q_net_surface_out[i] = float(surf_terms["q_net_surface"])
        q_conv_loss_out[i] = float(surf_terms["q_conv_loss"])
        q_rad_loss_out[i] = float(surf_terms["q_rad_loss"])
        h_amb_out[i] = float(surf_terms["h_conv_total"])
        eps_surface_out[i] = float(surf_terms["eps_eff"])
        q_in_source_out.append(str(q_in_src_i))

    conv_mode_raw = str(getattr(fuel_cfg, "convection_mode", "auto") or "auto").strip().lower()
    if conv_mode_raw == "auto":
        conv_mode_resolved = "forced" if float(getattr(fuel_cfg, "u_inf_m_s", 0.0)) > 1.0e-9 else "natural"
    elif conv_mode_raw in {"forced", "natural", "mixed"}:
        conv_mode_resolved = conv_mode_raw
    else:
        conv_mode_resolved = conv_mode_raw if conv_mode_raw else "unknown"
    if abs(float(getattr(fuel_cfg, "C_h_conv", 1.0))) <= 1.0e-12 and abs(float(getattr(fuel_cfg, "h_amb", 0.0))) > 0.0:
        conv_mode_resolved = "constant"
    h_amb_model = (
        f"{conv_mode_resolved} + params("
        f"C_h_conv={float(getattr(fuel_cfg, 'C_h_conv', 1.0)):.6g},"
        f"h_amb={float(getattr(fuel_cfg, 'h_amb', 0.0)):.6g},"
        f"u_inf={float(getattr(fuel_cfg, 'u_inf_m_s', 0.0)):.6g},"
        f"L_m={float(getattr(fuel_cfg, 'L_m', 1.0)):.6g},"
        f"orientation={str(getattr(fuel_cfg, 'orientation', 'vertical'))})"
    )
    if has_schedule:
        q_in_value = float(max((q for _, q in q_in_schedule_w_m2), default=0.0))
    elif q_in_constant_cfg_w_m2 is not None:
        q_in_value = float(q_in_constant_cfg_w_m2)
    else:
        q_in_value = float(np.max(q_in_applied_out)) if q_in_applied_out.size else 0.0
    rom_eps_value = float(np.max(eps_surface_out)) if eps_surface_out.size else float(fuel_cfg.eps)

    debug_enable = os.environ.get("PMMA_DEBUG_CSV", "").strip().lower() in {"1", "true", "yes"}
    if debug_enable and thermal_node_order != 2:
        print("[PMMA debug] skipped for thermal_model_order=3 (debug CSV writer assumes 2-node layout)")
    elif debug_enable:
        tag = _slug(case_id or subcase_token or "run_rom")
        debug_path_env = os.environ.get("PMMA_DEBUG_CSV_PATH", "").strip()
        debug_path = Path(debug_path_env) if debug_path_env else Path("test_debug") / f"pmma_debug_{tag}.csv"
        _write_pmma_debug_csv(
            out_path=debug_path,
            t=t_solver,
            y=result.y,
            fuel_cfg=fuel_cfg,
            env_cfg=env_cfg,
            q_raw=q_raw_func,
            q_ramped=q_func,
            ramp_mode=ramp_mode,
            ramp_tau=ramp_tau,
        )
        q_raw_peak = max(float(q_raw_func(float(tt))) for tt in t_solver) if len(t_solver) else 0.0
        q_ramp_peak = max(float(q_func(float(tt))) for tt in t_solver) if len(t_solver) else 0.0
        print(
            "[PMMA debug] "
            f"wrote={debug_path} mode={ramp_mode} tau={ramp_tau:.3g}s "
            f"q_raw_peak={q_raw_peak:.2f} q_ramped_peak={q_ramp_peak:.2f}"
        )

    m_seq_reactive_total_out = None
    if m_seq_stage1_out is not None and m_seq_stage2_out is not None:
        m_seq_reactive_total_out = np.asarray(m_seq_stage1_out, dtype=float) + np.asarray(m_seq_stage2_out, dtype=float)
    seq_mass_balance_max = None
    seq_source_id = None
    reactive_access_mode = "none"
    reactive_access_source_id = None
    if pyro_state_trace is not None:
        seq_mass_balance_max = pyro_state_trace.mass_balance_max_residual_kg_m2_s
        seq_source_id = pyro_state_trace.sequential_source_id
        if pyro_state_trace.reactive_access_mode:
            reactive_access_mode = str(pyro_state_trace.reactive_access_mode)
        reactive_access_source_id = pyro_state_trace.reactive_access_source_id

    # Interpolate smoldering pool to output grid
    _sm_pool_out: Optional[np.ndarray] = None
    if _sm_m_arr_solver is not None:
        _sm_pool_out = _interp_series_to_grid(t_out, t_solver, _sm_m_arr_solver)
        if _sm_pool_out is None:
            _sm_pool_out = np.zeros_like(t_out)

    # Build per-node temperature and char fraction arrays for output
    _T_nodes_out: Optional[list] = None
    _alpha_nodes_out: Optional[list] = None
    if _N_therm >= 3:
        _T_nodes_out = []
        for _ni in range(_N_therm):
            _T_ni_solver = np.asarray(result.y[:, _ni], dtype=float)
            _T_ni_out = _interp_series_to_grid(t_out, t_solver, _T_ni_solver)
            if _T_ni_out is None:
                _T_ni_out = np.full_like(t_out, float(_T_ni_solver[-1]) if _T_ni_solver.size else env_cfg.Tamb)
            _T_nodes_out.append(_T_ni_out)
        if _ode_alpha_all is not None:
            _alpha_nodes_out = []
            for _ni in range(_N_therm):
                _a_solver = _ode_alpha_all[_ni]
                _a_out = _interp_series_to_grid(t_out, t_solver, _a_solver)
                if _a_out is None:
                    _a_out = np.zeros_like(t_out, dtype=float)
                _alpha_nodes_out.append(np.clip(_a_out, 0.0, 1.0))

    return RomSignals(
        t=t_out,
        T_surf=T1,
        T_inner=T2,
        M1_moisture=M1_out,
        hrrpua=hrrpua,
        hrrpua_diag=hrrpua_diag,
        m_py=m_py,
        mlr=mlr,
        mass_total=mass_total,
        mass_total_units="kg_total",
        m_fuel_remaining_kg_m2=m_fuel_remaining_out,
        pyrolysis_mass_source=pyro_mass_source,
        m_seq_stage1_kg_m2=m_seq_stage1_out,
        m_seq_stage2_kg_m2=m_seq_stage2_out,
        m_seq_residue_kg_m2=m_seq_residue_out,
        mdot_seq_step1_kg_m2_s=mdot_seq_step1_out,
        mdot_seq_step2_kg_m2_s=mdot_seq_step2_out,
        mdot_seq_char_sink_kg_m2_s=mdot_seq_char_sink_out,
        mdot_seq_vol_kg_m2_s=mdot_seq_vol_out,
        m_seq_reactive_total_kg_m2=m_seq_reactive_total_out,
        sequential_kinetics_enabled=bool(sequential_mode_active),
        sequential_kinetics_source_id=seq_source_id,
        sequential_mass_balance_max_residual_kg_m2_s=seq_mass_balance_max,
        reactive_access_mode=reactive_access_mode,
        reactive_access_source_id=reactive_access_source_id,
        access_factor_stage2=access_factor_stage2_out,
        time_grid_mode="uniform",
        dt_out=float(dt_out),
        t_solver=t_solver.copy(),
        mass_total_solver=mass_total_solver.copy(),
        m_py_solver=m_py_solver.copy(),
        hrrpua_diag_solver=(hrrpua_diag_solver.copy() if hrrpua_diag_solver is not None else None),
        hoc_eff_raw=float(hoc_eff_raw),
        hoc_units=str(hoc_units_local),
        hoc_eff_J_kg=float(hoc_eff_j_kg),
        hoc_eff=float(hoc_eff_kj_kg),
        q_in_incident_W_m2=q_in_applied_out,
        q_net_into_surface_W_m2=q_net_surface_out,
        q_conv_loss_W_m2=q_conv_loss_out,
        q_rad_loss_W_m2=q_rad_loss_out,
        q_in_W_m2=q_in_applied_out,
        q_conv_W_m2=q_conv_applied_out,
        q_rad_W_m2=q_rad_applied_out,
        q_net_surface_W_m2=q_net_surface_out,
        h_amb_W_m2K=h_amb_out,
        eps_surface=eps_surface_out,
        q_in_mode=q_in_mode,
        q_in_value_W_m2=q_in_value,
        h_amb_model=h_amb_model,
        rom_eps=rom_eps_value,
        q_in_source=q_in_source_out,
        deck_q_in_units_raw=str(deck_q_in_units_raw),
        deck_q_in_constant_raw=deck_q_in_constant_raw,
        cfg_q_in_mode=str(q_in_mode_cfg),
        cfg_q_in_constant_W_m2=float(q_in_constant_cfg_w_m2 or 0.0),
        cfg_q_in_constant_source=str(cfg_q_in_constant_source),
        cfg_q_in_constant_altkey_raw=cfg_q_in_constant_altkey_raw,
        cfg_has_schedule=bool(has_schedule),
        cfg_schedule_len=int(len(q_in_schedule_w_m2)),
        q_in_source_at_t0=str(q_in_source_at_t0),
        q_in_applied_at_t0_W_m2=float(q_in_applied_at_t0),
        qin_guardrail_error=qin_guardrail_error,
        registry_exposure_q_kW_m2=registry_exposure_q_kW_m2,
        pyro_m_remaining_kg_m2=pyro_m_remaining,
        pyro_mdot_kin_kg_m2_s=pyro_mdot_kin,
        pyro_mdot_cap_kg_m2_s=pyro_mdot_cap,
        pyro_mdot_limit_kg_m2_s=pyro_mdot_limit,
        pyro_mdot_final_kg_m2_s=pyro_mdot_final,
        pyro_limiter_active=pyro_limiter_active,
        pyro_cap_active=pyro_cap_active,
        pyro_kinetics_gate_active=pyro_gate_active,
        pyro_gate_factor=pyro_gate_factor,
        area_m2_used=float(area_m2),
        thickness_m_used=(
            float(getattr(fuel_cfg, "thickness_m"))
            if getattr(fuel_cfg, "thickness_m", None) is not None
            else float(np.nan)
        ),
        rho_kg_m3_used=(
            float(getattr(fuel_cfg, "rho"))
            if getattr(fuel_cfg, "rho", None) is not None
            else float(np.nan)
        ),
        T_mid=T_mid_out,
        fuel_cfg_used=fuel_cfg,
        T_nodes=_T_nodes_out,
        alpha_nodes=_alpha_nodes_out,
        m_smolder_pool_kg_m2=_sm_pool_out,
        flame_height_m=_flame_height_arr,
        flame_view_factor_t=_flame_vf_arr,
        plume_T_K=_plume_T_arrs,
        plume_u_m_s=_plume_u_arrs,
    )
