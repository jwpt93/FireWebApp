from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from .config.defaults import default_configs
from .io.text_input import (
    load_text_input,
    q_in_callable,
    q_inc_ramp_factor,
    ramped_q_in_callable,
    convert_q_in,
    convert_m_py,
    normalize_hoc_units,
    resolve_geometry,
    apply_material_geometry,
)
from .fuel.heat_transfer import open_face_loss_flux
from .fuel.two_node import compute_front_limit_terms, integrate_fuel
from .fuel.pyrolysis import pyrolysis_flux
from .fuel.depletion import apply_depletion


@dataclass
class DemoForcing:
    """Simple prescribed forcing for demo runs."""

    def q_in(self, t: float) -> float:
        if t < 10.0:
            return 5.0e4 * (t / 10.0)
        if t < 40.0:
            return 5.0e4
        if t < 50.0:
            return 5.0e4 * (1.0 - (t - 40.0) / 10.0)
        return 0.0

    def rewet_rate(self, t: float) -> float:
        return 0.0

    def M1_eq(self, t: float) -> float:
        return 0.0


def _q_inc_ramp_factor(t: float, mode: str, tau: float) -> float:
    return float(q_inc_ramp_factor(t, mode, tau))


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
    rows = []
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
        rows.append(
            [
                float(t[i]),
                float(T1[i]),
                float(T2[i]),
                float(M1[i]),
                float(delta_arr[i]),
                float(m_c_arr[i]),
                float(L_arr[i]),
                float(q_raw(float(t[i]))),
                float(q_ramped(float(t[i]))),
                float(_q_inc_ramp_factor(float(t[i]), ramp_mode, ramp_tau)),
                terms["m_dot_kin"],
                terms["m_dot_cap"],
                terms["m_dot_pp"],
                terms["m_dot_used"],
                terms["m_avail"],
                terms["d_delta_dt"],
                terms["delta_ratio"],
                terms["handoff_blend"],
                q_open,
                1.0 if (terms["m_dot_pp"] + 1.0e-12 < terms["m_dot_kin"]) else 0.0,
                1.0 if (terms["handoff_blend"] > 1.0e-9) else 0.0,
                1.0 if (terms["m_avail"] <= 0.0) else 0.0,
            ]
        )

    header = (
        "t_s,T1_K,T2_K,M1,delta_py_m,m_c_kg_m2,L_m,"
        "q_inc_raw_W_m2,q_inc_ramped_W_m2,ramp_factor,"
        "m_dot_kin_kg_m2_s,m_dot_cap_kg_m2_s,m_dot_pp_kg_m2_s,m_dot_used_kg_m2_s,"
        "m_avail_kg_m2,d_delta_dt_m_s,delta_ratio,handoff_blend,q_open_W_m2,"
        "cap_limited,handoff_active,avail_limited"
    )
    np.savetxt(out_path, np.asarray(rows, dtype=float), delimiter=",", header=header, comments="")


def run_demo(input_path: Path | None = None) -> None:
    fuel_cfg, env_cfg, sim_cfg, thresholds = default_configs()

    y0 = np.array([300.0, 300.0, 0.2], dtype=float)
    forcing = DemoForcing()
    q_raw_func = forcing.q_in

    hoc_eff_raw = 1.0
    hoc_units = "kJ/kg"
    if input_path is not None and input_path.exists():
        inputs = load_text_input(input_path)
        hoc_units = normalize_hoc_units(inputs.hoc_units)
        if inputs.hoc_eff is not None:
            hoc_eff_raw = inputs.hoc_eff
        # apply geometry/material-derived values first
        area_m2, L_m = resolve_geometry(inputs, area_default=1.0)
        if L_m is not None:
            fuel_cfg.L_m = L_m
        if inputs.T_sur is not None:
            env_cfg.T_sur = inputs.T_sur
        apply_material_geometry(inputs, fuel_cfg)
        if inputs.pyrolysis_mode:
            fuel_cfg.pyrolysis_mode = inputs.pyrolysis_mode
        if inputs.m_py_schedule:
            schedule = [
                (t, convert_m_py(m, inputs.m_py_units, hoc_eff_raw, hoc_units))
                for t, m in inputs.m_py_schedule
            ]
            fuel_cfg.pyrolysis_mode = "prescribed"
            fuel_cfg.m_py_schedule = schedule

        # apply explicit overrides
        for key, val in inputs.fuel_overrides.items():
            if hasattr(fuel_cfg, key):
                setattr(fuel_cfg, key, val)
        for key, val in inputs.sim_overrides.items():
            if hasattr(sim_cfg, key):
                setattr(sim_cfg, key, val)
        if inputs.Tamb is not None:
            env_cfg.Tamb = inputs.Tamb
        if inputs.t_end is not None:
            sim_cfg.t_end = inputs.t_end
        if inputs.T1 is not None:
            y0[0] = inputs.T1
        if inputs.T2 is not None:
            y0[1] = inputs.T2
        if inputs.M1 is not None:
            y0[2] = inputs.M1

        if inputs.force_htc_zero:
            fuel_cfg.h_amb = 0.0
            fuel_cfg.C_h_conv = 0.0

        def _wrap_preburn(q_func):
            if not (inputs.preburn_enable or inputs.preburn_q_in is not None):
                return q_func
            if inputs.preburn_q_in is None:
                return q_func
            q_pre = convert_q_in(inputs.preburn_q_in, inputs.preburn_units)
            t0 = inputs.preburn_start_s if inputs.preburn_start_s is not None else 0.0
            t1 = inputs.preburn_end_s if inputs.preburn_end_s is not None else t0
            if t1 < t0:
                t0, t1 = t1, t0

            def _q_with_preburn(t: float, base=q_func, qp=q_pre, a=t0, b=t1) -> float:
                return base(t) + (qp if a <= t <= b else 0.0)

            return _q_with_preburn

        if inputs.q_in_schedule:
            schedule = [(t, convert_q_in(q, inputs.q_in_units)) for t, q in inputs.q_in_schedule]
            q_raw_func = _wrap_preburn(q_in_callable(schedule, hold_last=True))

            class _Forcing:
                def q_in(self, t: float) -> float:
                    return float(q_raw_func(t))

                def rewet_rate(self, t: float) -> float:
                    return 0.0

                def M1_eq(self, t: float) -> float:
                    return 0.0

            forcing = _Forcing()
        elif inputs.q_in_constant is not None:
            q_val = convert_q_in(inputs.q_in_constant, inputs.q_in_units)

            def _base_q(_: float, q=q_val) -> float:
                return float(q)

            q_raw_func = _wrap_preburn(_base_q)

            class _Forcing:
                def q_in(self, t: float) -> float:
                    return float(q_raw_func(t))

                def rewet_rate(self, t: float) -> float:
                    return 0.0

                def M1_eq(self, t: float) -> float:
                    return 0.0

            forcing = _Forcing()
        elif inputs.preburn_enable or inputs.preburn_q_in is not None:
            # Apply preburn on top of zero base forcing.
            def _zero(_: float) -> float:
                return 0.0

            q_raw_func = _wrap_preburn(_zero)

            class _Forcing:
                def q_in(self, t: float) -> float:
                    return float(q_raw_func(t))

                def rewet_rate(self, t: float) -> float:
                    return 0.0

                def M1_eq(self, t: float) -> float:
                    return 0.0

            forcing = _Forcing()

    ramp_mode = str(getattr(sim_cfg, "q_inc_ramp_mode", "none") or "none").strip().lower()
    ramp_tau = float(getattr(sim_cfg, "q_inc_ramp_tau", 1.0) or 1.0)

    q_func = ramped_q_in_callable(q_raw_func, ramp_mode, ramp_tau)

    base_forcing = forcing

    class _RampedForcing:
        def q_in(self, t: float) -> float:
            return float(q_func(t))

        def rewet_rate(self, t: float) -> float:
            return float(base_forcing.rewet_rate(t))

        def M1_eq(self, t: float) -> float:
            return float(base_forcing.M1_eq(t))

    forcing = _RampedForcing()

    if sim_cfg.warn_on_initial_temp_offset:
        dT1 = abs(y0[0] - env_cfg.Tamb)
        dT2 = abs(y0[1] - env_cfg.Tamb)
        if dT1 > sim_cfg.initial_temp_warn_K or dT2 > sim_cfg.initial_temp_warn_K:
            print(
                "Warning: initial T1/T2 differ from Tamb by more than "
                f"{sim_cfg.initial_temp_warn_K:.1f} K "
                f"(dT1={dT1:.1f} K, dT2={dT2:.1f} K)."
            )

    result = integrate_fuel(y0, (0.0, sim_cfg.t_end), fuel_cfg, env_cfg, forcing, sim_cfg)
    thermal_order = int(getattr(result, "thermal_node_order", 2) or 2)

    T1 = result.y[:, 0]
    if thermal_order == 3:
        T2_mid = result.y[:, 1]
        T2 = result.y[:, 2]   # deepest node
        M1 = result.y[:, 3]
    else:
        T2_mid = None
        T2 = result.y[:, 1]
        M1 = result.y[:, 2]

    m_py_raw = np.array([pyrolysis_flux(t1, m1, fuel_cfg, t=t) for t1, m1, t in zip(T1, M1, result.t)])
    if fuel_cfg.enable_depletion:
        m_py, _ = apply_depletion(result.t, m_py_raw, fuel_cfg.m_fuel_kg_m2)
    else:
        m_py = m_py_raw

    print(f"final T1_K: {T1[-1]:.2f}")
    if T2_mid is not None:
        print(f"final T2_mid_K: {T2_mid[-1]:.2f}")
    print(f"final T2_K: {T2[-1]:.2f}")
    print(f"final M1: {M1[-1]:.4f}")
    print(
        "pyrolysis flux [kg/m^2/s] min/mean/max: "
        f"{m_py.min():.6f} / {m_py.mean():.6f} / {m_py.max():.6f}"
    )

    csv_path = Path("model_demo.csv")
    if T2_mid is not None:
        header = "t_s,T1_K,T2_mid_K,T2_deep_K,M1,m_py_kg_m2_s"
        data = np.column_stack([result.t, T1, T2_mid, T2, M1, m_py])
    else:
        header = "t_s,T1_K,T2_K,M1,m_py_kg_m2_s"
        data = np.column_stack([result.t, T1, T2, M1, m_py])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"wrote: {csv_path}")

    debug_enable = os.environ.get("PMMA_DEBUG_CSV", "").strip().lower() in {"1", "true", "yes"}
    if debug_enable:
        debug_path = Path(os.environ.get("PMMA_DEBUG_CSV_PATH", "test_debug/pmma_debug_runner.csv"))
        _write_pmma_debug_csv(
            out_path=debug_path,
            t=result.t,
            y=result.y,
            fuel_cfg=fuel_cfg,
            env_cfg=env_cfg,
            q_raw=q_raw_func,
            q_ramped=q_func,
            ramp_mode=ramp_mode,
            ramp_tau=ramp_tau,
        )
        print(f"wrote: {debug_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fuel model demo.")
    parser.add_argument("--input", type=Path, default=None, help="Path to text input file.")
    args = parser.parse_args()
    run_demo(args.input)


if __name__ == "__main__":
    main()
