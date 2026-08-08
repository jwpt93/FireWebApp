"""Outdoor single-element fire runner.

Wraps the existing lumped pyrolysis ROM (model.runner.run_rom) with
outdoor boundary conditions:

  1. Dead fuel moisture pre-conditioning (Nelson 2000 lag ODE)
     → sets initial M1 from pre-ignition drying under ambient T / RH
  2. Wind-enhanced forced convection
     → sets fuel_overrides["u_inf_m_s"] = midflame wind speed
  3. Spray suppression (Rasbash 1962, Johansson et al. 2018)
     → subtracts Q_water from q_in after spray start time
     → applies foam blanket scaling to incident flux

The existing pyrolysis kinetics, thermal ODE, and flame feedback all run
unchanged inside run_rom.

Known limitation: foam blanket scaling is applied to q_in only; the flame
radiation feedback q_fb inside run_rom is not yet foam-scaled.  This is a
conservative bias (slightly over-predicts flame contribution when foam is on).

Usage::
    from pathlib import Path
    from model_outdoor.fuel_element import run_outdoor_element

    signals, diag = run_outdoor_element(Path("inputs/validation_cases/Outdoor_Grass_GR1__no_wind.txt"))

    # Check suppression diagnostics
    print(f"Fireline intensity I_B = {diag['I_B_kW_m']:.1f} kW/m")
    print(f"Critical water rate W_crit = {diag['W_crit_kg_m2_s']:.4f} kg/m²/s")
    print(f"Suppression ratio = {diag['suppression_ratio']:.2f}")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Union

import numpy as np

from model.io.text_input import load_text_input, RomInputs
from model.runner import run_rom

from model_outdoor.config import (
    OutdoorEnvConfig,
    SprayConfig,
    outdoor_env_from_dict,
    spray_config_from_dict,
)
from model_outdoor.moisture import solve_dead_fuel_moisture
from model_outdoor.boundary import (
    midflame_wind_speed,
    byram_flame_length,
    fireline_intensity,
    flame_tilt_angle,
    view_factor_tilt_enhancement,
)
from model_outdoor.suppression import (
    spray_heat_sink_W_m2,
    foam_flux_factor,
    w_critical,
    suppression_summary,
)


def run_outdoor_element(
    deck: Union[Path, str, RomInputs],
    *,
    q_in_kW_m2: Optional[float] = None,
    t_end_s: Optional[float] = None,
    area_m2: float = 0.09,
    Tamb_K: Optional[float] = None,
    M1_init: Optional[float] = None,
    hoc_eff: float = 17000.0,
    case_id: Optional[str] = None,
    pre_dry_duration_s: float = 3600.0,
):
    """Run the ROM for a single outdoor fuel element.

    Applies outdoor boundary conditions (wind, dead fuel moisture, spray)
    on top of the existing single-element pyrolysis ROM.

    Parameters
    ----------
    deck : Path | str | RomInputs
        Input deck path or already-parsed RomInputs.
    q_in_kW_m2 : float, optional
        External irradiance [kW/m²].  If None, taken from deck.
    t_end_s : float, optional
        Simulation end time [s].  If None, taken from deck.
    area_m2 : float
        Fuel element area [m²].  Default 0.09 m² (30 cm × 30 cm cone pan).
    Tamb_K : float, optional
        Ambient temperature [K].  Overrides outdoor.ambient_T_K from deck if set.
    M1_init : float, optional
        Override initial moisture fraction.  If None, uses Nelson (2000) pre-dry result.
    hoc_eff : float
        Effective heat of combustion [kJ/kg].
    case_id : str, optional
        Case identifier for debug output.
    pre_dry_duration_s : float
        Duration of pre-ignition drying ODE [s].  Default 3600 s (1-hr lag).

    Returns
    -------
    signals : RomSignals
        Full pyrolysis ROM output (same as run_rom return value).
    diagnostics : dict
        Outdoor fire diagnostics:
          outdoor_cfg     OutdoorEnvConfig used
          spray_cfg       SprayConfig used
          M1_predry       moisture after pre-drying [-]
          U_mf_m_s        midflame wind speed [m/s]
          peak_HRRPUA     peak HRRPUA [W/m²]
          I_B_kW_m        Byram fireline intensity at peak [kW/m]
          L_f_m           Byram flame length at peak [m]
          theta_rad       flame tilt angle at peak [rad]
          vf_tilt         view factor tilt enhancement at peak [-]
          W_crit_kg_m2_s  critical water application rate [kg/m²/s]
          suppression_ratio  applied / critical water rate
    """
    # ── Parse deck ────────────────────────────────────────────────────────────
    if isinstance(deck, (Path, str)):
        rom_inputs = load_text_input(Path(deck))
    else:
        rom_inputs = copy.deepcopy(deck)

    # ── Build outdoor and spray configs from deck overrides ───────────────────
    outdoor_cfg = outdoor_env_from_dict(rom_inputs.outdoor_overrides)
    spray_cfg = spray_config_from_dict(rom_inputs.spray_overrides)

    # Tamb: prefer explicit argument, then deck outdoor.ambient_T_K, then default
    Tamb = Tamb_K if Tamb_K is not None else outdoor_cfg.ambient_T_K
    rom_inputs.Tamb = Tamb

    # ── Dead fuel moisture pre-conditioning (Nelson 2000) ─────────────────────
    if M1_init is not None:
        M_predry = float(M1_init)
    else:
        M_predry = solve_dead_fuel_moisture(
            t_dry_s=pre_dry_duration_s,
            M0=outdoor_cfg.initial_moisture_frac,
            T_K=outdoor_cfg.ambient_T_K,
            RH_frac=outdoor_cfg.ambient_RH_frac,
            tau_s=outdoor_cfg.fuel_lag_time_s,
        )

    # Set initial moisture on rom_inputs.  M1 in RomInputs maps to the M1
    # initial condition passed to the ODE integrator.  When M1_represents =
    # "fraction" the value is a fraction [0–1] of the moisture pool.
    # For dead fuel moisture we use it as a dimensionless fraction.
    rom_inputs.M1 = M_predry

    # ── Wind convection: set midflame wind speed on fuel_overrides ────────────
    # fuel_overrides["u_inf_m_s"] feeds FuelConfig.u_inf_m_s → heat_transfer.h_conv()
    # using the existing forced flat-plate correlation.
    # Rothermel (1972) WAF applied here.
    U_mf = midflame_wind_speed(outdoor_cfg.wind_speed_m_s, outdoor_cfg.terrain)
    if U_mf > 0.0:
        # Two pathways (plan step 4): wind_speed_m_s is the physical input;
        # direct C_h_conv override is legacy-compatible.
        # fuel.wind_speed_m_s takes precedence.
        rom_inputs.fuel_overrides["u_inf_m_s"] = U_mf

    # ── Spray suppression: adjust q_in_constant / q_in_schedule ─────────────
    #
    # Physical basis: q_net_surface = q_in - q_conv - q_rad - Q_water
    # Subtracting Q_water from q_in yields identical q_net_surface.
    # Foam coverage scales incoming irradiance before spray is added.
    #
    # Rasbash (1962) thermal quench:
    #   Q_water = m_dot_w * (c_p_w * dT_w + eta * L_v)
    # Johansson et al. (2018): W_crit = I_B / Q_water_per_kg
    #
    Q_spray = 0.0
    t_spray = 0.0
    foam_f = 1.0

    if spray_cfg.enable and spray_cfg.m_dot_water_kg_m2_s > 0.0:
        Q_spray = spray_heat_sink_W_m2(spray_cfg)
        t_spray = float(spray_cfg.t_start_s)
        foam_f = foam_flux_factor(spray_cfg)
        _apply_spray_to_q_in(rom_inputs, Q_spray, t_spray, foam_f)

    # ── Free-burning mode: ignition pulse then q_in=0 ─────────────────────────
    if outdoor_cfg.free_burning_mode:
        _apply_free_burning_to_q_in(
            rom_inputs,
            outdoor_cfg.ignition_q_kW_m2,
            outdoor_cfg.ignition_duration_s,
        )

    # ── Run ROM ───────────────────────────────────────────────────────────────
    signals = run_rom(
        q_in_kW_m2=q_in_kW_m2,
        t_end_s=float(t_end_s or rom_inputs.t_end or 600.0),
        area_m2=area_m2,
        Tamb_K=Tamb,
        M1_init=M_predry,
        hoc_eff=hoc_eff,
        subcase_token=case_id or "outdoor",
        rom_inputs=rom_inputs,
        case_id=case_id,
    )

    # ── Post-process: outdoor fire diagnostics ────────────────────────────────
    hrrpua = np.asarray(signals.hrrpua, dtype=float)
    t_arr = np.asarray(signals.t, dtype=float)
    peak_idx = int(np.argmax(hrrpua))
    peak_hrrpua = float(hrrpua[peak_idx])

    I_B = fireline_intensity(peak_hrrpua, outdoor_cfg.fuel_depth_m)
    L_f = byram_flame_length(peak_hrrpua, outdoor_cfg.fuel_depth_m)
    theta = flame_tilt_angle(outdoor_cfg.wind_speed_m_s, L_f, outdoor_cfg.terrain)
    vf_tilt = view_factor_tilt_enhancement(theta)

    sup = suppression_summary(I_B, spray_cfg, outdoor_cfg.fuel_depth_m)

    diagnostics = {
        "outdoor_cfg": outdoor_cfg,
        "spray_cfg": spray_cfg,
        "M1_predry": M_predry,
        "U_mf_m_s": U_mf,
        "peak_HRRPUA_W_m2": peak_hrrpua,
        "I_B_kW_m": I_B,
        "L_f_m": L_f,
        "theta_rad": theta,
        "vf_tilt": vf_tilt,
        **sup,
    }

    return signals, diagnostics


def _apply_spray_to_q_in(
    rom_inputs: RomInputs,
    Q_spray_W_m2: float,
    t_start_s: float,
    foam_factor: float,
) -> None:
    """Modify rom_inputs q_in to account for spray suppression and foam.

    Converts the flux specification to a two-piece schedule:
      [0, t_start_s):     q_in unchanged
      [t_start_s, t_end]: q_in * foam_factor - Q_spray_W_m2  (clamped to ≥ 0)

    Modifies rom_inputs in-place.

    Parameters
    ----------
    rom_inputs : RomInputs
    Q_spray_W_m2 : float
        Spray heat sink [W/m²] (always positive; subtracted from q_in).
    t_start_s : float
        Time spray begins [s].
    foam_factor : float
        Multiplicative reduction of incident flux due to foam blanket (≤ 1).
    """
    t_end = float(rom_inputs.t_end or 600.0)

    # Determine base q_in value in W/m²
    if rom_inputs.q_in_constant is not None:
        from model.io.text_input import _parse_float
        raw = float(rom_inputs.q_in_constant)
        units = str(rom_inputs.q_in_units or "W/m2").strip().lower()
        if units in {"kw/m2", "kw/m^2"}:
            q_base_W_m2 = raw * 1000.0
        else:
            q_base_W_m2 = raw
        # Build two-piece schedule
        q_after = max(0.0, q_base_W_m2 * foam_factor - Q_spray_W_m2)
        rom_inputs.q_in_schedule = [
            (0.0, q_base_W_m2),
            (t_start_s, q_after),
            (t_end + 1.0, q_after),
        ]
        rom_inputs.q_in_units = "W/m2"
        # Clear constant so schedule takes precedence
        rom_inputs.q_in_constant = None

    elif rom_inputs.q_in_schedule is not None:
        # Modify existing schedule: scale values at or after t_start_s
        new_sched = []
        _inserted = False
        for t_s, q_s in rom_inputs.q_in_schedule:
            if t_s >= t_start_s:
                if not _inserted:
                    # Insert a transition point at t_start_s
                    new_sched.append((t_start_s, max(0.0, q_s * foam_factor - Q_spray_W_m2)))
                    _inserted = True
                new_sched.append((t_s, max(0.0, q_s * foam_factor - Q_spray_W_m2)))
            else:
                new_sched.append((t_s, q_s))
        rom_inputs.q_in_schedule = new_sched


def _apply_free_burning_to_q_in(
    rom_inputs: "RomInputs",
    ignition_q_kW_m2: float,
    ignition_duration_s: float,
) -> None:
    """Set q_in to an ignition pulse followed by zero external irradiance.

    Creates a two-piece schedule:
      [0, ignition_duration_s):     ignition_q_kW_m2   (ignition source)
      [ignition_duration_s, t_end]: 0.0                (flame feedback only)

    After the pulse, q_fb from the flame state machine is the sole heat source.
    Flame must be enabled in the deck (fuel.flame_enable = true) for self-sustaining
    burning.

    Physical basis: represents initial ignition by a torch or adjacent burning
    element, followed by self-sustained burning driven by flame radiation feedback.
    Rule #1: ignition_q is not EXP-fitted; it represents a physical ignition source.

    Modifies rom_inputs in-place.
    """
    t_end = float(rom_inputs.t_end or 600.0)
    q_ign_W = ignition_q_kW_m2 * 1000.0  # kW/m² → W/m²

    rom_inputs.q_in_schedule = [
        (0.0,                   q_ign_W),
        (ignition_duration_s,   0.0),
        (t_end + 1.0,           0.0),
    ]
    rom_inputs.q_in_units = "W/m2"
    rom_inputs.q_in_constant = None
