"""Configuration dataclasses for outdoor fire and spray suppression physics.

Outdoor environment parameters follow Rothermel (1972) USDA FS INT-115
conventions.  Spray suppression parameters follow Rasbash (1962) and
Johansson et al. (2018, Fire 2(1):3).

Typical usage in an input deck:
    outdoor.wind_speed_m_s = 3.0          # 10-m reference; Rothermel (1972)
    outdoor.ambient_RH_frac = 0.30        # ambient relative humidity [-]
    outdoor.ambient_T_K = 300.0           # ambient temperature [K]
    outdoor.fuel_depth_m = 0.30           # bed depth [m]; Anderson (1982) GR1
    outdoor.bulk_density_kg_m3 = 0.24     # oven-dry bulk density; Anderson (1982) GR1
    outdoor.sav_ratio_1_m = 7218.0        # SAV ratio [1/m]; Anderson (1982) GR1
    outdoor.initial_moisture_frac = 0.06  # dead fuel MC [-]; Nelson (2000)
    outdoor.fuel_lag_time_s = 3600.0      # 1-hr lag time [s]; Nelson (2000)

    spray.enable = false
    spray.t_start_s = 60.0
    spray.m_dot_water_kg_m2_s = 0.025     # Johansson (2018): W_crit ~0.016–0.042
    spray.eta_evap = 0.70                  # evaporation efficiency; Rasbash (1962)
    spray.foam_cover_frac = 0.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutdoorEnvConfig:
    """Environmental and fuel-bed parameters for outdoor fire scenarios.

    References:
        Rothermel (1972) USDA FS INT-115 — wind speed convention and fuel params
        Anderson (1982) USDA FS INT-122 — 13 NFFL standard fuel models
        Nelson (2000) Can. J. For. Res. 30:1071 — dead fuel moisture lag
    """

    # Wind field
    wind_speed_m_s: float = 0.0
    """10-m reference wind speed [m/s] (Rothermel 1972 convention)."""

    # Ambient atmospheric state
    ambient_RH_frac: float = 0.30
    """Ambient relative humidity [-] (0–1)."""

    ambient_T_K: float = 300.0
    """Ambient air temperature [K]."""

    # Fuel bed geometry (Rothermel 1972 / Anderson 1982 NFFL parameterisation)
    fuel_depth_m: float = 0.30
    """Fuel bed depth [m]; used for Byram (1959) fireline intensity I_B = HRRPUA × depth."""

    bulk_density_kg_m3: float = 3.0
    """Oven-dry bulk density [kg/m³] (Anderson 1982: GR1 = 0.24, SH2 = 0.48)."""

    sav_ratio_1_m: float = 6562.0
    """Fuel particle surface-area-to-volume ratio [1/m]
    (Anderson 1982: GR1 = 7218, GR3 = 4921)."""

    # Dead fuel moisture (Nelson 2000 lag-time model)
    initial_moisture_frac: float = 0.08
    """Initial dead fuel moisture content [kg water / kg dry fuel] at simulation start.
    Typical cured grass: 0.03–0.08.  Nelson (2000)."""

    fuel_lag_time_s: float = 3600.0
    """Moisture equilibration lag time [s].
    1-hr fine fuel: 3600 s; 10-hr: 36000 s.  Nelson (2000)."""

    # Terrain type (controls midflame wind speed conversion)
    terrain: str = "open"
    """Terrain type for wind profile adjustment: 'open' (grass) or 'shrub'.
    Open grassland: U_mf = 0.9 * U_10m.  Shrubland: U_mf = 0.6 * U_10m.
    Rothermel (1972) WAF values."""

    # Free-burning (flame-feedback-only) mode
    free_burning_mode: bool = False
    """When True, apply a brief ignition pulse then cut external irradiance to zero.
    Flame radiation feedback (q_fb) becomes the sole heat source after the pulse.
    Tests whether calibrated kinetics are self-sustaining without external irradiance —
    the fundamental transition from cone calorimeter to outdoor free-burning fire."""

    ignition_q_kW_m2: float = 40.0
    """Irradiance of the ignition pulse [kW/m²].  Applied for t < ignition_duration_s.
    Represents initial ignition source (torch, burning neighbour).
    Rule #1: not an EXP-fitted parameter; represents a physical ignition trigger."""

    ignition_duration_s: float = 30.0
    """Duration of ignition pulse [s].  After this time q_in_external drops to 0.0."""

    # Rothermel (1972) packing-ratio reaction-velocity correction
    rothermel_packing_enable: bool = False
    """When True, multiply gas-phase heat release by Γ'(β)/Γ'(β_op) from
    Rothermel (1972) Eq. 36/38. β_op = 3.348·σ_ft^(−0.8189) is the
    optimum packing ratio for a given SAV; both loose and dense beds
    burn less efficiently. Default False preserves existing deck behavior
    (Rule #11 no silent effects)."""

    # Phase 14aw (2026-05-27): volume-weighted low-Mach projection.  When
    # True, the projection operator becomes ∇·((α_g/ρ)·∇p) and divergence
    # becomes ∇·(α_g·u), where α_g = 1 - α_s.  Canonical E-E projection
    # (Anderson & Jackson 1967; Pember et al. 1998).  Default False:
    # prior in-tree attempt (commit f9b5893, reverted) confirmed an
    # AMG-CG pathology under α_g modulation that breaks marginal-wind
    # Cheney sweep cases (Nat 4% U=0.5/1.0 → ROS=0).  PARDISO path runs
    # mechanically (V_real Lx=40 smoke completes 1046 steps without NaN,
    # proj_iter=1 every step, ‖∇·(α_g u)‖∞ ≤ 1e-8) but the AMG-CG path
    # needs investigation before flipping default.  See memory notes
    # phase14aw_volume_weighted_projection_wip.
    volume_weighted_projection: bool = False
    """Phase 14aw-2 opt-in: gas-volume-fraction weighting for the low-Mach
    projection (∇·((α_g/ρ)·∇p) operator + ∇·(α_g u) divergence).  When
    False (default), the standard ∇·((1/ρ)·∇p) operator + ∇·u divergence
    is used (pre-14aw bit-exact).  Set True ONLY when investigating the
    volume-weighted path; expect AMG-CG instability on marginal-wind
    cases until the conditioning issue is resolved."""

    # Phase 14ap (2026-05-22, re-added on Phase 14ax base 2026-05-29):
    # 3D Synthetic Eddy Method (Jarrin et al. 2006) for inlet turbulence.
    sem_enable: bool = False
    """When True, inject y-asymmetric u/v/w perturbations at the inlet plane
    each step from a fixed-seed Synthetic Eddy Method generator.
    Default False (Rule #11)."""
    sem_seed: int = 42
    """Seed for the PCG64 generator that places eddies and signs."""
    sem_I_t: float = 0.20
    """Turbulence intensity I_t = σ_u'/U for inlet eddies."""
    sem_N: int = 200
    """Number of synthetic eddies (Jarrin 2006 recommendation)."""

    # Phase 14ap-2 (2026-05-22, re-added): cold-flow spin-up window.
    spin_up_s: float = 0.0
    """Cold-flow spin-up duration [s] before combustion is enabled.  During
    [0, spin_up_s] only momentum + turbulence + SEM run.  Default 0.0."""

    # Phase 14aq (2026-05-24, re-added): IC fuel-bed perturbation.
    fuel_pert_enable: bool = False
    """When True, apply y-periodic Fourier perturbation to bed alpha_s and
    m_hemi at IC: factor = 1 + amp · Σₖ cos(2π·k·j/Ny + φₖ).
    Default False (Rule #11)."""
    fuel_pert_amp: float = 0.05
    """Perturbation amplitude (relative).  0.05 = ±5%."""
    fuel_pert_kmax: int = 2
    """Max y-wavenumber for the perturbation."""
    fuel_pert_seed: int = 13
    """Seed for the phase RNG (deterministic per Rule #17)."""

    # Phase 14at re-added 2026-05-30: solid-phase form-drag coefficient
    # promoted from hardcoded constant (drag_3d.C_D_DEFAULT) to deck flag.
    # See drag_3d.py module docstring for the canonical literature values:
    #   0.30 — Wilson & Shaw 1977 (default, dense canopy)
    #   0.20 — Lalic et al. 2004
    #   0.15 — Mueller 2021 (specifically calibrated for pasture grass)
    #   0.50 — Morvan & Dupuy 2004 (FIRESTAR shrub canopy)
    # Cold-flow BL diagnostic 2026-05-30 (reference_cold_flow_BL_diagnostics)
    # showed Wilson-Shaw 0.30 over-extracts in pasture by factor ~2 in IBL
    # at z=0.5m; Mueller 2021 0.15 is the literature-supported pasture value.
    canopy_C_d: float = 0.30
    """Solid-phase canopy form-drag coefficient.  Used as the C_D in the
    Forchheimer/quadratic term `½·C_D·σ·α_s·ρ·|u|·u` of the Ergun two-term
    drag.  Default 0.30 (Wilson & Shaw 1977) preserves prior production
    behavior; literature-supported alternatives in module docstring."""

    # Phase 14at re-added 2026-05-30: Sanz 2003 canopy turbulence
    # closure coefficients promoted from hardcoded constants
    # (turbulence_3d.{BETA_P,BETA_D}_CANOPY_DEFAULT) to deck flags.
    # Sanz 2003 calibrated against WT data for DENSE forest canopy
    # (β_p=1.0, β_d=4.0).  For sparse pasture (LAI < 2), Brunet 1994
    # and Massman 1997 suggest β_d closer to 1.5-2.0 (less sub-grid
    # wake-wake interaction → less TKE short-circuit dissipation).
    canopy_beta_p: float = 1.0
    """Sanz 2003 β_p: mean-flow KE → TKE conversion factor in the canopy
    drag term `S_k = β_p · C_D · σ · α_s · |u|³`.  Default 1.0 (Sanz 2003
    Table 4).  Production source term."""

    canopy_beta_d: float = 4.0
    """Sanz 2003 β_d: sub-grid wake-breakup short-circuit dissipation
    factor in `S_k -= β_d · C_D · σ · α_s · |u| · k`.  Default 4.0
    (Sanz 2003 dense canopy).  For pasture grass try 1.5-2.0."""


@dataclass
class SprayConfig:
    """Spray suppression device parameters.

    Three suppression mechanisms modelled (Rasbash 1962):
    1. Thermal quench:    Q_water = m_dot_w * (c_p_w * dT_w + eta * L_v)
    2. Steam displacement of O2 (foam_cover_frac)
    3. Foam blanket shielding of incoming flux (foam_cover_frac)

    Critical water application rate (Johansson et al. 2018, Fire 2(1):3):
        W_crit [kg/m²/s] = I_B / (eta * (c_p_w * dT_w + L_v))

    References:
        Rasbash (1962) Fire Research Abstracts and Reviews 4–5
        Johansson et al. (2018) Fire 2(1):3
    """

    enable: bool = False
    """Master switch for spray suppression model."""

    t_start_s: float = 0.0
    """Time to begin water/foam application [s]."""

    m_dot_water_kg_m2_s: float = 0.0
    """Water mass application rate [kg/m²/s].
    Johansson (2018) W_crit: light fuels 0.016–0.042, heavy fuels up to 0.35."""

    eta_evap: float = 0.7
    """Evaporation efficiency [-] (fraction of water that vaporises before runoff).
    Well-atomised spray: 0.6–0.8.  Rasbash (1962)."""

    foam_cover_frac: float = 0.0
    """Foam blanket area coverage fraction [-] (0 = no foam, 1 = full coverage).
    Scales both incoming flux and flame feedback: q_eff *= (1 - foam_cover_frac)."""

    # Physical constants — should not normally be changed in a deck
    c_p_water_J_kg_K: float = 4180.0
    """Specific heat of water [J/(kg·K)]."""

    delta_T_water_K: float = 80.0
    """Temperature rise of water droplets from ambient to boiling [K]
    (300 K ambient → 373 K boiling ≈ 73 K; 80 K includes some subcooling)."""

    L_v_J_kg: float = 2_256_000.0
    """Latent heat of vaporisation of water [J/kg] at 100 °C.  NIST Chemistry WebBook."""


def outdoor_env_from_dict(d: dict) -> OutdoorEnvConfig:
    """Build an OutdoorEnvConfig from the outdoor_overrides dict parsed from a deck.

    Performs case-insensitive matching so that deck keys like ``ambient_rh_frac``
    (parser lowercases all keys) match the dataclass field ``ambient_RH_frac``.
    """
    cfg = OutdoorEnvConfig()
    # Build a lowercase → actual-name mapping for case-insensitive lookup
    field_map = {f.lower(): f for f in cfg.__dataclass_fields__}
    for key, val in d.items():
        actual = field_map.get(key.lower())
        if actual is not None:
            setattr(cfg, actual, val)
    return cfg


def spray_config_from_dict(d: dict) -> SprayConfig:
    """Build a SprayConfig from the spray_overrides dict parsed from a deck.

    Performs case-insensitive matching (deck keys are lowercased by the parser).
    """
    cfg = SprayConfig()
    field_map = {f.lower(): f for f in cfg.__dataclass_fields__}
    for key, val in d.items():
        actual = field_map.get(key.lower())
        if actual is not None:
            setattr(cfg, actual, val)
    return cfg
