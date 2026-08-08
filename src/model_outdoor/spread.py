"""1-D fire line spread.

Sequential coupling: element 0 is an externally-ignited source (sustained q_in
or free-burning mode); each subsequent element receives the spread radiation flux
from ALL burning predecessors as its external irradiance, then runs its own
pyrolysis ROM with flame feedback enabled.

Spread flux formula (Albini 1981/1985 line-source geometry with Beer-Lambert blocking):
    q_spread = chi_rad × HRRPUA_j × F_adj(j×dx) × Π_{k=1}^{j-1} τ_k

    F_adj(r)  = 0.5 × (1 − r_eff / √(L_f² + r_eff²)),  r_eff = max(r − L_f sin θ, ε)
    τ_k(t)   = exp(−κ × L_f_k(t))

Physical interpretation:
  - F_adj: Albini (1981) line-source view factor — geometric dilution with distance
  - τ_k: Beer-Lambert transmittance through intermediate burning cell k.
    Intermediate flames are optically participating — they partially absorb
    radiation from deeper predecessors before it reaches the target cell.
    κ [m⁻¹] is the mean flame extinction coefficient.
    τ = 1 when a cell is not burning (transparent); τ → 0 for a deeply burning
    tall-flame cell (opaque).
  - (1 + 0.4 sin θ): Albini (1981) tilt enhancement toward unburned fuel.

All predecessors (j = 1, 2, ..., i) contribute simultaneously.  No multi_hop_n
cap is imposed — the cascade is self-limiting because: (a) view factor decays
as 1/r, and (b) blocking transmittance compounds exponentially with predecessor
depth, suppressing contributions from distant cells whose radiation has passed
through many burning intermediates.

Rate of spread (ROS):
    ROS [m/s] = cell_spacing_m × (n_ign − 1) / (G[n_ign] − G[1])
where G[k] is the global wall-clock time at which cell k was ignited.

References:
    Albini (1981) Combustion and Flame 43:155 — flame tilt + forward view factor
    Albini (1985) Combust. Sci. Tech. 42:229 — chi_rad 0.25 for wildland fuels
    Byram (1959) Forest Fire: Control and Use, McGraw-Hill — flame length
    Cheney, Gould & Catchpole (1993) Int. J. Wildland Fire 3:31 — ROS field data
    Morvan & Dupuy (2004) Int. J. Wildland Fire 13(1):115 — κ flame 0.5–1.5 m⁻¹
    Sung et al. (2025) NIST TN 2314 — chi_rad = 0.34 little bluestem (total radiant
        fraction; includes both gas-phase and solid-surface radiation components)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_linalg
from scipy.integrate import solve_ivp

from model.io.text_input import load_text_input, RomInputs
from model_outdoor.fuel_element import run_outdoor_element
from model_outdoor.boundary import (
    byram_flame_length,
    flame_tilt_angle,
    midflame_wind_speed,
    wind_h_conv,
    wind_profile_in_bed,
)
from model_outdoor.config import outdoor_env_from_dict

# Maximum schedule points passed to the ODE integrator per cell.
# The adaptive solver evaluates q_in at every internal step; a dense schedule
# adds lookup overhead without improving accuracy.
_SCHED_N_POINTS = 200


# ── Configuration dataclasses ────────────────────────────────────────────────

@dataclass
class SpreadConfig:
    """Configuration for 1-D fire line spread.

    Parameters
    ----------
    max_cells : int
        Maximum number of fuel elements to simulate (safety cap).
        The adaptive loop exits earlier when ROS converges or fire stops.
    cell_width_m : float
        Width of each fuel cell [m].  Used for fireline geometry only.
    dx_m : float
        Centre-to-centre spacing between adjacent cells [m].
        Typically equal to cell_width_m for touching cells.
    chi_rad_spread : float
        Radiative fraction for spread flux [–].
        Albini (1985): 0.25 for wildland grass fuels.
    hrrpua_ign_kW_m2 : float
        HRRPUA threshold used to detect ignition in the ROM output [kW/m²].
        A cell is considered "ignited" when its HRRPUA first exceeds this value.
        Not a physical ignition trigger — the ROM's internal flame state machine
        controls actual ignition.  This value is for post-processing ROS only.
    kappa_flame_m : float
        Mean flame extinction coefficient [m⁻¹] for inter-element Beer-Lambert
        blocking.  Default 0.5 m⁻¹ from Morvan & Dupuy (2004).
    ros_converge_frac : float
        Stop when the last ros_min_cells inter-ignition intervals agree within
        this fractional tolerance.  0.10 = 10% relative spread in intervals.
        Pure numerical convergence control — not a physics parameter.
    ros_min_cells : int
        Minimum number of cells that must ignite (after cell 0) before the
        convergence check is applied.  Also the sliding window size.
    q_spread_crit_W_m2 : float
        Retained for API compatibility; not used in the current implementation.
    persist_s : float
        Retained for API compatibility; not used in the current implementation.
    """

    max_cells: int = 20
    cell_width_m: float = 0.30
    dx_m: float = 0.30

    chi_rad_spread: float = 0.25
    """Albini (1985) wildland grass; lower than wood default 0.30."""

    hrrpua_ign_kW_m2: float = 10.0
    """HRRPUA threshold for recording ignition time [kW/m²]."""

    kappa_flame_m: float = 0.5
    """Mean flame extinction coefficient [m⁻¹] for Beer-Lambert inter-element blocking.

    Intermediate burning cells partially absorb radiation from deeper predecessors.
    Transmittance through cell k: τ_k = exp(−κ × L_f_k), where L_f_k is the
    Byram flame length of cell k at the moment of evaluation.

    τ = 1 when a cell is dark (L_f = 0); τ decreases as HRRPUA — and therefore
    L_f — grows.  A cell that has burned out returns to τ = 1 automatically.

    Default κ = 0.5 m⁻¹: lower bound of measured values for wildland fire flames.
    Reference: Morvan & Dupuy (2004) Int. J. Wildland Fire 13(1):115 — κ
    0.5–1.5 m⁻¹ for flame radiation in grass and shrub fires (Table 1).
    """

    chi_conv_spread: float = 0.0
    """Radiation wind-enhancement coefficient [–].  DEPRECATED — superseded by
    alpha_conv_preheat (explicit convective pathway).

    Legacy: scaled ALL predecessor radiation by chi_total = 1 + chi_conv × U^B
    (Rothermel 1972 phi_w applied to radiation flux).  Physically conflates
    radiation enhancement with convective preheating; causes ROS ceiling at
    U ≥ 5 m/s because a flux multiplier on radiation cannot produce shorter
    ignition intervals than the source ramp-up time allows.

    Set to 0.0 when alpha_conv_preheat > 0 to avoid double-counting the wind
    contribution.  Retained for backward compatibility; default 0.0.
    """

    alpha_conv_preheat: float = 0.0
    """Convective pre-heating efficiency from nearest burning predecessor [–].

    Represents the fraction of the nearest predecessor's HRRPUA delivered as
    direct convective heat flux to the target cell:

        q_conv(τ) = alpha_conv_preheat × HRRPUA_{i-1}(τ) × U^n_conv_preheat  [W/m²]

    where HRRPUA_{i-1} is in kW/m² and the formula includes a ×1000 W→kW factor.

    Physical basis: at moderate–high wind, hot combustion gases are advected ahead
    of the fire front by the wind and heat unburned fuel by forced convection.
    This mechanism is separate from and additive to the radiative cascade
    (chi_rad × F_adj × HRRPUA × Beer-Lambert blocking).  Unlike radiation, the
    convective pathway:
      • is NOT Beer-Lambert blocked (gas phase, not photons)
      • involves only the NEAREST predecessor (gas cools rapidly with distance)
      • grows as U^n_conv with exponent n_conv = 1.5 (not Beer-attenuated)

    The U^1.5 exponent matches the empirical Cheney (1993) field ROS scaling
    ROS ~ (1 + U)^1.5 and is consistent with the combined mass-flow (∝U) and
    forced-convection heat-transfer coefficient (∝U^0.5) contributions from a
    tilted flame gas plume (Beer 1991).

    Alternative (n_conv = 1.0, linear): supported by Beer (1991) simple model but
    fails to reproduce U^1.5 empirical scaling at high wind — rejected in favour
    of the Cheney-consistent exponent n_conv_preheat = 1.5.

    Calibration (Rule #2/#3):
      Calibration fuel:  GR1 grass, Cheney (1993) Eq. 2, FFDI=10
      Calibration case:  U = 8 m/s (worst structural failure of chi_conv model)
      Pre-declared band: ratio ∈ [0.33, 3.0] all 6 wind cases, monotone
      Method:            scan from 0.005→0.10; pick minimum α giving 6/6 PASS

    Only meaningful when wind_speed_m_s > 0.  Default 0.0 = radiation-only.

    References:
        Beer (1991) Combust. Sci. Tech. 77:55 — convective mechanism in grass fires
        Cheney, Gould & Catchpole (1993) Int. J. Wildland Fire 3(1):31 — U^1.5
        Albini (1981) Combust. Flame 43:155 — tilted flame forward gas transport
    """

    n_conv_preheat: float = 1.5
    """Wind speed exponent for convective pre-heating [–].  Fixed = 1.5.

    Matches empirical Cheney (1993) ROS ~ (1 + U)^1.5 scaling and Beer (1991)
    combined mass-flow + forced-convection theory.  Not a free calibration
    parameter — do not adjust.
    """

    flame_immersion_enable: bool = False
    """Enable direct flame-body immersion + gas-column enthalpy preheating.

    Unified mechanism: when the tilted flame overhangs the target (f_overlap > 0),
    AND for the multi-predecessor enthalpy column (distance-weighted gas preheating).

    Two additive sub-terms, both computed from physical states with no free parameters:

    (1) Flame-body immersion (f_overlap > 0):
        q_imm(τ) = h_conv(U) × (T_flame − T_amb) × f_overlap(τ)
        T_flame = 1473 K (Drysdale 2011 cellulosic adiabatic), T_amb = 300 K.
        h_conv from Beer (1991) forced-convection scaling (wind_h_conv, boundary.py).

    (2) Enthalpy-column gas preheating from all burning predecessors:
        q_gas_i(τ) = (1−χ_rad) × Σ_j HRRPUA_j(τ_j) × w_j
        w_j = (dx/L_decay) × exp(−(i−j)×dx/L_decay)   [distance-weighted sight function]

    L_decay is the gas-column enthalpy decay length derived from fuel-bed structure:
        h_p = Nu × k_gas / d_p;  d_p = 4/σ;  Nu = 2 + 0.6×Re^0.5×Pr^(1/3) (Ranz-Marshall)
        a_v = σ × (ρ_bulk/ρ_particle)       [volumetric surface area, m⁻¹]
        L_decay = ρ_gas × c_p_gas × U_mf / (h_p × a_v)

    Gas properties evaluated at ~800°C (mean of 300 K ambient and 1473 K flame):
        ρ_gas = 0.44 kg/m³,  c_p = 1099 J/kg/K,  μ = 3.7e-5 Pa·s,  k = 0.057 W/m/K

    ρ_particle = 500 kg/m³ (universal dry cellulosic; Drysdale 2011 — not a free parameter).

    Properties:
      • Zero at U = 0 (no advection → no enthalpy transport).
      • Non-divergent: Σ_j w_j → r/(e^r−1) < 1 as N→∞ (r = dx/L_decay).
      • Fuel-type specific: GR1 (σ=7218, dense) L_decay≈1.25m;
        SH2 (σ=5600, sparse) L_decay≈0.72m — large a_v reduces L_decay.
      • Nearest predecessor always dominates (exponential weighting).

    References:
        Drysdale (2011) An Introduction to Fire Dynamics, 3rd ed. — T_ad, ρ_particle
        Beer (1991) Prog. Energy Combust. Sci. — forced-convection h_conv
        Ranz & Marshall (1952) Chem. Eng. Prog. 48:141 — Nu = 2 + 0.6 Re^0.5 Pr^1/3
        Incropera et al. (2007) Fundamentals of Heat Transfer — gas props at 800°C
        Butler et al. (2004) Int. J. Wildland Fire 13(4):481 — flame contact zone
        Albini (1981) Combust. Flame 43:155 — tilt geometry
    """

    flame_jump_enable: bool = False
    """Enable ROS-corrected jump stencil: each cascade ignition event advances
    the fire front by n_jump × dx_m physical cells rather than dx_m.

    Physical basis: the tilted flame from source element extends L_f × sin(θ) metres
    ahead.  When that overhang exceeds the fire-front depth h_fuel, the leading edge
    has advanced beyond the current fire front — each excess dx is an additional jump:
        n_jump = 1                              if overhang ≤ h_fuel
        n_jump = 1 + ⌈(overhang − h_fuel)/dx⌉  otherwise
    (Anderson 1982 fuel-model depth h_fuel as fire-front spatial scale).

    n_jump is constant across the cascade — derived from the source element (cell 0)
    HRRPUA, not the immediate predecessor, to avoid self-reinforcing cascade runaway
    where downstream cells accumulate predecessor flux → higher HRRPUA → larger n_jump.

    This changes two things:
      (a) F_adj is computed at the effective spacing n_jump × dx for all predecessor
          distances, restoring wind-dependent view-factor behaviour at high U.
      (b) The ROS denominator uses the physical advance Σ(n_jump_k × dx) /
          elapsed_time rather than n_cells × dx / elapsed_time.

    All inputs from fuel deck + Byram (1959) + Albini (1981) — no free parameters.

    References:
        Byram (1959) — fireline intensity and flame length
        Albini (1981) Combust. Flame 43:155 — tilt geometry
    """

    ros_converge_frac: float = 0.10
    """Stop when the last ros_min_cells inter-ignition intervals agree within
    this fractional tolerance.  0.10 = 10% relative spread.
    Pure numerical convergence control — not a physics parameter."""

    ros_min_cells: int = 3
    """Minimum ignited cells (after cell 0) before convergence check; also the
    sliding window size.  3 intervals = 4 ignitions required."""

    T_gas_spread_K: float = 1473.0
    """Effective gas temperature for convective exchange of the SOURCE element [K].

    In a real fire front, the burning source element is fully surrounded by hot
    combustion products (≈ T_flame), not cold ambient air.  The conventional
    convective loss term h_conv × (T_surf − T_amb) uses T_amb=300 K and becomes
    a large cooling sink at high wind that collapses the cascade — this is
    physically wrong for an element embedded in the fire front.

    Replacing T_amb with T_gas_spread_K converts the exchange to:
        q_gas = h_conv(U) × (T_gas − T_surf)
    which is a NET HEAT INPUT when T_surf < T_gas (always true during burning).

    Applied to the SOURCE element (cell 0) ONLY.  Cascade cells (i>0) retain
    T_amb=300 K because their hot-gas environment is explicitly captured by Phase 5
    immersion sub-terms (predecessor flame); applying T_gas there would double-count.

    Default: 1473 K (Drysdale 2011 cellulosic adiabatic flame temperature — same
    value used in Phase 5 sub-terms A and B).  User-adjustable for other fuels.
    References:
        Drysdale (2011) Fire Dynamics, Table 1.1 — cellulosic T_flame ≈ 1200–1700 K
        Byram (1959) — source element as ambient fire-front reference condition
    """

    use_tau_r_fire_front_depth: bool = False
    """When True, replace the static h_bed fire-front depth with the Rothermel (1972)
    residence-time formula d_H = τ_r × ROS, iterated to self-consistency.

    τ_r = 384/σ^1.5 [min] (σ in ft⁻¹) — Rothermel (1972) INT-115 Eq. 53.
    d_H = τ_r × ROS — horizontal reaction zone depth.

    Both n_front (stopping condition) and Byram L_f (via I_B = HRRPUA × d_H)
    use d_H instead of h_bed.  Jump stencil fire-front-depth comparison also
    uses d_H.  Outer loop iterates until n_front converges (integer-valued;
    typically 2–4 passes).

    Backward-compatible: defaults False; existing GR1/SH2 scripts unchanged.
    Enable for deep-bed fuels (h_bed > 0.35 m) where h_bed >> d_H.

    Reference: Rothermel (1972) USDA FS INT-115.
    """

    tau_r_l_f_from_h_bed: bool = False
    """When True (requires use_tau_r_fire_front_depth=True): d_H sets n_front only;
    _outdoor_cfg_eff.fuel_depth_m stays as h_bed for Byram L_f and jump stencil.

    Physical basis: d_H = τ_r × ROS is the HORIZONTAL reaction-zone width (how many
    cells burn simultaneously in the spread direction).  L_f = Byram(I_B = HRRPUA × h_bed)
    is the VERTICAL flame height set by the full fuel column (h_bed = Anderson 1982
    bed depth).  These are orthogonal dimensions — Phase 8a conflated them by using
    d_H for both, collapsing L_f from ~0.75m to ~0.006m and eliminating flame radiation.

    Default False preserves Phase 8a behaviour (d_H for both n_front and L_f).
    References: Rothermel (1972) INT-115; Byram (1959) fireline intensity; Anderson (1982).
    """

    tau_r_jump_from_d_H: bool = False
    """When True (requires tau_r_l_f_from_h_bed=True): jump stencil comparison uses d_H
    (horizontal fire-front edge) instead of h_bed.

    Physical basis: the flame leap occurs when the tilted flame tip extends past the
    HORIZONTAL leading edge of the fire front (d_H), not past the full column depth (h_bed).
    For tall grass at U=5 m/s: overhang ≈ 0.64m >> d_H ≈ 0.05m → n_jump = 13.
    This represents the real flame-leap physics where a thin active reaction zone (d_H)
    allows the long flame to jump far ahead.

    Default False → jump comparison uses h_bed (conservative, n_jump=1 for GR3).
    References: Byram (1959); Albini (1981) tilted flame forward transport.
    """

    alpha_conv_h_scale: bool = False
    """When True: effective convective preheat = alpha_conv_preheat × (fuel_depth_m / 0.30).

    Physical basis: Beer (1991) convective preheat scales with fireline intensity ∝ h_bed.
    alpha_conv_preheat=0.010 was calibrated for GR1 (h_bed=0.30m, Anderson 1982).
    For deeper beds, the larger flame column drives proportionally stronger gas convection.
    h_ref = 0.30m is the Anderson (1982) GR1 fuel bed depth (fixed, not a fit parameter).

    This does NOT violate Rule #5 (shared parameters): alpha_conv_preheat=0.010 is unchanged;
    the scaling factor is physically derived from h_bed (a deck parameter, not a calibration).
    Default False preserves the existing calibrated alpha_conv_preheat behaviour.
    References: Beer (1991) Combust. Sci. Tech. 77:55; Anderson (1982) USDA Gen. Tech. Rep.
    """

    thermal_absorption_floor: float = 1.0
    """Beer-Lambert thermal density correction floor for coupled spread [--].

    When < 1.0, enables porous-bed thermal correction for cascade cells in
    run_1d_spread_coupled.  The effective thermal density for each receiver
    cell (i > 0) is:

        f_abs     = 1 - exp(-σ × β × dx)           # Beer-Lambert absorption
        f_thermal = max(f_abs, thermal_absorption_floor)
        ρ_thermal = ρ_eff × f_thermal

    where σ = SAV [1/m], β = ρ_bulk / ρ_particle [-], dx = cell spacing [m].

    Physical basis: σ×β is the extinction coefficient of the porous fuel bed.
    f_abs is the fraction of incident radiation absorbed per cell depth.  The
    blade-scale slab model packs all column fuel mass into a thin element,
    giving thermal inertia proportional to the full fuel load.  Scaling by
    f_abs corrects for the porous structure where only a fraction of the mass
    participates in the thermal response at any given depth.

    The floor prevents source-transient artifacts: if ρ_thermal produces
    per-cell t_ign < ~0.5 s, cells ignite during the source HRRPUA ramp
    (0 → peak in ~3 s), creating non-physical timing.  Floor = 0.50 gives
    ρ_thermal ≥ 114 kg/m³ for GR3 (t_ign ≈ 0.6 s, safe).

    Default 1.0 = no correction (preserves existing behaviour).
    Only affects run_1d_spread_coupled (not run_1d_spread).
    Source cell (i=0) always uses full ρ_eff for correct HRRPUA.
    m_fuel_total_kg_m2 is preserved (fuel depletion unaffected).

    References:
        Beer-Lambert law: I(x) = I₀ × exp(−a_v × x); a_v = σ × β
        Anderson (1982) USDA FS INT-122 — σ, ρ_bulk
        Drysdale (2011) Fire Dynamics — ρ_particle = 500 kg/m³
    """

    # Legacy fields (not used by run_1d_spread)
    q_spread_crit_W_m2: float = 10_000.0
    persist_s: float = 2.0


@dataclass
class SpreadResult:
    """Output of :func:`run_1d_spread`.

    Attributes
    ----------
    t_ignition : list of float or None
        Wall-clock ignition time per cell [s].
        Element 0 is always 0.0.  None if cell never ignited.
    cell_t : list of ndarray
        Wall-clock time arrays for each cell run [s].
    cell_hrrpua : list of ndarray
        HRRPUA [kW/m²] time traces for each cell run.
    ros_m_s : float
        Mean rate of spread [m/s] over all ignited cells.
        0.0 if fewer than 2 cells ignited.
    n_cells_ignited : int
        Number of cells that reached ``hrrpua_ign_kW_m2``.
    spread_cfg : SpreadConfig
        Configuration used.
    """

    t_ignition: List[Optional[float]]
    cell_t: List[np.ndarray]
    cell_hrrpua: List[np.ndarray]
    ros_m_s: float
    n_cells_ignited: int
    spread_cfg: SpreadConfig
    n_jump_list: List[int] = field(default_factory=list)
    """Per-step jump counts (length = n_cells_ignited − 1).  All 1s when
    flame_jump_enable=False.  Physical advance = sum(n_jump_list) × dx_m."""


# ── Physics helpers ───────────────────────────────────────────────────────────


def _adjacent_view_factor(L_f_m: float, dx_m: float, theta_rad: float) -> float:
    """View factor from burning element flame to the adjacent fuel surface.

    Tilt-projected geometry (physically correct for non-zero wind):

        r_eff = max(dx − L_f × sin θ, ε)
        F     = 0.5 × (1 − r_eff / √(L_f² + r_eff²))

    Physical basis: at flame tilt angle θ from vertical, the flame tip projects
    horizontally by L_f × sin θ toward the unburned fuel.  The effective
    line-source-to-surface separation is reduced from dx to r_eff.  When the
    flame tip reaches the receiver (L_f sin θ ≥ dx), r_eff → ε and F → 0.5
    (maximum: half-space, flame directly overhead).

    This replaces the Albini (1985) scalar (1 + 0.4 sin θ) linearisation, which
    saturates at a 40% enhancement and cannot represent the steep increase in
    view factor when the tilted flame tip approaches or overhangs the receiver.
    At θ = 0 the formula reduces exactly to the Albini (1985) no-wind result.

    Reference: Albini (1981) Combustion and Flame 43:155 — tilt geometry;
               Butler et al. (2004) Int. J. Wildland Fire 13(4):481 — tilted
               flame tip projects toward unburned fuel (confirmed field obs).

    Returns 0.0 for zero flame height.  Clamped to [0, 0.5].

    Parameters
    ----------
    L_f_m : float     Flame height (Byram length) [m].
    dx_m : float      Cell spacing (centre-to-centre) [m].
    theta_rad : float Flame tilt angle from vertical [rad].
    """
    if L_f_m <= 0.0:
        return 0.0
    r_eff = max(float(dx_m) - L_f_m * math.sin(theta_rad), 1e-3)
    F = 0.5 * (1.0 - r_eff / math.sqrt(L_f_m**2 + r_eff**2))
    return float(min(max(F, 0.0), 0.5))


def _make_spread_callable(
    t_wall: np.ndarray,
    hrrpua_kW_arr: np.ndarray,
    outdoor_cfg,
    chi_rad: float,
    dx_m: float,
    G_src: float,
) -> Callable[[float], float]:
    """Return a callable q(t_global) → spread flux [W/m²] from one predecessor.

    Spread flux = chi_rad × HRRPUA × F_adj   (Albini 1981/1985 line-source geometry).

    chi_rad is the **total** radiant fraction (NIST TN 2314 Table 8: 0.34 for little
    bluestem).  It includes both gas-phase flame radiation and solid-surface radiation;
    no separate surface-radiation term is applied (that would double-count).

    The predecessor's simulation runs on its own local clock starting at t=0.
    It started at *global* time G_src.  The callable translates:
        t_local = t_global - G_src
    and returns 0 for t_global < G_src (predecessor has not started yet).
    Beyond the trace end, the last flux value is held.

    Pre-computes the Albini spread-flux array once on the predecessor's local
    time axis, then uses np.interp for O(log n) lookup at each ODE step.

    Parameters
    ----------
    t_wall : ndarray   Predecessor's local time axis [s].
    hrrpua_kW_arr : ndarray  Predecessor HRRPUA [kW/m²] on the same axis.
    outdoor_cfg    OutdoorEnvConfig (wind speed, terrain, fuel depth).
    chi_rad : float  Total radiative fraction for spread [-].
    dx_m : float     Centre-to-centre spacing from predecessor to receiver [m].
    G_src : float    Global start time of the predecessor cell [s].
                     G_src = 0 for cell 0; G_src = Σ t_ign[0..k-1] for cell k.

    Returns
    -------
    Callable[[float], float]
        q(t_global) in W/m².  Thread-safe (no shared mutable state).
    """
    # Pre-compute spread-flux array on predecessor's local time axis.
    q_local = np.empty(len(t_wall), dtype=float)
    for k, hrr_kW in enumerate(hrrpua_kW_arr):
        hrr_W = float(hrr_kW) * 1000.0
        L_f = byram_flame_length(hrr_W, outdoor_cfg.fuel_depth_m)
        theta = flame_tilt_angle(outdoor_cfg.wind_speed_m_s, L_f, outdoor_cfg.terrain)
        F_adj = _adjacent_view_factor(L_f, dx_m, theta)
        q_local[k] = max(chi_rad * hrr_W * F_adj, 0.0)

    # Capture arrays by value (closure-safe).
    _t = t_wall.copy()
    _q = q_local
    _G = float(G_src)
    _q_last = float(q_local[-1]) if len(q_local) > 0 else 0.0

    def _callable(t_global: float) -> float:
        t_local = t_global - _G
        if t_local < 0.0:
            return 0.0
        return float(np.interp(t_local, _t, _q, left=0.0, right=_q_last))

    return _callable


def _blocking_transmittance_callable(
    t_wall: np.ndarray,
    hrrpua_kW_arr: np.ndarray,
    outdoor_cfg,
    kappa_m: float,
    G_src: float,
) -> Callable[[float], float]:
    """Return callable τ(t_global) for Beer-Lambert blocking through one intermediate cell.

    The transmittance is:
        τ(t) = exp(−κ × L_f(HRRPUA(t)))

    where L_f is the Byram flame length of the blocking cell at global time t.
    τ = 1 when the cell is dark (not yet ignited or fully burned out).
    τ < 1 during active burning; strongly suppressed for tall, high-HRRPUA flames.

    Parameters
    ----------
    t_wall : ndarray
        Blocking cell's local time axis [s].
    hrrpua_kW_arr : ndarray
        Blocking cell HRRPUA [kW/m²] on the same axis.
    outdoor_cfg
        OutdoorEnvConfig — provides fuel_depth_m for Byram L_f calculation.
    kappa_m : float
        Flame extinction coefficient [m⁻¹].
        Morvan & Dupuy (2004) Int. J. Wildland Fire 13(1):115: 0.5–1.5 m⁻¹.
    G_src : float
        Global start time of this blocking cell [s].

    Returns
    -------
    Callable[[float], float]
        τ(t_global) ∈ (0, 1].  Returns 1.0 before cell starts.
    """
    L_f_arr = np.array([
        byram_flame_length(float(h) * 1000.0, outdoor_cfg.fuel_depth_m)
        for h in hrrpua_kW_arr
    ])
    tau_arr = np.exp(-kappa_m * L_f_arr)  # τ=1 when L_f=0, decreases as flame grows
    _t = t_wall.copy()
    _tau = tau_arr
    _G = float(G_src)

    def _callable(t_global: float) -> float:
        t_local = t_global - _G
        if t_local < 0.0:
            return 1.0  # cell hasn't started yet — transparent
        # right=last value: hold final (typically ≈1 as cell burns out)
        return float(np.interp(t_local, _t, _tau, left=1.0, right=float(_tau[-1])))

    return _callable


def _find_ignition_time(
    t_arr: np.ndarray,
    hrrpua_arr: np.ndarray,
    threshold_kW_m2: float,
) -> Optional[float]:
    """Return wall-clock time when HRRPUA first exceeds threshold, or None.

    Uses linear interpolation between the last sub-threshold and first
    super-threshold ODE output points so that fast-igniting cells (where the
    crossing may span a large ODE step) yield a precise crossing time rather
    than snapping to the nearest output point.  Without interpolation the
    minimum resolvable ignition interval equals the ODE output step, which
    caps ROS at cell_width / dt_out regardless of preheat flux.
    """
    h = np.asarray(hrrpua_arr)
    mask = h >= threshold_kW_m2
    if not np.any(mask):
        return None
    idx = int(np.argmax(mask))
    if idx == 0:
        return float(t_arr[0])
    h0, h1 = float(h[idx - 1]), float(h[idx])
    t0, t1 = float(t_arr[idx - 1]), float(t_arr[idx])
    if h1 == h0:
        return t0
    frac = (threshold_kW_m2 - h0) / (h1 - h0)
    return t0 + frac * (t1 - t0)


def _ros_converged(
    t_ignition: List[Optional[float]],
    min_cells: int,
    converge_frac: float,
) -> bool:
    """Return True when the last min_cells inter-ignition intervals agree within converge_frac.

    Convergence means the instantaneous ROS has reached a quasi-steady value: the
    ratio of the longest to shortest of the last min_cells intervals is within
    (1 + converge_frac).  This is a purely numerical stopping criterion — it detects
    when adding more cells would not change the ROS estimate meaningfully.
    """
    valid = [t for t in t_ignition if t is not None]
    if len(valid) < min_cells + 1:
        return False
    intervals = [valid[k] - valid[k - 1] for k in range(-min_cells, 0)]
    lo, hi = min(intervals), max(intervals)
    if lo <= 0.0:
        return False
    return (hi / lo - 1.0) < converge_frac


# ── Gas-column enthalpy decay length ──────────────────────────────────────────

# Hot-gas properties at ~800°C (arithmetic mean of 300 K ambient and 1473 K flame).
# Incropera et al. (2007) Fundamentals of Heat Transfer, Table A.4 air at ~800 K.
_RHO_GAS = 0.44      # [kg/m³]
_CP_GAS  = 1099.0    # [J/(kg·K)]
_MU_GAS  = 3.7e-5    # [Pa·s]
_K_GAS   = 0.057     # [W/(m·K)]
_PR_GAS  = 0.70      # [-]
_RHO_PARTICLE = 500.0  # [kg/m³] dry cellulosic; Drysdale (2011) — universal constant


def _gas_enthalpy_decay_length(outdoor_cfg, U_mf: float) -> float:
    """Gas-column enthalpy decay length L_decay [m].

    Derived from fuel-bed microstructure with no free parameters:

        d_p   = 4 / σ                  [m] particle diameter from SAV
        Re    = ρ_gas × U_mf × d_p / μ
        Nu    = 2 + 0.6 × Re^0.5 × Pr^(1/3)   (Ranz-Marshall 1952)
        h_p   = Nu × k_gas / d_p       [W/m²/K] particle-scale heat transfer
        β     = ρ_bulk / ρ_particle    [-]  solid volume fraction
        a_v   = σ × β                  [m⁻¹] volumetric surface area
        L     = ρ_gas × c_p × U_mf / (h_p × a_v)   [m]

    Gas props at ~800°C (mean of ambient 300 K and flame 1473 K):
        ρ = 0.44 kg/m³, c_p = 1099 J/kg/K, μ = 3.7e-5 Pa·s, k = 0.057 W/m·K

    References:
        Ranz & Marshall (1952) Chem. Eng. Prog. 48:141 — Nu correlation
        Incropera (2007) Fundamentals of Heat Transfer — gas properties
        Anderson (1982) USDA FS INT-122 — σ, ρ_bulk for NFFL fuel models
        Drysdale (2011) Fire Dynamics — ρ_particle = 500 kg/m³ dry cellulosic
    """
    if U_mf <= 0.0:
        return math.inf  # no advection → no decay (immersion still applies at U>0)
    sigma = outdoor_cfg.sav_ratio_1_m          # [1/m]
    rho_bulk = outdoor_cfg.bulk_density_kg_m3  # [kg/m³]
    d_p = 4.0 / sigma                          # [m]
    Re = _RHO_GAS * U_mf * d_p / _MU_GAS
    Nu = 2.0 + 0.6 * Re**0.5 * _PR_GAS**(1.0/3.0)
    h_p = Nu * _K_GAS / d_p                   # [W/m²/K]
    beta = rho_bulk / _RHO_PARTICLE
    a_v = sigma * beta                         # [m⁻¹]
    if a_v <= 0.0 or h_p <= 0.0:
        return math.inf
    return _RHO_GAS * _CP_GAS * U_mf / (h_p * a_v)  # [m]


# ── Main spread runner ────────────────────────────────────────────────────────

def run_1d_spread(
    deck: Union[Path, str, "RomInputs"],
    spread_cfg: Optional[SpreadConfig] = None,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
) -> SpreadResult:
    """Run 1-D fire line spread simulation.

    Element 0 is the ignition source; it uses the deck's q_in configuration
    unchanged (typically a constant irradiance or free_burning_mode pulse).
    Each subsequent element (i > 0) receives the spread radiation flux from
    element i-1 as its external irradiance and runs its own pyrolysis ROM.

    For grass fires, element 0 should have a sustained external flux
    (e.g. ``q_in_constant = 50 kW/m²``) to maintain high HRRPUA that produces
    meaningful spread flux.  The free_burning_mode deck (40 kW/m² / 30 s pulse)
    is insufficient: the cone element never reaches the HRRPUA levels needed
    for spread (Phase 1 POC result, 2026-03-27).

    Parameters
    ----------
    deck : Path | str | RomInputs
        Input deck path or already-parsed inputs.
        Element 0 uses this deck as-is (q_in, flame settings, kinetics).
        All subsequent elements use the same kinetics/material parameters
        but with q_in replaced by the computed spread flux schedule.
    spread_cfg : SpreadConfig, optional
        Line spread geometry and ignition parameters.  Defaults to
        ``SpreadConfig()`` (5 cells, 0.30 m spacing, chi_rad=0.25).
    wind_speed_m_s : float
        10-m wind speed [m/s]; overrides ``outdoor.wind_speed_m_s`` in deck.
    max_wall_time_s : float
        Simulation end time [s]; same for all cells.

    Returns
    -------
    SpreadResult
        Ignition times, HRRPUA traces per cell, rate of spread, diagnostics.

    Notes
    -----
    All burning predecessors j = 1, 2, ..., i contribute to cell i's spread
    flux.  Each predecessor's contribution is attenuated by Beer-Lambert
    blocking through the j−1 intermediate burning cells between it and the
    receiver.  The cascade is self-limiting: view factor decays with distance
    and blocking compounds, so the sum converges without an explicit hop cap.

    The rate of spread is computed as::

        ROS = dx_m × (n_ign − 1) / (G[n_ign] − G[1])

    where G[k] is the global wall-clock time at which cell k was ignited.
    """
    if spread_cfg is None:
        spread_cfg = SpreadConfig()

    # ── Parse base deck ───────────────────────────────────────────────────────
    if isinstance(deck, (Path, str)):
        ri_base = load_text_input(Path(deck))
    else:
        ri_base = copy.deepcopy(deck)

    ri_base.outdoor_overrides["wind_speed_m_s"] = float(wind_speed_m_s)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    # Source element (cell 0) runs at ACTUAL wind speed with T_gas = T_flame.
    # In a real fire front, burning fuel is surrounded by hot combustion products
    # (T_gas ≈ T_flame ≈ 1473 K), not cold ambient air (300 K).  With T_gas=T_flame,
    # the convective exchange h_conv×(T_gas−T_surf) becomes a HEAT INPUT, correctly
    # representing the fire-front environment.  Previously wind was zeroed here as
    # a workaround to prevent the cold-ambient cooling sink from collapsing the cascade.
    #
    # Cascade cells (i>0) retain wind=0 (zeroed below at ri_i assignment) because
    # their hot-gas heating is explicitly captured by Phase 5 immersion sub-terms
    # from the predecessor flame; applying T_gas there would double-count.
    #
    # Wind effects (tilt → n_jump, chi_conv, F_adj) use outdoor_cfg throughout.
    # Source element T_gas set below via ri_base.Tamb before running cell 0.
    _ri_base_Tamb_saved = ri_base.Tamb  # save original (restore after source run)

    # ── Rothermel phi_w wind exponent ─────────────────────────────────────────
    # B = 0.02526 × σ_ft^0.54  (Rothermel 1972 INT-115, σ in ft⁻¹)
    # Converts SAV from m⁻¹ to ft⁻¹ (1 ft⁻¹ = 3.281 m⁻¹).
    # chi_total = 1 + chi_conv × U^B  (super-linear in U).
    _sav_ft = outdoor_cfg.sav_ratio_1_m / 3.281
    _phi_w_B = 0.02526 * (_sav_ft ** 0.54)

    # ── τ_r self-consistent fire-front depth (Phase 8, opt-in) ───────────────
    # Rothermel (1972) INT-115 Eq. 53: τ_r = 384/σ^1.5 [min] (σ in ft⁻¹).
    # d_H = τ_r × ROS is the physical horizontal reaction zone depth.
    # When use_tau_r_fire_front_depth=True, replaces h_bed in BOTH n_front
    # formula and Byram L_f (I_B = HRRPUA × d_H).  Outer loop iterates until
    # n_front(d_H) converges (integer-valued; typically 2–4 passes).
    # Backward-compatible: when False (default), _d_H_eff = h_bed throughout.
    # Reference: Rothermel (1972) USDA FS INT-115.
    if spread_cfg.use_tau_r_fire_front_depth:
        _sav_ft_tau = outdoor_cfg.sav_ratio_1_m / 3.281
        _tau_r_s = (384.0 / _sav_ft_tau ** 1.5) * 60.0  # seconds
    # Warm-start: h_bed (same as normal mode).  τ_r iteration then converges
    # downward from this upper bound to the self-consistent d_H = τ_r × ROS.
    # Starting at dx_m would find the trivial n_front=1 fixed point immediately.
    _d_H_eff = outdoor_cfg.fuel_depth_m
    _outdoor_cfg_eff = copy.copy(outdoor_cfg)

    # ── Run source element (cell 0) once — result is τ_r-independent ────────
    # Cell 0's simulation does not depend on _outdoor_cfg_eff.fuel_depth_m;
    # fuel_depth_m only affects cascade spread-flux / L_f for cells i > 0.
    # Caching cell 0 avoids running its 300s ODE on every τ_r iteration.
    ri_base.Tamb = spread_cfg.T_gas_spread_K  # fire-front gas temperature
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_base_Tamb_saved  # restore for cascade cells
    _signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    _src_t = np.asarray(_signals_src.t, dtype=float)
    _src_hrrpua = np.asarray(_signals_src.hrrpua, dtype=float)

    for _tau_pass in range(6 if spread_cfg.use_tau_r_fire_front_depth else 1):
        # Gate: when tau_r_l_f_from_h_bed=True, d_H sets n_front ONLY;
        # fuel_depth_m (= h_bed) is preserved for Byram L_f and jump stencil.
        # Physical basis: d_H is the HORIZONTAL reaction zone; h_bed is the
        # VERTICAL column height — orthogonal dimensions (Phase 8b correction).
        if not (spread_cfg.use_tau_r_fire_front_depth
                and spread_cfg.tau_r_l_f_from_h_bed):
            _outdoor_cfg_eff.fuel_depth_m = _d_H_eff
        # At quasi-steady ROS, d_H is the physical reaction zone depth.
        # When use_tau_r_fire_front_depth=True, d_H = τ_r × ROS replaces h_bed;
        # otherwise d_H = h_bed (Anderson 1982 fuel bed depth).
        _n_front_eff = max(1, min(
            spread_cfg.max_cells,
            round(_d_H_eff / spread_cfg.dx_m),
        ))

        # Initialize with cached cell 0 result.  G[1]=0: cell 1 starts when
        # cell 0 ignites, which is at t=0 (source element, always immediate).
        cell_t: List[np.ndarray] = [_src_t]
        cell_hrrpua: List[np.ndarray] = [_src_hrrpua]
        t_ignition: List[Optional[float]] = [0.0]  # cell 0: ignited at t=0
        _n_jump_list: List[int] = []
    
        # G[k] = global start time of cell k.  Cell 1 starts at t=0 (source
        # ignites immediately); subsequent cells start when predecessor ignites.
        G: List[float] = [0.0, 0.0]  # G[0]=0 (cell 0), G[1]=0 (cell 1 start)
    
        i = 1  # cell 0 cached; start from cascade cells
        while True:
            G_i = G[i]
            # Cell i runs for the remaining simulation window on the global clock.
            t_remain = max(max_wall_time_s - G_i, 1.0)
    
            # ── Deck for element i (always i>0; cell 0 is pre-cached) ─────────
            # Cascade cells receive spread flux; wind BC zeroed below.
            ri_i = copy.deepcopy(ri_base)
            # Source element (i=0): run at actual wind with fire-environment T_gas.
            # Cascade cells (i>0): deepcopy of ri_base (wind will be zeroed below).
    
            # Disable free_burning_mode; q_in from spread callable.
            ri_i.outdoor_overrides["free_burning_mode"] = False
    
            # ── Jump stencil: compute effective cell spacing ──────────────────
            # When flame_jump_enable=True, the tilted predecessor flame overhangs
            # n_jump = ⌈L_f × sin θ / dx⌉ cells simultaneously.  The fire front
            # advances n_jump × dx per ignition event rather than dx.
            # F_adj is evaluated at j × n_jump × dx (not saturated at ε for j=1).
            # Physical basis: Byram (1959) flame length + Albini (1981) tilt.
            # n_jump derived from source-element (cell 0) HRRPUA — no free parameters.
            if spread_cfg.flame_jump_enable and wind_speed_m_s > 0.0:
                # n_jump is computed from the SOURCE element (cell 0) HRRPUA, not the
                # immediate predecessor.  Using cell_hrrpua[i-1] causes cascade runaway
                # because downstream cells accumulate predecessor flux → higher HRRPUA
                # → larger n_jump → faster ignition → yet higher HRRPUA.  The source
                # element represents the ambient fire-front condition (Byram 1959):
                # a steady quasi-1-D fire front in equilibrium with the fuel bed.
                _peak_src = float(np.max(cell_hrrpua[0]))
                _L_f_pred = byram_flame_length(
                    _peak_src * 1000.0, _outdoor_cfg_eff.fuel_depth_m
                )
                _theta_pred = flame_tilt_angle(
                    wind_speed_m_s, _L_f_pred, outdoor_cfg.terrain
                )
                _overhang = _L_f_pred * math.sin(_theta_pred)
                # Gate: when tau_r_jump_from_d_H=True, jump comparison uses d_H
                # (horizontal reaction-zone edge) not h_bed (column depth).
                # Physical basis: flame leap occurs when the flame tip extends past
                # the thin active reaction zone (d_H), not the full column (h_bed).
                # Default: use _outdoor_cfg_eff.fuel_depth_m (h_bed when tau_r_l_f_from_h_bed,
                # or d_H when Phase 8a mode).
                if (spread_cfg.use_tau_r_fire_front_depth
                        and spread_cfg.tau_r_jump_from_d_H):
                    _fire_front_depth = _d_H_eff
                else:
                    _fire_front_depth = _outdoor_cfg_eff.fuel_depth_m
                if _overhang <= _fire_front_depth:
                    # Flame tip lands within fire-front body — no leading-edge advance.
                    _n_jump = 1
                else:
                    # Flame tip extends beyond the fire-front depth; each excess dx
                    # is an additional cell the front advances per ignition event.
                    # Physical basis: fuel_depth_m (Anderson 1982) defines the spatial
                    # scale of the active fire front; jumps beyond it are real advance.
                    _n_jump = 1 + int(
                        math.ceil((_overhang - _fire_front_depth) / spread_cfg.dx_m)
                    )
            else:
                _n_jump = 1
            _n_jump_list.append(_n_jump)
            _dx_eff = float(_n_jump) * spread_cfg.dx_m

            # Build spread flux callables — one per predecessor.
            # All j = 1..i predecessors contribute; no hop cap is imposed.
            # Each callable q_j(t_global) returns chi_rad × HRRPUA_j × F_adj(j×dx_eff),
            # the unblocked Albini spread flux from predecessor j.
            q_callables: List[Callable[[float], float]] = []
            for j in range(1, i + 1):
                q_callables.append(
                    _make_spread_callable(
                        cell_t[i - j],
                        cell_hrrpua[i - j],
                        _outdoor_cfg_eff,
                        spread_cfg.chi_rad_spread,
                        float(j) * _dx_eff,
                        G_src=G[i - j],
                    )
                )
    
            # Build Beer-Lambert blocking callable chains.
            # For predecessor j, the radiation from cell (i-j) passes through
            # intermediate cells (i-j+1), (i-j+2), ..., (i-1) before reaching i.
            # That is j-1 intermediate cells, indexed k = 1..j-1 from the receiver.
            # blocking_per_pred[j-1] is a list of transmittance callables τ_k(t_global)
            # for each intermediate cell.  Product of all τ_k gives total transmittance.
            blocking_per_pred: List[List[Callable[[float], float]]] = []
            for j in range(1, i + 1):
                blk: List[Callable[[float], float]] = []
                for k in range(1, j):  # k=1: nearest to receiver, k=j-1: adjacent to pred
                    blk.append(
                        _blocking_transmittance_callable(
                            cell_t[i - k],
                            cell_hrrpua[i - k],
                            _outdoor_cfg_eff,
                            spread_cfg.kappa_flame_m,
                            G_src=G[i - k],
                        )
                    )
                blocking_per_pred.append(blk)
    
            # Wrap into a single q_in on cell i's LOCAL time axis
            # (τ = t_global - G_i).  Convective pre-heating scales total flux
            # by (1 + χ_conv × U) (Rothermel 1972 INT-115 Eq. 47).
            # chi_total = 1 + chi_conv × U^B  (Rothermel 1972 INT-115 phi_w form)
            # B derived from fuel SAV at runtime; chi_conv calibrated at U_cal=5 m/s.
            chi_total = 1.0 + (
                spread_cfg.chi_conv_spread * (wind_speed_m_s ** _phi_w_B)
                if spread_cfg.chi_conv_spread > 0.0 and wind_speed_m_s > 0.0
                else 0.0
            )
    
            def _q_in_local(
                tau: float,
                _G: float = G_i,
                _chi: float = chi_total,
                _q_cbs: List[Callable[[float], float]] = q_callables,
                _blk: List[List[Callable[[float], float]]] = blocking_per_pred,
            ) -> float:
                t_global = tau + _G
                q_total = 0.0
                for q_cb, blk_cbs in zip(_q_cbs, _blk):
                    q_j = q_cb(t_global)
                    for tau_cb in blk_cbs:
                        q_j *= tau_cb(t_global)
                    q_total += q_j
                return _chi * q_total
    
            # Build schedule on a union of ALL predecessors' actual ODE time axes
            # (both flux traces and blocking traces share the same cells' time axes).
            # Dense near ignition; coarse baseline ensures coverage everywhere.
            tau_pieces: List[np.ndarray] = [np.linspace(0.0, max_wall_time_s, 50)]
            for j in range(1, i + 1):
                t_pred = cell_t[i - j]
                t_offset_j = G_i - G[i - j]
                tau_j = t_pred - t_offset_j
                tau_j = tau_j[(tau_j >= 0.0) & (tau_j <= max_wall_time_s)]
                if len(tau_j) > 0:
                    tau_pieces.append(tau_j)
            t_grid = np.unique(np.concatenate(tau_pieces))
            q_grid = np.array([_q_in_local(t) for t in t_grid], dtype=float)
    
            # ── Convective pre-heating from nearest burning predecessor ───────
            # Hot combustion gases advected ahead by wind from cell i-1.
            # q_conv(τ) = alpha × HRRPUA_{i-1}(τ + G_i − G_{i-1}) × U^n  [W/m²]
            # Additive to radiation; no Beer-Lambert blocking (gas phase).
            # Only nearest predecessor — gas temperature decays rapidly with
            # distance (Beer 1991: characteristic length ~ flame scale ~0.3 m).
            if spread_cfg.alpha_conv_preheat > 0.0 and wind_speed_m_s > 0.0:
                # alpha_conv_h_scale: Beer (1991) convective preheat ∝ h_bed.
                # alpha_conv=0.010 calibrated for GR1 (h_ref=0.30m, Anderson 1982).
                # Scale factor = fuel_depth_m / h_ref (no new free parameter).
                _alpha_conv_eff = spread_cfg.alpha_conv_preheat
                if spread_cfg.alpha_conv_h_scale:
                    _alpha_conv_eff *= outdoor_cfg.fuel_depth_m / 0.30
                _conv_scale = (
                    _alpha_conv_eff
                    * (wind_speed_m_s ** spread_cfg.n_conv_preheat)
                    * 1000.0  # kW/m² → W/m²
                )
                # Gap 2 fix (Phase 9): convective preheat decays with stride distance.
                # When jump stencil is active (dx_eff > dx_m), the hot gas stream cools
                # over the stride distance.  Beer (1991): exponential decay with
                # characteristic length L_decay (same as enthalpy column sub-term B).
                # At dx_eff = dx_m (no jump), decay factor = 1.0 (no change).
                if _dx_eff > spread_cfg.dx_m:
                    _U_mf_conv = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)
                    _L_decay_conv = _gas_enthalpy_decay_length(
                        _outdoor_cfg_eff, _U_mf_conv
                    )
                    if _L_decay_conv < math.inf:
                        _conv_scale *= math.exp(
                            -(_dx_eff - spread_cfg.dx_m) / _L_decay_conv
                        )
                _t_near = cell_t[i - 1]
                _h_near = cell_hrrpua[i - 1]   # kW/m²
                _G_near = G[i - 1]
                for k, tau in enumerate(t_grid):
                    t_src = tau + G_i - _G_near  # elapsed time in predecessor frame
                    if 0.0 < t_src <= _t_near[-1]:
                        q_grid[k] += _conv_scale * float(
                            np.interp(t_src, _t_near, _h_near, left=0.0, right=0.0)
                        )
    
            # ── Flame immersion + enthalpy-column gas preheating ─────────────
            # Unified block (flame_immersion_enable).  Two additive sub-terms:
            #
            # (A) Direct flame-body immersion from nearest predecessor only:
            #     q_imm = h_conv(U) × (T_flame − T_amb) × f_overlap(τ)
            #     f_overlap = max(0, sin θ − dx/L_f)  (Byram/Albini tilt geometry)
            #     T_flame = 1473 K (Drysdale 2011), T_amb = 300 K.
            #     h_conv = wind_h_conv(U)  (Beer 1991, boundary.py).
            #     Active only when L_f × sin θ > dx (flame physically overhangs target).
            #
            # (B) Distance-weighted enthalpy preheating from ALL predecessors j<i:
            #     q_gas = (1−χ_rad) × Σ_j  HRRPUA_j(τ_j) × w_j
            #     w_j = (dx/L_decay) × exp(−(i−j)×dx/L_decay)
            #     L_decay = ρ_gas×c_p×U_mf / (h_p×a_v)  (fuel-bed microstructure)
            #     h_p from Ranz-Marshall Nu = 2+0.6Re^0.5Pr^1/3 on d_p = 4/σ.
            #     Self-limiting: Σw → r/(e^r−1) < 1 (r = dx/L_decay).
            #
            # No free calibration parameters — all quantities from fuel deck + gas props.
            if spread_cfg.flame_immersion_enable and wind_speed_m_s > 0.0:
                _U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)
                _h_conv_imm = wind_h_conv(wind_speed_m_s, outdoor_cfg.terrain)
                _dT_imm = 1473.0 - 300.0  # T_flame − T_amb [K]
                _L_decay = _gas_enthalpy_decay_length(_outdoor_cfg_eff, _U_mf)
                # Gap 3: normalization uses dx_eff (actual stride) for consistency.
                _r_decay = _dx_eff / _L_decay if _L_decay < math.inf else 0.0
                _chi_conv_frac = 1.0 - spread_cfg.chi_rad_spread
    
                # (A) nearest predecessor: direct flame-body immersion.
                #
                # f_overlap = max(0, sin θ − dx/L_f): geometric fraction of target
                # cell body that is overlapped by the tilted predecessor flame.
                # L_f from Byram (1959) using instantaneous element HRRPUA × fuel_depth.
                #
                # Note: F_adj (primary radiation view factor) saturates at ~0.5 for
                # all wind speeds because L_f × sin(θ) >> dx even at U=1 m/s — the
                # flame already overhangs the target cell.  Switching to fire-front
                # ROS-based L_f does not change F_adj (confirmed: r_eff → ε, F→0.499
                # for both L_f formulations at all U).  Concavity of model ROS vs U
                # relative to Cheney (1993) is therefore structural: primary radiation
                # is constant; wind only adds convective/immersion terms that have
                # diminishing returns.  Rule #4: structural limitation, accepted.
                _t_near = cell_t[i - 1]
                _h_near = cell_hrrpua[i - 1]
                _G_near = G[i - 1]
                for k, tau in enumerate(t_grid):
                    t_src = tau + G_i - _G_near
                    if 0.0 < t_src <= _t_near[-1]:
                        hrr_kW = float(
                            np.interp(t_src, _t_near, _h_near, left=0.0, right=0.0)
                        )
                        if hrr_kW > 0.0:
                            _L_f = byram_flame_length(
                                hrr_kW * 1000.0, _outdoor_cfg_eff.fuel_depth_m
                            )
                            if _L_f > 0.0:
                                _theta = flame_tilt_angle(
                                    wind_speed_m_s, _L_f, outdoor_cfg.terrain
                                )
                                # Gap 1 fix (Phase 9): immersion overlap uses dx_m
                                # (nearest physical cell spacing), not dx_eff (stride).
                                # When the jump stencil is active (n_jump > 1), the
                                # flame still overhangs the nearest unburned fuel at
                                # distance dx_m.  Using dx_eff zeroes the immersion
                                # term when dx_eff ≈ overhang (Butler et al. 2004:
                                # fuel at the flame tip receives 100-200 kW/m² contact).
                                _f_ov = max(
                                    0.0, math.sin(_theta) - spread_cfg.dx_m / _L_f
                                )
                                q_grid[k] += _h_conv_imm * _dT_imm * _f_ov
    
                # (B) all predecessors: enthalpy-column distance-weighted preheating
                # Gap 3 fix (Phase 9): use dx_eff (actual stride) not dx_m for distances.
                # When jump stencil is active, predecessor j is at j × dx_eff physical
                # distance, not j × dx_m.  Using dx_m overestimates the gas-column
                # contribution from distant predecessors.
                if _r_decay > 0.0:
                    _dx_enthalpy = _dx_eff  # physical distance per cascade step
                    for j_rel in range(1, i + 1):   # j_rel = i−j (distance in cells)
                        _w_j = _r_decay * math.exp(-j_rel * _dx_enthalpy / _L_decay)
                        if _w_j < 1e-6:
                            break  # exponential tail negligible — stop early
                        _j_abs = i - j_rel           # absolute predecessor index
                        _t_j = cell_t[_j_abs]
                        _h_j = cell_hrrpua[_j_abs]
                        _G_j = G[_j_abs]
                        for k, tau in enumerate(t_grid):
                            t_src = tau + G_i - _G_j
                            if 0.0 < t_src <= _t_j[-1]:
                                hrr_kW = float(
                                    np.interp(t_src, _t_j, _h_j, left=0.0, right=0.0)
                                )
                                if hrr_kW > 0.0:
                                    # Gas temperature at fire front ≈ T_flame = 1473 K
                                    # regardless of HRRPUA.  Distance-weighting via w_j
                                    # captures decay of hot-gas column along wind path.
                                    # Using HRRPUA here would create positive cascade
                                    # feedback (higher burn → hotter gas → faster ignition
                                    # → higher burn…) that diverges at low wind.
                                    q_grid[k] += _h_conv_imm * _dT_imm * _w_j
    
            ri_i.q_in_schedule = list(zip(t_grid.tolist(), q_grid.tolist()))
            ri_i.q_in_units = "W/m2"
            ri_i.q_in_constant = None
    
            # Receiving-element wind BC: in the fire-front context the gas
            # immediately adjacent to the unburned fuel is hot combustion gas
            # advected from the flame, not ambient-temperature air.  Applying
            # h_c × (T_amb - T_surf) would give an incorrect cooling sink.
            # Wind's correct effect (tilt-enhanced F_adj and convective pre-heat)
            # is already embedded in the schedule above.  Zero the wind BC so
            # convective cooling is not double-applied.
            # Physical basis: Albini (1985) radiation-only spread model.
            ri_i.outdoor_overrides["wind_speed_m_s"] = 0.0
    
            # ── Run element ───────────────────────────────────────────────────────
            signals, _ = run_outdoor_element(ri_i, t_end_s=t_remain)
    
            t_arr = np.asarray(signals.t, dtype=float)
            hrr_arr = np.asarray(signals.hrrpua, dtype=float)
            cell_t.append(t_arr)
            cell_hrrpua.append(hrr_arr)
    
            # ── Record ignition time and check stopping conditions ────────────────
            t_ign = _find_ignition_time(t_arr, hrr_arr, spread_cfg.hrrpua_ign_kW_m2)
            t_ignition.append(t_ign)
            # Update global start time for the next cell.
            G.append(G_i + (float(t_ign) if t_ign is not None else 0.0))
            if t_ign is None:
                # Fire stopped — downstream cells won't ignite either
                break
            if _ros_converged(t_ignition, spread_cfg.ros_min_cells,
                               spread_cfg.ros_converge_frac):
                # Instantaneous ROS has reached quasi-steady state
                break
    
            # Fire-front depth cap: stop after one full fire-front crossing.
            # _n_front = round(fuel_depth_m / dx_m) — Anderson (1982) fuel bed depth.
            # spread_cfg.max_cells is a hard safety cap above this.
            if len(cell_t) >= _n_front_eff:
                break
            i += 1
    
        # ── Rate of spread ────────────────────────────────────────────────────────
        # G[k+1] = cumulative sum of local ignition times = global wall-clock time
        # at which cell k was ignited.  G[1] = 0 (source ignited at t=0).
        # t_ignition stores PER-CELL LOCAL times; using t_ign_valid[-1] - t_ign_valid[0]
        # as the denominator gives only the last cell's local time (e.g. 1 s), not the
        # total elapsed fire-front time.  Use G instead.
        t_ign_valid = [t for t in t_ignition if t is not None]
        n_ign = len(t_ign_valid)
    
        if n_ign >= 2:
            # Global ignition time of cell k = G[k+1]  (G has length n_cells_run + 1)
            # G[1] = 0 (cell 0), G[n_ign] = global ign time of the last ignited cell.
            global_elapsed = G[n_ign] - G[1]  # G[1] = 0 always
            if global_elapsed > 0.0:
                # Physical advance: when jump stencil is active, each ignition step
                # covers n_jump × dx physical metres (not just dx).
                # _n_jump_list has one entry per downstream cell (i=1,2,...).
                # We use the first (n_ign-1) entries = the steps that produced ignitions.
                _n_jumps_used = _n_jump_list[: n_ign - 1]  # steps 1..n_ign-1
                physical_advance_m = (
                    float(sum(_n_jumps_used)) * spread_cfg.dx_m
                    if _n_jumps_used
                    else spread_cfg.dx_m * (n_ign - 1)
                )
                ros_m_s = physical_advance_m / global_elapsed
            else:
                ros_m_s = 0.0
                physical_advance_m = 0.0
        else:
            ros_m_s = 0.0
            physical_advance_m = 0.0
    
        # ── τ_r convergence check (Phase 8) ───────────────────────────────────
        # Compute new d_H = τ_r × ROS from this pass's result.
        # If n_front(d_H) is unchanged (integer convergence), stop iterating.
        if spread_cfg.use_tau_r_fire_front_depth:
            _d_H_new = (
                max(spread_cfg.dx_m, _tau_r_s * ros_m_s)
                if ros_m_s > 0.0 else spread_cfg.dx_m
            )
            _n_new = max(1, min(
                spread_cfg.max_cells,
                round(_d_H_new / spread_cfg.dx_m),
            ))
            if _n_new == _n_front_eff:
                break  # converged — n_front unchanged
            _d_H_eff = _d_H_new

    return SpreadResult(
        t_ignition=t_ignition,
        cell_t=cell_t,
        cell_hrrpua=cell_hrrpua,
        ros_m_s=ros_m_s,
        n_cells_ignited=n_ign,
        spread_cfg=spread_cfg,
        n_jump_list=_n_jump_list,
    )


# ── Coupled multi-cell spread model using full ROM (Phase 9) ─────────────────
#
# Same sequential cascade architecture as run_1d_spread, but each cell receives
# **preheat** from predecessor radiation during its waiting period [0, G_i].
# The preheat energy is injected as an elevated q_in schedule that starts at
# t=0 (global) and ramps up as predecessors ignite.  This captures the physics
# missing from the sequential cascade: cells ahead of the fire warm gradually.
#
# When no jump stencil is active, this reduces to the standard cascade with
# preheat — smooth in U because t_ign depends continuously on the accumulated
# preheat energy.  No n_front, no n_jump, no τ_r iteration.
#
# Each cell runs the FULL pyrolysis ROM (2-node thermal, kinetics, moisture,
# flame feedback) — same fidelity as run_1d_spread.
#
# References:
#   Sullivan (2009) Int. J. Wildland Fire 18:349 — physics-based spread review
#   Weber (1991) Prog. Energy Combust. Sci. 17:67 — reaction-diffusion models
#   Beer (1991) Combust. Sci. Tech. 77:55 — convective mechanism in grass fires


def run_1d_spread_coupled(
    ri_or_path: Union[Path, str, RomInputs],
    spread_cfg: SpreadConfig,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
) -> SpreadResult:
    """Coupled multi-cell fire spread using full ROM per cell (Phase 9).

    Runs the same cascade as ``run_1d_spread`` but with preheat: each cell's
    q_in schedule starts from t=0 (global time), not from G_i.  Predecessors
    that are burning at any earlier time contribute radiation to downstream
    cells, warming them before the fire front reaches them.

    The cell still runs the full pyrolysis ROM ODE — not a simplified 1-node
    approximation.  The preheat is delivered via an extended q_in_schedule
    that covers [0, G_i + t_remain] instead of [G_i, G_i + t_remain].

    No n_front cap, no jump stencil, no τ_r iteration.  ROS emerges from
    the ignition-front advance through the cell array.

    Returns SpreadResult (same interface as run_1d_spread).
    """
    # ── Parse deck ────────────────────────────────────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri_base = load_text_input(Path(ri_or_path))
    else:
        ri_base = copy.deepcopy(ri_or_path)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    _ri_base_Tamb_saved = ri_base.Tamb

    # ── Grid ─────────────────────────────────────────────────────────────────
    dx = spread_cfg.dx_m
    N = max(2, min(spread_cfg.max_cells, round(outdoor_cfg.fuel_depth_m / dx)))

    # ── Source element (cell 0) ──────────────────────────────────────────────
    ri_base.Tamb = spread_cfg.T_gas_spread_K
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_base_Tamb_saved
    signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    src_t = np.asarray(signals_src.t, dtype=float)
    src_hrrpua = np.asarray(signals_src.hrrpua, dtype=float)

    # ── Sustained source plateau ─────────────────────────────────────────
    # In a real fire, the fire line is quasi-steady: fresh fuel continuously
    # enters the reaction zone as burned fuel exits behind.  The single-blade
    # ROM produces a transient profile (ramp → peak → burnout) because the
    # 0.8mm blade has only 0.182 kg/m² of fuel.  For the cascade, the source
    # should represent the aggregate fire-line HRRPUA, not one blade's transient.
    # Hold the HRRPUA at 80% of peak after the peak — physical: the fire line
    # is slightly below peak due to mixing with burned fuel at the trailing edge.
    _peak_idx = int(np.argmax(src_hrrpua))
    _plateau = float(src_hrrpua[_peak_idx]) * 0.8
    src_hrrpua = src_hrrpua.copy()
    src_hrrpua[_peak_idx:] = np.maximum(src_hrrpua[_peak_idx:], _plateau)

    cell_t: List[np.ndarray] = [src_t]
    cell_hrrpua: List[np.ndarray] = [src_hrrpua]
    t_ignition: List[Optional[float]] = [0.0]
    G: List[float] = [0.0, 0.0]  # G[0]=0, G[1]=0

    # ── Flame / convective parameters ────────────────────────────────────────
    _outdoor_cfg_eff = copy.copy(outdoor_cfg)
    chi_rad = spread_cfg.chi_rad_spread
    kappa = spread_cfg.kappa_flame_m

    # ── Sequential cascade with full ROM per cell ────────────────────────────
    i = 1
    while i < N:
        G_i = G[i]
        t_remain = max(max_wall_time_s - G_i, 1.0)

        ri_i = copy.deepcopy(ri_base)
        ri_i.outdoor_overrides["free_burning_mode"] = False

        # ── Split-flux thermal mass correction ────────────────────────────
        # The flame is dx wide.  When it tilts, hot gas enters the next cell
        # as a thin stream — only dx wide, not h_bed tall.  The convective/
        # immersion flux contacts a thin layer of the cell (∝ dx/h_bed), but
        # the model applies it to the full slab thermal mass C_slab.
        #
        # Split-flux: dT/dt = q_rad/C_slab + q_wind/C_conv
        #   where C_conv = C_slab × (dx / h_bed)
        #
        # Equivalent single-C form: q_eff = q_rad + q_wind × (h_bed / dx)
        # The ROM uses C_slab unchanged; the amplified q_wind produces the same
        # dT/dt as if C_conv were smaller.
        #
        # h_bed/dx = 1 for thin beds (GR1: 0.30/0.05=6) → modest amplification.
        # h_bed/dx >> 1 for deep beds (GR3: 0.76/0.05=15.2) → strong amplification.
        # At U=0: q_wind=0, amplification irrelevant — radiation uses full C_slab.
        # No free parameters.  No density override.  No floor.
        _split_flux_amp = outdoor_cfg.fuel_depth_m / dx if (
            spread_cfg.thermal_absorption_floor < 1.0 and dx > 0.0
        ) else 1.0

        # ── Build q_in schedule from ALL burning predecessors ────────────────
        # Same radiation + Beer-Lambert + convective + immersion physics as
        # run_1d_spread, but NO jump stencil — dx_eff = dx always.
        _dx_eff = dx  # physical cell spacing, no jump

        # Radiation from all predecessors (same as existing cascade).
        q_callables: List[Callable[[float], float]] = []
        for j in range(1, i + 1):
            q_callables.append(
                _make_spread_callable(
                    cell_t[i - j],
                    cell_hrrpua[i - j],
                    _outdoor_cfg_eff,
                    chi_rad,
                    float(j) * _dx_eff,
                    G_src=G[i - j],
                )
            )

        # Beer-Lambert blocking.
        blocking_per_pred: List[List[Callable[[float], float]]] = []
        for j in range(1, i + 1):
            blk: List[Callable[[float], float]] = []
            for k_rel in range(1, j):
                idx_blk = i - j + k_rel
                blk.append(
                    _blocking_transmittance_callable(
                        cell_t[idx_blk],
                        cell_hrrpua[idx_blk],
                        _outdoor_cfg_eff,
                        kappa,
                        G_src=G[idx_blk],
                    )
                )
            blocking_per_pred.append(blk)

        def _q_in_local(t_global, _qc=q_callables, _bp=blocking_per_pred):
            total = 0.0
            for jj, q_j in enumerate(_qc):
                q_raw = q_j(t_global)
                if q_raw <= 0.0:
                    continue
                tau = 1.0
                for blk_fn in _bp[jj]:
                    tau *= blk_fn(t_global)
                    if tau < 1e-6:
                        break
                total += q_raw * tau
            return total

        # Build dense time grid for the schedule.
        tau_pieces = []
        for j in range(1, i + 1):
            t_pred = cell_t[i - j]
            t_offset_j = G_i - G[i - j]
            tau_j = t_pred - t_offset_j
            tau_j = tau_j[(tau_j >= 0.0) & (tau_j <= max_wall_time_s)]
            if len(tau_j) > 0:
                tau_pieces.append(tau_j)
        t_grid = np.unique(np.concatenate(tau_pieces))
        q_grid_rad = np.array([_q_in_local(t) for t in t_grid], dtype=float)
        q_grid_wind = np.zeros_like(q_grid_rad)

        # ── Convective preheat (nearest predecessor, with distance decay) ────
        if spread_cfg.alpha_conv_preheat > 0.0 and wind_speed_m_s > 0.0:
            _alpha_conv_eff = spread_cfg.alpha_conv_preheat
            if spread_cfg.alpha_conv_h_scale:
                _alpha_conv_eff *= outdoor_cfg.fuel_depth_m / 0.30
            _conv_scale = (
                _alpha_conv_eff
                * (wind_speed_m_s ** spread_cfg.n_conv_preheat)
                * 1000.0
            )
            if _dx_eff > dx:
                _U_mf_conv = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)
                _L_decay_conv = _gas_enthalpy_decay_length(_outdoor_cfg_eff, _U_mf_conv)
                if _L_decay_conv < math.inf:
                    _conv_scale *= math.exp(-(_dx_eff - dx) / _L_decay_conv)
            _t_near = cell_t[i - 1]
            _h_near = cell_hrrpua[i - 1]
            _G_near = G[i - 1]
            for k, tau in enumerate(t_grid):
                t_src = tau + G_i - _G_near
                if 0.0 < t_src <= _t_near[-1]:
                    q_grid_wind[k] += _conv_scale * float(
                        np.interp(t_src, _t_near, _h_near, left=0.0, right=0.0)
                    )

        # ── Flame immersion (nearest predecessor, Gap 1: dx_m for overlap) ───
        if spread_cfg.flame_immersion_enable and wind_speed_m_s > 0.0:
            _U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)
            _h_conv_imm = wind_h_conv(wind_speed_m_s, outdoor_cfg.terrain)
            _dT_imm = 1473.0 - 300.0
            _L_decay = _gas_enthalpy_decay_length(_outdoor_cfg_eff, _U_mf)
            _r_decay = _dx_eff / _L_decay if _L_decay < math.inf else 0.0

            # Immersion from ALL overlapping predecessors (distance-weighted sum).
            # Each predecessor j contributes immersion inversely weighted by
            # distance: w = 1/(1 + d_ij/dx).  This gives full weight at dx
            # (nearest cell) and decays for distant predecessors.
            # f_overlap cancellation applies: if the flame overlaps at distance
            # d_ij, apply h_conv×dT×w (no f_ov reduction on the flux itself).
            for k, tau in enumerate(t_grid):
                _imm_sum = 0.0
                for j_imm in range(i - 1, -1, -1):
                    _d_imm = float(i - j_imm) * dx
                    t_src = tau + G_i - G[j_imm]
                    if not (0.0 < t_src <= cell_t[j_imm][-1]):
                        continue
                    hrr_kW = float(np.interp(t_src, cell_t[j_imm],
                                             cell_hrrpua[j_imm],
                                             left=0.0, right=0.0))
                    if hrr_kW <= 0.0:
                        continue
                    _L_f = byram_flame_length(hrr_kW * 1000.0,
                                              _outdoor_cfg_eff.fuel_depth_m)
                    if _L_f <= 0.0:
                        continue
                    _theta = flame_tilt_angle(wind_speed_m_s, _L_f,
                                              outdoor_cfg.terrain)
                    _f_ov = max(0.0, math.sin(_theta) - _d_imm / _L_f)
                    if _f_ov > 0.0:
                        _w_dist = 1.0 / (1.0 + _d_imm / dx)
                        _imm_sum += _h_conv_imm * _dT_imm * _w_dist
                q_grid_wind[k] += _imm_sum

            # Enthalpy column (all predecessors, Gap 3: dx_eff for distances).
            if _r_decay > 0.0:
                _dx_enthalpy = _dx_eff
                for j_rel in range(1, i + 1):
                    _w_j = _r_decay * math.exp(-j_rel * _dx_enthalpy / _L_decay)
                    if _w_j < 1e-6:
                        break
                    _j_abs = i - j_rel
                    _t_j = cell_t[_j_abs]
                    _h_j = cell_hrrpua[_j_abs]
                    _G_j = G[_j_abs]
                    for k, tau in enumerate(t_grid):
                        t_src = tau + G_i - _G_j
                        if 0.0 < t_src <= _t_j[-1]:
                            hrr_kW = float(np.interp(t_src, _t_j, _h_j, left=0.0, right=0.0))
                            if hrr_kW > 0.0:
                                q_grid_wind[k] += _h_conv_imm * _dT_imm * _w_j

        # ── Combine radiation + wind flux ────────────────────────────────────
        q_grid = q_grid_rad + q_grid_wind

        # ── Set q_in schedule and run element ────────────────────────────────
        ri_i.q_in_schedule = list(zip(t_grid.tolist(), q_grid.tolist()))
        ri_i.q_in_units = "W/m2"
        ri_i.q_in_constant = None
        ri_i.outdoor_overrides["wind_speed_m_s"] = 0.0

        signals, _ = run_outdoor_element(ri_i, t_end_s=t_remain)

        t_arr = np.asarray(signals.t, dtype=float)
        hrr_arr = np.asarray(signals.hrrpua, dtype=float)
        # Apply sustained plateau to cascade cells (same as source):
        # each position in the fire line sustains burning as fuel feeds in.
        if spread_cfg.thermal_absorption_floor < 1.0:
            _pk_i = int(np.argmax(hrr_arr))
            _plat_i = float(hrr_arr[_pk_i]) * 0.8
            hrr_arr = hrr_arr.copy()
            hrr_arr[_pk_i:] = np.maximum(hrr_arr[_pk_i:], _plat_i)
        cell_t.append(t_arr)
        cell_hrrpua.append(hrr_arr)

        t_ign = _find_ignition_time(t_arr, hrr_arr, spread_cfg.hrrpua_ign_kW_m2)
        t_ignition.append(t_ign)
        G.append(G_i + (float(t_ign) if t_ign is not None else 0.0))
        if t_ign is None:
            break
        if _ros_converged(t_ignition, spread_cfg.ros_min_cells,
                           spread_cfg.ros_converge_frac):
            break
        if len(t_ignition) >= N:
            break
        i += 1

    # ── ROS ──────────────────────────────────────────────────────────────────
    t_ign_valid = [t for t in t_ignition if t is not None]
    n_ign = len(t_ign_valid)
    if n_ign >= 2:
        global_elapsed = G[n_ign] - G[1]
        if global_elapsed > 0.0:
            physical_advance_m = float(n_ign - 1) * dx
            ros_m_s = physical_advance_m / global_elapsed
        else:
            ros_m_s = 0.0
    else:
        ros_m_s = 0.0

    return SpreadResult(
        t_ignition=t_ignition,
        cell_t=cell_t,
        cell_hrrpua=cell_hrrpua,
        ros_m_s=ros_m_s,
        n_cells_ignited=n_ign,
        spread_cfg=spread_cfg,
        n_jump_list=[1] * max(0, n_ign - 1),
    )


# ── 2-equation PDE fire spread model (Phase 10) ─────────────────────────────
#
# Two-temperature reaction-advection-diffusion:
#   Solid:  C_s × ∂T_s/∂t = h_p×a_v×(T_g - T_s) + q_rad(x) - q_loss(T_s)
#   Gas:    ρ_g×c_g×(∂T_g/∂t + U_mf×∂T_g/∂x) = -h_p×a_v×(T_g - T_s) + q_comb
#
# Wind sensitivity from gas ADVECTION (U_mf×∂T_g/∂x): hot gas from fire
# carried forward, heating solid fuel via particle convection (h_p×a_v).
# Grid-convergent: diffusion stencil + upwind advection.
# No free parameters: h_p from Ranz-Marshall, a_v from SAV×β, all from deck.
#
# References:
#   Weber (1991) Prog. Energy Combust. Sci. 17:67
#   Morvan & Dupuy (2001) Combust. Flame 127:1981
#   Beer (1991) Combust. Sci. Tech. 77:55


def run_1d_spread_pde(
    ri_or_path: Union[Path, str, RomInputs],
    spread_cfg: SpreadConfig,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
    pde_dx: float = 0.005,
    pde_domain_m: float = 5.0,
    variable_density: bool = False,
) -> SpreadResult:
    """2-equation PDE fire spread (T_solid, T_gas) — Phase 10.

    Grid-convergent formulation with gas advection providing wind sensitivity.

    Parameters
    ----------
    variable_density : bool
        If True, gas thermal mass uses ideal-gas ρ(T) = P₀/(R·T) instead
        of constant ρ₀.  Cold gas (300 K) becomes 2.7× heavier; hot gas
        (1473 K) becomes 1.8× lighter.  No free parameters.
    """
    # ── Parse deck ────────────────────────────────────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri_base = load_text_input(Path(ri_or_path))
    else:
        ri_base = copy.deepcopy(ri_or_path)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    # ── Source element ───────────────────────────────────────────────────────
    _ri_base_Tamb = ri_base.Tamb
    ri_base.Tamb = spread_cfg.T_gas_spread_K
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_base_Tamb
    signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    src_t = np.asarray(signals_src.t, dtype=float)
    src_hrrpua = np.asarray(signals_src.hrrpua, dtype=float)
    _peak_idx = int(np.argmax(src_hrrpua))
    _plateau = float(src_hrrpua[_peak_idx]) * 0.8
    src_hrrpua = src_hrrpua.copy()
    src_hrrpua[_peak_idx:] = np.maximum(src_hrrpua[_peak_idx:], _plateau)

    # ── Physical parameters ──────────────────────────────────────────────────
    T_amb = outdoor_cfg.ambient_T_K if hasattr(outdoor_cfg, 'ambient_T_K') else 300.0
    T_ign = T_amb + 300.0  # solid ignition temperature [K]
    T_flame_adiabatic = 1473.0  # adiabatic flame temperature [K]

    # Solid fuel.
    rho_bulk_s = outdoor_cfg.bulk_density_kg_m3  # [kg/m³] bed bulk density
    cp_s = 1300.0
    eps = 0.90
    sigma_sb = 5.67e-8
    # C_s set after grid setup (depends on dx).

    # Gas.
    h_bed = outdoor_cfg.fuel_depth_m
    U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)

    # Effective gas temperature — needs U_mf and h_bed.
    # Only the CONVECTIVE fraction (1 - chi_rad) heats the gas; the
    # radiative fraction leaves via photons without raising gas temperature.
    _HRRPUA_W = float(np.max(src_hrrpua)) * 1000.0 * 0.8
    _chi_rad_deck = spread_cfg.chi_rad_spread
    _U_char = max(U_mf, math.sqrt(9.81 * h_bed * 0.5))
    _dT_gas = (1.0 - _chi_rad_deck) * _HRRPUA_W / max(_RHO_GAS * _CP_GAS * _U_char, 1.0)
    T_flame = min(T_amb + _dT_gas, T_flame_adiabatic)

    # Particle-gas heat transfer: h_p × a_v [W/m³/K].
    sigma_sav = outdoor_cfg.sav_ratio_1_m
    rho_bulk = outdoor_cfg.bulk_density_kg_m3
    d_p = 4.0 / sigma_sav
    _Re = max(_RHO_GAS * max(U_mf, 0.1) * d_p / _MU_GAS, 0.1)
    _Nu = 2.0 + 0.6 * _Re**0.5 * _PR_GAS**(1.0 / 3.0)
    h_p = _Nu * _K_GAS / d_p       # [W/m²/K]
    beta = rho_bulk / _RHO_PARTICLE
    a_v = sigma_sav * beta          # [m⁻¹]
    hp_av = h_p * a_v               # [W/m³/K] volumetric coupling

    # Gas thermal capacity per unit bed depth [J/m³/K].
    # Porosity ≈ 1 for sparse grass beds.
    rho_g_cp_g = _RHO_GAS * _CP_GAS  # [J/m³/K]

    # Source HRRPUA for combustion term.
    _peak_W = float(np.max(src_hrrpua)) * 1000.0
    hoc_eff = 14900.0 * 1000.0  # [J/kg]

    # ── Grid setup ───────────────────────────────────────────────────────────
    N = max(10, int(math.ceil(pde_domain_m / pde_dx)))
    x_mid = np.linspace(0.5 * pde_dx, pde_domain_m - 0.5 * pde_dx, N)
    dx = float(x_mid[1] - x_mid[0]) if N > 1 else pde_dx

    # Solid thermal capacity per cell: fuel mass in a dx-wide slice.
    # C_s = ρ_bulk × cp × dx  [J/m²/K per cell].
    # Both coupling (hp_eff × dx) and C_s scale with dx → dT/dt is dx-independent.
    C_s = rho_bulk_s * cp_s * dx

    # States.
    T_s = np.full(N, T_amb)   # solid temperature [K]
    T_g = np.full(N, T_amb)   # gas temperature [K]
    burning = np.zeros(N, dtype=bool)

    # Source: first cell burning from t=0.
    T_s[0] = T_flame
    T_g[0] = T_flame
    burning[0] = True
    x_front = float(x_mid[0])

    front_history_t: List[float] = [0.0]
    front_history_x: List[float] = [x_front]

    # ── Flame radiation source (Albini view factor + Beer-Lambert) ─────────
    # Precompute flame geometry: L_f, theta, and the sustained HRRPUA.
    # The radiation from burning cells provides the primary wind-dependent
    # spread mechanism (flame tilt increases view factor at high wind).
    # Gas-solid coupling alone cannot drive high-wind spread because gas
    # transit time through burning cells is too short at high U.
    chi_rad = spread_cfg.chi_rad_spread
    _sustained_HRRPUA_W = _HRRPUA_W  # W/m² (sustained 80% of peak)
    L_f = byram_flame_length(_sustained_HRRPUA_W, h_bed)
    theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f, outdoor_cfg.terrain)
    kappa = spread_cfg.kappa_flame_m

    # Radiation uses the continuous slab view factor, not per-cell summation.
    # For a contiguous burning zone, q_rad at distance d from the front edge:
    #   q_rad = chi_rad × HRRPUA × [F(d_near) - F(d_far)]
    # This is dx-independent: it depends on physical distance, not cell count.

    # ── Turbulent gas diffusion ─────────────────────────────────────────────
    # Mixing-length theory: α_turb = C_μ × L_mix × U_char.
    # L_mix = min(1/a_v, h_bed): the inter-particle gap (gas mean free path
    # through the porous bed) or the bed height, whichever limits the eddy.
    # C_μ = 0.09 (Launder & Spalding 1974, standard k-ε constant).
    # At U=0: buoyancy-driven velocity replaces U_mf.
    # No free parameters: a_v from SAV×β (deck), C_μ from literature.
    _C_mu = 0.09
    _L_mix = min(1.0 / max(a_v, 1e-3), h_bed)
    if U_mf > 0.0:
        _alpha_turb = _C_mu * _L_mix * U_mf  # [m²/s]
    else:
        _g = 9.81
        _buoy_vel = math.sqrt(_g * h_bed * (T_flame - T_amb) / T_amb)
        _alpha_turb = _C_mu * _L_mix * _buoy_vel
    D_g_turb = rho_g_cp_g * _alpha_turb  # [W/m/K]

    # ── Time stepping ────────────────────────────────────────────────────────
    t = 0.0
    dt_adv = dx / max(U_mf, 0.01)
    dt_diff_g = 0.4 * dx**2 / max(_alpha_turb, 1e-8)
    _hp_eff_max = (h_p + 4.0 * eps * sigma_sb * T_flame**3) * a_v
    dt_couple = 0.5 * min(C_s, rho_g_cp_g * dx) / max(_hp_eff_max * dx, 1.0)
    dt_max = min(0.5 * dt_adv, dt_diff_g, dt_couple, 0.1)
    dt = max(dt_max, 1e-6)

    # Precompute index mask for unburned cells (updated when ignition occurs).
    _unburned = ~burning  # bool array, cells 1..N-1 that are not burning

    # Flame tip projection distance [m] — cells within this distance of the
    # burning zone front edge are immersed in the flame plume.
    _flame_proj = L_f * math.sin(theta_tilt) if theta_tilt > 0 else 0.0

    while t < max_wall_time_s:
        # Clamp burning cells.
        T_s[burning] = T_flame
        T_g[burning] = T_flame

        # Flame immersion: clamp gas in unburned cells within the flame
        # projection to T_flame_adiabatic (undiluted flame temperature).
        # These cells are physically inside the tilted flame plume.
        if _flame_proj > 0.0 and len(np.where(burning)[0]) > 0:
            _last_b_imm = int(np.where(burning)[0][-1])
            _front_imm = x_mid[_last_b_imm] + 0.5 * dx
            _imm_mask = _unburned & (x_mid < _front_imm + _flame_proj)
            T_g[_imm_mask] = T_flame_adiabatic

        # ── Vectorized solid equation (cells 1..N-1, unburned only) ──
        # h_rad = 4εσT³ at mean local T.
        _T_local = 0.5 * (T_g + T_s)
        _h_rad_arr = 4.0 * eps * sigma_sb * _T_local**3
        _hp_eff_arr = (h_p + _h_rad_arr) * a_v  # [W/m³/K]

        # Gas-to-solid coupling: hp_eff × dx × (T_g - T_s).
        q_gs = _hp_eff_arr * dx * (T_g - T_s)
        # Radiative loss from particle surfaces to surroundings.
        # Loss per unit particle surface × volumetric surface area × dx
        # → scales with dx (same as coupling) → dx-independent dT/dt.
        # Convective loss to ambient is handled by the gas equation (no double-counting).
        q_loss_arr = eps * sigma_sb * (T_s**4 - T_amb**4) * a_v * dx

        # ── Flame radiation: continuous slab view factor ──────────────
        # q_rad[i] = chi_rad × HRRPUA × [F(d_near) - F(d_far)]
        # d_near = distance from cell i to front edge of burning zone
        # d_far = distance from cell i to back edge of burning zone
        # dx-independent: depends on physical distances only.
        q_rad = np.zeros(N)
        _burn_idx = np.where(burning)[0]
        if len(_burn_idx) > 0:
            _last_b = int(_burn_idx[-1])
            _front_edge = x_mid[_last_b] + 0.5 * dx
            _back_edge = x_mid[_burn_idx[0]] - 0.5 * dx
            _i_start = _last_b + 1
            if _i_start < N:
                _d_near = np.maximum(x_mid[_i_start:] - _front_edge, 0.5 * dx)
                _d_far = x_mid[_i_start:] - _back_edge
                _sin_t = math.sin(theta_tilt)
                _r_near = np.maximum(_d_near - L_f * _sin_t, 1e-3)
                _r_far = np.maximum(_d_far - L_f * _sin_t, 1e-3)
                _F_near = 0.5 * (1.0 - _r_near / np.sqrt(L_f**2 + _r_near**2))
                _F_far = 0.5 * (1.0 - _r_far / np.sqrt(L_f**2 + _r_far**2))
                _F_slab = np.maximum(_F_near - _F_far, 0.0)
                # Incident flux [W/m²] × absorption in a dx-wide fuel strip.
                # Beer-Lambert: only fraction (1-exp(-a_v×dx)) is absorbed.
                # This makes q_rad_abs ∝ dx → dT/dt dx-independent.
                _f_abs = 1.0 - math.exp(-a_v * dx)
                q_rad[_i_start:] = chi_rad * _sustained_HRRPUA_W * _F_slab * _f_abs

        dTs = (q_gs + q_rad - q_loss_arr) / C_s

        # ── Vectorized gas equation ──────────────────────────────────
        # Temperature form: ∂T/∂t = -u ∂T/∂x + α ∂²T/∂x² + sources/(ρ cp)
        # Transport terms (advection, diffusion) are ρ-independent.
        # Source terms (coupling) use local ρ(T) when variable_density=True.
        dTg = np.zeros(N)
        # Upwind advection: -U × (T[i] - T[i-1]) / dx.
        dTg[1:] -= U_mf * (T_g[1:] - T_g[:-1]) / dx
        # Turbulent diffusion (central difference, zero-gradient at right).
        dTg[1:-1] += _alpha_turb * (T_g[:-2] - 2.0 * T_g[1:-1] + T_g[2:]) / dx**2
        dTg[-1] += _alpha_turb * (T_g[-2] - T_g[-1]) / dx**2
        # Gas-solid coupling: source/(ρ cp), ρ = ideal-gas or constant.
        if variable_density:
            _rho_local = 101325.0 / (287.0 * np.maximum(T_g, T_amb))
            dTg += -_hp_eff_arr * (T_g - T_s) / (_rho_local * _CP_GAS)
        else:
            dTg += -_hp_eff_arr * (T_g - T_s) / rho_g_cp_g

        # Zero out burning cells (they stay clamped).
        dTs[burning] = 0.0
        dTg[burning] = 0.0
        # Cell 0 is source — always clamped.
        dTs[0] = 0.0
        dTg[0] = 0.0

        # Update states.
        T_s += dTs * dt
        T_g += dTg * dt
        t += dt

        # Check for new ignitions.
        newly = (T_s >= T_ign) & _unburned
        if np.any(newly):
            burning |= newly
            _unburned &= ~newly
            _new_idx = np.where(newly)[0]
            _new_front = float(x_mid[_new_idx[-1]])
            if _new_front > x_front:
                x_front = _new_front
                front_history_t.append(t)
                front_history_x.append(x_front)

        if x_front > pde_domain_m * 0.9:
            break

    # ── ROS ──────────────────────────────────────────────────────────────────
    ft = np.array(front_history_t)
    fx = np.array(front_history_x)

    if len(ft) >= 3:
        n_half = len(ft) // 2
        dx_total = fx[-1] - fx[n_half]
        dt_total = ft[-1] - ft[n_half]
        ros_m_s = dx_total / dt_total if dt_total > 0.0 else 0.0
    elif len(ft) >= 2:
        ros_m_s = (fx[-1] - fx[0]) / (ft[-1] - ft[0]) if ft[-1] > ft[0] else 0.0
    else:
        ros_m_s = 0.0

    n_ign = int(np.sum(burning))

    return SpreadResult(
        t_ignition=[0.0],
        cell_t=[src_t],
        cell_hrrpua=[src_hrrpua],
        ros_m_s=ros_m_s,
        n_cells_ignited=n_ign,
        spread_cfg=spread_cfg,
        n_jump_list=[],
    )


# ── 2D (x,z) PDE spread model ───────────────────────────────────────────────
# Extends the 1D PDE with an explicit vertical dimension so that buoyancy-
# driven gas escape emerges naturally from the equations rather than being
# approximated with parameterised correction terms.
#
# References:
#   Cionco (1965) J. Appl. Meteorol. 4:517 — wind profile in canopy
#   Morton, Taylor & Turner (1956) Proc. R. Soc. A 234:1 — buoyant plumes


def run_2d_spread_pde(
    ri_or_path: Union[Path, str, RomInputs],
    spread_cfg: SpreadConfig,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
    pde_dx: float = 0.01,
    pde_domain_m: float = 10.0,
    n_z_bed: int = 4,
    n_z_buffer: int = 2,
) -> SpreadResult:
    """2D (x,z) PDE fire spread — Phase 11.

    Adds a vertical dimension to resolve gas buoyancy explicitly.
    Hot gas rises through the fuel bed and exits via the open top boundary,
    limiting horizontal heat transport naturally.

    Parameters
    ----------
    n_z_bed : int     Vertical cells within the fuel bed (z = 0..h_bed).
    n_z_buffer : int  Vertical cells above the bed (free atmosphere).
    """
    # ── Parse deck (same as 1D) ──────────────────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri_base = load_text_input(Path(ri_or_path))
    else:
        ri_base = copy.deepcopy(ri_or_path)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    # ── Source element (same as 1D) ──────────────────────────────────────
    _ri_base_Tamb = ri_base.Tamb
    ri_base.Tamb = spread_cfg.T_gas_spread_K
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_base_Tamb
    signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    src_t = np.asarray(signals_src.t, dtype=float)
    src_hrrpua = np.asarray(signals_src.hrrpua, dtype=float)
    _peak_idx = int(np.argmax(src_hrrpua))
    _plateau = float(src_hrrpua[_peak_idx]) * 0.8
    src_hrrpua = src_hrrpua.copy()
    src_hrrpua[_peak_idx:] = np.maximum(src_hrrpua[_peak_idx:], _plateau)

    # ── Physical parameters ──────────────────────────────────────────────
    T_amb = outdoor_cfg.ambient_T_K if hasattr(outdoor_cfg, 'ambient_T_K') else 300.0
    T_ign = T_amb + 300.0
    T_flame_adiabatic = 1473.0

    h_bed = outdoor_cfg.fuel_depth_m
    rho_bulk = outdoor_cfg.bulk_density_kg_m3
    cp_s = 1300.0
    eps = 0.90
    sigma_sb = 5.67e-8
    _g = 9.81

    U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)

    # Convective gas temperature.
    _HRRPUA_W = float(np.max(src_hrrpua)) * 1000.0 * 0.8
    _chi_rad_deck = spread_cfg.chi_rad_spread
    _U_char = max(U_mf, math.sqrt(_g * h_bed * 0.5))
    _dT_gas = (1.0 - _chi_rad_deck) * _HRRPUA_W / max(_RHO_GAS * _CP_GAS * _U_char, 1.0)
    T_flame = min(T_amb + _dT_gas, T_flame_adiabatic)

    # Particle-gas coupling.
    sigma_sav = outdoor_cfg.sav_ratio_1_m
    d_p = 4.0 / sigma_sav
    beta = rho_bulk / _RHO_PARTICLE
    a_v = sigma_sav * beta
    _Re = max(_RHO_GAS * max(U_mf, 0.1) * d_p / _MU_GAS, 0.1)
    _Nu = 2.0 + 0.6 * _Re**0.5 * _PR_GAS**(1.0 / 3.0)
    h_p = _Nu * _K_GAS / d_p
    rho_g_cp_g = _RHO_GAS * _CP_GAS

    # Slab radiation.
    chi_rad = spread_cfg.chi_rad_spread
    _sustained_HRRPUA_W = _HRRPUA_W
    L_f = byram_flame_length(_sustained_HRRPUA_W, h_bed)
    theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f, outdoor_cfg.terrain)

    # ── 2D grid setup ────────────────────────────────────────────────────
    Nx = max(10, int(math.ceil(pde_domain_m / pde_dx)))
    dx = pde_domain_m / Nx
    Nz = n_z_bed + n_z_buffer
    dz = h_bed / max(n_z_bed, 1)

    x_mid = np.linspace(0.5 * dx, pde_domain_m - 0.5 * dx, Nx)
    z_mid = np.linspace(0.5 * dz, (Nz - 0.5) * dz, Nz)

    # Thermal capacity per cell (fuel zone only): C_s = ρ_bulk × cp × dz.
    C_s = rho_bulk * cp_s * dz

    # Wind profile: Cionco (1965) within bed, free-stream above.
    U_z = np.zeros(Nz)
    for k in range(Nz):
        if z_mid[k] <= h_bed:
            U_z[k] = wind_profile_in_bed(z_mid[k], h_bed, U_mf)
        else:
            U_z[k] = U_mf
    # Shape (Nz, 1) for broadcasting over x.
    U_z_2d = U_z[:, np.newaxis]

    # Gas-solid coupling (volumetric, fuel zone only).
    hp_eff_base = h_p * a_v  # convective only at T_amb

    # Turbulent diffusivity: mixing-length within bed.
    _C_mu = 0.09
    _L_mix = min(1.0 / max(a_v, 1e-3), h_bed)
    _alpha_turb = _C_mu * _L_mix * max(U_mf, 0.1)
    D_g_turb = rho_g_cp_g * _alpha_turb

    # ── State arrays (Nz, Nx) ────────────────────────────────────────────
    T_s = np.full((Nz, Nx), T_amb)
    T_g = np.full((Nz, Nx), T_amb)
    burning = np.zeros((Nz, Nx), dtype=bool)

    # Source: column i=0 burning at all fuel z-levels.
    for k in range(n_z_bed):
        T_s[k, 0] = T_flame
        T_g[k, 0] = T_flame
        burning[k, 0] = True
    x_front = float(x_mid[0])
    front_history_t: List[float] = [0.0]
    front_history_x: List[float] = [x_front]

    # Column-averaged solid temperature for ignition check.
    _col_burning = np.zeros(Nx, dtype=bool)
    _col_burning[0] = True

    # ── CFL time step ────────────────────────────────────────────────────
    _w_buoy_max = math.sqrt(_g * h_bed * max(T_flame - T_amb, 0) / T_amb)
    dt_adv_x = dx / max(U_mf, 0.01)
    dt_adv_z = dz / max(_w_buoy_max, 0.01)
    dt_diff = 0.25 / max(_alpha_turb * (1.0 / dx**2 + 1.0 / dz**2), 1e-8)
    _hp_eff_max = (h_p + 4.0 * eps * sigma_sb * T_flame**3) * a_v
    dt_couple = 0.5 * min(C_s, rho_g_cp_g * min(dx, dz)) / max(_hp_eff_max * min(dx, dz), 1.0)
    dt = max(min(0.5 * dt_adv_x, 0.5 * dt_adv_z, dt_diff, dt_couple, 0.1), 1e-6)

    # Beer-Lambert absorption fraction.
    _f_abs = 1.0 - math.exp(-a_v * dz)

    # ── Time loop ────────────────────────────────────────────────────────
    t = 0.0
    while t < max_wall_time_s:
        # Clamp burning cells.
        T_s[burning] = T_flame
        T_g[burning] = T_flame

        # Above-bed flame: burning columns produce hot gas in the buffer
        # zone above the bed. This gas advects at free-stream wind speed
        # and diffuses downward into unburned fuel.
        if n_z_buffer > 0:
            for k in range(n_z_bed, Nz):
                _z_above_bed = z_mid[k] - h_bed
                if _z_above_bed < L_f:
                    T_g[k, _col_burning] = T_flame

        # Flame penetration: tilted flame enters the TOP of the fuel bed
        # ahead of the fire front. At high wind, the flame base is nearly
        # horizontal and lies on the bed surface. Unburned columns within
        # the flame projection distance get T_g = T_flame at the top
        # fuel layer only — the flame touches the bed surface, not the
        # full depth.
        if theta_tilt > 0 and L_f > 0:
            _flame_proj = L_f * math.sin(theta_tilt)
            if _flame_proj > 0 and len(np.where(_col_burning)[0]) > 0:
                _last_burn_col = int(np.where(_col_burning)[0][-1])
                _front_x = x_mid[_last_burn_col] + 0.5 * dx
                _pen_mask = (~_col_burning) & (x_mid < _front_x + _flame_proj)
                T_g[n_z_bed - 1, _pen_mask] = np.maximum(
                    T_g[n_z_bed - 1, _pen_mask], T_flame)

        # ── Solid equation (fuel zone: k < n_z_bed) ─────────────────
        _T_local = 0.5 * (T_g[:n_z_bed] + T_s[:n_z_bed])
        _h_rad = 4.0 * eps * sigma_sb * _T_local**3
        _hp_eff = (h_p + _h_rad) * a_v

        q_gs = _hp_eff * dz * (T_g[:n_z_bed] - T_s[:n_z_bed])
        q_loss = eps * sigma_sb * (T_s[:n_z_bed]**4 - T_amb**4) * a_v * dz

        # Slab radiation (applied to top fuel layer only for simplicity).
        q_rad = np.zeros((n_z_bed, Nx))
        _burn_cols = np.where(_col_burning)[0]
        if len(_burn_cols) > 0:
            _last_b = int(_burn_cols[-1])
            _front_edge = x_mid[_last_b] + 0.5 * dx
            _back_edge = x_mid[_burn_cols[0]] - 0.5 * dx
            _i_start = _last_b + 1
            if _i_start < Nx and L_f > 0:
                _d_near = np.maximum(x_mid[_i_start:] - _front_edge, 0.5 * dx)
                _d_far = x_mid[_i_start:] - _back_edge
                _sin_t = math.sin(theta_tilt)
                _r_near = np.maximum(_d_near - L_f * _sin_t, 1e-3)
                _r_far = np.maximum(_d_far - L_f * _sin_t, 1e-3)
                _F_near = 0.5 * (1.0 - _r_near / np.sqrt(L_f**2 + _r_near**2))
                _F_far = 0.5 * (1.0 - _r_far / np.sqrt(L_f**2 + _r_far**2))
                _F_slab = np.maximum(_F_near - _F_far, 0.0)
                _q_rad_1d = chi_rad * _sustained_HRRPUA_W * _F_slab * _f_abs
                # Apply to top fuel layer (most exposed to flame).
                q_rad[n_z_bed - 1, _i_start:] = _q_rad_1d

        dTs = np.zeros_like(T_s)
        dTs[:n_z_bed] = (q_gs + q_rad - q_loss) / C_s

        # ── Gas equation (all cells) ─────────────────────────────────
        # Horizontal advection (upwind).
        q_adv_x = np.zeros_like(T_g)
        q_adv_x[:, 1:] = -rho_g_cp_g * U_z_2d * (T_g[:, 1:] - T_g[:, :-1]) / dx

        # Vertical buoyancy advection (upwind from below).
        # Buoyancy velocity: use physical bed height as acceleration scale
        # (not dz, which is grid-dependent). For cells near the bed top,
        # use the remaining height to the top as the scale.
        _dT_g = np.maximum(T_g - T_amb, 0.0)
        _z_to_top = np.maximum(h_bed - z_mid, dz)[:, np.newaxis]  # (Nz,1)
        _w_buoy = np.sqrt(_g * _z_to_top * _dT_g / max(T_amb, 1.0))
        q_adv_z = np.zeros_like(T_g)
        q_adv_z[1:, :] = -rho_g_cp_g * _w_buoy[1:, :] * (T_g[1:, :] - T_g[:-1, :]) / dz

        # Horizontal diffusion.
        q_diff_x = np.zeros_like(T_g)
        q_diff_x[:, 1:-1] = D_g_turb * (T_g[:, :-2] - 2.0 * T_g[:, 1:-1] + T_g[:, 2:]) / dx**2

        # Vertical diffusion.
        q_diff_z = np.zeros_like(T_g)
        q_diff_z[1:-1, :] = D_g_turb * (T_g[:-2, :] - 2.0 * T_g[1:-1, :] + T_g[2:, :]) / dz**2

        # Gas-solid coupling (fuel zone only).
        q_sg = np.zeros_like(T_g)
        q_sg[:n_z_bed] = -_hp_eff * (T_g[:n_z_bed] - T_s[:n_z_bed])

        dTg = (q_adv_x + q_adv_z + q_diff_x + q_diff_z + q_sg) / rho_g_cp_g

        # Top boundary: T_g = T_amb (open atmosphere).
        dTg[-1, :] = 0.0
        T_g[-1, :] = T_amb

        # Zero out burning cells.
        dTs[burning] = 0.0
        dTg[burning] = 0.0
        for k in range(n_z_bed):
            dTs[k, 0] = 0.0
            dTg[k, 0] = 0.0

        # Update.
        T_s += dTs * dt
        T_g += dTg * dt
        t += dt

        # ── Column ignition check ────────────────────────────────────
        # Column ignites when vertically-averaged T_s exceeds T_ign.
        # Column ignites when ANY fuel layer reaches T_ign. For thin
        # grass blades, surface ignition propagates through the blade
        # height almost instantly.
        _T_col_max = np.max(T_s[:n_z_bed], axis=0)
        newly = (_T_col_max >= T_ign) & (~_col_burning)
        if np.any(newly):
            _new_cols = np.where(newly)[0]
            _col_burning |= newly
            for ci in _new_cols:
                for k in range(n_z_bed):
                    burning[k, ci] = True
            _new_front = float(x_mid[_new_cols[-1]])
            if _new_front > x_front:
                x_front = _new_front
                front_history_t.append(t)
                front_history_x.append(x_front)

        if x_front > pde_domain_m * 0.9:
            break

    # ── ROS (same as 1D) ─────────────────────────────────────────────────
    ft = np.array(front_history_t)
    fx = np.array(front_history_x)
    if len(ft) >= 3:
        n_half = len(ft) // 2
        ros_m_s = (fx[-1] - fx[n_half]) / max(ft[-1] - ft[n_half], 1e-8)
    elif len(ft) >= 2:
        ros_m_s = (fx[-1] - fx[0]) / max(ft[-1] - ft[0], 1e-8)
    else:
        ros_m_s = 0.0

    n_ign = int(np.sum(_col_burning))

    return SpreadResult(
        t_ignition=[0.0],
        cell_t=[src_t],
        cell_hrrpua=[src_hrrpua],
        ros_m_s=ros_m_s,
        n_cells_ignited=n_ign,
        spread_cfg=spread_cfg,
        n_jump_list=[],
    )



# ── 2D Boussinesq momentum + energy spread model ────────────────────────────


def _build_poisson_matrix_3d(Nz: int, Ny: int, Nx: int,
                             dx: float, dy: float, dz: float):
    """Sparse 3D Laplacian for pressure Poisson.

    Neumann BC on left/right/bottom and y-faces, Dirichlet P=0 at top.
    Index ordering: n = k*Ny*Nx + j*Nx + i  (k=z, j=y, i=x).
    """
    N = Nz * Ny * Nx
    rows, cols, vals = [], [], []
    idx_dx2 = 1.0 / dx**2
    idx_dy2 = 1.0 / dy**2
    idx_dz2 = 1.0 / dz**2
    NyNx = Ny * Nx

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                n = k * NyNx + j * Nx + i
                c = -2.0 * idx_dx2 - 2.0 * idx_dy2 - 2.0 * idx_dz2
                # x-neighbours
                if i < Nx - 1:
                    rows.append(n); cols.append(n + 1); vals.append(idx_dx2)
                else:
                    c += idx_dx2   # Neumann right
                if i > 0:
                    rows.append(n); cols.append(n - 1); vals.append(idx_dx2)
                else:
                    c += idx_dx2   # Neumann left
                # y-neighbours (periodic)
                j_plus = (j + 1) % Ny
                j_minus = (j - 1) % Ny
                rows.append(n); cols.append(k * NyNx + j_plus * Nx + i); vals.append(idx_dy2)
                rows.append(n); cols.append(k * NyNx + j_minus * Nx + i); vals.append(idx_dy2)
                # z-neighbours
                if k < Nz - 1:
                    rows.append(n); cols.append(n + NyNx); vals.append(idx_dz2)
                if k > 0:
                    rows.append(n); cols.append(n - NyNx); vals.append(idx_dz2)
                else:
                    c += idx_dz2   # Neumann bottom
                rows.append(n); cols.append(n); vals.append(c)

    mat = sp_sparse.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(N, N),
    )
    # Reference pressure at top corner.
    mat[N - 1, :] = 0
    mat[N - 1, N - 1] = 1.0
    mat.eliminate_zeros()
    return mat, sp_linalg.splu(mat)


def _sem_tent(r):
    """Tent shape function for Synthetic Eddy Method (Jarrin 2006).

    Variance-normalized: ∫f²dx = 1 over [-1, 1].
    """
    return np.sqrt(1.5) * np.maximum(1.0 - np.abs(r), 0.0)


def _build_poisson_matrix(Nz: int, Nx: int, dx: float, dz: float):
    """Sparse 2D Laplacian for pressure Poisson.

    Neumann BC on left/right/bottom, Dirichlet P=0 at top.
    """
    N = Nz * Nx
    rows, cols, vals = [], [], []
    idx_dx2 = 1.0 / dx**2
    idx_dz2 = 1.0 / dz**2

    for k in range(Nz):
        for i in range(Nx):
            n = k * Nx + i
            c = -2.0 * idx_dx2 - 2.0 * idx_dz2
            if i < Nx - 1:
                rows.append(n); cols.append(n + 1); vals.append(idx_dx2)
            else:
                c += idx_dx2
            if i > 0:
                rows.append(n); cols.append(n - 1); vals.append(idx_dx2)
            else:
                c += idx_dx2
            if k < Nz - 1:
                rows.append(n); cols.append(n + Nx); vals.append(idx_dz2)
            if k > 0:
                rows.append(n); cols.append(n - Nx); vals.append(idx_dz2)
            else:
                c += idx_dz2
            rows.append(n); cols.append(n); vals.append(c)

    mat = sp_sparse.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(N, N),
    )
    mat[N - 1, :] = 0
    mat[N - 1, N - 1] = 1.0
    mat.eliminate_zeros()
    return mat, sp_linalg.splu(mat)


def run_2d_momentum_spread(
    ri_or_path: Union[Path, str, RomInputs],
    spread_cfg: SpreadConfig,
    *,
    wind_speed_m_s: float = 0.0,
    max_wall_time_s: float = 300.0,
    pde_dx: float = 0.01,
    pde_domain_m: float = 10.0,
    n_z_bed: int = 4,
    n_z_buffer: int = 2,
    method: str = "explicit",
    y_loss_L: Optional[float] = None,
    y_loss_W: Optional[float] = None,
    variable_density: bool = False,
    low_mach: bool = False,
    smagorinsky: bool = False,
    turbulence_model: str = "k_epsilon",
) -> SpreadResult:
    """2D Boussinesq momentum + energy — Phase 11.

    Parameters
    ----------
    method : 'explicit', 'semi_implicit', or 'implicit'.
    y_loss_L : float or None
        Ghost-cell distance [m] from the fire-line edge.
    y_loss_W : float or None
        Fire-line y-width [m].
    variable_density : bool
        If True, gas thermal mass uses ideal-gas ρ(T) = P₀/(R·T).
    low_mach : bool
        If True, adds thermal-expansion divergence source to the Poisson
        equation: ∇·u = (1/T) DT/Dt (Rehm & Baum 1978, J. Res. NBS
        83:297).  Accounts for gas expansion at the flame that
        Boussinesq suppresses.  No free parameters.
    smagorinsky : bool
        If True, adds Smagorinsky sub-grid eddy viscosity to momentum
        and energy diffusion: ν_t = (C_s Δ)² |S|.  Models the 3D
        breakup of 2D coherent convection rolls.
        C_s = 0.17 (Lilly 1967), Pr_t = 0.5 (buoyancy flows).
        No free parameters beyond standard literature constants.
    """
    # ── Parse deck ───────────────────────────────────────────────────────
    if isinstance(ri_or_path, (Path, str)):
        ri_base = load_text_input(Path(ri_or_path))
    else:
        ri_base = copy.deepcopy(ri_or_path)
    outdoor_cfg = outdoor_env_from_dict(ri_base.outdoor_overrides)
    outdoor_cfg.wind_speed_m_s = wind_speed_m_s

    # ── Source element ───────────────────────────────────────────────────
    _ri_Tamb = ri_base.Tamb
    ri_base.Tamb = spread_cfg.T_gas_spread_K
    _ri_src = copy.deepcopy(ri_base)
    ri_base.Tamb = _ri_Tamb
    signals_src, _ = run_outdoor_element(_ri_src, t_end_s=max_wall_time_s)
    src_t = np.asarray(signals_src.t, dtype=float)
    src_hrrpua = np.asarray(signals_src.hrrpua, dtype=float)
    _pk = int(np.argmax(src_hrrpua))
    _plat = float(src_hrrpua[_pk]) * 0.8
    src_hrrpua = src_hrrpua.copy()
    src_hrrpua[_pk:] = np.maximum(src_hrrpua[_pk:], _plat)

    # ── Physical parameters ──────────────────────────────────────────────
    T_amb = outdoor_cfg.ambient_T_K if hasattr(outdoor_cfg, 'ambient_T_K') else 300.0
    T_ign = T_amb + 300.0
    _g = 9.81
    h_bed = outdoor_cfg.fuel_depth_m
    rho_bulk = outdoor_cfg.bulk_density_kg_m3
    cp_s = 1300.0; eps_s = 0.90; sigma_sb = 5.67e-8
    U_mf = midflame_wind_speed(wind_speed_m_s, outdoor_cfg.terrain)

    rho0 = _RHO_GAS; cp_g = _CP_GAS
    nu = _MU_GAS / rho0
    alpha_th = nu / _PR_GAS

    # Porous-media drag coefficient for fuel bed.
    # C_D = 1.0 for cylinders at moderate Re (Morvan & Dupuy 2001,
    # Combust. Flame 127:1981; Linn et al. 2002, IJWF 11:233).
    _C_D_drag = 1.0

    # Smagorinsky sub-grid turbulence (Lilly 1967, Smagorinsky 1963).
    # C_s = 0.17 (Lilly 1967 theoretical for isotropic turbulence).
    # Pr_t = 0.5 (standard for buoyancy-driven flows, Ince & Launder 1989).
    _C_smag = 0.17
    _Pr_t = 0.5

    _HRRPUA_W = float(np.max(src_hrrpua)) * 1000.0 * 0.8
    _chi_rad = spread_cfg.chi_rad_spread
    T_flame_adiabatic = 1473.0
    _U_char = max(U_mf, math.sqrt(_g * h_bed * 0.5))
    _dT_gas = (1.0 - _chi_rad) * _HRRPUA_W / max(rho0 * cp_g * _U_char, 1.0)
    T_flame = min(300.0 + _dT_gas, T_flame_adiabatic)

    Q_comb_static = (1.0 - _chi_rad) * _HRRPUA_W / h_bed  # [W/m³] (fallback)

    # ── Per-cell Arrhenius pyrolysis (TGA kinetics) ─────────────────
    # Uses TGA-measured intrinsic kinetics — the fundamental chemistry
    # independent of sample geometry.  No A_eff scaling needed.
    # Two-component blend: cellulose + lignin (Orfão 1999, Fuel 78:349).
    # Lignin fraction from deck seq_mr_frac0 (char residue ≈ lignin).
    _R_gas = 8.314
    # Three-component whole-grass pseudo-component kinetics.
    # E values from Berghel et al. (2023) J. Thermal Anal. Calorim.,
    # 3-step Kissinger analysis of wheat straw.  Di Blasi (2008)
    # PECS 34:47 gives E=40-120 kJ/mol as the "bulk fire kinetics"
    # range for cellulosic biomass.
    # Pure-component values (Orfão 1999: E_cell=178.7, E_hemi=139.8)
    # are too steep — correct at T_ign but give unphysical depletion
    # times at T_ign+50K due to missing component interactions,
    # mineral catalysis (Saddawi 2012), and transport limitations
    # (Antal & Varhegyi 1995).
    # A values recalculated via kinetic compensation relation to
    # preserve onset temperature at T_onset ≈ 600K.
    _E_hemi = 92000.0     # [J/mol] Berghel (2023) wheat straw hemicellulose
    _E_cell = 120000.0    # [J/mol] Berghel (2023) wheat straw cellulose
    _E_lign = 60800.0     # [J/mol] unchanged — already reasonable
    # A from compensation: A_new = A_old × exp((E_new-E_old)/(R×T_onset))
    _A_hemi = 9.71e11 * math.exp((_E_hemi - 139800.0) / (_R_gas * 600.0))
    _A_cell = 2.07e14 * math.exp((_E_cell - 178700.0) / (_R_gas * 600.0))
    _A_lign = 2.59e1      # [1/s] unchanged
    # Mass fractions — from proximate analysis or deck.
    _f_lignin = float(getattr(ri_base, 'seq_mr_frac0', 0) or 0.0)
    _f_hemi = 0.15        # cured grass (Chen 2021: 28% total, reduced for
                          # field curing/weathering; Di Blasi 2008: 20-35%)
    _f_cell = max(0.0, 1.0 - _f_hemi - _f_lignin)  # remainder = cellulose
    # Flame feedback view factor (flame radiates back to fuel surface).
    _flame_vf = getattr(ri_base, 'flame_view_factor', None)
    if _flame_vf is None:
        _flame_vf = ri_base.fuel_overrides.get('flame_view_factor', 0.40)
    _flame_vf = float(_flame_vf)

    sigma_sav = outdoor_cfg.sav_ratio_1_m
    d_p = 4.0 / sigma_sav
    beta = rho_bulk / _RHO_PARTICLE
    a_v = sigma_sav * beta
    _Re_p = max(rho0 * max(U_mf, 0.1) * d_p / _MU_GAS, 0.1)
    _Nu_p = 2.0 + 0.6 * _Re_p**0.5 * _PR_GAS**(1.0 / 3.0)
    h_p = _Nu_p * _K_GAS / d_p
    hp_av = h_p * a_v

    chi_rad = _chi_rad
    # Initial L_f from static source HRRPUA; updated dynamically in the
    # time loop from I_B = HoC × w_0 × ROS (Byram 1959).
    _w_0 = rho_bulk * h_bed   # [kg/m²] fuel load
    _hoc_J = 14900.0 * 1000.0  # [J/kg] heat of combustion
    L_f = byram_flame_length(_HRRPUA_W, h_bed)
    theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f, outdoor_cfg.terrain)

    # ── Y-direction ghost-cell BC ──────────────────────────────────────
    # The 2D (x,z) model is a cross-section of a 3D fire.  In 3D,
    # buoyancy-driven continuity ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 shares
    # vertical outflow between x AND y; the 2D model forces it all
    # into x, over-driving horizontal gas transport.
    #
    # Fix: ghost cells at y = ±L_y with T_g = T_amb (far-field).
    # Centre cell has y-width W (fire depth in y-direction).
    # FV diffusion:  dT/dt = 2 α_y (T_amb − T_g) / (d_half × W)
    #   where d_half = (W + L_y) / 2  (centre-to-ghost distance).
    # α_y = C_μ × L_mix × w_buoy — buoyancy-driven eddy diffusivity.
    # No free parameters: α_y from fuel deck + turbulence constants.
    _k_y_loss = 0.0   # [1/s] — disabled if y_loss_L is None
    if y_loss_L is not None and y_loss_L > 0:
        _C_mu = 0.09
        _L_mix_y = min(1.0 / max(a_v, 1e-3), h_bed)
        _w_buoy_y = math.sqrt(_g * h_bed * max(T_flame - T_amb, 0) / max(T_amb, 1.0))
        _alpha_y = _C_mu * _L_mix_y * _w_buoy_y
        # Centre-cell y-width (fire line length) and ghost distance.
        _W_y = y_loss_W if y_loss_W is not None else y_loss_L
        _d_half = (_W_y + y_loss_L) / 2.0   # centre-to-ghost
        _k_y_loss = 2.0 * _alpha_y / (_d_half * _W_y)

    # ── Grid ─────────────────────────────────────────────────────────────
    Nx = max(10, int(math.ceil(pde_domain_m / pde_dx)))
    dx = pde_domain_m / Nx
    Nz = n_z_bed + n_z_buffer
    dz = h_bed / max(n_z_bed, 1)
    x_mid = np.linspace(0.5 * dx, pde_domain_m - 0.5 * dx, Nx)
    z_mid = np.linspace(0.5 * dz, (Nz - 0.5) * dz, Nz)
    # Solid thermal capacity (dry — moisture handled as explicit phase
    # change below, not smeared into cp).  Grishin (1984), Margerit &
    # Séro-Guillaume (2002): moisture evaporates at T = 373K with
    # latent heat L_v = 2257 kJ/kg, and pyrolysis is deferred until
    # the local water mass is gone.  The previous cp_eff approach
    # smeared latent heat over T_amb → T_ign which double-suppressed
    # propagation at high moisture.
    _M_f = outdoor_cfg.initial_moisture_frac if hasattr(outdoor_cfg, 'initial_moisture_frac') else 0.0
    _L_v = 2257000.0   # [J/kg] latent heat of water
    _T_evap = 373.15   # [K] water boiling point
    C_s = rho_bulk * cp_s * dz
    _f_abs = 1.0 - math.exp(-a_v * dz)

    # Inlet wind profile (Cionco 1965 within bed, free-stream above).
    # The drag term acts on the PERTURBATION velocity (u - U_mean) only,
    # so the Cionco profile represents the steady-state mean wind that
    # is maintained by the large-scale pressure gradient (not resolved).
    u_inlet = np.zeros(Nz)
    for k in range(Nz):
        u_inlet[k] = wind_profile_in_bed(z_mid[k], h_bed, U_mf) if z_mid[k] <= h_bed else U_mf

    # ── State ────────────────────────────────────────────────────────────
    vel_u = np.zeros((Nz, Nx))
    vel_w = np.zeros((Nz, Nx))
    T_g = np.full((Nz, Nx), T_amb)
    T_s = np.full((Nz, Nx), T_amb)

    vel_u[:, :] = u_inlet[:, np.newaxis]

    # Fuel mass per POOL per layer [kg/m²].  Sequential depletion:
    # hemicellulose depletes first (lower E), then cellulose, then lignin.
    _m_total = rho_bulk * dz   # total per layer
    _m_hemi = np.full((n_z_bed, Nx), _f_hemi * _m_total)
    _m_cell = np.full((n_z_bed, Nx), _f_cell * _m_total)
    _m_lign = np.full((n_z_bed, Nx), _f_lignin * _m_total)
    _m_fuel_min = 1e-6 * _m_total

    # ── Gas-phase combustible volatile mass ──────────────────────────
    # Two-mass model: solid fuel pyrolyses → combustible volatiles →
    # gas-phase combustion.  Volatiles are advected by the momentum
    # field (u, w) and consumed by mixing-limited combustion.
    # Morvan & Dupuy (2001) Combust. Flame 127:1981.
    _m_vol = np.zeros((n_z_bed, Nx))
    # Explicit moisture state per cell — water mass (kg/m²/layer).
    # Grishin (1984); Margerit & Séro-Guillaume (2002).  Consumed by
    # incoming heat at T = T_evap; pyrolysis suppressed until local
    # water = 0.  Initial water mass = ρ × M_f × dz per layer.
    _m_water = np.full((n_z_bed, Nx), rho_bulk * _M_f * dz)
    # Heat-of-gasification cap state (external q_in from previous step).
    # Used to limit m_dot_py to physically achievable rates given the
    # current external heating.  Quintiere "Fundamentals of Fire
    # Phenomena" (2006) Ch.7: m_dot ≤ q_net / L_HoG where L_HoG is the
    # heat to fully gasify unit fuel mass.  For grass: 2.5 MJ/kg
    # (cellulose pyrolysis enthalpy ~1.6 MJ/kg + sensible heating
    # 300K → 700K ~0.5 MJ/kg + moisture evaporation).  Without this
    # cap, Arrhenius kinetics at flame T (>900K) deplete fuel in <10ms
    # which is 10× faster than required for self-sustained spread
    # (Rothermel 1972 fuel residence τ_r ≈ 0.1s for grass at 1 m/s).
    _q_in_ext_prev = np.zeros((n_z_bed, Nx))   # external heat input [W/m²]
    # L_HoG range from literature for grass cellulose:
    #   Antal & Várhegyi (1995): pyrolysis enthalpy alone 1.6 MJ/kg
    #   Quintiere (2006) Ch.7: + sensible heating 300→700K = 0.5 MJ/kg
    # Total: 2.1 MJ/kg.  Moisture is now handled explicitly via the
    # _m_water state (not folded into L_HoG), so use the dry-fuel
    # gasification value at the upper end of the literature range.
    _L_HoG = 2.1e6   # [J/kg]
    # Combustible fraction of pyrolysis products by pool.
    # Yang et al. (2007) Fuel 86:1781; Shen et al. (2010) JAAP 87:199.
    _eta_hemi = 0.65  # hemicellulose: ~35% non-combustible (CO₂, H₂O);
                      # remainder (acetic acid, furfural, organics) burns.
                      # Yang (2007): ~20% CO₂ + ~15% H₂O = 35% inert.
    _eta_cell = 0.9   # cellulose: ~95% combustible (levoglucosan, tar)
    _eta_lign = 0.5   # lignin: ~45% char, rest combust/non-condensable
    # ── Turbulence model ─────────────────────────────────────────────
    # k-ε URANS with RNG correction (Yakhot & Orszag 1986;
    # Morvan & Dupuy 2001).  Standard constants with RNG η-correction
    # on C_2ε for high-strain fire plumes.
    _C_mu = 0.09
    _C_1eps = 1.44
    _C_2eps = 1.92
    _sigma_k = 1.0
    _sigma_eps = 1.3
    _use_ke = (turbulence_model == "k_epsilon")
    if smagorinsky and _use_ke:
        import warnings as _w
        _w.warn("smagorinsky=True ignored when turbulence_model='k_epsilon'")
    # k-ε state arrays initialized after SEM block (needs _I_t, _u_turb_ref).
    _k_turb = None; _eps_turb = None
    _k_inlet = 0.0; _eps_inlet = 0.0
    # Fallback τ_mix for non-k-ε modes (FDS buoyancy regime).
    _Delta_mix = math.sqrt(dx * dz)
    _tau_mix_fallback = math.sqrt(2.0 * _Delta_mix / _g)

    # Source: initial burning zone.
    _n_source = max(1, int(0.5 / dx))
    _col_burning = np.zeros(Nx, dtype=bool)
    _col_burning[:_n_source] = True
    for k in range(n_z_bed):
        T_s[k, :_n_source] = T_ign + 100.0   # ignited by drip torch

    x_front = float(x_mid[min(_n_source - 1, Nx - 1)])
    front_history_t: List[float] = [0.0]
    front_history_x: List[float] = [x_front]

    # ── Turbulent inflow BC ────────────────────────────────────────────
    # Real atmospheric wind has turbulence intensity I_t ≈ 0.15–0.25
    # for open terrain (Simiu & Scanlan 1996, Table 2.2; ASCE 7-22
    # Table 26.11-1 Exposure C).  At U=0, buoyancy-driven convective
    # gusts provide the fluctuation (Morton, Taylor & Turner 1956).
    # The inlet perturbation seeds instabilities that break 2D coherent
    # convection rolls — same physics as the 3D v-perturbation but
    # applied as a boundary condition.
    # ── Synthetic Eddy Method (Jarrin et al. 2006) ──────────────────
    # Broadband, spatially-correlated turbulent inflow.  Deterministic
    # via seeded RNG.  Replaces single-frequency sinusoidal.
    # Jarrin N. et al. (2006) Int. J. Heat Fluid Flow 27:585-593.
    # I_t = 0.20 (ASCE 7-22 Table 26.11-1 Exposure C, open terrain).
    # σ = h_bed (integral length scale).
    _I_t = 0.20
    _alpha_e = 0.1   # MTT (1956)
    _w_buoy_turb = math.sqrt(_g * h_bed * max(T_flame - T_amb, 0) / max(T_amb, 1.0))
    _u_turb_ref = max(U_mf, _alpha_e * _w_buoy_turb)
    _sem_N = 100
    _sem_sigma = h_bed
    _sem_zmin = 0.0
    _sem_zmax = Nz * dz
    _sem_rng = np.random.Generator(np.random.PCG64(seed=42))
    _sem_xk = _sem_rng.uniform(-_sem_sigma, _sem_sigma, size=_sem_N)
    _sem_zk = _sem_rng.uniform(_sem_zmin - _sem_sigma,
                                _sem_zmax + _sem_sigma, size=_sem_N)
    _sem_eps = _sem_rng.choice([-1.0, 1.0], size=(_sem_N, 2))
    _sem_Uc = max(U_mf, 0.1)
    _sem_amp = _I_t * _u_turb_ref
    _sem_inv_sqN = 1.0 / math.sqrt(_sem_N)
    _t_ign_col = np.full(Nx, np.inf)
    _t_ign_col[:_n_source] = 0.0

    # ── Deferred k-ε initialization ─────────────────────────────────
    # Richards & Hoxey (1993) equilibrium ABL profiles.
    # u* = κ U_10 / ln((z_ref+z0)/z0);  k = u*²/√C_μ;
    # ε = u*³ / (κ(z+z0)).  z0=0.03m for open grassland.
    if _use_ke:
        _z0_grass = 0.03   # [m] roughness length (grassland)
        _kappa = 0.41      # von Karman constant
        _u_star = _kappa * wind_speed_m_s / max(math.log((10.0 + _z0_grass) / _z0_grass), 0.1)
        _u_star = max(_u_star, 0.01)
        _k_inlet = _u_star ** 2 / math.sqrt(_C_mu)    # 3.33 u*²
        _eps_inlet = _u_star ** 3 / (_kappa * max(h_bed + _z0_grass, 0.01))
        _k_turb = np.full((Nz, Nx), max(_k_inlet, 1e-8))
        _eps_turb = np.full((Nz, Nx), max(_eps_inlet, 1e-8))
        _nu_t = _C_mu * _k_turb ** 2 / _eps_turb

    # ── Poisson ──────────────────────────────────────────────────────────
    _, _poi_lu = _build_poisson_matrix(Nz, Nx, dx, dz)

    # ── Timestep ─────────────────────────────────────────────────────────
    _w_est = math.sqrt(_g * h_bed * max(Q_comb_static / max(hp_av, 1.0), 1.0) / T_amb)
    # Drag CFL: dt < 2 / (C_D × a_v × |u_max|).
    _u_est = max(U_mf, _w_est, 0.1)
    _dt_drag = 2.0 / (_C_D_drag * a_v * _u_est) if a_v > 0 else 1.0
    if method == "explicit":
        dt = max(min(0.3 * dx / max(U_mf, 0.01),
                     0.3 * dz / max(_w_est, 0.01),
                     0.2 * min(dx, dz)**2 / max(nu, 1e-8),
                     0.5 * _dt_drag,
                     0.1), 1e-6)
    elif method == "semi_implicit":
        dt = max(min(0.5 * dx / max(U_mf, 0.01),
                     0.5 * dz / max(_w_est, 0.01), 0.5), 1e-4)
    else:
        dt = max(min(dx / max(U_mf, 0.01),
                     dz / max(_w_est, 0.01), 1.0), 1e-4)

    # ── Time loop ────────────────────────────────────────────────────────
    _step = 0
    _print_every = max(1, int(10.0 / max(dt, 1e-6)))
    _stall_check_every = max(1, int(30.0 / max(dt, 1e-6)))  # every 30s
    _last_front = x_front
    t = 0.0
    while t < max_wall_time_s:
        _step += 1
        if _step % _print_every == 0:
            _n_burn = int(np.sum(_col_burning))
            print(f"  t={t:.1f}s  front={x_front:.2f}m  burning={_n_burn}  "
                  f"dt={dt:.5f}s", flush=True)
        # Early exit if fire stalled (no advance in 30s).
        if _step % _stall_check_every == 0:
            if x_front <= _last_front + dx:
                break  # fire stalled
            _last_front = x_front
        # Source cells maintain T at ignition (external heat source).
        # For TGA kinetics (steep E ≈ 179 kJ/mol), 900K gives instant
        # burnout.  T_ign gives burn_time ≈ 17s — source lasts long
        # enough to establish the fire.
        if t < outdoor_cfg.ignition_duration_s:
            for k in range(n_z_bed):
                T_s[k, :_n_source] = np.maximum(T_s[k, :_n_source], T_ign)
            # Bootstrap heat input for source cells (matches deck
            # q_in_constant=50 kW/m² external source).  Without this,
            # HoG cap prevents source cells from pyrolyzing because
            # their own q_in_ext starts at 0 (gas not heated yet, no
            # slab radiation onto burning cells).  The clamp is
            # physically equivalent to an external heater applying
            # the deck's ignition flux.
            _q_src_kW = 50.0  # [kW/m²] from deck q_in_constant
            _q_in_ext_prev[:, :_n_source] = np.maximum(
                _q_in_ext_prev[:, :_n_source], _q_src_kW * 1000.0)

        # ── SIMPLE inner iterations (URANS pressure-velocity coupling) ──
        # Iterate momentum + pressure + k-ε until ν_t converges.
        # Without inner iterations, the single-pass fractional step
        # leaves velocity and ν_t inconsistent, causing k divergence.
        _n_inner = 10 if _use_ke else 1
        for _inner in range(_n_inner):

            # ── Momentum ─────────────────────────────────────────────────
            # Advection (upwind, sign-split).
            du = np.zeros_like(vel_u)
            dw = np.zeros_like(vel_w)

            _up = np.maximum(vel_u, 0.0)
            _un = np.minimum(vel_u, 0.0)
            _wp = np.maximum(vel_w, 0.0)
            _wn = np.minimum(vel_w, 0.0)

            # u-momentum advection.
            du[:, 1:] -= _up[:, 1:] * (vel_u[:, 1:] - vel_u[:, :-1]) / dx
            du[:, :-1] -= _un[:, :-1] * (vel_u[:, 1:] - vel_u[:, :-1]) / dx
            du[1:, :] -= _wp[1:, :] * (vel_u[1:, :] - vel_u[:-1, :]) / dz
            du[:-1, :] -= _wn[:-1, :] * (vel_u[1:, :] - vel_u[:-1, :]) / dz
            # w-momentum advection.
            dw[:, 1:] -= _up[:, 1:] * (vel_w[:, 1:] - vel_w[:, :-1]) / dx
            dw[:, :-1] -= _un[:, :-1] * (vel_w[:, 1:] - vel_w[:, :-1]) / dx
            dw[1:, :] -= _wp[1:, :] * (vel_w[1:, :] - vel_w[:-1, :]) / dz
            dw[:-1, :] -= _wn[:-1, :] * (vel_w[1:, :] - vel_w[:-1, :]) / dz

            # Diffusion (central) — eddy viscosity from turbulence model.
            # Strain-rate tensor (needed for both k-ε production and Smagorinsky).
            _S11 = np.zeros_like(vel_u)
            _S22 = np.zeros_like(vel_w)
            _S12 = np.zeros_like(vel_u)
            _S11[:, 1:-1] = (vel_u[:, 2:] - vel_u[:, :-2]) / (2.0 * dx)
            _S22[1:-1, :] = (vel_w[2:, :] - vel_w[:-2, :]) / (2.0 * dz)
            _S12[1:-1, 1:-1] = 0.5 * (
                (vel_u[2:, 1:-1] - vel_u[:-2, 1:-1]) / (2.0 * dz) +
                (vel_w[1:-1, 2:] - vel_w[1:-1, :-2]) / (2.0 * dx))
            _S_mag2 = 2.0 * (_S11**2 + _S22**2 + 2.0 * _S12**2)
            _S_mag = np.sqrt(_S_mag2)
            if _use_ke:
                # k-ε URANS: ν_t = C_μ k²/ε (Launder & Spalding 1974).
                # Under-relaxation (α=0.7, OpenFOAM default) for SIMPLE
                # convergence — prevents ν_t jumps between inner iterations.
                _nu_t_raw = _C_mu * _k_turb**2 / np.maximum(_eps_turb, 1e-10)
                _alpha_urf = 0.7
                _nu_t = _alpha_urf * _nu_t_raw + (1.0 - _alpha_urf) * _nu_t
            elif smagorinsky:
                _Delta = math.sqrt(dx * dz)
                _nu_t = (_C_smag * _Delta)**2 * _S_mag
            else:
                _nu_t = np.zeros((Nz, Nx))
            _nu_eff_arr = nu + _nu_t
            # Variable viscosity diffusion.
            du[:, 1:-1] += _nu_eff_arr[:, 1:-1] * (vel_u[:, :-2] - 2*vel_u[:, 1:-1] + vel_u[:, 2:]) / dx**2
            du[1:-1, :] += _nu_eff_arr[1:-1, :] * (vel_u[:-2, :] - 2*vel_u[1:-1, :] + vel_u[2:, :]) / dz**2
            dw[:, 1:-1] += _nu_eff_arr[:, 1:-1] * (vel_w[:, :-2] - 2*vel_w[:, 1:-1] + vel_w[:, 2:]) / dx**2
            dw[1:-1, :] += _nu_eff_arr[1:-1, :] * (vel_w[:-2, :] - 2*vel_w[1:-1, :] + vel_w[2:, :]) / dz**2

            # Buoyancy.
            buoy = _g * (T_g - T_amb) / max(T_amb, 1.0)

            # Porous-media aerodynamic drag (fuel bed only).
            # F_drag = -C_D × ρ × (a_v/2) × |u'| × u'   [N/m³]
            # Applied to PERTURBATION velocity u' = u - U_mean only:
            # the mean wind profile (Cionco) is sustained by the large-scale
            # pressure gradient (not resolved).  Drag damps the buoyancy-
            # driven perturbation that creates artificial horizontal transport.
            # C_D = 1.0 for cylinders at moderate Re (Morvan & Dupuy 2001,
            # Combust. Flame 127:1981; Linn et al. 2002, IJWF 11:233).
            _u_pert = vel_u[:n_z_bed] - u_inlet[:n_z_bed, np.newaxis]
            _w_pert = vel_w[:n_z_bed]   # mean w = 0
            _speed_pert = np.sqrt(_u_pert**2 + _w_pert**2)
            _drag_coeff = _C_D_drag * a_v * 0.5   # [1/m]
            du[:n_z_bed] -= _drag_coeff * _speed_pert * _u_pert
            dw[:n_z_bed] -= _drag_coeff * _speed_pert * _w_pert

            # ── k-ε transport (Launder & Spalding 1974) ──────────────
            if _use_ke:
                # Production: P_k = ν_t × |S|²
                _Pk = _nu_t * _S_mag2
                # Buoyancy production: G_k = (ν_t/Pr_t) × (g/T_amb) × dT/dz
                # (positive = unstable stratification = buoyancy generates k)
                _Gk = np.zeros_like(_k_turb)
                _Gk[1:-1, :] = (_nu_t[1:-1, :] / _Pr_t) * (_g / max(T_amb, 1.0)) * \
                    (T_g[2:, :] - T_g[:-2, :]) / (2.0 * dz)
                np.clip(_Gk, 0.0, None, out=_Gk)  # only unstable
                # Limit buoyancy production to shear production.
                # Prevents runaway in fire plumes where dT/dz is extreme
                # (1000s K/m).  Standard practice for buoyancy-dominated
                # flows (Rodi 1987; ANSYS Fluent Theory Guide §4.4.2).
                np.minimum(_Gk, _Pk, out=_Gk)
                # Porous drag dissipation (fuel bed only).
                _Dk = np.zeros_like(_k_turb)
                _Dk[:n_z_bed] = _C_D_drag * a_v * _speed_pert * _k_turb[:n_z_bed]
                # k equation: advect + diffuse + source.
                _dk = np.zeros_like(_k_turb)
                _dk[:, 1:]  -= _up[:, 1:]  * (_k_turb[:, 1:]  - _k_turb[:, :-1]) / dx
                _dk[:, :-1] -= _un[:, :-1] * (_k_turb[:, 1:]  - _k_turb[:, :-1]) / dx
                _dk[1:, :]  -= _wp[1:, :]  * (_k_turb[1:, :]  - _k_turb[:-1, :]) / dz
                _dk[:-1, :] -= _wn[:-1, :] * (_k_turb[1:, :]  - _k_turb[:-1, :]) / dz
                _alpha_k = nu + _nu_t / _sigma_k
                _dk[:, 1:-1] += _alpha_k[:, 1:-1] * (_k_turb[:, :-2] - 2*_k_turb[:, 1:-1] + _k_turb[:, 2:]) / dx**2
                _dk[1:-1, :] += _alpha_k[1:-1, :] * (_k_turb[:-2, :] - 2*_k_turb[1:-1, :] + _k_turb[2:, :]) / dz**2
                # Implicit source treatment for stability.
                # Split: dk/dt = ... + (P+G) - ε - D
                # Treat destruction (ε + D) implicitly:
                # k^(n+1) = (k^n + S_pos*dt) / (1 + S_neg*dt/k^n)
                _S_pos_k = _Pk + _Gk
                _S_neg_k = _eps_turb + _Dk
                _dk += _S_pos_k  # explicit positive sources
                _k_new = (_k_turb + _dk * dt) / (1.0 + _S_neg_k * dt / np.maximum(_k_turb, 1e-10))
                _alpha_k_urf = 0.8
                _k_turb = _alpha_k_urf * _k_new + (1.0 - _alpha_k_urf) * _k_turb
                np.clip(_k_turb, 1e-8, None, out=_k_turb)
                # ε equation: advect + diffuse + source.
                # dε/dt = ... + C_1ε(P+G)ε/k - C_2ε ε²/k
                # Treat C_2ε ε²/k implicitly (stiff destruction).
                _deps = np.zeros_like(_eps_turb)
                _deps[:, 1:]  -= _up[:, 1:]  * (_eps_turb[:, 1:]  - _eps_turb[:, :-1]) / dx
                _deps[:, :-1] -= _un[:, :-1] * (_eps_turb[:, 1:]  - _eps_turb[:, :-1]) / dx
                _deps[1:, :]  -= _wp[1:, :]  * (_eps_turb[1:, :]  - _eps_turb[:-1, :]) / dz
                _deps[:-1, :] -= _wn[:-1, :] * (_eps_turb[1:, :]  - _eps_turb[:-1, :]) / dz
                _alpha_eps = nu + _nu_t / _sigma_eps
                _deps[:, 1:-1] += _alpha_eps[:, 1:-1] * (_eps_turb[:, :-2] - 2*_eps_turb[:, 1:-1] + _eps_turb[:, 2:]) / dx**2
                _deps[1:-1, :] += _alpha_eps[1:-1, :] * (_eps_turb[:-2, :] - 2*_eps_turb[1:-1, :] + _eps_turb[2:, :]) / dz**2
                _ek_ratio = _eps_turb / np.maximum(_k_turb, 1e-10)
                # RNG correction (Yakhot & Orszag 1986): increases ε
                # dissipation in high-strain regions (fire plumes).
                # η = |S| k/ε;  C*_2ε = C_2ε + C_μ η³(1-η/η₀)/(1+β η³)
                _eta_rng = _S_mag * _k_turb / np.maximum(_eps_turb, 1e-10)
                _eta0 = 4.38; _beta_rng = 0.012
                _rng_corr = _C_mu * _eta_rng**3 * (1.0 - _eta_rng / _eta0) / \
                    (1.0 + _beta_rng * _eta_rng**3)
                # RNG correction adds dissipation (positive) when η < η₀
                # and reduces it (negative) when η > η₀.  Clip at zero
                # to only ADD dissipation in high-strain regions — the
                # physically intended effect for fire plumes.
                _C_2eff = _C_2eps + np.maximum(_rng_corr, 0.0)
                _S_pos_eps = _C_1eps * (_Pk + _Gk) * _ek_ratio
                _S_neg_eps = _C_2eff * _eps_turb * _ek_ratio  # C*_2ε ε²/k
                _deps += _S_pos_eps
                _eps_new = (_eps_turb + _deps * dt) / (1.0 + _S_neg_eps * dt / np.maximum(_eps_turb, 1e-10))
                _alpha_eps_urf = 0.8
                _eps_turb = _alpha_eps_urf * _eps_new + (1.0 - _alpha_eps_urf) * _eps_turb
                np.clip(_eps_turb, 1e-8, None, out=_eps_turb)
                # BCs: inlet Dirichlet, elsewhere zero-gradient.
                _k_turb[:, 0] = max(_k_inlet, 1e-8)
                _eps_turb[:, 0] = max(_eps_inlet, 1e-8)
                _k_turb[0, :] = _k_turb[1, :]     # ground: zero-gradient
                _eps_turb[0, :] = _eps_turb[1, :]
                _k_turb[-1, :] = _k_turb[-2, :]   # top: zero-gradient
                _eps_turb[-1, :] = _eps_turb[-2, :]

            u_star = vel_u + dt * du
            w_star = vel_w + dt * (dw + buoy)

            # Velocity BCs at inlet (x=0).
            if _use_ke:
                # URANS: steady mean profile, no velocity fluctuations.
                # Turbulence enters via k,ε at inlet (ghost cell), not
                # via resolved velocity perturbations.
                u_star[:, 0] = u_inlet
                w_star[:, 0] = 0.0
            else:
                # LES/SEM: Synthetic Eddy Method (Jarrin et al. 2006).
                _u_fluct = np.zeros(Nz)
                _w_fluct = np.zeros(Nz)
                for _ek in range(_sem_N):
                    _fx = _sem_tent(_sem_xk[_ek] / _sem_sigma)
                    _fz = _sem_tent((z_mid - _sem_zk[_ek]) / _sem_sigma)
                    _c = _fx * _fz
                    _u_fluct += _sem_eps[_ek, 0] * _c
                    _w_fluct += _sem_eps[_ek, 1] * _c
                _u_fluct *= _sem_inv_sqN * _sem_amp
                _w_fluct *= _sem_inv_sqN * _sem_amp
                _sem_xk += _sem_Uc * dt
                _ex = _sem_xk > _sem_sigma
                if np.any(_ex):
                    _n_ex = int(np.sum(_ex))
                    _sem_xk[_ex] = -_sem_sigma
                    _sem_zk[_ex] = _sem_rng.uniform(
                        _sem_zmin - _sem_sigma, _sem_zmax + _sem_sigma, size=_n_ex)
                    _sem_eps[_ex] = _sem_rng.choice([-1.0, 1.0], size=(_n_ex, 2))
                u_star[:, 0] = u_inlet + _u_fluct
                w_star[:, 0] = _w_fluct
            u_star[0, :] = 0.0
            u_star[-1, :] = u_star[-2, :]
            w_star[0, :] = 0.0
            w_star[-1, :] = 0.0

            # Projection (approximate variable-density: Almgren et al. 2000).
            # Poisson matrix uses constant coefficients (LU reused).
            # Variable density enters the velocity CORRECTION only.
            if variable_density:
                _rho_g = 101325.0 / (287.0 * np.maximum(T_g, T_amb))
            div = np.zeros((Nz, Nx))
            div[:, :-1] += (u_star[:, 1:] - u_star[:, :-1]) / dx
            div[:-1, :] += (w_star[1:, :] - w_star[:-1, :]) / dz
            # Low-Mach divergence: ∇·u = (1/T) DT/Dt (Rehm & Baum 1978).
            if low_mach:
                _Q_exp = np.zeros((Nz, Nx))
                _T_safe = np.maximum(T_g, T_amb)
                _rcp_lm = (_rho_g * cp_g) if variable_density else (rho0 * cp_g)
                _Q_exp[:n_z_bed] = hp_av * (T_s[:n_z_bed] - T_g[:n_z_bed]) / _rcp_lm[:n_z_bed] if variable_density else \
                    hp_av * (T_s[:n_z_bed] - T_g[:n_z_bed]) / (rho0 * cp_g)
                for k in range(n_z_bed):
                    _hd = np.maximum(T_flame_adiabatic - T_g[k, :], 0.0) / max(T_flame_adiabatic - T_amb, 1.0)
                    if variable_density:
                        _Q_exp[k, _col_burning] += Q_comb_static / (_rho_g[k, _col_burning] * cp_g) * _hd[_col_burning]
                    else:
                        _Q_exp[k, _col_burning] += Q_comb_static / (rho0 * cp_g) * _hd[_col_burning]
                _Q_exp /= _T_safe
                div -= _Q_exp
            rhs = (rho0 / dt) * div.ravel()
            rhs[-1] = 0.0
            P_flat = _poi_lu.solve(rhs)
            P = P_flat.reshape(Nz, Nx)

            dPdx = np.zeros_like(vel_u)
            dPdx[:, 1:] = (P[:, 1:] - P[:, :-1]) / dx
            dPdz = np.zeros_like(vel_w)
            dPdz[1:, :] = (P[1:, :] - P[:-1, :]) / dz

            # Velocity correction with y-ghost-cell leakage.
            # In 3D, buoyancy compensation splits across x, y, z (1/3 each
            # for isotropic turbulence).  In 2D, all goes to x (1/2 each).
            # Ghost cell: fraction f_y of the horizontal pressure correction
            # "escapes to y" instead of driving x-velocity.
            # f_y = 1/3 — isotropic partition, geometrically derived.
            _f_y = 1.0 / 3.0
            _vel_u_new = u_star - (1.0 - _f_y) * (dt / rho0) * dPdx
            _vel_w_new = w_star - (dt / rho0) * dPdz
            # Under-relax velocity (SIMPLE, Patankar 1980).
            _alpha_u = 0.8
            vel_u = _alpha_u * _vel_u_new + (1.0 - _alpha_u) * vel_u
            vel_w = _alpha_u * _vel_w_new + (1.0 - _alpha_u) * vel_w
            vel_u[:, 0] = u_inlet if _use_ke else u_inlet + _u_fluct
            vel_u[0, :] = 0.0
            vel_w[0, :] = 0.0
            vel_w[-1, :] = 0.0
            # ── End SIMPLE inner iteration ────────────────────────

        # ── Energy ───────────────────────────────────────────────────
        dTg = np.zeros_like(T_g)
        # Advection (sign-split upwind).
        _up2 = np.maximum(vel_u, 0.0)
        _un2 = np.minimum(vel_u, 0.0)
        _wp2 = np.maximum(vel_w, 0.0)
        _wn2 = np.minimum(vel_w, 0.0)
        dTg[:, 1:] -= _up2[:, 1:] * (T_g[:, 1:] - T_g[:, :-1]) / dx
        dTg[:, :-1] -= _un2[:, :-1] * (T_g[:, 1:] - T_g[:, :-1]) / dx
        dTg[1:, :] -= _wp2[1:, :] * (T_g[1:, :] - T_g[:-1, :]) / dz
        dTg[:-1, :] -= _wn2[:-1, :] * (T_g[1:, :] - T_g[:-1, :]) / dz
        # Diffusion — turbulent diffusivity from k-ε or Smagorinsky.
        _alpha_eff_arr = alpha_th + _nu_t / _Pr_t
        dTg[:, 1:-1] += _alpha_eff_arr[:, 1:-1] * (T_g[:, :-2] - 2*T_g[:, 1:-1] + T_g[:, 2:]) / dx**2
        dTg[1:-1, :] += _alpha_eff_arr[1:-1, :] * (T_g[:-2, :] - 2*T_g[1:-1, :] + T_g[2:, :]) / dz**2
        # Coupling + combustion source (fuel zone).
        # With variable_density: ρ(T) = P₀/(R·T) replaces constant ρ₀
        # in the thermal-mass denominator.  Transport terms (advection,
        # diffusion above) are already ρ-independent in temperature form.
        # Gas-solid coupling: CONVECTIVE only (hp_av = h_p × a_v).
        # The gas is transparent to thermal radiation — h_rad acts
        # between solid particles, not gas↔solid.  The solid equation
        # uses hp_eff = (h_conv + h_rad) × a_v because particles
        # exchange radiation with each other; the gas equation uses
        # h_conv × a_v only.  This asymmetry is physical.
        # Gas-solid coupling (convective only).
        if variable_density:
            _rho_loc = 101325.0 / (287.0 * np.maximum(T_g, T_amb))
            _rho_cp_loc = _rho_loc * cp_g
            dTg[:n_z_bed] -= hp_av * (T_g[:n_z_bed] - T_s[:n_z_bed]) / _rho_cp_loc[:n_z_bed]
        else:
            dTg[:n_z_bed] -= hp_av * (T_g[:n_z_bed] - T_s[:n_z_bed]) / (rho0 * cp_g)
        # Per-cell Arrhenius — sequential pool depletion.
        # Each pool has its own mass; m_dot_pool = A × exp(-E/RT) × m_pool.
        # Hemicellulose depletes first (lower E), bootstrapping cellulose.
        _T_s_bed = T_s[:n_z_bed]
        _T_safe = np.maximum(_T_s_bed, T_amb)
        _mdot_hemi = _A_hemi * np.exp(-_E_hemi / (_R_gas * _T_safe)) * _m_hemi
        _mdot_cell = _A_cell * np.exp(-_E_cell / (_R_gas * _T_safe)) * _m_cell
        _mdot_lign = _A_lign * np.exp(-_E_lign / (_R_gas * _T_safe)) * _m_lign
        _m_dot_py = _mdot_hemi + _mdot_cell + _mdot_lign          # [kg/m²/s]
        # Moisture suppression: pyrolysis blocked while local water > 0.
        # Grishin (1984); Margerit & Séro-Guillaume (2002).  Physically:
        # while water is evaporating, T_s is pinned near T_evap (373K)
        # and the bed temperature can't rise above water evaporation
        # until water is gone.  Models this via a smooth gate that
        # vanishes when water mass exceeds 1% of initial.
        _m_water_init = rho_bulk * _M_f * dz
        if _m_water_init > 0:
            _wet = _m_water[:n_z_bed] / _m_water_init
            _moist_gate = np.maximum(0.0, 1.0 - 100.0 * _wet)  # 0 if wet > 1%
            _mdot_hemi *= _moist_gate
            _mdot_cell *= _moist_gate
            _mdot_lign *= _moist_gate
        # Heat-of-gasification cap: m_dot ≤ q_in_ext / L_HoG.  Limits
        # Arrhenius kinetics to physically achievable rates given the
        # external heat input (radiation + convection from neighbors).
        # Quintiere (2006); standard cone-calorimeter physics extended
        # to the spread context.  Excludes flame_back (self-feedback)
        # to avoid runaway: external heat sets the maximum, not local
        # flame from this cell's own combustion.
        _m_dot_max = np.maximum(_q_in_ext_prev, 0.0) / _L_HoG       # [kg/m²/s]
        _scale = np.minimum(1.0, _m_dot_max / np.maximum(_m_dot_py, 1e-12))
        _mdot_hemi *= _scale
        _mdot_cell *= _scale
        _mdot_lign *= _scale
        _m_dot_py = _mdot_hemi + _mdot_cell + _mdot_lign

        # ── Gas-phase volatile transport + combustion ─────────────
        # Operator split: produce → advect → combust.
        # Volatiles produced by pyrolysis enter the gas phase, advect
        # with the momentum field, and burn at a mixing-limited rate.
        # McDermott et al. (2011) NIST; Morvan & Dupuy (2001).
        #
        # 1. Pyrolysis → combustible volatiles (η_comb knockdown).
        _m_dot_vol = (_eta_hemi * _mdot_hemi +
                      _eta_cell * _mdot_cell +
                      _eta_lign * _mdot_lign)                    # [kg/m²/s]
        _m_vol += _m_dot_vol * dt
        # 2. Advect volatiles (upwind, follows momentum field).
        _dm_adv = np.zeros_like(_m_vol)
        _ub = np.maximum(vel_u[:n_z_bed], 0.0)
        _un = np.minimum(vel_u[:n_z_bed], 0.0)
        _wb = np.maximum(vel_w[:n_z_bed], 0.0)
        _wn = np.minimum(vel_w[:n_z_bed], 0.0)
        _dm_adv[:, 1:]  -= _ub[:, 1:]  * (_m_vol[:, 1:]  - _m_vol[:, :-1]) / dx
        _dm_adv[:, :-1] -= _un[:, :-1] * (_m_vol[:, 1:]  - _m_vol[:, :-1]) / dx
        if n_z_bed > 1:
            _dm_adv[1:, :]  -= _wb[1:, :]  * (_m_vol[1:, :]  - _m_vol[:-1, :]) / dz
            _dm_adv[:-1, :] -= _wn[:-1, :] * (_m_vol[1:, :]  - _m_vol[:-1, :]) / dz
        _m_vol += _dm_adv * dt
        np.clip(_m_vol, 0.0, None, out=_m_vol)
        # 3. Mixing-limited combustion — τ_mix from turbulence model.
        if _use_ke:
            # τ_mix = k/ε — physical, grid-independent (Morvan & Dupuy 2001).
            _tau_mix_arr = _k_turb[:n_z_bed] / np.maximum(_eps_turb[:n_z_bed], 1e-10)
            _m_dot_comb = _m_vol / np.maximum(_tau_mix_arr, dt)
        else:
            _m_dot_comb = _m_vol / _tau_mix_fallback
        _m_dot_comb = np.minimum(_m_dot_comb, _m_vol / max(dt, 1e-10))
        _m_vol -= _m_dot_comb * dt
        np.clip(_m_vol, 0.0, None, out=_m_vol)
        # Heat release from gas-phase combustion.
        _Q_gas = _m_dot_comb * _hoc_J * (1.0 - chi_rad) / h_bed  # [W/m³]
        for k in range(n_z_bed):
            _headroom = np.maximum(T_flame_adiabatic - T_g[k, :], 0.0) / \
                max(T_flame_adiabatic - T_amb, 1.0)
            _Q_k = _Q_gas[k, :] * _headroom
            if variable_density:
                dTg[k, _col_burning] += _Q_k[_col_burning] / _rho_cp_loc[k, _col_burning]
            else:
                dTg[k, _col_burning] += _Q_k[_col_burning] / (rho0 * cp_g)
        # Y-direction ghost-cell diffusion (far-field T_amb at y=±L_y).
        # Applied to all cells — combustion source in burning cells
        # sustains their temperature; the y-loss damps forward-leaked
        # hot gas in unburned cells where there is no combustion source.
        if _k_y_loss > 0:
            dTg -= _k_y_loss * (T_g - T_amb)

        T_g += dTg * dt
        # Cap at adiabatic flame temperature (physical upper bound).
        np.clip(T_g, T_amb, T_flame_adiabatic, out=T_g)
        T_g[-1, :] = T_amb  # top BC
        T_g[0, :] = T_g[1, :]  # ground: zero gradient

        # Solid: hp_eff = (h_conv + h_rad) × a_v includes inter-particle
        # radiation (transparent gas, particles radiate to each other).
        _Tl = 0.5 * (T_g[:n_z_bed] + T_s[:n_z_bed])
        _hr = 4.0 * eps_s * sigma_sb * _Tl**3
        _hpe = (h_p + _hr) * a_v
        q_gs = _hpe * dz * (T_g[:n_z_bed] - T_s[:n_z_bed])
        q_loss = eps_s * sigma_sb * (T_s[:n_z_bed]**4 - T_amb**4) * a_v * dz
        # Slab radiation: flame radiates horizontally to all fuel layers.
        # Each layer absorbs f_abs and transmits (1-f_abs) to the next.
        # Top-down attenuation through the bed: layer k sees the incident
        # flux reduced by the layers above it.
        q_rad = np.zeros((n_z_bed, Nx))
        _bc = np.where(_col_burning)[0]
        if len(_bc) > 0 and L_f > 0:
            _lb = int(_bc[-1])
            # Slab radiation source HRRPUA scaled by w_0/w_0_ref to handle
            # different fuel loads.  The deck cone-element (density=171,
            # thickness=35mm) calibrates to outdoor w_0 ≈ 0.072 kg/m² (GR1).
            # Wildland fires with denser fuel produce stronger flames per
            # Byram (1959) I_B = HoC × w_0 × ROS; without rescaling the
            # static cone HRRPUA, dense beds (Cheney Nat/Cut) stall
            # because radiative reach is insufficient to ignite the
            # higher thermal mass per cell.  Albini (1985) IJWF: flame
            # radiation tracks fireline intensity, not external
            # ignition source.
            _w0_ref = 0.072
            _HRRPUA_loc = _HRRPUA_W * max(_w_0 / _w0_ref, 1.0)
            _fe = x_mid[_lb] + 0.5 * dx
            _be = x_mid[_bc[0]] - 0.5 * dx
            _is = _lb + 1
            if _is < Nx:
                _dn = np.maximum(x_mid[_is:] - _fe, 0.5 * dx)
                _df = x_mid[_is:] - _be
                _st = math.sin(theta_tilt)
                _rn = np.maximum(_dn - L_f * _st, 1e-3)
                _rf = np.maximum(_df - L_f * _st, 1e-3)
                _Fn = 0.5 * (1.0 - _rn / np.sqrt(L_f**2 + _rn**2))
                _Ff = 0.5 * (1.0 - _rf / np.sqrt(L_f**2 + _rf**2))
                _Fs = np.maximum(_Fn - _Ff, 0.0)
                _q_inc = chi_rad * _HRRPUA_loc * _Fs
                _transmit = 1.0 - _f_abs
                for k in range(n_z_bed - 1, -1, -1):
                    q_rad[k, _is:] = _q_inc * _f_abs
                    _q_inc = _q_inc * _transmit

        # Flame radiation feedback to solid surface [W/m²].
        # Each cell receives max(own flame, upstream neighbor's flame).
        # A newly-ignited cell's own m_dot is near zero, but the cell
        # behind it has been burning longer — its flame illuminates the
        # new cell's surface.  This is the dominant ignition mechanism
        # at the fire front.
        # Flame feedback from cell's own pyrolysis only.
        # Flame feedback from gas-phase combustion rate.
        # Morvan & Dupuy (2001): flame intensity depends on volatile
        # combustion (gas-phase), not instantaneous pyrolysis (solid).
        _q_flame_back = chi_rad * _m_dot_comb * _hoc_J * _flame_vf

        # Save EXTERNAL heat input (radiation + gas-solid coupling minus
        # losses) for next-iteration HoG cap on pyrolysis.  Excludes
        # _q_flame_back (self-feedback) to prevent runaway.
        _q_in_ext_prev = q_rad + np.maximum(q_gs, 0.0) - q_loss

        # Moisture evaporation: consume q_in to evaporate water.
        # Grishin (1984); Margerit & Séro-Guillaume (2002).  Heat
        # going into water doesn't heat the solid until water is gone.
        # Only positive (heating) fluxes contribute to evaporation;
        # negative q_gs (gas cools solid) and q_loss still apply to dTs.
        _q_in_pos = np.maximum(q_gs, 0.0) + q_rad + _q_flame_back  # [W/m²]
        _q_in_neg = np.minimum(q_gs, 0.0)                          # [W/m²]
        _has_water = _m_water > 0
        if np.any(_has_water):
            _dm_evap = np.where(_has_water,
                np.minimum(_m_water, _q_in_pos * dt / _L_v), 0.0)
            _m_water -= _dm_evap
            np.clip(_m_water, 0.0, None, out=_m_water)
            _q_used = _dm_evap * _L_v / max(dt, 1e-12)
            _q_in_pos = np.maximum(_q_in_pos - _q_used, 0.0)

        dTs = np.zeros_like(T_s)
        dTs[:n_z_bed] = (_q_in_pos + _q_in_neg - q_loss) / C_s

        # ── Buoyancy vortex flame contact (Finney et al. 2015) ───────
        # At low wind, counter-rotating vortex pairs create periodic
        # downward flame bursts that contact unburned fuel ahead of
        # the fire front.  This is the dominant ignition mechanism
        # when wind-driven gas transport is weak (U < U_buoy).
        # Finney et al. (2015) PNAS 112:9833; Tang et al. (2017) Proc.
        # Combust. Inst.; Tang et al. (2019) Front. Mech. Eng. 5:34.
        _bc = np.where(_col_burning)[0]
        if len(_bc) > 0 and L_f > 0:
            _delta_burst = 0.2 * L_f   # burst reach (Fig. 1I)
            # Peak convective contact flux during flame burst [W/m²].
            # Frankman et al. (2013) IJWF 22:157: 5-22 kW/m² surface.
            _q_burst_peak = 5000.0
            if _delta_burst > dx:
                _lb = int(_bc[-1])
                _is = _lb + 1
                _ie = min(_is + max(1, int(_delta_burst / dx)), Nx)
                for i in range(_is, _ie):
                    _dist = (i - _lb) * dx
                    # Local Richardson number Ri_x = g ΔT x / (T_amb U²).
                    # Contact frequency:
                    #   Ri_x < 1 (attached): f_C = 0 — gas-solid handles it
                    #   Ri_x > 1 (intermittent): f_C ~ Ri_x^-0.7 (Tang 2019)
                    _dT_fire = max(float(np.mean(T_g[:n_z_bed, _lb])) - T_amb, 1.0)
                    _Ri_x = _g * _dT_fire * max(_dist, dx) / \
                        (max(T_amb, 1.0) * max(U_mf, 0.01)**2)
                    if _Ri_x > 1.0:
                        _f_contact = _Ri_x ** (-0.7)
                    else:
                        _f_contact = 0.0
                    _atten = max(0.0, 1.0 - _dist / _delta_burst)
                    _q_burst = _q_burst_peak * _f_contact * _atten
                    dTs[:n_z_bed, i] += _q_burst / C_s

        T_s += dTs * dt

        # Deplete fuel mass.
        # Deplete each pool independently.
        _m_hemi -= _mdot_hemi * dt
        _m_cell -= _mdot_cell * dt
        _m_lign -= _mdot_lign * dt
        np.clip(_m_hemi, 0.0, None, out=_m_hemi)
        np.clip(_m_cell, 0.0, None, out=_m_cell)
        np.clip(_m_lign, 0.0, None, out=_m_lign)
        t += dt

        # ── Ignition ─────────────────────────────────────────────────
        _Ttop = T_s[n_z_bed - 1]
        newly = (_Ttop >= T_ign) & (~_col_burning)
        if np.any(newly):
            _col_burning |= newly
            _nc = np.where(newly)[0]
            _t_ign_col[_nc] = t
            _nf = float(x_mid[_nc[-1]])
            if _nf > x_front:
                x_front = _nf
                front_history_t.append(t)
                front_history_x.append(x_front)
                # ── Dynamic flame length (Byram 1959) ────────────
                # I_B = HoC × w_0 × ROS — grows with fire intensity.
                # L_f = 0.0475 × I_B_kW^0.493 [m] (same formula as
                # byram_flame_length but with dynamic I_B).
                # No new free parameters: HoC and w_0 from fuel deck.
                if len(front_history_t) >= 3:
                    _ros_cur = (front_history_x[-1] - front_history_x[-2]) / \
                        max(front_history_t[-1] - front_history_t[-2], 1e-8)
                    _I_B_kW = _hoc_J * _w_0 * _ros_cur / 1000.0  # [kW/m]
                    if _I_B_kW > 0:
                        L_f = 0.0475 * _I_B_kW ** 0.493
                        theta_tilt = flame_tilt_angle(wind_speed_m_s, L_f,
                                                      outdoor_cfg.terrain)

        # ── Burnout (mass-based) ─────────────────────────────────────
        _col_fuel_sum = np.sum(_m_hemi + _m_cell + _m_lign, axis=0)
        _burned_out = _col_burning & (_col_fuel_sum < n_z_bed * _m_fuel_min)
        if np.any(_burned_out):
            _col_burning[_burned_out] = False

        if x_front > pde_domain_m * 0.9:
            break

    # ── ROS (steady-state, burst-mode rejected) ─────────────────────────
    # Burst-mode is the dominant non-physical artifact at high fuel loads:
    # the source-clamp creates a heat plume that transiently ignites many
    # cells in <1s, then the fire stalls when source clamp releases.  If we
    # used the full front history, ROS would capture the burst.
    #
    # Steady-state criterion: ROS is meaningful only if the fire is still
    # advancing AT THE END of the simulation.  If it stalled (no front
    # advance in the last 30s), report ROS=0 — the fire is not self-
    # sustaining at this wind/fuel/moisture combination.
    ft = np.array(front_history_t)
    fx = np.array(front_history_x)
    _source_x = float(x_mid[min(_n_source - 1, Nx - 1)])
    _advance_cells = (fx[-1] - _source_x) / dx if len(fx) > 0 else 0
    # Stall rejection: if last advance was > 30s before sim end, fire died.
    _t_since_advance = t - ft[-1] if len(ft) > 0 else 1e9
    _reached_far = (fx[-1] > pde_domain_m * 0.7) if len(fx) > 0 else False
    if _advance_cells < 5 or (_t_since_advance > 30.0 and not _reached_far):
        ros_m_s = 0.0
    else:
        # Use the last 30s of history (or all if shorter) — captures
        # steady-state, rejects initial source-driven burst.
        _t_window = 30.0
        _t_min = max(ft[-1] - _t_window, ft[0])
        _idx = np.searchsorted(ft, _t_min)
        if _idx >= len(ft) - 1:
            _idx = max(0, len(ft) - 2)
        if ft[-1] > ft[_idx]:
            ros_m_s = (fx[-1] - fx[_idx]) / (ft[-1] - ft[_idx])
        else:
            ros_m_s = 0.0

    return SpreadResult(
        t_ignition=[0.0], cell_t=[src_t], cell_hrrpua=[src_hrrpua],
        ros_m_s=ros_m_s, n_cells_ignited=int(np.sum(_col_burning)),
        spread_cfg=spread_cfg, n_jump_list=[],
    )
