from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FuelConfig:
    """Lumped fuel model parameters (2-node default, optional 3-node prototype)."""

    C1: float = 2.0e5  # [J/m^2/K] surface node heat capacity
    C2: float = 5.0e5  # [J/m^2/K] interior node heat capacity
    K12: float = 60.0  # [W/m^2/K] inter-node coupling
    thermal_model_order: int = 2  # [-] 2 | 3 thermal nodes (3-node is prototype)
    C3: float | None = None  # [J/m^2/K] deep interior/back node heat capacity (3-node)
    K23: float | None = None  # [W/m^2/K] node2-node3 coupling (3-node)
    C4: float | None = None  # [J/m^2/K] node-4 heat capacity (4/5-node)
    K34: float | None = None  # [W/m^2/K] node3-node4 coupling (4/5-node)
    C5: float | None = None  # [J/m^2/K] node-5 heat capacity (5-node)
    K45: float | None = None  # [W/m^2/K] node4-node5 coupling (5-node)
    eps: float = 0.9  # [-] surface emissivity
    h_amb: float = 10.0  # [W/m^2/K] legacy/extra convective HTC (kept for compatibility)
    h_fg: float = 2.26e6  # [J/kg] latent heat of evaporation
    dH_py: float = 0.0  # [J/kg] optional endothermic pyrolysis sink
    A_py: float = 5.0e5  # [1/s] if A_basis=mass, [kg/m^2/s] if A_basis=flux
    E_py: float = 1.2e5  # [J/mol] activation energy
    R: float = 8.314462618  # [J/mol/K] gas constant
    alpha_moist: float = 3.0  # [-] moisture suppression factor
    kinetics_mode: str = "arrhenius"  # arrhenius | sigmoid | two_step | two_step_sequential | semi_global_seq_yield
    sigmoid_T0_K: float = 650.0  # [K] sigmoid midpoint
    sigmoid_dT_K: float = 25.0  # [K] sigmoid width
    A1_py: float | None = None  # branch 1 pre-exponential (same units as A_py)
    E1_py: float | None = None  # [J/mol] two-step branch 1 activation
    A2_py: float | None = None  # branch 2 pre-exponential (same units as A_py)
    E2_py: float | None = None  # [J/mol] two-step branch 2 activation
    A3_py: float | None = None  # branch 3 pre-exponential (secondary-char sink in sequential mode)
    E3_py: float | None = None  # [J/mol] branch 3 activation (secondary-char sink in sequential mode)
    # Sequential two-step kinetics (PB fuel_state prototype)
    seq_y1_vol: float = 0.0  # [-] volatile yield from stage-1 reaction
    seq_y2_vol: float = 0.7  # [-] volatile yield from stage-2 reaction
    seq_f12_to_m2: float = 1.0  # [-] fraction of stage-1 nonvolatile remainder routed to stage-2 pool
    seq_secondary_char_enable: bool = False  # [-] enable optional stage-2 secondary charring sink (A3/E3)
    seq_m1_frac: float = 1.0  # [-] initial stage-1 condensed mass fraction
    seq_m2_frac0: float = 0.0  # [-] initial stage-2 condensed mass fraction
    seq_mr_frac0: float = 0.0  # [-] initial residue condensed mass fraction
    seq_clamp_yields: bool = True  # [-] clamp yields to [0,1] (otherwise raise on invalid)
    seq_T_ign_K: float = 560.0    # [K] surface-temperature gate for staged vol_frac: HRRPUA=0 below this
    seq_vol_interp_n: float = 0.0  # [-] >0: power-law vol_frac = y2+(y1-y2)*(m1/m1_0)^n; 0=rate-weighted
    seq_pool2_use_back_node: bool = False  # [-] Pool 2 Arrhenius uses T_back (last thermal node) instead of T1
    # Semi-global sequential product-yield prototype (PB fuel_state mode)
    sg_n1: float = 1.0  # [-] stage-1 reaction order
    sg_n2: float = 1.0  # [-] stage-2/intermediate reaction order
    sg_n3: float = 1.0  # [-] optional secondary-char sink reaction order (A3/E3)
    sg_y_g1: float = 0.15  # [-] stage-1 direct volatile/gas yield
    sg_y_i1: float = 0.55  # [-] stage-1 intermediate condensed yield (feeds stage-2 pool)
    sg_y_c1: float = 0.30  # [-] stage-1 direct char/residue yield
    sg_y_g2: float = 0.70  # [-] stage-2 volatile/gas yield
    sg_y_c2: float = 0.30  # [-] stage-2 char/residue yield
    sg_clamp_yields: bool = True  # [-] clamp semi-global yields to [0,1] (otherwise raise on invalid)
    # Optional reduced transport/accessibility limiter for delayed pathways in thick charring solids
    reactive_access_mode: str = "none"  # none | transport_reduced_wood_char
    access_reduction_beta: float = 2.0  # [-] accessibility reduction strength (bounded reduction control)
    access_min: float = 0.2  # [-] minimum delayed-path accessibility factor
    access_driver: str = "residue_fraction"  # residue_fraction | residue_and_depth
    access_target: str = "stage2_only"  # stage2_only | delayed_paths
    access_depth_weight_pow: float = 1.0  # [-] depth weighting exponent for residue_and_depth driver
    chem_depth_partition_mode: str = "none"  # none | thermal_nodes (3-node local-rate chemistry partition)
    # Pyrolysis state interpretation and basis selection
    M1_represents: str = "fraction"  # fraction | kg_m2
    pyrolysis_mass_source: str = "legacy_M1"  # legacy_M1 | fuel_state
    m_fuel_total_kg_m2: float | None = None  # [kg/m^2] optional explicit slab mass/area
    A_basis: str = "mass"  # mass -> A in 1/s, flux -> A in kg/m^2/s
    # Legacy alias retained for backward compatibility with older tests/decks
    pyrolysis_rate_basis: str = "mass"  # deprecated alias of A_basis
    k_evap0: float = 0.02  # [1/s] base evaporation rate
    T_evap_onset: float = 373.0  # [K] evaporation onset temperature
    m1_max_kg_m2: float = 0.5  # [kg/m^2] surface water storage capacity
    q_loss2: float = 0.0  # [W/m^2] interior heat loss
    q_loss3: float | None = None  # [W/m^2] deep-node heat loss (3-node); defaults to q_loss2
    m_fuel_kg_m2: float = 1.0  # [kg/m^2] dry fuel mass per area (legacy)
    rho_solid: float | None = None  # [kg/m^3] optional solid density for front-limit cap
    rho: float | None = None  # [kg/m^3] optional density used for m_tot fallback
    thickness_m: float | None = None  # [m] specimen thickness used for m_tot fallback
    enable_depletion: bool = True  # [-] apply simple fuel depletion to m_py''
    pyrolysis_mode: str = "arrhenius"  # arrhenius | prescribed
    m_py_schedule: list[tuple[float, float]] = field(default_factory=list)  # [(t_s, m_py_kg_m2_s)]
    # PMMA front-limit/regression controls
    front_limit_enable: bool = False  # [-] enable delta/mass-limited PMMA branch
    front_limit_surface_only: bool = False  # [-] Stefan cap on surface node only; Arrhenius for deeper nodes
    regression_alpha: float = 0.0  # [m^2/s] depth-growth coefficient in alpha/max(delta,delta_min)
    regression_T_py_K: float | None = None  # [K] if set, enables dynamic Stefan: alpha = k_char*(T-T_py)/(rho*dH_py)
    regression_delta_min_m: float = 1.0e-6  # [m] minimum depth for stable growth
    regression_delta_cap_m: float | None = None  # [m] char depth cap for Stefan rate (None = uncapped)
    regression_spall_onset_frac: float = 0.0     # [-] delta_py/L at which char fall-off begins (0=disabled)
    regression_spall_reduction_frac: float = 0.0  # [-] max fractional reduction of delta_capped at full penetration
    softmin_beta: float = 30.0  # [-] smooth-min sharpness
    handoff_start_frac: float = 0.9  # [-] start blending cap-limited -> kinetics-limited
    handoff_end_frac: float = 1.0  # [-] finish blending cap-limited -> kinetics-limited
    delta_py0_m: float = 0.0  # [m] initial pyrolyzed depth
    m_char0_kg_m2: float = 0.0  # [kg/m^2] initial consumed/charred mass basis
    regression_L0_m: float = 1.0  # [m] initial full thickness/depth basis
    # Flame radiation feedback — De Ris / Tewarson closure: q_fb = chi_rad × F × HRRPUA
    # Activated by 2-pass coupling in rom_adapter (fuel.flame_enable = true in deck)
    flame_enable: bool = False
    flame_chi_rad: float = 0.35          # [-] radiative fraction of HRR (wood: 0.25–0.35, Tewarson SFPE §3.4)
    flame_view_factor: float = 0.40      # [-] view factor from flame to fuel surface (cone cal geometry)
    flame_persistence_s: float = 5.0    # [s] flame persistence window after extinction criterion
    flame_m_py_ignite: float = 0.005    # [kg/m²/s] pyrolysis flux threshold for ignition
    flame_m_py_crit: float = 0.001      # [kg/m²/s] pyrolysis flux threshold for extinction
    flame_T_ignite: float = 600.0       # [K] surface temperature threshold for ignition
    flame_T_py: float = 500.0           # [K] minimum surface temperature to sustain burning
    flame_hog_split_flux: bool = False  # [-] if True, HoG ceiling uses cone flux only (not cone+feedback); avoids double-counting when L_eff was calibrated against EXP
    flame_tau_growth_s: float = -1.0    # [s] flame growth ramp timescale; -1 = no ramp (instant full flame).
                                         # >0: q_fb *= (1 - exp(-(t-t_ign)/tau)) — exponential saturation.
                                         # Physical basis: flame grows from pilot to full coverage over ~τ s;
                                         # weaker flux → slower flame establishment. Set to EXP rise time.
    flame_coupling_passes: int = 10      # [-] max flame coupling iterations; exits early via tolerance below.
                                         # Contraction ratio = chi_rad×view_factor ≈ 0.10 → converges in ~3-4 passes.
                                         # Set to 1 to reproduce legacy single-pass (2-ODE-solve) behavior.
    flame_coupling_tol_W_m2: float = 1.0  # [W/m²] L∞ HRRPUA convergence tolerance for early exit.
    # Finite-bed flame geometry (Heskestad 1983 + Drysdale 1999) — sub-1 m³ free-burning
    # Activated when flame_area_m2 is set AND flame_geometry_mode = "heskestad".
    # The static deck flame_view_factor is then overridden per-step with F(HRRPUA, area).
    flame_area_m2: float | None = None          # [m²] fuel bed plan area; None = disabled (deck F used)
    flame_geometry_mode: str = "deck"           # "deck" | "heskestad" — view-factor source
    flame_plume_heights_m: str | None = None    # semicolon-separated heights [m] for McCaffrey plume output
    # Stefan char-front crack enhancement (Di Blasi 2002, Moghtaderi 2006, Shi & Chew 2023)
    # k_char_eff = k_char * (1 + k_crack_frac * alpha_bar); 0 = intact char (default), 2-4 = fire-cracked range
    k_crack_frac: float = 0.0   # [-] char crack conductivity enhancement factor
    k_crack_ode_enable: bool = False  # [-] apply crack enhancement to ODE inter-node K_char links (N-node only)
    # Pre-ignition volatile pool burst (Sanned et al. 2023, Babrauskas 2023)
    # Integrates Arrhenius m_dot_kin from t=0 to t_ignite; burns as Gaussian burst at ignition
    vol_pool_enable: bool = False
    vol_pool_tau_auto: bool = False  # if True, compute τ = 1/k(T_py+80K) from Arrhenius params
    vol_pool_tau_s: float = 8.0   # [s] FWHM of ignition burst; ignored when vol_pool_tau_auto=True
    vol_pool_t_peak_s: float = 0.0  # [s] time from ignition to burst peak; 0=at ignition
                                     # Set from EXP: argmax(HRRPUA_exp) — e.g. 9.5s at 25 kW/m²
    vol_pool_tau_decay_s: float = -1.0  # [s] right-side exponential decay timescale for burst tail.
                                         # -1 (default) = symmetric Gaussian (backward compatible).
                                         # >0 = asymmetric: Gaussian rise, exp(-t/tau_decay) fall.
                                         # Physical basis: flame-limited pool burn-down rate.
    vol_pool_a_preheat_1_s: float = 0.0        # [1/s] hemicellulose pre-ignition Arrhenius A (Orfão 1999; 0=disabled)
    vol_pool_e_preheat_j_mol: float = 80000.0  # [J/mol] hemicellulose E_a (Orfão 1999 TGA: 76–92 kJ/mol)
    # Lignin slow-release pool — post-processing addend to HRRPUA during main burn phase
    # Lignin comprises ~25-30% of wood by mass (Orfão 1999, NIST). Higher E_a than cellulose;
    # releases volatiles slowly throughout burn. Adds to sustained HRRPUA without affecting peak.
    lig_enable: bool = False                    # [-] enable lignin pool contribution
    lig_m_frac: float = 0.27                   # [-] lignin mass fraction of total fuel (~27% SPF; Orfão 1999)
    lig_a_1_s: float = 0.0                     # [1/s] lignin Arrhenius pre-exponential; 0=disabled
    lig_e_j_mol: float = 200000.0              # [J/mol] lignin activation energy (Orfão 1999 TGA: 160-210 kJ/mol)
    # Heat-of-gasification cap (Tewarson model — no A/E calibration)
    hog_enable: bool = False
    hog_L_eff_J_kg: float | None = None    # [J/kg] effective heat of gasification; ~3-8 MJ/kg for wood
    hog_q_crit_W_m2: float | None = None   # [W/m²] critical flux for pyrolysis onset; ~10-15 kW/m² for wood
    # Thermal penetration cap (flux-independent; uses k/rho/cp — available for future use)
    therm_pen_enable: bool = False
    # Char surface oxidation addend (Vermesi 2020 decomposition framework)
    # HRRPUA_char_ox limited by char pool: accumulates from pyrolysis, depletes from oxidation
    char_ox_enable: bool = False
    char_ox_q_ref_W_m2: float | None = None   # [W/m²] char oxidation ceiling rate; ~100-200 kW/m² for wood
    char_ox_q_stefan0_W_m2: float | None = None  # [W/m²] Stefan-flow suppression flux; ~75-100 kW/m²
    char_ox_char_yield: float = 0.25          # [-] fraction of pyrolyzed mass that becomes char; ~0.20-0.30 for wood
    char_ox_char_hoc_J_kg: float = 32.7e6    # [J/kg] heat of combustion of char (carbon); ~30-33 MJ/kg
    char_ox_m_py_stefan0_kg_m2_s: float | None = None  # [kg/m²/s] if set, Stefan blow uses actual pyrolysis rate: f_stefan = max(1 - m_py/m_py_stefan0, 0). Activates char_ox at end-of-burn when pyrolysis drops → 2nd peak mechanism. If None, falls back to cone-flux formula.
    # Smoldering char oxidation — O₂-diffusion-limited slow glowing combustion (Frandsen 1991)
    # Separate from char_ox: activated after flaming phase when pyrolysis drops below m_py_s0.
    # Physical basis: pine needle / shrub char glows at ~1–8 kW/m² for minutes (Frandsen 1991).
    # Re-ignition hazard after suppression agent application — sustains until char depletes.
    char_smolder_enable: bool = False
    char_smolder_q_ref_W_m2: float | None = None      # [W/m²] O₂-diffusion-limited HRR; 1–8 kW/m² litter/shrub
    char_smolder_char_yield: float | None = None       # [-] if None, falls back to seq_mr_frac0
    char_smolder_hoc_J_kg: float = 32.7e6             # [J/kg] char HoC; NIST carbon combustion
    char_smolder_m_py_s0_kg_m2_s: float | None = None  # [kg/m²/s] blow suppression: smoldering active when m_py < this
    # Backside boundary condition (node-2 for 2-node, node-3 for 3-node)
    back_bc_mode: str = "adiabatic"  # adiabatic | open
    h_open: float = 0.0  # [W/m^2/K] open-back convective HTC
    eps_open: float | None = None  # [-] open-back emissivity; defaults to eps
    # Back-face incident flux + pyrolysis (general, experiment-agnostic)
    # Works with back_bc_mode="open". Covers: cone cal (q_in=0), FDS coupling, furnace.
    back_face_q_in_W_m2: float = 0.0        # [W/m²] incident flux on back face (0=unirradiated)
    back_face_pyrolysis_enable: bool = False  # [-] add back-face pyrolysis HRR contribution
    back_face_node_frac: float = 0.333       # [-] fraction of total fuel mass in node-3 zone
    back_face_T_py_K: float = -1.0          # [K] back-face pyrolysis onset; -1 = use regression_T_py_K
    #   Decouples back-face trigger from front Stefan T_py. Use when T_py is lowered for front-Stefan
    #   pre-ignition charring physics but the back-face secondary peak requires a higher onset temp.
    back_face_hog_enable: bool = False      # [-] use thermal energy-balance (HoG) for back-face rate
    #   When True: m_dot_back = (q_cond_to_back - q_loss_back) / dH_py  (energy-limited, no Arrhenius clamp)
    #   When False (legacy): Arrhenius at T_back clamped to T_py_bp (produces negligible rate)
    #   Use True for thin panels where secondary hump is thermally-driven (back-face breakthrough).
    back_face_hog_min_char_frac: float = 0.0  # [-] min delta_py/L to activate HoG; 0=always on
    #   Delays back-face contribution until significant char depth (cracks not formed in thin char).
    #   Typical: 0.35–0.50. Creates valley between primary and secondary HRRPUA humps.
    back_face_hog_k_crack_frac: float = 0.0  # [-] k_crack for back-face HoG K23 only (decoupled from ODE)
    #   Allows separate tuning: ODE uses fuel_cfg.k_crack_frac (Stefan front rate),
    #   HoG post-processing uses back_face_hog_k_crack_frac (K23_eff enhancement).
    #   If 0.0, falls back to fuel_cfg.k_crack_frac.  Thin panels: 20–60 typical.
    back_face_hog_T_min_K: float = 0.0       # [K] T3 threshold to activate HoG; 0=use legacy delta_py/L trigger
    #   Physical basis: extractive/hemicellulose volatilization onset at back face.
    #   Basswood: ~363 K (90°C) for terpene/phenol extractives (Shafizadeh 1982).
    #   If >0, overrides back_face_hog_min_char_frac. Legacy path used when 0.0.
    back_face_hog_T_ramp_dT_K: float = 20.0  # [K] ramp width for T3-based activation: 0→1 over T_min→T_min+dT
    #   Activation ramp: T_min_K → T_min_K + T_ramp_dT_K.  Default 20 K (90→110°C for Basswood).
    back_face_hog_ramp_width: float = 0.20   # [-] legacy char-front ramp width (only used when T_min_K = 0)
    front_hog_floor_enable: bool = False        # [-] post-processing HoG floor for Stefan front face
    front_hog_floor_L_eff_J_kg: float = 5.5e6  # [J/kg] effective latent heat for floor
    #   Represents distributed pyrolysis zone behind sharp Stefan front (thin-panel effect).
    #   Applies as: hrrpua = max(Stefan, q_in/L_eff × (1-alpha_bar)).
    #   Only active when front_limit_enable=True. Calibrate L_eff to match EXP middle plateau.
    # Per-node burnthrough threshold (fraction) for sequential front-face regression
    # Values > 1.0 (e.g. default 1.1) disable the feature — must be set explicitly per deck
    alpha_burnthrough: float = 1.1  # [-] per-node char fraction for burnthrough; >1.0 = disabled
    # Char (fully-converted) thermal properties for evolving-property model
    k_char: float | None = None  # [W/m/K] char conductivity (typically < k_virgin for wood)
    rho_char: float | None = None  # [kg/m^3] char density (typically ~0.3x virgin)
    cp_char: float | None = None  # [J/kg/K] char specific heat
    # Derived per-node char capacitances and conductances (set by apply_material_geometry)
    C1_char: float | None = None  # [J/m^2/K] surface node char heat capacity
    C2_char: float | None = None  # [J/m^2/K] interior node char heat capacity
    C3_char: float | None = None  # [J/m^2/K] deep node char heat capacity (3-node)
    K12_char: float | None = None  # [W/m^2/K] char inter-node coupling (nodes 1-2)
    K23_char: float | None = None  # [W/m^2/K] char inter-node coupling (nodes 2-3, 3-node)
    C4_char: float | None = None  # [J/m^2/K] node-4 char heat capacity (4/5-node)
    K34_char: float | None = None  # [W/m^2/K] char coupling nodes 3-4 (4/5-node)
    C5_char: float | None = None  # [J/m^2/K] node-5 char heat capacity (5-node)
    K45_char: float | None = None  # [W/m^2/K] char coupling nodes 4-5 (5-node)
    # Staggered coupling passes: 0=disabled, 1=one extra pass, 2=two passes (recommended)
    evolving_props_passes: int = 0
    # Kinetic char state for evolving thermal properties (ODE state extension)
    char_state_mode: str = "none"  # none | kinetic
    A_char: float | None = None    # [1/s] char formation pre-exponential (default: uses A_py/A1_py)
    E_char: float | None = None    # [J/mol] char formation activation energy (default: uses E_py/E1_py)
    # Temperature-dependent conductivity (optional)
    k_temp_mode: str = "constant"  # constant | pmma_piecewise
    k_ref: float | None = None  # [W/m/K] reference conductivity for scaling K12
    K12_ref: float | None = None  # [W/m^2/K] reference inter-node coupling
    # Heat transfer model inputs
    L_m: float = 1.0  # [m] characteristic length
    u_inf_m_s: float = 0.0  # [m/s] free-stream velocity
    convection_mode: str = "auto"  # auto, forced, natural, mixed
    orientation: str = "vertical"  # vertical, horizontal_up, horizontal_down
    C_h_conv: float = 1.0  # [-] calibration multiplier for h_conv
    C_eps: float = 1.0  # [-] calibration multiplier for emissivity
    # Bed collapse model — loose fibrous fuel beds (dry grass, straw in cone holder)
    # When True: inter-node conductances scale as 1/m_frac as fuel burns (nodes become
    # physically closer as bed height h(t) = h₀ × m_frac(t) shrinks). Heat capacity per
    # unit area is unchanged (same mass in compressed space). No free parameters —
    # pure mass conservation. Default False → identical behavior to existing models.
    bed_collapse_enable: bool = False


@dataclass
class EnvConfig:
    """Ambient environment parameters."""

    Tamb: float = 300.0  # [K] ambient temperature
    sigma: float = 5.670374419e-8  # [W/m^2/K^4] Stefan-Boltzmann constant
    T_sur: float | None = None  # [K] surrounding radiative temperature


@dataclass
class SimConfig:
    """Simulation controls."""

    t_end: float = 60.0  # [s] end time
    dt_out: float = 1.0  # [s] uniform output time step for exported signals
    dt_chunk: float = 0.5  # [s] coupling chunk size
    max_step: float | None = None  # [s] maximum solver step size
    rtol: float = 1.0e-6  # [-] solver relative tolerance
    atol: float = 1.0e-8  # [-] solver absolute tolerance
    method: str = "RK45"  # solve_ivp method
    q_in_mode: str = "incident"  # incident | net
    q_inc_ramp_mode: str = "none"  # none | exp | cosine
    q_inc_ramp_tau: float = 1.0  # [s] ramp time constant/window
    warn_on_initial_temp_offset: bool = True  # [-] warn if T1/T2 far from Tamb
    initial_temp_warn_K: float = 50.0  # [K] threshold for initial T offset warning


@dataclass
class Thresholds:
    """Fuel-related thresholds (pyrolysis and flame state machine)."""

    m_py_crit: float = 0.001    # [kg/m²/s] extinction pyrolysis flux
    m_py_ignite: float = 0.005  # [kg/m²/s] ignition pyrolysis flux
    T_py: float = 500.0         # [K] nominal pyrolysis sustain temperature
    T_ignite: float = 600.0     # [K] surface temperature threshold for ignition
    persistence_window: float = 5.0  # [s] extinction persistence window


@dataclass
class OutputConfig:
    """Controls what files and plots are written after a ROM run."""

    base_dir: str = "plots"          # output directory (relative to cwd or absolute)
    case_subdir: bool = False        # create a subdirectory named after the case_id
    # ── PNG plot ──────────────────────────────────────────────────────────────
    png_enable: bool = True          # write a comparison plot PNG
    png_dpi: int = 160               # plot resolution
    # ── CSV time series ───────────────────────────────────────────────────────
    csv_enable: bool = False         # write time-series CSV
    csv_columns: str = "t,hrrpua,mlr,T_surf,T_mid,T_inner,alpha1,alpha2,alpha3"
    # ── JSON scalar metrics ────────────────────────────────────────────────────
    json_metrics_enable: bool = False  # write peak/AUC/sustained metrics as JSON
    # ── Console output ────────────────────────────────────────────────────────
    metrics_console: bool = True     # print metrics table to stdout
    # ── EXP overlay ───────────────────────────────────────────────────────────
    exp_csv_path: str = ""           # path to experimental CSV for overlay (empty = no overlay)
