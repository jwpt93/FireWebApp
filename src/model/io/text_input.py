from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class RomInputs:
    """Parsed user inputs from a text file."""

    q_in_schedule: Optional[List[Tuple[float, float]]] = None  # (t_s, q_W_m2)
    q_in_constant: Optional[float] = None  # W/m^2
    q_in_units: str = "W/m2"
    q_in_constant_key: Optional[str] = None
    q_in_constant_altkey_raw: Optional[float] = None
    hold_last: bool = True
    preburn_enable: bool = False
    preburn_start_s: Optional[float] = None
    preburn_end_s: Optional[float] = None
    preburn_q_in: Optional[float] = None
    preburn_units: str = "W/m2"
    pyrolysis_mode: Optional[str] = None
    m_py_schedule: Optional[List[Tuple[float, float]]] = None
    m_py_units: str = "kg/m2/s"

    T1: Optional[float] = None
    T2: Optional[float] = None
    T3: Optional[float] = None
    M1: Optional[float] = None
    Tamb: Optional[float] = None
    T_sur: Optional[float] = None
    t_end: Optional[float] = None
    method: Optional[str] = None
    q_in_mode: Optional[str] = None
    q_inc_ramp_mode: Optional[str] = None
    q_inc_ramp_tau: Optional[float] = None
    dt_out: Optional[float] = None

    # Geometry / size
    area_m2: Optional[float] = None
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    thickness_m: Optional[float] = None
    node1_frac: Optional[float] = None
    node2_frac: Optional[float] = None
    node3_frac: Optional[float] = None
    node4_frac: Optional[float] = None
    node5_frac: Optional[float] = None

    # Material properties (prefer lookup, allow overrides)
    material_name: Optional[str] = None
    density: Optional[float] = None
    cp: Optional[float] = None
    k: Optional[float] = None
    thermal_model_order: Optional[int] = None
    C3: Optional[float] = None
    K23: Optional[float] = None
    eps: Optional[float] = None
    k_temp_mode: Optional[str] = None
    dH_py: Optional[float] = None
    back_bc_mode: Optional[str] = None
    h_open: Optional[float] = None
    eps_open: Optional[float] = None
    back_face_q_in_W_m2: Optional[float] = None
    back_face_pyrolysis_enable: Optional[bool] = None
    back_face_node_frac: Optional[float] = None
    back_face_T_py_K: Optional[float] = None
    back_face_hog_enable: Optional[bool] = None
    back_face_hog_min_char_frac: Optional[float] = None
    back_face_hog_k_crack_frac: Optional[float] = None
    back_face_hog_T_min_K: Optional[float] = None
    back_face_hog_T_ramp_dT_K: Optional[float] = None
    back_face_hog_ramp_width: Optional[float] = None
    front_hog_floor_enable: Optional[bool] = None
    front_hog_floor_L_eff_J_kg: Optional[float] = None
    q_loss3: Optional[float] = None
    A_py: Optional[float] = None
    E_py: Optional[float] = None
    alpha_moist: Optional[float] = None
    alpha_burnthrough: Optional[float] = None
    kinetics_mode: Optional[str] = None
    sigmoid_T0_K: Optional[float] = None
    sigmoid_dT_K: Optional[float] = None
    A1_py: Optional[float] = None
    E1_py: Optional[float] = None
    A2_py: Optional[float] = None
    E2_py: Optional[float] = None
    A3_py: Optional[float] = None
    E3_py: Optional[float] = None
    seq_y1_vol: Optional[float] = None
    seq_y2_vol: Optional[float] = None
    seq_f12_to_m2: Optional[float] = None
    seq_secondary_char_enable: Optional[bool] = None
    seq_m1_frac: Optional[float] = None
    seq_m2_frac0: Optional[float] = None
    seq_mr_frac0: Optional[float] = None
    seq_clamp_yields: Optional[bool] = None
    seq_T_ign_K: Optional[float] = None
    seq_vol_interp_n: Optional[float] = None
    seq_pool2_use_back_node: Optional[bool] = None
    sg_n1: Optional[float] = None
    sg_n2: Optional[float] = None
    sg_n3: Optional[float] = None
    sg_y_g1: Optional[float] = None
    sg_y_i1: Optional[float] = None
    sg_y_c1: Optional[float] = None
    sg_y_g2: Optional[float] = None
    sg_y_c2: Optional[float] = None
    sg_clamp_yields: Optional[bool] = None
    reactive_access_mode: Optional[str] = None
    access_reduction_beta: Optional[float] = None
    access_min: Optional[float] = None
    access_driver: Optional[str] = None
    access_target: Optional[str] = None
    access_depth_weight_pow: Optional[float] = None
    chem_depth_partition_mode: Optional[str] = None
    M1_represents: Optional[str] = None
    pyrolysis_mass_source: Optional[str] = None
    m_fuel_total_kg_m2: Optional[float] = None
    A_basis: Optional[str] = None
    front_limit_enable: Optional[bool] = None
    front_limit_surface_only: Optional[bool] = None
    front_model_mode: Optional[str] = None
    regression_alpha: Optional[float] = None
    regression_T_py_K: Optional[float] = None
    hog_enable: Optional[bool] = None
    hog_L_eff_J_kg: Optional[float] = None
    hog_q_crit_W_m2: Optional[float] = None
    therm_pen_enable: Optional[bool] = None
    char_ox_enable: Optional[bool] = None
    char_ox_q_ref_W_m2: Optional[float] = None
    char_ox_q_stefan0_W_m2: Optional[float] = None
    char_ox_char_yield: Optional[float] = None
    char_ox_char_hoc_J_kg: Optional[float] = None
    char_ox_m_py_stefan0_kg_m2_s: Optional[float] = None
    char_smolder_enable: Optional[bool] = None
    char_smolder_q_ref_W_m2: Optional[float] = None
    char_smolder_char_yield: Optional[float] = None
    char_smolder_hoc_J_kg: Optional[float] = None
    char_smolder_m_py_s0_kg_m2_s: Optional[float] = None
    regression_delta_min_m: Optional[float] = None
    regression_delta_cap_m: Optional[float] = None
    regression_spall_onset_frac: Optional[float] = None
    regression_spall_reduction_frac: Optional[float] = None
    softmin_beta: Optional[float] = None
    handoff_start_frac: Optional[float] = None
    handoff_end_frac: Optional[float] = None
    delta_py0_m: Optional[float] = None
    m_char0_kg_m2: Optional[float] = None
    regression_L0_m: Optional[float] = None
    rho_solid: Optional[float] = None
    k_evap0: Optional[float] = None
    T_evap_onset: Optional[float] = None
    # Char (fully-converted) thermal properties for evolving-property model
    k_char: Optional[float] = None
    rho_char: Optional[float] = None
    cp_char: Optional[float] = None
    evolving_props_passes: int = 0
    char_state_mode: Optional[str] = None
    A_char: Optional[float] = None
    E_char: Optional[float] = None

    fuel_overrides: Dict[str, Any] = field(default_factory=dict)
    sim_overrides: Dict[str, Any] = field(default_factory=dict)
    # output control — parsed from output.* keys in deck
    output_overrides: Dict[str, Any] = field(default_factory=dict)
    force_htc_zero: bool = False
    hrr_from_mlr: bool = False
    hoc_units: str = "kJ/kg"
    hoc_eff: Optional[float] = None
    hoc_eff_J_kg: Optional[float] = None

    # ── MoL (method-of-lines) 1D pyrolysis solver settings ──────────────────
    # Enable with:  mol.enable = true
    # Default N=20 cells is accurate for L ≥ 5mm with Radau solver.
    mol_enable: bool = False
    mol_n_cells: int = 20       # number of spatial cells
    mol_n_species: int = 1      # 1 = single Arrhenius; 2 = two-species (uses A1/E1 + A2/E2)
    mol_n_reactions: int = 0    # number of reactions (0 = infer from n_species)
    mol_h_conv: Optional[float] = None   # surface convective HTC [W/m²/K]; default 15.0
    mol_back_bc: Optional[str] = None    # "adiabatic" (default) or "open"
    mol_k_crack_frac: Optional[float] = None  # char cracking conductivity factor (Shi & Chew 2023)
    mol_grid_stretch: float = 1.0             # geometric grid-stretch ratio (1.0 = uniform)
    mol_char_ox_enable: bool = False          # enable char oxidation post-processing
    mol_char_ox_q_ref_W_m2: float = 70000.0  # max char ox heat release [W/m²]
    mol_char_ox_m_py_stefan0: float = 0.010  # blow suppression threshold [kg/m²/s]
    # Surface recession (non-charring polymers, e.g. PMMA)
    mol_surface_recession_enable: bool = False
    mol_surface_rho_floor: float = 0.01      # delete cell when rho_total < floor × rho0_sum
    # Char spalling (phenomenological char detachment for charring wood)
    mol_spall_enable: bool = False
    mol_spall_depth_m: float = 0.003         # char conversion depth threshold [m]
    mol_spall_rho_floor: float = 0.05        # residual char fraction after detachment
    mol_cp_gas: float = 0.0                  # volatile gas cp [J/kg/K] for gas convection; 0 = disabled
    mol_in_depth_rad_kappa: float = 0.0      # Beer-Lambert extinction coeff [m⁻¹]; 0 = surface-opaque
    mol_in_depth_rad_density_weighted: bool = False  # density-weighted κ (T9b); False = fixed κ
    mol_surface_y_o2: float = 0.21           # surface O2 mass fraction for oxidative rxns (nO2>0); 0.21=air
    mol_material_coords: bool = False        # material-coordinate moving mesh (PMMA recession fix)
    mol_lagrangian_mode: bool = False        # surface-cell Lagrangian replenishment (PMMA ablation BC)
    mol_charring_front_bc: bool = False      # Stefan front BC for charring materials (wood, PB)
    mol_charring_T_py: float = 600.0         # charring front temperature [K]; Drysdale 2011
    mol_surface_ablation_bc: bool = False    # energy-balance ablation BC for non-charring receding polymers
    mol_surface_ablation_L_py: float = 3.5e6  # effective heat of gasification [J/kg]
    mol_surface_ablation_T_min: float = 600.0  # ablation activation temperature [K]
    # Explicit multi-species specification (Lautenberger 2009 deck format)
    # mol.species.N.{name,k0,nk,rho0,cp0,nc,eps,gamma}
    # mol.reaction.N.{from,to,Z,E,n,dH_vol,nu_gas,dH_sol,nO2}
    mol_species_list: List[Dict[str, Any]] = field(default_factory=list)
    mol_reactions_list: List[Dict[str, Any]] = field(default_factory=list)

    # ── Outdoor fire / brush-fire extension ─────────────────────────────────
    # Enable with:  outdoor.wind_speed_m_s = 3.0  (etc.)
    # See model_outdoor/config.py for OutdoorEnvConfig field definitions.
    # References: Rothermel (1972), Anderson (1982), Nelson (2000)
    outdoor_overrides: Dict[str, Any] = field(default_factory=dict)

    # ── Spray suppression device ─────────────────────────────────────────────
    # Enable with:  spray.enable = true
    # See model_outdoor/config.py for SprayConfig field definitions.
    # References: Rasbash (1962), Johansson et al. (2018)
    spray_overrides: Dict[str, Any] = field(default_factory=dict)


def _parse_bool(text: str) -> Optional[bool]:
    val = text.strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_float(text: str) -> Optional[float]:
    val = text.strip().lower()
    if val in {"none", "null", "nan", ""}:
        return None
    return _parse_float(text)


def normalize_hoc_units(units: Optional[str]) -> str:
    token = (units or "kJ/kg").strip().lower().replace(" ", "")
    token = token.replace("_", "").replace("-", "")
    if token in {"j/kg", "jkg", "jperkg", "joule/kg", "joules/kg"}:
        return "J/kg"
    if token in {"kj/kg", "kjkg", "kjperkg", "kilojoule/kg", "kilojoules/kg"}:
        return "kJ/kg"
    return "kJ/kg"


def hoc_eff_to_j_per_kg(hoc_eff: Optional[float], hoc_units: Optional[str] = "kJ/kg") -> Optional[float]:
    if hoc_eff is None:
        return None
    try:
        hoc_val = float(hoc_eff)
    except Exception:
        return None
    if not math.isfinite(hoc_val):
        return None
    if normalize_hoc_units(hoc_units) == "J/kg":
        return hoc_val
    return hoc_val * 1000.0


def hoc_eff_to_kj_per_kg(hoc_eff: Optional[float], hoc_units: Optional[str] = "kJ/kg") -> Optional[float]:
    hoc_j = hoc_eff_to_j_per_kg(hoc_eff, hoc_units)
    if hoc_j is None:
        return None
    return hoc_j / 1000.0


def _parse_schedule(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        parts = [p.strip() for p in chunk.replace(",", " ").split()]
        if len(parts) < 2:
            continue
        t = _parse_float(parts[0])
        q = _parse_float(parts[1])
        if t is None or q is None:
            continue
        pairs.append((t, q))
    return pairs


# ── Deck parser ───────────────────────────────────────────────────────────────
# load_text_input() reads a plain key=value text file and populates RomInputs.
# Keys are dispatched by prefix:
#   (no prefix)       → top-level fields (t_end, Tamb, q_in_*, hoc_eff, …)
#   geometry.*        → sample geometry (area_m2, thickness_m, node*_frac)
#   material.*        → material/kinetics properties → FuelConfig fields
#   fuel.*            → FuelConfig overrides (flame, char_ox, vol_pool, …)
#   sim.*             → SimConfig overrides (max_step, method, dt_out)
#   output.*          → OutputConfig overrides (png_enable, csv_enable, …)
#   back_face.*       → back-face BC and pyrolysis settings
# Comments start with # or ;

def load_text_input(path: Path) -> RomInputs:
    inputs = RomInputs()
    _mol_species_tmp: Dict[int, Dict[str, Any]] = {}
    _mol_reactions_tmp: Dict[int, Dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, value = [s.strip() for s in line.split("=", 1)]
        # Strip inline comments (e.g. "true  # explanation")
        if "#" in value:
            value = value[:value.index("#")].strip()
        key_l = key.lower()
        if "sigma" in key_l:
            # Sigma is a physical constant; ignore if provided.
            continue

        if key_l in {"q_in_schedule", "forcing.q_in_schedule"}:
            inputs.q_in_schedule = _parse_schedule(value)
            continue
        if key_l in {"q_in", "q_in_constant", "forcing.q_in"}:
            q_const = _parse_float(value)
            inputs.q_in_constant = q_const
            inputs.q_in_constant_key = key_l
            if key_l != "q_in_constant":
                inputs.q_in_constant_altkey_raw = q_const
            else:
                inputs.q_in_constant_altkey_raw = None
            continue
        if key_l in {"q_in_units", "forcing.q_in_units"}:
            inputs.q_in_units = value
            continue
        if key_l in {"preburn.enable", "preburn_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.preburn_enable = b
            continue
        if key_l in {"preburn.start_s", "preburn_start_s", "preburn.start"}:
            inputs.preburn_start_s = _parse_float(value)
            continue
        if key_l in {"preburn.end_s", "preburn_end_s", "preburn.end"}:
            inputs.preburn_end_s = _parse_float(value)
            continue
        if key_l in {"preburn.q_in", "preburn_q_in"}:
            inputs.preburn_q_in = _parse_float(value)
            continue
        if key_l in {"preburn.q_in_units", "preburn_units"}:
            inputs.preburn_units = value
            continue
        if key_l in {"pyrolysis.mode", "pyrolysis_mode"}:
            inputs.pyrolysis_mode = value.strip().lower()
            continue
        if key_l in {"pyrolysis.schedule", "pyrolysis.m_py_schedule", "m_py_schedule"}:
            inputs.m_py_schedule = _parse_schedule(value)
            continue
        if key_l in {"pyrolysis.units", "pyrolysis.m_py_units", "m_py_units"}:
            inputs.m_py_units = value
            continue
        if key_l in {"force_htc_zero", "force_htc0", "htc_zero"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.force_htc_zero = b
            continue
        if key_l in {"hrr_from_mlr", "hrr_from_mass_loss", "diagnostic_hrr"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.hrr_from_mlr = b
            continue
        if key_l in {"hoc_units", "material.hoc_units", "hoc_unit"}:
            inputs.hoc_units = normalize_hoc_units(value)
            continue
        if key_l in {"hoc_eff", "material.hoc", "hoc"}:
            inputs.hoc_eff = _parse_float(value)
            continue
        if key_l in {"hoc_eff_j_kg", "material.hoc_eff_j_kg", "hoc_j_kg"}:
            hoc_j = _parse_float(value)
            if hoc_j is not None:
                inputs.hoc_units = "J/kg"
                inputs.hoc_eff = hoc_j
                inputs.hoc_eff_J_kg = hoc_j
            continue

        if key_l in {"t1", "init.t1", "initial.t1"}:
            inputs.T1 = _parse_float(value)
            continue
        if key_l in {"t2", "init.t2", "initial.t2"}:
            inputs.T2 = _parse_float(value)
            continue
        if key_l in {"t3", "init.t3", "initial.t3"}:
            inputs.T3 = _parse_float(value)
            continue
        if key_l in {"m1", "init.m1", "initial.m1"}:
            inputs.M1 = _parse_float(value)
            continue
        if key_l in {"tamb", "env.tamb"}:
            inputs.Tamb = _parse_float(value)
            continue
        if key_l in {"t_sur", "env.t_sur"}:
            inputs.T_sur = _parse_float(value)
            continue
        if key_l in {"t_end", "sim.t_end"}:
            inputs.t_end = _parse_float(value)
            continue
        if key_l in {"method", "sim.method"}:
            inputs.method = value.strip()
            if inputs.method:
                inputs.sim_overrides["method"] = inputs.method
            continue
        if key_l in {"q_in_mode", "sim.q_in_mode", "forcing.q_in_mode"}:
            mode = value.strip().lower()
            inputs.q_in_mode = mode
            inputs.sim_overrides["q_in_mode"] = mode
            continue
        if key_l in {"q_inc_ramp_mode", "sim.q_inc_ramp_mode", "forcing.q_inc_ramp_mode"}:
            mode = value.strip().lower()
            inputs.q_inc_ramp_mode = mode
            inputs.sim_overrides["q_inc_ramp_mode"] = mode
            continue
        if key_l in {"q_inc_ramp_tau", "sim.q_inc_ramp_tau", "forcing.q_inc_ramp_tau"}:
            tau = _parse_float(value)
            inputs.q_inc_ramp_tau = tau
            if tau is not None:
                inputs.sim_overrides["q_inc_ramp_tau"] = tau
            continue
        if key_l in {"dt_out", "sim.dt_out"}:
            val = _parse_float(value)
            inputs.dt_out = val
            if val is not None:
                inputs.sim_overrides["dt_out"] = val
            continue
        if key_l in {"max_step", "max_dt", "sim.max_step"}:
            val = _parse_float(value)
            if val is not None:
                inputs.sim_overrides["max_step"] = val
            continue

        if key_l in {"geometry.area_m2", "area_m2"}:
            inputs.area_m2 = _parse_float(value)
            continue
        if key_l in {"geometry.length_m", "length_m"}:
            inputs.length_m = _parse_float(value)
            continue
        if key_l in {"geometry.width_m", "width_m"}:
            inputs.width_m = _parse_float(value)
            continue
        if key_l in {"geometry.thickness_m", "thickness_m"}:
            inputs.thickness_m = _parse_float(value)
            continue
        if key_l in {"geometry.node1_frac", "node1_frac"}:
            inputs.node1_frac = _parse_float(value)
            continue
        if key_l in {"geometry.node2_frac", "node2_frac"}:
            inputs.node2_frac = _parse_float(value)
            continue
        if key_l in {"geometry.node3_frac", "node3_frac"}:
            inputs.node3_frac = _parse_float(value)
            continue
        if key_l in {"geometry.node4_frac", "node4_frac"}:
            inputs.node4_frac = _parse_float(value)
            continue
        if key_l in {"geometry.node5_frac", "node5_frac"}:
            inputs.node5_frac = _parse_float(value)
            continue

        if key_l in {"material.name", "material_name"}:
            inputs.material_name = value
            continue
        if key_l in {"material.density", "density"}:
            inputs.density = _parse_float(value)
            continue
        if key_l in {"material.cp", "cp"}:
            inputs.cp = _parse_float(value)
            continue
        if key_l in {"material.k", "k"}:
            inputs.k = _parse_float(value)
            continue
        if key_l in {"material.k_char", "k_char"}:
            inputs.k_char = _parse_float(value)
            continue
        if key_l in {"material.rho_char", "rho_char"}:
            inputs.rho_char = _parse_float(value)
            continue
        if key_l in {"material.cp_char", "cp_char"}:
            inputs.cp_char = _parse_float(value)
            continue
        if key_l in {"simulation.evolving_props_passes", "evolving_props_passes"}:
            ival = _parse_float(value)
            if ival is not None:
                inputs.evolving_props_passes = int(ival)
            continue
        if key_l in {"material.char_state_mode", "char_state_mode"}:
            inputs.char_state_mode = value.strip().lower()
            continue
        if key_l in {"material.a_char", "a_char"}:
            inputs.A_char = _parse_float(value)
            continue
        if key_l in {"material.e_char", "e_char"}:
            inputs.E_char = _parse_float(value)
            continue
        if key_l in {"material.thermal_model_order", "thermal_model_order"}:
            ival = _parse_float(value)
            if ival is not None:
                inputs.thermal_model_order = int(ival)
            continue
        if key_l in {"material.c3", "c3"}:
            inputs.C3 = _parse_optional_float(value)
            continue
        if key_l in {"material.k23", "k23"}:
            inputs.K23 = _parse_optional_float(value)
            continue
        if key_l in {"material.k_temp_mode", "k_temp_mode"}:
            inputs.k_temp_mode = value.strip().lower()
            continue
        if key_l in {"material.eps", "eps"}:
            inputs.eps = _parse_float(value)
            continue
        if key_l in {"material.dh_py", "dh_py"}:
            inputs.dH_py = _parse_float(value)
            continue
        if key_l in {"material.back_bc_mode", "back_bc_mode"}:
            inputs.back_bc_mode = value.strip().lower()
            continue
        if key_l in {"material.h_open", "h_open"}:
            inputs.h_open = _parse_float(value)
            continue
        if key_l in {"material.eps_open", "eps_open"}:
            inputs.eps_open = _parse_optional_float(value)
            continue
        if key_l in {"back_face.q_in_w_m2", "back_face_q_in_w_m2",
                     "back_face.q_in_W_m2", "back_face_q_in_W_m2"}:
            inputs.back_face_q_in_W_m2 = _parse_float(value)
            continue
        if key_l in {"back_face.pyrolysis", "back_face_pyrolysis",
                     "back_face.pyrolysis_enable", "back_face_pyrolysis_enable"}:
            inputs.back_face_pyrolysis_enable = value.strip().lower() in {"true", "1", "yes"}
            continue
        if key_l in {"back_face.node_frac", "back_face_node_frac"}:
            inputs.back_face_node_frac = _parse_float(value)
            continue
        if key_l in {"back_face.t_py_k", "back_face_t_py_k"}:
            inputs.back_face_T_py_K = _parse_float(value)
            continue
        if key_l in {"back_face.hog_enable", "back_face_hog_enable",
                     "material.back_face_hog_enable"}:
            inputs.back_face_hog_enable = value.strip().lower() in {"true", "1", "yes"}
            continue
        if key_l in {"back_face.hog_min_char_frac", "back_face_hog_min_char_frac",
                     "material.back_face_hog_min_char_frac"}:
            inputs.back_face_hog_min_char_frac = _parse_float(value)
            continue
        if key_l in {"back_face.hog_k_crack_frac", "back_face_hog_k_crack_frac",
                     "material.back_face_hog_k_crack_frac"}:
            inputs.back_face_hog_k_crack_frac = _parse_float(value)
            continue
        if key_l in {"back_face.hog_t_min_k", "back_face_hog_t_min_k",
                     "material.back_face_hog_t_min_k"}:
            inputs.back_face_hog_T_min_K = _parse_float(value)
            continue
        if key_l in {"back_face.hog_t_ramp_dt_k", "back_face_hog_t_ramp_dt_k",
                     "material.back_face_hog_t_ramp_dt_k"}:
            inputs.back_face_hog_T_ramp_dT_K = _parse_float(value)
            continue
        if key_l in {"back_face.hog_ramp_width", "back_face_hog_ramp_width",
                     "material.back_face_hog_ramp_width"}:
            inputs.back_face_hog_ramp_width = _parse_float(value)
            continue
        if key_l in {"front_hog_floor_enable", "material.front_hog_floor_enable"}:
            inputs.front_hog_floor_enable = value.strip().lower() in {"true", "1", "yes"}
            continue
        if key_l in {"front_hog_floor_l_eff_j_kg", "material.front_hog_floor_l_eff_j_kg"}:
            inputs.front_hog_floor_L_eff_J_kg = _parse_float(value)
            continue
        if key_l in {"material.q_loss3", "q_loss3"}:
            inputs.q_loss3 = _parse_optional_float(value)
            continue
        if key_l in {"material.a_py", "a_py"}:
            inputs.A_py = _parse_float(value)
            continue
        if key_l in {"material.e_py", "e_py"}:
            inputs.E_py = _parse_float(value)
            continue
        if key_l in {"material.alpha_moist", "alpha_moist"}:
            inputs.alpha_moist = _parse_float(value)
            continue
        if key_l in {"material.alpha_burnthrough", "alpha_burnthrough"}:
            inputs.alpha_burnthrough = _parse_float(value)
            continue
        if key_l in {"material.kinetics_mode", "kinetics_mode"}:
            inputs.kinetics_mode = value.strip().lower()
            continue
        if key_l in {"material.sigmoid_t0_k", "sigmoid_t0_k"}:
            inputs.sigmoid_T0_K = _parse_float(value)
            continue
        if key_l in {"material.sigmoid_dt_k", "sigmoid_dt_k"}:
            inputs.sigmoid_dT_K = _parse_float(value)
            continue
        if key_l in {"material.a1_py", "a1_py"}:
            inputs.A1_py = _parse_optional_float(value)
            continue
        if key_l in {"material.e1_py", "e1_py"}:
            inputs.E1_py = _parse_optional_float(value)
            continue
        if key_l in {"material.a2_py", "a2_py"}:
            inputs.A2_py = _parse_optional_float(value)
            continue
        if key_l in {"material.e2_py", "e2_py"}:
            inputs.E2_py = _parse_optional_float(value)
            continue
        if key_l in {"material.a3_py", "a3_py"}:
            inputs.A3_py = _parse_optional_float(value)
            continue
        if key_l in {"material.e3_py", "e3_py"}:
            inputs.E3_py = _parse_optional_float(value)
            continue
        if key_l in {"material.seq_y1_vol", "seq_y1_vol"}:
            inputs.seq_y1_vol = _parse_float(value)
            continue
        if key_l in {"material.seq_y2_vol", "seq_y2_vol"}:
            inputs.seq_y2_vol = _parse_float(value)
            continue
        if key_l in {"material.seq_t_ign_k", "seq_t_ign_k"}:
            inputs.seq_T_ign_K = _parse_float(value)
            continue
        if key_l in {"material.seq_vol_interp_n", "seq_vol_interp_n"}:
            inputs.seq_vol_interp_n = _parse_float(value)
            continue
        if key_l in {"material.seq_f12_to_m2", "seq_f12_to_m2"}:
            inputs.seq_f12_to_m2 = _parse_float(value)
            continue
        if key_l in {"material.seq_secondary_char_enable", "seq_secondary_char_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.seq_secondary_char_enable = b
            continue
        if key_l in {"material.seq_m1_frac", "seq_m1_frac"}:
            inputs.seq_m1_frac = _parse_float(value)
            continue
        if key_l in {"material.seq_m2_frac0", "seq_m2_frac0"}:
            inputs.seq_m2_frac0 = _parse_float(value)
            continue
        if key_l in {"material.seq_mr_frac0", "seq_mr_frac0"}:
            inputs.seq_mr_frac0 = _parse_float(value)
            continue
        if key_l in {"material.seq_clamp_yields", "seq_clamp_yields"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.seq_clamp_yields = b
            continue
        if key_l in {"material.seq_pool2_use_back_node", "seq_pool2_use_back_node"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.seq_pool2_use_back_node = b
            continue
        if key_l in {"material.sg_n1", "sg_n1"}:
            inputs.sg_n1 = _parse_float(value)
            continue
        if key_l in {"material.sg_n2", "sg_n2"}:
            inputs.sg_n2 = _parse_float(value)
            continue
        if key_l in {"material.sg_n3", "sg_n3"}:
            inputs.sg_n3 = _parse_float(value)
            continue
        if key_l in {"material.sg_y_g1", "sg_y_g1"}:
            inputs.sg_y_g1 = _parse_float(value)
            continue
        if key_l in {"material.sg_y_i1", "sg_y_i1"}:
            inputs.sg_y_i1 = _parse_float(value)
            continue
        if key_l in {"material.sg_y_c1", "sg_y_c1"}:
            inputs.sg_y_c1 = _parse_float(value)
            continue
        if key_l in {"material.sg_y_g2", "sg_y_g2"}:
            inputs.sg_y_g2 = _parse_float(value)
            continue
        if key_l in {"material.sg_y_c2", "sg_y_c2"}:
            inputs.sg_y_c2 = _parse_float(value)
            continue
        if key_l in {"material.sg_clamp_yields", "sg_clamp_yields"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.sg_clamp_yields = b
            continue
        if key_l in {"material.reactive_access_mode", "reactive_access_mode"}:
            inputs.reactive_access_mode = value.strip().lower()
            continue
        if key_l in {"material.access_reduction_beta", "access_reduction_beta"}:
            inputs.access_reduction_beta = _parse_float(value)
            continue
        if key_l in {"material.access_min", "access_min"}:
            inputs.access_min = _parse_float(value)
            continue
        if key_l in {"material.access_driver", "access_driver"}:
            inputs.access_driver = value.strip().lower()
            continue
        if key_l in {"material.access_target", "access_target"}:
            inputs.access_target = value.strip().lower()
            continue
        if key_l in {"material.access_depth_weight_pow", "access_depth_weight_pow"}:
            inputs.access_depth_weight_pow = _parse_float(value)
            continue
        if key_l in {"material.chem_depth_partition_mode", "chem_depth_partition_mode"}:
            inputs.chem_depth_partition_mode = value.strip().lower()
            continue
        if key_l in {"material.m1_represents", "m1_represents"}:
            inputs.M1_represents = value.strip().lower()
            continue
        if key_l in {"material.pyrolysis_mass_source", "pyrolysis_mass_source"}:
            inputs.pyrolysis_mass_source = value.strip()
            continue
        if key_l in {"material.m_fuel_total_kg_m2", "m_fuel_total_kg_m2"}:
            inputs.m_fuel_total_kg_m2 = _parse_optional_float(value)
            continue
        if key_l in {"material.a_basis", "a_basis", "material.pyrolysis_rate_basis", "pyrolysis_rate_basis"}:
            inputs.A_basis = value.strip().lower()
            continue
        if key_l in {"material.front_limit_enable", "front_limit_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.front_limit_enable = b
            continue
        if key_l in {"material.front_limit_surface_only", "front_limit_surface_only"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.front_limit_surface_only = b
            continue
        if key_l in {"material.front_model_mode", "front_model_mode"}:
            mode = value.strip().lower()
            inputs.front_model_mode = mode
            if mode in {"on", "enabled", "enable", "true", "1", "yes", "y"}:
                inputs.front_limit_enable = True
            elif mode in {"off", "disabled", "disable", "false", "0", "no", "n"}:
                inputs.front_limit_enable = False
            continue
        if key_l in {"material.regression_alpha", "regression_alpha"}:
            inputs.regression_alpha = _parse_float(value)
            continue
        if key_l in {"material.regression_t_py_k", "regression_t_py_k", "material.regression_t_py", "regression_t_py"}:
            inputs.regression_T_py_K = _parse_float(value)
            continue
        if key_l in {"material.hog_enable", "hog_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.hog_enable = b
            continue
        if key_l in {"material.hog_l_eff_j_kg", "hog_l_eff_j_kg", "material.hog_l_eff", "hog_l_eff"}:
            inputs.hog_L_eff_J_kg = _parse_float(value)
            continue
        if key_l in {"material.hog_q_crit_w_m2", "hog_q_crit_w_m2", "material.hog_q_crit", "hog_q_crit"}:
            inputs.hog_q_crit_W_m2 = _parse_float(value)
            continue
        if key_l in {"material.therm_pen_enable", "therm_pen_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.therm_pen_enable = b
            continue
        if key_l in {"material.char_ox_enable", "char_ox_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.char_ox_enable = b
            continue
        if key_l in {"material.char_ox_q_ref_w_m2", "char_ox_q_ref_w_m2", "material.char_ox_q_ref", "char_ox_q_ref"}:
            inputs.char_ox_q_ref_W_m2 = _parse_float(value)
            continue
        if key_l in {"material.char_ox_q_stefan0_w_m2", "char_ox_q_stefan0_w_m2", "material.char_ox_q_stefan0", "char_ox_q_stefan0"}:
            inputs.char_ox_q_stefan0_W_m2 = _parse_float(value)
            continue
        if key_l in {"material.char_ox_char_yield", "char_ox_char_yield"}:
            inputs.char_ox_char_yield = _parse_float(value)
            continue
        if key_l in {"material.char_ox_char_hoc_j_kg", "char_ox_char_hoc_j_kg", "material.char_ox_char_hoc", "char_ox_char_hoc"}:
            inputs.char_ox_char_hoc_J_kg = _parse_float(value)
            continue
        if key_l in {"material.char_ox_m_py_stefan0_kg_m2_s", "char_ox_m_py_stefan0_kg_m2_s", "char_ox_m_py_stefan0"}:
            inputs.char_ox_m_py_stefan0_kg_m2_s = _parse_float(value)
            continue
        if key_l in {"material.char_smolder_enable", "char_smolder_enable"}:
            b = _parse_bool(value)
            if b is not None:
                inputs.char_smolder_enable = b
            continue
        if key_l in {"material.char_smolder_q_ref_w_m2", "char_smolder_q_ref_w_m2", "material.char_smolder_q_ref", "char_smolder_q_ref"}:
            inputs.char_smolder_q_ref_W_m2 = _parse_float(value)
            continue
        if key_l in {"material.char_smolder_char_yield", "char_smolder_char_yield"}:
            inputs.char_smolder_char_yield = _parse_float(value)
            continue
        if key_l in {"material.char_smolder_hoc_j_kg", "char_smolder_hoc_j_kg", "material.char_smolder_hoc", "char_smolder_hoc"}:
            inputs.char_smolder_hoc_J_kg = _parse_float(value)
            continue
        if key_l in {"material.char_smolder_m_py_s0_kg_m2_s", "char_smolder_m_py_s0_kg_m2_s", "char_smolder_m_py_s0"}:
            inputs.char_smolder_m_py_s0_kg_m2_s = _parse_float(value)
            continue
        if key_l in {"material.regression_delta_min_m", "regression_delta_min_m", "delta_min_m"}:
            inputs.regression_delta_min_m = _parse_float(value)
            continue
        if key_l in {"material.regression_delta_cap_m", "regression_delta_cap_m", "delta_cap_m"}:
            inputs.regression_delta_cap_m = _parse_float(value)
            continue
        if key_l in {"material.regression_spall_onset_frac", "regression_spall_onset_frac", "spall_onset_frac"}:
            inputs.regression_spall_onset_frac = _parse_float(value)
            continue
        if key_l in {"material.regression_spall_reduction_frac", "regression_spall_reduction_frac", "spall_reduction_frac"}:
            inputs.regression_spall_reduction_frac = _parse_float(value)
            continue
        if key_l in {"material.softmin_beta", "softmin_beta"}:
            inputs.softmin_beta = _parse_float(value)
            continue
        if key_l in {"material.handoff_start_frac", "handoff_start_frac"}:
            inputs.handoff_start_frac = _parse_float(value)
            continue
        if key_l in {"material.handoff_end_frac", "handoff_end_frac"}:
            inputs.handoff_end_frac = _parse_float(value)
            continue
        if key_l in {"material.delta_py0_m", "delta_py0_m"}:
            inputs.delta_py0_m = _parse_float(value)
            continue
        if key_l in {"material.m_char0_kg_m2", "m_char0_kg_m2"}:
            inputs.m_char0_kg_m2 = _parse_float(value)
            continue
        if key_l in {"material.regression_l0_m", "regression_l0_m"}:
            inputs.regression_L0_m = _parse_float(value)
            continue
        if key_l in {"material.rho_solid", "rho_solid"}:
            inputs.rho_solid = _parse_float(value)
            continue
        if key_l in {"material.k_evap0", "k_evap0"}:
            inputs.k_evap0 = _parse_float(value)
            continue
        if key_l in {"material.t_evap_onset", "t_evap_onset"}:
            inputs.T_evap_onset = _parse_float(value)
            continue

        if key_l.startswith("fuel."):
            field = key_l.split(".", 1)[1]
            b = _parse_bool(value)
            if b is not None:
                inputs.fuel_overrides[field] = b
                continue
            val_opt = _parse_optional_float(value)
            if val_opt is not None or value.strip().lower() in {"none", "null", "nan", ""}:
                inputs.fuel_overrides[field] = val_opt
                continue
            inputs.fuel_overrides[field] = value.strip()
            continue
        if key_l.startswith("mol."):
            subkey = key_l.split(".", 1)[1]
            if subkey == "enable":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_enable = b
            elif subkey == "n_cells":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_n_cells = int(v)
            elif subkey == "n_species":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_n_species = int(v)
            elif subkey == "n_reactions":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_n_reactions = int(v)
            elif subkey == "h_conv":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_h_conv = v
            elif subkey == "back_bc":
                inputs.mol_back_bc = value.strip().lower()
            elif subkey == "k_crack_frac":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_k_crack_frac = v
            elif subkey == "grid_stretch":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_grid_stretch = v
            elif subkey == "char_ox_enable":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_char_ox_enable = b
            elif subkey == "char_ox_q_ref_w_m2":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_char_ox_q_ref_W_m2 = v
            elif subkey == "char_ox_m_py_stefan0":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_char_ox_m_py_stefan0 = v
            elif subkey == "surface_recession_enable":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_surface_recession_enable = b
            elif subkey == "surface_rho_floor":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_surface_rho_floor = v
            elif subkey == "in_depth_rad_kappa":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_in_depth_rad_kappa = v
            elif subkey == "in_depth_rad_density_weighted":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_in_depth_rad_density_weighted = b
            elif subkey == "spall_enable":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_spall_enable = b
            elif subkey == "spall_depth_m":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_spall_depth_m = v
            elif subkey == "spall_rho_floor":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_spall_rho_floor = v
            elif subkey == "cp_gas":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_cp_gas = v
            elif subkey == "surface_y_o2":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_surface_y_o2 = v
            elif subkey == "material_coords":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_material_coords = b
            elif subkey == "lagrangian_mode":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_lagrangian_mode = b
            elif subkey == "charring_front_bc":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_charring_front_bc = b
            elif subkey == "charring_t_py":
                inputs.mol_charring_T_py = float(value)
            elif subkey == "surface_ablation_bc":
                b = _parse_bool(value)
                if b is not None:
                    inputs.mol_surface_ablation_bc = b
            elif subkey == "surface_ablation_l_py":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_surface_ablation_L_py = v
            elif subkey == "surface_ablation_t_min":
                v = _parse_float(value)
                if v is not None:
                    inputs.mol_surface_ablation_T_min = v
            elif subkey.startswith("species."):
                # mol.species.IDX.field  →  _mol_species_tmp[IDX][field]
                # Field names stored lowercase; no bool parsing (0/1 are numeric, not flags)
                parts = subkey.split(".", 2)  # ["species", "0", "field"]
                if len(parts) == 3:
                    try:
                        idx = int(parts[1])
                    except ValueError:
                        pass
                    else:
                        field_name = parts[2]  # already lowercase via key_l
                        if idx not in _mol_species_tmp:
                            _mol_species_tmp[idx] = {}
                        fv = _parse_optional_float(value)
                        if fv is not None:
                            _mol_species_tmp[idx][field_name] = fv
                        else:
                            _mol_species_tmp[idx][field_name] = value.strip()
            elif subkey.startswith("reaction."):
                # mol.reaction.IDX.field  →  _mol_reactions_tmp[IDX][field]
                # Field names stored lowercase; no bool parsing (0/1 are numeric indices)
                parts = subkey.split(".", 2)  # ["reaction", "0", "field"]
                if len(parts) == 3:
                    try:
                        idx = int(parts[1])
                    except ValueError:
                        pass
                    else:
                        field_name = parts[2]  # already lowercase via key_l
                        if idx not in _mol_reactions_tmp:
                            _mol_reactions_tmp[idx] = {}
                        fv = _parse_optional_float(value)
                        if fv is not None:
                            _mol_reactions_tmp[idx][field_name] = fv
                        else:
                            _mol_reactions_tmp[idx][field_name] = value.strip()
            continue
        if key_l.startswith("sim."):
            field = key_l.split(".", 1)[1]
            b = _parse_bool(value)
            if b is not None:
                inputs.sim_overrides[field] = b
                continue
            val = _parse_float(value)
            if val is not None:
                inputs.sim_overrides[field] = val
                continue
            inputs.sim_overrides[field] = value.strip()
            continue
        if key_l.startswith("output."):
            field = key_l.split(".", 1)[1]
            b = _parse_bool(value)
            if b is not None:
                inputs.output_overrides[field] = b
                continue
            val = _parse_float(value)
            if val is not None:
                # store int for dpi-like integer fields
                inputs.output_overrides[field] = int(val) if val == int(val) else val
                continue
            inputs.output_overrides[field] = value.strip()
            continue
        # ── Outdoor environment (Rothermel 1972, Anderson 1982, Nelson 2000) ──
        if key_l.startswith("outdoor."):
            field = key_l.split(".", 1)[1]
            b = _parse_bool(value)
            if b is not None:
                inputs.outdoor_overrides[field] = b
                continue
            val_opt = _parse_optional_float(value)
            if val_opt is not None or value.strip().lower() in {"none", "null", "nan", ""}:
                inputs.outdoor_overrides[field] = val_opt
                continue
            inputs.outdoor_overrides[field] = value.strip()
            continue
        # ── Spray suppression device (Rasbash 1962, Johansson et al. 2018) ──
        if key_l.startswith("spray."):
            field = key_l.split(".", 1)[1]
            b = _parse_bool(value)
            if b is not None:
                inputs.spray_overrides[field] = b
                continue
            val_opt = _parse_optional_float(value)
            if val_opt is not None or value.strip().lower() in {"none", "null", "nan", ""}:
                inputs.spray_overrides[field] = val_opt
                continue
            inputs.spray_overrides[field] = value.strip()
            continue

    if inputs.hoc_eff is not None:
        inputs.hoc_eff_J_kg = hoc_eff_to_j_per_kg(inputs.hoc_eff, inputs.hoc_units)

    # Consolidate mol species / reactions from explicit deck specification
    if _mol_species_tmp:
        n_sp = max(_mol_species_tmp.keys()) + 1
        inputs.mol_species_list = [_mol_species_tmp.get(i, {}) for i in range(n_sp)]
    if _mol_reactions_tmp:
        n_rxn = max(_mol_reactions_tmp.keys()) + 1
        inputs.mol_reactions_list = [_mol_reactions_tmp.get(i, {}) for i in range(n_rxn)]

    # ── Materials database backfill (Rule #11: all applied values logged) ─
    if inputs.material_name is not None:
        from model.materials_db import apply_material_db
        applied = apply_material_db(inputs)
        if applied:
            print(f"[materials_db] '{inputs.material_name}' — "
                  f"applied {len(applied)} defaults:")
            for line in applied:
                print(f"  {line}")
        else:
            print(f"[materials_db] '{inputs.material_name}' — "
                  f"all parameters explicitly set in deck.")
    # ── Re-derive hoc_eff_J_kg after DB backfill in case hoc_eff was filled ─
    if inputs.hoc_eff is not None and inputs.hoc_eff_J_kg is None:
        inputs.hoc_eff_J_kg = hoc_eff_to_j_per_kg(inputs.hoc_eff, inputs.hoc_units)

    return inputs


def output_config_from_inputs(inputs: "RomInputs") -> "OutputConfig":
    """Build an OutputConfig from the output_overrides parsed from a deck."""
    from model.config.schemas import OutputConfig  # avoid circular at module level
    cfg = OutputConfig()
    for key, val in inputs.output_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def _warn(msg: str) -> None:
    print(f"Warning: {msg}")


def resolve_geometry(inputs: RomInputs, area_default: float) -> tuple[float, float | None]:
    area = None
    if inputs.area_m2 is not None:
        area = inputs.area_m2
    if inputs.length_m is not None and inputs.width_m is not None:
        area_from_dims = inputs.length_m * inputs.width_m
        if area is not None and abs(area - area_from_dims) / max(area, 1e-9) > 0.05:
            _warn("Both area_m2 and length/width provided; using area_m2.")
        if area is None:
            area = area_from_dims

    if area is None:
        area = area_default

    L_m = None
    if area is not None and area > 0.0:
        L_m = area**0.5
    return area, L_m


def resolve_node_fracs(inputs: RomInputs) -> tuple[float | None, float | None]:
    f1 = inputs.node1_frac
    f2 = inputs.node2_frac
    if f1 is None and f2 is None:
        return None, None
    if f1 is None and f2 is not None:
        f1 = 1.0 - f2
    if f2 is None and f1 is not None:
        f2 = 1.0 - f1
    if f1 is None or f2 is None:
        return f1, f2
    total = f1 + f2
    if abs(total - 1.0) > 1e-6:
        _warn("node1_frac + node2_frac != 1; renormalizing.")
        f1 = f1 / total
        f2 = f2 / total
    return f1, f2


def resolve_node_fracs_three(inputs: RomInputs) -> tuple[float | None, float | None, float | None]:
    f1 = inputs.node1_frac
    f2 = inputs.node2_frac
    f3 = inputs.node3_frac
    if f1 is None and f2 is None and f3 is None:
        return None, None, None

    if f3 is None:
        # Backward-compatible split: treat the legacy two-node interior fraction as two equal
        # control volumes when the user enables a 3-node thermal model without specifying node3.
        f1_two, f2_two = resolve_node_fracs(inputs)
        if f1_two is None or f2_two is None:
            return None, None, None
        return float(f1_two), float(0.5 * f2_two), float(0.5 * f2_two)

    vals = [f1, f2, f3]
    if sum(v is None for v in vals) > 1:
        _warn("Need at least two of node1_frac/node2_frac/node3_frac for 3-node derivation.")
        return None, None, None
    if f1 is None:
        f1 = 1.0 - float(f2) - float(f3)
    if f2 is None:
        f2 = 1.0 - float(f1) - float(f3)
    if f3 is None:
        f3 = 1.0 - float(f1) - float(f2)
    if f1 is None or f2 is None or f3 is None:
        return None, None, None

    total = float(f1 + f2 + f3)
    if total <= 0.0:
        _warn("node1_frac + node2_frac + node3_frac <= 0; cannot derive 3-node thermal chain.")
        return None, None, None
    if abs(total - 1.0) > 1e-6:
        _warn("node1_frac + node2_frac + node3_frac != 1; renormalizing.")
        f1 = float(f1) / total
        f2 = float(f2) / total
        f3 = float(f3) / total
    return float(f1), float(f2), float(f3)


def resolve_node_fracs_N(inputs: RomInputs, N: int) -> list[float] | None:
    """Resolve and renormalize N node fractions from inputs.nodeI_frac fields.

    Returns a list of N floats summing to 1.0, or None if fractions are missing/invalid.
    Requires all N fractions to be explicitly specified (no backward-compat fallback for N>3).
    """
    frac_attrs = ["node1_frac", "node2_frac", "node3_frac", "node4_frac", "node5_frac"]
    if N > len(frac_attrs):
        _warn(f"resolve_node_fracs_N: N={N} exceeds maximum supported nodes ({len(frac_attrs)}).")
        return None
    fracs = [getattr(inputs, frac_attrs[i]) for i in range(N)]
    if any(f is None for f in fracs):
        _warn(f"resolve_node_fracs_N: not all node fracs specified for N={N} thermal nodes.")
        return None
    fracs = [float(f) for f in fracs]
    total = sum(fracs)
    if total <= 0.0:
        _warn(f"resolve_node_fracs_N: node fracs sum to {total}; cannot derive thermal chain.")
        return None
    if abs(total - 1.0) > 1e-6:
        _warn(f"resolve_node_fracs_N: node fracs sum {total:.6f} != 1; renormalizing.")
        fracs = [f / total for f in fracs]
    return fracs


def apply_material_geometry(inputs: RomInputs, fuel_cfg) -> None:
    # Apply direct material overrides
    if inputs.thermal_model_order is not None:
        fuel_cfg.thermal_model_order = int(inputs.thermal_model_order)
    if inputs.C3 is not None:
        fuel_cfg.C3 = inputs.C3
    if inputs.K23 is not None:
        fuel_cfg.K23 = inputs.K23
    if inputs.eps is not None:
        fuel_cfg.eps = inputs.eps
    if inputs.dH_py is not None:
        fuel_cfg.dH_py = inputs.dH_py
    if inputs.back_bc_mode is not None:
        fuel_cfg.back_bc_mode = inputs.back_bc_mode
    if inputs.h_open is not None:
        fuel_cfg.h_open = inputs.h_open
    if inputs.eps_open is not None:
        fuel_cfg.eps_open = inputs.eps_open
    if inputs.back_face_q_in_W_m2 is not None:
        fuel_cfg.back_face_q_in_W_m2 = inputs.back_face_q_in_W_m2
    if inputs.back_face_pyrolysis_enable is not None:
        fuel_cfg.back_face_pyrolysis_enable = inputs.back_face_pyrolysis_enable
    if inputs.back_face_node_frac is not None:
        fuel_cfg.back_face_node_frac = inputs.back_face_node_frac
    if inputs.back_face_T_py_K is not None:
        fuel_cfg.back_face_T_py_K = inputs.back_face_T_py_K
    if inputs.back_face_hog_enable is not None:
        fuel_cfg.back_face_hog_enable = inputs.back_face_hog_enable
    if inputs.back_face_hog_min_char_frac is not None:
        fuel_cfg.back_face_hog_min_char_frac = inputs.back_face_hog_min_char_frac
    if inputs.back_face_hog_k_crack_frac is not None:
        fuel_cfg.back_face_hog_k_crack_frac = inputs.back_face_hog_k_crack_frac
    if inputs.back_face_hog_T_min_K is not None:
        fuel_cfg.back_face_hog_T_min_K = inputs.back_face_hog_T_min_K
    if inputs.back_face_hog_T_ramp_dT_K is not None:
        fuel_cfg.back_face_hog_T_ramp_dT_K = inputs.back_face_hog_T_ramp_dT_K
    if inputs.back_face_hog_ramp_width is not None:
        fuel_cfg.back_face_hog_ramp_width = inputs.back_face_hog_ramp_width
    if inputs.front_hog_floor_enable is not None:
        fuel_cfg.front_hog_floor_enable = inputs.front_hog_floor_enable
    if inputs.front_hog_floor_L_eff_J_kg is not None:
        fuel_cfg.front_hog_floor_L_eff_J_kg = inputs.front_hog_floor_L_eff_J_kg
    if inputs.q_loss3 is not None:
        fuel_cfg.q_loss3 = inputs.q_loss3
    if inputs.A_py is not None:
        fuel_cfg.A_py = inputs.A_py
    if inputs.E_py is not None:
        fuel_cfg.E_py = inputs.E_py
    if inputs.alpha_moist is not None:
        fuel_cfg.alpha_moist = inputs.alpha_moist
    if inputs.alpha_burnthrough is not None:
        fuel_cfg.alpha_burnthrough = inputs.alpha_burnthrough
    if inputs.kinetics_mode is not None:
        fuel_cfg.kinetics_mode = inputs.kinetics_mode
    if inputs.sigmoid_T0_K is not None:
        fuel_cfg.sigmoid_T0_K = inputs.sigmoid_T0_K
    if inputs.sigmoid_dT_K is not None:
        fuel_cfg.sigmoid_dT_K = inputs.sigmoid_dT_K
    if inputs.A1_py is not None:
        fuel_cfg.A1_py = inputs.A1_py
    if inputs.E1_py is not None:
        fuel_cfg.E1_py = inputs.E1_py
    if inputs.A2_py is not None:
        fuel_cfg.A2_py = inputs.A2_py
    if inputs.E2_py is not None:
        fuel_cfg.E2_py = inputs.E2_py
    if inputs.A3_py is not None:
        fuel_cfg.A3_py = inputs.A3_py
    if inputs.E3_py is not None:
        fuel_cfg.E3_py = inputs.E3_py
    if inputs.seq_y1_vol is not None:
        fuel_cfg.seq_y1_vol = inputs.seq_y1_vol
    if inputs.seq_y2_vol is not None:
        fuel_cfg.seq_y2_vol = inputs.seq_y2_vol
    if inputs.seq_T_ign_K is not None:
        fuel_cfg.seq_T_ign_K = inputs.seq_T_ign_K
    if inputs.seq_vol_interp_n is not None:
        fuel_cfg.seq_vol_interp_n = inputs.seq_vol_interp_n
    if inputs.seq_f12_to_m2 is not None:
        fuel_cfg.seq_f12_to_m2 = inputs.seq_f12_to_m2
    if inputs.seq_secondary_char_enable is not None:
        fuel_cfg.seq_secondary_char_enable = bool(inputs.seq_secondary_char_enable)
    if inputs.seq_pool2_use_back_node is not None:
        fuel_cfg.seq_pool2_use_back_node = bool(inputs.seq_pool2_use_back_node)
    if inputs.seq_m1_frac is not None:
        fuel_cfg.seq_m1_frac = inputs.seq_m1_frac
    if inputs.seq_m2_frac0 is not None:
        fuel_cfg.seq_m2_frac0 = inputs.seq_m2_frac0
    if inputs.seq_mr_frac0 is not None:
        fuel_cfg.seq_mr_frac0 = inputs.seq_mr_frac0
    if inputs.seq_clamp_yields is not None:
        fuel_cfg.seq_clamp_yields = bool(inputs.seq_clamp_yields)
    if inputs.sg_n1 is not None:
        fuel_cfg.sg_n1 = inputs.sg_n1
    if inputs.sg_n2 is not None:
        fuel_cfg.sg_n2 = inputs.sg_n2
    if inputs.sg_n3 is not None:
        fuel_cfg.sg_n3 = inputs.sg_n3
    if inputs.sg_y_g1 is not None:
        fuel_cfg.sg_y_g1 = inputs.sg_y_g1
    if inputs.sg_y_i1 is not None:
        fuel_cfg.sg_y_i1 = inputs.sg_y_i1
    if inputs.sg_y_c1 is not None:
        fuel_cfg.sg_y_c1 = inputs.sg_y_c1
    if inputs.sg_y_g2 is not None:
        fuel_cfg.sg_y_g2 = inputs.sg_y_g2
    if inputs.sg_y_c2 is not None:
        fuel_cfg.sg_y_c2 = inputs.sg_y_c2
    if inputs.sg_clamp_yields is not None:
        fuel_cfg.sg_clamp_yields = bool(inputs.sg_clamp_yields)
    if inputs.reactive_access_mode is not None:
        fuel_cfg.reactive_access_mode = str(inputs.reactive_access_mode).strip().lower()
    if inputs.access_reduction_beta is not None:
        fuel_cfg.access_reduction_beta = inputs.access_reduction_beta
    if inputs.access_min is not None:
        fuel_cfg.access_min = inputs.access_min
    if inputs.access_driver is not None:
        fuel_cfg.access_driver = str(inputs.access_driver).strip().lower()
    if inputs.access_target is not None:
        fuel_cfg.access_target = str(inputs.access_target).strip().lower()
    if inputs.access_depth_weight_pow is not None:
        fuel_cfg.access_depth_weight_pow = inputs.access_depth_weight_pow
    if inputs.chem_depth_partition_mode is not None:
        fuel_cfg.chem_depth_partition_mode = str(inputs.chem_depth_partition_mode).strip().lower()
    if inputs.M1_represents is not None:
        fuel_cfg.M1_represents = inputs.M1_represents
    if inputs.pyrolysis_mass_source is not None:
        fuel_cfg.pyrolysis_mass_source = inputs.pyrolysis_mass_source
    if inputs.m_fuel_total_kg_m2 is not None:
        fuel_cfg.m_fuel_total_kg_m2 = inputs.m_fuel_total_kg_m2
    if inputs.A_basis is not None:
        fuel_cfg.A_basis = inputs.A_basis
        fuel_cfg.pyrolysis_rate_basis = inputs.A_basis
    if inputs.front_limit_enable is not None:
        fuel_cfg.front_limit_enable = inputs.front_limit_enable
    if inputs.front_limit_surface_only is not None:
        fuel_cfg.front_limit_surface_only = inputs.front_limit_surface_only
    if inputs.regression_alpha is not None:
        fuel_cfg.regression_alpha = inputs.regression_alpha
    if inputs.regression_T_py_K is not None:
        fuel_cfg.regression_T_py_K = inputs.regression_T_py_K
    if inputs.regression_delta_min_m is not None:
        fuel_cfg.regression_delta_min_m = inputs.regression_delta_min_m
    if inputs.regression_delta_cap_m is not None:
        fuel_cfg.regression_delta_cap_m = inputs.regression_delta_cap_m
    if inputs.regression_spall_onset_frac is not None:
        fuel_cfg.regression_spall_onset_frac = inputs.regression_spall_onset_frac
    if inputs.regression_spall_reduction_frac is not None:
        fuel_cfg.regression_spall_reduction_frac = inputs.regression_spall_reduction_frac
    if inputs.softmin_beta is not None:
        fuel_cfg.softmin_beta = inputs.softmin_beta
    if inputs.handoff_start_frac is not None:
        fuel_cfg.handoff_start_frac = inputs.handoff_start_frac
    if inputs.handoff_end_frac is not None:
        fuel_cfg.handoff_end_frac = inputs.handoff_end_frac
    if inputs.delta_py0_m is not None:
        fuel_cfg.delta_py0_m = inputs.delta_py0_m
    if inputs.m_char0_kg_m2 is not None:
        fuel_cfg.m_char0_kg_m2 = inputs.m_char0_kg_m2
    if inputs.regression_L0_m is not None:
        fuel_cfg.regression_L0_m = inputs.regression_L0_m
    if inputs.hog_enable is not None:
        fuel_cfg.hog_enable = inputs.hog_enable
    if inputs.hog_L_eff_J_kg is not None:
        fuel_cfg.hog_L_eff_J_kg = inputs.hog_L_eff_J_kg
    if inputs.hog_q_crit_W_m2 is not None:
        fuel_cfg.hog_q_crit_W_m2 = inputs.hog_q_crit_W_m2
    if inputs.therm_pen_enable is not None:
        fuel_cfg.therm_pen_enable = inputs.therm_pen_enable
    if inputs.char_ox_enable is not None:
        fuel_cfg.char_ox_enable = inputs.char_ox_enable
    if inputs.char_ox_q_ref_W_m2 is not None:
        fuel_cfg.char_ox_q_ref_W_m2 = inputs.char_ox_q_ref_W_m2
    if inputs.char_ox_q_stefan0_W_m2 is not None:
        fuel_cfg.char_ox_q_stefan0_W_m2 = inputs.char_ox_q_stefan0_W_m2
    if inputs.char_ox_char_yield is not None:
        fuel_cfg.char_ox_char_yield = inputs.char_ox_char_yield
    if inputs.char_ox_char_hoc_J_kg is not None:
        fuel_cfg.char_ox_char_hoc_J_kg = inputs.char_ox_char_hoc_J_kg
    if inputs.char_ox_m_py_stefan0_kg_m2_s is not None:
        fuel_cfg.char_ox_m_py_stefan0_kg_m2_s = inputs.char_ox_m_py_stefan0_kg_m2_s
    if inputs.char_smolder_enable is not None:
        fuel_cfg.char_smolder_enable = inputs.char_smolder_enable
    if inputs.char_smolder_q_ref_W_m2 is not None:
        fuel_cfg.char_smolder_q_ref_W_m2 = inputs.char_smolder_q_ref_W_m2
    if inputs.char_smolder_char_yield is not None:
        fuel_cfg.char_smolder_char_yield = inputs.char_smolder_char_yield
    if inputs.char_smolder_hoc_J_kg is not None:
        fuel_cfg.char_smolder_hoc_J_kg = inputs.char_smolder_hoc_J_kg
    if inputs.char_smolder_m_py_s0_kg_m2_s is not None:
        fuel_cfg.char_smolder_m_py_s0_kg_m2_s = inputs.char_smolder_m_py_s0_kg_m2_s
    if inputs.rho_solid is not None:
        fuel_cfg.rho_solid = inputs.rho_solid
    if inputs.k_char is not None and inputs.rho_char is None:
        # k_char without full char material properties: Stefan-front-only mode.
        # Full char capacitances (C_char) are derived in the geometry section below
        # only when rho_char+cp_char are also provided (char_state_mode=kinetic).
        fuel_cfg.k_char = inputs.k_char
    if inputs.k_evap0 is not None:
        fuel_cfg.k_evap0 = inputs.k_evap0
    if inputs.T_evap_onset is not None:
        fuel_cfg.T_evap_onset = inputs.T_evap_onset

    kinetics_mode = str(getattr(fuel_cfg, "kinetics_mode", "arrhenius") or "arrhenius").strip().lower()
    if kinetics_mode == "two_step_sequential":
        missing = [
            name
            for name in ("A1_py", "E1_py", "A2_py", "E2_py")
            if getattr(fuel_cfg, name, None) is None
        ]
        if missing:
            raise ValueError(
                "two_step_sequential requires explicit branch parameters; missing: "
                + ", ".join(missing)
            )
        if bool(getattr(fuel_cfg, "seq_secondary_char_enable", False)):
            missing_char = [
                name for name in ("A3_py", "E3_py") if getattr(fuel_cfg, name, None) is None
            ]
            if missing_char:
                raise ValueError(
                    "two_step_sequential with seq_secondary_char_enable requires explicit branch parameters; missing: "
                    + ", ".join(missing_char)
                )

        clamp_yields = bool(getattr(fuel_cfg, "seq_clamp_yields", True))
        for attr in ("seq_y1_vol", "seq_y2_vol", "seq_f12_to_m2"):
            val = getattr(fuel_cfg, attr, None)
            if val is None:
                raise ValueError(f"two_step_sequential missing required unit-interval field: {attr}")
            fval = float(val)
            if clamp_yields:
                fval = max(0.0, min(1.0, fval))
                setattr(fuel_cfg, attr, fval)
            elif not (0.0 <= fval <= 1.0):
                raise ValueError(f"two_step_sequential yield out of bounds for {attr}: {fval}")

        frac_sum = (
            float(getattr(fuel_cfg, "seq_m1_frac", 0.0))
            + float(getattr(fuel_cfg, "seq_m2_frac0", 0.0))
            + float(getattr(fuel_cfg, "seq_mr_frac0", 0.0))
        )
        if abs(frac_sum - 1.0) > 1.0e-9:
            raise ValueError(
                "two_step_sequential requires seq_m1_frac + seq_m2_frac0 + seq_mr_frac0 == 1 "
                f"(got {frac_sum:.12g})"
            )
    if kinetics_mode == "semi_global_seq_yield":
        missing = [
            name
            for name in ("A1_py", "E1_py", "A2_py", "E2_py")
            if getattr(fuel_cfg, name, None) is None
        ]
        if missing:
            raise ValueError(
                "semi_global_seq_yield requires explicit branch parameters; missing: "
                + ", ".join(missing)
            )
        if bool(getattr(fuel_cfg, "seq_secondary_char_enable", False)):
            missing_char = [
                name for name in ("A3_py", "E3_py") if getattr(fuel_cfg, name, None) is None
            ]
            if missing_char:
                raise ValueError(
                    "semi_global_seq_yield with seq_secondary_char_enable requires explicit branch parameters; missing: "
                    + ", ".join(missing_char)
                )
        clamp_sg = bool(getattr(fuel_cfg, "sg_clamp_yields", True))
        for attr in ("sg_y_g1", "sg_y_i1", "sg_y_c1", "sg_y_g2", "sg_y_c2"):
            val = getattr(fuel_cfg, attr, None)
            if val is None:
                raise ValueError(f"semi_global_seq_yield missing required yield field: {attr}")
            fval = float(val)
            if clamp_sg:
                fval = max(0.0, min(1.0, fval))
                setattr(fuel_cfg, attr, fval)
            elif not (0.0 <= fval <= 1.0):
                raise ValueError(f"semi_global_seq_yield yield out of bounds for {attr}: {fval}")
        sum1 = (
            float(getattr(fuel_cfg, "sg_y_g1", 0.0))
            + float(getattr(fuel_cfg, "sg_y_i1", 0.0))
            + float(getattr(fuel_cfg, "sg_y_c1", 0.0))
        )
        sum2 = float(getattr(fuel_cfg, "sg_y_g2", 0.0)) + float(getattr(fuel_cfg, "sg_y_c2", 0.0))
        if abs(sum1 - 1.0) > 1.0e-9:
            raise ValueError(
                "semi_global_seq_yield requires sg_y_g1 + sg_y_i1 + sg_y_c1 == 1 "
                f"(got {sum1:.12g})"
            )
        if abs(sum2 - 1.0) > 1.0e-9:
            raise ValueError(
                "semi_global_seq_yield requires sg_y_g2 + sg_y_c2 == 1 "
                f"(got {sum2:.12g})"
            )
        for attr in ("sg_n1", "sg_n2", "sg_n3"):
            val = getattr(fuel_cfg, attr, None)
            if val is None:
                raise ValueError(f"semi_global_seq_yield missing reaction-order field: {attr}")
            if float(val) <= 0.0:
                raise ValueError(f"semi_global_seq_yield reaction order must be > 0 for {attr}: {val}")
        frac_sum = (
            float(getattr(fuel_cfg, "seq_m1_frac", 0.0))
            + float(getattr(fuel_cfg, "seq_m2_frac0", 0.0))
            + float(getattr(fuel_cfg, "seq_mr_frac0", 0.0))
        )
        if abs(frac_sum - 1.0) > 1.0e-9:
            raise ValueError(
                "semi_global_seq_yield requires seq_m1_frac + seq_m2_frac0 + seq_mr_frac0 == 1 "
                f"(got {frac_sum:.12g})"
            )
    if inputs.k_temp_mode is not None:
        fuel_cfg.k_temp_mode = inputs.k_temp_mode

    thermal_order = int(getattr(fuel_cfg, "thermal_model_order", 2) or 2)
    if thermal_order < 2:
        raise ValueError(f"thermal_model_order must be >= 2 (got {thermal_order})")
    chem_depth_mode = str(getattr(fuel_cfg, "chem_depth_partition_mode", "none") or "none").strip().lower()
    if chem_depth_mode not in {"none", "thermal_nodes"}:
        raise ValueError(
            "chem_depth_partition_mode must be 'none' or 'thermal_nodes' "
            f"(got {getattr(fuel_cfg, 'chem_depth_partition_mode', None)!r})"
        )
    fuel_cfg.chem_depth_partition_mode = chem_depth_mode
    if chem_depth_mode == "thermal_nodes" and thermal_order < 3:
        raise ValueError("chem_depth_partition_mode='thermal_nodes' requires thermal_model_order >= 3")

    access_mode = str(getattr(fuel_cfg, "reactive_access_mode", "none") or "none").strip().lower()
    if access_mode not in {"none", "transport_reduced_wood_char"}:
        raise ValueError(
            "reactive_access_mode must be 'none' or 'transport_reduced_wood_char' "
            f"(got {getattr(fuel_cfg, 'reactive_access_mode', None)!r})"
        )
    fuel_cfg.reactive_access_mode = access_mode

    access_driver = str(getattr(fuel_cfg, "access_driver", "residue_fraction") or "residue_fraction").strip().lower()
    if access_driver not in {"residue_fraction", "residue_and_depth"}:
        raise ValueError(
            "access_driver must be 'residue_fraction' or 'residue_and_depth' "
            f"(got {getattr(fuel_cfg, 'access_driver', None)!r})"
        )
    fuel_cfg.access_driver = access_driver

    access_target = str(getattr(fuel_cfg, "access_target", "stage2_only") or "stage2_only").strip().lower()
    if access_target not in {"stage2_only", "delayed_paths"}:
        raise ValueError(
            "access_target must be 'stage2_only' or 'delayed_paths' "
            f"(got {getattr(fuel_cfg, 'access_target', None)!r})"
        )
    fuel_cfg.access_target = access_target

    for attr in ("access_reduction_beta", "access_min", "access_depth_weight_pow"):
        val = getattr(fuel_cfg, attr, None)
        try:
            fval = float(val)
        except Exception:
            raise ValueError(f"{attr} must be a finite number (got {val!r})")
        if not math.isfinite(fval):
            raise ValueError(f"{attr} must be finite (got {val!r})")
        setattr(fuel_cfg, attr, fval)
    if float(getattr(fuel_cfg, "access_reduction_beta", 0.0)) < 0.0:
        raise ValueError("access_reduction_beta must be >= 0")
    if float(getattr(fuel_cfg, "access_depth_weight_pow", 0.0)) < 0.0:
        raise ValueError("access_depth_weight_pow must be >= 0")
    access_min_val = float(getattr(fuel_cfg, "access_min", 0.0))
    if not (0.0 <= access_min_val <= 1.0):
        raise ValueError(f"access_min must be in [0,1] (got {access_min_val})")

    # Derive thermal control-volume capacities/couplings from geometry + material
    if inputs.thickness_m is None:
        return
    fuel_cfg.thickness_m = inputs.thickness_m
    if not hasattr(fuel_cfg, "regression_L0_m") or getattr(fuel_cfg, "regression_L0_m", None) is None:
        fuel_cfg.regression_L0_m = inputs.thickness_m
    else:
        fuel_cfg.regression_L0_m = inputs.thickness_m if inputs.regression_L0_m is None else inputs.regression_L0_m
    if inputs.density is not None:
        fuel_cfg.rho = inputs.density
        derived_m_tot = inputs.density * inputs.thickness_m
        fuel_cfg.m_fuel_kg_m2 = derived_m_tot
        if inputs.m_fuel_total_kg_m2 is None:
            fuel_cfg.m_fuel_total_kg_m2 = derived_m_tot
        fuel_cfg.rho_solid = inputs.density
    if inputs.density is None or inputs.cp is None or inputs.k is None:
        _warn("thickness_m provided without density/cp/k; cannot derive thermal C/K chain.")
        return

    if thermal_order == 3:
        f1, f2, f3 = resolve_node_fracs_three(inputs)
        if f1 is None or f2 is None or f3 is None:
            _warn("node fractions missing; cannot derive C1/C2/C3/K12/K23.")
            return
        th1 = inputs.thickness_m * f1
        th2 = inputs.thickness_m * f2
        th3 = inputs.thickness_m * f3
        fuel_cfg.C1 = inputs.density * inputs.cp * th1
        fuel_cfg.C2 = inputs.density * inputs.cp * th2
        fuel_cfg.C3 = inputs.density * inputs.cp * th3
        fuel_cfg.K12 = inputs.k / max(0.5 * (th1 + th2), 1.0e-9)
        fuel_cfg.K23 = inputs.k / max(0.5 * (th2 + th3), 1.0e-9)
        if inputs.k_char is not None and inputs.rho_char is not None and inputs.cp_char is not None:
            fuel_cfg.k_char = inputs.k_char
            fuel_cfg.rho_char = inputs.rho_char
            fuel_cfg.cp_char = inputs.cp_char
            fuel_cfg.C1_char = inputs.rho_char * inputs.cp_char * th1
            fuel_cfg.C2_char = inputs.rho_char * inputs.cp_char * th2
            fuel_cfg.C3_char = inputs.rho_char * inputs.cp_char * th3
            fuel_cfg.K12_char = inputs.k_char / max(0.5 * (th1 + th2), 1.0e-9)
            fuel_cfg.K23_char = inputs.k_char / max(0.5 * (th2 + th3), 1.0e-9)
    elif thermal_order >= 4:
        fracs = resolve_node_fracs_N(inputs, thermal_order)
        if fracs is None:
            _warn(f"node fractions missing; cannot derive thermal chain for {thermal_order}-node model.")
            return
        ths = [inputs.thickness_m * f for f in fracs]
        has_char = (inputs.k_char is not None and inputs.rho_char is not None and inputs.cp_char is not None)
        if has_char:
            fuel_cfg.k_char = inputs.k_char
            fuel_cfg.rho_char = inputs.rho_char
            fuel_cfg.cp_char = inputs.cp_char
        for i in range(thermal_order):
            setattr(fuel_cfg, f"C{i + 1}", inputs.density * inputs.cp * ths[i])
            if has_char:
                setattr(fuel_cfg, f"C{i + 1}_char", inputs.rho_char * inputs.cp_char * ths[i])
            if i < thermal_order - 1:
                k_link = inputs.k / max(0.5 * (ths[i] + ths[i + 1]), 1.0e-9)
                setattr(fuel_cfg, f"K{i + 1}{i + 2}", k_link)
                if has_char:
                    k_link_char = inputs.k_char / max(0.5 * (ths[i] + ths[i + 1]), 1.0e-9)
                    setattr(fuel_cfg, f"K{i + 1}{i + 2}_char", k_link_char)
    else:
        f1, f2 = resolve_node_fracs(inputs)
        if f1 is None or f2 is None:
            _warn("node fractions missing; cannot derive C1/C2/K12.")
            return
        th1 = inputs.thickness_m * f1
        th2 = inputs.thickness_m * f2
        fuel_cfg.C1 = inputs.density * inputs.cp * th1
        fuel_cfg.C2 = inputs.density * inputs.cp * th2
        fuel_cfg.K12 = inputs.k / max(0.5 * (th1 + th2), 1.0e-9)
        if inputs.k_char is not None and inputs.rho_char is not None and inputs.cp_char is not None:
            fuel_cfg.k_char = inputs.k_char
            fuel_cfg.rho_char = inputs.rho_char
            fuel_cfg.cp_char = inputs.cp_char
            fuel_cfg.C1_char = inputs.rho_char * inputs.cp_char * th1
            fuel_cfg.C2_char = inputs.rho_char * inputs.cp_char * th2
            fuel_cfg.K12_char = inputs.k_char / max(0.5 * (th1 + th2), 1.0e-9)
    fuel_cfg.K12_ref = fuel_cfg.K12
    fuel_cfg.k_ref = inputs.k
    fuel_cfg.evolving_props_passes = inputs.evolving_props_passes
    if inputs.char_state_mode is not None:
        fuel_cfg.char_state_mode = inputs.char_state_mode
    if inputs.A_char is not None:
        fuel_cfg.A_char = inputs.A_char
    if inputs.E_char is not None:
        fuel_cfg.E_char = inputs.E_char


def q_in_callable(schedule: List[Tuple[float, float]], hold_last: bool = True):
    if not schedule:
        return lambda t: 0.0
    schedule = sorted(schedule, key=lambda x: x[0])
    times = [p[0] for p in schedule]
    values = [p[1] for p in schedule]

    def _q_in(t: float) -> float:
        if t <= times[0]:
            return values[0]
        for i in range(1, len(times)):
            if t <= times[i]:
                t0, t1 = times[i - 1], times[i]
                q0, q1 = values[i - 1], values[i]
                if t1 == t0:
                    return q1
                frac = (t - t0) / (t1 - t0)
                return q0 + frac * (q1 - q0)
        return values[-1] if hold_last else 0.0

    return _q_in


def q_inc_ramp_factor(t: float, mode: str, tau: float) -> float:
    mode_l = (mode or "none").strip().lower()
    if mode_l == "none":
        return 1.0
    tau_eff = float(tau)
    if tau_eff <= 0.0:
        return 1.0
    t_eff = max(float(t), 0.0)
    if mode_l == "exp":
        return 1.0 - math.exp(-t_eff / tau_eff)
    if mode_l == "cosine":
        if t_eff >= tau_eff:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * (t_eff / tau_eff)))
    return 1.0


def ramped_q_in_callable(q_raw: Callable[[float], float], mode: str, tau: float) -> Callable[[float], float]:
    def _q(t: float, q0=q_raw, m=mode, tau_s=tau) -> float:
        return float(q0(t) * q_inc_ramp_factor(t, m, tau_s))

    return _q


def convert_q_in(value: float, units: str) -> float:
    u = units.strip().lower()
    if "kw" in u and "/m2" in u:
        return value * 1.0e3
    return value


def convert_m_py(value: float, units: str, hoc_eff: float = 1.0, hoc_units: str = "kJ/kg") -> float:
    """Convert a prescribed schedule value to m_py [kg/m^2/s]."""
    u = units.strip().lower().replace(" ", "")
    if "kw" in u and "/m2" in u:
        # Treat as HRRPUA [kW/m2]; convert using hoc_eff [kJ/kg].
        hoc_kj = hoc_eff_to_kj_per_kg(hoc_eff, hoc_units)
        denom = max(hoc_kj if hoc_kj is not None else 0.0, 1.0e-9)
        return value / denom
    if "kg" in u:
        return value
    if "g" in u and "/m2" in u:
        return value / 1000.0
    return value
