"""
Material properties database for the unitiedmodel2 ROM.

Usage in an input deck::

    material.name = wheat_straw

Any parameter explicitly set in the deck overrides the database default.
When a material name is given, the solver prints every parameter it supplies
so the user can verify what was applied (Rule #11 compliance).

Adding a new material
---------------------
1. Copy the template below, fill in all required fields.
2. Add a ``_sources`` entry for every non-trivial parameter (Rule #6).
3. Add a ``_cal_case`` string naming the calibration deck and achieved R/E.
4. Register alternate names in ``_ALIASES``.

Kinetics calibration basis
--------------------------
Kinetics (A_py, E_py) are CONE-CALIBRATED: fitted to reproduce the cone
calorimeter HRRPUA curve using the ROM's heat transfer model.  This is
the correct basis for the PDE spread model because:

  1. The PDE uses the SAME gas-solid coupling (Ranz-Marshall h_p × a_v)
  2. The A_eff scaling (A_py × m_element / m_bed) handles geometry
  3. Flame feedback is separately modeled (χ_rad × m_dot × HoC × VF)

TGA values (A ~ 10^13-10^15, E ~ 200-250 kJ/mol) are more fundamental
but describe mass loss at controlled heating rates — they would require
re-deriving the heat transfer coupling for the ROM.  Store TGA values
as ``_tga_A``, ``_tga_E`` reference fields, not as primary kinetics.

PDE spread model usage
----------------------
The spread solver computes A_eff from the DB entry's element geometry:

  A_eff = A_py × (density × thickness_m) / (rho_bulk × h_bed)

This scales the cone-calibrated rate to the outdoor fuel bed mass basis.
For materials where the element IS the bed (e.g. SH2 bed-scale deck),
density × thickness = rho_bulk × h_bed → A_eff = A_py (no scaling).

Future: chemistry-class interpolation (Tier 2, near-term)
---------------------------------------------------------
For fuels WITHOUT cone data, interpolate kinetics from the nearest
calibrated material by proximate analysis (cellulose/lignin fractions):

  Grasses:  A ~ 1-10 /s,    E ~ 50-60 kJ/mol  (cellulose-dominated)
  Shrubs:   A ~ 50-200 /s,  E ~ 50-70 kJ/mol  (higher lignin)
  Litter:   A ~ 0.1-5 /s,   E ~ 40-55 kJ/mol  (partially degraded)

Future: Bayesian Hierarchical Model (Tier 3, long-term)
-------------------------------------------------------
With 20+ calibrated materials, pool information across fuel classes
using BHM to predict (A, E) for new materials from proximate analysis
with uncertainty quantification.

Reserved keys (not forwarded to RomInputs):
    _sources   — provenance strings for Rule #6 console print
    _cal_case  — calibration summary string (informational only)
    _fuel      — dict of fuel.* override keys (forwarded to fuel_overrides)
    _tga_A     — TGA-measured pre-exponential (reference, not primary)
    _tga_E     — TGA-measured activation energy (reference, not primary)
    _spread    — PDE spread model notes (A_eff, validation status)
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Database
# Each entry maps RomInputs field names → values, with reserved _* keys.
# ---------------------------------------------------------------------------

MATERIALS_DB: dict[str, dict[str, Any]] = {

    # ═══════════════════════════════════════════════════════════════════════
    # Wheat straw (Triticum aestivum) — packed cone calorimeter
    # Calibration: Chen et al. (2021) cone, 50 kW/m²
    # Shared (Rule #5) by rice straw and corn straw (same cellulosic class)
    # ═══════════════════════════════════════════════════════════════════════
    "wheat_straw": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":   171.0,     # kg/m³  — Chen et al. (2021) 60g / 100×100×35mm
        "cp":       1300.0,     # J/(kg·K) — Janssens (1993) dry cellulosic
        "k":           0.08,    # W/(m·K)  — Koufopanos et al. (1991) packed dry biomass
        "eps":         0.90,    # [-]      — Dietenberger (2002) typical dry biomass

        # ── Geometry defaults (Chen 2021 cone holder) ──────────────────────
        "thermal_model_order": 3,
        "thickness_m":   0.035,
        "node1_frac":    0.10,  # surface; dx1 = 3.5 mm
        "node2_frac":    0.40,  # intermediate
        "node3_frac":    0.50,  # bulk

        # ── Kinetics — two_step_sequential (cone-calibrated for ROM) ─────
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,         # dry fuel — no evaporation heat sink
        "A1_py":  10.0,         # [1/s]   fast surface pool — CALIBRATED 2026-03-26
        "E1_py": 5.6e4,         # [J/mol]
        "A2_py":   1.2,         # [1/s]   bulk pool — CALIBRATED 2026-03-29 (3-node+flame)

        # ── TGA intrinsic kinetics (for PDE spread model) ─────────────────
        # Cellulose-dominated grass: Orfão (1999) Fuel 78:349 Table 2.
        # These are INTRINSIC chemistry — no geometry dependence.
        # The PDE spread solver uses these by default (no A_eff scaling).
        "_tga_A": 2.07e14,      # [1/s]  microcrystalline cellulose (Orfão 1999)
        "_tga_E": 178700.0,     # [J/mol] (Orfão 1999)
        "E2_py": 5.6e4,         # [J/mol] Di Blasi (2008) range 40–120 kJ/mol
        "seq_m1_frac":  0.05,   # [-]  5% fast surface fraction
        "seq_m2_frac0": 0.95,   # [-]  bulk
        "seq_mr_frac0": 0.00,   # [-]  no inert residue (dry grass fully volatile)
        "seq_f12_to_m2": 0.0,   # [-]  independent pools
        "seq_y1_vol": 1.0,      # [-]  Pool 1 fully volatile
        "seq_y2_vol": 1.0,      # [-]  Pool 2 fully volatile

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 12500.0,     # kJ/kg — Leventon et al. (2025) NIST TN 2314 bluestem grass

        # ── Char oxidation (smoldering) ────────────────────────────────────
        "char_ox_enable":                    True,
        "char_ox_char_yield":               0.15,   # [-]        Orfão (1999) lignin fraction
        "char_ox_q_ref_W_m2":          45000.0,     # [W/m²]     T6a energy balance 2026-03-30
        "char_ox_m_py_stefan0_kg_m2_s":  0.005,     # [kg/m²/s]  blow suppression threshold

        # ── fuel.* overrides ────────────────────────────────────────────────
        "_fuel": {
            "flame_enable":      True,
            "flame_chi_rad":     0.34,   # NIST TN 2314 Sung et al. 2025 little bluestem
            "flame_view_factor": 0.40,   # standard cone calorimeter geometry
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density/cp/k/eps": "Chen et al. (2021); Janssens (1993); Koufopanos (1991); Dietenberger (2002)",
            "kinetics":         "CALIBRATED — Chen_Wheat_Straw__CONE_50 (2026-03-29); pk R/E=1.013, avg=1.036, late=0.877",
            "char_ox":          "Orfão (1999) lignin fraction 0.15; T6a energy balance q_ref=45 kW/m² (2026-03-30)",
            "hoc_eff":          "Leventon et al. (2025) NIST TN 2314 ΔHc,gas bluestem grass 12,500 kJ/kg",
            "flame":            "NIST TN 2314 Sung et al. (2025) chi_rad=0.34 little bluestem; vf=0.40 cone geometry",
        },
        "_cal_case": "Chen_Wheat_Straw__CONE_50; pk R/E=1.013 avg R/E=1.039 late R/E=0.877 (all PASS)",

        # ── PDE spread model (per-cell Arrhenius) ──────────────────────────
        "_spread": {
            "A_eff_basis": "A2_py × (density × thickness_m) / (rho_bulk × h_bed)",
            "A_eff_GR1": "2.0 × 5.985 / 0.072 = 166 /s (h_bed=0.30m, rho=0.24)",
            "A_eff_GR3": "2.0 × 5.985 / 0.182 = 65.6 /s (h_bed=0.76m, rho=0.24)",
            "burn_time_900K_GR1": "10.7 s",
            "validation_3D_50m": "6/6 PASS ratios 0.94-1.71 vs Cheney (1993)",
            "domain_needed": "50m for quasi-steady (d_H up to 23m at U=8)",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # Rice straw (Oryza sativa) — identical material properties to wheat straw
    # Per Rule #5: same cellulosic class → shared parameters.
    # Designation: VALIDATION (Rice_Straw_50 kW/m²)
    # ═══════════════════════════════════════════════════════════════════════
    "rice_straw": None,   # resolved via alias → wheat_straw (see _ALIASES)

    # ═══════════════════════════════════════════════════════════════════════
    # Corn straw (Zea mays) — identical material properties to wheat straw
    # Per Rule #5 (same cellulosic material class — shared parameters).
    # ═══════════════════════════════════════════════════════════════════════
    "corn_straw": None,   # resolved via alias → wheat_straw (see _ALIASES)

    # ═══════════════════════════════════════════════════════════════════════
    # FSRI Wood Stud (SPF 2×4, 38mm) — Stefan char-front, 3-node
    # Calibration: FSRI database, 50 kW/m² (or all three flux levels)
    # ═══════════════════════════════════════════════════════════════════════
    "fsri_wood_stud": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":   379.0,     # kg/m³  — FSRI HFM / database
        "cp":       1242.0,     # J/(kg·K) — FSRI specific heat (dried)
        "k":           0.10,    # W/(m·K)  — FSRI thermal conductivity
        "eps":         0.857,   # [-]      — FSRI emissivity measurement

        # ── Char properties ─────────────────────────────────────────────────
        "k_char":    0.08,      # W/(m·K)  — Drysdale (2011) / FSRI literature
        "rho_char":  130.0,     # kg/m³    — Drysdale (2011)
        "cp_char":  1100.0,     # J/(kg·K) — Drysdale (2011)
        "char_state_mode": "kinetic",

        # ── Geometry defaults (38mm FSRI stud section, 3-node) ─────────────
        "thermal_model_order": 3,
        "thickness_m":   0.038,
        "node1_frac":    0.20,
        "node2_frac":    0.30,
        "node3_frac":    0.50,

        # ── Kinetics — arrhenius + Stefan front ────────────────────────────
        "kinetics_mode":         "arrhenius",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "A_py":  3.1e11,        # [1/s]   — FSRI calibration
        "E_py":  1.62e5,        # [J/mol] — FSRI calibration
        "regression_L0_m":  0.038,
        "front_limit_enable":    True,
        "regression_T_py_K": 548.0,    # K — Drysdale (2011) char onset ~250°C
        "dH_py":          1800000.0,   # J/kg — Janssens (1994) DSC for SPF
        "regression_delta_min_m": 1e-4,
        "softmin_beta":    0.0,
        "hog_enable":      False,
        "therm_pen_enable": False,
        "regression_spall_onset_frac":    0.0,
        "regression_spall_reduction_frac": 0.0,

        # ── Char oxidation ─────────────────────────────────────────────────
        "char_ox_enable":                    True,
        "char_ox_q_ref_W_m2":          70000.0,
        "char_ox_q_stefan0_W_m2":      80000.0,
        "char_ox_m_py_stefan0_kg_m2_s":  0.010,

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 15500.0,     # kJ/kg — FSRI / Tewarson wood value

        # ── Back-face pyrolysis defaults ────────────────────────────────────
        "back_face_pyrolysis_enable": True,
        "back_face_node_frac": 0.10,
        "back_face_T_py_K":  600.0,

        # ── fuel.* overrides ─────────────────────────────────────────────────
        "_fuel": {
            "flame_enable":       True,
            "flame_chi_rad":      0.35,    # Tewarson SFPE §3.4 wood
            "flame_view_factor":  0.40,
            "flame_persistence_s": 5.0,
            "flame_m_py_ignite":  0.005,
            "flame_m_py_crit":    0.001,
            "flame_T_ignite":   600.0,
            "flame_T_py":       500.0,
            "k_crack_frac":      10.0,    # Di Blasi (2002); Shi & Chew (2023)
            "vol_pool_enable":    True,
            "vol_pool_tau_auto":  False,
            "vol_pool_tau_s":    20.0,
            "vol_pool_t_peak_s":  0.0,
            "vol_pool_a_preheat_1_s":   2.0e4,
            "vol_pool_e_preheat_j_mol": 80000.0,
            "lig_enable": False,
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density/cp/k/eps":    "FSRI materials database (FSRI_Wood_Stud)",
            "k_char/rho_char":     "Drysdale (2011) Table 5.5 fully-converted char",
            "kinetics":            "CALIBRATED — FSRI_Wood_Stud_3NODE__CONE_50 (2026-03-16)",
            "regression_T_py_K":   "Drysdale (2011) ~250°C; Kashiwagi (1981) 250–270°C",
            "dH_py":               "Janssens (1994) DSC for SPF 1.8 MJ/kg",
            "char_ox":             "Calibrated to FSRI 3-node decks (q_ref=70 kW/m²)",
            "hoc_eff":             "Tewarson SFPE §3.4 softwood 15–16 MJ/kg",
            "k_crack_frac":        "Di Blasi (2002); Moghtaderi (2006); Shi & Chew (2023)",
            "flame":               "Tewarson SFPE §3.4 chi_rad=0.35 generic wood",
        },
        "_cal_case": "FSRI_Wood_Stud_3NODE__CONE_50; pk R/E=1.142 avg R/E=1.039 (PASS)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # RISE FR Particle Board 16mm — two_step_sequential + HoG rate cap
    # Calibration: RISE dataset, 3-flux (25/35/50 kW/m²)
    # ═══════════════════════════════════════════════════════════════════════
    "rise_fr_pb_16mm": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":  583.0,      # kg/m³  — RISE measured
        "cp":      1437.0,      # J/(kg·K) — RISE DSC
        "k":         0.124,     # W/(m·K)  — RISE HFM
        "eps":       0.929,     # [-]      — RISE

        # ── Char properties ─────────────────────────────────────────────────
        "k_char":    0.10,      # W/(m·K)  — cellulosic char, Di Blasi (2008)
        "rho_char":  150.0,     # kg/m³
        "cp_char":  1100.0,     # J/(kg·K)
        "char_state_mode": "none",

        # ── Geometry defaults (16mm, 3-node) ───────────────────────────────
        "thermal_model_order": 3,
        "thickness_m":  0.016,
        "node1_frac":   0.20,
        "node2_frac":   0.30,
        "node3_frac":   0.50,

        # ── Kinetics — two_step_sequential + HoG ───────────────────────────
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "hog_enable":       True,
        "hog_L_eff_J_kg": 4270000.0,    # J/kg — calibrated to cone peak scaling
        "hog_q_crit_W_m2":    0.0,
        "therm_pen_enable": False,
        "softmin_beta":    30.0,
        "front_limit_enable": False,
        "regression_L0_m":  0.016,
        "char_ox_enable":   False,

        "A1_py": 1.03e7,    # [1/s]   Orfão (1999) glucomannan (hemicellulose proxy)
        "E1_py": 9.0e4,     # [J/mol] midpoint UF/PF resin (Gaur 1995) + hemicellulose
        "seq_y1_vol": 1.0,
        "seq_m1_frac": 0.10,
        "A2_py": 3.1e4,     # [1/s]   Kung (1972) + FR barrier suppression
        "E2_py": 1.62e5,    # [J/mol]
        "seq_y2_vol": 0.15,
        "seq_m2_frac0": 0.90,

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 11780.0,     # kJ/kg — RISE measured effective HoC

        # ── fuel.* overrides ─────────────────────────────────────────────────
        "_fuel": {},  # no flame feedback for FR PB in production decks

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density/cp/k/eps":  "RISE measurements (Scaling_Pyrolysis dataset)",
            "k_char/rho_char":   "Di Blasi (2008) review cellulosic char",
            "kinetics":          "CALIBRATED — RISE_FR_PB_16mm_3NODE__CONE_* (2026-03-18); pk R/E 0.992–1.018",
            "A1_py/E1_py":       "Orfão (1999) glucomannan; Gaur & Reed (1995) UF/PF resin 80–90 kJ/mol",
            "A2_py/E2_py":       "Kung (1972) + FR polyphosphoric glass barrier suppression (7 OOM)",
            "hog_L_eff":         "Calibrated: EXP peaks scale ∝ q_in → L_eff = q_in/peak × hoc_eff ≈ 4.27 MJ/kg",
            "hoc_eff":           "RISE cone calorimeter derived effective HoC = 11,780 kJ/kg",
        },
        "_cal_case": "RISE_FR_PB_16mm_3NODE__CONE_25/35/50; pk R/E=0.992/1.018/1.000 (all PASS)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # Grass blade-scale element — outdoor spread model (thermally thin regime)
    # Calibration: Cheney et al. (1993) field ROS, U=0–8 m/s
    #
    # DISTINCT from wheat_straw (cone geometry, 35mm, heat-transfer limited).
    # Wildfire blade elements are ~0.3–1 mm → thermally thin (τ_thermal ≈ 0.07 s)
    # → pyrolysis is kinetically controlled.  Rate and density differ from cone.
    #
    # Kinetics are empirically calibrated to Cheney (1993) field ROS, NOT to TGA.
    # MoL + Orfão TGA kinetics (20 K/min) fail structurally on 1D slab (T6a 2026-03-30).
    # Effective A1=750/s at E=56 kJ/mol is consistent with Grønli (2002) prediction
    # for thin hemicellulosic fuels: A~10³–10⁴/s at E~56 kJ/mol basis.
    # ═══════════════════════════════════════════════════════════════════════
    "wheat_straw_blade": {
        # ── Thermal & physical (effective blade density matches field fuel load) ─
        "density":   72.0,     # kg/m³  ρ_eff = ρ_bulk × h_bed / δ = 0.24×0.30/0.001
        "cp":      1300.0,     # J/(kg·K) — Janssens (1993) dry cellulosic
        "k":          0.08,    # W/(m·K)  — Koufopanos et al. (1991) dry biomass
        "eps":        0.90,    # [-]      — Dietenberger (2002) dry biomass

        # ── Geometry (1 mm blade element, 2-node) ──────────────────────────────
        "thermal_model_order": 2,
        "thickness_m":  0.001,
        "node1_frac":   0.50,  # 2-node symmetric; thin slab, spatial mesh irrelevant

        # ── Kinetics — two_step_sequential, empirically fitted to Cheney ROS ───
        # A1=750/s: production value — passes monotone criterion at U=0..8 m/s
        # A1=600/s: free-burn variant (closer to U=0 ratio=1.0 but fails monotone)
        # Decks that need the free-burn value should override A1_py explicitly.
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "A1_py":  750.0,    # [1/s]   Cheney (1993) CAL; monotone U=0..8 m/s
        "E1_py": 5.6e4,     # [J/mol] Di Blasi (2008) cellulosic range
        "A2_py":  100.0,    # [1/s]   A2/A1 ≈ 0.13; consistent with Chen (2021) ratio
        "E2_py": 5.6e4,     # [J/mol]
        "seq_m1_frac":   0.05,  # 5% fast surface fraction
        "seq_m2_frac0":  0.95,  # 95% bulk
        "seq_mr_frac0":  0.00,  # dry grass fully volatile
        "seq_f12_to_m2": 0.0,
        "seq_y1_vol":    1.0,
        "seq_y2_vol":    1.0,

        # ── fuel.* overrides ────────────────────────────────────────────────────
        # No char_ox: blade burns out in ~30–60 s; no sustained smoldering phase.
        "_fuel": {
            "flame_enable":          True,
            "flame_chi_rad":         0.34,  # NIST TN 2314 Sung et al. (2025) little bluestem
            "flame_view_factor":     0.70,  # blade-in-bed; Rothermel (1972) ≥0.7 for grass bed
            "flame_coupling_passes": 3,
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────────
        "_sources": {
            "density":       "ρ_eff = ρ_bulk × h_bed / δ = 0.24×0.30/0.001 — Anderson (1982) GR1",
            "cp/k/eps":      "Janssens (1993); Koufopanos (1991); Dietenberger (2002)",
            "kinetics":      "CALIBRATED — Cheney (1993) ROS U=0; monotone at U=0–8 m/s (2026-03-28)."
                             " Grønli et al. (2002): thin hemicellulosic fuels A~10³–10⁴/s at E=56 kJ/mol",
            "flame_vf":      "Rothermel (1972) INT-115 radiation exchange factor ≥0.70 for grass bed",
            "flame_chi_rad": "NIST TN 2314 Sung et al. (2025) little bluestem measured mean",
        },
        "_cal_case": "Cheney (1993) ROS; 6/6 wind cases PASS (band 0.30–3.0); monotone U=0–8 m/s",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GR3 Tall Grass blade-scale element — outdoor spread model
    # Anderson (1982) NFFL GR3: fuel_depth=0.76m, SAV=4921/m, ρ_bulk=0.24 kg/m³
    # δ = 4/SAV = 0.81mm → 0.8mm; ρ_eff = 0.24×0.76/0.0008 = 228 kg/m³
    # fuel_load = ρ_eff × δ = 228 × 0.0008 = 0.182 kg/m² ✓ (Anderson 1982 GR3)
    # Same cellulosic chemistry as GR1 blade; A1 scan pending CAL run.
    # Calibration: Cheney (1993) Eq. 2 at U=0 m/s; Rule #3 acceptance [0.33,3.0];
    #              monotone ROS(U=0–8 m/s) required.
    # ═══════════════════════════════════════════════════════════════════════
    "gr3_tall_grass": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":  228.0,      # kg/m³  ρ_eff = ρ_bulk × h_bed / δ = 0.24×0.76/0.0008
        "cp":      1300.0,      # J/(kg·K) — Janssens (1993) dry cellulosic
        "k":          0.08,     # W/(m·K)  — Koufopanos et al. (1991) dry biomass
        "eps":        0.90,     # [-]      — Dietenberger (2002) dry biomass

        # ── Geometry (0.8mm blade element, 2-node) ──────────────────────────
        "thermal_model_order": 2,
        "thickness_m":  0.0008,  # δ = 4/4921 m = 0.81mm; Anderson (1982) GR3 SAV=4921/m
        "node1_frac":   0.50,

        # ── Kinetics — two_step_sequential, same class as wheat_straw_blade ─
        # A1=750/s: element free-burn PASS (peak 355 kW/m², Rothermel GR3 I_R [350–500]).
        # Cascade ROS deferred — deep-bed structural limitation (Rule #4, 2026-03-31):
        # h=0.76m drives Byram intensity 3.7× GR1 → spread flux ~4× → runaway cascade.
        # A1 scan [50–750/s] gives floor ratio=9–17 (temperature-limited at T>800K).
        # Domain boundary: cascade valid for thin-bed grass h≤~0.30m only.
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "A1_py":  750.0,    # [1/s]   A1=750 → elem free-burn PASS (355 kW/m²); cascade deferred (Rule #4)
        "E1_py": 5.6e4,     # [J/mol] Di Blasi (2008) cellulosic range; same class as GR1
        "A2_py":  100.0,    # [1/s]
        "E2_py": 5.6e4,     # [J/mol]
        "seq_m1_frac":   0.05,   # 5% fast surface fraction
        "seq_m2_frac0":  0.95,   # 95% bulk
        "seq_mr_frac0":  0.00,   # dry grass fully volatile (same as GR1 blade)
        "seq_f12_to_m2": 0.0,
        "seq_y1_vol":    1.0,
        "seq_y2_vol":    1.0,

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 14900.0,  # kJ/kg — Sung et al. (2025) NIST TN 2314 little bluestem
                              # Same grass class as GR1; direct measurement.

        # ── fuel.* overrides ─────────────────────────────────────────────────
        # No char_ox: blade burns out in ~30–60 s; no sustained smoldering.
        "_fuel": {
            "flame_enable":          True,
            "flame_chi_rad":         0.34,  # NIST TN 2314 Sung et al. (2025) little bluestem
            "flame_view_factor":     0.70,  # blade-in-bed; deeper GR3 bed ≥ GR1 0.70 (Rothermel 1972)
            "flame_coupling_passes": 3,
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density":       "ρ_eff = ρ_bulk × h_bed / δ = 0.24×0.76/0.0008 — Anderson (1982) GR3 NFFL",
            "thickness_m":   "δ = 4/SAV = 4/4921 m — Anderson (1982) GR3 SAV=4921 /m",
            "cp/k/eps":      "Janssens (1993); Koufopanos (1991); Dietenberger (2002) — same cellulosic class",
            "kinetics":      "A1=750/s — elem free-burn PASS 355 kW/m² (Rothermel GR3 I_R [350–500]); cascade deferred Rule #4 deep-bed 2026-03-31",
            "hoc_eff":       "Sung et al. (2025) NIST TN 2314 Table 3 little bluestem 14,900 kJ/kg",
            "flame_chi_rad": "NIST TN 2314 Sung et al. (2025) little bluestem measured mean",
            "flame_vf":      "Rothermel (1972) INT-115 grass bed radiation exchange ≥0.70",
        },
        "_cal_case": "STRUCT LIMIT Rule #4 (2026-03-31) — elem free-burn PASS 355 kW/m²; cascade ratio=23 (deep bed h=0.76m, domain h≤0.30m)",

        # ── PDE spread model ───────────────────────────────────────────────
        # NOTE: This blade-element entry (rho=228, delta=0.8mm) is for the
        # CASCADE spread model.  The PDE spread model uses the CONE element
        # (wheat_straw DB entry: rho=171, L=35mm) with GR3 OUTDOOR params.
        # Deck: Outdoor_GR3_CONE__spread_source.txt
        "_spread": {
            "element_source": "wheat_straw (cone geometry, Rule #5 shared kinetics)",
            "A_eff": "65.6 /s (A2=2.0 × 5.985 / 0.182)",
            "burn_time_900K": "27.1 s",
            "domain_needed": "50m+ (d_H up to 60m at U=8)",
            "status": "pending 50m 3D validation",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SH2 Dead Brush / Chamise — outdoor spread model (thermally thin element)
    # Anderson (1982) NFFL SH2: fuel_depth=0.20m, SAV=5600/m, ρ_bulk=0.48 kg/m³
    # δ = 4/SAV = 0.71mm → 0.7mm; ρ_eff = 0.48×0.20/0.0007 = 137 kg/m³
    # fuel_load = 137 × 0.0007 = 0.096 kg/m² ✓ (Anderson 1982 SH2)
    #
    # Chemistry: dead chamise/shrub — woody-cellulosic + higher lignin fraction.
    # seq_mr_frac0=0.25: Orfão (1999) softwood lignin fraction 25–30%.
    # Slightly higher hoc_eff vs grass: dead woody shrub with resin (Babrauskas 2002).
    # Calibration: Cruz et al. (2010) Int. J. Wildland Fire 19:218 shrub field ROS,
    #              OR Andrews & Chase (1989) BEHAVE SH2 at U=0 m/s.
    # ═══════════════════════════════════════════════════════════════════════
    "sh2_brush": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":    0.48,     # kg/m³  ρ_bulk = fuel_load/h_bed = 0.096/0.200 (Anderson 1982 SH2)
        "cp":      1300.0,      # J/(kg·K) — Janssens (1993) dry cellulosic biomass
        "k":          0.03,     # W/(m·K)  — k_bed sparse shrub; void fraction≈0.999 (ρ_bulk=0.48, ρ_solid≈500)
                                #             k_eff ≈ ε×k_air + contact = 0.999×0.026 + 0.004 ≈ 0.03
                                #             Rothermel (1972) Appendix: heat exchange in porous fuel beds
        "eps":        0.90,     # [-]      — Dietenberger (2002) dry biomass

        # ── Geometry (bed-scale element h_bed=0.200m, 2-node) ───────────────
        "thermal_model_order": 2,
        "thickness_m":  0.200,   # h_bed = 0.200 m — Anderson (1982) SH2 fuel bed depth (bed-scale model)
                                 # fuel_load = ρ_bulk × h_bed = 0.48 × 0.200 = 0.096 kg/m² ✓
                                 # K12 = k_bed/(node1_frac×h_bed) = 0.03/(0.5×0.200) = 0.30 W/m²/K
                                 # (vs 229 W/m²/K for particle-scale δ=0.7mm — 760× reduction)
        "node1_frac":   0.50,

        # ── Kinetics — two_step_sequential, woody-cellulosic with lignin residue ─
        # Pool fractions adjusted for dead shrub / chamise chemistry:
        #   m1=0.05 (hemicellulose fast fraction, same as grass)
        #   m2=0.70 (cellulosic bulk; reduced from 0.95 by 0.25 lignin)
        #   mr=0.25 (lignin residue — non-volatile char; Orfão 1999 softwood ~25–30%)
        # Structural limitation (Rule #4, 2026-03-31):
        #   Free-burn peak 78.8 kW/m² (target [100–400]: FAIL).
        #   A2 scan [100–900/s]: 0/6 PASS; best U=0 ratio=3.72 (A2=450–680, marginally FAIL).
        #   Root cause: mr=0.25 → source HRRPUA ~162 kW/m² < Rothermel SH2 I_R [200–340].
        #   Domain boundary: requires smoldering/glowing-combustion physics (Rule #7).
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "A1_py":  750.0,    # [1/s]   struct limit Rule #4; free-burn 78.8 kW/m² (target [100–400]: FAIL)
        "E1_py": 5.6e4,     # [J/mol] Di Blasi (2008) hemicellulosic range
        "A2_py":  100.0,    # [1/s]   struct limit; A2 scan 0/6 PASS at all values
        "E2_py": 5.6e4,     # [J/mol] cellulosic chemistry Di Blasi (2008)
        "seq_m1_frac":   0.05,   # 5% fast hemicellulose fraction
        "seq_m2_frac0":  0.70,   # 70% cellulosic bulk pool
        "seq_mr_frac0":  0.25,   # 25% lignin residue (non-volatile char; Orfão 1999)
        "seq_f12_to_m2": 0.0,
        "seq_y1_vol":    1.0,
        "seq_y2_vol":    1.0,

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 17000.0,  # kJ/kg — Babrauskas (2002) SFPE Handbook Table 3-4.2
                              # dead woody shrub / chamise; resin-enriched vs grass.

        # ── fuel.* overrides ─────────────────────────────────────────────────
        "_fuel": {
            "flame_enable":          True,
            "flame_chi_rad":         0.30,  # shrub bed; conservative Rule #1
                                             # between grass (0.34) and generic wood (0.35)
                                             # Albini (1985) wildland fuels baseline 0.25–0.35
            "flame_view_factor":     0.55,  # dense shrub bed geometry; Rothermel (1972)
                                             # radiation exchange factor; between grass 0.70
                                             # and horizontal cone 0.40
            "flame_coupling_passes": 3,
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density":       "ρ_bulk = fuel_load/h_bed = 0.096/0.200 = 0.48 kg/m³ — Anderson (1982) SH2 NFFL bed-scale",
            "thickness_m":   "h_bed = 0.200 m — Anderson (1982) SH2 NFFL fuel bed depth (bed-scale model replaces δ=0.7mm)",
            "cp/k/eps":      "cp: Janssens (1993) dry biomass; k: bed-scale k_eff≈0.03 W/m·K sparse shrub (Rothermel 1972 Appendix); eps: Dietenberger (2002)",
            "seq_mr_frac0":  "Orfão (1999) Table 1: softwood lignin fraction 25–30%; dead chamise similar",
            "hoc_eff":       "Babrauskas (2002) SFPE Handbook Table 3-4.2 dead woody shrub ~17 MJ/kg",
            "flame_chi_rad": "Albini (1985) wildland fuels; conservative between grass 0.34 and wood 0.35 (Rule #1)",
            "flame_vf":      "Rothermel (1972) INT-115 radiation exchange; denser shrub bed than grass",
            "kinetics":      "two_step_sequential woody-cellulosic; A2=100/s (Di Blasi 2008); bed-scale model (2026-04-01) replaces particle-scale; CAL pending",
        },
        "_cal_case": "BED-SCALE (2026-04-01) — thickness 0.7mm→200mm, density 137→0.48 kg/m³, k_solid→k_bed=0.03; K12 229→0.30 W/m²/K; free-burn peak 116.7 kW/m² PASS [100,400]; cascade 5/6 PASS (U=8 1D-cascade boundary)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # TL1 Timber Litter / Pine Needle — outdoor free-burn element
    # Anderson (1982) NFFL TL1: fuel_load=0.165 kg/m², fuel_depth=0.03m, SAV=6562/m
    # ρ_bulk = 0.165/0.03 = 5.5 kg/m³; δ = 4/SAV = 0.61mm → 0.6mm
    # ρ_eff = 5.5×0.03/0.0006 = 275 kg/m³
    # fuel_load = 275 × 0.0006 = 0.165 kg/m² ✓ (Anderson 1982 TL1)
    #
    # Chemistry: coniferous needle — char-forming (lignin 15–25%), resin-enriched.
    # Higher E2=75 kJ/mol: cellulose + resin activation (Di Blasi 2008 65–80 kJ/mol).
    # Higher hoc_eff: pine resin contribution (Tewarson 1995 SFPE).
    # No spread cascade: compact litter burns in smoldering mode; 1D slab
    # structural limitation (Rule #4). Free-burn HRRPUA target only.
    # Calibration: HRRPUA peak ∈ [50, 150 kW/m²] (Rothermel 1972 I_R for TL1).
    # ═══════════════════════════════════════════════════════════════════════
    "tl1_timber_litter": {
        # ── Thermal & physical ─────────────────────────────────────────────
        "density":    5.5,      # kg/m³  ρ_bulk = fuel_load/h_bed = 0.165/0.030 (Anderson 1982 TL1)
        "cp":      1600.0,      # J/(kg·K) — Janssens (1993) coniferous; higher resin
        "k":          0.05,     # W/(m·K)  — k_bed packed pine needle bed (Manzello et al. 2012
                                #             Int. J. Wildland Fire 21(4):388–396: k_eff 0.04–0.07)
        "eps":        0.95,     # [-]      — dark litter; Dietenberger (2002) upper range

        # ── Geometry (bed-scale element h_bed=0.030m, 2-node) ───────────────
        "thermal_model_order": 2,
        "thickness_m":  0.030,   # h_bed = 0.030 m — Anderson (1982) TL1 litter bed depth (bed-scale model)
                                 # fuel_load = ρ_bulk × h_bed = 5.5 × 0.030 = 0.165 kg/m² ✓
                                 # K12 = k_bed/(node1_frac×h_bed) = 0.05/(0.5×0.030) = 3.33 W/m²/K
                                 # (vs 333 W/m²/K for particle-scale δ=0.6mm — 100× reduction)
        "node1_frac":   0.50,

        # ── Kinetics — two_step_sequential, coniferous litter chemistry ──────
        # Pool fractions for coniferous needle chemistry:
        #   m1=0.05 (fast hemicellulose surface fraction)
        #   m2=0.75 (cellulosic bulk + resin)
        #   mr=0.20 (lignin char residue; Van der Weide 2021 pine needle ~15–25%)
        # E2=75 kJ/mol: cellulose + resin activation energy (Di Blasi 2008 65–80 kJ/mol)
        # A2=500/s: faster cellulosic pool consistent with higher E2 (Di Blasi 2008 range)
        # Structural limitation (Rule #4, 2026-03-31):
        #   Free-burn peak 32.1 kW/m² at t=7s (target [50–150]: FAIL); element extinguishes
        #   at t≈10s after ignition pulse ends. Compact litter burns by smoldering, not flaming.
        #   Domain boundary: requires Frandsen (1991) glowing-combustion physics (Rule #7).
        "kinetics_mode":         "two_step_sequential",
        "pyrolysis_mass_source": "fuel_state",
        "k_evap0": 0.0,
        "A1_py":  750.0,    # [1/s]   struct limit Rule #4; free-burn 32.1 kW/m² FAIL (target [50–150])
        "E1_py": 5.6e4,     # [J/mol] hemicellulose basis (Di Blasi 2008)
        "A2_py":  500.0,    # [1/s]   cellulosic + resin pool (Di Blasi 2008; faster at E2=75)
        "E2_py": 7.5e4,     # [J/mol] cellulose + resin 65–80 kJ/mol; Di Blasi (2008)
        "seq_m1_frac":   0.05,   # 5% fast hemicellulose
        "seq_m2_frac0":  0.75,   # 75% cellulosic + resin bulk
        "seq_mr_frac0":  0.20,   # 20% lignin char residue (Van der Weide 2021)
        "seq_f12_to_m2": 0.0,
        "seq_y1_vol":    1.0,
        "seq_y2_vol":    1.0,

        # ── Combustion ─────────────────────────────────────────────────────
        "hoc_eff": 18500.0,  # kJ/kg — Tewarson (1995) SFPE §3.4 pine; resin contribution
                              # pine ~18–20 MJ/kg vs grass ~15 MJ/kg (Babrauskas 2002)

        # ── fuel.* overrides ─────────────────────────────────────────────────
        # Note: flame_view_factor=0.70 — litter BED geometry (needles irradiate adjacent
        # needles in a packed layer; same class as grass blade in grass bed).
        # Corrected 2026-04-01 from 0.40 (cone/flat-specimen) to 0.70 (BED geometry,
        # Rothermel 1972 grass bed ≥0.70).  Rule #1: geometry correction, not re-tuning.
        "_fuel": {
            "flame_enable":          True,
            "flame_chi_rad":         0.25,  # compact horizontal litter; Albini (1985) wildland
                                             # baseline — lower than grass (0.34) due to compact
                                             # horizontal geometry with low flame height
            "flame_view_factor":     0.70,  # litter BED — needles irradiate adjacent needles;
                                             # Rothermel (1972) grass bed ≥0.70; corrected 2026-04-01
            "flame_coupling_passes": 3,
        },

        # ── Provenance (Rule #6) ────────────────────────────────────────────
        "_sources": {
            "density":       "ρ_bulk = fuel_load/h_bed = 0.165/0.030 = 5.5 kg/m³ — Anderson (1982) TL1 NFFL bed-scale",
            "thickness_m":   "h_bed = 0.030 m — Anderson (1982) TL1 litter bed depth (bed-scale model replaces δ=0.6mm needle)",
            "cp":            "Janssens (1993) coniferous biomass; elevated vs grass due to resin",
            "k":             "k_bed packed pine needle bed: Manzello et al. (2012) Int. J. Wildland Fire 21(4):388 k_eff=0.04–0.07 W/m·K; midpoint 0.05",
            "seq_mr_frac0":  "Van der Weide (2021): pine needle lignin fraction 15–25%",
            "E2_py/A2_py":   "Di Blasi (2008) Prog. Energy Combust. Sci. 34: cellulose+resin 65–80 kJ/mol",
            "hoc_eff":       "Tewarson (1995) SFPE §3.4 Table 3-4.2 pine ~18–20 MJ/kg (resin contribution)",
            "flame_chi_rad": "Albini (1985) generic wildland fuels; compact litter geometry (Rule #1)",
            "flame_vf":      "litter BED geometry — needles irradiate adjacent needles; vf corrected 0.40→0.70 (2026-04-01); Rothermel (1972) grass bed ≥0.70",
            "kinetics":      "two_step_sequential coniferous; E2=75 kJ/mol (Di Blasi 2008); bed-scale model (2026-04-01) replaces particle-scale; CAL pending",
        },
        "_cal_case": "BED-SCALE (2026-04-01) — thickness 0.6mm→30mm, density 275→5.5 kg/m³, k_solid→k_bed=0.05; K12 333→3.33 W/m²/K; free-burn peak 69.1 kW/m² PASS [50,150] (Rothermel I_R 65–90 kW/m²); self-sustain t=120s FAIL (compact smoldering litter boundary, Rule #4)",
    },
}

# ---------------------------------------------------------------------------
# Alias resolution (case-insensitive, space ↔ underscore normalised)
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    "wheat straw":         "wheat_straw",
    "rice straw":          "rice_straw",
    "corn straw":          "corn_straw",
    "gr1 grass":           "wheat_straw",
    "gr1":                 "wheat_straw",
    "nffl gr1":            "wheat_straw",
    "straw":               "wheat_straw",
    "fsri wood stud":      "fsri_wood_stud",
    "wood stud":           "fsri_wood_stud",
    "fsri_wood_stud":      "fsri_wood_stud",
    "rise fr pb 16mm":     "rise_fr_pb_16mm",
    "rise fr pb":          "rise_fr_pb_16mm",
    "fr pb 16mm":          "rise_fr_pb_16mm",
    "fr_pb_16mm":          "rise_fr_pb_16mm",
    # GR1 blade-scale entries
    "wheat straw blade":   "wheat_straw_blade",
    "gr1 blade":           "wheat_straw_blade",
    "gr1 grass blade":     "wheat_straw_blade",
    "grass blade":         "wheat_straw_blade",
    "outdoor grass blade": "wheat_straw_blade",
    # GR3 tall grass
    "gr3 tall grass":      "gr3_tall_grass",
    "gr3 grass":           "gr3_tall_grass",
    "gr3":                 "gr3_tall_grass",
    "gr3 blade":           "gr3_tall_grass",
    "tall grass":          "gr3_tall_grass",
    # SH2 brush
    "sh2 brush":           "sh2_brush",
    "sh2":                 "sh2_brush",
    "chaparral":           "sh2_brush",
    "dead brush":          "sh2_brush",
    "brush":               "sh2_brush",
    "chamise":             "sh2_brush",
    # TL1 timber litter
    "tl1 litter":          "tl1_timber_litter",
    "tl1":                 "tl1_timber_litter",
    "timber litter":       "tl1_timber_litter",
    "pine litter":         "tl1_timber_litter",
    "pine needle":         "tl1_timber_litter",
}

# rice/corn straw share all parameters with wheat straw (Rule #5)
MATERIALS_DB["rice_straw"] = MATERIALS_DB["wheat_straw"]
MATERIALS_DB["corn_straw"] = MATERIALS_DB["wheat_straw"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Lower-case and collapse spaces/underscores for alias matching."""
    return name.strip().lower().replace("_", " ")


def lookup_material(name: str) -> dict[str, Any] | None:
    """Return the DB entry for *name* (case-insensitive, alias-aware).

    Returns ``None`` if the material is not found.
    """
    key = _normalise_name(name)
    # Try alias table first, then direct key match
    canonical = _ALIASES.get(key) or _ALIASES.get(name.strip().lower())
    if canonical is None:
        # Try direct match (underscored keys)
        direct = name.strip().lower().replace(" ", "_")
        if direct in MATERIALS_DB:
            canonical = direct
    if canonical is None:
        return None
    return MATERIALS_DB.get(canonical)


def apply_material_db(inputs: "RomInputs") -> list[str]:  # type: ignore[name-defined]
    """Backfill *inputs* fields from the materials database.

    Only fills fields that are currently ``None`` in *inputs* — deck-explicit
    values are never overwritten.  ``fuel.*`` keys are merged into
    ``inputs.fuel_overrides`` (DB values lose on conflict).

    Returns a list of human-readable strings describing applied parameters
    (for Rule #11 console transparency).  Returns an empty list if all
    parameters were already explicitly set in the deck.
    """
    name = inputs.material_name
    if name is None:
        return []

    entry = lookup_material(name)
    if entry is None:
        print(f"[materials_db] WARNING: material '{name}' not found in database. "
              f"Available: {sorted(MATERIALS_DB.keys())}")
        return []

    applied: list[str] = []

    # -- Direct RomInputs fields ------------------------------------------
    reserved = {"_fuel", "_sources", "_cal_case"}
    for key, value in entry.items():
        if key in reserved:
            continue
        if not hasattr(inputs, key):
            print(f"[materials_db] WARNING: DB key '{key}' is not a RomInputs field — skipped.")
            continue
        if getattr(inputs, key) is None:
            setattr(inputs, key, value)
            applied.append(f"{key} = {value!r}")

    # -- fuel.* overrides -------------------------------------------------
    fuel_dict: dict[str, Any] = entry.get("_fuel", {})
    for fuel_key, fuel_val in fuel_dict.items():
        if fuel_key not in inputs.fuel_overrides:
            inputs.fuel_overrides[fuel_key] = fuel_val
            applied.append(f"fuel.{fuel_key} = {fuel_val!r}")

    # -- Provenance summary -----------------------------------------------
    sources = entry.get("_sources", {})
    cal_case = entry.get("_cal_case", "")
    if applied and sources:
        applied.append(f"  [sources: {'; '.join(f'{k}: {v}' for k, v in sources.items())}]")
    if applied and cal_case:
        applied.append(f"  [calibration: {cal_case}]")

    return applied
