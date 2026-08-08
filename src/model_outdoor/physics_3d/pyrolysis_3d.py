"""Pyrolysis kernels for cellulosic biomass (3-pool TGA + single-pool MD2004).

This module exposes two kernels:

1. ``step_pyrolysis`` — 3-pool Arrhenius for hemicellulose, cellulose, lignin
   (Berghel 2023 + Orfão 1999 + Yang 2007).  Used for cone-cal / TGA
   conditions where component identities matter.

2. ``step_pyrolysis_md2004`` — single-pool grass kinetics (Morvan & Dupuy
   2004 Combust. Flame 138:199, calibrated from Grishin 1997 Tomsk Univ.
   Press dry-grass TGA, validated for fire-spread CFD by Mell et al. 2007
   IJWF 16:1).  Used for field-scale wildland grass simulations where the
   bulk fuel is already cured (hemicellulose pre-degraded) and a single
   first-order rate captures the dominant volatile release.

3-pool form (kernel ``step_pyrolysis``):
    dm_i / dt = -A_i * exp(-E_i / (R * T_s)) * m_i      (i = hemi, cell, lign)
    S_pyro = Σ_i  η_i * (-dm_i/dt)
where η_hemi=0.65, η_cell=0.90, η_lign=0.50 (Yang 2007, Shen 2010).

Single-pool form (kernel ``step_pyrolysis_md2004``):
    dm / dt = -A * exp(-E / (R * T_s)) * m
    S_pyro = -dm/dt    (η = 1, all volatiles assumed combustible at field scale)
where A = 36280 s⁻¹, E = 58600 J/mol (MD2004 §2.3 + Eq. 11).

References:
- Berghel et al. (2023) J. Therm. Anal. Calorim. — wheat straw TGA
- Orfão et al. (1999) Fuel 78:349 — biomass component kinetics
- Yang et al. (2007) Fuel 86:1781 — pyrolysis product yields
- Shen et al. (2010) JAAP 87:199 — combustible volatile fractions
- Di Blasi (2008) PECS 34:47 — bulk fire kinetics review
- Morvan & Dupuy (2004) Combust. Flame 138:199 — grass-fire CFD kinetics
- Grishin (1997) Tomsk Univ. Press — original dry-grass TGA
- Mell et al. (2007) IJWF 16:1 — WFDS Cheney (1993) validation
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# Universal gas constant.
_R_GAS = 8.314           # [J/mol/K]

# Activation energies (Phase 14 Try 4, 2026-05-11):
#   Hemi/cell from Várhegyi et al. (1997) *J Anal Appl Pyrolysis* 42:73
#   — multi-rate TGA Kissinger plots on xylan (hemi proxy) and Avicel
#   (pure cellulose).  Higher E values match Yang 2007 DTG peak T's at
#   10°C/min within ±10K; cellulose peak rate matches Yang within 2%.
#   Hemi peak rate is 3× higher than Yang (single-step Arrhenius is too
#   sharp for biomass hemi multi-sugar mix; documented limitation —
#   Várhegyi 2-reaction successive scheme would close this).
# Pre-Try-4: Berghel 2023 wheat-straw (E_HEMI=92, E_CELL=120, A's via
# Orfão kinetic-compensation at T_onset=600K).  Those E's were ~2× low.
E_HEMI = 193_000.0       # [J/mol] Várhegyi 1997 Table 1 xylan primary (193 ± 1 kJ/mol)
E_CELL = 238_000.0       # [J/mol] Várhegyi 1997 §4.2 Avicel cellulose (238 ± 10 kJ/mol)
E_LIGN = 60_800.0        # [J/mol] Orfão lignin (unchanged; lignin peak already in band)

# Pre-exponentials:
#   A_HEMI: Várhegyi 1997 Table 1 (log A1 = 17.0 → A = 1.0e17 1/s)
#   A_CELL: Kissinger-derived to match Yang 2007 peak T = 628 K with
#           Várhegyi E_CELL = 238 kJ/mol (A = (β·E/RT²)·exp(E/RT) at peak).
#   A_LIGN: Orfão (unchanged).
A_HEMI = 1.0e17          # [1/s] Várhegyi 1997 xylan primary reaction
A_CELL = 7.57e17         # [1/s] Kissinger-fit to Yang 2007 cell peak T 628K
A_LIGN = 2.59e1          # [1/s] Orfão lignin (unchanged)

# Combustible volatile fractions per pool.
ETA_HEMI = 0.65          # Yang 2007: ~35 % CO2/H2O inert; balance burns
ETA_CELL = 0.90          # Cellulose: levoglucosan + light tar
ETA_LIGN = 0.50          # Lignin: ~45 % char, balance combust

# Char yield fractions per pool — fraction of pyrolyzed mass that stays
# in solid phase as char (Yang et al. 2007 *Fuel* 86:1781 §3.2.1, solid
# residue at 900°C in N₂ atmosphere at 10°C/min).  When step_pyrolysis is
# called with the `m_char` output array, this fraction of the mass loss
# rate is routed to m_char (instead of being implicitly lost).
# Mass balance per pool:
#   ETA (combustible volatile gas) + CHAR_YIELD (solid char) +
#   (1 - ETA - CHAR_YIELD) (inert volatile gas, CO2/H2O — discarded)
#   = 1.0
CHAR_YIELD_HEMI = 0.20    # Yang 2007 §3.2.1; consistent across biomass sources
CHAR_YIELD_CELL = 0.065   # Yang 2007 §3.2.1; pure cellulose almost fully volatilizes
CHAR_YIELD_LIGN = 0.40    # Yang 2007 §3.2.1; highest char yield (charring-dominated)

# Heat of pyrolysis per pool (Phase 14 Try 3, lit-grounded per Anca-Couce
# 2016 *Prog Energy Combust Sci* 53:41 §4.4 + Yang 2007 char yields).
# Positive = endothermic (energy absorbed); negative = exothermic.
#
# Anca-Couce 2016 documents that biomass pyrolysis is the sum of primary
# devolatilization (endothermic) + secondary charring (exothermic).  Net
# HoR per pool depends on char yield: high-charring pools (lign, hemi)
# are net exothermic; low-charring cellulose remains net endothermic.
#
# Cellulose: primary endo +538 kJ/kg (Milosavljevic & Suuberg 1995) minus
#   charring 0.065 × 3.5 MJ/kg-char ≈ -228 kJ/kg → +310 kJ/kg net endo.
# Hemicellulose: charring 0.20 × 3.5 MJ/kg-char ≈ -700 kJ/kg overtakes
#   smaller primary endo → net -300 kJ/kg exo.  Yang DSC confirms exo peak
#   at 275°C (Yang 2007 §3.2.2).
# Lignin: charring 0.40 × 3.5 MJ/kg-char ≈ -1400 kJ/kg dominates → net
#   -1200 kJ/kg exo.  Yang DSC confirms exo peak at 365°C.
#
# Charring HoR median 3.5 kJ/g-char from Mok-Antal, Milosavljevic, Cho,
# Rath, Basile — convergent across cellulose, beech, spruce, switchgrass,
# corn, poplar (Anca-Couce 2016 §4.4).
HOR_HEMI = -3.0e5         # [J/kg]  exothermic; Anca-Couce 2016 §4.4
HOR_CELL = +3.1e5         # [J/kg]  endothermic; Milosavljevic 1995 + Mok-Antal
HOR_LIGN = -1.2e6         # [J/kg]  exothermic (charring-dominated); Anca-Couce 2016

# Legacy lumped HoR — retained for backward compat / debugging.  Equals
# the mass-weighted mix for Cheney natural grass (42% C / 30% H / 28% L):
#   0.42 × 310 + 0.30 × (-300) + 0.28 × (-1200) ≈ -296 kJ/kg net.
# Active kernel uses per-pool HOR_* values below; HEAT_OF_PYROLYSIS is no
# longer referenced in step_pyrolysis (Phase 14 Try 3, 2026-05-11).
HEAT_OF_PYROLYSIS = 4.0e5  # [J/kg]  (legacy; superseded by HOR_HEMI/CELL/LIGN)


@njit(cache=True, parallel=True)
def step_pyrolysis(
    T_s: np.ndarray,            # (Nz, Ny, Nx) [K]
    m_hemi: np.ndarray,         # (Nz, Ny, Nx) [kg/m³]
    m_cell: np.ndarray,
    m_lign: np.ndarray,
    m_char: np.ndarray,         # (Nz, Ny, Nx) [kg/m³] char accumulator (in/out)
    alpha_s: np.ndarray,        # (Nz, Ny, Nx) [-] used only as a >0 mask
    dt: float,
    S_pyro_out: np.ndarray,     # (Nz, Ny, Nx) volatile source [kg/m³/s] (overwritten)
    Q_pyro_out: np.ndarray,     # (Nz, Ny, Nx) endothermic sink [W/m³] (overwritten)
) -> None:
    """One pyrolysis step.

    In-place update of m_hemi, m_cell, m_lign (mass loss), m_char
    (accumulation).  Fills S_pyro_out (volatile mass added to gas,
    kg/m³/s) and Q_pyro_out (endothermic heat sink for the solid energy
    equation, W/m³).

    Per-pool mass balance (per kg of pool consumed):
      ETA_pool         → combustible volatile gas (entered as S_pyro_out → Y_F)
      CHAR_YIELD_pool  → solid char (accumulated into m_char)
      (1 - ETA - CHAR_YIELD) → inert volatile gas (CO₂/H₂O, currently discarded)

    Cells with alpha_s == 0 are skipped (no fuel).  Mass is clipped at
    zero so depletion is monotone.
    """
    Nz, Ny, Nx = T_s.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s[k, j, i] <= 0.0:
                    S_pyro_out[k, j, i] = 0.0
                    Q_pyro_out[k, j, i] = 0.0
                    continue
                T = T_s[k, j, i]
                if T <= 0.0:
                    S_pyro_out[k, j, i] = 0.0
                    Q_pyro_out[k, j, i] = 0.0
                    continue
                inv_RT = 1.0 / (_R_GAS * T)
                k_hemi = A_HEMI * math.exp(-E_HEMI * inv_RT)
                k_cell = A_CELL * math.exp(-E_CELL * inv_RT)
                k_lign = A_LIGN * math.exp(-E_LIGN * inv_RT)

                mh = m_hemi[k, j, i]
                mc = m_cell[k, j, i]
                ml = m_lign[k, j, i]

                # Implicit single-pool decay over dt: m(t+dt) = m(t)·exp(-k·dt)
                # — preserves positivity and conserves mass without a tiny dt.
                mh_new = mh * math.exp(-k_hemi * dt)
                mc_new = mc * math.exp(-k_cell * dt)
                ml_new = ml * math.exp(-k_lign * dt)

                dmh = mh - mh_new
                dmc = mc - mc_new
                dml = ml - ml_new

                # Mass-loss rates (positive) [kg/m³/s].
                rate_hemi = dmh / dt if dt > 0 else 0.0
                rate_cell = dmc / dt if dt > 0 else 0.0
                rate_lign = dml / dt if dt > 0 else 0.0

                # Volatile source (combustible fraction sums into gas).
                S_pyro_out[k, j, i] = (
                    ETA_HEMI * rate_hemi
                    + ETA_CELL * rate_cell
                    + ETA_LIGN * rate_lign
                )
                # Char accumulation — char fraction of each pool's mass loss
                # stays in solid as char residue (Yang 2007 char yields).
                m_char[k, j, i] += (
                    CHAR_YIELD_HEMI * dmh
                    + CHAR_YIELD_CELL * dmc
                    + CHAR_YIELD_LIGN * dml
                )
                # Heat-of-reaction sink/source per pool (Phase 14 Try 3,
                # Anca-Couce 2016).  Positive = endothermic sink, negative
                # = exothermic source.  Cellulose endo; hemi+lign exo (charring
                # dominated).  Sign of Q_pyro_out matches caller convention:
                # POSITIVE = energy ABSORBED by solid (cools bed).
                Q_pyro_out[k, j, i] = (
                    rate_hemi * HOR_HEMI
                    + rate_cell * HOR_CELL
                    + rate_lign * HOR_LIGN
                )

                m_hemi[k, j, i] = mh_new
                m_cell[k, j, i] = mc_new
                m_lign[k, j, i] = ml_new


# ── Single-pool MD2004 grass kinetics ────────────────────────────────────────
# Morvan & Dupuy (2004) Combust. Flame 138:199 §2.3 Eq. 11; calibrated from
# Grishin (1997) Tomsk Univ. Press dry-grass TGA.  Used by FDS / WFDS
# wildland-fire simulations (Mell et al. 2007 IJWF 16:1).
A_MD2004 = 36_280.0          # [1/s] pre-exponential, thermal pyrolysis Rp
E_MD2004 = 58_600.0          # [J/mol] activation energy, thermal pyrolysis Rp
# Phase 14 Try 5b: split m_loss into volatile + char per Shafizadeh 1968
# char fraction.  Pre-5b: ETA_MD2004=1 with floor-based char retention in
# m_solid; post-5b: m_solid decays to 0, CHAR_YIELD_MD2004 fraction routes
# to dedicated m_char field for char_ox + smolder consumption.
CHAR_YIELD_MD2004 = 0.15      # [-] char yield fraction (Shafizadeh 1968,
                              # cellulose 12-18%; weighted avg 42/30/28
                              # hemi/cell/lign with Yang 2007 yields ≈ 0.20
                              # — kept at 0.15 per Shafizadeh single-pool ref)
ETA_MD2004 = 1.0 - CHAR_YIELD_MD2004  # 0.85 — combustible volatile fraction
CHAR_FRAC_MD2004 = CHAR_YIELD_MD2004   # legacy alias (referenced elsewhere)

# Phase 14 Try 6 (2026-05-11): oxidative pyrolysis path R_op.
# Rate form: m_dot_op = A_op · exp(-E_op/RT) · m_solid · Y_O2^n_O2_OP.
# Parallel to thermal R_p; total mass-loss rate is sum of both paths.
# Lit basis: Lautenberger & Fernandez-Pello 2009 *Fire Saf J* 44:819
# Table 3 PMMA pattern (R2 thermal + R3 oxidative parallel, same substrate
# bpmma).  PMMA ratios: A_op/A_p ≈ 2×; E_op/E_p ≈ 0.94×; n_O2 = 1.31.
# Anca-Couce 2016 §4.5.1: oxidative pyrolysis peak T is 30-50°C lower
# than thermal-only, rate ~1.5× higher in air vs N₂ at low heating rate.
#
# Grass-specific R_op constants are sparse in lit (Anca-Couce notes
# Ohlemiller-scheme grass kinetics are "rare").  Adopt PMMA-pattern
# scaling on MD2004:
A_OP_MD2004 = 7.26e4          # [1/s] = 2× A_MD2004 (PMMA Table 3 R3/R2 ratio)
E_OP_MD2004 = 55_000.0        # [J/mol] = 0.94× E_MD2004 (PMMA pattern)
N_O2_OP = 1.0                 # [-] O₂ exponent (simpler than PMMA 1.31)
# Heat of reaction (R_op exothermic — partial combustion at the bed
# surface releases ~1 MJ/kg of biomass consumed; Anca-Couce 2016 §4.5.1).
HOR_OP_MD2004 = -1.0e6        # [J/kg] negative = exothermic (heat released)
Y_O2_MIN_OP = 1.0e-3          # [-] skip R_op below this O₂ fraction


# ── Explicit drying reaction R_d (Phase 14 Try 7, 2026-05-11) ────────────────
# Arrhenius-form water removal kernel.  Matches the standard wildland-fire
# CFD 4-reaction state model (Lautenberger & Fernandez-Pello 2009 *Fire
# Saf J* 44:819 Table 6, wet wood → dry wood; Ahmed 2024).
#
# Pre-Try-7 the model used pure energy-balance drying inside
# coupling_3d.step_gas_solid_coupling (Phase 14h, line 163).  Energy-
# balance drying remains as a redundant safety after this kernel — both
# operate in series, drying is monotone, can't over-deplete water.
#
# Rate form: dm_water/dt = -k_dry(T) · m_water
#   k_dry = A_DRY · exp(-E_DRY / RT_s)
# Heat:    Q_dry = (mass evaporated)/dt · L_VAP_WATER (positive,
#                                                     endothermic sink)
A_DRY = 4.29e3            # [1/s]  Lautenberger 2009 white pine Table 6 R1
E_DRY = 43_800.0          # [J/mol] Lautenberger 2009 white pine Table 6 R1
L_VAP_WATER = 2.26e6      # [J/kg] latent heat of vaporization @ 100°C


@njit(cache=True, parallel=True)
def step_drying(
    T_s: np.ndarray,            # (Nz, Ny, Nx) [K]
    m_water: np.ndarray,        # (Nz, Ny, Nx) [kg/m³] in/out (decreases)
    alpha_s: np.ndarray,        # (Nz, Ny, Nx) [-] used only as a >0 mask
    dt: float,
    Q_dry_out: np.ndarray,      # (Nz, Ny, Nx) [W/m³] endothermic sink (overwritten)
) -> None:
    """One drying step (Lautenberger-style Arrhenius water removal).

    In-place update of m_water (decreases) and Q_dry_out (overwritten).
    Q_dry_out is POSITIVE (energy absorbed by solid for vaporization);
    caller should add to Q_pyro for the coupling step's endothermic sink.

    Mass is clipped at zero so depletion is monotone.  Cells with
    alpha_s == 0 are skipped (no solid present).
    """
    Nz, Ny, Nx = T_s.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s[k, j, i] <= 0.0:
                    Q_dry_out[k, j, i] = 0.0
                    continue
                mw = m_water[k, j, i]
                if mw <= 0.0:
                    Q_dry_out[k, j, i] = 0.0
                    continue
                T = T_s[k, j, i]
                if T <= 0.0:
                    Q_dry_out[k, j, i] = 0.0
                    continue
                k_dry = A_DRY * math.exp(-E_DRY / (_R_GAS * T))
                # Exact first-order decay: mw_new = mw · exp(-k·dt)
                mw_new = mw * math.exp(-k_dry * dt)
                dm_evap = mw - mw_new
                m_water[k, j, i] = mw_new
                Q_dry_out[k, j, i] = (dm_evap * L_VAP_WATER / dt
                                      if dt > 0.0 else 0.0)


@njit(cache=True, parallel=True)
def step_pyrolysis_md2004(
    T_s: np.ndarray,            # (Nz, Ny, Nx) [K]
    m_solid: np.ndarray,        # (Nz, Ny, Nx) [kg/m³] reactive biomass (decays to 0)
    m_initial: np.ndarray,      # (Nz, Ny, Nx) [kg/m³] initial fuel mass (diagnostic; kept for compat)
    m_char: np.ndarray,         # (Nz, Ny, Nx) [kg/m³] accumulated char (in/out)
    Y_O2: np.ndarray,           # (Nz, Ny, Nx) [-] gas O₂ mass fraction (for R_op path)
    alpha_s: np.ndarray,        # (Nz, Ny, Nx) [-] used only as a >0 mask
    m_water: np.ndarray,        # (Nz, Ny, Nx) [kg/m³] water mass per cell
    m_water_init: float,        # [kg/m³] initial water density (scalar; 0 disables gate)
    dt: float,
    S_pyro_out: np.ndarray,     # (Nz, Ny, Nx) volatile source [kg/m³/s] (overwritten)
    Q_pyro_out: np.ndarray,     # (Nz, Ny, Nx) heat sink/source [W/m³] (overwritten)
    A_p: float = A_MD2004,      # Phase 15I — thermal R_p pre-exponential
                                 # override [1/s].  Default = MD2004 lumped
                                 # (36280); set to a literature single-step
                                 # cellulose value to swap kinetics, e.g.
                                 # Antal-Varhegyi 1998 (5e17, E=236 kJ/mol)
                                 # or Miller-Bellan 1997 cellulose-R1 (2.8e19,
                                 # E=242.4 kJ/mol).  See Rule #1 — any
                                 # non-MD2004 value must be lit-cited in
                                 # the calling deck/worker.
    E_p: float = E_MD2004,      # Phase 15I — thermal R_p activation energy
                                 # override [J/mol].  Default = MD2004 (58600).
) -> None:
    """One pyrolysis step using MD2004 single-pool grass kinetics, with
    explicit char accumulation (Phase 14 Try 5b, Yang 2007).

    In-place update of m_solid (decays to 0 — pre-Layer-B floor removed)
    and m_char (accumulates char from each Δm of pyrolysis).  Mass balance:
      Δm of reactive biomass leaves m_solid each step.
      ETA_MD2004 fraction goes to gas via S_pyro_out (volatile).
      CHAR_YIELD_MD2004 fraction goes to m_char (solid residue).
      ETA_MD2004 + CHAR_YIELD_MD2004 = 1.0 (no inert gas split in MD2004).

    char_ox and smolder kernels consume m_char (not m_solid).  This fixes
    a pre-Layer-B structural issue where char_ox fired on raw biomass.

    **Moisture gate (Phase 14h)**: when local water > ~1% of initial,
    pyrolysis kinetics are suppressed inside the kernel and m_solid is
    NOT decremented (Grishin 1984; Margerit & Séro-Guillaume 2002).

    Soft ramp gate = max(0, 1 - 100·m_water/m_water_init):
      wet ≥ 1%  → gate=0 (skip; m_solid unchanged)
      wet = 0   → gate=1 (full kinetics)
      0 < wet < 1%: linear ramp.

    Pass m_water_init = 0 to disable the gate (dry-fuel case).
    """
    inv_mw_init = 1.0 / m_water_init if m_water_init > 0.0 else 0.0
    Nz, Ny, Nx = T_s.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s[k, j, i] <= 0.0:
                    S_pyro_out[k, j, i] = 0.0
                    Q_pyro_out[k, j, i] = 0.0
                    continue
                T = T_s[k, j, i]
                if T <= 0.0:
                    S_pyro_out[k, j, i] = 0.0
                    Q_pyro_out[k, j, i] = 0.0
                    continue

                # Moisture gate (Grishin 1984): no pyrolysis until water gone.
                if inv_mw_init > 0.0:
                    wet = m_water[k, j, i] * inv_mw_init
                    moist_gate = 1.0 - 100.0 * wet
                    if moist_gate <= 0.0:
                        S_pyro_out[k, j, i] = 0.0
                        Q_pyro_out[k, j, i] = 0.0
                        continue
                else:
                    moist_gate = 1.0

                m = m_solid[k, j, i]
                if m <= 0.0:
                    S_pyro_out[k, j, i] = 0.0
                    Q_pyro_out[k, j, i] = 0.0
                    continue

                # Phase 14 Try 6: parallel R_p (thermal) + R_op (oxidative).
                k_thermal = A_p * math.exp(-E_p / (_R_GAS * T))
                yO2 = Y_O2[k, j, i]
                if yO2 > Y_O2_MIN_OP:
                    k_oxidative = (A_OP_MD2004 *
                                   math.exp(-E_OP_MD2004 / (_R_GAS * T)) *
                                   yO2 ** N_O2_OP)
                else:
                    k_oxidative = 0.0
                k_total = k_thermal + k_oxidative
                m_new = m * math.exp(-k_total * dt)
                dm_full = m - m_new
                # Apply moisture-gate ramp consistently to mass loss & gas source.
                dm = dm_full * moist_gate
                rate = dm / dt if dt > 0 else 0.0
                # Split dm by path share for heat-of-reaction accounting
                # (each path has different HoR: thermal endo, oxidative exo).
                if k_total > 0.0:
                    f_thermal = k_thermal / k_total
                    f_oxidative = k_oxidative / k_total
                else:
                    f_thermal = 1.0
                    f_oxidative = 0.0

                # ETA_MD2004 fraction → volatile gas; CHAR_YIELD_MD2004 → m_char.
                S_pyro_out[k, j, i] = ETA_MD2004 * rate
                Q_pyro_out[k, j, i] = rate * (
                    f_thermal * HEAT_OF_PYROLYSIS    # endothermic R_p
                    + f_oxidative * HOR_OP_MD2004    # exothermic R_op (-)
                )
                m_char[k, j, i] += CHAR_YIELD_MD2004 * dm
                m_solid[k, j, i] = m - dm


# ── Char oxidation (Phase 14y-char) ───────────────────────────────────────────
# C(s) + O2 → CO2 + heat.  Sustained heat source from glowing char layer
# behind the volatile-flame front — provides "hot trailing edge" that
# radiates/convects to the bed-ahead-of-front cells, enabling propagation
# in marginal cases where EDC volatile combustion alone can't sustain.
#
# Standard physics in WFDS (Mell et al. 2007 IJWF 16:1 §2.4) and FIRETEC
# (Linn 2002).  Was missing from our model — bootstrap was silently
# substituting for it for marginal Nat-bed cases (sparse, deep grass).
#
# Arrhenius rate (Boonmee & Quintiere 2005 Combust. Flame 141:283):
#   m_dot_char = A_CHAR · exp(-E_CHAR/RT_s) · m_solid · Y_O2
# Heat release: HOC_CHAR ≈ 32 MJ/kg (carbon → CO2)
# T_CHAR_ONSET ~ 600 K — below this, k_char is negligible
#
# NOTE: m_solid is used as the char inventory (when m_solid > char_limit
# this represents char + unpyrolyzed mixed; when m_solid ≤ char_limit
# only char remains).  Char ox can drive m_solid below the pyrolysis
# floor; pyrolysis kernel naturally stops when m_avail < 0.

A_CHAR        = 7.09e5       # [1/s]   Peterson & Brown (2020) DOE OSTI 1648151
                              # switchgrass biochar TGA in air, Regime I
                              # (333-433°C).  Replaces Mell 2007 generic
                              # vegetation A=1e5 with grass-specific value.
                              # Operates in HIGH-T regime (>600K).  Marginal
                              # cases that need lower-T scaffolding go through
                              # the smoldering kernel (below).
E_CHAR        = 94_000.0     # [J/mol] Peterson & Brown (2020) switchgrass.
                              # Replaces Mell 2007 generic E=75 kJ/mol; lit-
                              # grounded re-tune closes 0D validation gap
                              # (model rate was 6.4× faster than EXP at 600K).
HOC_CHAR      = 32_000_000.0 # [J/kg]  Mell 2007 §2.4 (carbon → CO2)
T_CHAR_ONSET  = 600.0        # [K]     below this, k_char effectively zero
Y_O2_MIN_CHAR = 1.0e-3       # [-]     skip char ox if no oxygen
Q_CHAR_MAX    = 5.0e5        # [W/m³]  abs cap on per-cell char heat release
                              # (mass-transfer limited regime safeguard).
                              # Matches old Q_BOOTSTRAP magnitude (500 kW/m³).


@njit(cache=True, parallel=True)
def step_char_oxidation(
    T_s: np.ndarray,         # (Nz, Ny, Nx) [K]
    m_solid: np.ndarray,     # (Nz, Ny, Nx) [kg/m³] mutated (decreases)
    Y_O2: np.ndarray,        # (Nz, Ny, Nx) [-] gas O2 in same cell (bed cells)
    alpha_s: np.ndarray,     # (Nz, Ny, Nx) [-] mask for bed cells (alpha_s > 0)
    dt: float,
    Q_char_out: np.ndarray,  # (Nz, Ny, Nx) [W/m³] heat release (output)
) -> None:
    """One char oxidation step.  Consumes m_solid, releases Q_char.

    No O2 deduction from gas (assumes plenty of O2 reaches the bed via
    canopy mixing — appropriate at the coarse grid; precise O2 budget
    would require coupling to gas-phase O2 transport which is an
    incremental refinement).

    Heat goes to T_s via Q_pyro_out (added by caller before coupling).
    """
    Nz, Ny, Nx = T_s.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s[k, j, i] <= 0.0:
                    Q_char_out[k, j, i] = 0.0
                    continue
                m = m_solid[k, j, i]
                if m <= 1.0e-6:
                    Q_char_out[k, j, i] = 0.0
                    continue
                T = T_s[k, j, i]
                if T < T_CHAR_ONSET:
                    Q_char_out[k, j, i] = 0.0
                    continue
                yO2 = Y_O2[k, j, i]
                if yO2 < Y_O2_MIN_CHAR:
                    Q_char_out[k, j, i] = 0.0
                    continue
                k_char = A_CHAR * math.exp(-E_CHAR / (_R_GAS * T))
                m_dot = k_char * m * yO2          # [kg/m³/s]
                # Cap heat release at Q_CHAR_MAX (mass-transfer-limited
                # regime safeguard; real char ox at high T is O2-supply
                # limited not Arrhenius limited).
                Q_arrh = m_dot * HOC_CHAR
                if Q_arrh > Q_CHAR_MAX:
                    m_dot = Q_CHAR_MAX / HOC_CHAR
                # Limit consumption per step to avoid overshoot
                m_consumed = m_dot * dt
                if m_consumed > 0.5 * m:
                    m_consumed = 0.5 * m
                m_solid[k, j, i] = m - m_consumed
                Q_char_out[k, j, i] = m_consumed * HOC_CHAR / dt if dt > 0.0 else 0.0


# ── Smoldering combustion (Phase 14z-A1) ──────────────────────────────────────
# Slow exothermic surface oxidation that operates in the LOW-T regime
# (T_s ~ 350-700 K), bridging the gap between drip-torch heating and
# flaming-combustion-onset.  Provides the physically-correct scaffolding
# that bootstrap was approximating, but lit-grounded.
#
# In real grass fires, smoldering combustion of finely-divided cellulosic
# material is observable at T as low as 200-300°C (~470-570 K).  Heat
# release from the slow C(s) + O2 → CO + CO2 reaction (at low T, CO is
# the major product, then transitions to CO2 at higher T).
#
# Lit refs:
# - Ohlemiller, T.J. (1985) "Modeling of smoldering combustion propagation"
#   Combust. Flame 60:33  (kinetic regime)
# - Rein, G. (2014) "Smoldering combustion" Int. Rev. Chem. Eng. 6:25
#   (vegetation smoldering review; A=10²-10⁴, E=80-100 kJ/mol typical)
# - Rein, G. et al. (2008) Catena 74:304  (peat smoldering; A=4.4e7,
#   E=104 kJ/mol — used as upper bound)
# - Boonmee & Quintiere (2005) Combust. Flame 141:283  (wood smoldering)
#
# Distinct from char ox (high-T flaming-regime cap-limited); smoldering
# IS the kinetic regime, has lower E and operates with low-T onset.

A_SMOLD       = 1.0e6        # [1/s]   Rein 2014 vegetation review (10²-10⁸).
                              # 1e8 was too fast — combined w/ char ox the
                              # solid-side heat release ramped ρ field too
                              # fast for the projection.  1e6 gradual ramp:
                              # ~290 W/m³ at T=400K, capped at 200 kW/m³
                              # by T~600K.  Stable + provides scaffolding.
E_SMOLD       = 80_000.0     # [J/mol] Lower than char ox (75-100 kJ/mol typical).
                              # NOTE: known low vs Blunck 2022 / Huang-Rein
                              # multi-step kinetics (cellulose 278, hemi 294,
                              # lignin 289 kJ/mol oxidation E).  Single-step
                              # lumped E is intentionally lower as an effective
                              # rate constant; a future Phase 14 step will
                              # implement the multi-step Huang-Rein scheme.
HOC_SMOLD     = 28_000_000.0 # [J/kg]  C → CO+CO2 mix (lower than complete C → CO2)
T_SMOLD_ONSET = 473.0        # [K]     Blunck et al. 2022 SERDP RC-2651:
                              # IR-camera measurement of grass-blend smolder
                              # surface T spans 200-500°C (473-773 K).
                              # 473 K = chemically-grounded lower bound of
                              # observed smolder.  Was 400 K (Rein 2014
                              # generic vegetation, less-grass-specific).
Y_O2_MIN_SMOLD = 1.0e-3      # [-]     skip if no oxygen
Q_SMOLD_MAX   = 2.0e5        # [W/m³]  modest cap — smoldering is slow
                              # (200 kW/m³ ≈ 40% of bootstrap-era scaffolding,
                              # appropriate "slow surface burn" magnitude)


@njit(cache=True, parallel=True)
def step_smoldering_oxidation(
    T_s: np.ndarray,           # (Nz, Ny, Nx) [K]
    m_solid: np.ndarray,       # (Nz, Ny, Nx) [kg/m³] mutated (decreases)
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-] gas O2 in same cell
    alpha_s: np.ndarray,       # (Nz, Ny, Nx) [-] mask for bed cells
    dt: float,
    Q_smold_out: np.ndarray,   # (Nz, Ny, Nx) [W/m³] heat release (output)
) -> None:
    """One smoldering oxidation step.  Slow surface oxidation operative at
    T > T_SMOLD_ONSET ~ 400K — well below char-ox 600K threshold.

    Same Arrhenius form as char ox but with lower E (faster rate at lower
    T) and lower Q_max (slow regime).  Acts on m_solid (treats whatever
    bulk mass is present as oxidizable; in the smoldering regime, the
    bed surface is what oxidizes, slowly).
    """
    Nz, Ny, Nx = T_s.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if alpha_s[k, j, i] <= 0.0:
                    Q_smold_out[k, j, i] = 0.0
                    continue
                m = m_solid[k, j, i]
                if m <= 1.0e-6:
                    Q_smold_out[k, j, i] = 0.0
                    continue
                T = T_s[k, j, i]
                if T < T_SMOLD_ONSET:
                    Q_smold_out[k, j, i] = 0.0
                    continue
                yO2 = Y_O2[k, j, i]
                if yO2 < Y_O2_MIN_SMOLD:
                    Q_smold_out[k, j, i] = 0.0
                    continue
                k_smold = A_SMOLD * math.exp(-E_SMOLD / (_R_GAS * T))
                m_dot = k_smold * m * yO2          # [kg/m³/s]
                # Cap heat release in smoldering "diffusion-limited" regime
                Q_arrh = m_dot * HOC_SMOLD
                if Q_arrh > Q_SMOLD_MAX:
                    m_dot = Q_SMOLD_MAX / HOC_SMOLD
                # Per-step consumption cap
                m_consumed = m_dot * dt
                if m_consumed > 0.5 * m:
                    m_consumed = 0.5 * m
                m_solid[k, j, i] = m - m_consumed
                Q_smold_out[k, j, i] = m_consumed * HOC_SMOLD / dt if dt > 0.0 else 0.0
