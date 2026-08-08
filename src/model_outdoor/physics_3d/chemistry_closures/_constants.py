"""Physical and stoichiometric constants shared across chemistry closures.

Closure-specific tuning constants (e.g., Magnussen 1981 EDC fine-structure
C_GAMMA_EDC, C_TAU_EDC) live in their respective closure module.

See combustion_3d.py module docstring for the full literature trail
(Susott 1980, Magnussen-Hjertager 1977, Westbrook-Dryer 1981,
Spalding 1971, Pruyn 2018, Grishin-Perminov 2002, Drysdale 2011).
"""

# Stoichiometric mass ratio kg O₂ / kg fuel for grass-pyrolysis volatile
# (CH_{1.4}O_{0.6}-like, Susott 1980).  Phase 14w-D fix: was 4.0 (CH₄-
# equivalence) but HoC is for grass biomass, not methane.  s=1.3 restores
# correct closed-reactor T_ad ≈ 1850 K matching lit grass flame ~1500-1800 K.
S_STOICH = 1.3

# Mass fraction of O₂ in fresh air.
Y_O2_AIR = 0.232

# Westbrook & Dryer / Morvan & Dupuy single-step Arrhenius kinetics
# (FDS wood-volatile one-step).
A_COMB = 1.0e9       # [1/s]
E_COMB = 84_000.0    # [J/mol]
_R_GAS = 8.314       # [J/mol/K]

# EBU constant (Phase 14w-B retained at Spalding 1971 default).
# Lit-canonical M&H 1977 value is 4.0; we run at 1.0 because at our
# 10 cm bed dx a 4× higher source-cell ω drives ∂ρ/∂t spikes the
# variable-density Chorin projection cannot resolve.  See Phase 14r
# memory for the RANS+k-ε+EBU structural-limit discussion.
C_EBU = 1.0

# Raw chemical heat of combustion (Susott 1980 grass biomass).
# Apply (1−χ_rad) explicitly downstream so the radiation budget is
# tracked once (by DOM/P1), not double-counted via an effective HoC.
HOC_J = 17_000_000.0   # [J/kg]


# ── Phase 23: chemistry-family presets ─────────────────────────────────────
# The five constants above (S_STOICH, HOC_J, A_COMB, E_COMB, C_EBU) are
# grass-biomass volatile defaults, hard-wired at module import.  For
# non-biomass fuels (methane cup burner, etc.) the closure kernels now
# accept these five values as explicit arguments; the caller resolves
# them from a family name via ``resolve_chemistry_family``.  Grass
# behaviour is bit-exact preserved because the biomass preset returns
# exactly the same numbers as the top-of-file module constants.

CHEMISTRY_FAMILIES = {
    # Susott 1980 grass biomass + Westbrook-Dryer 1981 wood single-step.
    # Match the module constants above exactly (Rule #17 bit-exact
    # invariant for all pre-Phase-23 validation cases).
    "biomass": dict(
        s_stoich=1.3,
        hoc_J=17_000_000.0,
        a_comb=1.0e9,
        e_comb=84_000.0,
        c_ebu=1.0,
    ),
    # Westbrook-Dryer 1981 single-step CH4 oxidation.  HoC = LHV of
    # methane (NIST WebBook).  C_EBU = M&H 1977 canonical for turbulent
    # diffusion flames.  Used by the Phase 23 cup burner validation.
    "methane": dict(
        s_stoich=4.0,           # CH4 + 2 O2 → CO2 + 2 H2O
        hoc_J=50_000_000.0,     # LHV
        a_comb=2.1e11,          # Westbrook & Dryer 1981
        e_comb=125_000.0,       # J/mol
        c_ebu=4.0,              # Magnussen-Hjertager 1977 turbulent diffusion
    ),
}


def resolve_chemistry_family(family: str) -> dict:
    """Return the 5-scalar chemistry-preset dict for ``family``.

    Raises ValueError on unknown family.  Kernel callers unpack as
    ``**resolve_chemistry_family("biomass")`` or pass individually.
    """
    if family not in CHEMISTRY_FAMILIES:
        raise ValueError(
            f"Unknown chemistry family '{family}'.  "
            f"Registered: {sorted(CHEMISTRY_FAMILIES.keys())}"
        )
    return dict(CHEMISTRY_FAMILIES[family])   # copy to prevent mutation
