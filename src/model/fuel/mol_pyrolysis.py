"""
Method-of-Lines (MoL) 1D pyrolysis solver.

Implementation follows Lautenberger & Fernandez-Pello (2009), "Generalized pyrolysis
model for combustible solids," Fire Safety Journal 44(8): 819-839.

State vector
------------
    y = [T_0, ..., T_{N-1},
         rho_{0,0}, ..., rho_{0,N-1},    # bulk density of species 0 [kg/m³]
         rho_{1,0}, ..., rho_{1,N-1},    # bulk density of species 1
         ...]
Total states: N × (1 + M), where M = number of condensed-phase species.

Species densities rho_{s,i} are bulk densities in the ORIGINAL (fixed) cell volume
[kg/m³].  As species decompose, gas escapes and the cell becomes more porous;
solid densities decrease accordingly.

Reactions (Lautenberger 2009 Eq. 3.31–3.32)
---------------------------------------------
    r_k = Z_k × exp(-E_k / (R T_i)) × max(rho_{from_k, i}, 0)^{n_k}   [kg/m³/s]

    Species update per cell i:
        d rho_{from_k,i}/dt  -= r_k
        d rho_{to_k,i}  /dt  += r_k × (1 - nu_gas_k)

    Endothermic/exothermic source per cell i:
        q_py_i = sum_k  dH_vol_k × r_k × nu_gas_k           [W/m³]
              (positive dH_vol = endothermic; same sign convention as Lautenberger)

    Mass flux of gas produced per cell i:
        omega_gas_i = sum_k r_k × nu_gas_k                   [kg_gas/m³/s]
    Total surface mass flux:
        m_dot = sum_i omega_gas_i × dx                       [kg/m²/s]

Thermal properties (Lautenberger 2009 Table 6.4 power-law form)
-----------------------------------------------------------------
    k_s(T)  = k0_s × (T / T_ref)^{nk_s}   +  4 σ γ_s T³    (in-pore radiation)
    cp_s(T) = cp0_s × (T / T_ref)^{nc_s}

    Mixing per cell (density-weighted, Lautenberger §3.2):
        k_i    = sum_s [rho_{s,i} / rho_total_i] × k_s(T_i)
        (rho cp)_i = sum_s rho_{s,i} × cp_s(T_i)

Surface emissivity is taken from the dominant (highest-mass-fraction) surface species.

References
----------
Lautenberger, C. & Fernandez-Pello, C. (2009). Fire Safety Journal 44(8): 819-839.
Kung, H.C. (1972). Combust. Flame 18: 185-195.   (cellulose kinetics)
Drysdale, D. (2011). Introduction to Fire Dynamics, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import copy as _copy

from scipy.integrate import solve_ivp

_R_GAS: float = 8.314            # J/mol/K
_SIGMA: float = 5.670374419e-8   # Stefan-Boltzmann [W/m²/K⁴]
_T_REF: float = 293.0            # reference temperature for power-law props [K]


# ── Grid builder ───────────────────────────────────────────────────────────────

def _build_grid(L: float, N: int, stretch: float) -> np.ndarray:
    """Cell-width array with geometric surface refinement.

    Cell widths increase geometrically from surface to back face:
        dx[i] = dx_0 × stretch^i,  sum(dx) = L

    Parameters
    ----------
    L : float
        Slab thickness [m].
    N : int
        Number of cells.
    stretch : float
        Geometric ratio.  stretch > 1 refines toward the surface.
        stretch = 1.0 gives a uniform grid (dx = L/N for all cells).

    Returns
    -------
    dx_arr : ndarray, shape (N,)
        Cell widths from surface (index 0) to back face (index N-1).
    """
    if abs(stretch - 1.0) < 1e-10:
        return np.full(N, L / N)
    r = float(stretch)
    dx_0 = L * (1.0 - r) / (1.0 - r ** N)
    return dx_0 * r ** np.arange(N)


def _build_jac_sparsity(N: int, M: int) -> np.ndarray:
    """Sparsity pattern for the N-cell M-species MoL Jacobian.

    State ordering: [T_0..T_{N-1}, rho_{0,0}..rho_{0,N-1}, ..., rho_{M-1,0}..rho_{M-1,N-1}]

    Nonzero structure:
      T-T         : tridiagonal (conduction couples adjacent cells)
      T ↔ rho_s   : diagonal (k_loc, rho_cp, reaction source are cell-local)
      rho_s ↔ rho_s': diagonal per cell (kinetics are cell-local)

    Passed to scipy solve_ivp(jac_sparsity=...) for CPR column grouping, reducing
    finite-difference Jacobian evaluations from N(1+M) → ~3(1+M) groups.
    For N=40, M=3: 160 → ~12 RHS evaluations per Jacobian update (~13× speedup).
    """
    n = N * (1 + M)
    S = np.zeros((n, n), dtype=np.int8)
    # T-T: tridiagonal (conduction couples adjacent cells)
    for i in range(N):
        for j in range(max(0, i - 1), min(N, i + 2)):
            S[i, j] = 1
    # T <-> rho: diagonal (same cell only — k, rho_cp, reaction rates are local)
    for s in range(M):
        for i in range(N):
            S[i, N + s * N + i] = 1          # dT_i/drho_{s,i}
            S[N + s * N + i, i] = 1          # drho_{s,i}/dT_i
    # rho-rho: same cell, all species pairs (kinetics are cell-local)
    for s in range(M):
        for sp in range(M):
            for i in range(N):
                S[N + s * N + i, N + sp * N + i] = 1
    return S


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class MolSolidSpecies:
    """Condensed-phase material species (Lautenberger 2009 Table 6.4 format).

    Properties follow power-law temperature dependence:
        k(T)  = k0  × (T / T_ref)^nk  +  4 σ γ T³   [W/m/K]
        cp(T) = cp0 × (T / T_ref)^nc                 [J/kg/K]

    Attributes
    ----------
    name : str
    k0 : float
        Thermal conductivity at T_ref [W/m/K].
    nk : float
        Temperature exponent for k (0 = constant).
    rho0 : float
        Initial bulk density [kg/m³].  For product species (e.g. char) that start
        at zero and are produced by reactions, set rho0 = 0.
    cp0 : float
        Specific heat at T_ref [J/kg/K].
    nc : float
        Temperature exponent for cp (0 = constant).
    eps : float
        Surface emissivity [-] (used only for the surface cell, weighted by mass
        fraction).
    gamma : float
        In-pore radiation length scale [m].  0 = disabled.
        Adds radiant conductivity k_rad = 4 σ γ T³ (Lautenberger §3.2).
    """

    name: str = "species"
    k0: float = 0.12
    nk: float = 0.0
    rho0: float = 400.0
    cp0: float = 1500.0
    nc: float = 0.0
    eps: float = 0.87
    gamma: float = 0.0

    def k_at(self, T: np.ndarray) -> np.ndarray:
        """Thermal conductivity [W/m/K] at temperature T (array)."""
        T_safe = np.maximum(T, 1.0)
        k = self.k0 * (T_safe / _T_REF) ** self.nk
        if self.gamma > 0.0:
            k = k + 4.0 * _SIGMA * self.eps * self.gamma * T_safe ** 3
        return k

    def cp_at(self, T: np.ndarray) -> np.ndarray:
        """Specific heat [J/kg/K] at temperature T (array)."""
        T_safe = np.maximum(T, 1.0)
        return self.cp0 * (T_safe / _T_REF) ** self.nc


@dataclass
class MolReaction:
    """Condensed-phase kinetic reaction (Lautenberger 2009 Eq. 3.31).

        reactant[from_idx]  →  (1-nu_gas) × product[to_idx]  +  nu_gas × gas

    Rate:  r = Z × exp(-E / (R T)) × max(rho_from, 0)^n   [kg/m³/s]

    Attributes
    ----------
    from_idx : int
        Index into MolParams.species for the reactant species.
    to_idx : int
        Index into MolParams.species for the solid product (e.g. char).
        Ignored when nu_gas = 1.0 (all mass goes to gas).
    Z : float
        Pre-exponential factor [1/s] (for n=1; units generalise for n ≠ 1).
    E : float
        Activation energy [J/mol].
    n : float
        Reaction order with respect to reactant density (Lautenberger: n=5.42
        for wood charring; n=1.0 for simple first-order).
    dH_vol : float
        Heat of volatilisation [J/kg of gas produced].  Positive = endothermic.
        Lautenberger: dH_vol = 6.74e5 J/kg (anaerobic wood charring); 2.41e6
        J/kg (water vaporisation).
    nu_gas : float
        Fraction of reactant mass that becomes gas (0–1).
        For charring wood:  nu_gas = (rho_virgin - rho_char) / rho_virgin.
        For drying:          nu_gas = (rho_wet - rho_dry)   / rho_wet.
        For full gasification (PMMA): nu_gas = 1.0.
    nO2 : float
        Oxygen reaction order (0 = anaerobic).  Not currently coupled to an O2
        transport equation; set nO2 > 0 only if explicitly handling gas-phase
        oxygen (out of scope for basic cone simulations).
    dH_sol : float
        Heat of solid-product formation [J/kg of solid product formed].
        Usually 0; use for reactions with significant condensed-phase enthalpy
        change (e.g. PMMA bubbling reaction has dH_sol = 4580 J/kg).
    """

    from_idx: int = 0
    to_idx: int = 1
    Z: float = 1.0e10
    E: float = 1.62e5
    n: float = 1.0
    dH_vol: float = 0.0
    nu_gas: float = 0.0
    nO2: float = 0.0
    dH_sol: float = 0.0
    # GPYRO-style density normalization: r = Z × exp(-E/RT) × (ρ/rho_ref)^(n-1) × ρ
    # Equivalent to r = Z × exp(-E/RT) × ρ^n / rho_ref^(n-1).
    # Default 1.0 → absolute density (backward compatible for n=1 cases).
    # Set to the FROM species reference/initial density for n>1 Lautenberger params.
    rho_ref: float = 1.0
    # Sigmoid temperature gate: S = 1/(1+exp(-(T-T_gate_K)/dT_gate_K))
    # Applied as r_eff = S × r_Arrhenius.  T_gate_K=0.0 → gate disabled (pure Arrhenius).
    # Physical basis: TGA-measurable charring onset temperature (Orfão 1999: ~450-550°C for wood).
    T_gate_K: float = 0.0    # gate midpoint [K]; 0 = disabled
    dT_gate_K: float = 25.0  # gate half-width [K]; larger = broader transition


@dataclass
class MolParams:
    """Parameters for the MoL 1D pyrolysis solver (Lautenberger 2009 formulation)."""

    # Geometry
    L: float              # slab thickness [m]
    N: int = 20           # number of uniform cells

    # Condensed-phase species list (MolSolidSpecies)
    species: list = field(default_factory=list)

    # Kinetic reactions list (MolReaction)
    reactions: list = field(default_factory=list)

    # Ambient / boundary conditions
    Tamb: float = 300.0    # ambient temperature [K]
    T_sur: float = 300.0   # surroundings temp for radiation [K]
    h_conv: float = 15.0   # surface convective HTC [W/m²/K]

    # Back-face BC
    back_bc: str = "adiabatic"  # "adiabatic" or "open"
    h_back: float = 10.0        # [W/m²/K] if open
    eps_back: float = 0.87      # back-face emissivity if open

    # Grid — geometric stretching toward surface (stretch > 1 refines surface)
    grid_stretch: float = 1.0   # geometric ratio; 1.0 = uniform grid
    k_crack_frac: float = 0.0   # char cracking conductivity factor (Shi & Chew 2023)
                                 # k_eff_i = k_i × (1 + k_crack_frac × delta_conv_i / L)

    # Surface recession: delete fully-depleted surface cells (for non-charring PMMA)
    surface_recession_enable: bool = False
    surface_rho_floor: float = 0.01   # delete cell when rho_total < floor × sum(rho0_s)

    # In-depth radiation absorption (Beer-Lambert for porous fuel beds)
    # κ = 0: surface-opaque (default, backward compatible with all existing cases)
    # κ > 0: incident irradiance distributed as I(x) = I₀·exp(−κx) [Albini 1985]
    in_depth_rad_kappa: float = 0.0   # extinction coefficient [m⁻¹]

    # Density-weighted Beer-Lambert (T9b variant):
    # κ_local,i = κ₀ × (ρ_reactive,i / ρ_reactive,0)
    # As cells deplete/char, their optical extinction falls proportionally →
    # radiation advances into unburned interior rather than heating depleted cells.
    # Physical basis: optical extinction ∝ remaining absorbing solid (Beer 1852).
    # False (default): fixed κ everywhere (original T9 behaviour).
    in_depth_rad_density_weighted: bool = False

    # Char spalling: phenomenological char detachment (Shi & Chew 2023; Drysdale 2011)
    spall_enable: bool = False
    spall_depth_m: float = 0.003    # char conversion depth threshold [m]
    spall_rho_floor: float = 0.05   # residual char fraction after detachment

    # Gas-phase convective energy transport (GPYRO eq. −∂/∂z(ρ_g c_g v_g T))
    cp_gas: float = 0.0   # volatile gas specific heat [J/kg/K]; 0 = disabled
                           # GPYRO paper uses 1100 J/kg/K (propane surrogate)

    # Surface O₂ mass fraction for oxidative reactions (nO2 > 0)
    # Default 0.21 = ambient air. Set 0.0 to disable all O2-dependent reactions.
    # Constant-Y_O2 approximation: no gas-phase transport equation needed for
    # cone calorimeter conditions (well-mixed ambient, moderate blowing).
    y_o2_surf: float = 0.21

    # Material-coordinate moving mesh (Lagrangian)
    # Each cell tracks a fixed initial mass Δm_i = ρ_total0 × dx_i.
    # Physical thickness updates dynamically: dx_i(t) = Δm_i / ρ_total_i(t).
    # As bpmma reacts to gas and ρ_total decreases, the depleted cell EXPANDS,
    # creating a large thermal resistance that protects interior cells from
    # pre-heating.  Required for non-charring receding polymers (PMMA).
    # Backward compatible: char-forming decks leave False → dx_i = dx_arr (fixed).
    material_coords: bool = False

    # Bed collapse model — loose fibrous fuel beds (dry grass, straw)
    # Physical basis: as fuel burns, the char/ash layer loses structural support and
    # remaining unburned fuel settles toward the heat source. The inter-cell conduction
    # distances shrink proportionally: d_half(t) = d_half_0 × β(t) where
    #   β = m_solid_total(t) / m_solid_0   (volume-fraction collapse, alpha=1).
    # This increases inter-cell thermal conductance without changing heat capacity per
    # unit area (same mass in compressed space: (ρ/β) × cp × (dx×β) = ρ × cp × dx).
    # Backward compatible: False (default) → identical behavior to existing decks.
    bed_collapse_enable: bool = False

    # Surface ablation BC (non-charring receding polymers, e.g. PMMA)
    # When T_surf >= surface_ablation_T_min, cell-0 Arrhenius kinetics are replaced
    # by an energy-balance formula:
    #   m_dot [kg/m²/s] = max(q_in − q_conv − q_rad − J[0], 0) / surface_ablation_L_py
    # This forces dT[0]/dt → 0 (surface T locks at T_min) and gives quasi-steady MLR.
    # Physical basis: PMMA quasi-steady ablation at ~600 K (Kashiwagi & Nambu 1992).
    # Arrhenius kinetics fail for PMMA due to sub-cell reaction zone (~47 nm << dx_0).
    # L_py: effective heat of gasification incl. sensible heat + endotherm + conduction.
    #   Literature hg = 2.23 MJ/kg (Tewarson 1995 SFPE); effective ≈ 3.5 MJ/kg for
    #   10 mm cone slab accounting for interior conduction per unit mass ablated.
    # Backward compatible: surface_ablation_bc=False → normal Arrhenius throughout.
    surface_ablation_bc: bool = False
    surface_ablation_L_py: float = 3.5e6   # effective heat of gasification [J/kg]
    surface_ablation_T_min: float = 600.0  # activation temperature [K]
    surface_ablation_active: bool = False   # one-way latch: True once T≥T_min first time;
    # prevents dips when cell deletion exposes a new cell-0 with T slightly below T_min.

    # ODE solver settings
    method: str = "Radau"
    rtol: float = 1e-4
    atol: float = 1e-6
    max_step: float = 0.5   # internal adaptive step ceiling [s]
    dt_eval: float = 0.5    # uniform output grid spacing [s]; also flame-coupling interval

    # Surface-cell Lagrangian replenishment (non-charring receding polymers e.g. PMMA)
    # Adds v_s×ρ[s,1]/dx_0 advective flux INTO cell 0, balancing ablation depletion.
    # Keeps ρ[0] ≈ ρ₀ → no recession cycling → smooth quasi-steady m_dot.
    # Requires surface_ablation_bc=true. Backward compatible: False → no change.
    lagrangian_mode: bool = False

    # Stefan front BC for charring materials (wood, PB — NOT for non-charring polymers).
    # Replaces Arrhenius kinetics for the charring reaction with an energy-balance front
    # velocity at the T_py isotherm.  Concentrates pyrolysis at a mathematical interface
    # rather than spreading it over ~1 cell (Arrhenius δ_rxn ≈ 133 µm < dx[0] = 370 µm).
    # Drying reaction (Rxn 0) stays as Arrhenius — distributed and well-resolved.
    # Backward compatible: False → Arrhenius unchanged for all existing decks.
    charring_front_bc: bool = False
    charring_T_py: float = 600.0   # pyrolysis front temperature [K]; Drysdale 2011 for wood

    # Precomputed fields — built by __post_init__, do not set manually
    dx_arr: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    dm_arr: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # ^ dm_arr[i] = ρ_total0 × dx_arr[i]  — initial solid mass per unit area per cell [kg/m²]
    #   Used by material_coords mode to compute dynamic dx_i = dm_arr[i] / ρ_total_i(t)
    rho0_total: float = field(default=0.0, init=False, repr=False)
    # ^ sum of initial species densities; used for v_s = omega_gas[0]×dx_0/ρ₀_total
    m_solid_0: float = field(default=0.0, init=False, repr=False)
    # ^ total initial solid mass per unit area [kg/m²] = rho0_total × L
    #   Used by bed_collapse_enable to compute β = m_solid_total(t) / m_solid_0

    def __post_init__(self) -> None:
        self.dx_arr = _build_grid(self.L, self.N, self.grid_stretch)
        rho0 = float(sum(max(sp.rho0, 0.0) for sp in self.species))
        self.dm_arr = self.dx_arr * rho0
        self.rho0_total = rho0
        self.m_solid_0 = rho0 * self.L


@dataclass
class MolResult:
    """Output from integrate_mol()."""

    t: np.ndarray          # time grid [s], shape (n_t,)
    T: np.ndarray          # temperature [K], shape (N, n_t); T[0]=surface
    rho: np.ndarray        # species densities [kg/m³], shape (M, N, n_t)
    m_dot: np.ndarray      # total volatile mass flux [kg/m²/s], shape (n_t,)
    T_surf: np.ndarray     # surface temperature [K], shape (n_t,)
    hrrpua_kW: np.ndarray  # HRRPUA [kW/m²], shape (n_t,)

    @property
    def alpha(self) -> np.ndarray:
        """Conversion fraction per species per cell, shape (M, N, n_t).

        alpha[s,i,t] = 1 - rho[s,i,t] / rho0[s]  for reactant species (rho0>0).
        Clamped to [0, 1].
        """
        M = self.rho.shape[0]
        a = np.zeros_like(self.rho)
        for s in range(M):
            rho0_s = self.rho[s, :, 0:1]  # initial density, shape (N,1)
            mask = rho0_s > 1e-6
            a[s] = np.where(mask, np.clip(1.0 - self.rho[s] / np.maximum(rho0_s, 1e-6), 0.0, 1.0), 0.0)
        return a


# ── Core ODE RHS ───────────────────────────────────────────────────────────────

def _mol_rhs(
    t: float,
    y: np.ndarray,
    params: MolParams,
    q_surface_fn: Callable[[float], float],
) -> np.ndarray:
    """ODE right-hand side for the N-cell M-species 1D pyrolysis model.

    Implements Lautenberger & Fernandez-Pello (2009) §3:
      - Cell-centred finite-volume energy equation with harmonic-mean interface k
      - Species conservation with arbitrary-order kinetics
      - Endothermic/exothermic heat sources per cell
      - Surface BC: incident + flame - convective loss - radiative loss
      - Back BC: adiabatic or open (convective + radiative loss)
    """
    N = params.N
    M = len(params.species)

    # Unpack state
    T = y[:N]                              # (N,) temperatures
    rho_all = y[N:].reshape(M, N)         # (M, N) species densities


    T_safe = np.maximum(T, 200.0)

    # ── Local total density and thermal properties ────────────────────────────
    rho_total = np.sum(rho_all, axis=0)                    # (N,)
    rho_total_safe = np.maximum(rho_total, 1e-6)

    # Volume-fraction reference density for each species:
    #   Virgin species (rho0 > 0): use rho0.
    #   Product species (rho0 = 0, e.g. dry wood, char): infer from reaction
    #     stoichiometry so that rho_vf_basis = max achievable bulk density.
    #   This gives the GPYRO volume-fraction k mixing rule:
    #     k = Σ_s (ρ_s/ρ_s,ref) × k_s  /  Σ_s (ρ_s/ρ_s,ref)
    #   Physical basis: char expands 5× by volume relative to wood; volume fraction
    #   gives equal weight to a half-converted cell (50% char by mass ≈ 83% char by
    #   volume), reducing effective k vs the mass-fraction rule and slowing the
    #   thermal wave — consistent with GPYRO behaviour.
    rho_vf_basis = np.array([max(sp.rho0, 0.0) for sp in params.species])
    for rxn in params.reactions:
        if rho_vf_basis[rxn.to_idx] < 1e-6:
            from_basis = max(rho_vf_basis[rxn.from_idx], rxn.rho_ref)
            if from_basis > 1e-6:
                rho_vf_basis[rxn.to_idx] = from_basis * (1.0 - rxn.nu_gas)

    # k mixing by volume fraction; rho*cp by mass (correct for energy storage)
    # Clamp each species density to ≥ 0 for thermal property calculations.
    # The ODE solver (Radau) may probe slightly-negative rho during step rejection;
    # allowing negative rho_cp causes floor(rho_cp, 1.0) to kick in, which makes
    # dT/dt ∝ 1e9 K/s and drives the solver to diverge.
    rho_all_nn = np.maximum(rho_all, 0.0)   # non-negative density for thermo props
    k_loc  = np.zeros(N)
    vol_wt = np.zeros(N)
    rho_cp = np.zeros(N)
    for s, sp in enumerate(params.species):
        if rho_vf_basis[s] > 1e-6:
            vf_s = rho_all_nn[s] / rho_vf_basis[s]     # volume-fraction proxy
        else:
            vf_s = rho_all_nn[s] / rho_total_safe       # fallback: mass fraction
        k_loc  += vf_s * sp.k_at(T_safe)
        vol_wt += vf_s
        rho_cp += rho_all_nn[s] * sp.cp_at(T_safe)     # [J/m³/K], mass-based

    k_loc = k_loc / np.maximum(vol_wt, 1e-12)

    # Guard rho_cp against near-zero (fully depleted cell)
    rho_cp = np.maximum(rho_cp, 1.0)

    # Surface emissivity: dominant-species value at surface cell
    w_surf = rho_all[:, 0] / rho_total_safe[0]
    eps_surf = float(np.dot(w_surf, [sp.eps for sp in params.species]))

    # ── Char cracking: enhance k_loc by cumulative converted depth ───────────
    # Uses depletion of the FIRST (virgin) species as conversion indicator.
    # This gives geometric depth consistent with the 3-node Stefan front formula:
    #   k_eff = k × (1 + k_crack_frac × delta_front / L)
    # where delta_front is the depth at which virgin material has been thermally
    # processed (dried / charred), matching the 3-node k_crack behaviour.
    dx_arr = params.dx_arr  # (N,) initial cell widths (static reference grid)

    # ── Material-coordinate moving mesh ──────────────────────────────────────
    # In material coordinates, each cell tracks a fixed initial mass Δm_i.
    # Physical thickness: dx_i(t) = Δm_i / ρ_total_i(t).
    # As solid reacts to gas, ρ_total drops and depleted cells EXPAND in physical
    # space, creating a thermal resistance that shields interior fresh material.
    # Backward compatible: material_coords=False → dx_dyn = dx_arr (no change).
    if params.material_coords:
        rho_total_mc = np.maximum(rho_total_safe, 1e-3)
        dx_dyn = params.dm_arr / rho_total_mc          # (N,) dynamic cell widths [m]
    else:
        dx_dyn = dx_arr

    # ── Bed collapse: scale inter-cell conduction distances ───────────────────
    # As loose fibrous fuel burns, remaining solid settles toward heat source:
    #   h(t) = h₀ × β(t)  where β = m_solid(t) / m_solid_0  (mass conservation)
    # → inter-cell distances shrink by β → higher conduction to interior cells.
    # Only conduction distances (d_half) are scaled — NOT dx_dyn for heat capacity:
    # heat capacity per unit area is conserved (ρ_phys × cp × dx_phys = ρ × cp × dx).
    if params.bed_collapse_enable:
        _m_solid_now = float((rho_total_safe * dx_arr).sum())
        beta_collapse = max(_m_solid_now / max(params.m_solid_0, 1e-9), 0.05)
    else:
        beta_collapse = 1.0

    if params.k_crack_frac > 0.0:
        sp0_rho0 = params.species[0].rho0
        if sp0_rho0 > 1e-6:
            conv_frac = np.clip(1.0 - rho_all[0] / sp0_rho0, 0.0, 1.0)
        else:
            conv_frac = np.zeros(N)
        # Cumulative geometric depth of processed material from surface
        delta_conv = np.cumsum(conv_frac * dx_arr)
        k_loc = k_loc * (1.0 + params.k_crack_frac * delta_conv / params.L)

    # ── Interface conductivities (harmonic mean) ──────────────────────────────
    k_sum  = k_loc[:-1] + k_loc[1:]
    k_sum  = np.maximum(k_sum, 1e-12)
    k_half = 2.0 * k_loc[:-1] * k_loc[1:] / k_sum         # (N-1,)

    # ── Surface ablation BC pre-computation ──────────────────────────────────
    # Must run BEFORE reaction loop so we can later override cell-0 kinetics.
    # k_half[0] and dx_dyn are available here; q_surface_fn(t) called once.
    _gas_rate_0_abl = 0.0   # [kg_gas/m³/s] ablation rate at cell 0; 0 = inactive
    if params.surface_ablation_bc and N > 1 and T[0] >= params.surface_ablation_T_min:
        _q_in_abl    = float(q_surface_fn(t))
        _q_conv_abl  = params.h_conv * (T[0] - params.Tamb)
        _q_rad_abl   = eps_surf * _SIGMA * (T[0] ** 4 - params.T_sur ** 4)
        # Correct Eulerian surface energy balance: subtract J[0] (heat conducted
        # into interior).  J[0] = k*(T[0]-T[1])/d_half[0] > 0 when surface is
        # hotter than interior; this energy is NOT available for ablation.
        # Clamped to max(J,0): inward conduction (J<0) does not add to ablation.
        # Physical basis: Kashiwagi & Nambu (1992) surface energy balance.
        _d_half_0 = (dx_dyn[0] + dx_dyn[1]) / 2.0
        _J_0 = float(k_half[0] * (T[0] - T[1]) / _d_half_0)
        _q_avail_abl = max(_q_in_abl - _q_conv_abl - _q_rad_abl - max(_J_0, 0.0), 0.0)
        _gas_rate_0_abl = _q_avail_abl / (params.surface_ablation_L_py * dx_dyn[0])

    # ── Reaction rates and source terms ──────────────────────────────────────
    drho_dt = np.zeros((M, N))
    q_py    = np.zeros(N)     # endothermic source [W/m³], positive = energy absorbed
    omega_gas = np.zeros(N)   # gas production rate [kg_gas/m³/s]

    # Stefan front BC: identify the charring reaction (last reaction with solid product
    # and positive dH_vol).  Locate the front cell and decide whether Stefan is active
    # BEFORE the Arrhenius loop so the skip decision is consistent.
    #
    # Physical correction (net-flux Stefan condition):
    #   v_front = q_net / (ρ_from × dH_eff)
    #   q_net   = q_in − q_out  (net heat flux at front cell, NOT one-sided incoming)
    #   dH_eff  = dH_vol × nu_gas  [J/kg material consumed]
    # The one-sided incoming flux overestimates v_front by ~100×, causing ODE failure.
    # The net flux is small (~1-5 kW/m²) because most incoming heat heats cold wood.
    _charring_rxn_idx = -1
    _charring_from_idx = -1
    _i_fr_stefan = -1          # front cell index for Stefan BC
    _stefan_active = False     # True when Stefan BC will replace Arrhenius this call

    if params.charring_front_bc:
        for _ci, _rxn_c in enumerate(params.reactions):
            if _rxn_c.to_idx != _rxn_c.from_idx and _rxn_c.dH_vol > 0.0:
                _charring_rxn_idx = _ci   # last qualifying reaction wins

        if _charring_rxn_idx >= 0:
            _charring_from_idx = params.reactions[_charring_rxn_idx].from_idx
            _rho_from_pre = rho_all[_charring_from_idx]
            _rho_max_pre  = max(float(np.max(_rho_from_pre)), 1e-6)
            _rho_thr_pre  = 1e-2 * _rho_max_pre

            for _ci in range(N):
                if T_safe[_ci] < params.charring_T_py and _rho_from_pre[_ci] > _rho_thr_pre:
                    _i_fr_stefan = _ci
                    break

            # Stefan is active only when the front cell is interior (needs left AND right
            # neighbor) and has ≥ 10% of the current maximum from-species density.
            if _i_fr_stefan >= 1 and _i_fr_stefan < N - 1:
                _rho_at_fr_pre = float(_rho_from_pre[_i_fr_stefan])
                _stefan_active = _rho_at_fr_pre >= 0.10 * _rho_max_pre

    for _ri, rxn in enumerate(params.reactions):
        # Skip ALL charring reactions on the same from-species when Stefan front BC
        # is active this call.  For materials with multiple charring reactions (e.g.
        # White Pine cellulose + hemicellulose), both Arrhenius contributions are
        # replaced by a single Stefan energy-balance BC.
        # When Stefan is NOT active (front not established), fall back to Arrhenius.
        if (_stefan_active
                and rxn.from_idx == _charring_from_idx
                and rxn.to_idx != rxn.from_idx
                and rxn.dH_vol > 0.0):
            continue
        rho_from = np.maximum(rho_all[rxn.from_idx], 0.0)
        # Arrhenius rate [kg/m³/s] — GPYRO normalization:
        #   r = Z × exp(-E/RT) × ρ^n / rho_ref^(n-1)
        # rho_ref=1.0 (default) → absolute density (n=1 cases unchanged).
        # Set rho_ref = FROM-species reference density for n>1 Lautenberger params.
        k_rxn = rxn.Z * np.exp(-rxn.E / (_R_GAS * T_safe))
        if rxn.n == 1.0:
            r = k_rxn * rho_from
        elif rxn.rho_ref > 1.0:
            r = k_rxn * rho_from ** rxn.n / rxn.rho_ref ** (rxn.n - 1.0)
        else:
            r = k_rxn * rho_from ** rxn.n

        # Optional sigmoid temperature gate: S = 1/(1+exp(-(T-T_gate)/dT_gate))
        # Physically: smooth onset temp for charring induction period (TGA-observable).
        # Enabled when T_gate_K > 0; skipped otherwise (pure Arrhenius, backward compat).
        if rxn.T_gate_K > 0.0:
            dT_g = max(rxn.dT_gate_K, 1.0)
            S = 1.0 / (1.0 + np.exp(np.clip(-(T_safe - rxn.T_gate_K) / dT_g, -50.0, 50.0)))
            r = r * S

        # O2-dependent rate: r *= Y_O2^nO2 when nO2 > 0
        # Uses constant surface Y_O2 (ambient cone approximation; no transport equation).
        # Backward compatible: all existing decks have nO2=0.0 → block never executes.
        if rxn.nO2 > 0.0:
            r = r * (params.y_o2_surf ** rxn.nO2)

        # Species mass balance
        drho_dt[rxn.from_idx] -= r
        solid_prod = r * (1.0 - rxn.nu_gas)
        if rxn.to_idx != rxn.from_idx:
            drho_dt[rxn.to_idx] += solid_prod

        # Endotherm: dH_vol per kg of gas + dH_sol per kg of solid product
        gas_rate  = r * rxn.nu_gas
        omega_gas += gas_rate
        q_py      += rxn.dH_vol * gas_rate
        if rxn.dH_sol != 0.0:
            q_py += rxn.dH_sol * solid_prod

    # ── Surface ablation BC override ─────────────────────────────────────────
    # Interior cells (1..N-1): always suppressed.  In the Eulerian frame the
    # reaction zone (~47 nm) is far smaller than any cell width, so interior
    # Arrhenius causes pre-depletion that is unphysical.
    #
    # Cell-0 treatment:
    #   T[0] < T_min  → leave cell-0 Arrhenius as-is (pre-ignition gas production
    #                    and early pyrolysis needed to reach ablation onset + flame).
    #   T[0] >= T_min → suppress cell-0 Arrhenius; apply energy-balance ablation:
    #                    m_dot = max(q_in − q_conv − q_rad − J[0], 0) / L_py
    #
    # Backward compatible: surface_ablation_bc=False → this block skipped entirely.
    if params.surface_ablation_bc and N > 1:
        # Always suppress interior cells (unphysical Arrhenius in Eulerian frame)
        drho_dt[:, 1:] = 0.0
        omega_gas[1:]  = 0.0
        q_py[1:]       = 0.0
        # Cell-0: switch to ablation BC when T[0] >= T_min
        if T[0] >= params.surface_ablation_T_min:
            drho_dt[:, 0] = 0.0
            omega_gas[0]  = 0.0
            q_py[0]       = 0.0
            if _gas_rate_0_abl > 0.0 and rho_total_safe[0] > 1e-6:
                # Clamp denominator at recession threshold to bound the Jacobian.
                # Without this, rho_total_safe floors to 1e-6 when cell is nearly
                # depleted → ∂drho/∂rho = -gas_rate/1e-6 ≈ -3e8 s⁻¹ → ODE failure.
                # At threshold (1% of ρ₀), Jacobian = -gas_rate/threshold ≈ -29 s⁻¹
                # — manageable.  Recession fires at step end to delete the cell.
                _rho0_sum_abl = sum(max(sp.rho0, 0.0) for sp in params.species)
                _rho_denom = max(rho_total_safe[0],
                                 params.surface_rho_floor * max(_rho0_sum_abl, 1.0))
                for s in range(M):
                    drho_dt[s, 0] = -_gas_rate_0_abl * rho_all[s, 0] / _rho_denom
                omega_gas[0] = _gas_rate_0_abl
                q_py[0]      = _gas_rate_0_abl * params.surface_ablation_L_py
        # else: T[0] < T_min → leave cell-0 Arrhenius as-is

    # ── Surface-cell Lagrangian replenishment ─────────────────────────────────
    # Eulerian FV flux balance for cell 0 with ablation surface:
    #   dρ_s[0]/dt += v_s × ρ_s[1] / dx_0   (flux IN from cell 1)
    #   flux OUT at surface face = ablation; already in drho_dt[s,0] above.
    # At quasi-steady: drho[s,0]/dt = 0 → ρ[s,0] ≈ ρ₀_s (stays fresh). ✓
    # v_s = omega_gas[0]×dx_0 / ρ₀_total = m_dot_surface / ρ₀_total [m/s]
    # Backward compatible: lagrangian_mode=False → skipped entirely.
    if params.lagrangian_mode and params.surface_ablation_bc and N > 1:
        if params.rho0_total > 1e-6 and omega_gas[0] > 0.0:
            _v_s = omega_gas[0] * dx_dyn[0] / params.rho0_total
            if _v_s > 1e-12:
                for _s in range(M):
                    drho_dt[_s, 0] += _v_s * rho_all[_s, 1] / dx_dyn[0]

    # ── Charring Stefan front BC ───────────────────────────────────────────────
    # Replaces Arrhenius for the charring reaction(s) when _stefan_active is True.
    # Uses the NET heat flux at the front cell (conduction in − conduction out).
    # Derivation:  Stefan condition: ρ × v_front × L_py = q_net
    #   L_py  = dH_vol × nu_gas       [J/kg material consumed]
    #   q_net = q_in − q_out          [W/m²]
    #   v_fr  = q_net / (ρ_at_fr × L_py)
    # Why net, not one-sided?  The incoming conduction from the char side is dominated by
    # sensible heating of cold wood; only the excess (net) flux drives endothermic pyrolysis.
    # Using one-sided flux overestimates v_front by ~100×, causing ODE stiffness / failure.
    # Energy balance check: q_py = dH_vol × gas_fr = dH_vol × v_fr × ρ × ν / dx
    #                            = q_net × ν_gas × dH_vol / (L_py × dx)
    #                            = q_net / dx  ✓  (absorbs entire net flux at front cell)
    # Backward compatible: charring_front_bc=False → this block skipped entirely.
    if _stefan_active:
        _char_rxn    = params.reactions[_charring_rxn_idx]
        _rho_from_arr = rho_all[_char_rxn.from_idx]
        _rho_at_fr   = float(_rho_from_arr[_i_fr_stefan])

        # Harmonic-mean conductivities at left and right faces of front cell
        _kL  = k_loc[_i_fr_stefan - 1]; _kC = k_loc[_i_fr_stefan]; _kRR = k_loc[_i_fr_stefan + 1]
        _k_L = 2.0 * _kL * _kC / max(_kL + _kC, 1e-12)
        _k_R = 2.0 * _kC * _kRR / max(_kC + _kRR, 1e-12)
        _dxL = (dx_dyn[_i_fr_stefan - 1] + dx_dyn[_i_fr_stefan]) / 2.0
        _dxR = (dx_dyn[_i_fr_stefan] + dx_dyn[_i_fr_stefan + 1]) / 2.0
        _q_in  = _k_L * (T_safe[_i_fr_stefan - 1] - T_safe[_i_fr_stefan]) / _dxL
        _q_out = _k_R * (T_safe[_i_fr_stefan] - T_safe[_i_fr_stefan + 1]) / _dxR
        _q_fr  = max(_q_in - _q_out, 0.0)   # net flux; clamp at zero (no backward propagation)

        # Effective heat of pyrolysis: dH_vol [J/kg gas] × nu_gas = [J/kg material consumed]
        _dH_eff = max(_char_rxn.dH_vol * _char_rxn.nu_gas, 1e3)
        _v_fr     = _q_fr / max(_rho_at_fr * _dH_eff, 1.0)
        _gas_fr   = _v_fr * _rho_at_fr * _char_rxn.nu_gas / dx_dyn[_i_fr_stefan]
        _solid_fr = _v_fr * _rho_at_fr / dx_dyn[_i_fr_stefan]

        drho_dt[_char_rxn.from_idx, _i_fr_stefan] -= _solid_fr
        drho_dt[_char_rxn.to_idx,   _i_fr_stefan] += _solid_fr * (1.0 - _char_rxn.nu_gas)
        omega_gas[_i_fr_stefan] += _gas_fr
        q_py[_i_fr_stefan]      += _char_rxn.dH_vol * _gas_fr   # endothermic heat sink [W/m³]

    # ── Conduction: non-uniform cell-centred finite volume ────────────────────
    # Inter-cell distances (center-to-center): d_half[i] = (dx[i]+dx[i+1])/2
    # Uses dx_dyn (dynamic in material_coords mode, else equals dx_arr).
    # Bed collapse: d_half is scaled by β → shorter distances → higher conductance.
    d_half = (dx_dyn[:-1] + dx_dyn[1:]) / 2.0 * beta_collapse  # (N-1,)
    # Rightward heat flux at each interface [W/m²]:
    #   J[i] > 0 means heat flows from cell i to cell i+1 (deeper into slab)
    J = k_half * (T[:-1] - T[1:]) / d_half                    # (N-1,)

    # Net conductive flux INTO each cell [W/m²] (before BC terms)
    q_cond_m2 = np.zeros(N)
    if N >= 2:
        q_cond_m2[0]    = -J[0]            # surface: only right interface flux out
        q_cond_m2[-1]   =  J[-1]           # back:    only left interface flux in
    if N >= 3:
        q_cond_m2[1:-1] =  J[:-1] - J[1:] # interior: left in minus right out

    # ── Surface BC: net external flux [W/m²] added to surface cell ───────────
    q_in_surf  = float(q_surface_fn(t))
    q_conv     = params.h_conv * (T[0] - params.Tamb)
    q_rad      = eps_surf * _SIGMA * (T[0] ** 4 - params.T_sur ** 4)
    if params.in_depth_rad_kappa > 0.0:
        # Surface cell receives only convective/radiative losses; q_in_surf is
        # distributed across all cells below via Beer-Lambert (see dT_dt block).
        q_net_surf = -q_conv - q_rad
    else:
        q_net_surf = q_in_surf - q_conv - q_rad   # surface-opaque (default)
    q_cond_m2[0] += q_net_surf

    # ── Back BC: subtract back-face loss from back cell (open only) ───────────
    if params.back_bc == "open":
        q_back = (
            params.h_back  * (T[-1] - params.Tamb)
            + params.eps_back * _SIGMA * (T[-1] ** 4 - params.T_sur ** 4)
        )
        q_cond_m2[-1] -= q_back

    # ── Energy ODE: (rho cp) dT/dt = q_cond/dx - q_py ────────────────────────
    # Divide [W/m²] by dx_dyn [m] to get [W/m³]; use dx_dyn for cell heat capacity.
    dT_dt = (q_cond_m2 / dx_dyn - q_py) / rho_cp

    # ── Gas-phase convective energy transport (GPYRO eq. −∂/∂z(ρ_g c_g v_g T)) ──
    # Full discretisation of the gas-enthalpy-flux divergence:
    #
    #   −∂(ṁ''_g cp_g T)/∂z  for cell i  =
    #       [cum[i+1]×T[i+1] − cum[i]×T[i]] × cp_g / dx[i]
    #     = cum[i+1]×cp_g×(T[i+1]−T[i])/dx[i]  −  ω_gas[i]×cp_g×T[i]
    #       ↑ advective transport (flux through cell)  ↑ production source
    #
    #   cum[i] = Σ_{k=i}^{N-1} ω_gas[k]×dx[k]  (upward mass flux at top face of cell i)
    #
    # Both terms are subtracted from dT_dt because they extract energy from the solid:
    #   • Advective: gas transiting cell i is heated from T[i+1] to T[i] — cools cell.
    #   • Production: newly-formed gas at temperature T[i] carries away sensible heat
    #     cp_g×T[i] per kg — directly cools the active pyrolysis zone.
    if params.cp_gas > 0.0:
        cum = np.cumsum((omega_gas * dx_dyn)[::-1])[::-1]  # cum[i] = Σ_{k≥i} ω×dx
        # Advective term: uses flux at BOTTOM interface of cell i = cum[i+1]
        flux_bottom = np.zeros(N)
        flux_bottom[:-1] = cum[1:]
        dT_advect = np.zeros(N)
        dT_advect[:-1] = (flux_bottom[:-1] * params.cp_gas * (T[:-1] - T[1:])) / (dx_dyn[:-1] * rho_cp[:-1])
        # Production source term: gas formed at T[i] removes cp_g×T[i] per kg
        dT_prod = omega_gas * params.cp_gas * T_safe / rho_cp
        dT_dt -= (dT_advect + dT_prod)

    # ── In-depth radiation absorption (Beer-Lambert, porous fuel bed) ─────────
    # For loose fibrous beds (straw, grass) the cone irradiance penetrates
    # several mm to cm before being fully absorbed.  Distributes heat source
    # across multiple cell depths simultaneously, sustaining interior pyrolysis.
    # References: Albini (1985) Combust. Sci. Tech. 42:229 (wildland fire spread);
    #             Anderson (1969) USDA Forest Service Res. Paper INT-56 (fuel SAV).
    if params.in_depth_rad_kappa > 0.0:
        _kappa_0 = params.in_depth_rad_kappa
        if params.in_depth_rad_density_weighted:
            # κ_local,i = κ₀ × (ρ_reactive,i / ρ₀_reactive)
            # Reactive species = those with initial rho0 > 0 (excludes char product).
            # As cells deplete, κ_local → 0 → depleted/charred cells become
            # transparent and radiation advances into unburned interior.
            # Physical: optical extinction ∝ remaining absorbing solid (Beer 1852;
            # Albini 1985 eq. 14; Baughman & Albini 1980 fuel ignition).
            _rho0_reactive = sum(sp.rho0 for sp in params.species if sp.rho0 > 0)
            _rho_reactive  = np.zeros(N)
            for _s, _sp in enumerate(params.species):
                if _sp.rho0 > 0:
                    _rho_reactive += rho_all_nn[_s]
            _kappa_local = _kappa_0 * _rho_reactive / max(_rho0_reactive, 1e-6)  # (N,)
            # Optical depth at each cell interface: τ[j] = Σ_{i<j} κ_local_i × dx_i
            _tau = np.concatenate([[0.0], np.cumsum(_kappa_local * dx_dyn)])  # (N+1,)
            _q_abs = q_in_surf * (np.exp(-_tau[:-1]) - np.exp(-_tau[1:]))    # (N,)
        else:
            # Fixed κ everywhere (original T9 behaviour)
            _x_front = np.concatenate([[0.0], np.cumsum(dx_dyn[:-1])])
            _x_back  = _x_front + dx_dyn
            _q_abs = q_in_surf * (np.exp(-_kappa_0 * _x_front) - np.exp(-_kappa_0 * _x_back))
        # Add volumetric source: [W/m²] / ([J/m³/K] × [m]) = [K/s]
        dT_dt += _q_abs / (rho_cp * dx_dyn)

    # ── Freeze T[0] when surface cell depleted below recession threshold ──────
    # When rho < recession threshold, the cell should be deleted but the end-of-step
    # check hasn't fired yet.  With rho_cp floored to 1 J/m³/K and dx=10 µm,
    # the unfrozen dT_dt[0] ≈ q_net/dx/1 ≈ 3×10⁹ K/s → ODE step failure.
    # Setting dT_dt[0]=0 is conservative (T stays frozen for < one step; ~J/m² error),
    # and removes the catastrophic eigenvalue from the Jacobian until recession fires.
    # Backward compatible: skipped when surface_ablation_bc=False.
    if params.surface_ablation_bc and N > 1:
        _rho0_s = sum(max(sp.rho0, 0.0) for sp in params.species)
        if rho_total_safe[0] <= params.surface_rho_floor * max(_rho0_s, 1.0):
            dT_dt[0] = 0.0

    # ── Assemble dy/dt ────────────────────────────────────────────────────────
    dy_dt = np.empty_like(y)
    dy_dt[:N]  = dT_dt
    dy_dt[N:]  = drho_dt.ravel()
    return dy_dt


# ── Interval-averaged mass flux helper ─────────────────────────────────────────

def _compute_m_dot_delta(
    params: MolParams,
    rho_cur: np.ndarray,
    rho_prev: np.ndarray,
    dt: float,
) -> float:
    """Interval-averaged volatile mass flux from finite-difference Δrho/Δt.

    m_dot = dx × sum_s  sum_i  max(-Δrho_{s,i}/Δt, 0) × nu_gas_eff_s

    Using Δrho rather than instantaneous drho/dt eliminates Arrhenius spikes
    at rapid mid-conversion.  The net decrease in each species density over the
    interval represents the mass of volatiles released during that interval.

    Parameters
    ----------
    params : MolParams
    rho_cur, rho_prev : ndarray, shape (M*N,)
        Flattened species density arrays at end and start of interval.
    dt : float
        Interval length [s].

    Returns
    -------
    m_dot_avg : float  [kg/m²/s], ≥ 0
    """
    if dt <= 0.0:
        return 0.0
    M = len(params.species)
    N = params.N
    dx_arr = params.dx_arr  # (N,) non-uniform cell widths

    drho = (rho_cur - rho_prev).reshape(M, N) / dt   # [kg/m³/s] per species

    # Net total density decrease = gas production rate (char redistribution cancels)
    drho_total = np.sum(drho, axis=0)   # (N,) [kg/m³/s]
    # Integrate over non-uniform cells [kg/m²/s]
    m_dot = float(np.dot(np.maximum(-drho_total, 0.0), dx_arr))
    return m_dot


# ── Integration entry point ────────────────────────────────────────────────────

def integrate_mol(
    params: MolParams,
    t_span: tuple,
    T0_K: float,
    q_incident_fn: Callable[[float], float],
    hoc_eff_J_kg: float,
    flame_enable: bool = False,
    flame_cfg=None,
    fuel_viability=None,
    n_passes: int = 3,      # kept for API compatibility; not used in sequential mode
    tau_growth_s: float = 20.0,
    flame_tol_W_m2: float = 1.0,
) -> MolResult:
    """Integrate MoL pyrolysis ODE with sequential (operator-split) flame coupling.

    Advances the ODE one dt_eval step, evaluates m_dot at the new state,
    updates the flame state machine, then continues.  Flame feedback latency =
    dt_eval (≤ 0.5 s), eliminating the pass-lag oscillations of fixed-point
    iteration.

    Parameters
    ----------
    params : MolParams
    t_span : (t0, t_end) [s]
    T0_K : float
        Uniform initial temperature [K].
    q_incident_fn : callable(t) -> float
        Incident irradiance [W/m²] (cone + any external sources).
    hoc_eff_J_kg : float
        Effective heat of combustion [J/kg volatile].
    flame_enable : bool
    flame_cfg, fuel_viability : model.flame objects or None
    n_passes : int
        Ignored; kept for backward-compatible call sites.
    tau_growth_s : float
        Flame growth ramp time constant [s].
    flame_tol_W_m2 : float
        Ignored; kept for backward-compatible call sites.

    Returns
    -------
    MolResult
    """
    N = params.N
    M = len(params.species)

    # Uniform output / coupling grid
    _dt_ev = params.dt_eval if params.dt_eval > 0.0 else params.max_step
    t_eval = np.arange(float(t_span[0]), float(t_span[1]) + _dt_ev * 0.5, _dt_ev)
    n_out = t_eval.size

    # State and output arrays
    N_states = N * (1 + M)
    y_hist    = np.zeros((N_states, n_out))
    m_dot_out = np.zeros(n_out)
    T_surf_track = np.zeros(n_out)

    # Initial conditions: uniform temperature, initial species densities
    y0 = np.zeros(N_states)
    y0[:N] = T0_K
    for s, sp in enumerate(params.species):
        y0[N + s * N: N + (s + 1) * N] = sp.rho0
    y_hist[:, 0] = y0
    T_surf_track[0] = T0_K

    # Flame state
    q_flame: float = 0.0
    _t_ign: float | None = None
    _flame_internal = None
    if flame_enable:
        from model.flame import FlameInternalState, flame_step  # noqa: PLC0415
        _flame_internal = FlameInternalState()

    # ── Sparse Jacobian (CPR grouping: ~12 groups vs 160 RHS eval for N=40,M=3) ──
    jac_sp = _build_jac_sparsity(N, M)

    # charring_front_bc: Stefan BC adds drho_{s,i*}/dT_{i*-1} off-diagonal coupling.
    # The front cell i* changes over time, so a static extended sparsity would add
    # T[i-1]→ρ[s,i] for ALL i — this destroys CPR column grouping efficiency (100× slower).
    # Instead, use the original sparsity: the T[i*-1]→ρ[s,i*] entry is secondary; the
    # solver compensates with a few extra Newton iterations per step (minor cost).

    # ── Dynamic grid state (for surface recession) ────────────────────────────
    N_cur = N
    dx_cur = params.dx_arr.copy()
    dm_cur = params.dm_arr.copy()   # mirror of dx_cur; must be sliced with dx_cur
    params_cur = params
    # rho0 total for recession threshold (sum of all initial species densities)
    _rho0_total = sum(max(sp.rho0, 0.0) for sp in params.species)

    # One-way ablation latch: once T_surf first reaches T_min, keep ablation active
    # even if cell deletion briefly exposes a new cell-0 with T slightly < T_min.
    _abl_latch: bool = False  # set True permanently after first activation

    # y_prev_rho: post-spall rho from previous step (for Δrho m_dot computation).
    # Using this instead of y_hist[N:,i] avoids counting spalled char as volatiles.
    y_prev_rho = y0[N:].copy()

    # Spall hysteresis: track char depth at last spall event.  Only fire again
    # when NEW char depth (total minus spalled baseline) exceeds spall_depth_m.
    _spall_depth_baseline: float = 0.0

    y_cur = y0.copy()

    for i in range(n_out - 1):
        t0_step = float(t_eval[i])
        t1_step = float(t_eval[i + 1])
        _dt_step = t1_step - t0_step
        _q_fl_i = q_flame  # capture by value for this step's closure

        # Rebuild jac_sparsity only if grid shrank due to surface recession.
        # material_coords mode: dx_dyn depends on rho_total, adding T-rho off-diagonal
        # coupling not in the default sparsity pattern; use full numerical Jacobian.
        # lagrangian_mode: drho_dt[s,0] += v_s×rho[s,1]/dx_0 adds rho[s,0]←rho[s,1]
        # off-diagonal coupling not in the default sparsity pattern.
        if params.material_coords or params.lagrangian_mode:
            _jac_sp_cur = None
        else:
            _jac_sp_cur = jac_sp if N_cur == N else _build_jac_sparsity(N_cur, M)

        sol_step = solve_ivp(
            lambda t_, y_: _mol_rhs(
                t_, y_, params_cur,
                lambda t__: float(q_incident_fn(t__)) + _q_fl_i,
            ),
            (t0_step, t1_step),
            y_cur,
            method=params.method,
            rtol=params.rtol,
            atol=params.atol,
            max_step=params.max_step,
            jac_sparsity=_jac_sp_cur,
        )
        if sol_step.status < 0:
            raise RuntimeError(
                f"MoL ODE step failed at t={t0_step:.2f}–{t1_step:.2f}: "
                f"{sol_step.message}"
            )

        y_cur = sol_step.y[:, -1]

        # Store in y_hist (only when N unchanged; used for full T/rho output)
        if N_cur == N:
            y_hist[:, i + 1] = y_cur

        # ── m_dot: Δrho method (post-spall-prev vs pre-spall-current) ─────────
        # y_prev_rho is the POST-spall rho from the previous step; y_cur[N_cur:]
        # is the PRE-spall rho from the current step.  The difference captures
        # only natural pyrolysis — spalled char is added separately below.
        if params.surface_recession_enable:
            # Mass-balance approach (handles variable N_cur after recession)
            _mass_before = float(
                np.dot(np.sum(y_prev_rho.reshape(M, N_cur), axis=0), dx_cur)
            )
            _mass_after = float(
                np.dot(np.sum(y_cur[N_cur:].reshape(M, N_cur), axis=0), dx_cur)
            )
            m_dot_out[i + 1] = max(_mass_before - _mass_after, 0.0) / _dt_step
        elif params.lagrangian_mode and params.surface_ablation_bc and N_cur > 1:
            # With Lagrangian replenishment drho_dt[s,0] ≈ 0 → Δrho method gives 0.
            # Compute m_dot directly from energy-balance formula (mirrors _mol_rhs
            # ablation BC pre-computation block).
            _T0_l = float(y_cur[0])
            _T1_l = float(y_cur[1])
            if _T0_l >= params_cur.surface_ablation_T_min:
                _rs01 = np.maximum(y_cur[N_cur:].reshape(M, N_cur)[:, :2], 0.0)
                _rt0_l = max(float(np.sum(_rs01[:, 0])), 1e-6)
                # rho_vf_basis: mirrors _mol_rhs volume-fraction reference
                _rvf = np.array([max(sp.rho0, 0.0) for sp in params_cur.species])
                for _rxn_l in params_cur.reactions:
                    if _rvf[_rxn_l.to_idx] < 1e-6:
                        _fb_l = max(_rvf[_rxn_l.from_idx], _rxn_l.rho_ref)
                        if _fb_l > 1e-6:
                            _rvf[_rxn_l.to_idx] = _fb_l * (1.0 - _rxn_l.nu_gas)
                _k01_l = np.zeros(2); _vw01_l = np.zeros(2); _eps0_l = 0.0
                _T01_l = np.array([max(_T0_l, 1.0), max(_T1_l, 1.0)])
                for _si_l, _sp_l in enumerate(params_cur.species):
                    _vf_l = _rs01[_si_l] / max(_rvf[_si_l], 1e-6)
                    _k01_l += _vf_l * _sp_l.k_at(_T01_l)
                    _vw01_l += _vf_l
                    _eps0_l += float(_rs01[_si_l, 0]) / _rt0_l * _sp_l.eps
                _k01_l /= np.maximum(_vw01_l, 1e-12)
                _kh0_l = 2.0 * _k01_l[0] * _k01_l[1] / max(_k01_l[0] + _k01_l[1], 1e-12)
                _dh0_l = (dx_cur[0] + dx_cur[1]) / 2.0
                _J0_l  = _kh0_l * (_T0_l - _T1_l) / _dh0_l
                _qt_l  = float(q_incident_fn(t1_step)) + _q_fl_i
                _qc_l  = params_cur.h_conv * (_T0_l - params_cur.Tamb)
                _qr_l  = _eps0_l * _SIGMA * (_T0_l**4 - params_cur.T_sur**4)
                _qa_l  = max(_qt_l - _qc_l - _qr_l - max(_J0_l, 0.0), 0.0)
                m_dot_out[i + 1] = _qa_l / params_cur.surface_ablation_L_py
            else:
                m_dot_out[i + 1] = _compute_m_dot_delta(
                    params_cur, y_cur[N_cur:], y_prev_rho, _dt_step
                )
        else:
            m_dot_out[i + 1] = _compute_m_dot_delta(
                params_cur, y_cur[N_cur:], y_prev_rho, _dt_step
            )

        # ── Char spalling (between ODE steps) ─────────────────────────────────
        # When NEW char depth (total minus spalled baseline) exceeds spall_depth_m,
        # DELETE the surface cell if it is predominantly char (non-char species < 5%
        # of original).  Cell deletion is the correct Eulerian analog of spalling —
        # resetting rho_char to 5% would cause repeated firing and thermal capacity
        # collapse.  Physical basis: Shi & Chew 2023; char cracking at ~2–5mm.
        # Spalled char departs as solid fragment — no gas-phase m_dot contribution.
        # Important: modify y_cur AFTER m_dot Δrho computation to avoid double-count.
        if params.spall_enable and M >= 2 and N_cur > 2:
            rho_arr_sp = y_cur[N_cur:].reshape(M, N_cur)
            # Char depth: cells from surface where char has meaningfully formed
            rho_char_arr = rho_arr_sp[M - 1]
            char_formed = (rho_char_arr > 1.0).astype(float)
            delta_conv = float(np.dot(char_formed, dx_cur))
            if (delta_conv - _spall_depth_baseline) > params.spall_depth_m:
                # Check if surface cell is predominantly char (non-char species depleted)
                _rho0_total_sp = sum(max(sp.rho0, 0.0) for sp in params.species)
                rho_nonchar_surf = float(np.sum(rho_arr_sp[: M - 1, 0]))
                _char_thresh = 0.05 * max(_rho0_total_sp, 1.0)
                if rho_nonchar_surf < _char_thresh:
                    # Delete surface cell (char layer peels off as solid)
                    T_new = y_cur[1:N_cur]
                    rho_blocks = [
                        y_cur[N_cur + s * N_cur + 1: N_cur + (s + 1) * N_cur]
                        for s in range(M)
                    ]
                    y_cur = np.concatenate([T_new] + rho_blocks)
                    dx_cur = dx_cur[1:]
                    dm_cur = dm_cur[1:]
                    N_cur -= 1
                    params_cur = _copy.copy(params)
                    params_cur.N = N_cur
                    params_cur.dx_arr = dx_cur.copy()
                    params_cur.dm_arr = dm_cur.copy()
                    y_prev_rho = y_cur[N_cur:].copy()
                    _spall_depth_baseline = delta_conv - dx_cur[0]  # new baseline

        # Update y_prev_rho to POST-spall rho (reference for next step's Δrho)
        y_prev_rho = y_cur[N_cur:].copy()

        # ── Surface recession: delete depleted surface cell ────────────────────
        # For non-charring polymers (nu_gas=1.0): surface cells fully deplete
        # with no char residue; delete them to allow the regression front to
        # advance and sustain quasi-steady MLR.
        # For charring materials the remaining solid at deletion is char, not
        # volatile gas — the deletion pulse is therefore omitted (no m_dot spike).
        if params.surface_recession_enable and N_cur > 2:
            rho_arr = y_cur[N_cur:].reshape(M, N_cur)
            rho_surf_total = float(np.sum(rho_arr[:, 0]))
            if _rho0_total > 1e-6 and rho_surf_total < params.surface_rho_floor * _rho0_total:
                # Omit instantaneous gasification pulse; the Arrhenius m_dot
                # from the cell's last integration step already accounts for
                # volatile production.  Remaining residue is char/ash.
                pass
                # Delete surface cell: shift state arrays
                T_new = y_cur[1:N_cur]
                rho_blocks = [
                    y_cur[N_cur + s * N_cur + 1: N_cur + (s + 1) * N_cur]
                    for s in range(M)
                ]
                y_cur = np.concatenate([T_new] + rho_blocks)
                dx_cur = dx_cur[1:]
                dm_cur = dm_cur[1:]
                N_cur -= 1
                # Update params_cur with new grid (shallow copy; dx_arr/dm_arr replaced)
                params_cur = _copy.copy(params)
                params_cur.N = N_cur
                params_cur.dx_arr = dx_cur.copy()
                params_cur.dm_arr = dm_cur.copy()
                # Update y_prev_rho for new N
                y_prev_rho = y_cur[N_cur:].copy()

        # ── Surface temperature ────────────────────────────────────────────────
        T_surf_track[i + 1] = float(y_cur[0])

        # ── Ablation latch: activate once; propagate to params_cur ─────────────
        if (params.surface_ablation_bc and not _abl_latch
                and T_surf_track[i + 1] >= params.surface_ablation_T_min):
            _abl_latch = True
        if _abl_latch and not params_cur.surface_ablation_active:
            params_cur = _copy.copy(params_cur)
            params_cur.surface_ablation_active = True

        # ── Flame feedback for NEXT step (operator splitting) ──────────────────
        if flame_enable:
            _hrrpua_w = m_dot_out[i + 1] * hoc_eff_J_kg
            _fo = {
                "HRRPUA_W_m2": _hrrpua_w,
                "T_surf_K": float(y_cur[0]),
                "m_py": m_dot_out[i + 1],
            }
            _q_fi, _, _flame_internal = flame_step(
                t1_step, _fo, flame_cfg, fuel_viability, _flame_internal
            )
            if tau_growth_s > 0.0 and _q_fi > 0.0:
                if _t_ign is None:
                    _t_ign = t1_step
                _q_fi *= 1.0 - float(np.exp(-(t1_step - _t_ign) / tau_growth_s))
            q_flame = _q_fi

    # ── Build result ───────────────────────────────────────────────────────────
    if params.surface_recession_enable and N_cur < N:
        # Grid shrank; T and rho arrays are not spatially consistent.
        # T_surf_track is always valid; fill T[0,:] from it.
        T_hist = np.zeros((N, n_out))
        T_hist[0, :] = T_surf_track
        rho_hist = np.zeros((M, N, n_out))
    else:
        T_hist   = y_hist[:N, :]
        rho_hist = y_hist[N:, :].reshape(M, N, n_out)

    # ── Smooth m_dot when surface recession is enabled ────────────────────────
    # Discrete cell-deletion events create step-wise oscillations in the per-step
    # mass-balance m_dot (period ≈ cell-lifetime, typically 1–5 s for PMMA).
    # A 10 s centred uniform window represents what a cone-calorimeter load cell
    # would measure; it does not affect the physics (flame feedback already ran).
    # Backward compatible: recession_enable=False → no smoothing applied.
    if params.surface_recession_enable and _dt_ev > 0.0:
        _n_win = max(1, int(round(10.0 / _dt_ev)))   # 10 s / dt_eval
        if _n_win > 1:
            _kernel = np.ones(_n_win) / _n_win
            m_dot_out = np.convolve(m_dot_out, _kernel, mode="same")

    hrrpua_kW = m_dot_out * hoc_eff_J_kg / 1000.0

    return MolResult(
        t=t_eval,
        T=T_hist,
        rho=rho_hist,
        m_dot=m_dot_out,
        T_surf=T_surf_track,
        hrrpua_kW=hrrpua_kW,
    )


# ── Convenience builder for single-component charring solid ────────────────────

def build_single_species_params(
    L: float,
    rho_v: float,
    cp_v: float,
    k_v: float,
    eps: float,
    rho_char: float,
    cp_char: float,
    k_char: float,
    A_py: float,
    E_py: float,
    dH_py: float,
    Tamb: float = 300.0,
    T_sur: float = 300.0,
    h_conv: float = 15.0,
    back_bc: str = "adiabatic",
    h_back: float = 10.0,
    eps_back: float = 0.87,
    N: int = 20,
    method: str = "Radau",
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_step: float = 0.5,
    dt_eval: float = 0.5,
    n_rxn: float = 1.0,
    nk_v: float = 0.0,
    nc_v: float = 0.0,
    nk_char: float = 0.0,
    nc_char: float = 0.0,
    gamma_char: float = 0.0,
    grid_stretch: float = 1.0,
    k_crack_frac: float = 0.0,
) -> MolParams:
    """Build MolParams for a two-species (virgin → char + gas) model.

    dH_py is interpreted as dH_vol — J per kg of gas produced — per
    Lautenberger (2009).  The volatile fraction nu_gas is derived from the
    density ratio: nu_gas = (rho_v - rho_char) / rho_v.

    The char species starts at rho0 = 0 and accumulates as the reaction
    proceeds; at full conversion rho_char_final = rho_v × (1 - nu_gas) = rho_char.
    """
    nu_gas = (rho_v - rho_char) / max(rho_v, 1e-6)
    virgin = MolSolidSpecies(
        name="virgin", k0=k_v, nk=nk_v, rho0=rho_v, cp0=cp_v, nc=nc_v, eps=eps,
    )
    char = MolSolidSpecies(
        name="char", k0=k_char, nk=nk_char, rho0=0.0, cp0=cp_char, nc=nc_char,
        eps=0.87, gamma=gamma_char,
    )
    rxn = MolReaction(
        from_idx=0, to_idx=1, Z=A_py, E=E_py, n=n_rxn,
        dH_vol=dH_py, nu_gas=nu_gas,
    )
    return MolParams(
        L=L, N=N,
        species=[virgin, char],
        reactions=[rxn],
        Tamb=Tamb, T_sur=T_sur, h_conv=h_conv,
        back_bc=back_bc, h_back=h_back, eps_back=eps_back,
        method=method, rtol=rtol, atol=atol, max_step=max_step, dt_eval=dt_eval,
        grid_stretch=grid_stretch, k_crack_frac=k_crack_frac,
    )
