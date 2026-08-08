"""1D vertical soil heat conduction sub-model — Phase 14v-bc-soil.

Couples to the gas-phase DOM radiation solver via the wall-face BC at
z=0.  Each horizontal (j, i) location has an independent 1D vertical
soil column representing the dirt heat reservoir below the gas grid.

Physics (Carslaw & Jaeger 1959 §2; FIRESTAR Morvan & Dupuy 2004 §3.2;
FIRETEC Pimont et al. 2006 *Combust. Sci. Tech.* 178:1389 §2.4):

    ρ_s · c_s · ∂T/∂t = ∂/∂z (k_s · ∂T/∂z)              [soil interior]
    -k_s · ∂T/∂z|_{z=0⁻} = q_in_radiation − ε_s σ T_surface⁴   [BC top]
    T(z = -δ_deep)        = T_amb                                [BC bot]

Discretization:
    Geometric vertical stretch from dz_first ≈ 1 mm at the surface to
    a few mm at depth.  Total soil depth ~30 mm covers the thermal
    penetration depth at fire timescales (√(α_s · 30 s) ≈ 4 mm) with
    several×safety factor.

    5–8 cells per (j, i) is the FIRESTAR / FIRETEC standard.

Numerical:
    Explicit Euler — dt_max = dz_first² / (2 α_s) ≈ 1.25 s for
    dz_first = 1 mm and α_s = 4·10⁻⁷ m²/s.  Always larger than the
    gas-phase dt (~1 ms) so no substepping required.

Soil properties (Hahn 1981 *J. Atmos. Sci.* 38:1601 dry topsoil; matches
FIRESTAR baseline):
    ρ_s   = 1500 kg/m³        — dry soil bulk density
    c_p_s = 800  J/kg/K        — soil specific heat
    k_s   = 0.48 W/m/K         — soil thermal conductivity
    α_s   = k/(ρc) = 4.0e-7 m²/s
    ε_s   = 0.85              — soil IR emissivity (Hahn 1981 dry soil)

Coupling to DOM (z=0 wall BC):
    q_in_soil(j,i) = ∫_{Ω↓} I(0, j, i, Ω) |μ| dΩ  (DOM provides this)
    DOM uses T_soil_surface = T_soil[0, j, i] as the wall T for I_w
        I_w = ε_s σ T_soil_surface⁴ / π + (1−ε_s) · q_in_diffuse / π

This recycles a substantial fraction of fire radiation back into the gas
phase as the soil heats up, addressing the cold-blackbody-ground
limitation (T_soil rising to 500-700 K during peak fire reduces net
radiation loss to the ground from ~all-incident to ~half-incident).

References:
- Carslaw, H.S. & Jaeger, J.C. (1959) Conduction of Heat in Solids,
  2nd ed. — semi-infinite solid with surface flux BC (the canonical
  reference for short-duration surface-heating problems)
- Morvan, D. & Dupuy, J.L. (2004) Combust. Flame 138:199 — FIRESTAR
  soil sub-model description
- Pimont, F., Dupuy, J.L., Linn, R.R., Sauer, J.A. (2006) Combust.
  Sci. Tech. 178:1389 §2.4 — FIRETEC soil treatment
- Hahn, J. (1981) J. Atmos. Sci. 38:1601 — soil IR emissivity and
  thermal properties
- Albini, F.A. (1985) Combust. Sci. Tech. 42:229 — semi-infinite soil
  treatment for grass-fire heat transfer
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# Soil properties (Hahn 1981 dry topsoil; FIRESTAR baseline)
RHO_SOIL_DEFAULT = 1500.0      # [kg/m³]
CP_SOIL_DEFAULT  = 800.0       # [J/kg/K]
K_SOIL_DEFAULT   = 0.48        # [W/m/K]
EPS_SOIL_DEFAULT = 0.85        # [-] IR emissivity
SIGMA_SB         = 5.67e-8     # [W/m²/K⁴]


def build_soil_grid(
    n_soil: int = 6,
    dz_first: float = 0.001,    # [m] 1 mm surface cell
    growth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build geometrically-stretched soil grid.

    Returns:
        soil_dz   (n_soil,)   per-cell dz [m]
        d_above   (n_soil,)   center-to-center distance to cell above
                              (d_above[0] = dz_first/2: half-cell to surface BC)
        d_below   (n_soil,)   center-to-center distance to cell below
                              (d_below[-1] = dz[-1]/2: half-cell to deep T_amb BC)
        depth_total (m)       total soil depth covered
    """
    soil_dz = np.array(
        [dz_first * growth ** k for k in range(n_soil)],
        dtype=np.float64,
    )
    d_above = np.empty(n_soil, dtype=np.float64)
    d_below = np.empty(n_soil, dtype=np.float64)
    # Top BC face is at z=0 (surface), distance from cell 0 center is dz[0]/2
    d_above[0] = soil_dz[0] / 2.0
    for k in range(1, n_soil):
        d_above[k] = 0.5 * (soil_dz[k - 1] + soil_dz[k])
    for k in range(n_soil - 1):
        d_below[k] = 0.5 * (soil_dz[k] + soil_dz[k + 1])
    # Bottom BC face is at z=-depth, distance from cell N-1 center is dz[N-1]/2
    d_below[-1] = soil_dz[-1] / 2.0
    return soil_dz, d_above, d_below, float(soil_dz.sum())


@njit(cache=True, parallel=True)
def step_soil_conduction(
    T_soil: np.ndarray,            # (N_soil, Ny, Nx) [K] — updated in place
    q_in_surface: np.ndarray,      # (Ny, Nx) [W/m²] downward radiation flux from DOM
    dt: float,
    soil_dz: np.ndarray,           # (N_soil,) [m]
    d_above: np.ndarray,           # (N_soil,) [m] center-to-center distances
    d_below: np.ndarray,           # (N_soil,) [m]
    alpha_s: float,                # [m²/s] soil thermal diffusivity
    k_s: float,                    # [W/m/K] soil thermal conductivity
    rho_s: float,                  # [kg/m³] soil density
    cp_s: float,                   # [J/kg/K] soil specific heat
    eps_s: float,                  # [-] IR emissivity
    T_amb: float,                  # [K] deep-soil & ambient temperature
) -> None:
    """One explicit-Euler step of 1D vertical soil heat conduction at
    every horizontal (j, i) location.

    Top BC (z=0):       net flux = q_in − ε σ T[0]⁴
    Bottom BC (z=-δ):   T = T_amb (Dirichlet, semi-infinite assumption)

    Discretization is finite-volume — flux through each face computed
    with Fourier's law using d_above / d_below distances.
    """
    N_soil, Ny, Nx = T_soil.shape
    rho_cp_inv = 1.0 / (rho_s * cp_s)
    inv_d_above = 1.0 / d_above
    inv_d_below = 1.0 / d_below

    for j in prange(Ny):
        for i in range(Nx):
            for k in range(N_soil):
                T_k = T_soil[k, j, i]
                # Flux INTO cell from the top face (z = -sum(dz[0..k-1]))
                if k == 0:
                    # Top BC: net surface heat flux
                    q_top = q_in_surface[j, i] - eps_s * SIGMA_SB * T_k ** 4
                else:
                    T_above = T_soil[k - 1, j, i]
                    q_top = k_s * (T_above - T_k) * inv_d_above[k]
                # Flux OUT of cell through the bottom face
                if k == N_soil - 1:
                    # Bottom BC: deep soil at T_amb
                    q_bot = k_s * (T_k - T_amb) * inv_d_below[k]
                else:
                    T_below = T_soil[k + 1, j, i]
                    q_bot = k_s * (T_k - T_below) * inv_d_below[k]
                # Energy balance: dT/dt = (q_top − q_bot) / (ρ c dz)
                T_soil[k, j, i] = T_k + dt * (q_top - q_bot) * rho_cp_inv / soil_dz[k]
