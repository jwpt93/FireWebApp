"""3D k-ε turbulence model (Phase 14b).

Standard high-Reynolds k-ε with buoyancy production and porous-bed
dissipation, based on the 2D implementation in ``model_outdoor/spread.py``.

Governing equations (Launder & Spalding 1974 Comput. Methods Appl. Mech.
Eng. 3:269; Rodi 1987 J. Geophys. Res. 92:5305 buoyancy correction):

    Dk/Dt = ∇·((ν + ν_t/σ_k) ∇k) + P_k + G_k − ε − D_drag
    Dε/Dt = ∇·((ν + ν_t/σ_ε) ∇ε) + (ε/k)(C_1ε(P_k+G_k) − C_2ε* ε)

where:
    ν_t = C_μ k²/ε                   [m²/s] eddy viscosity
    P_k = ν_t |S|²                  shear production
    G_k = (ν_t/Pr_t)(g/T)(∂T/∂z)    buoyancy production (>0 unstable)
    D_drag = C_D σ_SAV α_s u_p k    porous-bed dissipation
    |S|² = 2 S_ij S_ij               strain-rate magnitude
    S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)

Constants (Launder-Spalding standard k-ε):
    C_μ=0.09, C_1ε=1.44, C_2ε=1.92, σ_k=1.0, σ_ε=1.3, Pr_t=0.85

RNG correction (Yakhot & Orszag 1986 J. Sci. Comput. 1:3):
    C_2ε* = C_2ε + C_μ η³(1−η/η₀)/(1+βη³),  η = |S|k/ε
    Adds dissipation in high-strain fire-plume regions.

Buoyancy clamp G_k ≤ P_k (Rodi 1987 standard practice; ANSYS Fluent
Theory Guide §4.4.2): prevents runaway growth of k from extreme
∂T/∂z gradients in fire plumes.

Time integration: explicit advection-diffusion with implicit destruction
(both k and ε have stiff sink terms ε, C_2ε ε²/k that must be
treated implicitly for stability).

Outputs the eddy viscosity ν_t which is used:
- as additional momentum diffusion (ν_eff = ν + ν_t)
- as turbulent species diffusion (D_eff = D + ν_t/Sc_t)
- to compute τ_mix = k/ε for EDM combustion

References:
- Launder, B.E. & Spalding, D.B. (1974) — standard k-ε model
- Yakhot, V. & Orszag, S.A. (1986) — RNG k-ε
- Rodi, W. (1987) — buoyancy production
- Henkes, R.A.W.M. et al. (1991) IJHMT 34:377 — buoyancy-modified k-ε C_3ε
- Morvan, D. & Dupuy, J.L. (2004) Combust. Flame 138:199 — k-ε for grass fire
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from model_outdoor.physics_3d.muscl_3d import muscl_face_value


# Air properties (Drysdale 2011 Table 2.4)
_NU_GAS = 1.5e-5      # [m²/s] kinematic viscosity at 300 K
_G = 9.81             # [m/s²]

# Standard k-ε constants
C_MU    = 0.09
C_1EPS  = 1.44
C_2EPS  = 1.92
SIGMA_K   = 1.0
SIGMA_EPS = 1.3
PR_T    = 0.85   # turbulent Prandtl number (gas-energy diffusion)
SC_T    = 0.7    # turbulent Schmidt number (species diffusion)

# Porous-bed drag dissipation
C_D_DRAG = 1.0   # cylinder/fiber drag coefficient at moderate Re
                 # (Morvan & Dupuy 2001 Combust. Flame 127:1981)

# Phase 14l — Sanz (2003) canopy/vegetation turbulence closure.
# Reference: Sanz (2003) Boundary-Layer Meteorol. 108:191-217.
# Used in FIRETEC (Pimont & Linn 2009) and many vegetation-canopy ABL codes.
#
# Drag work on the mean flow extracts kinetic energy and SOURCES TKE
# (β_p term), while sub-grid wake breakup adds an extra k-proportional
# SINK (β_d term).  Combined:
#
#   S_k_canopy = C_D σ_β · (β_p |u|³ − β_d |u| k)
#   S_ε_canopy = C_D σ_β · (ε/k) · (C_ε4 β_p |u|³ − C_ε5 β_d |u| k)
#
# Calibration DEFAULTS (Sanz 2003 Table 4, calibrated against WT data
# for DENSE forest canopies).  For sparse canopies (pasture, LAI < 2)
# Brunet et al. (1994) BLM 71:135 and Massman (1997) BLM 83:407 suggest
# β_d closer to 1.5-2.0 (less sub-grid wake-wake interaction).  Phase
# 14at re-added 2026-05-30: now deck inputs `outdoor.canopy_beta_p` and
# `outdoor.canopy_beta_d`.
BETA_P_CANOPY_DEFAULT = 1.0   # mean-flow KE → TKE conversion (production)
BETA_D_CANOPY_DEFAULT = 4.0   # sub-grid wake-breakup short-circuit dissipation
C_EPS4_CANOPY = 0.9   # ε analog of C_ε1 for canopy production
C_EPS5_CANOPY = 0.9   # ε analog of C_ε2 for canopy dissipation
# Backward-compat aliases for legacy imports.
BETA_P_CANOPY = BETA_P_CANOPY_DEFAULT
BETA_D_CANOPY = BETA_D_CANOPY_DEFAULT
#
# Pre-14l: only the sink was modeled (β_d=1 implicit, β_p=0 missing).
# This under-predicted bed-region k for porous wildland fuels — diagnosed
# via Cut bed downstream-of-source diagnostic (plots/phase14h/cut4_u4_*.png)
# where T_g in buffer reached 750K but T_s in bed below stayed at 303K
# because turbulent diffusion ν_t·∇T_g across the bed-buffer interface
# was too small.  Sanz 2003 raises canopy k → ν_t → cross-interface flux.

# RNG correction
ETA0 = 4.38
BETA_RNG = 0.012

# Phase 14ai — Sandia BVG (buoyant vorticity generation) k-source.
# Reference: Nicolette V.F., Tieszen S.R., Black A.R., Domino S.P., O'Hern T.J.
# (2005) "A Turbulence Model for Buoyant Flows Based on Vorticity Generation,"
# Sandia Tech Report SAND2005-6273.  k-source term derived from baroclinic
# vorticity generation rate, ∇ρ × ∇p:
#
#   G_B = C_BVG · (ν + ν_t) · |∇ρ × ∇p| / ρ²                       (SAND Eq. 14)
#
# Approximation used here (hydrostatic-only ∇p ≈ −ρg·ẑ): the dominant pressure
# gradient in buoyancy-driven fire flow is hydrostatic.  Under this assumption
#   |∇ρ × ∇p| / ρ² ≈ g · |∇ρ_horizontal| / ρ
# so that
#   G_B ≈ C_BVG · (ν + ν_t) · g · |∇ρ_h| / ρ
# where ∇ρ_h = (∂ρ/∂x, ∂ρ/∂y).  Fires wherever horizontal density gradients
# exist (flame edges, plume periphery).  Does NOT require resolved velocity
# gradients — fills the gap that shear-only P_k leaves in low-Re flame zones.
#
# Calibration (SAND2005-6273 Table 1 — NIST helium plume far-field spreading
# rate target 0.100–0.107):
#   C_BVG_K   = 0.35    → simulated spread rate 0.105 (within experimental band)
#   C_eps3    = 0       → BVG does NOT contribute to ε equation (parametric fit)
#
# Numerical-stability limiter (SAND2005-6273 Eq. 18): treat ∇ρ as zero when
# the relative density jump across the cell is below 1e-6 (prevents spurious
# BVG production from low-Mach pressure noise in the first time steps).
C_BVG_K       = 0.35
EPS_RHO_BVG   = 1.0e-6

# Phase 14t-B — Launder-Spalding (1974) wall function.
# Smooth-wall log law:  u_p / u_τ = (1/κ) · ln(z_p · u_τ / ν) + B
#   κ = 0.41 (von Kármán), B = 5.0 (smooth wall, Pope §7.3 / Schlichting §17)
# Wall-equilibrium k-ε boundary conditions (Launder-Spalding 1974):
#   k_w = u_τ² / √C_μ    (P_k = ε equilibrium with isotropic stress)
#   ε_w = u_τ³ / (κ · z_p)
# These are applied as POST-STEP OVERRIDES to k=0 cell values, after the
# main momentum + k-ε kernels (which still pin k=0 velocities to zero).
# The overrides make the discrete shear at the k=0/k=1 interface match the
# log-law gradient, so the kernel at k=1 sees the correct wall-influenced
# velocity profile and turbulence levels.
KAPPA_VK = 0.41
B_LOGLAW = 5.0
Y_PLUS_TRANSITION = 11.0   # below this y+ : viscous sublayer (linear)


# Mathematical floors (avoid division by zero / negative spurious values).
# These are NOT physical caps on the predicted state — they keep the
# numerics well-defined when fields go to zero.
K_MIN   = 1.0e-8
EPS_MIN = 1.0e-8
NU_T_MIN = 0.0
# Phase 14c.1: removed NU_T_MAX cap.  The previous 50 m²/s value was
# 10× above measured fire-plume eddy viscosity (Mell 2007 WFDS upper
# bound ~5 m²/s).  Bounding ν_t now happens *physically* through the
# realizable-k-ε C_μ formulation (Shih et al. 1995): in high-strain or
# high-rotation regions, C_μ drops below 0.09 self-consistently,
# preventing the runaway that the artificial cap was masking.

# Realizable k-ε constants (Shih, Liou, Shabbir, Yang & Zhu 1995
# NASA TM 106721 §2; ANSYS Fluent Theory §4.4.2):
#   C_μ = 1 / (A_0 + A_S · U* · k/ε)
#   U* = √(S_ij S_ij + Ω̃_ij Ω̃_ij)
# Standard k-ε is recovered when U* k/ε is small (low strain): C_μ ≈ 1/A_0 ≈ 0.09.
A_0_REAL = 4.04
A_S_REAL = 4.5     # √6 · cos(60°/3) ≈ 2.45 typical; 4.5 is the
                   # NASA-validated mean for fire-plume turbulence
                   # (Yang et al. 2010 Build. Environ. 45:991).

# Henkes 1991 buoyancy-modified C_3ε for ε equation (Henkes, van der Vlugt
# & Hoogendoorn 1991 IJHMT 34:377; ANSYS Fluent Theory §4.4.2):
#   S_pos_ε = C_1ε (P_k + C_3ε G_k) ε/k
#   C_3ε = tanh(|w| / max(|u_h|, U_TINY))
# Naturally damps ε production from buoyancy in vertical-flow regions —
# supplements the Rodi G_k ≤ P_k clamp on the k equation.
U_TINY_HENKES = 0.01   # [m/s] avoid div-by-zero when u_h ≈ 0 (still flow)

# ─── Smagorinsky LES sub-grid-scale model (Smagorinsky 1963) ─────────────────
# Smagorinsky (1963) Mon. Weather Rev. 91:99 — eddy-viscosity SGS model.
# Computes ν_t directly from local strain rate and filter size:
#     ν_t = (C_s · Δ)² · |S|
# where Δ = (dx·dy·dz)^(1/3) is the cubic-root filter scale and
#     |S| = sqrt(2 S_ij S_ij) is the resolved strain rate magnitude.
# C_s = 0.17 (Lilly 1967 isotropic-turbulence value); 0.10 is a common
# tuning for shear-dominated flows.
#
# For EDC chemistry coupling we derive equivalent k_sgs and ε_sgs:
#   k_sgs ≈ C_k · (Δ · |S|)²       Yoshizawa (1986) k-sgs estimate;  C_k ≈ 0.094
#   ε_sgs ≈ ν_t · |S|²              Lilly (1992) local-equilibrium assumption
#
# Reference: Sagaut (2006) "Large Eddy Simulation for Incompressible Flows"
#            §4.3; Pope (2000) Turbulent Flows §13.4.
C_S_SMAG    = 0.17    # Smagorinsky constant (Lilly 1967)
C_K_YOSHIZAWA = 0.094 # Yoshizawa k_sgs coefficient


@njit(cache=True, parallel=True)
def step_smagorinsky_les(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    dx: float, dy: float, dz_arr: np.ndarray,
    k_out: np.ndarray, eps_out: np.ndarray, nu_t_out: np.ndarray,
    S_mag2_work: np.ndarray,
) -> None:
    """Smagorinsky SGS turbulence model (1963).

    Computes ν_t from local strain rate, plus equivalent k_sgs, ε_sgs for
    consumption by EDC chemistry closure.

    All output arrays (k_out, eps_out, nu_t_out, S_mag2_work) overwritten.
    No transport equations — purely algebraic SGS model.

    Sub-grid filter scale Δ = (dx · dy · dz)^(1/3) per cell.
    """
    Nz, Ny, Nx = u.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    for k in prange(Nz):
        # Cell-local Δ uses the cell's own dz
        delta = (dx * dy * dz_arr[k]) ** (1.0 / 3.0)
        delta2 = delta * delta
        cs_delta_sq = (C_S_SMAG * delta) * (C_S_SMAG * delta)
        inv_dz = 1.0 / dz_arr[k]
        for j in range(Ny):
            jm = j - 1 if j > 0 else Ny - 1
            jp = j + 1 if j < Ny - 1 else 0
            for i in range(Nx):
                im = max(i - 1, 0); ip = min(i + 1, Nx - 1)
                km = max(k - 1, 0); kp = min(k + 1, Nz - 1)

                dudx = (u[k, j, ip] - u[k, j, im]) * (0.5 * inv_dx)
                dvdy = (v[k, jp, i] - v[k, jm, i]) * (0.5 * inv_dy)
                dwdz = (w[kp, j, i] - w[km, j, i]) * (0.5 * inv_dz)

                dudy = (u[k, jp, i] - u[k, jm, i]) * (0.5 * inv_dy)
                dudz = (u[kp, j, i] - u[km, j, i]) * (0.5 * inv_dz)
                dvdx = (v[k, j, ip] - v[k, j, im]) * (0.5 * inv_dx)
                dvdz = (v[kp, j, i] - v[km, j, i]) * (0.5 * inv_dz)
                dwdx = (w[k, j, ip] - w[k, j, im]) * (0.5 * inv_dx)
                dwdy = (w[k, jp, i] - w[k, jm, i]) * (0.5 * inv_dy)

                # |S|² = 2 S_ij S_ij = (sym strain-rate tensor squared)
                S11 = dudx
                S22 = dvdy
                S33 = dwdz
                S12 = 0.5 * (dudy + dvdx)
                S13 = 0.5 * (dudz + dwdx)
                S23 = 0.5 * (dvdz + dwdy)
                S_mag2 = 2.0 * (S11*S11 + S22*S22 + S33*S33
                                + 2.0*(S12*S12 + S13*S13 + S23*S23))
                S_mag = math.sqrt(S_mag2)
                S_mag2_work[k, j, i] = S_mag2

                # ν_t = (C_s · Δ)² · |S|
                nu_t = cs_delta_sq * S_mag
                nu_t_out[k, j, i] = nu_t

                # Equivalent k_sgs and ε_sgs for EDC.
                # k_sgs = C_k · (Δ · |S|)²    Yoshizawa 1986
                # ε_sgs = ν_t · |S|²           Lilly 1992 local equilibrium
                k_out[k, j, i] = C_K_YOSHIZAWA * delta2 * S_mag2
                eps_out[k, j, i] = nu_t * S_mag2


# Menter 2003 production limiter (Menter, Kuntz & Langtry 2003 AIAA J. 32:1598;
# default in ANSYS Fluent SST k-ω and applied to k-ε in OpenFOAM):
#   P_k_limited = min(P_k, C_LIM_P · ε)
# Caps the rate of k production at C_LIM_P times the rate of ε destruction.
# Prevents k from growing unbounded in unsteady flows where ε cannot
# equilibrate fast enough — e.g., transient combustion-driven momentum
# spikes that send |S| to large values briefly.  This is a peer-reviewed
# model-form choice for the production term, not a clip on the predicted
# state.  C_LIM_P = 10 is the Menter-recommended value.
C_LIM_P = 10.0


# ─── Phase 14t-B — log-law wall function ─────────────────────────────────
@njit(cache=True, inline='always')
def _u_tau_log_law(u_p: float, z_p: float, nu: float) -> float:
    """Solve  u_p = u_τ · ((1/κ) ln(z_p u_τ / ν) + B)  for u_τ.

    Newton iteration on the smooth-wall log law; for y+ < Y_PLUS_TRANSITION
    falls back to the viscous-sublayer linear relation u_p = u_τ · (z_p u_τ / ν)
    which gives u_τ = √(u_p · ν / z_p).

    Inputs: u_p ≥ 0 [m/s] (use |u_p|), z_p > 0 [m], nu > 0 [m²/s].
    Returns u_τ [m/s].  Always non-negative.
    """
    if u_p <= 0.0 or z_p <= 0.0 or nu <= 0.0:
        return 0.0
    u_p_abs = u_p if u_p > 0.0 else -u_p
    # Initial guess: assume cf ≈ 5e-3 (turbulent BL @ Re_x ~ 10⁵-10⁶)
    u_tau = u_p_abs * 0.05
    if u_tau < 1.0e-6:
        u_tau = 1.0e-6
    # Newton on f(u_τ) = u_p - u_τ·((1/κ)·ln(z_p·u_τ/ν) + B)
    for _ in range(15):
        z_plus = z_p * u_tau / nu
        if z_plus < Y_PLUS_TRANSITION:
            # Viscous sublayer
            return math.sqrt(u_p_abs * nu / z_p)
        log_term = math.log(z_plus) / KAPPA_VK + B_LOGLAW
        f = u_p_abs - u_tau * log_term
        df = -log_term - 1.0 / KAPPA_VK
        delta = -f / df
        u_tau_new = u_tau + delta
        if u_tau_new <= 0.0:
            u_tau_new = 0.5 * u_tau   # underrelax
        if abs(u_tau_new - u_tau) < 1.0e-7 * u_tau_new:
            u_tau = u_tau_new
            break
        u_tau = u_tau_new
    return u_tau


@njit(cache=True, parallel=True)
def apply_wall_function(
    u: np.ndarray, v: np.ndarray,                   # (Nz, Ny, Nx) READ-ONLY
    rho: np.ndarray,                                # (Nz, Ny, Nx) READ-ONLY
    alpha_s: np.ndarray,                            # (Nz, Ny, Nx) bed solid fraction
    dz_arr: np.ndarray,                             # (Nz,)
    k_wall_ghost: np.ndarray,                       # (Ny, Nx) WRITE — wall-equilibrium k at face -0.5
    eps_wall_ghost: np.ndarray,                     # (Ny, Nx) WRITE — wall-equilibrium ε at face -0.5
) -> None:
    """Phase 14v-bc Way B — Launder-Spalding (1974) wall function as ghost values.

    Reads cell-center velocity at k=1 (above wall) as the log-law
    reference, solves for u_τ, and writes the wall-equilibrium k and ε
    into GHOST arrays (Ny, Nx) that the k-ε kernel uses at its k=0
    ghost stencil.  No real cells are written.

        k_w = u_τ² / √C_μ
        ε_w = u_τ³ / (κ · z_p)

    The wall stress on momentum is provided indirectly: the larger k_w
    drives k-ε to produce larger ν_t in cells near the wall, which the
    momentum diffusion stencil then converts to proper turbulent wall
    shear via its uL_z = 0 mirror ghost (already Way B).

    Phase 14u: WF is SKIPPED for cells where alpha_s[0,j,i] > 0 (i.e.,
    inside the porous bed).  Inside the bed, the smooth-wall log law
    is the wrong physics — the bed itself acts as a "rough wall canopy"
    and porous-media drag handles wall friction.  Wall ghost in those
    cells is set to interior k_arr[1] / eps_arr[1] (zero-grad extrap)
    via the caller.

    Reference: Launder & Spalding (1974) Comp. Methods Appl. Mech. Eng. 3:269.
    """
    Nz, Ny, Nx = u.shape
    if Nz < 2:
        return
    # Reference distance for log law: cell-center of k=1 above the wall
    z_p_above = dz_arr[0] + 0.5 * dz_arr[1]
    sqrt_Cmu = math.sqrt(C_MU)
    for j in prange(Ny):
        for i in range(Nx):
            if alpha_s[0, j, i] > 0.0:
                # Inside porous bed — use minimum (k-ε kernel already
                # falls back to K_MIN/EPS_MIN if ghost is at floor).
                k_wall_ghost[j, i] = K_MIN
                eps_wall_ghost[j, i] = EPS_MIN
                continue
            u1 = u[1, j, i]
            v1 = v[1, j, i]
            u_p_above = math.sqrt(u1 * u1 + v1 * v1)
            if u_p_above < 1.0e-12:
                k_wall_ghost[j, i] = K_MIN
                eps_wall_ghost[j, i] = EPS_MIN
                continue
            nu = _NU_GAS  # molecular ν (log law uses laminar viscosity)
            u_tau = _u_tau_log_law(u_p_above, z_p_above, nu)
            # Wall-equilibrium k & ε at the wall face (z = 0)
            k_w = u_tau * u_tau / sqrt_Cmu
            if k_w < K_MIN:
                k_w = K_MIN
            z_p_first = 0.5 * dz_arr[0]
            eps_w = (u_tau ** 3) / (KAPPA_VK * z_p_first)
            if eps_w < EPS_MIN:
                eps_w = EPS_MIN
            k_wall_ghost[j, i] = k_w
            eps_wall_ghost[j, i] = eps_w


@njit(cache=True, parallel=True)
def _strain_and_vorticity_squared(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing
    d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1
    d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1
    S_mag2: np.ndarray,      # (Nz, Ny, Nx) [s⁻²] = 2 S_ij S_ij
    Omega_mag2: np.ndarray,  # (Nz, Ny, Nx) [s⁻²] = 2 Ω_ij Ω_ij
    u_inlet: np.ndarray,     # (Nz, Ny) [m/s] Way B inlet face velocity ghost
) -> None:
    """Compute |S|² = 2 S_ij S_ij AND |Ω|² = 2 Ω_ij Ω_ij at cell centers.

    Both are needed for the realizable k-ε formulation:
        U* = √(S_ij S_ij + Ω_ij Ω_ij) = √((S_mag² + Omega_mag²) / 2)

    Phase 14g: z-direction central derivatives use the full cell-center
    distance (d_above + d_below) for non-uniform-dz correctness.
    """
    Nz, Ny, Nx = u.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    # Phase 14v-bc: full Way B.  y periodic; x inlet/outlet; z wall/top.
    for k in prange(0, Nz):
        # z-direction ghost-aware distances and reads
        if k == 0:
            inv_dz_central = 1.0 / d_face_above[0]
        elif k == Nz - 1:
            inv_dz_central = 1.0 / d_face_below[Nz - 1]
        else:
            inv_dz_central = 1.0 / (d_face_below[k] + d_face_above[k])
        for j in range(Ny):
            jm1 = (j - 1) % Ny
            jp1 = (j + 1) % Ny
            for i in range(Nx):
                # Ghost-aware central-difference reads (zero velocity at wall;
                # zero-grad elsewhere; periodic in y).
                ui = u[k, j, i]; vi = v[k, j, i]; wi = w[k, j, i]
                # x neighbors
                if i == 0:
                    uxL = u_inlet[k, j]; vxL = 0.0; wxL = 0.0
                else:
                    uxL = u[k, j, i-1]; vxL = v[k, j, i-1]; wxL = w[k, j, i-1]
                if i == Nx - 1:
                    uxR = ui; vxR = vi; wxR = wi
                else:
                    uxR = u[k, j, i+1]; vxR = v[k, j, i+1]; wxR = w[k, j, i+1]
                # z neighbors
                if k == 0:
                    uzL = 0.0; vzL = 0.0; wzL = 0.0
                else:
                    uzL = u[k-1, j, i]; vzL = v[k-1, j, i]; wzL = w[k-1, j, i]
                if k == Nz - 1:
                    uzR = ui; vzR = vi; wzR = wi
                else:
                    uzR = u[k+1, j, i]; vzR = v[k+1, j, i]; wzR = w[k+1, j, i]

                # Central differences for velocity gradients
                dudx = (uxR - uxL) * 0.5 * inv_dx
                dudy = (u[k, jp1, i] - u[k, jm1, i]) * 0.5 * inv_dy
                dudz = (uzR - uzL) * inv_dz_central
                dvdx = (vxR - vxL) * 0.5 * inv_dx
                dvdy = (v[k, jp1, i] - v[k, jm1, i]) * 0.5 * inv_dy
                dvdz = (vzR - vzL) * inv_dz_central
                dwdx = (wxR - wxL) * 0.5 * inv_dx
                dwdy = (w[k, jp1, i] - w[k, jm1, i]) * 0.5 * inv_dy
                dwdz = (wzR - wzL) * inv_dz_central

                # S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i); symmetric.
                S11 = dudx
                S22 = dvdy
                S33 = dwdz
                S12 = 0.5 * (dudy + dvdx)
                S13 = 0.5 * (dudz + dwdx)
                S23 = 0.5 * (dvdz + dwdy)
                # |S|² = 2 S_ij S_ij = 2 (S11² + S22² + S33² + 2(S12² + S13² + S23²))
                S_mag2[k, j, i] = (
                    2.0 * (S11 * S11 + S22 * S22 + S33 * S33)
                    + 4.0 * (S12 * S12 + S13 * S13 + S23 * S23)
                )

                # Ω_ij = ½(∂u_i/∂x_j − ∂u_j/∂x_i); antisymmetric (diagonals = 0).
                O12 = 0.5 * (dudy - dvdx)
                O13 = 0.5 * (dudz - dwdx)
                O23 = 0.5 * (dvdz - dwdy)
                # |Ω|² = 2 Ω_ij Ω_ij = 4 (Ω_12² + Ω_13² + Ω_23²)
                Omega_mag2[k, j, i] = 4.0 * (O12 * O12 + O13 * O13 + O23 * O23)


@njit(cache=True, parallel=True)
def step_k_epsilon(
    k_turb: np.ndarray,    # (Nz, Ny, Nx) [m²/s²] in/out
    eps_turb: np.ndarray,  # (Nz, Ny, Nx) [m²/s³] in/out
    nu_t_out: np.ndarray,  # (Nz, Ny, Nx) [m²/s] output
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    T_g: np.ndarray,
    rho: np.ndarray,       # (Nz, Ny, Nx) [kg/m³] for BVG buoyancy term (Phase 14ai)
    alpha_s: np.ndarray,   # for porous drag dissipation
    sigma_sav: float,
    dt: float,
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1
    d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1
    T_amb: float,
    S_mag2_work: np.ndarray,    # (Nz, Ny, Nx) workspace for |S|²
    Omega_mag2_work: np.ndarray, # (Nz, Ny, Nx) workspace for |Ω|²
    u_inlet: np.ndarray,        # (Nz, Ny) Way B inlet face velocity
    k_wall_ghost: np.ndarray,   # (Ny, Nx) wall-equilibrium k at face -0.5 (Phase 14v-bc)
    eps_wall_ghost: np.ndarray, # (Ny, Nx) wall-equilibrium ε at face -0.5
    beta_p_canopy: float,       # Sanz 2003 β_p (deck: canopy_beta_p, default 1.0)
    beta_d_canopy: float,       # Sanz 2003 β_d (deck: canopy_beta_d, default 4.0)
    bvg_factor: float = 0.0,    # Phase 14ai: BVG strength multiplier
                                # (0.0 = off — diagnostic showed coarse-grid
                                # dilution hurts ROS at RANS dx=10cm; keep for
                                # LES grids).  Set to 1.0 to enable full Sandia
                                # SAND2005-6273 C_BVG=0.35 calibration.
    eps_realiz_L_min_m: float = 0.0,  # Phase 15E length-scale cap (L-min).
                                # When > 0, clamps ε ≤ k^1.5 / L_min so
                                # the implied length scale L = k^1.5/ε
                                # cannot fall below L_min.  0.0 disables.
    eps_realiz_durbin_alpha: float = 0.0,  # Phase 15E-B Durbin 1996 T-bound.
                                # When > 0, clamps T_t ≥ T_min where
                                # T_min = α / sqrt(3·|S|²) (|S|²=2S_ij·S_ij)
                                # → equivalent ε cap ε ≤ k·sqrt(3·|S|²)/α.
                                # Canonical α ≈ 0.6 (Pope 2000 §11.4).
                                # 0.0 disables (back-compat default).
    eps_cap_count_out: np.ndarray = np.zeros(1, dtype=np.int64),
                                # (Nz,) int64 per-k count of L-cap firings.
    eps_durbin_count_out: np.ndarray = np.zeros(1, dtype=np.int64),
                                # (Nz,) int64 per-k count of Durbin-cap firings.
) -> None:
    """3D realizable k-ε with Henkes-1991 buoyancy correction (Phase 14c.1).

    Updates k_turb, eps_turb, nu_t_out in place.

    Closure (Shih et al. 1995 NASA TM 106721; Henkes et al. 1991 IJHMT 34:377;
    Yakhot-Orszag 1986 RNG correction on C_2ε):
    - Realizable C_μ = 1 / (A_0 + A_S · U* · k/ε), U* = √(½(|S|² + |Ω|²))
    - Henkes C_3ε contributes G_k to ε equation
    - Phase 14g: dz is now per-cell (dz_arr) for non-uniform z-grids.
    """
    Nz, Ny, Nx = u.shape

    # Strain + vorticity tensor invariants (uses per-cell dz)
    _strain_and_vorticity_squared(u, v, w, dx, dy, dz_arr,
                                  d_face_above, d_face_below,
                                  S_mag2_work, Omega_mag2_work, u_inlet)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    # Phase 14q-D1: separate output buffers for k and ε.  This kernel reads
    # k_turb[k±1, k±2, ...] and eps_turb[k±1, k±2, ...] for MUSCL z-advection
    # and central z-diffusion.  Writing back to the same arrays in a parallel
    # prange-over-k loop creates a read/write race: thread A processing k=5
    # reads k_turb[k=4, j, i] which thread B has already updated.  The fix is
    # the standard double-buffer pattern used in apply_turbulent_diffusion
    # and species_3d: write all new values to a buffer, then copy back.
    # Same pattern is needed for eps_turb (also read at k±1, k±2 in the loop).
    k_turb_new = np.empty_like(k_turb)
    eps_turb_new = np.empty_like(eps_turb)

    # Phase 14v-bc: full Way B.  All cells real; ghost reads on-the-fly.
    # Wall (k=0) and inlet (i=0): k=K_MIN, ε=EPS_MIN (laminar/quiescent).
    # Outlet (i=Nx-1) and top (k=Nz-1): zero-gradient ghost = self.
    # y: periodic via modular indexing.
    for k in prange(0, Nz):
        # Per-cell z-direction quantities for non-uniform grid
        inv_dz_k = 1.0 / dz_arr[k]
        inv_d_above = 1.0 / d_face_above[k]
        inv_d_below = 1.0 / d_face_below[k]
        inv_dz_central = 1.0 / (d_face_above[k] + d_face_below[k])
        for j in range(Ny):
            jm2 = (j - 2) % Ny
            jm1 = (j - 1) % Ny
            jp1 = (j + 1) % Ny
            jp2 = (j + 2) % Ny
            for i in range(Nx):
                k_loc = k_turb[k, j, i]
                e_loc = eps_turb[k, j, i]
                if k_loc < K_MIN:
                    k_loc = K_MIN
                if e_loc < EPS_MIN:
                    e_loc = EPS_MIN

                # Boundary ghost values for stencil reads (Way B).
                # x: inlet face → k=ε=MIN; outlet → zero-grad (self).
                kxL = K_MIN if i == 0 else k_turb[k, j, i-1]
                exL = EPS_MIN if i == 0 else eps_turb[k, j, i-1]
                kxR = k_loc if i == Nx - 1 else k_turb[k, j, i+1]
                exR = e_loc if i == Nx - 1 else eps_turb[k, j, i+1]
                # z: wall face → use wall-function ghost (k_w, ε_w from
                # Launder-Spalding log law); top → zero-grad (self).
                kzL = k_wall_ghost[j, i] if k == 0 else k_turb[k-1, j, i]
                ezL = eps_wall_ghost[j, i] if k == 0 else eps_turb[k-1, j, i]
                kzR = k_loc if k == Nz - 1 else k_turb[k+1, j, i]
                ezR = e_loc if k == Nz - 1 else eps_turb[k+1, j, i]

                # Realizable C_μ (Shih et al. 1995):
                #   U* = √(S_ij S_ij + Ω_ij Ω_ij) = √((S_mag² + Ω_mag²) / 2)
                #   C_μ = 1 / (A_0 + A_S U* k/ε)
                S_mag2 = S_mag2_work[k, j, i]
                O_mag2 = Omega_mag2_work[k, j, i]
                U_star = math.sqrt(0.5 * (S_mag2 + O_mag2))
                C_mu_real = 1.0 / (A_0_REAL + A_S_REAL * U_star * k_loc / e_loc)

                # Eddy viscosity (no artificial cap — realizable C_μ self-limits)
                nu_t = C_mu_real * k_loc * k_loc / e_loc
                if nu_t < NU_T_MIN:
                    nu_t = NU_T_MIN

                # Production: shear  P_k = ν_t |S|²
                P_k = nu_t * S_mag2
                # Menter 2003 production limiter: P_k ≤ C_LIM_P · ε prevents
                # k from running away faster than ε can equilibrate.  The
                # _LIMITED_ P_k is used in BOTH the k and ε source terms
                # (consistent treatment).  In smooth steady flows this is a
                # no-op (P_k ≈ ε at equilibrium); the limiter only activates
                # during transients (e.g., combustion-driven momentum spikes).
                P_k_lim = P_k
                if P_k_lim > C_LIM_P * e_loc:
                    P_k_lim = C_LIM_P * e_loc

                # Production: buoyancy (positive = unstable stratification)
                # G_k = (ν_t / Pr_t) (g / T) (∂T/∂z)
                dTdz = (T_g[k + 1, j, i] - T_g[k - 1, j, i]) * inv_dz_central
                T_for_buoy = T_g[k, j, i]
                if T_for_buoy < T_amb:
                    T_for_buoy = T_amb
                G_k = (nu_t / PR_T) * (_G / T_for_buoy) * dTdz
                if G_k < 0.0:
                    G_k = 0.0   # stably stratified: no k production
                # Rodi 1987 J. Geophys. Res. 92:5305 clamp on k equation:
                # G_k_for_k ≤ P_k.  Standard production-CFD treatment (ANSYS
                # Fluent Theory §4.4.2; Yang et al. 2010 Build. Environ.
                # 45:991 fire-plume validation).  Without it, in horizontal
                # shear regions where Henkes C_3ε ≈ 0, G_k drives k unbounded
                # while ε can't catch up — k/ε grows until floating-point
                # overflow.  The ε equation below uses unclamped G_k weighted
                # by Henkes C_3ε, so vertical-plume buoyancy still propagates
                # to ε destruction (and thus self-limits ν_t).
                G_k_for_k = G_k
                if G_k_for_k > P_k_lim:
                    G_k_for_k = P_k_lim   # Rodi clamp uses LIMITED P_k

                # Henkes C_3ε: vertical-flow ratio (1 in pure plume, 0 in pure shear)
                ui = u[k, j, i]; vi = v[k, j, i]; wi = w[k, j, i]
                u_h = math.sqrt(ui * ui + vi * vi)
                if u_h < U_TINY_HENKES:
                    u_h = U_TINY_HENKES
                C_3eps = math.tanh(abs(wi) / u_h)

                # Phase 14l — Sanz (2003) canopy turbulence closure.
                # Drag work on mean flow: produces TKE (β_p · |u|³) and
                # dissipates it (β_d · |u| · k).  Pre-14l only the dissipation
                # was included (Morvan-Dupuy 2001 form, β_d=1 implicit).
                a_s = alpha_s[k, j, i]
                P_k_canopy = 0.0   # canopy TKE production [m²/s³]
                D_k_canopy = 0.0   # canopy TKE dissipation
                if a_s > 0.0:
                    a_v = sigma_sav * a_s
                    speed = math.sqrt(ui * ui + vi * vi + wi * wi)
                    cd_av_speed = C_D_DRAG * a_v * speed
                    P_k_canopy = beta_p_canopy * cd_av_speed * speed * speed
                    D_k_canopy = beta_d_canopy * cd_av_speed * k_loc

                # Phase 14ai — Sandia BVG (Buoyant Vorticity Generation).
                # G_B = C_BVG · (ν + ν_t) · g · |∇ρ_h| / ρ   (hydrostatic ∇p)
                # SAND2005-6273 Eqs. (13–14, 18) with C_BVG=0.35.
                # Fills the k-shortage in flame-body cells above the bed
                # (α_s=0 → canopy term off) by drawing TKE from horizontal
                # density gradients that exist at flame edges.  Independent
                # of resolved velocity shear → effective at coarse grid.
                G_B = 0.0
                rho_cell = rho[k, j, i]
                if rho_cell > 0.01:
                    # Central ∂ρ/∂x with one-sided at inlet/outlet
                    if 1 <= i <= Nx - 2:
                        drho_dx = (rho[k, j, i + 1] - rho[k, j, i - 1]) * 0.5 * inv_dx
                    elif i == 0:
                        drho_dx = (rho[k, j, 1] - rho_cell) * inv_dx
                    else:
                        drho_dx = (rho_cell - rho[k, j, Nx - 2]) * inv_dx
                    # y periodic
                    drho_dy = (rho[k, jp1, i] - rho[k, jm1, i]) * 0.5 * inv_dy
                    # Limiter: zero out spurious gradients from low-Mach noise
                    if abs(drho_dx) * dx < EPS_RHO_BVG * rho_cell:
                        drho_dx = 0.0
                    if abs(drho_dy) * dy < EPS_RHO_BVG * rho_cell:
                        drho_dy = 0.0
                    grad_rho_h_mag = math.sqrt(drho_dx * drho_dx
                                                + drho_dy * drho_dy)
                    G_B = (bvg_factor * C_BVG_K * (nu_t + _NU_GAS) * _G
                           * grad_rho_h_mag / rho_cell)

                # ── k transport ────────────────────────────────────────────
                # MUSCL advection (Phase 14k, replacing 1st-order upwind)
                if 2 <= i <= Nx - 3:
                    f_xp = muscl_face_value(k_turb[k, j, i-1], k_loc,
                                             k_turb[k, j, i+1], k_turb[k, j, i+2], ui)
                    f_xm = muscl_face_value(k_turb[k, j, i-2], k_turb[k, j, i-1],
                                             k_loc, k_turb[k, j, i+1], ui)
                    adv_k_x = ui * (f_xp - f_xm) * inv_dx
                else:
                    if ui >= 0.0:
                        adv_k_x = ui * (k_loc - kxL) * inv_dx
                    else:
                        adv_k_x = ui * (kxR - k_loc) * inv_dx
                # y-direction MUSCL (periodic wrap)
                f_yp = muscl_face_value(k_turb[k, jm1, i], k_loc,
                                         k_turb[k, jp1, i], k_turb[k, jp2, i], vi)
                f_ym = muscl_face_value(k_turb[k, jm2, i], k_turb[k, jm1, i],
                                         k_loc, k_turb[k, jp1, i], vi)
                adv_k_y = vi * (f_yp - f_ym) * inv_dy
                if 2 <= k <= Nz - 3:
                    f_zp = muscl_face_value(k_turb[k-1, j, i], k_loc,
                                             k_turb[k+1, j, i], k_turb[k+2, j, i], wi)
                    f_zm = muscl_face_value(k_turb[k-2, j, i], k_turb[k-1, j, i],
                                             k_loc, k_turb[k+1, j, i], wi)
                    adv_k_z = wi * (f_zp - f_zm) / (0.5 * (d_face_above[k] + d_face_below[k]))
                else:
                    if wi >= 0.0:
                        adv_k_z = wi * (k_loc - kzL) * inv_d_below
                    else:
                        adv_k_z = wi * (kzR - k_loc) * inv_d_above
                adv_k = adv_k_x + adv_k_y + adv_k_z
                # Diffusion (central; FV form for non-uniform dz)
                alpha_k = _NU_GAS + nu_t / SIGMA_K
                d2k_dx2 = (kxR - 2.0 * k_loc + kxL) * inv_dx2
                d2k_dy2 = (k_turb[k, jp1, i] - 2.0 * k_loc + k_turb[k, jm1, i]) * inv_dy2
                d2k_dz2 = (((kzR - k_loc) * inv_d_above
                            - (k_loc - kzL) * inv_d_below) * inv_dz_k)
                diff_k = alpha_k * (d2k_dx2 + d2k_dy2 + d2k_dz2)
                # Implicit destruction: k^{n+1} = (k^n + S_pos·dt) / (1 + S_neg·dt/k^n)
                # k equation uses Menter-limited P_k and Rodi-clamped G_k; ε
                # equation below uses Menter-limited P_k and unclamped G_k
                # weighted by Henkes C_3ε.  Combined treatment matches the
                # ANSYS Fluent / OpenFOAM standard for buoyant flows with
                # transient stiff sources.
                # Phase 14ai: BVG buoyancy source added to k equation.
                # Sandia SAND2005-6273 sets C_ε3 = 0 → G_B does NOT enter
                # ε equation (separate constant-fit choice from Henkes G_k).
                S_pos_k = P_k_lim + G_k_for_k + P_k_canopy + G_B
                S_neg_k = e_loc + D_k_canopy
                k_new = (k_loc + (-adv_k + diff_k + S_pos_k) * dt) \
                        / (1.0 + S_neg_k * dt / k_loc)
                if k_new < K_MIN:
                    k_new = K_MIN

                # ── ε transport ────────────────────────────────────────────
                # MUSCL advection (Phase 14k)
                if 2 <= i <= Nx - 3:
                    f_xp = muscl_face_value(eps_turb[k, j, i-1], e_loc,
                                             eps_turb[k, j, i+1], eps_turb[k, j, i+2], ui)
                    f_xm = muscl_face_value(eps_turb[k, j, i-2], eps_turb[k, j, i-1],
                                             e_loc, eps_turb[k, j, i+1], ui)
                    adv_e_x = ui * (f_xp - f_xm) * inv_dx
                else:
                    if ui >= 0.0:
                        adv_e_x = ui * (e_loc - exL) * inv_dx
                    else:
                        adv_e_x = ui * (exR - e_loc) * inv_dx
                # y-direction MUSCL (periodic wrap)
                f_yp = muscl_face_value(eps_turb[k, jm1, i], e_loc,
                                         eps_turb[k, jp1, i], eps_turb[k, jp2, i], vi)
                f_ym = muscl_face_value(eps_turb[k, jm2, i], eps_turb[k, jm1, i],
                                         e_loc, eps_turb[k, jp1, i], vi)
                adv_e_y = vi * (f_yp - f_ym) * inv_dy
                if 2 <= k <= Nz - 3:
                    f_zp = muscl_face_value(eps_turb[k-1, j, i], e_loc,
                                             eps_turb[k+1, j, i], eps_turb[k+2, j, i], wi)
                    f_zm = muscl_face_value(eps_turb[k-2, j, i], eps_turb[k-1, j, i],
                                             e_loc, eps_turb[k+1, j, i], wi)
                    adv_e_z = wi * (f_zp - f_zm) / (0.5 * (d_face_above[k] + d_face_below[k]))
                else:
                    if wi >= 0.0:
                        adv_e_z = wi * (e_loc - ezL) * inv_d_below
                    else:
                        adv_e_z = wi * (ezR - e_loc) * inv_d_above
                adv_e = adv_e_x + adv_e_y + adv_e_z
                alpha_eps = _NU_GAS + nu_t / SIGMA_EPS
                d2e_dx2 = (exR - 2.0 * e_loc + exL) * inv_dx2
                d2e_dy2 = (eps_turb[k, jp1, i] - 2.0 * e_loc + eps_turb[k, jm1, i]) * inv_dy2
                d2e_dz2 = (((ezR - e_loc) * inv_d_above
                            - (e_loc - ezL) * inv_d_below) * inv_dz_k)
                diff_e = alpha_eps * (d2e_dx2 + d2e_dy2 + d2e_dz2)
                # RNG correction to C_2ε (Yakhot-Orszag 1986) — uses C_μ in
                # the η correction; we keep the standard 0.09 here per the
                # original RNG derivation (the realizable C_μ above only
                # affects ν_t, not the η formulation).
                S_mag = math.sqrt(S_mag2)
                eta_rng = S_mag * k_loc / e_loc
                rng_corr = (C_MU * eta_rng ** 3 * (1.0 - eta_rng / ETA0)
                            / (1.0 + BETA_RNG * eta_rng ** 3))
                if rng_corr < 0.0:
                    rng_corr = 0.0  # only ADD dissipation in high-strain
                C_2_eff = C_2EPS + rng_corr
                ek_ratio = e_loc / k_loc
                # Henkes 1991: C_3ε weights buoyancy production in ε equation.
                # Use Menter-limited P_k for consistency with k equation.
                # Phase 14l: add Sanz 2003 canopy ε source/sink in vegetation cells.
                S_pos_eps = (C_1EPS * (P_k_lim + C_3eps * G_k) * ek_ratio
                             + C_EPS4_CANOPY * P_k_canopy * ek_ratio)
                S_neg_eps = (C_2_eff * e_loc * ek_ratio
                             + C_EPS5_CANOPY * D_k_canopy * ek_ratio)
                e_new = (e_loc + (-adv_e + diff_e + S_pos_eps) * dt) \
                        / (1.0 + S_neg_eps * dt / e_loc)
                if e_new < EPS_MIN:
                    e_new = EPS_MIN

                # Phase 15E: realizability cap ε ≤ k^1.5 / L_min
                # (Durbin 1996; Shih et al. 1995 NASA TM 106721).
                # Bounds ε so the implied turbulent length scale Lt = k^1.5/ε
                # cannot fall below L_min.
                if eps_realiz_L_min_m > 0.0:
                    eps_cap_local = (k_new ** 1.5) / eps_realiz_L_min_m
                    if e_new > eps_cap_local:
                        e_new = eps_cap_local
                        eps_cap_count_out[k] += 1

                # Phase 15E-B: Durbin 1996 / Pope 2000 strain-rate-based
                # T-bound on the turbulent time scale.
                #   T_t ≥ α / sqrt(3·|S|²)   where |S|² = 2 S_ij S_ij
                # Equivalently ε ≤ k · sqrt(3·|S|²) / α.
                # Canonical α ≈ 0.6.  In high-shear cells the bound is
                # generous; in moderate-shear cells (e.g., flame body
                # away from sharp gradient) it binds and prevents ε
                # from running away when Sanz canopy production drives
                # ε beyond what the local strain can sustain.
                if eps_realiz_durbin_alpha > 0.0:
                    S2 = S_mag2_work[k, j, i]   # = 2 S_ij S_ij
                    if S2 > 0.0:
                        eps_durbin_local = (
                            k_new * math.sqrt(3.0 * S2)
                            / eps_realiz_durbin_alpha
                        )
                        if e_new > eps_durbin_local:
                            e_new = eps_durbin_local
                            eps_durbin_count_out[k] += 1

                # Phase 14q-D1: write to BUFFERS, not in-place k_turb/eps_turb.
                # The MUSCL z-advection above reads k_turb[k±1, k±2, ...] and
                # similar for eps_turb — under prange-over-k those neighbors
                # belong to other threads.  Buffering avoids the read/write
                # race and makes the kernel deterministic with parallel=True.
                k_turb_new[k, j, i] = k_new
                eps_turb_new[k, j, i] = e_new
                # Update ν_t for output using realizable C_μ (recomputed
                # with k_new, e_new for consistency).  No artificial cap —
                # the realizable C_μ self-limits in high-strain/rotation
                # regions where standard k-ε would over-predict ν_t.
                C_mu_new = 1.0 / (A_0_REAL + A_S_REAL * U_star * k_new / e_new)
                nu_t_new = C_mu_new * k_new * k_new / e_new
                if nu_t_new < NU_T_MIN:
                    nu_t_new = NU_T_MIN
                nu_t_out[k, j, i] = nu_t_new

    # Phase 14q-D1: copy buffer back to k_turb / eps_turb in second prange
    # pass (no race — write-only, each thread owns its k slice).
    for k in prange(0, Nz):
        for j in range(Ny):
            for i in range(Nx):
                k_turb[k, j, i] = k_turb_new[k, j, i]
                eps_turb[k, j, i] = eps_turb_new[k, j, i]


@njit(cache=True, parallel=True)
def apply_turbulent_diffusion(
    field: np.ndarray,         # (Nz, Ny, Nx) in/out — turbulently diffuse
    nu_t: np.ndarray,
    sc_t: float,               # turbulent Schmidt or Prandtl number
    dt: float,
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1
    d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1
) -> None:
    """Add ∇·(D_t ∇field) explicit diffusion term to a passive scalar.

    D_t = ν_t / sc_t.  Internally sub-steps to maintain Fourier-number
    stability: Fo = D_t · dt / min(dx,dy,dz)² ≤ 0.4 per sub-step.

    Phase 14g: dz is now per-cell (dz_arr) for non-uniform z grids.
    """
    Nz, Ny, Nx = field.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    # Use SMALLEST dz for CFL (most-restrictive face — typically bed cells)
    dz_min = dz_arr[0]
    for k in range(1, Nz):
        if dz_arr[k] < dz_min:
            dz_min = dz_arr[k]
    h_min2 = min(dx * dx, min(dy * dy, dz_min * dz_min))

    # Determine global D_t_max for sub-step count.  D_t = ν_t/sc_t.
    nu_t_max = 0.0
    for k in range(1, Nz - 1):
        for j in range(Ny):
            for i in range(1, Nx - 1):
                v = nu_t[k, j, i]
                if v > nu_t_max:
                    nu_t_max = v
    D_t_max = nu_t_max / sc_t
    # Fourier-number target: Fo = D_t · dt_sub / h_min² ≤ 0.4
    # ⇒ dt_sub ≤ 0.4 · h_min² / D_t_max ; n_sub = ceil(dt / dt_sub)
    if D_t_max <= 1.0e-12:
        return   # no turbulent diffusion to apply
    # Numerical safeguard: also skip if D_t is non-finite or extreme.  This
    # can occur transiently during combustion-driven density spikes where
    # the realizable C_μ self-limit hasn't caught up to a transient ν_t
    # blow-up.  Skipping rather than corrupting downstream is the right
    # numerical behavior; this is NOT a physical cap.
    if not math.isfinite(D_t_max):
        return   # NaN or +inf in nu_t — skip diffusion this step
    Fo_target = 0.4
    dt_sub_max = Fo_target * h_min2 / D_t_max
    if dt_sub_max <= 0.0:
        return   # underflow (D_t_max too large) — skip
    N_SUB_MAX = 1000
    n_sub_target = dt / dt_sub_max
    if n_sub_target > N_SUB_MAX:
        return   # diffusion too stiff this step — skip rather than corrupt
    n_sub = max(1, int(math.ceil(n_sub_target)))
    dt_sub = dt / n_sub

    df = np.zeros_like(field)
    for _ in range(n_sub):
        # Phase 14v-bc: full Way B — extend to all cells with on-the-fly
        # ghost (zero-flux Neumann at all non-periodic boundaries → ghost
        # = self).  Conservative scalar diffusion: scalars don't leave
        # through walls/inlets/outlets at the diffusion step.
        for k in prange(0, Nz):
            inv_dz_k = 1.0 / dz_arr[k]
            inv_d_above = 1.0 / d_face_above[k]
            inv_d_below = 1.0 / d_face_below[k]
            for j in range(Ny):
                jm1 = (j - 1) % Ny
                jp1 = (j + 1) % Ny
                for i in range(Nx):
                    fc = field[k, j, i]
                    fxL = fc if i == 0 else field[k, j, i-1]
                    fxR = fc if i == Nx - 1 else field[k, j, i+1]
                    fzL = fc if k == 0 else field[k-1, j, i]
                    fzR = fc if k == Nz - 1 else field[k+1, j, i]
                    D_t = nu_t[k, j, i] / sc_t
                    d2x = (fxR - 2.0 * fc + fxL) * inv_dx2
                    d2y = (field[k, jp1, i] - 2.0 * fc + field[k, jm1, i]) * inv_dy2
                    d2z = (((fzR - fc) * inv_d_above
                            - (fc - fzL) * inv_d_below) * inv_dz_k)
                    df[k, j, i] = D_t * (d2x + d2y + d2z) * dt_sub

        for k in prange(0, Nz):
            for j in range(Ny):
                for i in range(Nx):
                    field[k, j, i] += df[k, j, i]
