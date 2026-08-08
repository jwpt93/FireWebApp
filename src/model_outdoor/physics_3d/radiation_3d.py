"""Radiative heat transfer from flame to unburned solid (Albini 1985).

Approximation: the flame is a planar slab radiating at an effective
flame temperature T_flame.  Each unburned-solid cell receives flux

    q_inc(x) = ε_flame · σ · T_flame⁴ · F_view(x, L_f, θ_tilt)

where ``F_view`` is the differential view factor from a finite-height
inclined slab to a target cell (Albini 1985 Eq. 4–5; geometric form
of a tilted radiating wall).  The flux is then attenuated through the
porous fuel bed by Beer-Lambert:

    q_abs(k) = q_inc · σ_β · exp(−σ_β · z_below_k) · dz

with σ_β = σ_SAV · β = volumetric absorption coefficient (Albini's
"radiation absorption coefficient" derivation).

Dimension reduction: the y-dimension is uniform (integrated assumption
for an infinite-fire-line); each (j, i) column gets the same incident
flux from the burning column at (j, i_burning) directly upstream.

Ground-truth flame parameters:
- ε_flame = 0.9       (Quintiere 2006; soot-laden hydrocarbon flames)
- T_flame = 1200 K    (Byram 1959 mean grass-fire flame temperature;
                       Drysdale 2011 Table 11.5 wood/grass fires)

These default values are used unless the deck specifies otherwise.
The flame length L_f and tilt θ are computed externally (Byram 1959
intensity → length; flame_tilt_angle from boundary module).

Reference:
- Albini (1985) Combust. Sci. Tech. 42:229 — flame radiation slab model
- Byram (1959) USDA Forest Service — fireline intensity, flame length
- Quintiere (2006) Fundamentals of Fire Phenomena
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange


# Stefan-Boltzmann.
SIGMA_SB = 5.67e-8       # [W/m²/K⁴]

# Default flame radiation parameters.
T_FLAME_DEFAULT = 1200.0   # [K]
EPS_FLAME_DEFAULT = 0.9    # [-]

# Phase 13.W FVM radiation parameters.
KAPPA_SOOT_HOT = 0.5       # [1/m] hydrocarbon-flame soot extinction (Tien 1998 SFPE Handbook §1-4)
OMEGA_COMB_THRESH = 1.0e-3 # [kg/m³/s] combustion rate above which gas phase is luminous


@njit(cache=True, parallel=True)
def step_slab_radiation(
    T_s: np.ndarray,           # (Nz, Ny, Nx) [K] solid temperature
    alpha_s: np.ndarray,       # (Nz, Ny, Nx) [-] solid volume fraction
    burning_mask: np.ndarray,  # (Nz, Ny, Nx) [-] 1 where fire-front column
    sigma_sav: float,          # [1/m] fuel SAV
    L_f: float,                # [m] flame length (Byram)
    theta_tilt: float,         # [rad] flame tilt from vertical (positive = forward)
    T_flame: float,            # [K]
    eps_flame: float,          # [-]
    dx: float, dy: float, dz: float,
    q_rad_out: np.ndarray,     # (Nz, Ny, Nx) [W/m²] absorbed radiative flux per cell
) -> None:
    """Compute radiative flux absorbed by each solid cell.

    Approach: for each (k, j, i_target), find the nearest burning column
    upstream in -x; compute view-factor + Beer-Lambert attenuation;
    deposit the absorbed flux at q_rad_out[k, j, i_target].

    Cells with alpha_s == 0 receive zero (no fuel to absorb radiation).

    NOTE: this is a per-step pass and does not modify T_s; the caller
    integrates the absorbed flux into the solid energy equation (see
    coupling_3d.py).
    """
    Nz, Ny, Nx = T_s.shape

    # Volumetric absorption coefficient (Beer-Lambert) for the porous bed.
    # σ_β = σ_SAV · β, where β = α_s.  For uniform per-cell α_s this
    # gives the absorption per cell length dz.
    sin_t = math.sin(theta_tilt)
    cos_t = math.cos(theta_tilt)

    # Emissive power of the flame slab.
    E_flame = eps_flame * SIGMA_SB * T_flame ** 4   # [W/m²]

    # Number of integration segments along flame axis (per Beyler 2002
    # SFPE Handbook Ch.3-4 line-flame view factor; 10 segments is
    # adequate for the line-to-point integration accuracy needed here).
    N_FLAME_SEGS = 10

    for k in prange(Nz):
        for j in range(Ny):
            # Find the most-downwind (highest i) burning column in this
            # (k_any, j) ridge — radiation only travels forward in +x.
            # We look across ALL k for the burning mask (a column burns
            # if any of its bed cells is currently a flame source).
            i_burn = -1
            for ii in range(Nx - 1, -1, -1):
                # A column is "burning" if any cell in (j, ii) has burning_mask>0.
                col_burn = False
                for kk in range(Nz):
                    if burning_mask[kk, j, ii] > 0.0:
                        col_burn = True
                        break
                if col_burn:
                    i_burn = ii
                    break
            if i_burn < 0:
                # No burning column in this y-slice; no radiation.
                for i in range(Nx):
                    for kk in range(Nz):
                        q_rad_out[kk, j, i] = 0.0
                continue

            # Burn-edge x-coordinate (downstream face of last burning cell).
            x_burn = (i_burn + 1.0) * dx

            for i in range(Nx):
                if i <= i_burn:
                    # Burning or upstream of fire; no incoming radiation.
                    for kk in range(Nz):
                        q_rad_out[kk, j, i] = 0.0
                    continue

                # Distance from fire edge to target cell center.
                x_target = (i + 0.5) * dx
                dxn = x_target - x_burn
                if dxn < 0.5 * dx:
                    dxn = 0.5 * dx

                # Albini 2D infinite-fireline view factor (Albini 1985
                # Combust. Sci. Tech. 42:229).  This is the proper 2D
                # configuration factor for an infinite-y line flame at
                # angle θ from vertical, target on ground at distance dxn:
                #   r = dxn - L_f sin θ  (effective distance after tilt)
                #   F = 0.5 (1 - r/√(L_f² + r²))
                # F ∈ [0, 0.5]; saturates at 0.5 when r → 0 (target
                # under flame tip).
                #
                # NB: an earlier "proper line integral" attempt
                # (∫dF/ds along flame axis with cos²θ kernel) was
                # dimensionally inconsistent — it computed 3D point-
                # source view factor without proper y-integration,
                # giving values ~2.5× too small.  Reverted.
                r = dxn - L_f * sin_t
                if r < 1.0e-3:
                    r = 1.0e-3
                F_view = 0.5 * (1.0 - r / math.sqrt(L_f * L_f + r * r))

                # Incident flux at the target column (per unit horizontal area).
                q_inc = E_flame * F_view   # [W/m²]

                # Beer-Lambert vertical attenuation through the bed:
                # each layer absorbs σ_β · α_s · dz of incoming flux.
                # Top layer (k = Nz-1) has no attenuation above; lower
                # layers see flux reduced by the cumulative absorption.
                # Volumetric absorption coefficient for porous bed:
                #   σ_β = σ_SAV · α_s   (Drysdale 2011 §11.3.3, Modest 2003)
                # Previously used σ_SAV alone (per-fiber surface area), which
                # gives σ·dz >> 1 → f_abs ≈ 1.0 → all radiation deposited
                # in top bed cell, lower cells receive nothing.  Multiplying
                # by α_s (volume fraction of solid in the bed) converts
                # to per-bed-volume absorption coefficient.  For Cheney:
                #   Nat: σ_β = 2000·0.0021 = 4.28 1/m → σ_β·dz=0.40 → f=0.33
                #   Cut: σ_β = 3500·0.0059 = 20.7 1/m → σ_β·dz=0.78 → f=0.54
                # Distributes radiation across multiple bed cells, so
                # downstream cells don't pre-pyrolyze as aggressively
                # before the front arrives.  Bed-side analog of the
                # flame optical-thickness correction in Phase 13.S.
                a_bed = 0.0
                for kk_scan in range(Nz - 1, -1, -1):
                    if alpha_s[kk_scan, j, i] > 0.0:
                        a_bed = alpha_s[kk_scan, j, i]
                        break
                sigma_beta = sigma_sav * a_bed
                f_abs_per_cell = 1.0 - math.exp(-sigma_beta * dz)
                # Iterate top-down so each cell gets transmitted flux.
                q_remaining = q_inc
                for kk in range(Nz - 1, -1, -1):
                    if alpha_s[kk, j, i] > 0.0:
                        # Per-area absorbed flux this cell:
                        q_abs = q_remaining * f_abs_per_cell
                        q_rad_out[kk, j, i] = q_abs
                        q_remaining = q_remaining - q_abs
                    else:
                        # Buffer cell, no absorption.
                        q_rad_out[kk, j, i] = 0.0


# ─── Phase 13.W: cell-to-cell FVM radiation ──────────────────────────────────
#
# Replaces step_slab_radiation (Albini 1985 ground-target view-factor +
# vertical-descent Beer-Lambert).  Solves a 2-stream gray-gas radiative
# transfer equation along ±x and ±z, with each cell radiating from its
# own state (T_s + T_g blended by the corresponding extinction
# coefficients).
#
# This is a simplified FVM (1 ray per axis) — a coarsened version of
# the discrete-ordinates method used by FDS/WFDS (Mell et al. 2007;
# McGrattan et al. 2013 NIST FDS Tech Ref Vol.1 §6).  The closest
# wildland-fire precedent is Morvan & Dupuy (2004) Combust. Flame 138:199
# (porous-bed P1 approximation for grass).  General reference: Modest
# (2003) Radiative Heat Transfer 2nd ed. §16 (FVM/DOM).
#
# Why this replaces Albini-slab + Frankman flame contact:
#  - Albini 1985 treated the flame as an external slab radiator outside
#    the bed — a workaround for 2D Rothermel-style models.  In our 3D
#    PDE the flame is internal: cells with active combustion (Q_comb > 0)
#    have hot T_g and luminous soot, so they radiate to neighbors via
#    the FVM directly.  No need for an external slab.
#  - Frankman 2013 flame-contact heating was a phenomenological term
#    forcing radiative+convective flux on cells under the projected
#    flame body, since the 2D model couldn't represent flame impingement.
#    In 3D, the buffer cells above the bed where combustion happens
#    radiate downward to the leading-edge top bed cell naturally.
#
# Per-cell radiative state:
#   κ_solid = σ_SAV · α_s         [1/m]  (Modest §11; Drysdale §11.3.3)
#   κ_gas   = κ_soot · 𝟙(Q_comb > thresh)  (Tien 1998: 0.5/m for hot HC flames)
#   κ_tot   = κ_solid + κ_gas
#   T_rad⁴  = (κ_solid · T_s⁴ + κ_gas · T_g⁴) / κ_tot
#             (κ-weighted blackbody source, Modest Eq. 9.21)
#   B       = σ · T_rad⁴          [W/m²]  Stefan-Boltzmann emission
#
# 1D RTE in each direction (e.g. +x):
#   dF⁺/dx = -κ_tot · F⁺ + κ_tot · B
#   F⁺(x+dx) = F⁺(x)·τ + B·(1-τ),    τ = exp(-κ_tot · dx)
#   (closed-form for cell-wise constant κ, B)
#
# Boundary conditions: F = σ·T_amb⁴ on cold-sky (top z), ground (bottom z),
# inlet (i=0), outlet (i=Nx) — passive cool surroundings.
#
# Net per-cell absorbed flux (per unit horizontal area, [W/m²]):
#   q_x_per_horiz = ((F⁺_left - F⁺_right) + (F⁻_right - F⁻_left)) · dz/dx
#   q_z_per_horiz = ((F_up_below - F_up_above) + (F_dn_above - F_dn_below))
#   q_total       = q_x + q_z   (positive = absorbed, negative = net emission)
#
# The kernel splits q_total by phase (solid vs gas) using the κ_solid /
# κ_tot fraction.  Coupling consumes q_rad_solid_out as a NET source —
# it INCLUDES self-emission of the solid surfaces (as the cell radiates
# its own B to neighbors), so the legacy q_loss=ε·σ·(T_s⁴-T_amb⁴) term
# in coupling_3d.py must be DISABLED when use_fvm=True (or the cooling
# is double-counted).


@njit(cache=True, parallel=True)
def step_cell_radiation_fvm(
    T_s: np.ndarray,           # (Nz, Ny, Nx) [K] solid temperature
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K] gas temperature
    alpha_s: np.ndarray,       # (Nz, Ny, Nx) [-] solid volume fraction
    omega_comb: np.ndarray,    # (Nz, Ny, Nx) [kg/m³/s] combustion rate (luminance proxy)
    sigma_sav: float,          # [1/m] fuel SAV
    dx: float, dy: float, dz: float,
    T_amb: float,              # [K] ambient (cold sky/ground BC)
    q_rad_solid_out: np.ndarray,  # (Nz, Ny, Nx) [W/m²] net solid absorption
    q_rad_gas_out: np.ndarray,    # (Nz, Ny, Nx) [W/m²] net gas absorption
) -> None:
    """One FVM radiation step.  Outputs are NET (incoming − self-emission)."""
    Nz, Ny, Nx = T_s.shape
    sigma_T_amb4 = SIGMA_SB * T_amb ** 4

    # ── Per-cell extinction & blackbody source ─────────────────────────────
    kappa_solid = np.empty((Nz, Ny, Nx), dtype=np.float64)
    kappa_gas   = np.empty((Nz, Ny, Nx), dtype=np.float64)
    kappa_tot   = np.empty((Nz, Ny, Nx), dtype=np.float64)
    B           = np.empty((Nz, Ny, Nx), dtype=np.float64)

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                ks = sigma_sav * alpha_s[k, j, i]
                if omega_comb[k, j, i] > OMEGA_COMB_THRESH:
                    kg = KAPPA_SOOT_HOT
                else:
                    kg = 0.0
                kt = ks + kg
                if kt > 1.0e-9:
                    Trad4 = (ks * T_s[k, j, i] ** 4 + kg * T_g[k, j, i] ** 4) / kt
                else:
                    Trad4 = T_amb ** 4    # transparent gas, no radiative source
                kappa_solid[k, j, i] = ks
                kappa_gas[k, j, i]   = kg
                kappa_tot[k, j, i]   = kt
                B[k, j, i]           = SIGMA_SB * Trad4

    # ── x-direction 2-stream sweep ──────────────────────────────────────────
    F_x_fwd = np.empty((Nz, Ny, Nx + 1), dtype=np.float64)
    F_x_bck = np.empty((Nz, Ny, Nx + 1), dtype=np.float64)
    for k in prange(Nz):
        for j in range(Ny):
            F_x_fwd[k, j, 0] = sigma_T_amb4
            for i in range(Nx):
                tau = math.exp(-kappa_tot[k, j, i] * dx)
                F_x_fwd[k, j, i + 1] = F_x_fwd[k, j, i] * tau + B[k, j, i] * (1.0 - tau)
            F_x_bck[k, j, Nx] = sigma_T_amb4
            for i in range(Nx - 1, -1, -1):
                tau = math.exp(-kappa_tot[k, j, i] * dx)
                F_x_bck[k, j, i] = F_x_bck[k, j, i + 1] * tau + B[k, j, i] * (1.0 - tau)

    # ── z-direction 2-stream sweep ──────────────────────────────────────────
    F_z_up = np.empty((Nz + 1, Ny, Nx), dtype=np.float64)
    F_z_dn = np.empty((Nz + 1, Ny, Nx), dtype=np.float64)
    for j in prange(Ny):
        for i in range(Nx):
            F_z_up[0, j, i] = sigma_T_amb4
            for k in range(Nz):
                tau = math.exp(-kappa_tot[k, j, i] * dz)
                F_z_up[k + 1, j, i] = F_z_up[k, j, i] * tau + B[k, j, i] * (1.0 - tau)
            F_z_dn[Nz, j, i] = sigma_T_amb4
            for k in range(Nz - 1, -1, -1):
                tau = math.exp(-kappa_tot[k, j, i] * dz)
                F_z_dn[k, j, i] = F_z_dn[k + 1, j, i] * tau + B[k, j, i] * (1.0 - tau)

    # ── Net per-cell absorption, split by phase ─────────────────────────────
    inv_dx = 1.0 / dx
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                # x net absorption [W/m² of x-perpendicular cross-section]
                dF_x = ((F_x_fwd[k, j, i] - F_x_fwd[k, j, i + 1])
                        + (F_x_bck[k, j, i + 1] - F_x_bck[k, j, i]))
                # z net absorption [W/m² of horizontal cross-section]
                dF_z = ((F_z_up[k, j, i] - F_z_up[k + 1, j, i])
                        + (F_z_dn[k + 1, j, i] - F_z_dn[k, j, i]))
                # Convert x to per-horizontal-area: cross_x/horiz = dz/dx
                net_horiz = dF_x * dz * inv_dx + dF_z   # [W/m² horizontal]
                kt = kappa_tot[k, j, i]
                if kt > 1.0e-9:
                    f_solid = kappa_solid[k, j, i] / kt
                    f_gas   = kappa_gas[k, j, i] / kt
                else:
                    f_solid = 0.0
                    f_gas   = 0.0
                q_rad_solid_out[k, j, i] = net_horiz * f_solid
                q_rad_gas_out[k, j, i]   = net_horiz * f_gas


# ─── Phase 14a: P1 (Eddington) radiation solver ──────────────────────────────
#
# P1 spherical-harmonic approximation: I(r,Ω) ≈ I₀(r) + 3 I₁(r)·Ω.  Reduces
# the radiative-transfer equation (4D) to a single elliptic PDE for the mean
# intensity G(r) = ∫I dΩ:
#
#     ∇·(D ∇G) - κ G = -4π κ B,    D = 1/(3κ)
#
# where B = σT_rad⁴/π is the blackbody source per unit solid angle (W/m²·sr⁻¹)
# and κ is the absorption coefficient.  The radiation source per unit cell
# volume for the energy equation is then:
#
#     ∇·q_rad = κ(4π B - G)   [W/m³]   (positive = net emission)
#
# Net absorption per cell = κ(G - 4π B), which is what coupling sees.
#
# References:
# - Modest (2003) Radiative Heat Transfer 2nd ed. §15 — P1 derivation
# - Morvan & Dupuy (2004) Combust. Flame 138:199 — P1 in wildland fire
# - Larini, Giroud, Porterie & Loraud (1998) Combust. Sci. Tech. 134:153
#
# Discretization: 7-point finite-volume Laplacian with harmonic mean for D
# at faces (preserves diffusivity-jump behavior at α_s discontinuities).
# Boundary condition: Dirichlet G = 4σT_amb⁴ on x-inlet/outlet, ground, top
# (cold-sky equivalent black walls); periodic or zero-gradient on y per
# the spread-domain choice.  The coefficient matrix A is rebuilt and
# factorized each step (O(N log N) with splu) since κ depends on T_g, α_s
# and ω which all evolve.


class P1RadiationSolver:
    """P1 (Eddington) approximation solver for cell-to-cell radiation.

    Each call to ``solve`` rebuilds the sparse Laplacian-Helmholtz matrix
    using the current κ field, factorizes via splu, and returns the
    per-cell net solid/gas radiation absorption.

    Boundary convention: black walls at T_amb on x, z (ground+sky), with
    y per ``y_bc`` (periodic or zero-gradient).
    """

    def __init__(
        self,
        Nz: int, Ny: int, Nx: int,
        dy: float, dx: float,
        dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing
        d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1
        d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1
        y_bc: str = "periodic",
    ) -> None:
        self.Nz = Nz; self.Ny = Ny; self.Nx = Nx
        self.dy = dy; self.dx = dx
        self.dz_arr = dz_arr
        self.d_face_above = d_face_above
        self.d_face_below = d_face_below
        # Backward-compat scalar (= bed-cell dz)
        self.dz = float(dz_arr[0])
        self.y_bc = y_bc
        self.N = Nz * Ny * Nx

    def _idx(self, k: int, j: int, i: int) -> int:
        return (k * self.Ny + j) * self.Nx + i

    def solve(
        self,
        T_s: np.ndarray,         # (Nz, Ny, Nx) [K]
        T_g: np.ndarray,         # (Nz, Ny, Nx) [K]
        alpha_s: np.ndarray,     # (Nz, Ny, Nx) [-]
        omega_comb: np.ndarray,  # (Nz, Ny, Nx) [kg/m³/s]
        sigma_sav: float,
        T_amb: float,
        q_rad_solid_out: np.ndarray,
        q_rad_gas_out: np.ndarray,
    ) -> None:
        """Solve P1 equation for current state; fill per-cell solid/gas net flux."""
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        dx, dy = self.dx, self.dy
        dz_arr = self.dz_arr
        d_above = self.d_face_above
        d_below = self.d_face_below
        N = self.N

        # ── Per-cell radiative properties ──────────────────────────────────
        kappa_solid = sigma_sav * alpha_s
        kappa_gas = np.where(omega_comb > OMEGA_COMB_THRESH, KAPPA_SOOT_HOT, 0.0)
        kappa = kappa_solid + kappa_gas
        # Floor κ to avoid singular D = 1/(3κ); 1e-3 keeps buffer/ambient cells
        # near-transparent but matrix still well-conditioned.
        KAPPA_FLOOR = 1.0e-3
        kappa_safe = np.maximum(kappa, KAPPA_FLOOR)
        D = 1.0 / (3.0 * kappa_safe)

        # Source: T_rad blends T_s (solid radiator) and T_g (luminous gas) by κ.
        kappa_safe_for_T = np.where(kappa > 1.0e-9, kappa_safe, 1.0)  # avoid 0/0
        T_rad4 = np.where(
            kappa > 1.0e-9,
            (kappa_solid * T_s ** 4 + kappa_gas * T_g ** 4) / kappa_safe_for_T,
            T_amb ** 4,
        )
        # B in P1 is intensity per solid angle [W/m²/sr]:
        B = SIGMA_SB * T_rad4 / math.pi   # [W/m²/sr]

        # Boundary value: G_amb = 4πB_amb = 4σT_amb⁴ (black-wall blackbody)
        G_amb = 4.0 * SIGMA_SB * T_amb ** 4   # [W/m²]

        # ── Build sparse 7-point Helmholtz matrix ──────────────────────────
        # Equation per cell c:
        #   Σ_faces D_face · (G_n − G_c) / d_n² − κ_c · G_c = −4π · κ_c · B_c
        # Matrix form: A G = RHS
        #   A[c,c] = -Σ D_face/d² - κ_c
        #   A[c,n] = +D_face/d²  (interior neighbor)
        #   For Dirichlet ghost: A[c,c] -= D_face/d², RHS[c] -= D_face/d² · G_amb
        # Harmonic mean for face D: D_face = 2·D_c·D_n/(D_c + D_n)

        rows = np.empty(N * 7, dtype=np.int64)
        cols = np.empty(N * 7, dtype=np.int64)
        data = np.empty(N * 7, dtype=np.float64)
        rhs  = np.zeros(N, dtype=np.float64)
        ne = 0

        dx2 = dx * dx
        dy2 = dy * dy

        # Inline helpers
        idx = self._idx

        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    c = idx(k, j, i)
                    Dc = D[k, j, i]
                    diag = -kappa[k, j, i]
                    rhs_c = -4.0 * math.pi * kappa[k, j, i] * B[k, j, i]

                    # x faces
                    if i + 1 < Nx:
                        Df = 2.0 * Dc * D[k, j, i + 1] / (Dc + D[k, j, i + 1])
                        coef = Df / dx2
                        diag -= coef
                        rows[ne] = c; cols[ne] = idx(k, j, i + 1); data[ne] = coef; ne += 1
                    else:
                        diag -= Dc / dx2
                        rhs_c -= Dc / dx2 * G_amb
                    if i - 1 >= 0:
                        Df = 2.0 * Dc * D[k, j, i - 1] / (Dc + D[k, j, i - 1])
                        coef = Df / dx2
                        diag -= coef
                        rows[ne] = c; cols[ne] = idx(k, j, i - 1); data[ne] = coef; ne += 1
                    else:
                        diag -= Dc / dx2
                        rhs_c -= Dc / dx2 * G_amb

                    # y faces
                    if Ny > 1:
                        if self.y_bc == "periodic":
                            jp = (j + 1) % Ny
                            jm = (j - 1) % Ny
                            Df = 2.0 * Dc * D[k, jp, i] / (Dc + D[k, jp, i])
                            coef = Df / dy2
                            diag -= coef
                            rows[ne] = c; cols[ne] = idx(k, jp, i); data[ne] = coef; ne += 1
                            Df = 2.0 * Dc * D[k, jm, i] / (Dc + D[k, jm, i])
                            coef = Df / dy2
                            diag -= coef
                            rows[ne] = c; cols[ne] = idx(k, jm, i); data[ne] = coef; ne += 1
                        else:  # zero-gradient (edge_loss): no flux at y boundary
                            if j + 1 < Ny:
                                Df = 2.0 * Dc * D[k, j + 1, i] / (Dc + D[k, j + 1, i])
                                coef = Df / dy2
                                diag -= coef
                                rows[ne] = c; cols[ne] = idx(k, j + 1, i); data[ne] = coef; ne += 1
                            if j - 1 >= 0:
                                Df = 2.0 * Dc * D[k, j - 1, i] / (Dc + D[k, j - 1, i])
                                coef = Df / dy2
                                diag -= coef
                                rows[ne] = c; cols[ne] = idx(k, j - 1, i); data[ne] = coef; ne += 1

                    # z faces (Dirichlet at ground k=-1 and top k=Nz)
                    # Phase 14g: per-cell coefficients for non-uniform dz.
                    # The FV-Laplacian face coef is D_face / (dz[k] · d_above[k])
                    # where d_above[k] = 0.5(dz[k]+dz[k+1]) is the cell-center
                    # distance.  Reduces to D_face/dz² when dz uniform.
                    inv_dz_k = 1.0 / dz_arr[k]
                    coef_z_above_factor = inv_dz_k / d_above[k]
                    coef_z_below_factor = inv_dz_k / d_below[k]
                    if k + 1 < Nz:
                        Df = 2.0 * Dc * D[k + 1, j, i] / (Dc + D[k + 1, j, i])
                        coef = Df * coef_z_above_factor
                        diag -= coef
                        rows[ne] = c; cols[ne] = idx(k + 1, j, i); data[ne] = coef; ne += 1
                    else:
                        # Top boundary: ghost cell at distance d_above[k]
                        coef_g = Dc * coef_z_above_factor
                        diag -= coef_g
                        rhs_c -= coef_g * G_amb
                    if k - 1 >= 0:
                        Df = 2.0 * Dc * D[k - 1, j, i] / (Dc + D[k - 1, j, i])
                        coef = Df * coef_z_below_factor
                        diag -= coef
                        rows[ne] = c; cols[ne] = idx(k - 1, j, i); data[ne] = coef; ne += 1
                    else:
                        # Ground boundary
                        coef_g = Dc * coef_z_below_factor
                        diag -= coef_g
                        rhs_c -= coef_g * G_amb

                    rows[ne] = c; cols[ne] = c; data[ne] = diag; ne += 1
                    rhs[c] = rhs_c

        A = sp.coo_matrix(
            (data[:ne], (rows[:ne], cols[:ne])), shape=(N, N)
        ).tocsc()
        A.sum_duplicates()

        # ── Solve A · G = RHS ──────────────────────────────────────────────
        try:
            G_flat = spla.spsolve(A, rhs)
        except RuntimeError:
            # Fallback: iterative GMRES if direct solve fails
            G_flat, _ = spla.gmres(A, rhs, atol=1e-6)
        G = G_flat.reshape((Nz, Ny, Nx))

        # ── Per-cell net absorption ────────────────────────────────────────
        # Volumetric: κ(G - 4πB) [W/m³]; per horizontal area: × dz_arr[k]
        net_volumetric = kappa * (G - 4.0 * math.pi * B)
        net_per_horiz = net_volumetric * dz_arr.reshape(-1, 1, 1)

        # Split by phase
        f_solid = np.where(kappa > 1.0e-9, kappa_solid / kappa_safe, 0.0)
        f_gas   = np.where(kappa > 1.0e-9, kappa_gas   / kappa_safe, 0.0)
        np.copyto(q_rad_solid_out, net_per_horiz * f_solid)
        np.copyto(q_rad_gas_out,   net_per_horiz * f_gas)
