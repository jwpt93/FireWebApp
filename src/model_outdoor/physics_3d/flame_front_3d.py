"""Phase 14x — Level-set front + flame-body bootstrap for grass-fire propagation.

Companion to the resolved CFD: tracks the burning-front position via a
3D signed-distance field φ(x, y, z, t), advances v_n from CFD heat fluxes
to ahead-of-front cells, and provides masks for flame-body bootstrap
heating + ignition handover.

Architecture: option B2 (EBU + Bootstrap) — see
docs/phase14x_levelset_flame_body_plan.md.

Level-set evolution (Sethian 1999 Level Set Methods §6; Osher-Sethian 1988):

    ∂φ/∂t + v_n |∇φ| = 0

with v_n ≥ 0 (front advances forward only in our grass-fire setting).
Sign convention: φ < 0 burned, φ > 0 unburned, φ = 0 front.  3D field
(Nz, Ny, Nx) — uses 3D for generality (terrain, crown spread); reduces
naturally to a 2D-like front for our flat-bed infinite-line case.

Discretization: 1st-order Godunov upwind (Sethian 1999 §6.4):

    |∇φ|² = max(D⁻x, 0)² + min(D⁺x, 0)² + (same for y, z)
    where D±α are 1-sided differences

Reinitialization: Sussman et al. 1994 J. Comput. Phys. 114:146 —

    ∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0

iterated for ~5 substeps every ~10 outer dt's.  Restores |∇φ|=1 in a
narrow band around the front.

Front velocity v_n is CFD-derived for mesh-convergence:

    v_n = q_in_at_front / E_ign_per_area

where q_in is the heat flux delivered to ahead-of-front bed cells
(Frankman flame-tip convection + DOM forward radiation), integrated
over a constant-physical-size band (DX_VN_BAND = 0.20 m).  E_ign is
the energy per unit bed area required to ignite the next strip
(ρ_b × cp_s × h_bed × (T_ign − T_amb)).

This makes v_n grid-independent (the integration band has fixed
physical size, not fixed cell count) — addresses Phase 14r grid
convergence pathology of single-step Arrhenius CFD.

References:
- Sethian, J.A. (1999) Level Set Methods and Fast Marching Methods —
  textbook Godunov upwind scheme + reinit
- Osher, S. & Sethian, J.A. (1988) J. Comput. Phys. 79:12 — original
- Sussman, M., Smereka, P., Osher, S. (1994) J. Comput. Phys. 114:146 —
  reinit of level-set distance fields
- Mell, W. et al. (2007) IJWF 16:1 — WFDS level-set option (pattern)
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# ── Lit-grounded constants (no per-case tuning, Rule #5) ──────────────────────
T_FLAME_K        = 1500.0    # Drysdale 2011 §1.2.3 grass flame T
L_BURNOUT_M      = 0.50      # Albini 1985 grass burning-zone length
H_FLAME_FRANKMAN = 200.0     # W/m²/K — Frankman 2013 mid-range Table 2
DX_VN_BAND_M     = 0.20      # WFDS Mell 2007 §3.4 preheating band (legacy
                             # constant; wind-dependent via flame_tilt_band_m)
G_ACCEL          = 9.81      # m/s², used in Albini buoyancy velocity
WIND_MIDFLAME_FRAC = 0.723   # Cheney 1993: U_10 → U_1.5 reduction


def flame_tilt_band_m(
    u_10_m_s: float,
    L_flame: float = L_BURNOUT_M,
    g: float = G_ACCEL,
    midflame_frac: float = WIND_MIDFLAME_FRAC,
) -> float:
    """Albini 1981 flame-tilt projected band length.

    Geometric forward extent of a tilted buoyant flame onto the unburned
    bed.  Tilt angle θ from vertical satisfies:

        tan(θ) = U_mid / sqrt(2 g L_flame)

    where U_mid = U_10 · midflame_frac is the wind speed at half-flame
    height (Cheney 1993 0.723 factor for 10m → 1.5m).  The forward-tilted
    flame projects onto the bed over a horizontal extent of L_flame · sin(θ).

    Returns
    -------
    band_m : float
        Bed-level forward extent of flame contact [m].  At U_10 = 0 the
        band is zero (flame stands vertically).  At high wind θ → π/2
        and band → L_flame.

    References
    ----------
    Albini, F.A. (1981) "A model for the wind-blown flame from a line
        fire", Combustion and Flame 43:155.
    Nelson, R.M. Jr. (2002) IJWF 11:153 — same form.
    Cheney, N.P. et al. (1993) IJWF 3:31 — 10m→1.5m wind reduction.
    """
    u_buoy = (2.0 * g * L_flame) ** 0.5     # ≈ 3.13 m/s for L=0.5
    u_mid = max(u_10_m_s, 0.0) * midflame_frac
    tan_th = u_mid / u_buoy
    sin_th = tan_th / (1.0 + tan_th * tan_th) ** 0.5
    return L_flame * sin_th

LEVELSET_REINIT_INTERVAL = 10   # outer steps between reinit calls
REINIT_SUBSTEPS          = 5    # Sussman reinit substeps per call
DT_REINIT_FRAC           = 0.5  # τ-step = DT_REINIT_FRAC × min(dx,dy,dz)
PHI_INIT_UNBURNED        = 100.0   # large positive (well outside any narrow band)


@njit(cache=True, parallel=True)
def godunov_grad_norm(
    phi: np.ndarray,            # (Nz, Ny, Nx)
    dx: float, dy: float, dz_arr: np.ndarray,    # dz_arr (Nz,)
    grad_out: np.ndarray,       # (Nz, Ny, Nx)
) -> None:
    """Compute |∇φ| via Godunov-upwind for advection in normal direction.

    Used inside Sussman reinit (with sign-of-phi0 to pick which one-sided
    difference) and inside the v_n-driven advection step.

    For v_n > 0 (front advances forward, phi decreases from + to -):
        |∇φ|² = max(D⁻α, 0)² + min(D⁺α, 0)²  for α = x, y, z
    where D⁻α = (φ_i − φ_i-1)/Δα,  D⁺α = (φ_i+1 − φ_i)/Δα.

    This keeps the level-set evolution stable for v_n ≥ 0 (Sethian 1999 §6.4).
    """
    Nz, Ny, Nx = phi.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for k in prange(Nz):
        # vertical step uses neighbor dz; for first/last cell, use one-sided
        if k == 0:
            inv_dz_minus = 1.0 / dz_arr[0]
            inv_dz_plus  = 2.0 / (dz_arr[0] + dz_arr[1]) if Nz > 1 else 1.0 / dz_arr[0]
        elif k == Nz - 1:
            inv_dz_plus  = 1.0 / dz_arr[Nz - 1]
            inv_dz_minus = 2.0 / (dz_arr[Nz - 1] + dz_arr[Nz - 2])
        else:
            inv_dz_minus = 2.0 / (dz_arr[k] + dz_arr[k - 1])
            inv_dz_plus  = 2.0 / (dz_arr[k] + dz_arr[k + 1])

        for j in range(Ny):
            for i in range(Nx):
                phi_c = phi[k, j, i]

                # x-direction one-sided differences
                if i > 0:
                    d_minus_x = (phi_c - phi[k, j, i - 1]) * inv_dx
                else:
                    d_minus_x = 0.0
                if i < Nx - 1:
                    d_plus_x = (phi[k, j, i + 1] - phi_c) * inv_dx
                else:
                    d_plus_x = 0.0

                # y-direction (periodic-y for our case; treat both as interior)
                jm = j - 1 if j > 0 else Ny - 1
                jp = j + 1 if j < Ny - 1 else 0
                d_minus_y = (phi_c - phi[k, jm, i]) * inv_dy
                d_plus_y  = (phi[k, jp, i] - phi_c) * inv_dy

                # z-direction
                if k > 0:
                    d_minus_z = (phi_c - phi[k - 1, j, i]) * inv_dz_minus
                else:
                    d_minus_z = 0.0
                if k < Nz - 1:
                    d_plus_z = (phi[k + 1, j, i] - phi_c) * inv_dz_plus
                else:
                    d_plus_z = 0.0

                # Godunov upwind for v_n > 0 advection of phi (decreases at front)
                gx_p = max(d_minus_x, 0.0)
                gx_m = min(d_plus_x, 0.0)
                gy_p = max(d_minus_y, 0.0)
                gy_m = min(d_plus_y, 0.0)
                gz_p = max(d_minus_z, 0.0)
                gz_m = min(d_plus_z, 0.0)

                grad_out[k, j, i] = (
                    gx_p * gx_p + gx_m * gx_m
                    + gy_p * gy_p + gy_m * gy_m
                    + gz_p * gz_p + gz_m * gz_m
                ) ** 0.5


@njit(cache=True, parallel=True)
def reinit_godunov_grad(
    phi: np.ndarray,            # (Nz, Ny, Nx)
    phi_0: np.ndarray,          # (Nz, Ny, Nx) — pre-reinit values for sign
    dx: float, dy: float, dz_arr: np.ndarray,
    grad_out: np.ndarray,       # (Nz, Ny, Nx)
) -> None:
    """Compute |∇φ| with sign-aware Godunov upwind (Sussman 1994 reinit step).

    For sign(φ_0) > 0 (unburned side): use forward-going characteristic
    For sign(φ_0) < 0 (burned side): use backward-going characteristic

    The pattern that makes |∇φ| → 1:
        sign>0: |∇φ|² = max(D⁻, 0)² + min(D⁺, 0)²   for each axis
        sign<0: |∇φ|² = min(D⁻, 0)² + max(D⁺, 0)²   for each axis
    """
    Nz, Ny, Nx = phi.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for k in prange(Nz):
        if k == 0:
            inv_dz_minus = 1.0 / dz_arr[0]
            inv_dz_plus  = 2.0 / (dz_arr[0] + dz_arr[1]) if Nz > 1 else 1.0 / dz_arr[0]
        elif k == Nz - 1:
            inv_dz_plus  = 1.0 / dz_arr[Nz - 1]
            inv_dz_minus = 2.0 / (dz_arr[Nz - 1] + dz_arr[Nz - 2])
        else:
            inv_dz_minus = 2.0 / (dz_arr[k] + dz_arr[k - 1])
            inv_dz_plus  = 2.0 / (dz_arr[k] + dz_arr[k + 1])

        for j in range(Ny):
            for i in range(Nx):
                phi_c = phi[k, j, i]
                s = 1.0 if phi_0[k, j, i] > 0.0 else (-1.0 if phi_0[k, j, i] < 0.0 else 0.0)

                if i > 0:
                    d_minus_x = (phi_c - phi[k, j, i - 1]) * inv_dx
                else:
                    d_minus_x = 0.0
                if i < Nx - 1:
                    d_plus_x = (phi[k, j, i + 1] - phi_c) * inv_dx
                else:
                    d_plus_x = 0.0

                jm = j - 1 if j > 0 else Ny - 1
                jp = j + 1 if j < Ny - 1 else 0
                d_minus_y = (phi_c - phi[k, jm, i]) * inv_dy
                d_plus_y  = (phi[k, jp, i] - phi_c) * inv_dy

                if k > 0:
                    d_minus_z = (phi_c - phi[k - 1, j, i]) * inv_dz_minus
                else:
                    d_minus_z = 0.0
                if k < Nz - 1:
                    d_plus_z = (phi[k + 1, j, i] - phi_c) * inv_dz_plus
                else:
                    d_plus_z = 0.0

                if s > 0.0:
                    gx_p = max(d_minus_x, 0.0); gx_m = min(d_plus_x, 0.0)
                    gy_p = max(d_minus_y, 0.0); gy_m = min(d_plus_y, 0.0)
                    gz_p = max(d_minus_z, 0.0); gz_m = min(d_plus_z, 0.0)
                else:
                    gx_p = min(d_minus_x, 0.0); gx_m = max(d_plus_x, 0.0)
                    gy_p = min(d_minus_y, 0.0); gy_m = max(d_plus_y, 0.0)
                    gz_p = min(d_minus_z, 0.0); gz_m = max(d_plus_z, 0.0)

                grad_out[k, j, i] = (
                    gx_p * gx_p + gx_m * gx_m
                    + gy_p * gy_p + gy_m * gy_m
                    + gz_p * gz_p + gz_m * gz_m
                ) ** 0.5


class LevelSetFront3D:
    """3D level-set front tracker for grass-fire propagation.

    Sign convention:
        φ < 0  burned region (post-flame)
        φ ≈ 0  flame front
        φ > 0  unburned region (ahead of front)

    Updated each outer dt via:
        evolve(dt, v_n_field)   — Godunov upwind advection
        reinitialize()          — restore |∇φ|=1 in narrow band

    Provides masks for downstream coupling:
        burned_mask()           — φ ≤ 0
        flame_body_mask()       — -L_burnout ≤ φ ≤ 0
        ahead_band_mask()       — 0 < φ ≤ DX_VN_BAND
    """

    def __init__(
        self,
        Nz: int, Ny: int, Nx: int,
        dx: float, dy: float, dz_arr: np.ndarray,
        L_burnout: float = L_BURNOUT_M,
    ) -> None:
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.dx = dx
        self.dy = dy
        self.dz_arr = dz_arr.astype(np.float64).copy()
        self.L_burnout = float(L_burnout)
        self.phi = np.full((Nz, Ny, Nx), PHI_INIT_UNBURNED, dtype=np.float64)
        # Workspace for Godunov |∇φ|
        self._grad = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        self._phi_0 = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        # Step counter for reinit scheduling
        self._step = 0

    def initialize_source_patch(
        self,
        i_start: int, i_end: int,
        k_top_bed: int,
        x_mid: np.ndarray,        # (Nx,) cell-centered x
    ) -> None:
        """Set φ to a signed-distance field for an x-strip source patch.

        Source bed cells (i ∈ [i_start, i_end), k ≤ k_top_bed) are pre-burned:
        φ = -L_burnout/2 inside, signed distance to the right edge outside.

        For grass-fire: source is a vertical sheet at x = x_mid[i_end] (front
        of source).  φ(x) = x - x_front  for cells outside the source.
        """
        x_front = x_mid[i_end - 1] + 0.5 * self.dx   # right edge of source
        # Inside source patch: φ = -L_burnout/2 (well into burned region)
        # Outside source patch: φ = x - x_front (signed distance to source front)
        for k in range(self.Nz):
            for j in range(self.Ny):
                for i in range(self.Nx):
                    # Source patch cells (in bed)
                    if i_start <= i < i_end and k <= k_top_bed:
                        self.phi[k, j, i] = -self.L_burnout / 2.0
                    else:
                        # Signed distance to the source front
                        d = x_mid[i] - x_front
                        self.phi[k, j, i] = d if d > 0.0 else max(d, -self.L_burnout / 2.0)

    def evolve(self, dt: float, v_n_field: np.ndarray) -> None:
        """Advance φ by one step: ∂φ/∂t + v_n |∇φ| = 0 (v_n ≥ 0)."""
        godunov_grad_norm(self.phi, self.dx, self.dy, self.dz_arr, self._grad)
        # phi_new = phi - dt * v_n * |grad_phi|
        self.phi -= dt * v_n_field * self._grad
        self._step += 1

    def reinitialize(self) -> None:
        """Restore |∇φ| = 1 in narrow band via Sussman 1994 iteration.

        Iterates ∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0 for REINIT_SUBSTEPS substeps
        with τ-step = DT_REINIT_FRAC × min(dx, dy, dz_min).
        """
        np.copyto(self._phi_0, self.phi)
        dz_min = float(self.dz_arr.min())
        dtau = DT_REINIT_FRAC * min(self.dx, self.dy, dz_min)
        for _ in range(REINIT_SUBSTEPS):
            reinit_godunov_grad(self.phi, self._phi_0, self.dx, self.dy,
                                self.dz_arr, self._grad)
            # sign of phi_0 (smooth via tanh of phi_0 / dx for stability)
            sign_phi0 = np.where(
                self._phi_0 > 0.0, 1.0, np.where(self._phi_0 < 0.0, -1.0, 0.0)
            )
            self.phi -= dtau * sign_phi0 * (self._grad - 1.0)

    def maybe_reinitialize(self) -> None:
        """Reinitialize only every LEVELSET_REINIT_INTERVAL outer steps."""
        if self._step % LEVELSET_REINIT_INTERVAL == 0 and self._step > 0:
            self.reinitialize()

    def enforce_z_uniformity(self) -> None:
        """Project phi onto its z-mean to restore the structural invariant
        that phi tracks a 2D bed-front (lifted to 3D as redundant storage).

        Phase 14y diagnostic showed that Sussman reinit + Godunov advection
        progressively introduce z-variation in phi (1.5 m drift over 17 s
        in the U=2 large-domain case at t=17.12s).  Since phi conceptually
        represents the bed-pyrolysis-front position — a strictly 2D
        quantity — any z-variation is numerical drift, not physics.  This
        method enforces the invariant.

        Called after evolve() + maybe_reinitialize() in the time loop.
        """
        phi_2d = self.phi.mean(axis=0)
        self.phi[:] = phi_2d[None, :, :]

    # ── Region masks for CFD coupling ─────────────────────────────────────────
    def burned_mask(self) -> np.ndarray:
        """Cells in burned region: φ ≤ 0."""
        return self.phi <= 0.0

    def flame_body_mask(self) -> np.ndarray:
        """Cells in flame body (active burning zone): -L_burnout ≤ φ ≤ 0."""
        return (self.phi >= -self.L_burnout) & (self.phi <= 0.0)

    def ahead_band_mask(self, band_m: float = DX_VN_BAND_M) -> np.ndarray:
        """Cells in preheating band ahead of front: 0 < φ ≤ band_m."""
        return (self.phi > 0.0) & (self.phi <= band_m)

    def front_x(self, k: int = 0, j: int = 0) -> float:
        """Approximate front x-coordinate at (k, j) by linear interpolation
        between sign-changing neighbors.  Returns +∞ if no front in this row.
        """
        for i in range(self.Nx - 1):
            phi0 = self.phi[k, j, i]
            phi1 = self.phi[k, j, i + 1]
            if phi0 < 0.0 < phi1 or phi1 < 0.0 < phi0:
                # Linear interpolation of zero-crossing
                frac = -phi0 / (phi1 - phi0)
                return (i + frac + 0.5) * self.dx
        return float('inf')


def compute_v_n(
    q_into_unburned_band: np.ndarray,      # (Ny, Nx) [W/m²]
    rho_b: float,
    cp_s: float,
    h_bed: float,
    T_ign: float,
    T_amb: float,
) -> np.ndarray:
    """Compute front-normal velocity v_n from heat flux integral.

    v_n [m/s] = q_in / E_ign_per_area

    where E_ign = ρ_b · cp_s · h_bed · (T_ign − T_amb) is the energy/area
    needed to bring an ahead-of-front bed cell from T_amb to T_ign.

    This is mesh-convergent: q_in is integrated over a constant-physical-
    size band (DX_VN_BAND), and E_ign is well-defined regardless of dx.

    LEGACY 2D path — kept for backward compatibility with cases that
    don't use the 3D forcing.  Production callers should prefer
    :func:`compute_v_n_3d` (per-cell q_in + per-cell moisture-aware
    E_ign per Drysdale §3.5 + Mell 2007 §3.4).
    """
    E_ign_per_area = rho_b * cp_s * h_bed * (T_ign - T_amb)   # [J/m²]
    if E_ign_per_area <= 0.0:
        return np.zeros_like(q_into_unburned_band)
    v_n = q_into_unburned_band / E_ign_per_area
    np.maximum(v_n, 0.0, out=v_n)   # front advances forward only
    return v_n


# Water latent heat of vaporization at boiling point (NIST).
L_VAP_WATER = 2.26e6   # [J/kg]


def compute_q_in_at_front_3d(
    q_frankman: np.ndarray,     # (Nz, Ny, Nx) [W/m²] per cell
    q_dom_fwd: np.ndarray,      # (Nz, Ny, Nx) [W/m²] per cell
    ahead_band_mask: np.ndarray,
    q_burst_conv_2d: np.ndarray | None = None,   # (Ny, Nx) optional
) -> np.ndarray:
    """Per-cell forward heat flux entering the ahead-of-front band [W/m²].

    Unlike :func:`compute_q_in_at_front` (which column-sums Frankman and
    takes the top-of-band DOM), this routine keeps the per-cell z-resolved
    structure so the downstream :func:`compute_v_n_3d` can produce a
    z-varying propagation speed.  The cell-z that gets the most forward
    radiation (top-of-bed) naturally produces the largest v_n; bottom-of-
    bed cells lag in ignition.  This makes the level-set genuinely 3D
    (Phase 17b — replaces the 2D-lifted forcing that previously masked
    z-variation via ``enforce_z_uniformity``).

    Phase 17b convention: q_burst_conv is 2D (top-of-bed flame-finger
    contact heating from Finney 2015); it's added to the top-of-band
    cell per column only.
    """
    out = np.where(ahead_band_mask, q_frankman + q_dom_fwd, 0.0)
    if q_burst_conv_2d is not None:
        # Add q_burst to the top-of-band cell only.
        Nz, Ny, Nx = q_frankman.shape
        any_in_band = ahead_band_mask.any(axis=0)
        k_top = Nz - 1 - np.argmax(ahead_band_mask[::-1, :, :], axis=0)
        for j in range(Ny):
            for i in range(Nx):
                if any_in_band[j, i]:
                    out[k_top[j, i], j, i] += q_burst_conv_2d[j, i]
    return out


def compute_v_n_3d(
    q_in_3d: np.ndarray,                   # (Nz, Ny, Nx) [W/m²]
    rho_b: float,
    cp_s: float,
    h_bed: float,
    T_ign: float,
    T_amb: float,
    M_local: np.ndarray | None = None,     # (Nz, Ny, Nx) [-] m_water/m_solid
    L_vap: float = L_VAP_WATER,
    f_dry_to_ignite: float = 1.0,
) -> np.ndarray:
    """Per-cell front-normal velocity v_n [m/s] including moisture.

    For each cell:
        v_n = q_in_per_cell / E_ign_per_area_per_cell

    where E_ign now includes the latent heat of evaporating moisture
    (Drysdale 2011 §3.5; Mell 2007 WFDS §3.4; Linn 2002 FIRETEC):

        E_ign = ρ_b·cp_s·h_bed·(T_ign − T_amb)         [sensible heat]
              + ρ_b·M_local·h_bed·L_vap·f_dry_to_ignite [latent heat]

    The latent term dominates at field-density grass for M ≥ 0.1
    (at M=0.30, latent ~5× sensible).  ``f_dry_to_ignite`` (default 1.0)
    is the fraction of cell water that must evaporate before ignition;
    a value of 1.0 is the conservative full-evaporation assumption.

    Returns (Nz, Ny, Nx).  Cells with M_local=0 reduce exactly to the
    legacy dry formula (regression-preserving).
    """
    E_sens = rho_b * cp_s * h_bed * (T_ign - T_amb)       # [J/m²]
    if M_local is None:
        E_ign = np.full_like(q_in_3d, E_sens)
    else:
        E_lat = rho_b * h_bed * L_vap * f_dry_to_ignite * M_local
        E_ign = E_sens + E_lat
    # Guard against divide-by-zero (E_sens > 0 always for non-trivial inputs)
    E_ign = np.maximum(E_ign, 1.0)
    v_n = q_in_3d / E_ign
    np.maximum(v_n, 0.0, out=v_n)
    return v_n


# ── Stage 2: CFD coupling helpers ─────────────────────────────────────────────

# Calibration-class constants (Pyne 1993 sets order-of-magnitude; exact values
# locked once + applied to all cases per Rule #5; declared as cal targets per
# Rule #2 BEFORE running validation).
Q_BOOTSTRAP_W_M3 = 500_000.0   # Pyne 1993 §11.3 drip-torch operational scale
T_BOOTSTRAP_S    = 2.0         # bridge T_amb → chem-bootstrap given Q_bootstrap


@njit(cache=True, parallel=True)
def step_frankman_flame_tip(
    T_g: np.ndarray,             # (Nz, Ny, Nx) [K]
    T_s: np.ndarray,             # (Nz, Ny, Nx) [K]
    flame_body_mask: np.ndarray, # (Nz, Ny, Nx) bool
    ahead_band_mask: np.ndarray, # (Nz, Ny, Nx) bool
    alpha_s: np.ndarray,         # (Nz, Ny, Nx)
    sigma_sav: float,
    dz_arr: np.ndarray,          # (Nz,)
    h_flame: float,
    q_frankman_out: np.ndarray,  # (Nz, Ny, Nx) [W/m²] — overwritten
) -> None:
    """Frankman 2013 flame-tip convective heat transfer.

    For each y-strip, find the max T_g across flame_body cells.  Apply
    Frankman h_flame convection to ahead-band bed cells in that y-strip:

        q_frankman = h_flame · (T_g_flame_y − T_s) · σ · α_s · dz_cell

    The flame body and ahead band are at DIFFERENT x-positions (flame at
    x < x_front, band at x > x_front), so we search per-y rather than
    per-column.  This represents the lateral / forward heat transfer from
    the established flame to ahead-of-front bed.

    Output is per horizontal footprint [W/m²], to be added directly to
    coupling kernel's q_in_solid alongside q_rad_volumetric.

    Reference: Frankman, D. et al. (2013) IJWF 22:157 — measured h ~100-500
    W/m²/K at flame impingement on wildland fuels; we use the lit mid-range
    h_flame = 200.
    """
    Nz, Ny, Nx = T_g.shape
    # Initialize output to zero
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                q_frankman_out[k, j, i] = 0.0

    # Per-y-strip: find max T_g across all flame-body cells at that j
    for j in prange(Ny):
        T_flame_strip = 0.0
        has_flame = False
        for i in range(Nx):
            for k in range(Nz):
                if flame_body_mask[k, j, i] and T_g[k, j, i] > T_flame_strip:
                    T_flame_strip = T_g[k, j, i]
                    has_flame = True

        if not has_flame:
            continue

        # Apply Frankman flux to ALL ahead-band bed cells at this j
        for i in range(Nx):
            for k in range(Nz):
                if not ahead_band_mask[k, j, i]:
                    continue
                a_s = alpha_s[k, j, i]
                if a_s <= 0.0:
                    continue
                Ts = T_s[k, j, i]
                if T_flame_strip <= Ts:
                    continue   # no heat transfer if flame not hotter
                # Volumetric flux × dz_k → per horizontal footprint [W/m²]
                a_v = sigma_sav * a_s
                q_per_vol = h_flame * a_v * (T_flame_strip - Ts)   # [W/m³]
                q_frankman_out[k, j, i] = q_per_vol * dz_arr[k]    # [W/m²]


def compute_q_dom_fwd_at_band(
    rad_solver,                 # DOMRadiationSolver instance
    ahead_band_mask: np.ndarray,
    q_dom_fwd_out: np.ndarray,  # (Nz, Ny, Nx) [W/m²] — overwritten
) -> None:
    """Extract DOM forward-pointing radiative flux density into ahead-of-front
    cells, expressed as a surface flux [W/m²] per cell.

    Σ_n w_n · |ξ_n| · I_n[k, j, i] over ordinates with ξ_n > 0 (forward in +x).
    For a DOM intensity field I_n [W/m²/sr] and weights w_n summing to 4π, the
    sum Σ w_n |ξ_n| I_n is the +x-direction net hemispheric flux at the cell
    [W/m²].

    Phase 15M (2026-06-08) bugfix: removed a spurious ``× dz_arr`` multiply
    that previously converted the per-cell flux [W/m²] into a per-cell line
    integrand [W/m].  Downstream :func:`compute_q_in_at_front` then column-
    summed the W/m quantity but labelled it W/m², and :func:`compute_v_n`
    treated that as a surface flux — giving v_n ~10× smaller than the true
    surface flux would imply.  With finite ignition (Phase 15L v2) this
    manifested as front-stalling at ROS_post_pulse ≈ 0.5 m/min on mickey L0.

    For grass-fire propagation in +x, "forward" radiation reaches ahead-of-
    front cells via ordinates pointing in +x.  Cold ambient gas is mostly
    transparent (κ_gas ≈ 0 at T<600K via Phase 14w-I continuous ramp), so
    forward intensity tracks across the flame-to-bed-top distance.
    """
    q_dom_fwd_out.fill(0.0)
    for n in range(rad_solver.M):
        xi_n = float(rad_solver.Omega[n, 0])
        if xi_n <= 0.0:
            continue   # only forward-pointing ordinates
        w_n = rad_solver.weights[n]
        I_n = rad_solver.I_set[n]
        # Vectorized accumulate: q_fwd += w · |ξ| · I_n  [W/m²]  in ahead band
        contribution = w_n * abs(xi_n) * I_n
        q_dom_fwd_out += np.where(ahead_band_mask, contribution, 0.0)
    # Phase 15M: do NOT multiply by dz here — Σ w·|ξ|·I is already W/m².


def compute_q_in_at_front(
    q_frankman: np.ndarray,     # (Nz, Ny, Nx) [W/m²] per cell — see below
    q_dom_fwd: np.ndarray,      # (Nz, Ny, Nx) [W/m²] per cell — see below
    ahead_band_mask: np.ndarray,
    dx: float, dy: float, dz_arr: np.ndarray,
    band_m: float = DX_VN_BAND_M,
    q_burst_conv_2d: np.ndarray | None = None,   # (Ny, Nx) [W/m²] Phase 15N optional
) -> np.ndarray:
    """Surface-flux equivalent of forward heat transport into ahead-band [W/m²].

    Two physically distinct inputs are combined into a single bed-surface
    power-per-horizontal-area for v_n:

    * ``q_frankman`` is set by :func:`compute_q_frankman_at_band` as
      ``q_per_vol [W/m³] × dz_k [m]`` — the per-cell volumetric Frankman
      forward-convective heating absorbed by a porous cell, expressed as a
      per-horizontal-footprint power [W/m²].  Column-summing across bed cells
      gives the total Frankman power absorbed per horizontal area [W/m²].

    * ``q_dom_fwd`` is set by :func:`compute_q_dom_fwd_at_band` (post-Phase-15M)
      as ``Σ_n w_n |ξ_n| I_n`` — the per-cell DOM net forward radiative flux
      density [W/m²].  We take the value at the top-of-band cell as the
      surface flux incident on the bed top.  (A more accurate bed-absorption
      treatment would multiply by ``(1 − exp(−κ_solid h_bed))`` ≈ 0.77 for
      Cheney grass, but the surface-flux approximation is exact in the
      thin-bed limit and within 25% over the canopy-scale range.)

    Returns (Ny, Nx) [W/m²] suitable as input to :func:`compute_v_n`.

    Phase 15M (2026-06-08) bugfix: pre-fix, this routine column-summed
    ``q_frankman + q_dom_fwd``, but ``q_dom_fwd`` had been spuriously
    multiplied by ``dz_arr`` in :func:`compute_q_dom_fwd_at_band` (units
    W/m, not W/m²).  The sum was dimensionally inconsistent; the resulting
    "W/m²" label hid a ~10× under-read.  With finite ignition this masked
    a complete failure to sustain propagation.
    """
    Nz, Ny, Nx = q_frankman.shape

    # ── Frankman: column-sum of per-cell absorbed power-per-horizontal-area ──
    q_frankman_masked = np.where(ahead_band_mask, q_frankman, 0.0)
    q_frankman_col = q_frankman_masked.sum(axis=0)   # (Ny, Nx) [W/m²]

    # ── DOM forward: top-of-band cell incoming surface flux ───────────────
    # For each (j,i), find the topmost k where ahead_band_mask is True;
    # take q_dom_fwd at that cell as the surface flux entering the bed top.
    # Phase 15F mask restricts to k < n_z_bed, so "top of band" is the top
    # bed cell beneath the band.
    any_in_band = ahead_band_mask.any(axis=0)
    # Argmax of mask flipped vertically gives offset from the top
    k_top_from_bottom = Nz - 1 - np.argmax(ahead_band_mask[::-1, :, :], axis=0)
    q_dom_top = np.take_along_axis(
        q_dom_fwd, k_top_from_bottom[np.newaxis, :, :], axis=0,
    )[0]
    q_dom_top = np.where(any_in_band, q_dom_top, 0.0)   # (Ny, Nx) [W/m²]

    result = q_frankman_col + q_dom_top   # (Ny, Nx) [W/m²]
    # Phase 15N — optional Finney burst-convective preheat contribution
    if q_burst_conv_2d is not None:
        result = result + q_burst_conv_2d
    return result


@njit(cache=True, parallel=True)
def apply_bootstrap_heat(
    Q_comb: np.ndarray,           # (Nz, Ny, Nx) [W/m³] mutated
    flame_body_mask: np.ndarray,  # (Nz, Ny, Nx) bool
    cell_age: np.ndarray,         # (Nz, Ny, Nx) [s]
    Q_bootstrap: float,
    t_bootstrap: float,
) -> None:
    """Add Q_bootstrap to Q_comb for cells in flame_body whose cell_age <
    t_bootstrap.

    Lit anchor: Pyne 1993 §11.3 drip-torch operational fuel-flow rate ×
    HoC ÷ line volume gives MW/m³ scale.  500 kW/m³ is the conservative
    end of that range; calibration-class per Rule #2.

    Used to bridge the chemistry-bootstrap gap: when level-set advances
    into a new cell (cell_age=0), Q_bootstrap drives T_g into the
    chemistry-active range over t_bootstrap seconds, after which
    resolved EBU + chemistry self-sustain (B2 architecture).
    """
    Nz, Ny, Nx = Q_comb.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if flame_body_mask[k, j, i] and cell_age[k, j, i] < t_bootstrap:
                    Q_comb[k, j, i] += Q_bootstrap


# ── Phase 14y-flame: state-derived flame level-set ────────────────────────────
# A SECOND level-set, separate from phi (the bed-pyrolysis front).  Tracks
# the active gas-phase flame body as a 3D signed-distance field, computed
# each outer step from gas state — no separate evolution equation needed
# since the gas state itself carries advection + chemistry dynamics.
#
# Architectural intent:
#   phi (bed)    — kinematic level-set, advances by v_n; tracks bed
#                   pyrolysis front; consumed by bed-side masks (heat
#                   flux integral into unburned bed, ROS, ahead_bed_band).
#   phi_flame    — state-derived signed distance to active-flame region
#                   in the gas; consumed by flame-side masks (Frankman
#                   SOURCE, bootstrap heat sink).
#
# The two level-sets are independent: bed can pyrolyze without flame
# (volatiles blow away unburned), flame can exist over already-pyrolyzed
# bed (hot products + advected fuel).
#
# Active-flame criterion (lit-grounded, conservative):
#   ω > OMEGA_MIN_FLAME  OR  (T_g > T_PLUME_MIN AND Y_F > Y_F_MIN_PLUME)
#   = "reaction zone" OR "fuel-bearing hot plume tail"
#
# Lit refs:
# - Magnussen 1989 IEA-TLM (EDC fine-structure: ω is direct combustion
#   indicator)
# - Drysdale 2011 Fire Dynamics §3.4 (plume-tail T ~ 800-1200 K above
#   reaction zone is still part of the radiant flame body)

OMEGA_MIN_FLAME   = 1.0e-3   # [kg/m³/s] reaction-zone threshold
T_PLUME_MIN       = 1000.0   # [K] plume tail T threshold
Y_F_MIN_PLUME     = 1.0e-3   # [-] fuel-bearing plume threshold


def compute_phi_flame_from_state(
    omega: np.ndarray,        # (Nz, Ny, Nx) [kg/m³/s] reaction rate
    T_g: np.ndarray,          # (Nz, Ny, Nx) [K]
    Y_fuel: np.ndarray,       # (Nz, Ny, Nx) [-]
    dx: float, dy: float, dz_arr: np.ndarray,
) -> np.ndarray:
    """Compute signed-distance to active-flame region from gas state.

    phi_flame[k, j, i] < 0 inside flame, > 0 outside; magnitude in metres.

    Active-flame criterion:
        active = (ω > OMEGA_MIN_FLAME) OR
                 (T_g > T_PLUME_MIN AND Y_F > Y_F_MIN_PLUME)

    Distance is anisotropic Euclidean using per-axis grid spacing.
    Implementation uses scipy.ndimage.distance_transform_edt with
    sampling=(dz_typical, dy, dx).

    Returns a (Nz, Ny, Nx) float array; |phi_flame| ≈ metres to nearest
    active-flame cell boundary.
    """
    from scipy import ndimage as _ndi

    active = (omega > OMEGA_MIN_FLAME) | (
        (T_g > T_PLUME_MIN) & (Y_fuel > Y_F_MIN_PLUME)
    )

    # Use a representative dz (mean of bed cells) for anisotropic sampling.
    # Strict anisotropy would feed the per-cell dz_arr into the EDT, but
    # scipy.ndimage takes a uniform sampling per axis.  For our grids,
    # cell-to-cell dz variation is modest (BL refinement + buffer expansion
    # both kept off in current Phase 14x setups: dz ≈ const).
    dz_eff = float(dz_arr.mean())
    sampling = (dz_eff, dy, dx)

    if active.any() and (~active).any():
        # signed distance: + outside, − inside
        dist_outside = _ndi.distance_transform_edt(~active, sampling=sampling)
        dist_inside  = _ndi.distance_transform_edt(active,  sampling=sampling)
        phi_flame = np.where(active, -dist_inside, dist_outside)
    elif active.all():
        phi_flame = np.full(active.shape, -1.0e6, dtype=np.float64)
    else:
        phi_flame = np.full(active.shape, +1.0e6, dtype=np.float64)
    return phi_flame.astype(np.float64)


def flame_body_mask_from_phi_flame(
    phi_flame: np.ndarray,
    band_m: float = 0.0,
) -> np.ndarray:
    """Boolean mask of cells inside (or within band_m of) the active flame.

    band_m = 0 → strict inside (phi_flame ≤ 0).
    band_m > 0 → also includes cells within band_m of the flame surface
                  on the OUTSIDE — useful for "near-flame" heating zones.
    """
    return phi_flame <= band_m


def update_cell_age(
    cell_age: np.ndarray,         # (Nz, Ny, Nx) [s] mutated
    flame_body_mask: np.ndarray,  # (Nz, Ny, Nx) bool
    dt: float,
) -> None:
    """Update per-cell time-since-ignition for bootstrap window tracking.

    Newly ignited cells (in flame_body for the first time) get cell_age = 0.
    Continuing flame_body cells age by dt.  Cells outside flame_body
    reset to inf (ready to "ignite" if level-set crosses them again
    in some future state).
    """
    # Newly ignited: in flame body now, age was inf
    newly_ignited = flame_body_mask & np.isinf(cell_age)
    cell_age[newly_ignited] = 0.0
    # Continuing flame body: increment age
    continuing = flame_body_mask & ~newly_ignited
    cell_age[continuing] += dt
    # Outside flame body: reset to inf
    cell_age[~flame_body_mask] = np.inf
