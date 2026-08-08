"""Discrete Ordinates Method (DOM) radiation solver — Phase 14m.

Replaces the isotropic P1 (Eddington) approximation with directional
transport: each ordinate (Ω̂_n, w_n) carries its own intensity I_n
along its direction.  Captures the angular distribution that P1
averages out — critical in opaque dense media (κ·dx ≳ 1) where P1
underpredicts direct downward flux from above-bed flame plume onto
next-bed strips.

The Cut-bed propagation diagnostic in Phase 14h (z-rake at
plots/phase14h/cut4_u4_zrake.png) showed gas in buffer above the
front reaching 750K while bed cells below stayed at 303K — P1's
isotropic flux distributed the buffer-cell emission equally in all
4π directions, so only ~5% reached the bed below within the
optical depth.  S_N quadrature samples that downward direction
specifically.

Governing equation (no scattering — appropriate for fire problems):

    Ω̂·∇I(r, Ω̂) + κ I(r, Ω̂) = κ B(T)               [W/m³/sr]

Cell-centered finite-volume step differencing (Modest 2003 §16.5):

    I_n[c] = (κ B V + Σ_face_in |Ω̂_n·n̂| A_face I_n[face_in])
             / (κ V + Σ_face |Ω̂_n·n̂| A_face)

Per-cell volumetric absorption:

    ∇·q_rad = κ (4π B - G),    G = Σ_n w_n I_n

Output convention matches P1RadiationSolver: cell-centered q_rad
in [W/m²] (per horizontal footprint) = (∇·q_rad) · dz_k.

Boundary conditions (matching the Phase 14m design choices):
- z = Lz (top):    Marshak (ε_w=1 wall at T_amb): I_in_n = σT_amb⁴/π
- z = 0  (ground): diffuse-gray wall, ε_w_ground = 0.85 (Hahn 1981
                   typical dry soil; FIRESTAR validation uses 0.8-0.9):
                   I_out_n = ε_w · σT_w⁴/π + (1-ε_w)/π · q_in_diffuse
- x = 0, x = Lx:   Dirichlet T = T_amb (open boundary, ε_w=1)
- y = 0, y = Ly:   periodic (infinite-fire-line interpretation)

Quadrature: level-symmetric S_N (Lathrop & Carlson 1968 LA-3186):
- S4: 24 ordinates (3 per octant × 8) — default
- S6: 48 ordinates (6 per octant)
- S8: 80 ordinates (10 per octant)

Source iteration with under-relaxation ω_relax = 0.7 for the
non-black ground BC; convergence ‖ΔG‖∞/‖G‖∞ < 1e-3, max 30 iters.
For periodic-y BC each ordinate also iterates internally to
self-consistency.

References:
- Lathrop, K.D. & Carlson, B.G. (1968) Numerical Solution of the
  Boltzmann Transport Equation, LASL LA-3186 — original S_N quadrature
- Modest, M.F. (2003) Radiative Heat Transfer, 2nd ed., §16 — DOM
- Fiveland, W.A. (1988) JTHT 2:309 — DOM in absorbing-emitting media
- Tien, C.L. (1968) Adv. Heat Transfer 5:253 — soot κ correlations
- FIRESTAR (Morvan 2009) IJWF 18:679 — DOM validation in fire CFD
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


SIGMA_SB = 5.67e-8       # [W/m²/K⁴]

# Soot extinction constants (Tien 1968; Hubbard-Tien 1978).
# Phase 14w-I: replaced binary on/off (ω_comb > 1e-3 ? 0.5 : 0) with a
# continuous OR ramp activated by EITHER active combustion OR hot post-
# combustion gas.  The binary form made the entire above-bed buoyant
# plume invisible to DOM the moment a cell's ω dropped below threshold
# — even at Tg=1500K.  Lit: hot gas emits Planck radiation regardless
# of whether reaction is currently active; soot + CO2 + H2O all
# contribute κ ~ 0.05–0.5 1/m at flame T (Tien 1968).
KAPPA_SOOT_HOT = 0.5
OMEGA_COMB_THRESH = 1.0e-3   # ω scale for full activation by combustion
T_GAS_RAD_REF = 600.0        # K — onset of significant gas IR emission
T_GAS_RAD_DT  = 400.0        # K — saturate at T_ref + dT = 1000K (Tien 1968)

# Ground emissivity.
# Hahn 1981 J. Atmos. Sci. 38:1601 reports dry-soil IR ε ∈ [0.80, 0.95];
# FIRESTAR uses 0.85; Pimont & Linn 2009 use 0.9 for canopy floor.
# Phase 14v-bc-soil: with T_soil(t) now evolved by soil_3d.py (1D vertical
# conduction sub-model), ε_w<1 is physically consistent: soil heats up,
# reflects (1-ε_w) fraction of incident flux back into the bed, AND
# re-emits ε_w σ T_soil⁴ via the wall BC.
EPS_W_GROUND = 0.85


# ── S_N quadrature ────────────────────────────────────────────────────────────
# Level-symmetric S4 (Lathrop & Carlson 1968, Table III): 3 ordinates per octant
# Direction cosines (one octant, +x +y +z):
#   permutations of (μ_1, μ_1, μ_2) where:
#   μ_1 = 0.295876, μ_2 = 0.908248 (level-symmetric: 2μ_1² + μ_2² = 1)
# Weights: all equal, w_n = 4π/24 = π/6
def _generate_sn_ordinates(N: int):
    """Return (Ω, w) where Ω is (M, 3) direction-cosine array and w is (M,) weight.

    Total directions M = N(N+2)/8 × 8 = N(N+2)  [for level-symmetric S_N].
    """
    if N == 4:
        # Lathrop-Carlson S4 (Table III): 3 ordinates per octant.
        # Level-symmetric constraint: 2μ_1² + μ_2² = 1
        #   → 2(0.295876)² + (0.908248)² = 0.087543 + 0.824906 = 1.000 ✓
        mu1 = 0.295876
        mu2 = 0.908248
        per_octant = [
            (mu1, mu1, mu2),
            (mu1, mu2, mu1),
            (mu2, mu1, mu1),
        ]
    else:
        # S6 / S8 use level-quantized direction sets with non-trivial
        # ordinate selection rules (each ordinate (μ_i, μ_j, μ_k) must
        # satisfy i + j + k = N/2 + 1 and μ_i² + μ_j² + μ_k² = 1; Lathrop
        # & Carlson 1968 Tables V, VII).  Implementing these correctly
        # requires the level-index-triplet enumeration.  S4 (24 ordinates)
        # is sufficient to capture the angular asymmetry that motivated
        # DOM here; finer quadratures can be added later if S4 is
        # insufficient for cases where direct flux geometry matters more.
        raise NotImplementedError(
            f"S{N} not implemented yet; only S4 supported in Phase 14m. "
            f"S6/S8 require level-index-triplet enumeration "
            f"(Lathrop & Carlson 1968 Tables V, VII)."
        )

    # Reflect into all 8 octants
    octants = [
        ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), (-1,  1,  1),
        ( 1, -1, -1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
    ]
    Ω = []
    for sx, sy, sz in octants:
        for o in per_octant:
            Ω.append((sx * o[0], sy * o[1], sz * o[2]))
    Ω = np.array(Ω, dtype=np.float64)
    M = Ω.shape[0]
    # All weights equal in level-symmetric quadrature
    w = np.full(M, 4.0 * math.pi / M, dtype=np.float64)
    return Ω, w


# ── Sweep kernel ──────────────────────────────────────────────────────────────
@njit(cache=True, parallel=False)   # sequential in cell-sweep ordering by design
def _sweep_one_ordinate(
    I_n: np.ndarray,           # (Nz, Ny, Nx)  intensity for this ordinate
    kappa: np.ndarray,         # (Nz, Ny, Nx)  total absorption coefficient
    B: np.ndarray,             # (Nz, Ny, Nx)  blackbody intensity = σT⁴/π
    xi: float, eta: float, mu: float,    # direction cosines (x, y, z)
    dx: float, dy: float, dz_arr: np.ndarray,
    I_left: float, I_right: float,        # x-boundary inflow values
    I_back: float, I_front: float,        # y-boundary inflow values (used only if non-periodic)
    I_top: float, I_ground_in: np.ndarray,  # z BC: top inflow + ground (Ny,Nx) for non-uniform
    y_periodic: bool,
):
    """Compute I_n for all cells via a single sweep in upwind order.

    Step differencing:
       I_n[c] = (κ B + |ξ|/dx I_x_in + |η|/dy I_y_in + |μ|/dz_k I_z_in) /
                (κ + |ξ|/dx + |η|/dy + |μ|/dz_k)

    For periodic-y, this is one pass; full periodicity requires
    iterating the sweep at the wrapper level.
    """
    Nz, Ny, Nx = I_n.shape
    aix = abs(xi)
    aiy = abs(eta)
    aiz = abs(mu)
    inv_dx = aix / dx
    inv_dy = aiy / dy

    # Determine sweep order
    i_start, i_end, i_step = (0, Nx, 1) if xi >= 0 else (Nx - 1, -1, -1)
    j_start, j_end, j_step = (0, Ny, 1) if eta >= 0 else (Ny - 1, -1, -1)
    k_start, k_end, k_step = (0, Nz, 1) if mu  >= 0 else (Nz - 1, -1, -1)

    for k in range(k_start, k_end, k_step):
        inv_dz_k = aiz / dz_arr[k]
        for j in range(j_start, j_end, j_step):
            for i in range(i_start, i_end, i_step):
                # x upwind neighbor
                if xi >= 0:
                    I_x = I_n[k, j, i - 1] if i > 0 else I_left
                else:
                    I_x = I_n[k, j, i + 1] if i < Nx - 1 else I_right
                # y upwind neighbor (with periodic option)
                if eta >= 0:
                    if j > 0:
                        I_y = I_n[k, j - 1, i]
                    elif y_periodic:
                        I_y = I_n[k, Ny - 1, i]
                    else:
                        I_y = I_back
                else:
                    if j < Ny - 1:
                        I_y = I_n[k, j + 1, i]
                    elif y_periodic:
                        I_y = I_n[k, 0, i]
                    else:
                        I_y = I_front
                # z upwind neighbor
                if mu >= 0:
                    I_z = I_n[k - 1, j, i] if k > 0 else I_ground_in[j, i]
                else:
                    I_z = I_n[k + 1, j, i] if k < Nz - 1 else I_top

                num = kappa[k, j, i] * B[k, j, i] + inv_dx * I_x + inv_dy * I_y + inv_dz_k * I_z
                den = kappa[k, j, i] + inv_dx + inv_dy + inv_dz_k
                I_n[k, j, i] = num / den


@njit(cache=True, parallel=True)
def _accumulate_G(I_set: np.ndarray, w_set: np.ndarray, G_out: np.ndarray):
    """Compute G = Σ_n w_n I_n at every cell."""
    M, Nz, Ny, Nx = I_set.shape
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                s = 0.0
                for n in range(M):
                    s += w_set[n] * I_set[n, k, j, i]
                G_out[k, j, i] = s


class DOMRadiationSolver:
    """DOM (Discrete Ordinates Method) radiation solver — Phase 14m.

    Same interface as P1RadiationSolver: ``solve`` fills q_rad_solid_out
    and q_rad_gas_out [W/m²] per cell.  Replaces P1 to capture angular
    distribution of flux in dense bed.
    """

    def __init__(
        self,
        Nz: int, Ny: int, Nx: int,
        dy: float, dx: float,
        dz_arr: np.ndarray,
        d_face_above: np.ndarray,
        d_face_below: np.ndarray,
        y_bc: str = "periodic",
        N_quadrature: int = 4,
        eps_w_ground: float = EPS_W_GROUND,
        max_source_iter: int = 30,
        tol_source_iter: float = 1.0e-3,
        omega_relax: float = 0.7,
        kappa_gas_max: float = KAPPA_SOOT_HOT,  # Phase 15K — diagnostic
                                                  # override for the hot-gas
                                                  # soot/molecular extinction
                                                  # ceiling [1/m].  Default
                                                  # 0.5 (Tien 1998 SFPE
                                                  # Handbook §1-4 cellulose
                                                  # smoke).  Mell 2007 WFDS
                                                  # uses 1.0; WSGGM full-
                                                  # spectrum for hot CO2 +
                                                  # H2O + soot gives 0.5–3
                                                  # depending on local T and
                                                  # species.  Diagnostic-only
                                                  # per Rule #2 until lit
                                                  # justification for non-
                                                  # default is added.
    ) -> None:
        self.Nz = Nz; self.Ny = Ny; self.Nx = Nx
        self.dx = dx; self.dy = dy
        self.dz_arr = dz_arr.astype(np.float64).copy()
        # d_face_above/below kept for API symmetry but DOM uses dz_arr directly.
        self.dz = float(dz_arr[0])
        self.y_bc = y_bc
        self.y_periodic = (y_bc == "periodic")
        self.eps_w_ground = float(eps_w_ground)
        self.max_iter = int(max_source_iter)
        self.tol = float(tol_source_iter)
        self.omega = float(omega_relax)
        self.kappa_gas_max = float(kappa_gas_max)

        # Generate quadrature
        self.Omega, self.weights = _generate_sn_ordinates(N_quadrature)
        self.M = self.Omega.shape[0]   # number of ordinates

        # Pre-allocate intensity workspace
        self.I_set = np.zeros((self.M, Nz, Ny, Nx), dtype=np.float64)
        self.G = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        self.G_prev = np.zeros((Nz, Ny, Nx), dtype=np.float64)

    def solve(
        self,
        T_s: np.ndarray,
        T_g: np.ndarray,
        alpha_s: np.ndarray,
        omega_comb: np.ndarray,
        sigma_sav: float,
        T_amb: float,
        q_rad_solid_out: np.ndarray,
        q_rad_gas_out: np.ndarray,
        T_soil_surface: np.ndarray | None = None,   # (Ny, Nx) [K] Phase 14v-bc-soil
        q_in_soil_out: np.ndarray | None = None,    # (Ny, Nx) [W/m²] DOM → soil flux
        Y_H2O: np.ndarray | None = None,            # (Nz, Ny, Nx) [-] H2O mass fraction
        rho: np.ndarray | None = None,              # (Nz, Ny, Nx) [kg/m³] gas density
        bed_moisture_per_cell: np.ndarray | None = None,  # (Nz, Ny, Nx) [-] m_water/m_solid
                                                           # per bed cell.  When provided,
                                                           # scales kappa_solid by
                                                           # (1 + BETA_KSOLID_WATER * M).
                                                           # Phase 17a (Mell 2007 WFDS,
                                                           # Linn 2002 FIRETEC pattern).
    ) -> None:
        """Solve DOM RTE for current state; fill q_rad_solid_out and q_rad_gas_out [W/m²].

        Phase 14v-bc-soil: when ``T_soil_surface`` is provided, the ground
        wall BC uses T_soil_surface[j,i] for I_w (instead of constant
        T_amb), and ``q_in_soil_out`` is filled with the net downward
        radiation flux at z=0 [W/m²], which the soil 1D conduction model
        uses as its surface BC heat input.
        """
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx

        # ── Per-cell radiative properties ──────────────────────────────────
        kappa_solid = sigma_sav * alpha_s
        # ── Phase 17a (2026-06-20): wet-bed κ_solid scaling ──
        # Wet bed absorbs more radiation per kg of solid because H2O
        # bands (1.4, 1.9, 2.7 µm) supplement cellulose absorption.
        # Mell 2007 WFDS / Linn 2002 FIRETEC both apply a similar
        # multiplier.  Per-cell M_local = m_water/m_solid (from bed
        # aggregation) drives the scaling.  Bracketed:
        #   M=0     → no change
        #   M=0.10  → κ_solid × 1.5
        #   M=0.30  → κ_solid × 2.5
        # Energy absorbed extra is preferentially routed to drying
        # via the existing equilibrium-drying physics on particles.
        BETA_KSOLID_WATER = 5.0   # [-] empirical multiplier; Mell 2007 effective range
        if bed_moisture_per_cell is not None:
            kappa_solid = kappa_solid * (1.0 + BETA_KSOLID_WATER
                                                * bed_moisture_per_cell)
        # Continuous κ_gas ramp (Phase 14w-I).  Activated by EITHER:
        #   - active combustion (ω_comb / ω_thresh, clipped to [0,1])
        #   - hot post-combustion gas (T_g − T_ref) / dT, clipped to [0,1]
        # Either source ramps κ_gas from 0 to KAPPA_SOOT_HOT.  Pre-fix
        # binary cutoff blocked emission from the above-bed buoyant plume
        # the moment its local ω dropped below 1e-3, even at Tg≫1000K.
        omega_factor = np.clip(omega_comb / OMEGA_COMB_THRESH, 0.0, 1.0)
        T_factor     = np.clip((T_g - T_GAS_RAD_REF) / T_GAS_RAD_DT, 0.0, 1.0)
        kappa_gas = self.kappa_gas_max * np.maximum(omega_factor, T_factor)
        # Phase 16 (2026-06-17): water-vapor radiation absorption.
        # H2O has strong absorption bands at 1.4, 1.9, 2.7 µm (Modest 2003
        # §10.7).  WSGG average for H2O at typical flame conditions gives
        # mass-specific extinction A_H2O ≈ 30 m²/kg (Modest Table 10.3,
        # 1000-2000 K window).  Per-cell κ_H2O = A_H2O × ρ × Y_H2O.
        # This is what closes the Cheney moisture-coefficient gap (per
        # Mell 2007 WFDS / Linn 2002 FIRETEC) — H2O ahead of the front
        # absorbs forward radiation that would otherwise preheat the bed.
        if Y_H2O is not None and rho is not None:
            A_H2O_RAD = 30.0   # [m²/kg] mass-specific H2O extinction
            kappa_h2o = A_H2O_RAD * rho * Y_H2O
            kappa_gas = kappa_gas + kappa_h2o
        kappa = kappa_solid + kappa_gas
        # Safety floor for cells with neither solid nor gas absorption.
        # 1e-3 1/m matches P1; far-field cells stay near-transparent.
        KAPPA_FLOOR = 1.0e-3
        kappa_safe = np.maximum(kappa, KAPPA_FLOOR).astype(np.float64)

        # Source: weighted blackbody intensity B = σT⁴/π [W/m²/sr].
        # T_rad blends solid (T_s) and gas (T_g) by κ contribution.
        kappa_safe_for_T = np.where(kappa > 1.0e-9, kappa_safe, 1.0)
        T_rad4 = np.where(
            kappa > 1.0e-9,
            (kappa_solid * T_s ** 4 + kappa_gas * T_g ** 4) / kappa_safe_for_T,
            T_amb ** 4,
        )
        B = SIGMA_SB * T_rad4 / math.pi   # [W/m²/sr]

        # Boundary intensities
        I_amb = SIGMA_SB * T_amb ** 4 / math.pi   # ambient blackbody intensity
        # Top BC (Marshak ε_w=1 at T_amb)
        I_top_inflow = I_amb
        # x=0 and x=Lx: Dirichlet T_amb
        I_x_inflow = I_amb

        # Phase 14v-bc-soil: ground wall T from soil 1D model if provided.
        # I_soil_emit is the *emitted* portion of I_ground_out.  Reflected
        # portion is added inside the source iteration (depends on q_in).
        if T_soil_surface is not None:
            I_soil_emit = (self.eps_w_ground * SIGMA_SB
                           * T_soil_surface ** 4 / math.pi)
        else:
            I_soil_emit = np.full((Ny, Nx), self.eps_w_ground * I_amb,
                                  dtype=np.float64)

        # ── Source iteration ──────────────────────────────────────────────
        # For pure absorbing-emitting media with NO reflective walls, a
        # single sweep per ordinate gives the exact I_n.  We have ε_w<1
        # at the ground, so iterate for the diffuse reflection contribution.
        # For periodic-y, the y-direction wraps within the sweep itself
        # (using the previous-iteration value at j=0/Ny-1 boundaries).

        # Initial ground intensity = emitted contribution (reflection added
        # in iteration once q_in is known).
        I_ground_out = I_soil_emit.copy()

        # Pre-extract ordinate components for the sweep loop
        Omega = self.Omega
        weights = self.weights
        kappa_arr = kappa_safe   # use floored field for sweep
        B_arr = B.astype(np.float64)
        dz_arr = self.dz_arr

        for it in range(self.max_iter):
            # Reset G_prev for convergence check
            np.copyto(self.G_prev, self.G)
            self.G.fill(0.0)

            # Per-ordinate sweep
            for n in range(self.M):
                xi  = float(Omega[n, 0])
                eta = float(Omega[n, 1])
                mu  = float(Omega[n, 2])
                # I_ground_in for this ordinate: equal to outgoing intensity
                # if ordinate points UP (mu > 0).  Otherwise unused.
                I_g_in = I_ground_out if mu > 0 else np.zeros_like(I_ground_out)

                _sweep_one_ordinate(
                    self.I_set[n], kappa_arr, B_arr,
                    xi, eta, mu,
                    self.dx, self.dy, dz_arr,
                    I_x_inflow, I_x_inflow,
                    I_x_inflow, I_x_inflow,   # y boundary (unused if periodic)
                    I_top_inflow,
                    I_g_in,
                    self.y_periodic,
                )

            # Accumulate G
            _accumulate_G(self.I_set, weights, self.G)

            # Compute downward flux at ground (k=0) for new BC.
            # Diffuse-gray wall: I_out = ε_w σ T_w⁴/π + (1-ε_w)/π · ∫_in I cos θ dΩ
            # Phase 14v-bc-soil: emitted = I_soil_emit (uses T_soil_surface);
            # reflected = (1-ε_w)/π · q_in_diffuse.
            q_in_ground = np.zeros((Ny, Nx), dtype=np.float64)
            for n in range(self.M):
                mu_n = float(Omega[n, 2])
                if mu_n < 0:
                    q_in_ground += weights[n] * abs(mu_n) * self.I_set[n, 0, :, :]

            I_ground_new = (I_soil_emit
                            + (1.0 - self.eps_w_ground) / math.pi * q_in_ground)
            # Under-relax
            I_ground_out = (self.omega * I_ground_new
                            + (1.0 - self.omega) * I_ground_out)

            # Convergence check on G
            if it > 0:
                G_max = max(self.G.max(), 1e-12)
                err = np.abs(self.G - self.G_prev).max() / G_max
                if err < self.tol:
                    break

        # ── Per-cell volumetric net absorption ─────────────────────────────
        # ∇·q_rad = κ (4π B - G)   [W/m³]
        # Per horizontal area: × dz_k
        net_volumetric = kappa * (self.G - 4.0 * math.pi * B)
        net_per_horiz = net_volumetric * dz_arr.reshape(-1, 1, 1)

        # Split solid / gas channels by κ ratio
        f_solid = np.where(kappa > 1.0e-9, kappa_solid / kappa_safe, 0.0)
        f_gas   = np.where(kappa > 1.0e-9, kappa_gas   / kappa_safe, 0.0)
        np.copyto(q_rad_solid_out, net_per_horiz * f_solid)
        np.copyto(q_rad_gas_out,   net_per_horiz * f_gas)

        # Phase 14v-bc-soil: ABSORBED incident radiation at z=0 [W/m²].
        # ε_w fraction of downward flux is absorbed by the soil; the soil
        # model handles its own σεT⁴ emission internally via T_soil[0,j,i].
        if q_in_soil_out is not None:
            np.copyto(q_in_soil_out, self.eps_w_ground * q_in_ground)
