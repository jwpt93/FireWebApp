"""Scalar species transport for the lumped-gas formulation.

The lumped model tracks one combustible-volatile mass fraction
``Y_fuel``; the air mass fraction is implicit (``Y_air = 1 − Y_fuel``).

Transport equation (strong conservation form, low-Mach):

    ∂(ρ Y) / ∂t + ∇·(ρ u Y) = ∇·(ρ D ∇Y) + S      [kg/m³/s]

For Boussinesq / slowly-varying ρ this simplifies to

    ∂Y/∂t + (u·∇)Y = D ∇²Y + S/ρ

which is what is implemented here.  Advection: upwind (first-order,
positivity-preserving for non-negative Y).  Diffusion: central second
differences.  Sources/sinks come from external modules
(pyrolysis_3d.S_pyro, combustion_3d.ω_comb).

Diffusivity: laminar mass diffusivity D ≈ 1·10⁻⁴ m²/s for CO/CO₂/H₂O
in air at ~800 K (Drysdale 2011 Table 2.4).  When turbulence is
enabled, D_eff = D + ν_t / Sc_t with Sc_t ≈ 0.7 (Spalding 1971); the
turbulence-aware variant is added in Phase D₂.

Reference: McDermott et al. (2011) NIST FDS Tech. Ref.; Patankar (1980)
finite-volume conventions.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from model_outdoor.physics_3d.muscl_3d import muscl_face_value


# Laminar mass diffusivity at fire temperatures.
D_LAMINAR = 1.0e-4   # [m²/s]


@njit(cache=True, parallel=True)
def step_species_transport(
    Y: np.ndarray,            # (Nz, Ny, Nx) [-] mass fraction of species
    rho: np.ndarray,          # (Nz, Ny, Nx) [kg/m³]
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    S_per_volume: np.ndarray, # (Nz, Ny, Nx) [kg/m³/s] effective net source
    dt: float,
    dx: float, dy: float,
    dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing (Phase 14g)
    d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1 (face k+½)
    d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1 (face k-½)
    D: float,
    Y_inlet: float,            # Phase 14v-bc: x-inlet face value (Way B ghost)
    # Phase 23 Refactor 2C: z-min inlet ghost (cup burner).  Caller
    # ALWAYS passes an array (dummy zeros are fine when unused); the
    # ``z_min_inlet_active`` flag decides whether the array is consulted
    # or the pre-Phase-23 zero-flux wall (ghost = self) is used.  When
    # active, each (j, i) cell reads its own ghost from Y_inlet_zmin[j, i]
    # (fuel-side vs coflow-side compositions at the cup rim).  Default
    # (False + dummy) preserves bit-exact-invariant outdoor behaviour.
    Y_inlet_zmin: np.ndarray = np.zeros((1, 1)),
    z_min_inlet_active: bool = False,
) -> None:
    """Advance Y by one step: upwind advection + central diffusion +
    explicit source S_per_volume (kg/m³/s).  Updates Y in place.
    Boundaries are handled with one-sided differences (no flux).

    Phase 14g: dz is now per-cell (dz_arr) for non-uniform grids.

    Phase 16 (2026-06-16) MULTISPECIES CALLERS: the proper mass-fraction
    conservation form is ∂Y/∂t = (S_i − Y·S_total)/ρ − u·∇Y, where
    S_total = Σ_j S_j is total mass injection by ALL species.  This
    function takes the EFFECTIVE source S_per_volume directly; for
    multi-species correctness, callers should pre-compute:
        S_per_volume = S_i − Y · S_total
    and pass that in.  Single-species legacy callers can pass S_i alone
    (Y stays bounded at 1 by clipping, behavior unchanged from pre-Phase
    16; no dilution interpretation needed).
    """
    Nz, Ny, Nx = Y.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy

    dY = np.zeros_like(Y)

    # Phase 14v-bc: full Way B ghost handling.
    #   x: inlet face Y_inlet, outlet zero-gradient
    #   z: wall (k=0) zero-flux Neumann (ghost = self), top zero-grad
    #   y: periodic wrap (modular indexing)
    for k in prange(0, Nz):
        for j in range(Ny):
            jm2 = (j - 2) % Ny
            jm1 = (j - 1) % Ny
            jp1 = (j + 1) % Ny
            jp2 = (j + 2) % Ny
            for i in range(Nx):
                ui = u[k, j, i]; vi = v[k, j, i]; wi = w[k, j, i]
                Yi = Y[k, j, i]

                # Boundary ghost reads (Way B).
                YL_x = Y_inlet if i == 0 else Y[k, j, i-1]
                YR_x = Yi if i == Nx - 1 else Y[k, j, i+1]
                # z-min: pre-Phase-23 wall (zero-flux, ghost=self) unless
                # cup-burner-style z-min inlet is active.
                if k == 0:
                    if z_min_inlet_active:
                        YL_z = Y_inlet_zmin[j, i]
                    else:
                        YL_z = Yi
                else:
                    YL_z = Y[k-1, j, i]
                YR_z = Yi if k == Nz - 1 else Y[k+1, j, i]

                # ── MUSCL advection (Phase 14k, replaces 1st-order upwind) ─
                if 2 <= i <= Nx - 3:
                    f_xp = muscl_face_value(Y[k, j, i-1], Yi,
                                             Y[k, j, i+1], Y[k, j, i+2], ui)
                    f_xm = muscl_face_value(Y[k, j, i-2], Y[k, j, i-1],
                                             Yi, Y[k, j, i+1], ui)
                    flux_x = ui * (f_xp - f_xm) * inv_dx
                else:
                    if ui >= 0.0:
                        flux_x = ui * (Yi - YL_x) * inv_dx
                    else:
                        flux_x = ui * (YR_x - Yi) * inv_dx
                # y-direction (periodic wrap)
                f_yp = muscl_face_value(Y[k, jm1, i], Yi,
                                         Y[k, jp1, i], Y[k, jp2, i], vi)
                f_ym = muscl_face_value(Y[k, jm2, i], Y[k, jm1, i],
                                         Yi, Y[k, jp1, i], vi)
                flux_y = vi * (f_yp - f_ym) * inv_dy
                inv_d_above = 1.0 / d_face_above[k]
                inv_d_below = 1.0 / d_face_below[k]
                inv_dz_k = 1.0 / dz_arr[k]
                if 2 <= k <= Nz - 3:
                    f_zp = muscl_face_value(Y[k-1, j, i], Yi,
                                             Y[k+1, j, i], Y[k+2, j, i], wi)
                    f_zm = muscl_face_value(Y[k-2, j, i], Y[k-1, j, i],
                                             Yi, Y[k+1, j, i], wi)
                    flux_z = wi * (f_zp - f_zm) / (0.5 * (d_face_above[k] + d_face_below[k]))
                else:
                    if wi >= 0.0:
                        flux_z = wi * (Yi - YL_z) * inv_d_below
                    else:
                        flux_z = wi * (YR_z - Yi) * inv_d_above
                adv = -(flux_x + flux_y + flux_z)

                # ── Central diffusion (FV form for non-uniform dz) ──
                d2Y_x = (YR_x - 2.0 * Yi + YL_x) * inv_dx2
                d2Y_y = (Y[k, jp1, i] - 2.0 * Yi + Y[k, jm1, i]) * inv_dy2
                d2Y_z = (((YR_z - Yi) * inv_d_above
                          - (Yi - YL_z) * inv_d_below) * inv_dz_k)
                diff = D * (d2Y_x + d2Y_y + d2Y_z)

                # ── Source / sink (kg/m³/s → /ρ → 1/s in mass-frac units)
                src = S_per_volume[k, j, i] / rho[k, j, i]

                dY[k, j, i] = (adv + diff + src) * dt

    # Apply update.  Clip to physical [0, 1] range — Y is a mass
    # fraction, so any out-of-range value is unphysical and indicates
    # operator-splitting drift; clipping is mass-conservative under
    # source/sink balance.
    for k in prange(0, Nz):
        for j in range(Ny):
            for i in range(Nx):
                new = Y[k, j, i] + dY[k, j, i]
                if new < 0.0:
                    new = 0.0
                elif new > 1.0:
                    new = 1.0
                Y[k, j, i] = new
