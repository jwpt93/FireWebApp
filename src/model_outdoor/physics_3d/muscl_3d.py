"""MUSCL 2nd-order advection helpers (Phase 14k).

Replaces first-order upwind everywhere we do hyperbolic advection.

The minmod-limited MUSCL scheme (van Leer 1979, Sweby 1984):

For a 1D conservation law ∂φ/∂t + u·∂φ/∂x = 0 on a uniform mesh,
the flux at face i+1/2 is reconstructed from the upwind side as

    if u_{i+1/2} ≥ 0 :   φ_face = φ_i   + 0.5 · minmod(φ_i - φ_{i-1},  φ_{i+1} - φ_i)
    if u_{i+1/2} < 0 :   φ_face = φ_{i+1} − 0.5 · minmod(φ_{i+1} - φ_i, φ_{i+2} - φ_{i+1})

with minmod(a, b) = a if (a*b > 0 and |a|<|b|), b if (a*b > 0 and |b|<|a|), 0 if a*b ≤ 0.

The minmod limiter is **TVD** (Sweby 1984): no spurious oscillations near
discontinuities.  On smooth solutions it gives 2nd-order accuracy in
space, decaying to 1st-order at extrema (a feature, not a bug — preserves
monotonicity).

Numerical diffusion of pure 1st-order upwind: D_num = u·dx/2 ≈ 0.2 m²/s
for U=4 m/s, dx=0.1 — comparable to physical ν_t.  MUSCL reduces this
by ~order 10 in smooth regions, sharpening fronts.

References:
- van Leer, B. (1979) JCP 32:101 — original MUSCL
- Sweby, P. (1984) SIAM J. Numer. Anal. 21:995 — TVD limiters incl. minmod
- LeVeque, R. (2002) Finite Volume Methods §6.13

Boundary cells (i=0 or i=N-1 for x; analogous in y, z) fall back to
1st-order upwind: not enough cells for a 3-point stencil.
"""
import numpy as np
from numba import njit, prange


@njit(inline='always', cache=True)
def minmod(a: float, b: float) -> float:
    """Minmod limiter: returns the smaller-magnitude argument when same sign,
    zero when opposite signs.  TVD."""
    if a * b <= 0.0:
        return 0.0
    if a > 0.0:
        return a if a < b else b
    return a if a > b else b


@njit(inline='always', cache=True)
def muscl_face_value(phi_im1: float, phi_i: float, phi_ip1: float,
                     phi_ip2: float, u_face: float) -> float:
    """Compute the face value φ_{i+1/2} via minmod-limited MUSCL.

    Reconstructs from the upwind side using a 3-cell stencil.  Caller
    multiplies by u_face to get the flux.

    Parameters:
        phi_im1, phi_i, phi_ip1, phi_ip2 : scalar at cells i-1, i, i+1, i+2
        u_face : velocity at face i+1/2 (used for direction only)
    """
    if u_face >= 0.0:
        slope = minmod(phi_i - phi_im1, phi_ip1 - phi_i)
        return phi_i + 0.5 * slope
    else:
        slope = minmod(phi_ip1 - phi_i, phi_ip2 - phi_ip1)
        return phi_ip1 - 0.5 * slope


@njit(cache=True, parallel=True)
def advect_3d_scalar_muscl(
    phi: np.ndarray,         # (Nz, Ny, Nx) field to advect (read-only)
    u: np.ndarray,           # (Nz, Ny, Nx) cell-centered x-velocity
    v: np.ndarray,           # (Nz, Ny, Nx) cell-centered y-velocity
    w: np.ndarray,           # (Nz, Ny, Nx) cell-centered z-velocity
    dx: float, dy: float,
    d_face_above: np.ndarray,    # (Nz,) cell-center distance to k+1
    d_face_below: np.ndarray,    # (Nz,) cell-center distance to k-1
    rhs: np.ndarray,         # (Nz, Ny, Nx) — accumulator: -u·∇φ added in-place
    phi_inlet: float,        # Phase 14v-bc: x-inlet face value (Way B ghost)
    # Phase 23 Refactor 2C: z-min inlet ghost (cup burner).  Caller
    # ALWAYS passes an array (dummy zeros are fine when unused); the
    # ``z_min_inlet_active`` flag decides whether the array is consulted
    # or the pre-Phase-23 zero-flux wall (ghost = self) is used.  Default
    # (False + dummy) preserves bit-exact-invariant outdoor behaviour.
    phi_inlet_zmin: np.ndarray = np.zeros((1, 1)),
    z_min_inlet_active: bool = False,
) -> None:
    """Add -(u·∇)φ to ``rhs`` using minmod-MUSCL flux differencing.

    Cell-centered velocity used as proxy for face velocity — fine for
    smooth subsonic flow with sign-consistent stencils.  Boundary cells
    (≤1 from edge) and 2nd-from-edge cells (where the 4-cell stencil
    can't fit) fall back to 1st-order upwind.
    """
    Nz, Ny, Nx = phi.shape
    # Phase 14v-bc: full Way B.  x: inlet phi_inlet, outlet zero-grad.
    # z: wall (k=0) zero-flux Neumann (ghost=self), top zero-grad.
    # y: periodic via modular indexing.
    for k in prange(0, Nz):
        inv_d_below_k = 1.0 / d_face_below[k]
        inv_d_above_k = 1.0 / d_face_above[k]
        for j in range(Ny):
            jm2 = (j - 2) % Ny
            jm1 = (j - 1) % Ny
            jp1 = (j + 1) % Ny
            jp2 = (j + 2) % Ny
            for i in range(Nx):
                u_c = u[k, j, i]
                v_c = v[k, j, i]
                w_c = w[k, j, i]
                phi_c = phi[k, j, i]

                # Boundary ghost reads (Way B)
                phi_xL = phi_inlet if i == 0 else phi[k, j, i-1]
                phi_xR = phi_c if i == Nx - 1 else phi[k, j, i+1]
                # z-min ghost: pre-Phase-23 zero-flux wall (ghost=self)
                # unless a cup-burner-style inlet is active.
                if k == 0:
                    if z_min_inlet_active:
                        phi_zL = phi_inlet_zmin[j, i]
                    else:
                        phi_zL = phi_c
                else:
                    phi_zL = phi[k-1, j, i]
                phi_zR = phi_c if k == Nz - 1 else phi[k+1, j, i]

                # ── x-direction MUSCL flux differencing ─────────────
                if 2 <= i <= Nx - 3:
                    # face i+1/2: cells i-1, i, i+1, i+2
                    f_xp = muscl_face_value(
                        phi[k, j, i - 1], phi_c,
                        phi[k, j, i + 1], phi[k, j, i + 2], u_c)
                    # face i-1/2: cells i-2, i-1, i, i+1
                    f_xm = muscl_face_value(
                        phi[k, j, i - 2], phi[k, j, i - 1],
                        phi_c, phi[k, j, i + 1], u_c)
                    flux_x = u_c * (f_xp - f_xm) / dx
                else:
                    # 1st-order upwind fallback (with ghost-aware reads)
                    if u_c >= 0.0:
                        flux_x = u_c * (phi_c - phi_xL) / dx
                    else:
                        flux_x = u_c * (phi_xR - phi_c) / dx

                # ── y-direction MUSCL (periodic wrap) ───────────────
                f_yp = muscl_face_value(
                    phi[k, jm1, i], phi_c,
                    phi[k, jp1, i], phi[k, jp2, i], v_c)
                f_ym = muscl_face_value(
                    phi[k, jm2, i], phi[k, jm1, i],
                    phi_c, phi[k, jp1, i], v_c)
                flux_y = v_c * (f_yp - f_ym) / dy

                # ── z-direction MUSCL (per-cell distances; non-uniform) ─
                # Use d_face_below/above for upwind divisor (matches
                # existing 1st-order upwind convention).
                if 2 <= k <= Nz - 3:
                    f_zp = muscl_face_value(
                        phi[k - 1, j, i], phi[k, j, i],
                        phi[k + 1, j, i], phi[k + 2, j, i], w_c)
                    f_zm = muscl_face_value(
                        phi[k - 2, j, i], phi[k - 1, j, i],
                        phi[k, j, i], phi[k + 1, j, i], w_c)
                    # Use average of d_above and d_below as effective dz
                    # for the flux gradient (good enough for smooth z-grids).
                    flux_z = w_c * (f_zp - f_zm) / (0.5 * (d_face_above[k] + d_face_below[k]))
                else:
                    if w_c >= 0.0:
                        flux_z = w_c * (phi_c - phi_zL) * inv_d_below_k
                    else:
                        flux_z = w_c * (phi_zR - phi_c) * inv_d_above_k

                rhs[k, j, i] -= flux_x + flux_y + flux_z
