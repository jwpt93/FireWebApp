/**
 * MUSCL 2nd-order advection helpers — JS port of
 * model_outdoor/physics_3d/muscl_3d.py (Phase 14k).
 *
 * Minmod-limited MUSCL (van Leer 1979; Sweby 1984). For
 * dphi/dt + u.grad phi = 0 the face value at i+1/2 is reconstructed from
 * the upwind side:
 *
 *   u_face >= 0:  phi_face = phi_i   + 0.5*minmod(phi_i - phi_i-1, phi_i+1 - phi_i)
 *   u_face <  0:  phi_face = phi_i+1 - 0.5*minmod(phi_i+1 - phi_i, phi_i+2 - phi_i+1)
 *
 * minmod is TVD, so no spurious oscillations near discontinuities; 2nd order
 * on smooth solutions, dropping to 1st at extrema (deliberate — it preserves
 * monotonicity).
 *
 * Why it matters here: 1st-order upwind carries numerical diffusion
 * D_num = u*dx/2 ~ 0.2 m^2/s at U=4, dx=0.1 — comparable to the physical
 * turbulent viscosity, which would smear the very fronts this model exists
 * to resolve.
 *
 * INDEXING.  Python uses phi[k, j, i] on (Nz, Ny, Nx); here arrays are flat
 * Float64Array with idx = (k*Ny + j)*Nx + i, matching NumPy C order exactly
 * so golden vectors transfer without reshaping.
 *
 * Ported for general Ny rather than specialised to the 2D Ny=1 case, so the
 * verification vectors can be generated at any shape. At Ny=1 every y
 * neighbour is the cell itself, both minmod slopes are zero, and flux_y
 * vanishes identically — which is the 2D slab assumption falling out rather
 * than being imposed.
 *
 * References: van Leer (1979) JCP 32:101; Sweby (1984) SIAM J. Numer. Anal.
 * 21:995; LeVeque (2002) Finite Volume Methods 6.13.
 */

/**
 * Minmod limiter: smaller-magnitude argument when the two agree in sign,
 * zero when they don't.
 */
export function minmod(a, b) {
  if (a * b <= 0.0) return 0.0;
  if (a > 0.0) return a < b ? a : b;
  return a > b ? a : b;
}

/**
 * Face value phi_{i+1/2} by minmod-limited MUSCL, reconstructed from the
 * upwind side of a 4-cell stencil. `uFace` is used for direction only.
 */
export function musclFaceValue(phiIm1, phiI, phiIp1, phiIp2, uFace) {
  if (uFace >= 0.0) {
    return phiI + 0.5 * minmod(phiI - phiIm1, phiIp1 - phiI);
  }
  return phiIp1 - 0.5 * minmod(phiIp1 - phiI, phiIp2 - phiIp1);
}

/**
 * Add -(u.grad)phi to `rhs` by MUSCL flux differencing.
 *
 * Boundary handling is "Way B" ghosting, matching the Python exactly:
 *   x  inlet face value `phiInlet`, outlet zero-gradient
 *   z  wall at k=0 is zero-flux (ghost = self) unless a z-min inlet is
 *      active; top is zero-gradient
 *   y  periodic, via modular indexing
 *
 * Cells within two of an x or z edge fall back to 1st-order upwind, because
 * the 4-cell stencil does not fit. y always has the stencil (it wraps).
 *
 * Cell-centred velocities stand in for face velocities — adequate for smooth
 * subsonic flow with sign-consistent stencils, and what the Python does.
 *
 * @param {Float64Array} phi   (Nz*Ny*Nx) advected field, read-only
 * @param {Float64Array} u     x-velocity, cell centred
 * @param {Float64Array} v     y-velocity
 * @param {Float64Array} w     z-velocity
 * @param {number} dx
 * @param {number} dy
 * @param {Float64Array} dFaceAbove (Nz) centre-to-centre distance to k+1
 * @param {Float64Array} dFaceBelow (Nz) centre-to-centre distance to k-1
 * @param {Float64Array} rhs   (Nz*Ny*Nx) accumulator, modified in place
 * @param {number} phiInlet    x-inlet ghost value
 * @param {object} [opts]
 * @param {Float64Array} [opts.phiInletZmin] (Ny*Nx) z-min inlet ghosts
 * @param {boolean} [opts.zMinInletActive]
 * @param {number} opts.nx
 * @param {number} opts.ny
 * @param {number} opts.nz
 */
export function advect3dScalarMuscl(
  phi, u, v, w, dx, dy, dFaceAbove, dFaceBelow, rhs, phiInlet,
  { nx, ny, nz, phiInletZmin = null, zMinInletActive = false } = {},
) {
  const nxy = ny * nx;

  for (let k = 0; k < nz; k++) {
    const invDBelowK = 1.0 / dFaceBelow[k];
    const invDAboveK = 1.0 / dFaceAbove[k];
    const kBase = k * nxy;

    for (let j = 0; j < ny; j++) {
      // Python's (j - 2) % Ny etc. — JS % keeps the sign of the dividend,
      // so add ny before taking the modulus.
      const jm2 = (((j - 2) % ny) + ny) % ny;
      const jm1 = (((j - 1) % ny) + ny) % ny;
      const jp1 = (j + 1) % ny;
      const jp2 = (j + 2) % ny;
      const row = kBase + j * nx;
      const rjm2 = kBase + jm2 * nx;
      const rjm1 = kBase + jm1 * nx;
      const rjp1 = kBase + jp1 * nx;
      const rjp2 = kBase + jp2 * nx;

      for (let i = 0; i < nx; i++) {
        const c = row + i;
        const uC = u[c];
        const vC = v[c];
        const wC = w[c];
        const phiC = phi[c];

        // ── Way B ghost reads ────────────────────────────────────────────
        const phiXL = i === 0 ? phiInlet : phi[c - 1];
        const phiXR = i === nx - 1 ? phiC : phi[c + 1];
        let phiZL;
        if (k === 0) {
          phiZL = zMinInletActive && phiInletZmin
            ? phiInletZmin[j * nx + i]
            : phiC;
        } else {
          phiZL = phi[c - nxy];
        }
        const phiZR = k === nz - 1 ? phiC : phi[c + nxy];

        // ── x ────────────────────────────────────────────────────────────
        let fluxX;
        if (i >= 2 && i <= nx - 3) {
          const fXp = musclFaceValue(phi[c - 1], phiC, phi[c + 1], phi[c + 2], uC);
          const fXm = musclFaceValue(phi[c - 2], phi[c - 1], phiC, phi[c + 1], uC);
          fluxX = (uC * (fXp - fXm)) / dx;
        } else if (uC >= 0.0) {
          fluxX = (uC * (phiC - phiXL)) / dx;
        } else {
          fluxX = (uC * (phiXR - phiC)) / dx;
        }

        // ── y (periodic, always has its stencil) ─────────────────────────
        const fYp = musclFaceValue(phi[rjm1 + i], phiC, phi[rjp1 + i], phi[rjp2 + i], vC);
        const fYm = musclFaceValue(phi[rjm2 + i], phi[rjm1 + i], phiC, phi[rjp1 + i], vC);
        const fluxY = (vC * (fYp - fYm)) / dy;

        // ── z (non-uniform spacing) ──────────────────────────────────────
        let fluxZ;
        if (k >= 2 && k <= nz - 3) {
          const fZp = musclFaceValue(phi[c - nxy], phiC, phi[c + nxy], phi[c + 2 * nxy], wC);
          const fZm = musclFaceValue(phi[c - 2 * nxy], phi[c - nxy], phiC, phi[c + nxy], wC);
          // Average of the two face distances as the effective dz — the
          // Python's choice, adequate for smooth z-grids.
          fluxZ = (wC * (fZp - fZm)) / (0.5 * (dFaceAbove[k] + dFaceBelow[k]));
        } else if (wC >= 0.0) {
          fluxZ = wC * (phiC - phiZL) * invDBelowK;
        } else {
          fluxZ = wC * (phiZR - phiC) * invDAboveK;
        }

        rhs[c] -= fluxX + fluxY + fluxZ;
      }
    }
  }
}
