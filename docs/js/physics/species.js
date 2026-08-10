/**
 * Species transport — JS port of
 * model_outdoor/physics_3d/species_3d.py :: step_species_transport.
 *
 * Advances a mass fraction Y by one step:
 *
 *     dY/dt = -(u.grad)Y  +  D*lap(Y)  +  S/rho
 *
 * with MUSCL advection (Phase 14k), central diffusion in flux-volume form so
 * it stays correct on the non-uniform z-grid, and an explicit volumetric
 * source. Y is updated in place.
 *
 * This is the single most expensive kernel in the 2D profile — 21.9% of the
 * loop even after the serial-dispatch fix, because it runs once per chemistry
 * substep.
 *
 * MULTISPECIES CALLERS: the conservative form is
 * dY_i/dt = (S_i - Y_i*S_total)/rho - u.grad Y_i. This kernel takes the
 * EFFECTIVE source directly, so a multi-species caller must pre-compute
 * `S_per_volume = S_i - Y_i * S_total`. Single-species callers can pass S_i
 * alone. (Phase 16.)
 *
 * The final clip to [0, 1] is deliberate and mass-conservative under
 * source/sink balance: Y is a mass fraction, so an out-of-range value means
 * operator-splitting drift rather than physics.
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i, matching NumPy C
 * order so golden vectors transfer without reshaping.
 */
import { musclFaceValue } from './muscl.js';

/** Laminar mass diffusivity at fire temperatures [m^2/s]. */
export const D_LAMINAR = 1.0e-4;

/**
 * Advance Y one step. Updates `Y` in place.
 *
 * Boundaries follow the Python's "Way B" ghosting exactly:
 *   x  inlet face value `Yinlet`, outlet zero-gradient
 *   z  wall at k=0 zero-flux (ghost = self) unless a z-min inlet is active;
 *      top zero-gradient
 *   y  periodic
 *
 * @param {Float64Array} Y            (Nz*Ny*Nx) mass fraction, updated in place
 * @param {Float64Array} rho          density [kg/m^3]
 * @param {Float64Array} u
 * @param {Float64Array} v
 * @param {Float64Array} w
 * @param {Float64Array} SPerVolume   effective net source [kg/m^3/s]
 * @param {number} dt
 * @param {number} dx
 * @param {number} dy
 * @param {Float64Array} dzArr        (Nz) per-cell vertical spacing
 * @param {Float64Array} dFaceAbove   (Nz) centre-to-centre distance to k+1
 * @param {Float64Array} dFaceBelow   (Nz) centre-to-centre distance to k-1
 * @param {number} D                  mass diffusivity
 * @param {number} Yinlet             x-inlet ghost value
 * @param {object} opts               {nx, ny, nz, YinletZmin?, zMinInletActive?}
 */
export function stepSpeciesTransport(
  Y, rho, u, v, w, SPerVolume, dt, dx, dy, dzArr, dFaceAbove, dFaceBelow,
  D, Yinlet,
  { nx, ny, nz, YinletZmin = null, zMinInletActive = false } = {},
) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const invDx2 = invDx * invDx;
  const invDy2 = invDy * invDy;
  const nxy = ny * nx;

  // Separate accumulator, exactly as the Python: every cell reads the OLD Y,
  // so the result cannot depend on sweep order.
  const dY = new Float64Array(Y.length);

  for (let k = 0; k < nz; k++) {
    const kBase = k * nxy;
    const invDAbove = 1.0 / dFaceAbove[k];
    const invDBelow = 1.0 / dFaceBelow[k];
    const invDzK = 1.0 / dzArr[k];

    for (let j = 0; j < ny; j++) {
      // JS % keeps the dividend's sign; Python's does not. Normalise.
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
        const ui = u[c];
        const vi = v[c];
        const wi = w[c];
        const Yi = Y[c];

        // ── Way B ghost reads ────────────────────────────────────────────
        const YLx = i === 0 ? Yinlet : Y[c - 1];
        const YRx = i === nx - 1 ? Yi : Y[c + 1];
        let YLz;
        if (k === 0) {
          YLz = zMinInletActive && YinletZmin ? YinletZmin[j * nx + i] : Yi;
        } else {
          YLz = Y[c - nxy];
        }
        const YRz = k === nz - 1 ? Yi : Y[c + nxy];

        // ── MUSCL advection ──────────────────────────────────────────────
        let fluxX;
        if (i >= 2 && i <= nx - 3) {
          const fXp = musclFaceValue(Y[c - 1], Yi, Y[c + 1], Y[c + 2], ui);
          const fXm = musclFaceValue(Y[c - 2], Y[c - 1], Yi, Y[c + 1], ui);
          fluxX = ui * (fXp - fXm) * invDx;
        } else if (ui >= 0.0) {
          fluxX = ui * (Yi - YLx) * invDx;
        } else {
          fluxX = ui * (YRx - Yi) * invDx;
        }

        const fYp = musclFaceValue(Y[rjm1 + i], Yi, Y[rjp1 + i], Y[rjp2 + i], vi);
        const fYm = musclFaceValue(Y[rjm2 + i], Y[rjm1 + i], Yi, Y[rjp1 + i], vi);
        const fluxY = vi * (fYp - fYm) * invDy;

        let fluxZ;
        if (k >= 2 && k <= nz - 3) {
          const fZp = musclFaceValue(Y[c - nxy], Yi, Y[c + nxy], Y[c + 2 * nxy], wi);
          const fZm = musclFaceValue(Y[c - 2 * nxy], Y[c - nxy], Yi, Y[c + nxy], wi);
          fluxZ = (wi * (fZp - fZm)) / (0.5 * (dFaceAbove[k] + dFaceBelow[k]));
        } else if (wi >= 0.0) {
          fluxZ = wi * (Yi - YLz) * invDBelow;
        } else {
          fluxZ = wi * (YRz - Yi) * invDAbove;
        }
        const adv = -(fluxX + fluxY + fluxZ);

        // ── central diffusion, finite-volume form in z ───────────────────
        const d2Yx = (YRx - 2.0 * Yi + YLx) * invDx2;
        const d2Yy = (Y[rjp1 + i] - 2.0 * Yi + Y[rjm1 + i]) * invDy2;
        const d2Yz = ((YRz - Yi) * invDAbove - (Yi - YLz) * invDBelow) * invDzK;
        const diff = D * (d2Yx + d2Yy + d2Yz);

        // ── volumetric source, converted to mass-fraction rate ───────────
        const src = SPerVolume[c] / rho[c];

        dY[c] = (adv + diff + src) * dt;
      }
    }
  }

  // Apply, clipping to the physical range.
  for (let c = 0; c < Y.length; c++) {
    let n = Y[c] + dY[c];
    if (n < 0.0) n = 0.0;
    else if (n > 1.0) n = 1.0;
    Y[c] = n;
  }
}
