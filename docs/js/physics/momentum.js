/**
 * Tentative-velocity update — JS port of
 * model_outdoor/physics_3d/momentum_3d.py :: step_tentative_velocity.
 *
 * Advances velocity by every RHS term EXCEPT the pressure gradient
 * (Chorin 1967 fractional step, part B.3a):
 *
 *     u* = u^n + dt * ( -(u.grad)u + nu*lap(u) + g_buoy + F_ext/rho )
 *
 * The projection that enforces div(u^{n+1}) = 0 is a separate step and takes
 * u* as its input.
 *
 * Advection is minmod-MUSCL in conservation form; diffusion is central, in
 * finite-volume form so it stays correct on the stretched z-grid; buoyancy is
 * Boussinesq and applies to w only.
 *
 * BOUNDARIES ("Way B" — ghosts computed on the fly, never stored):
 *   k=0     no-slip wall: all three ghost velocities are 0, at the half-cell
 *           distance d_face_below[0] = dz[0]/2. That is what generates the
 *           2*nu*u/dz wall shear — the no-slip condition emerges from the FV
 *           stencil rather than being pinned afterwards.
 *   k=Nz-1  zero-gradient (ghost = self)
 *   i=0     inlet Dirichlet from u_inlet / v_inlet / w_inlet
 *   i=Nx-1  outlet zero-gradient
 *   y       periodic
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i. Inlet arrays are
 * (Nz*Ny), idx = k*Ny + j.
 *
 * References: Chorin (1967) JCP 2:12; Boussinesq (1903); van Leer (1979).
 */
import { musclFaceValue } from './muscl.js';

/** Gravitational acceleration [m/s^2]. */
export const G = 9.81;

/** Dynamic viscosity of the gas [Pa.s]. */
export const MU_GAS = 1.8e-5;

/**
 * Advance u, v, w by one tentative step, in place.
 *
 * Every cell is updated from the OLD velocity field via separate du/dv/dw
 * accumulators, so the result cannot depend on sweep order.
 *
 * @param {Float64Array} u   (Nz*Ny*Nx) modified in place
 * @param {Float64Array} v
 * @param {Float64Array} w
 * @param {Float64Array} rho density [kg/m^3]
 * @param {Float64Array} Tg  gas temperature [K]
 * @param {Float64Array} FxExt body force [N/m^3]
 * @param {Float64Array} FyExt
 * @param {Float64Array} FzExt
 * @param {number} dt
 * @param {number} dx
 * @param {number} dy
 * @param {Float64Array} dzArr      (Nz)
 * @param {Float64Array} dFaceAbove (Nz)
 * @param {Float64Array} dFaceBelow (Nz)
 * @param {number} Tamb
 * @param {Float64Array} uInlet (Nz*Ny) inlet face u
 * @param {Float64Array} vInlet (Nz*Ny)
 * @param {Float64Array} wInlet (Nz*Ny)
 * @param {object} opts {nx, ny, nz}
 */
export function stepTentativeVelocity(
  u, v, w, rho, Tg, FxExt, FyExt, FzExt, dt, dx, dy,
  dzArr, dFaceAbove, dFaceBelow, Tamb, uInlet, vInlet, wInlet,
  { nx, ny, nz } = {},
) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const invDx2 = invDx * invDx;
  const invDy2 = invDy * invDy;
  const nxy = ny * nx;

  const du = new Float64Array(u.length);
  const dv = new Float64Array(v.length);
  const dw = new Float64Array(w.length);

  for (let k = 0; k < nz; k++) {
    const kBase = k * nxy;
    for (let j = 0; j < ny; j++) {
      const jm2 = (((j - 2) % ny) + ny) % ny;
      const jm1 = (((j - 1) % ny) + ny) % ny;
      const jp1 = (j + 1) % ny;
      const jp2 = (j + 2) % ny;
      const row = kBase + j * nx;
      const rjm2 = kBase + jm2 * nx;
      const rjm1 = kBase + jm1 * nx;
      const rjp1 = kBase + jp1 * nx;
      const rjp2 = kBase + jp2 * nx;
      const inl = k * ny + j;

      for (let i = 0; i < nx; i++) {
        const c = row + i;
        const ui = u[c];
        const vi = v[c];
        const wi = w[c];
        const rhoI = rho[c];
        const nuI = MU_GAS / rhoI;

        // ── Way B ghosts ─────────────────────────────────────────────────
        let uLx, vLx, wLx;
        if (i === 0) { uLx = uInlet[inl]; vLx = vInlet[inl]; wLx = wInlet[inl]; }
        else { uLx = u[c - 1]; vLx = v[c - 1]; wLx = w[c - 1]; }
        let uRx, vRx, wRx;
        if (i === nx - 1) { uRx = ui; vRx = vi; wRx = wi; }
        else { uRx = u[c + 1]; vRx = v[c + 1]; wRx = w[c + 1]; }
        let uLz, vLz, wLz;
        if (k === 0) { uLz = 0.0; vLz = 0.0; wLz = 0.0; }   // no-slip wall
        else { uLz = u[c - nxy]; vLz = v[c - nxy]; wLz = w[c - nxy]; }
        let uRz, vRz, wRz;
        if (k === nz - 1) { uRz = ui; vRz = vi; wRz = wi; }
        else { uRz = u[c + nxy]; vRz = v[c + nxy]; wRz = w[c + nxy]; }

        // ── advection, MUSCL in conservation form ────────────────────────
        let dudx, dvdx, dwdx;
        if (i >= 2 && i <= nx - 3) {
          const uXp = musclFaceValue(u[c - 1], ui, u[c + 1], u[c + 2], ui);
          const uXm = musclFaceValue(u[c - 2], u[c - 1], ui, u[c + 1], ui);
          const vXp = musclFaceValue(v[c - 1], vi, v[c + 1], v[c + 2], ui);
          const vXm = musclFaceValue(v[c - 2], v[c - 1], vi, v[c + 1], ui);
          const wXp = musclFaceValue(w[c - 1], wi, w[c + 1], w[c + 2], ui);
          const wXm = musclFaceValue(w[c - 2], w[c - 1], wi, w[c + 1], ui);
          dudx = (uXp - uXm) * invDx;
          dvdx = (vXp - vXm) * invDx;
          dwdx = (wXp - wXm) * invDx;
        } else if (ui >= 0.0) {
          dudx = (ui - uLx) * invDx;
          dvdx = (vi - vLx) * invDx;
          dwdx = (wi - wLx) * invDx;
        } else {
          dudx = (uRx - ui) * invDx;
          dvdx = (vRx - vi) * invDx;
          dwdx = (wRx - wi) * invDx;
        }

        const uYp = musclFaceValue(u[rjm1 + i], ui, u[rjp1 + i], u[rjp2 + i], vi);
        const uYm = musclFaceValue(u[rjm2 + i], u[rjm1 + i], ui, u[rjp1 + i], vi);
        const vYp = musclFaceValue(v[rjm1 + i], vi, v[rjp1 + i], v[rjp2 + i], vi);
        const vYm = musclFaceValue(v[rjm2 + i], v[rjm1 + i], vi, v[rjp1 + i], vi);
        const wYp = musclFaceValue(w[rjm1 + i], wi, w[rjp1 + i], w[rjp2 + i], vi);
        const wYm = musclFaceValue(w[rjm2 + i], w[rjm1 + i], wi, w[rjp1 + i], vi);
        const dudy = (uYp - uYm) * invDy;
        const dvdy = (vYp - vYm) * invDy;
        const dwdy = (wYp - wYm) * invDy;

        let dudz, dvdz, dwdz;
        if (k >= 2 && k <= nz - 3) {
          const invDzEff = 1.0 / (0.5 * (dFaceAbove[k] + dFaceBelow[k]));
          const uZp = musclFaceValue(u[c - nxy], ui, u[c + nxy], u[c + 2 * nxy], wi);
          const uZm = musclFaceValue(u[c - 2 * nxy], u[c - nxy], ui, u[c + nxy], wi);
          const vZp = musclFaceValue(v[c - nxy], vi, v[c + nxy], v[c + 2 * nxy], wi);
          const vZm = musclFaceValue(v[c - 2 * nxy], v[c - nxy], vi, v[c + nxy], wi);
          const wZp = musclFaceValue(w[c - nxy], wi, w[c + nxy], w[c + 2 * nxy], wi);
          const wZm = musclFaceValue(w[c - 2 * nxy], w[c - nxy], wi, w[c + nxy], wi);
          dudz = (uZp - uZm) * invDzEff;
          dvdz = (vZp - vZm) * invDzEff;
          dwdz = (wZp - wZm) * invDzEff;
        } else if (wi >= 0.0) {
          const invDBelow = 1.0 / dFaceBelow[k];
          dudz = (ui - uLz) * invDBelow;
          dvdz = (vi - vLz) * invDBelow;
          dwdz = (wi - wLz) * invDBelow;
        } else {
          const invDAbove = 1.0 / dFaceAbove[k];
          dudz = (uRz - ui) * invDAbove;
          dvdz = (vRz - vi) * invDAbove;
          dwdz = (wRz - wi) * invDAbove;
        }

        const advU = -(ui * dudx + vi * dudy + wi * dudz);
        const advV = -(ui * dvdx + vi * dvdy + wi * dvdz);
        const advW = -(ui * dwdx + vi * dwdy + wi * dwdz);

        // ── viscous diffusion, FV form in z ──────────────────────────────
        const invDzK = 1.0 / dzArr[k];
        const invDAbove = 1.0 / dFaceAbove[k];
        const invDBelow = 1.0 / dFaceBelow[k];
        const d2udx2 = (uRx - 2.0 * ui + uLx) * invDx2;
        const d2udy2 = (u[rjp1 + i] - 2.0 * ui + u[rjm1 + i]) * invDy2;
        const d2udz2 = ((uRz - ui) * invDAbove - (ui - uLz) * invDBelow) * invDzK;
        const d2vdx2 = (vRx - 2.0 * vi + vLx) * invDx2;
        const d2vdy2 = (v[rjp1 + i] - 2.0 * vi + v[rjm1 + i]) * invDy2;
        const d2vdz2 = ((vRz - vi) * invDAbove - (vi - vLz) * invDBelow) * invDzK;
        const d2wdx2 = (wRx - 2.0 * wi + wLx) * invDx2;
        const d2wdy2 = (w[rjp1 + i] - 2.0 * wi + w[rjm1 + i]) * invDy2;
        const d2wdz2 = ((wRz - wi) * invDAbove - (wi - wLz) * invDBelow) * invDzK;
        const viscU = nuI * (d2udx2 + d2udy2 + d2udz2);
        const viscV = nuI * (d2vdx2 + d2vdy2 + d2vdz2);
        const viscW = nuI * (d2wdx2 + d2wdy2 + d2wdz2);

        // ── Boussinesq buoyancy, z only: hot gas rises ───────────────────
        const buoyW = (G * (Tg[c] - Tamb)) / Tamb;

        // ── external body force (drag, etc.) ─────────────────────────────
        du[c] = (advU + viscU + FxExt[c] / rhoI) * dt;
        dv[c] = (advV + viscV + FyExt[c] / rhoI) * dt;
        dw[c] = (advW + viscW + buoyW + FzExt[c] / rhoI) * dt;
      }
    }
  }

  for (let c = 0; c < u.length; c++) {
    u[c] += du[c];
    v[c] += dv[c];
    w[c] += dw[c];
  }
}
