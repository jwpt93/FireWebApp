/**
 * Turbulent diffusion of a passive scalar — JS port of
 * model_outdoor/physics_3d/turbulence_3d.py :: apply_turbulent_diffusion.
 *
 * Adds div(D_t grad field) explicitly, with D_t = nu_t / sc_t. Used for both
 * species (Schmidt number 0.7) and gas energy (Prandtl 0.85), which together
 * are ~18% of the 2D loop.
 *
 * SUB-STEPS FOR STABILITY. Explicit diffusion needs a Fourier number
 * Fo = D_t*dt/h_min^2 <= 0.4, so the kernel sub-cycles internally rather than
 * constraining the outer timestep. h_min uses the SMALLEST dz — the bed cells
 * — because that is the most restrictive face.
 *
 * FOUR EARLY RETURNS, ALL DELIBERATE, and the port keeps every one:
 *   - D_t_max <= 1e-12   nothing to diffuse
 *   - D_t_max not finite  a NaN or inf in nu_t. This happens transiently
 *                         during combustion-driven density spikes, before the
 *                         realizable C_mu self-limit catches a nu_t blow-up.
 *                         Skipping beats corrupting everything downstream —
 *                         and note it is a numerical guard, NOT a physical cap.
 *   - dt_sub_max <= 0     underflow
 *   - n_sub > 1000        too stiff this step; skip rather than corrupt
 *
 * Boundaries are zero-flux Neumann everywhere non-periodic (ghost = self):
 * scalars do not leave through walls, inlet or outlet during the diffusion
 * step. y is periodic.
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i.
 */

/** Turbulent Schmidt number — species diffusion. */
export const SC_T = 0.7;

/** Turbulent Prandtl number — gas-energy diffusion. */
export const PR_T = 0.85;

const N_SUB_MAX = 1000;
const FO_TARGET = 0.4;

/**
 * Diffuse `field` in place.
 *
 * @param {Float64Array} field   (Nz*Ny*Nx) modified in place
 * @param {Float64Array} nuT     turbulent viscosity [m^2/s]
 * @param {number} scT           turbulent Schmidt or Prandtl number
 * @param {number} dt
 * @param {number} dx
 * @param {number} dy
 * @param {Float64Array} dzArr      (Nz)
 * @param {Float64Array} dFaceAbove (Nz)
 * @param {Float64Array} dFaceBelow (Nz)
 * @param {object} opts {nx, ny, nz}
 */
export function applyTurbulentDiffusion(
  field, nuT, scT, dt, dx, dy, dzArr, dFaceAbove, dFaceBelow,
  { nx, ny, nz } = {},
) {
  const invDx2 = 1.0 / (dx * dx);
  const invDy2 = 1.0 / (dy * dy);
  const nxy = ny * nx;

  let dzMin = dzArr[0];
  for (let k = 1; k < nz; k++) if (dzArr[k] < dzMin) dzMin = dzArr[k];
  const hMin2 = Math.min(dx * dx, Math.min(dy * dy, dzMin * dzMin));

  // Peak nu_t over the INTERIOR only, matching the Python's loop bounds
  // (k and i skip the first and last planes; j does not).
  let nuTMax = 0.0;
  for (let k = 1; k < nz - 1; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const v = nuT[(k * ny + j) * nx + i];
        if (v > nuTMax) nuTMax = v;
      }
    }
  }
  const DtMax = nuTMax / scT;

  if (DtMax <= 1.0e-12) return;         // nothing to diffuse
  if (!Number.isFinite(DtMax)) return;  // transient nu_t blow-up — skip, don't corrupt
  const dtSubMax = (FO_TARGET * hMin2) / DtMax;
  if (dtSubMax <= 0.0) return;          // underflow
  const nSubTarget = dt / dtSubMax;
  if (nSubTarget > N_SUB_MAX) return;   // too stiff this step
  const nSub = Math.max(1, Math.ceil(nSubTarget));
  const dtSub = dt / nSub;

  const df = new Float64Array(field.length);

  for (let s = 0; s < nSub; s++) {
    for (let k = 0; k < nz; k++) {
      const invDzK = 1.0 / dzArr[k];
      const invDAbove = 1.0 / dFaceAbove[k];
      const invDBelow = 1.0 / dFaceBelow[k];
      const kBase = k * nxy;
      for (let j = 0; j < ny; j++) {
        const jm1 = (((j - 1) % ny) + ny) % ny;
        const jp1 = (j + 1) % ny;
        const row = kBase + j * nx;
        const rjm1 = kBase + jm1 * nx;
        const rjp1 = kBase + jp1 * nx;
        for (let i = 0; i < nx; i++) {
          const c = row + i;
          const fc = field[c];
          const fxL = i === 0 ? fc : field[c - 1];
          const fxR = i === nx - 1 ? fc : field[c + 1];
          const fzL = k === 0 ? fc : field[c - nxy];
          const fzR = k === nz - 1 ? fc : field[c + nxy];
          const Dt = nuT[c] / scT;
          const d2x = (fxR - 2.0 * fc + fxL) * invDx2;
          const d2y = (field[rjp1 + i] - 2.0 * fc + field[rjm1 + i]) * invDy2;
          const d2z = ((fzR - fc) * invDAbove - (fc - fzL) * invDBelow) * invDzK;
          df[c] = Dt * (d2x + d2y + d2z) * dtSub;
        }
      }
    }
    for (let c = 0; c < field.length; c++) field[c] += df[c];
  }
}
