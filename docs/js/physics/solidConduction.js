/**
 * Vertical conduction in the solid — JS port of
 * model_outdoor/physics_3d/solid_conduction_3d.py ::
 * step_solid_conduction_vertical.
 *
 * Explicit-Euler conduction up and down the fuel stack:
 *
 *   flux_up = k_s * alpha_face_up * (T_s[k+1] - T_s[k]) / d_face_above[k]
 *   flux_dn = k_s * alpha_face_dn * (T_s[k-1] - T_s[k]) / d_face_below[k]
 *   dT_s/dt = (flux_up + flux_dn) / (dz[k] * rho_s * cp_s * alpha_s[k])
 *
 * The face conductance uses the HARMONIC mean of the two adjacent solid
 * fractions, which is what makes gas cells (alpha_s = 0) decouple from the
 * stack automatically: the harmonic mean with zero is zero, so the first cell
 * above the bed becomes a no-flux boundary without a special case.
 *
 * Double-buffered — every cell reads the OLD T_s and writes a separate array
 * that is copied back at the end. That is the pattern CLAUDE.md Rule #17
 * requires, and it is why the result cannot depend on sweep order.
 *
 * Stability: explicit Euler needs Fourier number k*dt/(rho*cp*dz^2) <= 0.5.
 * For k=0.2, rho=500, cp=1300, dz=0.0925 that allows dt ~ 144 s, orders of
 * magnitude above the CFL-limited outer step, so it is never the constraint.
 */

/**
 * @param {Float64Array} Ts        (Nz*Ny*Nx) [K] updated in place
 * @param {Float64Array} alphaS    solid volume fraction
 * @param {Float64Array} dzArr     (Nz) cell thickness
 * @param {Float64Array} dFaceAbove (Nz)
 * @param {Float64Array} dFaceBelow (Nz)
 * @param {number} kSolid   [W/m/K]
 * @param {number} rhoSolid [kg/m^3]
 * @param {number} cpSolid  [J/kg/K]
 * @param {number} dt
 * @param {object} opts {nx, ny, nz}
 */
export function stepSolidConductionVertical(
  Ts, alphaS, dzArr, dFaceAbove, dFaceBelow,
  kSolid, rhoSolid, cpSolid, dt, { nx, ny, nz } = {},
) {
  const TsNew = Float64Array.from(Ts);
  const invRhoCp = 1.0 / (rhoSolid * cpSolid);
  const nxy = ny * nx;

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      for (let k = 0; k < nz; k++) {
        const c = (k * ny + j) * nx + i;
        const aC = alphaS[c];
        if (aC <= 0.0) continue;

        let fluxUp = 0.0;
        if (k < nz - 1) {
          const aA = alphaS[c + nxy];
          if (aA > 0.0) {
            const aEffUp = (2.0 * aC * aA) / (aC + aA);
            fluxUp = (kSolid * aEffUp * (Ts[c + nxy] - Ts[c])) / dFaceAbove[k];
          }
        }

        let fluxDn = 0.0;
        if (k > 0) {
          const aB = alphaS[c - nxy];
          if (aB > 0.0) {
            const aEffDn = (2.0 * aC * aB) / (aC + aB);
            fluxDn = (kSolid * aEffDn * (Ts[c - nxy] - Ts[c])) / dFaceBelow[k];
          }
        }

        const src = (fluxUp + fluxDn) / dzArr[k];
        TsNew[c] = Ts[c] + (dt * src * invRhoCp) / aC;
      }
    }
  }

  Ts.set(TsNew);
}
