/**
 * Porous-bed drag — JS port of
 * model_outdoor/physics_3d/drag_3d.py :: step_drag_force.
 *
 * Ergun two-term volumetric drag from the fuel bed on the gas:
 *
 *     F = -( K_visc + K_quad ) * u
 *     K_visc = 50 * mu * sigma^2 * alpha^2 / (1 - alpha)^3     (Darcy/Ergun)
 *     K_quad = 0.5 * C_D * sigma * alpha * rho * |u|           (Forchheimer)
 *
 * The viscous term dominates in still air, the quadratic term once the wind
 * is up — which is why a grass bed both slows the mean flow and damps the
 * buoyant perturbations that would otherwise run away.
 *
 * Cells with no fuel (alpha_s <= 0) get exactly zero, not a small number:
 * the early return matters because (1 - alpha)^3 would otherwise divide by
 * zero as alpha approaches 1.
 *
 * Output arrays are OVERWRITTEN, not accumulated.
 */

/** Ergun viscous constant [-]. */
export const ERGUN_VISC_K = 50.0;

/** Gas dynamic viscosity used by the drag closure [Pa.s]. */
export const MU_GAS = 3.0e-5;

/** Literature default form-drag coefficient (Wilson-Shaw). */
export const C_D_DEFAULT = 0.30;

/**
 * @param {Float64Array} u
 * @param {Float64Array} v
 * @param {Float64Array} w
 * @param {Float64Array} rho      [kg/m^3]
 * @param {Float64Array} alphaS   solid volume fraction [-]
 * @param {number} sigmaSav       fuel surface-area-to-volume [1/m]
 * @param {Float64Array} FxOut    overwritten [N/m^3]
 * @param {Float64Array} FyOut
 * @param {Float64Array} FzOut
 * @param {number} CD             form-drag coefficient
 */
export function stepDragForce(u, v, w, rho, alphaS, sigmaSav,
                              FxOut, FyOut, FzOut, CD) {
  const sigma2 = sigmaSav * sigmaSav;
  for (let c = 0; c < u.length; c++) {
    const a = alphaS[c];
    if (a <= 0.0) {
      FxOut[c] = 0.0;
      FyOut[c] = 0.0;
      FzOut[c] = 0.0;
      continue;
    }
    const ui = u[c];
    const vi = v[c];
    const wi = w[c];
    const speed = Math.sqrt(ui * ui + vi * vi + wi * wi);
    const oneMa = 1.0 - a;
    const Kvisc = (ERGUN_VISC_K * MU_GAS * sigma2 * a * a)
                  / (oneMa * oneMa * oneMa);
    const Kquad = CD * sigmaSav * a * 0.5 * rho[c] * speed;
    const coeff = -(Kvisc + Kquad);
    FxOut[c] = coeff * ui;
    FyOut[c] = coeff * vi;
    FzOut[c] = coeff * wi;
  }
}
