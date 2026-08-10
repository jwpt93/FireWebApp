/**
 * Gas-solid coupling — JS port of
 * model_outdoor/physics_3d/coupling_3d.py :: step_gas_solid_coupling.
 *
 * One step of convective exchange, radiative gain and loss, moisture
 * evaporation and the solid energy balance. Updates T_g, T_s and m_water in
 * place; everything else is read-only.
 *
 * Convective coefficient is Ranz-Marshall on the particle scale:
 *
 *     d_p = 4 / sigma,  Re = rho*|u|*d_p/mu  (floored at 0.1),
 *     Nu  = 2 + 0.6*sqrt(Re)*Pr^(1/3),  h_p = Nu*k_gas/d_p
 *
 * The Re floor is the natural-convection limit — without it a still cell
 * would get zero convective coupling, which is wrong; buoyant exchange
 * continues even with no mean flow.
 *
 * ENERGY ORDER MATTERS AND IS DELIBERATE (Phase 14h). Energy reaching the
 * solid, q_rad + q_conv - q_loss - q_ground, is spent on evaporating water
 * FIRST, capped by the water actually present, and only the residual raises
 * T_s. Before Phase 14h evaporation was radiation-only, so dense opaque beds
 * (cut grass) never dried their downstream cells and the moisture gate locked
 * pyrolysis out indefinitely. Wildland-fuel drying is driven by radiative AND
 * convective heating (Frandsen 1971; Albini 1985).
 *
 * The gas loses q_conv regardless of whether that energy ends up as sensible
 * heat in the solid or as latent heat in the vapour — that keeps the gas
 * energy balance unchanged by the evaporation split.
 *
 * Ground loss applies only at k=0: bottom-face area per unit volume is
 * 1/dz[0], Newton cooling to soil at T_amb.
 */

const MU_GAS = 1.8e-5;      // [Pa.s]
const K_GAS = 0.026;        // [W/m/K]
const PR_GAS = 0.7;         // [-]
const RHO_SOLID = 500.0;    // [kg/m^3] particle density
const CP_SOLID = 1300.0;    // [J/kg/K]
const EPS_SOLID = 0.9;      // [-] surface emissivity
const SIGMA_SB = 5.67e-8;   // [W/m^2/K^4]
const H_GROUND = 5.0;       // [W/m^2/K]
const CP_GAS = 1100.0;      // [J/kg/K]

/**
 * @param {Float64Array} Tg      [K] updated in place
 * @param {Float64Array} Ts      [K] updated in place
 * @param {Float64Array} rho     gas density
 * @param {Float64Array} u
 * @param {Float64Array} v
 * @param {Float64Array} w
 * @param {Float64Array} alphaS  solid volume fraction
 * @param {number} sigmaSav      [1/m]
 * @param {Float64Array} qRadIn  absorbed flux per cell [W/m^2]
 * @param {Float64Array} Qpyro   endothermic sink [W/m^3]
 * @param {Float64Array} Qcomb   combustion heat to gas [W/m^3]
 * @param {Float64Array} mWater  moisture [kg/m^3] updated in place
 * @param {number} Lv            latent heat of vaporisation [J/kg]
 * @param {number} dt
 * @param {Float64Array} dzArr   (Nz)
 * @param {number} Tamb
 * @param {object} opts {nx, ny, nz, qLossEnable?, hConvMult?}
 */
export function stepGasSolidCoupling(
  Tg, Ts, rho, u, v, w, alphaS, sigmaSav, qRadIn, Qpyro, Qcomb,
  mWater, Lv, dt, dzArr, Tamb,
  { nx, ny, nz, qLossEnable = true, hConvMult = 1.0 } = {},
) {
  if (sigmaSav <= 0.0) return;
  const dP = 4.0 / sigmaSav;
  const prCbrt = Math.pow(PR_GAS, 1.0 / 3.0);
  const nxy = ny * nx;

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = (k * ny + j) * nx + i;
        const aS = alphaS[c];
        const TgC = Tg[c];
        const TsC = Ts[c];
        const rhoG = rho[c];

        // Gas-phase combustion heat.
        let TgNew = TgC + (Qcomb[c] / (rhoG * CP_GAS)) * dt;

        if (aS > 0.0) {
          const ui = u[c], vi = v[c], wi = w[c];
          const speed = Math.sqrt(ui * ui + vi * vi + wi * wi);
          let Re = (rhoG * speed * dP) / MU_GAS;
          if (Re < 0.1) Re = 0.1;              // natural-convection floor
          // Mirrors the Python's `Re ** 0.5`.  numba lowers a 0.5 power to
          // sqrt, so Math.pow and Math.sqrt are interchangeable here --
          // verified identical on a probe set.  Contrast the T^4 below,
          // where numba's lowering does NOT match pow().
          const Nu = 2.0 + 0.6 * Math.pow(Re, 0.5) * prCbrt;
          const hP = ((Nu * K_GAS) / dP) * hConvMult;
          const aV = sigmaSav * aS;

          const qConv = hP * aV * (TgNew - TsC);
          // T^4 by repeated multiplication, NOT `** 4`.  The reference is
          // numba-compiled, and numba lowers an integer power to multiplies
          // while CPython and JS both call pow() -- they disagree in the last
          // ulp (…44968 vs …44965 at T_s = 632.0754).  Matching the Python
          // SOURCE is not enough here; the port has to match what numba
          // actually emits.  Caught as a 1-ulp T_s error in one cell.
          const Ts2 = TsC * TsC;
          const Ta2 = Tamb * Tamb;
          const qLoss = qLossEnable
            ? EPS_SOLID * SIGMA_SB * (Ts2 * Ts2 - Ta2 * Ta2) * aV
            : 0.0;
          const qRadVol = qRadIn[c] / dzArr[k];
          const qLossGround = k === 0
            ? (H_GROUND * (TsC - Tamb)) / dzArr[0]
            : 0.0;

          // Evaporation takes its cut before the solid heats.
          const qInSolid = qRadVol + qConv - qLoss - qLossGround;
          const mw = mWater[c];
          let qResidual;
          if (mw > 0.0 && Lv > 0.0 && dt > 0.0 && qInSolid > 0.0) {
            const qEvapMax = (mw * Lv) / dt;   // capped by water present
            const qEvapUse = qInSolid < qEvapMax ? qInSolid : qEvapMax;
            const dmEvap = (qEvapUse * dt) / Lv;
            let newMw = mw - dmEvap;
            if (newMw < 0.0) newMw = 0.0;
            mWater[c] = newMw;
            qResidual = qInSolid - qEvapUse;
          } else {
            qResidual = qInSolid;
          }

          const Cs = RHO_SOLID * CP_SOLID * aS;
          const TsNew = Cs > 0.0
            ? TsC + ((qResidual - Qpyro[c]) / Cs) * dt
            : TsC;

          // Gas gives up q_conv either way.
          TgNew -= (qConv / (rhoG * CP_GAS)) * dt;
          Ts[c] = TsNew;
        }

        Tg[c] = TgNew;
      }
    }
  }
}
