/**
 * Eddy Dissipation Concept combustion — JS port of
 * model_outdoor/physics_3d/chemistry_closures/edc.py ::
 * step_chemistry_ode_edc.
 *
 * Magnussen (1981) EDC. The cell-averaged reaction rate is
 *
 *     omega = gamma* . rho . min(Y_F, Y_O2/s) / tau*
 *     gamma* = ( C_gamma . (nu.eps/k^2)^(1/4) )^3     fine-structure volume
 *     tau*   = C_tau . (nu/eps)^(1/2)                 fine-structure time
 *
 * The fine structure is assumed to sit at flame temperature — chemistry
 * inside it is fast — so reaction proceeds at the rate fuel mixes INTO the
 * fine structure, regardless of the cell-averaged temperature. That is the
 * standard fix for a coarse-grid closure mismatch: our cells are ~10 cm while
 * a real flame sheet is ~mm, so a cell-averaged Arrhenius rate would be
 * meaninglessly cold.
 *
 * Explicit Euler over n substeps is sufficient — the EDC rate is not stiff in
 * temperature, unlike Arrhenius.
 *
 * THREE SUPPRESSION MECHANISMS, in the order the Python applies them:
 *
 *  1. Y_H2O quench (Phase 17a, always on). Beyler (1992): hydrocarbon flames
 *     extinguish as water vapour rises. omega falls linearly to zero at
 *     Y_H2O = 0.18. This is the direct rate suppression that dilution of
 *     Y_F.Y_O2 alone underestimates at field grass density.
 *  2. Extinction gates (Phase 16, opt-in): inert-fraction suppression,
 *     a cold-flame floor below 1200 K, and a marginal heat-release-rate
 *     threshold below which the flame cannot beat its own losses.
 *  3. Wet-bulb cooling (Drysdale 3.5), triggered by either a substantial
 *     Y_H2O quench or an extinction gate firing. RATE-LIMITED toward 373.15 K
 *     with tau = 0.5 s, deliberately not an instant pin: the Python notes an
 *     instant pin drove u to 1e80 m/s in 0.3 s of sim through the momentum
 *     EoS reacting to the density jump.
 *
 * NUMBA LOWERING — the reference is COMPILED numba, so the port must match
 * what numba emits, not what the Python source says. Verified case by case:
 *
 *     x ** 3     -> x*x*x     (multiplies, not pow)
 *     x ** 0.5   -> sqrt(x)
 *     x ** 0.25  -> pow(x, 0.25)   (NOT sqrt(sqrt(x)) — they differ by an ulp)
 *
 * The 0.25 case cost two wrong turns: a small probe set where pow and
 * sqrt(sqrt) happened to agree, then a 1-ulp error in gamma* that propagated
 * into omega and on into T_g. Probe the values the kernel ACTUALLY sees.
 */

// ── Magnussen fine-structure constants ─────────────────────────────────────
const C_GAMMA_EDC = 2.1377;
const C_TAU_EDC = 0.4083;
const NU_GAS_EDC = 1.5e-5;      // [m^2/s] kinematic viscosity near ambient
const K_TURB_FLOOR_EDC = 1.0e-4;   // [m^2/s^2] avoid divide-by-zero
const EPS_TURB_FLOOR_EDC = 1.0e-6; // [m^2/s^3]
const CP_VAPOR_EDC = 2000.0;    // [J/kg/K]

// ── Suppression / extinction constants ─────────────────────────────────────
const Y_H2O_QUENCH = 0.18;        // Beyler 1992 quench limit
const T_BOIL_WATER_EDC = 373.15;  // [K]
const Y_H2O_SAT_THRESH = 0.10;    // [-]
const EXTINCTION_F_SAFETY = 1.5;  // Linn 2002 heat-loss safety factor
const Y_INERT_CRIT = 0.88;        // [-]
const T_IGNITION_MIN = 1200.0;    // [K] cellulose ignition floor
const Q_RATE_MIN_MARGINAL = 5.0e4; // [W/m^3]
const TAU_WB = 0.5;               // [s] wet-bulb relaxation timescale
const T_G_CAP = 2400.0;           // [K] adiabatic-ish cap

/** Biomass defaults, from chemistry_closures/_constants.py. */
/** Arrhenius pre-exponential and activation energy for the volatile-oxidation
 *  limb. Source of truth: chemistry_closures/_constants.py. */
export const A_COMB = 1.0e9;        // [1/s]
export const E_COMB = 84000.0;      // [J/mol]
export const R_GAS_EDC = 8.314;     // [J/mol/K]

export const S_STOICH = 1.3;

// ── FDS critical-flame-temperature extinction (Tech Guide 5.3.1-5.3.2) ──
// The mixing-controlled model burns fuel and oxygen wherever they coexist,
// "regardless of the local temperature, reactant concentration, or strain
// rate" (FDS Tech Guide 5.3). FDS suppresses reaction in a cell when burning
// the locally available limiting reactant cannot raise the mixture to a
// critical flame temperature. Theory: Beyler, SFPE Handbook.
//
//   dT_max = dH_O2 * Y_O2,avail / cp        (dH_O2 = HOC / s_stoich)
//   extinguish if  T_cell + dT_max < CFT
//
// dH_O2 = 17.0e6 / 1.3 = 13.1 MJ/kg O2, which is FDS's universal value.
export const CFT_DEFAULT_K = 1600.0;   // [K] FDS default critical flame temp
export const HOC_J = 17_000_000.0;   // [J/kg]

/**
 * Advance Y_fuel, Y_O2 and T_g by one chemistry step, in place, and write the
 * dt-averaged reaction rate into omegaIntOut.
 *
 * @param {Float64Array} rho
 * @param {Float64Array} Tg        [K] in place
 * @param {Float64Array} Yfuel     [-] in place
 * @param {Float64Array} YO2       [-] in place
 * @param {Float64Array} kTurb     TKE [m^2/s^2]
 * @param {Float64Array} epsTurb   dissipation [m^2/s^3]
 * @param {number} chiRad          radiant fraction
 * @param {number} cpG             dry-air baseline cp [J/kg/K]
 * @param {number} dt
 * @param {number} nSubsteps
 * @param {Float64Array} omegaIntOut  [kg/m^3/s] written
 * @param {Float64Array} YH2O      water-vapour mass fraction; zeros for
 *                                 legacy constant-cp behaviour
 * @param {object} [opts] {extinctionEnable, sStoich, hocJ}
 */
export function stepChemistryOdeEdc(
  rho, Tg, Yfuel, YO2, kTurb, epsTurb, chiRad, cpG, dt, nSubsteps,
  omegaIntOut, YH2O,
  { extinctionEnable = false, sStoich = S_STOICH, hocJ = HOC_J,
    laminarFloorSL = 0.0, dxCell = 0.0, chemistryLimb = false, omegaO2 = null, cftK = 0.0, cftFineStructure = true,
    aComb = A_COMB, eComb = E_COMB } = {},
) {
  // THE MISSING CHEMISTRY LIMB (default OFF; reference behaviour untouched).
  //
  // Magnussen EDC computes the rate from turbulence ALONE:
  //     omega = gamma* * rho * Y_lim / tau*
  // Temperature appears nowhere. There is no Arrhenius factor, so the model
  // never asks how fast the chemistry can actually go, or whether it is hot
  // enough to react at all. Its own docstring says so: "no Arrhenius".
  //
  // Two failure modes follow, and both are measured:
  //   - turbulence -> 0 gives rate -> 0 (omega ~ eps^1.25/k^1.5), when the
  //     physical limit is a LAMINAR FLAME at S_L ~ 0.4 m/s. Hence zero flame
  //     cells at U = 0.5.
  //   - flame length 9x short at U = 4 (0.21 m against Byram's 1.77 m).
  //
  // The sibling kernel combustion_3d.step_combustion already has the right
  // structure -- min(omega_chem, omega_mix), with omega_mix -> inf in the
  // laminar limit so the Arrhenius rate takes over. This gives EDC the same
  // limb, using the same constants from chemistry_closures/_constants.py:
  //
  //     omega = min( max(omega_EDC, omega_laminar), omega_chem )
  //
  // The Arrhenius CEILING is not optional. A bare floor would happily "burn"
  // gas at 400 K, since nothing else in the expression knows about temperature.
  //
  // DIAGNOSTIC (default OFF, so the reference behaviour is untouched).
  //
  // Magnussen EDC is a HIGH-Damkohler closure: it assumes chemistry is fast and
  // mixing limits the rate. Its rate scales as eps^1.25 / k^1.5, so as
  // turbulence vanishes the rate vanishes with it -- gamma* ~ eps^0.75 AND
  // 1/tau* ~ eps^0.5, a double penalty. The physical limit as turbulence goes
  // to zero is NOT zero, it is the laminar flame, propagating at S_L.
  //
  // This floors the rate at a flame crossing the cell at S_L:
  //     omega_laminar = rho * Y_lim * S_L / dx
  // dimensionally identical to the Damkohler cap the loop already builds as an
  // UPPER bound (and then discards -- SOLVER_PORT.md 7.7).
  //
  // Purely a diagnostic to test whether a flame forms at low wind. Not a
  // proposed closure: a floor that applies everywhere would also raise the rate
  // at high wind, where the mixing limit is the correct physics.
  const useLaminarFloor = laminarFloorSL > 0.0 && dxCell > 0.0;
  const nSub = Math.max(nSubsteps, 1);
  const h = dt / nSub;
  const hocEff = hocJ * (1.0 - chiRad);

  // DELIBERATELY FUNCTION-SCOPED, NOT PER-CELL.
  //
  // In the Python this flag is only assigned inside the `else` branch, so a
  // cell whose fuel or O2 has run out (Yf <= 1e-9) never resets it and
  // inherits whatever the PREVIOUS CELL left behind.  numba scopes it to the
  // function, not the loop body, so the value leaks from cell to cell.
  //
  // That is a latent bug in the reference -- a cell's wet-bulb cooling can be
  // triggered by its neighbour's moisture rather than its own -- but it is a
  // DETERMINISTIC one: `prange` splits over k while the leak travels along i
  // within a row, so the predecessor is always in the same chunk. Verified
  // identical at 1, 2, 4 and 12 threads.
  //
  // The port's job is to reproduce the reference, not to quietly fix it, so
  // the leak is reproduced here. Reported upstream rather than patched.
  // Observed cost of getting this wrong: 6.8 K of T_g in one cell.
  let h2oQuenchSubstantial = false;

  for (let c = 0; c < rho.length; c++) {
    let Yf = Yfuel[c];
    let yO2 = YO2[c];
    let T = Tg[c];
    const rhoI = rho[c];

    let kT = kTurb[c];
    if (kT < K_TURB_FLOOR_EDC) kT = K_TURB_FLOOR_EDC;
    let eT = epsTurb[c];
    if (eT < EPS_TURB_FLOOR_EDC) eT = EPS_TURB_FLOOR_EDC;

    // k and eps are frozen across the step (operator-split from the k-eps
    // solver), so the fine-structure parameters are constant here.
    let ratio = (NU_GAS_EDC * eT) / (kT * kT);
    if (ratio < 1e-30) ratio = 1e-30;
    // Math.pow, NOT sqrt(sqrt): numba lowers **0.25 through pow, and at the
    // values this kernel actually sees the two disagree in the last ulp
    // (ratio = 5.82e-6 gives ...399098 via pow, ...399099 via sqrt(sqrt)).
    // Verified against numba at the real value -- an earlier four-point probe
    // where they happened to agree sent me the wrong way.
    const gBase = C_GAMMA_EDC * Math.pow(ratio, 0.25);
    let gammaStar = gBase * gBase * gBase;   // numba lowers **3 to multiplies
    if (gammaStar > 1.0) gammaStar = 1.0;
    // sqrt, not pow: numba lowers **0.5 to sqrt.
    const tauStar = C_TAU_EDC * Math.sqrt(NU_GAS_EDC / eT);

    let omegaAcc = 0.0;
    for (let s = 0; s < nSub; s++) {
      let omega;
      if (Yf <= 1e-9 || yO2 <= 1e-9) {
        omega = 0.0;
      } else {
        const yLim = Yf < yO2 / sStoich ? Yf : yO2 / sStoich;
        omega = gammaStar * ((rhoI * yLim) / tauStar);
        if (useLaminarFloor) {
          // Sub-grid laminar flame propagation: a flame crossing the cell
          // at S_L. Dimensionally identical to the Damkohler cap the loop
          // already builds as an UPPER bound and then discards (7.7).
          const omegaLam = (rhoI * yLim * laminarFloorSL) / dxCell;
          if (omegaLam > omega) omega = omegaLam;
        }
        // FDS critical-flame-temperature extinction. Checked BEFORE the O2
        // supply cap so the two are independent.
        if (cftK > 0.0) {
          // Apply CFT to the FINE-STRUCTURE composition, not the cell mean.
          // EDC's premise is that reaction occurs in a small fraction
          // gamma* of the cell where reactants are concentrated; the cell
          // mean is diluted by the unmixed remainder. Testing the cell mean
          // extinguishes everything (measured Y_fuel ~1e-4 is 478x too lean
          // to reach CFT), which is an artifact of averaging a sub-grid
          // flame over dx = 80x the fuel-element scale, not real extinction.
          const gs = gammaStar > 1.0e-6 ? gammaStar : 1.0e-6;
          const YfStar = cftFineStructure ? Yf / gs : Yf;
          const yfCap = YfStar > 1.0 ? 1.0 : YfStar;
          const dHO2 = HOC_J / sStoich;                 // [J/kg O2]
          const yO2Avail = yfCap * sStoich < yO2 ? yfCap * sStoich : yO2;
          if (T + (dHO2 * yO2Avail) / cpG < cftK) omega = 0.0;
        }
        // O2-supply limit: a cell cannot burn faster than oxygen is delivered
        // across its faces. This is what keeps combustion OUT of a packed bed
        // interior and puts the flame above the fuel, where air is available.
        if (omegaO2 !== null && omegaO2[c] < omega) omega = omegaO2[c];
        if (chemistryLimb) {
          // Arrhenius ceiling, same form and constants as step_combustion.
          const kChem = aComb * Math.exp(-eComb / (R_GAS_EDC * T));
          const omegaChem = rhoI * kChem * Yf * yO2;
          if (omegaChem < omega) omega = omegaChem;
        }

        // (1) Beyler 1992 water-vapour quench — always on.
        const yH2Osup = YH2O[c];
        h2oQuenchSubstantial = false;   // reset only on this path, as Python does
        if (yH2Osup > 0.0) {
          let f = 1.0 - yH2Osup / Y_H2O_QUENCH;
          if (f < 0.0) f = 0.0;
          omega *= f;
          // A >=50% reduction routes through the wet-bulb cascade below, so
          // the moisture quench also cools the gas — that closes the
          // quench -> T_g -> sigma T^4 -> preheat -> ROS feedback loop.
          if (f < 0.5) h2oQuenchSubstantial = true;
        }
      }

      // (2) extinction gates, opt-in
      let extinctionFired = false;
      if (extinctionEnable && omega > 0.0) {
        const yInertBound = 1.0 - Yf - yO2;
        if (yInertBound > Y_INERT_CRIT) {
          let supp = (1.0 - yInertBound) / (1.0 - Y_INERT_CRIT);
          if (supp < 0.0) supp = 0.0;
          omega *= supp;
          if (supp < 0.5) extinctionFired = true;
        }
        if (omega > 0.0 && T < T_IGNITION_MIN) {
          omega = 0.0;
          extinctionFired = true;
        }
        if (omega > 0.0) {
          const qRate = omega * hocEff;
          if (qRate < EXTINCTION_F_SAFETY * Q_RATE_MIN_MARGINAL) {
            omega = 0.0;
            extinctionFired = true;
          }
        }
      }

      // (3) wet-bulb cooling, rate-limited (never an instant pin)
      if ((extinctionFired || h2oQuenchSubstantial) && T > T_BOIL_WATER_EDC) {
        const yH2Owb = YH2O[c];
        if (yH2Owb > 0.0) {
          let wbStrength = yH2Owb / Y_H2O_SAT_THRESH;
          if (wbStrength > 1.0) wbStrength = 1.0;
          const relax = 1.0 - Math.exp(-h / TAU_WB);
          T -= (T - T_BOIL_WATER_EDC) * relax * wbStrength;
        }
      }

      // explicit Euler
      const dY = (-omega * h) / rhoI;
      Yf += dY;
      if (Yf < 0.0) Yf = 0.0;
      yO2 += sStoich * dY;
      if (yO2 < 0.0) yO2 = 0.0;

      // Water vapour raises the mixture heat capacity, adding thermal
      // inertia in moisture-laden cells.
      const yH2Ocell = YH2O[c];
      const cpMix = (1.0 - yH2Ocell) * cpG + yH2Ocell * CP_VAPOR_EDC;
      T += (omega * hocEff * h) / (rhoI * cpMix);
      if (T > T_G_CAP) T = T_G_CAP;

      omegaAcc += omega * h;
    }

    Yfuel[c] = Yf;
    YO2[c] = yO2;
    Tg[c] = T;
    omegaIntOut[c] = dt > 0.0 ? omegaAcc / dt : 0.0;
  }
}
