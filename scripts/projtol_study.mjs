/**
 * Projection-tolerance study — node scripts/projtol_study.mjs
 *
 * Does loosening the projection's divergence tolerance change the answer on a
 * case where mass conservation is actually being stressed?
 *
 * WHY THE EARLIER TEST WAS NOT ENOUGH. The first comparison ran with
 * `levelSetPassive: true`, which pins v_n to zero — the front barely moves, the
 * plume is weak, and the projection has little to do. It showed no difference,
 * which is close to uninformative. This one runs a REAL propagating front
 * across three wind regimes, integrated four times longer.
 *
 * WIND RANGE: U = 4, 6, 8 m/s, and NOT below.
 *
 * Below U_10 = 2.5 the production configuration is the Phase 19/20 empirical
 * hybrid, ramping to fully-resolved at 3.5. The bare resolved solver in that
 * regime is running where Phase 18 established it cannot propagate — a
 * closure-class limit, not a tuning problem: spread at low wind is mediated by
 * intermittent flame contact that a RANS average removes (Finney 2015).
 *
 * A first version of this study included U = 1. It advanced 0.18 m in 12 s and
 * peaked at 477 K, and reported 0.0% deviation across all three tolerances —
 * which reads as a pass and is nothing of the kind. Nothing was happening, so
 * nothing could change. Worse, there is no plume there to stress mass
 * conservation, which is the entire quantity under test. Testing a numerical
 * tolerance in a regime the model is not valid in cannot support a conclusion
 * either way.
 *
 * ACCEPTANCE BANDS, FIXED BEFORE RUNNING (CLAUDE.md Rule #3 — criteria are
 * written down before results are examined and may not be widened afterwards):
 *
 *   ROS        within 10% of the 1e-4/1e-3 reference, at every wind.
 *              10%, not 5%, to stay consistent with the criterion this project
 *              uses elsewhere.
 *   T_g max    within 5%. Peak flame temperature sets forward radiation, which
 *              is what drives the front, so it is the leading indicator of a
 *              change that has not yet reached ROS.
 *   |u| max    within 10%. The velocity field is what the projection actually
 *              corrects, so a mass-conservation failure shows here first.
 *   propagation  a tolerance that fails to propagate a case the reference
 *              propagates is an automatic FAIL regardless of the numbers.
 *
 * A tolerance passes only if it passes at EVERY wind. One failure fails it.
 *
 * NOT a validation against experiment — there is no EXP here. It is a
 * self-consistency check of a numerical parameter against the tightest setting.
 */
import { runSpread3D } from '../docs/js/physics/solver.js';

const BANDS = { ros: 0.10, tg: 0.05, u: 0.10 };

const TOLS = [
  { name: '1e-4/1e-3', rtol: 1.0e-4, dtol: 1.0e-3, ref: true },
  { name: '1e-3/1e-2', rtol: 1.0e-3, dtol: 1.0e-2 },
  { name: '1e-2/1e-1', rtol: 1.0e-2, dtol: 1.0e-1 },
];
const WINDS = [4.0, 6.0, 8.0];

const BASE = {
  Ly: 0.10, dy: 0.10, Lz: 8.0, nZBed: 4, hBed: 0.10, rhoB: 1.07,
  sigmaSav: 2000.0, canopyCd: 0.30, initialMoistureFrac: 0.04,
  Lx: 12.0, dx: 0.10, bedXStart: 1.0, bedXEnd: 9.0,
  wallBlN: 1, wallBlFirstDz: 0.025, wallBlGrowth: 1.0,
  atmGrowth: 1.20, atmMaxDz: 1.0,
  cflFactor: 0.40, minDtS: 1.0e-4,
  ignitionDurationS: 3.0, ignitionQMult: 3.0, ignitionWidthMult: 3.0,
  ignitionTPinEnable: false,
  solidPhaseIgnitionEnable: true, solidPhaseIgnitionTsK: 1000.0,
  lagrangianBedNPerCell: 4, lagrangianBedDryingMode: 'combined',
  lagrangianBedHConv: 250.0, lagrangianBedViewFactorGeometric: true,
  domSubcycleEvery: 5, wallFunction: false,
  nSub: 1,
  // THE POINT OF THIS STUDY: a real front, not a pinned one.
  levelSetPassive: false,
  maxWallTimeS: 12.0,
};

function run(wind, tol) {
  let divMax = 0, itSum = 0, itN = 0;
  const t0 = Date.now();
  const r = runSpread3D({
    ...BASE, windSpeedMs: wind,
    projectionCgRtol: tol.rtol, projDivTol: tol.dtol, profile: true,
  }, (i) => {
    if (i.projDivMax > divMax) divMax = i.projDivMax;
    itSum += i.projNIter; itN++;
    return true;
  });
  let tg = 0, umax = 0;
  for (const v of r.state.T_g) if (v > tg) tg = v;
  for (const v of r.state.u) { const a = Math.abs(v); if (a > umax) umax = a; }
  const tot = Object.values(r.timings).reduce((a, b) => a + b, 0);
  return {
    ros: r.rosMMin, tg, umax, divMax, steps: r.steps, t: r.t,
    projIt: itSum / itN, msStep: tot / r.steps,
    projMs: r.timings.projection / r.steps,
    wall: (Date.now() - t0) / 1000,
    advanced: r.frontX.length ? r.frontX[r.frontX.length - 1] - r.frontX[0] : 0,
  };
}

const rel = (a, b) => (b === 0 ? Math.abs(a) : Math.abs(a - b) / Math.abs(b));

console.log('Projection-tolerance study — real propagating front');
console.log(`bands: ROS ${BANDS.ros * 100}%  T_g ${BANDS.tg * 100}%  `
          + `|u| ${BANDS.u * 100}%   (fixed before running)\n`);

const ref = {};
const verdict = {};
for (const wind of WINDS) {
  console.log(`── U = ${wind} m/s ${'─'.repeat(52)}`);
  console.log('  tol          ROS m/min   dev      T_g    |u|max   divMax   '
            + 'projIt  ms/stp  proj  steps  advance');
  for (const tol of TOLS) {
    const r = run(wind, tol);
    if (tol.ref) ref[wind] = r;
    const R = ref[wind];
    const dRos = rel(r.ros, R.ros), dTg = rel(r.tg, R.tg), dU = rel(r.umax, R.umax);
    const propagated = r.advanced > 1e-6;
    const ok = tol.ref || (dRos <= BANDS.ros && dTg <= BANDS.tg && dU <= BANDS.u
                           && propagated === (R.advanced > 1e-6));
    if (!tol.ref) verdict[tol.name] = (verdict[tol.name] !== false) && ok;
    console.log(
      `  ${tol.name.padEnd(11)} ${r.ros.toFixed(3).padStart(8)}  `
      + `${tol.ref ? '   ref' : (dRos * 100).toFixed(1).padStart(5) + '%'}  `
      + `${r.tg.toFixed(0).padStart(5)}K ${r.umax.toFixed(2).padStart(7)}  `
      + `${r.divMax.toExponential(1).padStart(7)}  ${r.projIt.toFixed(2).padStart(5)}  `
      + `${r.msStep.toFixed(1).padStart(6)} ${r.projMs.toFixed(1).padStart(5)}  `
      + `${String(r.steps).padStart(5)}  ${r.advanced.toFixed(2).padStart(6)}m`
      + `${tol.ref ? '' : (ok ? '  ok' : '  FAIL')}`);
  }
  console.log('');
}

console.log('── VERDICT ' + '─'.repeat(56));
for (const tol of TOLS.filter((t) => !t.ref)) {
  console.log(`  ${tol.name}: ${verdict[tol.name] ? 'PASS — within band at every wind'
                                                  : 'FAIL — outside band at one or more winds'}`);
}
console.log('\nPROJTOL_STUDY_DONE');
