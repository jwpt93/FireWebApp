/**
 * Whole-solver cross-check — node scripts/integration_test.mjs
 *
 * Runs the JS port on the same case as scripts/run_integration_python.py and
 * compares. This is the counterpart to docs/kerneltest.mjs: the vectors prove
 * each kernel in isolation, this proves the ORDERING. A swapped stage or a
 * missing state update is invisible to the vectors and shows up only here.
 *
 * THE STANDARD IS A BAND, NOT BIT-EXACTNESS, and that is not a compromise --
 * it is the correct standard for this comparison. Per SOLVER_PORT.md section 4,
 * the two codes are two valid solutions of the same model. EDC's extinction
 * gates are DISCONTINUOUS: a 1-ulp difference in omega can cross a `< 0.5`
 * threshold and move a cell's T_g by degrees. Over 217 steps those differences
 * accumulate in the detail while the bulk answer stays put. Demanding
 * agreement in the last digit would be demanding the wrong thing.
 *
 * Bands are set BEFORE looking at the result (CLAUDE.md Rule #3 -- acceptance
 * criteria fixed before running, never widened after seeing the numbers):
 *
 *   ROS            +/- 10%   the headline; what the applet reports
 *   T_g / T_s max  +/- 5%    peak temperatures set forward radiation
 *   field means    +/- 10%   bulk state
 *   step count     +/- 15%   adaptive dt drifts once trajectories differ
 *
 * Exits non-zero on any failure.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runSpread3D } from '../docs/js/physics/solver.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const REF = JSON.parse(
  readFileSync(join(HERE, '..', 'docs', 'data', 'integration_reference.json'), 'utf8'));

const results = [];
const check = (name, ok, detail) => results.push({ name, ok, detail });

/** Relative difference, absolute when the reference is zero. */
const rel = (got, want) =>
  (want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want));

function band(name, got, want, tol, unit = '') {
  const r = rel(got, want);
  check(name, r <= tol,
    `${got.toPrecision(6)}${unit} vs ${want.toPrecision(6)}${unit} ` +
    `(${(r * 100).toFixed(2)}%, band ${(tol * 100).toFixed(0)}%)`);
}

// Same case as scripts/integration_case.txt. Kept in sync by hand -- the deck
// parser is Python-side, and reimplementing it in JS to read one file would be
// more surface area than the duplication costs.
const CFG = {
  Lx: 3.0, Ly: 0.10, Lz: 0.6, dx: 0.10, dy: 0.10, nZBed: 4,
  hBed: 0.10, rhoB: 1.07, sigmaSav: 2000.0, canopyCd: 0.30,
  initialMoistureFrac: 0.04, windSpeedMs: 4.0,
  bedXStart: 0.5, bedXEnd: 2.5,
  atmGrowth: 1.20, atmMaxDz: 1.0,
  cflFactor: 0.40, maxWallTimeS: 0.6, minDtS: 1.0e-4,
  ignitionDurationS: 3.0, ignitionQMult: 3.0, ignitionWidthMult: 3.0,
  ignitionTPinEnable: false,
  solidPhaseIgnitionEnable: true, solidPhaseIgnitionTsK: 1000.0,
  lagrangianBedNPerCell: 4, lagrangianBedDryingMode: 'combined',
  lagrangianBedHConv: 250.0, lagrangianBedViewFactorGeometric: true,
  domSubcycleEvery: 5, levelSetPassive: true,
  wallFunction: false,
  // Pinned to the upstream default. The reference run used N_SUB = 10, so the
  // comparison has to as well -- the applet's choice of 1 is validated
  // separately and does not belong in a cross-language fidelity check.
  nSub: 10,
};

const t0 = Date.now();
const diagT = [], diagTg = [], diagTs = [];
const r = runSpread3D(CFG, (info) => {
  diagT.push(info.t); diagTg.push(info.TgMax); diagTs.push(info.TsMax);
});
const wall = (Date.now() - t0) / 1000;

// Grid first: if the meshes differ, nothing downstream is comparable and every
// other failure would be a symptom of this one.
const g = REF.grid;
check('grid.shape',
  r.grid.nz === g.nz && r.grid.ny === g.ny && r.grid.nx === g.nx
    && r.grid.nZBed === g.n_z_bed,
  `${r.grid.nz}x${r.grid.ny}x${r.grid.nx} n_z_bed=${r.grid.nZBed} vs ` +
  `${g.nz}x${g.ny}x${g.nx} n_z_bed=${g.n_z_bed}`);

band('ros', r.rosMs, REF.ros_m_s, 0.10, ' m/s');
band('steps', r.steps, REF.n_steps, 0.15);

const st = r.state;
const stat = (a) => {
  let mn = Infinity, mx = -Infinity, s = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
    s += a[i];
  }
  return { min: mn, max: mx, mean: s / a.length };
};
for (const [name, arr, tolMax, tolMean] of [
  ['T_g', st.T_g, 0.05, 0.10],
  ['T_s', st.T_s, 0.05, 0.10],
  ['Y_fuel', st.Y_fuel, 0.10, 0.10],
  ['Y_O2', st.Y_O2, 0.05, 0.10],
  ['rho', st.rho, 0.05, 0.10],
  ['u', st.u, 0.10, 0.10],
]) {
  const s = stat(arr);
  band(`${name}.max`, s.max, REF[`${name}_max`], tolMax);
  band(`${name}.mean`, s.mean, REF[`${name}_mean`], tolMean);
}

// Trajectory, not just the endpoint: two runs can land in the same place
// having taken very different routes, and for a spread model the route is the
// physics. Compared at the reference's own sample times by nearest neighbour,
// since the adaptive dt sequences are not identical.
{
  let worstTg = 0, worstTs = 0;
  for (let i = 1; i < REF.diag_t.length; i++) {
    const tRef = REF.diag_t[i];
    let best = 0, bestD = Infinity;
    for (let j = 0; j < diagT.length; j++) {
      const d = Math.abs(diagT[j] - tRef);
      if (d < bestD) { bestD = d; best = j; }
    }
    worstTg = Math.max(worstTg, rel(diagTg[best], REF.diag_Tg_max[i]));
    worstTs = Math.max(worstTs, rel(diagTs[best], REF.diag_Ts_max[i]));
  }
  check('trajectory.Tg_max', worstTg <= 0.10,
    `worst deviation over ${REF.diag_t.length} samples: ${(worstTg * 100).toFixed(2)}% (band 10%)`);
  check('trajectory.Ts_max', worstTs <= 0.10,
    `worst deviation over ${REF.diag_t.length} samples: ${(worstTs * 100).toFixed(2)}% (band 10%)`);
}

const w = Math.max(...results.map((x) => x.name.length));
for (const x of results) {
  console.log(`  ${x.ok ? 'PASS' : 'FAIL'}  ${x.name.padEnd(w)}  ${x.detail}`);
}
const nPass = results.filter((x) => x.ok).length;
console.log(`\n${nPass}/${results.length} integration checks passed ` +
            `(JS ${wall.toFixed(1)}s vs Python ${REF._meta.wall_s}s)`);
process.exit(nPass === results.length ? 0 : 1);
