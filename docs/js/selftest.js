/**
 * Cross-check the JS port against golden vectors from the Python reference.
 *
 * Shared by two front ends so the check logic cannot drift between them:
 *   docs/test.html  — browser, renders a table
 *   docs/test.mjs   — node, prints to stdout and sets the exit code
 *
 * Regenerate the vectors with scripts/gen_golden_vectors.py.
 */
import { rosFromU2, rosFromU10, firelineIntensity, flameLength, U2_PER_U10 }
  from './cheney.js';
import { LevelSet2D } from './levelset.js';

/**
 * Tolerance for the closed-form relations.
 *
 * Not exact, for two reasons that are both about the reference rather than
 * the port: cheney_eq6_ros_m_per_s() multiplies by 60.0 and divides by 60.0
 * again — a no-op round-trip that loses up to 1 ulp at some inputs — and
 * pow()/exp() may differ by an ulp between libm builds.
 */
const CLOSED_FORM_TOL = 1e-12;

/** Largest relative difference (absolute where the reference is zero). */
function maxRelDiff(got, ref) {
  let worst = 0, at = -1;
  for (let i = 0; i < ref.length; i++) {
    const r = ref[i], g = got[i];
    const d = r === 0 ? Math.abs(g) : Math.abs(g - r) / Math.abs(r);
    if (d > worst) { worst = d; at = i; }
  }
  return { worst, at };
}

/** Count of values that are not bit-identical. */
function bitDiff(got, ref) {
  let n = 0, worst = 0, at = -1;
  for (let i = 0; i < ref.length; i++) {
    if (got[i] !== ref[i]) {
      n++;
      const d = Math.abs(got[i] - ref[i]);
      if (d > worst) { worst = d; at = i; }
    }
  }
  return { n, worst, at };
}

/**
 * Rebuild one golden level-set case with the JS engine.
 * Kept here rather than in the runners so both drive it identically.
 */
function replayLevelSet(c) {
  const ls = new LevelSet2D({ nx: c.nx, ny: c.ny, dx: c.dx, dy: c.dy });
  ls.seedCircle(c.seed.cx, c.seed.cy, c.seed.r);

  if (c.seed.type === 'circle_scaled') {
    for (let k = 0; k < ls.phi.length; k++) ls.phi[k] *= c.seed.scale;
    ls.reinitialize(c.substeps, c.cfl);
    return ls.phi;
  }

  const vn = new Float64Array(c.nx * c.ny);
  if (c.vn.type === 'uniform') {
    vn.fill(c.vn.value);
  } else if (c.vn.type === 'linear_x') {
    for (let j = 0; j < c.ny; j++)
      for (let i = 0; i < c.nx; i++)
        vn[j * c.nx + i] = c.vn.a + c.vn.b * ((i + 0.5) * c.dx);
  } else {
    throw new Error(`unknown vn type ${c.vn.type}`);
  }
  for (let s = 0; s < c.steps; s++) ls.step(c.dt, vn);
  return ls.phi;
}

/**
 * @param {object} golden  parsed docs/data/golden.json
 * @returns {{name: string, detail: string, ok: boolean, extra: string}[]}
 */
export function runChecks(golden) {
  const groups = [];
  const add = (name, detail, ok, extra = '') =>
    groups.push({ name, detail, ok, extra });

  // ── 1. Cheney law, both wind conventions ────────────────────────────────
  for (const [key, fn, windKey] of [
    ['from_u10', rosFromU10, 'U10_m_s'],
    ['from_u2', rosFromU2, 'U2_m_s'],
    ['edge', rosFromU10, 'U10_m_s'],
  ]) {
    const cases = golden.cheney[key];
    const got = cases.map(c => fn(c[windKey], c.moisture_frac, c.a_ch));
    const ref = cases.map(c => c.ros_m_s);
    const { worst, at } = maxRelDiff(got, ref);
    add(`cheney.${key}`,
        `${cases.length} cases · max rel. diff ${worst.toExponential(2)}`,
        worst <= CLOSED_FORM_TOL,
        worst > CLOSED_FORM_TOL && at >= 0
          ? `worst at ${JSON.stringify(cases[at])} → got ${got[at]}` : '');
  }

  // The 0.723 factor must be the ONLY difference between the two entry
  // points — this is the guard against a silent double conversion.
  {
    const a = rosFromU10(4.0, 0.04, 0.406);
    const b = rosFromU2(U2_PER_U10 * 4.0, 0.04, 0.406);
    add('cheney.conventions_consistent',
        `rosFromU10(4) vs rosFromU2(0.723·4) · Δ ${Math.abs(a - b).toExponential(2)}`,
        a === b);
  }

  // ── 2. Byram derived quantities ─────────────────────────────────────────
  {
    const cases = golden.byram.cases;
    const dI = maxRelDiff(
      cases.map(c => firelineIntensity(c.ros_m_s, c.w0_kg_m2, c.H_kJ_kg)),
      cases.map(c => c.I_kW_m));
    const dL = maxRelDiff(
      cases.map(c => flameLength(c.I_kW_m)),
      cases.map(c => c.L_f_m));
    add('byram.fireline_intensity',
        `${cases.length} cases · max rel. diff ${dI.worst.toExponential(2)}`,
        dI.worst <= CLOSED_FORM_TOL);
    add('byram.flame_length',
        `${cases.length} cases · max rel. diff ${dL.worst.toExponential(2)}`,
        dL.worst <= CLOSED_FORM_TOL);
  }

  // ── 3. 2D level set ─────────────────────────────────────────────────────
  // The golden speed fields use only IEEE-exact operations, so these must
  // match bit for bit. Any difference means the discretisation diverged.
  for (const c of golden.levelset.cases) {
    const { n, worst, at } = bitDiff(replayLevelSet(c), c.phi);
    add(`levelset.${c.name}`,
        n === 0
          ? `${c.phi.length} cells · bit-exact`
          : `${n}/${c.phi.length} cells differ · max Δ ${worst.toExponential(2)}`,
        n === 0,
        n > 0 && at >= 0 ? `worst at cell ${at}` : '');
  }

  // ── 4. Determinism — kernel-level analogue of CLAUDE.md Rule #17 ────────
  {
    const build = () => {
      const ls = new LevelSet2D({ nx: 24, ny: 20, dx: 0.5, dy: 0.5 });
      ls.seedCircle(6, 5, 1.5);
      const vn = new Float64Array(24 * 20).fill(1);
      for (let s = 0; s < 12; s++) ls.step(0.1, vn);
      return ls.phi;
    };
    const { n } = bitDiff(build(), build());
    add('levelset.determinism',
        n === 0 ? 'two back-to-back runs bit-identical' : `${n} cells differ`,
        n === 0);
  }

  return groups;
}
