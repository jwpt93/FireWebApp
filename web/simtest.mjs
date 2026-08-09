/**
 * Behavioural smoke test for the Mode A simulation — node web/simtest.mjs
 *
 * The golden vectors in test.mjs pin the KERNELS to the Python reference.
 * This file checks the thing built on top of them actually behaves like a
 * fire: that the front advances at the rate the Cheney law specifies, that
 * wind makes it elongate, and that the whole loop is deterministic.
 *
 * The load-bearing check is `head rate matches Cheney`. Mode A's entire claim
 * is that the spread rate is correct by construction; if the level set does
 * not actually propagate the head at the published ROS, that claim is empty
 * no matter how well the kernels match.
 *
 * Exits non-zero on failure.
 */
import { FireSim, DOMAIN } from './js/sim.js';
import { rosFromU2 } from './js/cheney.js';
import { FUELS, fuelLoad, residenceTime_s } from './js/fuels.js';

let failures = 0;
const results = [];

function check(name, ok, detail) {
  results.push({ name, ok, detail });
  if (!ok) failures++;
}

/** Farthest extent of the burnt region along a unit direction, from origin. */
function extentAlong(sim, ox, oy, dx_, dy_) {
  const { nx, ny } = sim;
  const { dx } = DOMAIN;
  const { phi } = sim.ls;
  let best = -Infinity;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      if (phi[j * nx + i] >= 0) continue;
      const x = (i + 0.5) * dx - ox;
      const y = (j + 0.5) * dx - oy;
      const proj = x * dx_ + y * dy_;
      if (proj > best) best = proj;
    }
  }
  return best;
}

// ── 1. head rate matches the Cheney law ────────────────────────────────────
// Advance a wind-driven fire and measure how fast the downwind extent grows.
//
// TOLERANCE.  A PLANAR front reproduces the Cheney rate exactly (see check 1b)
// -- the scheme is right.  A CURVED front from a point ignition lags, because
// first-order Godunov upwind carries an O(dx x curvature) error.  Measured
// convergence for this case, 200x120 m domain, U2=4 M=6%:
//
//     dx = 2.0 m  ->  -10.4%      dx = 0.5 m  ->  -3.4%
//     dx = 1.0 m  ->   -5.9%      dx = 0.25 m ->  -1.8%
//
// Clean first order.  The applet runs dx = 1.0 m because dx = 0.5 costs
// 38 ms/frame against a 16.7 ms budget (cost scales as 1/dx^3), so the
// animated front advances a few percent slower than the quoted rate.
// The READOUTS are evaluated analytically from the law and are exact; only
// the picture carries this error.  Check 1c pins the convergence so a real
// regression cannot hide behind this tolerance.
{
  for (const [fuelKey, U2, mf] of [
    ['natural', 4.0, 0.06],
    ['natural', 6.0, 0.04],
    ['cut', 3.0, 0.10],
  ]) {
    const sim = new FireSim();
    Object.assign(sim.params, {
      fuelKey, U2_m_s: U2, moistureFrac: mf,
      windDirDeg: 0, shape: 'windProjected', ignition: 'point',
    });
    sim.reset();

    const ox = DOMAIN.Lx * 0.28;
    const oy = DOMAIN.Ly * 0.5;

    sim.advance(20);                       // let the front shake off the seed
    const t0 = sim.t;
    const e0 = extentAlong(sim, ox, oy, 1, 0);
    sim.advance(60);
    const measured = (extentAlong(sim, ox, oy, 1, 0) - e0) / (sim.t - t0);

    const expected = rosFromU2(U2, mf, FUELS[fuelKey].a_ch);
    const err = (measured - expected) / expected;
    // Lags, never leads: a scheme that ran FAST would be a real bug.
    check(`head rate ${fuelKey} U₂=${U2} M=${(mf * 100).toFixed(0)}%`,
          err < 0.02 && err > -0.14,
          `measured ${(measured * 60).toFixed(2)} vs Cheney ${(expected * 60).toFixed(2)} m/min ` +
          `(${(err * 100).toFixed(1)}%)`);
  }
}

// ── 1b. a PLANAR front is exact ────────────────────────────────────────────
// The sharp check on the scheme itself, with curvature removed. If this ever
// drifts, the discretisation is wrong -- not merely coarse.
{
  const { LevelSet2D } = await import('./js/levelset.js');
  const { windProjectedSpeed } = await import('./js/cheney.js');
  const dx = 1.0, nx = 200, ny = 60;
  const V = rosFromU2(4, 0.06, FUELS.natural.a_ch);
  const ls = new LevelSet2D({ nx, ny, dx, dy: dx });
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) ls.phi[j * nx + i] = (i + 0.5) * dx - 20;
  }
  const fn = windProjectedSpeed(4, 0.06, FUELS.natural.a_ch, 0);
  const vn = new Float64Array(nx * ny);
  const dt = ls.maxDt(V);
  const frontX = () => {
    const j = Math.floor(ny / 2);
    for (let i = 1; i < nx; i++) {
      const a = ls.phi[j * nx + i - 1], b = ls.phi[j * nx + i];
      if (a < 0 && b >= 0) return (i - 1 + 0.5 + a / (a - b)) * dx;
    }
    return NaN;
  };
  const x0 = frontX();
  let t = 0, steps = 0;
  while (t < 60) {
    ls.fillNormalSpeed(fn, vn); ls.step(dt, vn); t += dt;
    if (++steps % 10 === 0) ls.reinitialize();
  }
  const rate = (frontX() - x0) / t;
  const err = Math.abs(rate - V) / V;
  check('planar front is exact', err < 1e-3,
        `${(rate * 60).toFixed(3)} vs ${(V * 60).toFixed(3)} m/min (${(err * 100).toFixed(3)}%)`);
}

// ── 1c. the curved-front error is first-order in dx ────────────────────────
// Pins the tolerance above to convergence rather than to a magic number: a
// genuine regression would break the ordering even if it stayed under 14%.
{
  const { LevelSet2D } = await import('./js/levelset.js');
  const { windProjectedSpeed } = await import('./js/cheney.js');
  const V = rosFromU2(4, 0.06, FUELS.natural.a_ch);
  const errs = [];
  for (const dx of [2.0, 1.0, 0.5]) {
    const nx = Math.round(200 / dx), ny = Math.round(120 / dx);
    const ls = new LevelSet2D({ nx, ny, dx, dy: dx });
    ls.seedCircle(40, 60, 3);
    const fn = windProjectedSpeed(4, 0.06, FUELS.natural.a_ch, 0);
    const vn = new Float64Array(nx * ny);
    const dt = ls.maxDt(V);
    const headX = () => {
      const j = Math.floor(ny / 2);
      let best = NaN;
      for (let i = 1; i < nx; i++) {
        const a = ls.phi[j * nx + i - 1], b = ls.phi[j * nx + i];
        if (a < 0 && b >= 0) best = (i - 1 + 0.5 + a / (a - b)) * dx;
      }
      return best;
    };
    let t = 0, steps = 0;
    while (t < 30) {
      ls.fillNormalSpeed(fn, vn); ls.step(dt, vn); t += dt;
      if (++steps % 10 === 0) ls.reinitialize();
    }
    const x0 = headX(), t0 = t;
    while (t < 150) {
      ls.fillNormalSpeed(fn, vn); ls.step(dt, vn); t += dt;
      if (++steps % 10 === 0) ls.reinitialize();
    }
    errs.push(Math.abs(((headX() - x0) / (t - t0)) / V - 1));
  }
  const shrinking = errs[0] > errs[1] && errs[1] > errs[2];
  const order = Math.log2(errs[0] / errs[2]) / 2;   // per halving of dx
  check('curved-front error is first-order', shrinking && order > 0.6,
        `dx 2.0/1.0/0.5 -> ${errs.map((e) => (e * 100).toFixed(1) + '%').join(' / ')}, ` +
        `observed order ${order.toFixed(2)}`);
}

// ── 2. isotropic mode stays circular ───────────────────────────────────────
{
  const sim = new FireSim();
  Object.assign(sim.params, {
    fuelKey: 'natural', U2_m_s: 4, moistureFrac: 0.06,
    shape: 'isotropic', ignition: 'point', windDirDeg: 0,
  });
  sim.reset();
  sim.advance(60);
  const ox = DOMAIN.Lx * 0.28, oy = DOMAIN.Ly * 0.5;
  const down = extentAlong(sim, ox, oy, 1, 0);
  const cross = extentAlong(sim, ox, oy, 0, 1);
  const ratio = down / cross;
  check('isotropic stays circular', Math.abs(ratio - 1) < 0.06,
        `downwind/crosswind extent ${ratio.toFixed(3)} (want ~1)`);
}

// ── 3. wind-projected elongates downwind ───────────────────────────────────
{
  const sim = new FireSim();
  Object.assign(sim.params, {
    fuelKey: 'natural', U2_m_s: 6, moistureFrac: 0.05,
    shape: 'windProjected', ignition: 'point', windDirDeg: 0,
  });
  sim.reset();
  sim.advance(60);
  const ox = DOMAIN.Lx * 0.28, oy = DOMAIN.Ly * 0.5;
  const down = extentAlong(sim, ox, oy, 1, 0);
  const back = extentAlong(sim, ox, oy, -1, 0);
  const cross = extentAlong(sim, ox, oy, 0, 1);
  check('wind elongates the fire', down > 2 * cross && cross > back,
        `head ${down.toFixed(1)} m, flank ${cross.toFixed(1)} m, back ${back.toFixed(1)} m`);
}

// ── 4. wind direction actually steers ──────────────────────────────────────
{
  const sim = new FireSim();
  Object.assign(sim.params, {
    fuelKey: 'natural', U2_m_s: 6, moistureFrac: 0.05,
    shape: 'windProjected', ignition: 'point', windDirDeg: 90,
  });
  sim.reset();
  sim.advance(45);
  const ox = DOMAIN.Lx * 0.28, oy = DOMAIN.Ly * 0.5;
  const alongY = extentAlong(sim, ox, oy, 0, 1);
  const alongX = extentAlong(sim, ox, oy, 1, 0);
  check('wind direction steers the front', alongY > 2 * alongX,
        `+y extent ${alongY.toFixed(1)} m vs +x ${alongX.toFixed(1)} m at 90°`);
}

// ── 5. burnt area grows monotonically ──────────────────────────────────────
{
  const sim = new FireSim();
  sim.reset();
  let prev = sim.burntArea_ha;
  let ok = true;
  for (let n = 0; n < 12; n++) {
    sim.advance(5);
    const a = sim.burntArea_ha;
    if (a < prev - 1e-12) { ok = false; break; }
    prev = a;
  }
  check('burnt area grows monotonically', ok && prev > 0, `final ${prev.toFixed(3)} ha`);
}

// ── 6. determinism (CLAUDE.md Rule #17, applied to the whole loop) ─────────
{
  const run = () => {
    const s = new FireSim();
    Object.assign(s.params, {
      fuelKey: 'cut', U2_m_s: 5, moistureFrac: 0.07,
      shape: 'windProjected', ignition: 'line', windDirDeg: 30,
    });
    s.reset();
    for (let n = 0; n < 20; n++) s.advance(3);
    return s.ls.phi;
  };
  const a = run(), b = run();
  let diff = 0;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) diff++;
  check('whole-loop determinism', diff === 0,
        diff === 0 ? 'two runs bit-identical' : `${diff} cells differ`);
}

// ── 7. derived quantities are sane ─────────────────────────────────────────
{
  for (const key of Object.keys(FUELS)) {
    const w0 = fuelLoad(FUELS[key]);
    check(`${key} fuel load plausible`, w0 > 0.15 && w0 < 1.0,
          `${w0.toFixed(3)} kg/m² (grass beds run ~0.2–0.6)`);
    const tau = residenceTime_s(FUELS[key]);
    check(`${key} residence time plausible`, tau > 5 && tau < 60,
          `${tau.toFixed(1)} s`);
  }

  const sim = new FireSim();
  Object.assign(sim.params, { fuelKey: 'natural', U2_m_s: 5, moistureFrac: 0.05 });
  // Byram at ~0.4 kg/m² and a fast grass fire lands in the thousands of kW/m
  // with flame lengths of a few metres — the range Cheney photographed.
  check('Byram intensity in range',
        sim.intensity_kW_m > 500 && sim.intensity_kW_m < 20000,
        `${Math.round(sim.intensity_kW_m)} kW/m`);
  check('flame length in range',
        sim.flameLength_m > 0.5 && sim.flameLength_m < 8,
        `${sim.flameLength_m.toFixed(2)} m`);
}

// ── report ─────────────────────────────────────────────────────────────────
const w = Math.max(...results.map((r) => r.name.length));
for (const r of results) {
  console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name.padEnd(w)}  ${r.detail}`);
}
console.log(`\n${results.length - failures}/${results.length} checks passed`);
process.exit(failures ? 1 : 0);
