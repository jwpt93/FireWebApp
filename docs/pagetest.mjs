/**
 * Static checks for the live-solver page — node docs/pagetest.mjs
 *
 * I cannot open a browser from here, so this covers the failures that ARE
 * catchable without one, which are most of the ones that actually happen:
 * a getElementById id that does not exist in the markup, a frame packed at the
 * wrong stride, a palette that is not monotonic. It does not prove the page
 * renders — only that it is not broken in the boring ways.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { visibleRows, packFrame, fireColour, buildPalette } from './js/solverframe.js';
import { CFG } from './js/solverconfig.js';
import { runSpread3D } from './js/physics/solver.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const results = [];
const check = (name, ok, detail) => results.push({ name, ok, detail });

const html = readFileSync(join(HERE, 'solver.html'), 'utf8');
const page = readFileSync(join(HERE, 'js', 'solverpage.js'), 'utf8')
  + readFileSync(join(HERE, 'js', 'solverconfig.js'), 'utf8');

// 1. Every $('...') the script touches must exist in the markup. This is the
//    single most common way a page like this silently half-works.
{
  const ids = [...page.matchAll(/\$\('([^']+)'\)/g)].map((m) => m[1]);
  const uniq = [...new Set(ids)];
  const missing = uniq.filter((id) => !html.includes(`id="${id}"`));
  check('page.ids', missing.length === 0,
    missing.length === 0
      ? `all ${uniq.length} referenced ids present in solver.html`
      : `missing from markup: ${missing.join(', ')}`);
}

// 2. The page ships ONE measured configuration. Assert the selector and its
//    presets are really gone -- a leftover $('preset') would throw on load,
//    and the previous version of this check went vacuous when the dropdown
//    was removed (0 options, still "passing").
{
  const problems = [];
  if (/PRESETS/.test(page)) problems.push('PRESETS still referenced');
  if (/\$\('preset'\)/.test(page)) problems.push("$('preset') still referenced");
  if (/<select/.test(html)) problems.push('<select> still in markup');
  if (!/const CFG = \{/.test(page)) problems.push('CFG not defined');
  for (const k of ['levelSetPassive: true', 'nSub: 1', 'projectionCgRtol: 1.0e-4',
                   'empiricalRosEnable: true']) {
    if (!page.includes(k)) problems.push(`CFG missing ${k}`);
  }
  check('page.singleConfig', problems.length === 0,
    problems.length === 0
      ? 'one measured config; selector and presets removed; all 4 tuned '
        + 'settings present'
      : problems.join('; '));
}

// 3. The worker must not import anything that does not exist.
{
  const w = readFileSync(join(HERE, 'js', 'solverworker.js'), 'utf8');
  const imports = [...w.matchAll(/from '([^']+)'/g)].map((m) => m[1]);
  let ok = true, bad = '';
  for (const spec of imports) {
    try { readFileSync(join(HERE, 'js', spec.replace('./', ''))); }
    catch { ok = false; bad = spec; }
  }
  check('worker.imports', ok, ok ? `${imports.length} imports resolve` : `missing ${bad}`);
}

// 4. visibleRows on a real stretched axis.
{
  const dz = Float64Array.from([0.025, 0.025, 0.025, 0.025, 0.03, 0.036, 0.043,
                                0.052, 0.062, 0.075, 0.09, 0.108, 0.13, 0.156]);
  const zFace = new Float64Array(dz.length + 1);
  for (let k = 0; k < dz.length; k++) zFace[k + 1] = zFace[k] + dz[k];
  const k12 = visibleRows(zFace, dz.length, 1.2);
  const k0 = visibleRows(zFace, dz.length, 0.0);
  check('frame.visibleRows',
    k12 > 4 && k12 <= dz.length && k0 === 1,
    `zTop=1.2 -> ${k12} of ${dz.length} rows (z=${zFace[k12].toFixed(3)} m); ` +
    `zTop=0 clamps to ${k0}`);
}

// 5. packFrame must read the right cells. Fill T_g with its own flat index so
//    a stride error shows up as a wrong number rather than plausible noise.
{
  const nx = 5, ny = 3, nz = 4, kVis = 2, nZBed = 2, jSlice = 1;
  const Tg = new Float64Array(nz * ny * nx);
  const Ts = new Float64Array(nz * ny * nx);
  for (let i = 0; i < Tg.length; i++) { Tg[i] = i; Ts[i] = -i; }
  const f = packFrame(Tg, Ts, { nx, ny, nz, kVis, nZBed, jSlice });
  let ok = f.length === kVis * nx + nZBed * nx;
  for (let k = 0; k < kVis && ok; k++) {
    for (let i = 0; i < nx; i++) {
      if (f[k * nx + i] !== (k * ny + jSlice) * nx + i) { ok = false; break; }
    }
  }
  for (let k = 0; k < nZBed && ok; k++) {
    for (let i = 0; i < nx; i++) {
      if (f[kVis * nx + k * nx + i] !== -((k * ny + jSlice) * nx + i)) { ok = false; break; }
    }
  }
  check('frame.packFrame', ok,
    ok ? `${f.length} values, T_g then T_s, correct j-slice and stride`
       : 'stride or slice wrong');
}

// 6. The palette must be monotonically brightening. A temperature ramp that
//    dips is worse than useless — it reads as a cooler region that is not there.
{
  const lut = buildPalette(300, 1500);
  let ok = true, at = -1;
  let prev = -1;
  for (let i = 0; i < 256; i++) {
    const lum = 0.2126 * lut[i * 3] + 0.7152 * lut[i * 3 + 1] + 0.0722 * lut[i * 3 + 2];
    if (lum < prev - 0.5) { ok = false; at = i; break; }
    prev = lum;
  }
  const cold = fireColour(300), hot = fireColour(1500);
  check('frame.palette', ok && cold[0] === 0 && hot[0] > 240,
    ok ? `monotonic luminance across 256 steps; 300 K black, 1500 K ${hot.join(',')}`
       : `luminance dips at step ${at}`);
}

// 7. The shipped config must actually RUN. Cheap 1.5 s smoke -- this is the
//    one check that would catch a config the page offers but the solver
//    rejects (an unsupported deck option throws by design).
{
  let ok = true, detail = '';
  try {
    const r = runSpread3D({ ...CFG, windSpeedMs: 4.0, maxWallTimeS: 1.5 });
    ok = r.steps > 10 && Number.isFinite(r.rosMMin);
    detail = `${r.steps} steps, ${r.grid.nz}x${r.grid.nx} = `
           + `${r.grid.nz * r.grid.ny * r.grid.nx} cells, `
           + `ROS_Ts ${Number.isFinite(r.rosTsMMin) ? r.rosTsMMin.toFixed(2) : 'n/a'} m/min`;
  } catch (e) { ok = false; detail = e.message; }
  check('page.configRuns', ok, detail);
}

const w = Math.max(...results.map((r) => r.name.length));
for (const r of results) {
  console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name.padEnd(w)}  ${r.detail}`);
}
const nPass = results.filter((r) => r.ok).length;
console.log(`\n${nPass}/${results.length} page checks passed`);
process.exit(nPass === results.length ? 0 : 1);
