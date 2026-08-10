/**
 * Kernel cross-check — node docs/kerneltest.mjs
 *
 * Replays golden vectors from scripts/gen_kernel_vectors.py through the JS
 * ports in docs/js/physics/ and demands BIT-EXACT agreement.
 *
 * Not a tolerance test. These kernels are explicit stencils: every output
 * cell is written once, from reads of the input buffer, using only IEEE-754
 * exact operations and comparisons. There is no reduction whose order could
 * differ between languages. So the correct answer is identical bits, and a
 * single differing ulp means a transcription error — a swapped index, a
 * missing ghost, a `<` where the Python has `<=`.
 *
 * Exits non-zero on any failure.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { minmod, musclFaceValue, advect3dScalarMuscl } from './js/physics/muscl.js';
import { stepSpeciesTransport } from './js/physics/species.js';
import { stepTentativeVelocity } from './js/physics/momentum.js';
import { stepDragForce } from './js/physics/drag.js';
import { stepSolidConductionVertical } from './js/physics/solidConduction.js';
import { stepGasSolidCoupling } from './js/physics/coupling.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const G = JSON.parse(readFileSync(join(HERE, 'data', 'kernel_vectors.json'), 'utf8'));

const results = [];
const check = (name, ok, detail) => results.push({ name, ok, detail });

/** First index where two arrays differ in bits, or -1. */
function firstDiff(got, ref) {
  for (let i = 0; i < ref.length; i++) if (got[i] !== ref[i]) return i;
  return -1;
}

// ── scalar helpers ─────────────────────────────────────────────────────────
{
  const fns = { minmod, muscl_face_value: musclFaceValue };
  let bad = 0, worst = null;
  for (const h of G.muscl.helpers) {
    const got = fns[h.fn](...h.args);
    if (got !== h.want) {
      bad++;
      worst ??= `${h.fn}(${h.args.join(', ')}) = ${got}, want ${h.want}`;
    }
  }
  check('muscl.scalar_helpers',
        bad === 0,
        bad === 0 ? `${G.muscl.helpers.length} probes bit-exact`
                  : `${bad} wrong — e.g. ${worst}`);
}

// ── advect_3d_scalar_muscl ─────────────────────────────────────────────────
for (const c of G.muscl.cases) {
  const f = (a) => Float64Array.from(a);
  const rhs = f(c.rhs_in);
  advect3dScalarMuscl(
    f(c.phi), f(c.u), f(c.v), f(c.w), c.dx, c.dy,
    f(c.d_face_above), f(c.d_face_below), rhs, c.phi_inlet,
    { nx: c.nx, ny: c.ny, nz: c.nz },
  );
  const at = firstDiff(rhs, c.rhs_out);
  const n = c.rhs_out.length;
  check(
    `muscl.advect.${c.name}`,
    at < 0,
    at < 0
      ? `${c.nz}x${c.ny}x${c.nx} = ${n} cells, bit-exact`
      : `first diff at ${at} (k=${Math.floor(at / (c.ny * c.nx))}, ` +
        `j=${Math.floor(at / c.nx) % c.ny}, i=${at % c.nx}): ` +
        `got ${rhs[at]} ref ${c.rhs_out[at]}`,
  );
}

// ── step_species_transport ─────────────────────────────────────────────────
for (const c of G.species.cases) {
  const f = (a) => Float64Array.from(a);
  const Y = f(c.Y_in);
  stepSpeciesTransport(
    Y, f(c.rho), f(c.u), f(c.v), f(c.w), f(c.S), c.dt, c.dx, c.dy,
    f(c.dz_arr), f(c.d_face_above), f(c.d_face_below), c.D, c.Y_inlet,
    { nx: c.nx, ny: c.ny, nz: c.nz },
  );
  const at = firstDiff(Y, c.Y_out);
  check(
    `species.transport.${c.name}`,
    at < 0,
    at < 0
      ? `${c.nz}x${c.ny}x${c.nx} = ${c.Y_out.length} cells, bit-exact ` +
        `(${c.n_clipped} hit the [0,1] clip)`
      : `first diff at ${at} (k=${Math.floor(at / (c.ny * c.nx))}, ` +
        `j=${Math.floor(at / c.nx) % c.ny}, i=${at % c.nx}): ` +
        `got ${Y[at]} ref ${c.Y_out[at]}`,
  );
}

// ── step_tentative_velocity ────────────────────────────────────────────────
for (const c of G.momentum.cases) {
  const f = (a) => Float64Array.from(a);
  const u = f(c.u_in), v = f(c.v_in), w = f(c.w_in);
  stepTentativeVelocity(
    u, v, w, f(c.rho), f(c.T_g), f(c.Fx), f(c.Fy), f(c.Fz),
    c.dt, c.dx, c.dy, f(c.dz_arr), f(c.d_face_above), f(c.d_face_below),
    c.T_amb, f(c.u_inlet), f(c.v_inlet), f(c.w_inlet),
    { nx: c.nx, ny: c.ny, nz: c.nz },
  );
  const bad = [['u', u, c.u_out], ['v', v, c.v_out], ['w', w, c.w_out]]
    .map(([nm, got, ref]) => [nm, firstDiff(got, ref), got, ref])
    .filter(([, at]) => at >= 0);
  check(
    `momentum.tentative.${c.name}`,
    bad.length === 0,
    bad.length === 0
      ? `${c.nz}x${c.ny}x${c.nx}, u/v/w all bit-exact ` +
        `(${c.n_buoy_neg} cells buoyant downward)`
      : bad.map(([nm, at, got, ref]) =>
          `${nm} differs at ${at}: got ${got[at]} ref ${ref[at]}`).join('; '),
  );
}

// ── step_drag_force ────────────────────────────────────────────────────────
for (const c of G.drag.cases) {
  const f = (a) => Float64Array.from(a);
  const n = c.nz * c.ny * c.nx;
  // Pre-fill with a sentinel: the kernel OVERWRITES, so a port that
  // accumulated would show up here.
  const Fx = new Float64Array(n).fill(-7.0);
  const Fy = new Float64Array(n).fill(-7.0);
  const Fz = new Float64Array(n).fill(-7.0);
  stepDragForce(f(c.u), f(c.v), f(c.w), f(c.rho), f(c.alpha_s), c.sigma_sav,
                Fx, Fy, Fz, c.C_D);
  const bad = [['Fx', Fx, c.Fx], ['Fy', Fy, c.Fy], ['Fz', Fz, c.Fz]]
    .map(([nm, got, ref]) => [nm, firstDiff(got, ref), got, ref])
    .filter(([, at]) => at >= 0);
  check(`drag.force.${c.name}`, bad.length === 0,
        bad.length === 0
          ? `${c.nz}x${c.ny}x${c.nx}, Fx/Fy/Fz bit-exact (${c.n_nofuel} no-fuel cells)`
          : bad.map(([nm, at, got, ref]) =>
              `${nm} at ${at}: got ${got[at]} ref ${ref[at]}`).join('; '));
}

// ── step_solid_conduction_vertical ─────────────────────────────────────────
for (const c of G.solid_conduction.cases) {
  const f = (a) => Float64Array.from(a);
  const Ts = f(c.T_s_in);
  stepSolidConductionVertical(
    Ts, f(c.alpha_s), f(c.dz_arr), f(c.d_face_above), f(c.d_face_below),
    c.k_solid, c.rho_solid, c.cp_solid, c.dt,
    { nx: c.nx, ny: c.ny, nz: c.nz });
  const at = firstDiff(Ts, c.T_s_out);
  check(`solidConduction.${c.name}`, at < 0,
        at < 0 ? `${c.nz}x${c.ny}x${c.nx} = ${c.T_s_out.length} cells, bit-exact`
               : `first diff at ${at}: got ${Ts[at]} ref ${c.T_s_out[at]}`);
}

// ── step_gas_solid_coupling ────────────────────────────────────────────────
for (const c of G.coupling.cases) {
  const f = (a) => Float64Array.from(a);
  const Tg = f(c.T_g_in), Ts = f(c.T_s_in), mw = f(c.m_water_in);
  stepGasSolidCoupling(
    Tg, Ts, f(c.rho), f(c.u), f(c.v), f(c.w), f(c.alpha_s), c.sigma_sav,
    f(c.q_rad_in), f(c.Q_pyro), f(c.Q_comb), mw, c.L_v, c.dt,
    f(c.dz_arr), c.T_amb, { nx: c.nx, ny: c.ny, nz: c.nz });
  const bad = [['T_g', Tg, c.T_g_out], ['T_s', Ts, c.T_s_out],
               ['m_water', mw, c.m_water_out]]
    .map(([nm, got, ref]) => [nm, firstDiff(got, ref), got, ref])
    .filter(([, at]) => at >= 0);
  check(`coupling.gasSolid.${c.name}`, bad.length === 0,
        bad.length === 0
          ? `${c.nz}x${c.ny}x${c.nx}, T_g/T_s/m_water bit-exact ` +
            `(${c.n_dried} cells fully dried — evap cap binds)`
          : bad.map(([nm, at, got, ref]) =>
              `${nm} at ${at}: got ${got[at]} ref ${ref[at]}`).join('; '));
}

// ── report ─────────────────────────────────────────────────────────────────
const w = Math.max(...results.map((r) => r.name.length));
for (const r of results) {
  console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name.padEnd(w)}  ${r.detail}`);
}
const nPass = results.filter((r) => r.ok).length;
console.log(`\n${nPass}/${results.length} kernel checks passed`);
process.exit(nPass === results.length ? 0 : 1);
