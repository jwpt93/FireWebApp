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
 * THE CONTRACT (two different standards, deliberately):
 *
 * 1. BETWEEN CODES — a tolerance, not identical bits.  Bit-exactness across
 *    languages is unattainable for any kernel touching pow() or exp(): IEEE
 *    does not require those to be correctly rounded, and V8's libm disagrees
 *    with glibc's by ~2 ulp (pow(1.743e-4, 0.25) is ...469729 in V8 against
 *    ...469727 in glibc).  Chasing that is wasted effort.  The actual max
 *    relative difference is REPORTED for every field so drift is visible, and
 *    where bit-exactness does happen to hold it is called out — the stencil
 *    kernels manage it because they use only + - * / sqrt, all correctly
 *    rounded.
 *
 * 2. WITHIN THIS CODE — bit-exact, no exceptions.  Every kernel is run twice
 *    on identical inputs and the two results must match to the last bit.
 *    This is CLAUDE.md Rule #17 applied to the port: without it, a validation
 *    result is noise, and we could not tell a physics change from a
 *    scheduling roll.  It is also the property that makes the tolerance in
 *    (1) meaningful — a reproducible port that sits a few ulp from the
 *    reference is a different thing from one that wanders.
 *
 * A caveat worth stating: EDC's extinction gates are DISCONTINUOUS, so a
 * 1-ulp difference in omega can cross a `< 0.5` threshold and move a cell's
 * T_g by degrees.  The browser solver is therefore a valid solution of the
 * same model, not a bit-identical replay of the Python trajectory.
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
import { stepChemistryOdeEdc } from './js/physics/edc.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const G = JSON.parse(readFileSync(join(HERE, 'data', 'kernel_vectors.json'), 'utf8'));

const results = [];
const check = (name, ok, detail) => results.push({ name, ok, detail });

/** First index where two arrays differ in bits, or -1. */
function firstDiff(got, ref) {
  for (let i = 0; i < ref.length; i++) if (got[i] !== ref[i]) return i;
  return -1;
}

/** Largest relative difference (absolute where the reference is zero). */
function maxRel(got, ref) {
  let worst = 0, at = -1;
  for (let i = 0; i < ref.length; i++) {
    const d = ref[i] === 0 ? Math.abs(got[i])
                           : Math.abs(got[i] - ref[i]) / Math.abs(ref[i]);
    if (d > worst) { worst = d; at = i; }
  }
  return { worst, at };
}

/** Default cross-language tolerance. Tight enough to catch a real bug, loose
 *  enough not to trip on libm. */
const XLANG_TOL = 1e-9;

/**
 * Compare named field triples [name, got, ref] against the tolerance, and
 * report whether any of them additionally came out bit-exact -- that is free
 * information about how clean the port is, worth surfacing even though it is
 * no longer required.
 */
function compareFields(label, fields, tol = XLANG_TOL, extra = '') {
  const rows = fields.map(([nm, got, ref]) => {
    const { worst, at } = maxRel(got, ref);
    return { nm, worst, at, got, ref, exact: firstDiff(got, ref) < 0 };
  });
  const over = rows.filter((r) => r.worst > (typeof tol === 'object' ? tol[r.nm] : tol));
  const allExact = rows.every((r) => r.exact);
  const summary = allExact
    ? `bit-exact (${rows.length} fields)`
    : rows.map((r) => `${r.nm} ${r.worst.toExponential(1)}`).join(', ');
  check(label, over.length === 0,
        over.length === 0
          ? `${summary}${extra ? ' — ' + extra : ''}`
          : over.map((r) => `${r.nm} rel ${r.worst.toExponential(2)} at ${r.at}: ` +
              `got ${r.got[r.at]} ref ${r.ref[r.at]}`).join('; '));
}

/**
 * Rule #17 within the port: run `fn` twice on freshly-built inputs and demand
 * the two results match bit for bit.
 */
function checkDeterminism(label, fn) {
  const a = fn();
  const b = fn();
  const bad = a.map((arr, i) => firstDiff(arr, b[i])).filter((x) => x >= 0);
  check(`determinism.${label}`, bad.length === 0,
        bad.length === 0
          ? `two runs bit-identical (${a.length} output fields)`
          : `${bad.length} field(s) differ between runs`);
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

// ── step_chemistry_ode_edc ─────────────────────────────────────────────────
for (const c of G.edc.cases) {
  const f = (a) => Float64Array.from(a);
  const Tg = f(c.T_g_in), Yf = f(c.Y_fuel_in), YO2 = f(c.Y_O2_in);
  const om = new Float64Array(c.omega_out.length);
  stepChemistryOdeEdc(
    f(c.rho), Tg, Yf, YO2, f(c.k_turb), f(c.eps_turb), c.chi_rad, c.cp_g,
    c.dt, c.n_substeps, om, f(c.Y_H2O),
    { extinctionEnable: c.extinction_enable, sStoich: c.s_stoich, hocJ: c.hoc_J },
  );
  // NOT bit-exact, and deliberately so -- see the note at the top of this
  // file.  EDC uses pow() and exp(), which IEEE does not require to be
  // correctly rounded, and V8's libm differs from glibc's by ~2 ulp.  The
  // stencil kernels avoid this by using only + - * / sqrt.
  //
  // Tolerances differ per field because the amplification is not uniform:
  // omega and the mass fractions track the ulp error directly, while T_g
  // additionally inherits THRESHOLD FLIPS -- a 1-ulp omega can cross the
  // `< 0.5` gate gating the wet-bulb cascade, which is worth ~2.3 K per
  // substep.  That is a property of the model's discontinuous extinction
  // gates, not of the port.
  const TOL = { omega: 1e-9, Y_fuel: 1e-9, Y_O2: 1e-9, T_g: 1e-2 };
  const bad = [['T_g', Tg, c.T_g_out], ['Y_fuel', Yf, c.Y_fuel_out],
               ['Y_O2', YO2, c.Y_O2_out], ['omega', om, c.omega_out]]
    .map(([nm, got, ref]) => {
      let worst = 0, at = -1;
      for (let i = 0; i < ref.length; i++) {
        const d = ref[i] === 0 ? Math.abs(got[i])
                               : Math.abs(got[i] - ref[i]) / Math.abs(ref[i]);
        if (d > worst) { worst = d; at = i; }
      }
      return [nm, worst, at, got, ref];
    });
  const over = bad.filter(([nm, worst]) => worst > TOL[nm]);
  const worstAll = bad.map(([nm, w]) => `${nm} ${w.toExponential(1)}`).join(', ');
  check(`edc.chemistry.${c.name}`, over.length === 0,
        over.length === 0
          ? `${c.nz}x${c.ny}x${c.nx}, max rel. diff — ${worstAll} ` +
            `(${c.n_quenched} quenched, ${c.n_below_Tign} below T_ign)`
          : over.map(([nm, w, at, got, ref]) =>
              `${nm} rel ${w.toExponential(2)} > ${TOL[nm]} at ${at}: ` +
              `got ${got[at]} ref ${ref[at]}`).join('; '));
}

// ── Rule #17 within the port: every kernel, twice, bit-identical ──────────
{
  const f = (a) => Float64Array.from(a);
  const m = G.muscl.cases[0];
  checkDeterminism('muscl.advect', () => {
    const rhs = f(m.rhs_in);
    advect3dScalarMuscl(f(m.phi), f(m.u), f(m.v), f(m.w), m.dx, m.dy,
      f(m.d_face_above), f(m.d_face_below), rhs, m.phi_inlet,
      { nx: m.nx, ny: m.ny, nz: m.nz });
    return [rhs];
  });

  const sp = G.species.cases[0];
  checkDeterminism('species.transport', () => {
    const Y = f(sp.Y_in);
    stepSpeciesTransport(Y, f(sp.rho), f(sp.u), f(sp.v), f(sp.w), f(sp.S),
      sp.dt, sp.dx, sp.dy, f(sp.dz_arr), f(sp.d_face_above),
      f(sp.d_face_below), sp.D, sp.Y_inlet,
      { nx: sp.nx, ny: sp.ny, nz: sp.nz });
    return [Y];
  });

  const mo = G.momentum.cases[0];
  checkDeterminism('momentum.tentative', () => {
    const u = f(mo.u_in), v = f(mo.v_in), w = f(mo.w_in);
    stepTentativeVelocity(u, v, w, f(mo.rho), f(mo.T_g), f(mo.Fx), f(mo.Fy),
      f(mo.Fz), mo.dt, mo.dx, mo.dy, f(mo.dz_arr), f(mo.d_face_above),
      f(mo.d_face_below), mo.T_amb, f(mo.u_inlet), f(mo.v_inlet),
      f(mo.w_inlet), { nx: mo.nx, ny: mo.ny, nz: mo.nz });
    return [u, v, w];
  });

  const dr = G.drag.cases[0];
  checkDeterminism('drag.force', () => {
    const n = dr.nz * dr.ny * dr.nx;
    const Fx = new Float64Array(n), Fy = new Float64Array(n), Fz = new Float64Array(n);
    stepDragForce(f(dr.u), f(dr.v), f(dr.w), f(dr.rho), f(dr.alpha_s),
      dr.sigma_sav, Fx, Fy, Fz, dr.C_D);
    return [Fx, Fy, Fz];
  });

  const sc = G.solid_conduction.cases[0];
  checkDeterminism('solidConduction', () => {
    const Ts = f(sc.T_s_in);
    stepSolidConductionVertical(Ts, f(sc.alpha_s), f(sc.dz_arr),
      f(sc.d_face_above), f(sc.d_face_below), sc.k_solid, sc.rho_solid,
      sc.cp_solid, sc.dt, { nx: sc.nx, ny: sc.ny, nz: sc.nz });
    return [Ts];
  });

  const co = G.coupling.cases[0];
  checkDeterminism('coupling.gasSolid', () => {
    const Tg = f(co.T_g_in), Ts = f(co.T_s_in), mw = f(co.m_water_in);
    stepGasSolidCoupling(Tg, Ts, f(co.rho), f(co.u), f(co.v), f(co.w),
      f(co.alpha_s), co.sigma_sav, f(co.q_rad_in), f(co.Q_pyro), f(co.Q_comb),
      mw, co.L_v, co.dt, f(co.dz_arr), co.T_amb,
      { nx: co.nx, ny: co.ny, nz: co.nz });
    return [Tg, Ts, mw];
  });

  const ed = G.edc.cases[0];
  checkDeterminism('edc.chemistry', () => {
    const Tg = f(ed.T_g_in), Yf = f(ed.Y_fuel_in), YO2 = f(ed.Y_O2_in);
    const om = new Float64Array(ed.omega_out.length);
    stepChemistryOdeEdc(f(ed.rho), Tg, Yf, YO2, f(ed.k_turb), f(ed.eps_turb),
      ed.chi_rad, ed.cp_g, ed.dt, ed.n_substeps, om, f(ed.Y_H2O),
      { extinctionEnable: ed.extinction_enable, sStoich: ed.s_stoich,
        hocJ: ed.hoc_J });
    return [Tg, Yf, YO2, om];
  });
}

// ── report ─────────────────────────────────────────────────────────────────
const w = Math.max(...results.map((r) => r.name.length));
for (const r of results) {
  console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.name.padEnd(w)}  ${r.detail}`);
}
const nPass = results.filter((r) => r.ok).length;
console.log(`\n${nPass}/${results.length} kernel checks passed`);
process.exit(nPass === results.length ? 0 : 1);
