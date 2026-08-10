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
import { SeparableLaplacian3D } from './js/physics/poisson.js';
import { applyTurbulentDiffusion } from './js/physics/turbulentDiffusion.js';
import { stepKEpsilon } from './js/physics/kepsilon.js';
import { DOMRadiationSolver } from './js/physics/dom.js';
import {
  allocateBedParticleBuffers, initializeBedParticlesFromAlphaS,
  stepBedParticles, aggregateParticlesToTsGrid,
  aggregateParticlesToMLocalGrid, stepHorizontalSolidConductionScatter,
} from './js/physics/lagrangianBed.js';
import { ProjectionSolver3D } from './js/physics/projection.js';
import {
  LevelSetFront3D, godunovGradNorm, reinitGodunovGrad, flameTiltBandM,
  computeQInAtFront3d, computeVn3d, computePhiFlameFromState,
  flameBodyMaskFromPhiFlame, updateCellAge,
} from './js/physics/flameFront.js';
import {
  applyOutflowSponge, applyWallFunction, stepO2SupplyRate,
  buildSoilGrid, stepSoilConduction, advGasEnergy,
} from './js/physics/support.js';

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

// ── SeparableLaplacian3D.solve ────────────────────────────────────────────
// Compared on the SOLUTION, not on eigenvectors: those are defined only up to
// sign and ordering, so LAPACK and the implicit-QL solver here legitimately
// disagree on them while producing the same p.
for (const c of G.poisson.cases) {
  const f = (a) => Float64Array.from(a);
  const sol = new SeparableLaplacian3D({
    nz: c.nz, ny: 1, nx: c.nx, dx: c.dx, dy: c.dy,
    dzArr: f(c.dz_arr), dFaceAbove: f(c.d_face_above),
    dFaceBelow: f(c.d_face_below), epsReg: c.eps_reg,
  });
  const p = sol.solve(f(c.rhs));
  compareFields(`poisson.solve.${c.name}`, [['p', p, c.p_out]], XLANG_TOL,
                `${c.nz}x1x${c.nx}, own eigensolver (implicit QL) vs LAPACK`);
}

// ── apply_turbulent_diffusion ─────────────────────────────────────────────
for (const c of G.turb_diff.cases) {
  const f = (a) => Float64Array.from(a);
  const fld = f(c.field_in);
  applyTurbulentDiffusion(fld, f(c.nu_t), c.sc_t, c.dt, c.dx, c.dy,
    f(c.dz_arr), f(c.d_face_above), f(c.d_face_below),
    { nx: c.nx, ny: c.ny, nz: c.nz });
  compareFields(`turbDiff.${c.name}`, [['field', fld, c.field_out]], XLANG_TOL,
                `${c.nz}x${c.ny}x${c.nx}, ${c.n_sub} sub-steps`);
}

// ── step_k_epsilon ────────────────────────────────────────────────────────
// Vectors regenerated after the 2026-08-10 upstream fix to the dT/dz
// boundaries (the old code read out of bounds at k=0 and k=Nz-1).
for (const c of G.kepsilon.cases) {
  const f = (a) => Float64Array.from(a);
  const n = c.nz * c.ny * c.nx;
  const kT = f(c.k_in), eT = f(c.eps_in);
  const nuT = new Float64Array(n);
  stepKEpsilon(kT, eT, nuT, f(c.u), f(c.v), f(c.w), f(c.T_g), f(c.rho),
    f(c.alpha_s), c.sigma_sav, c.dt, c.dx, c.dy, f(c.dz_arr),
    f(c.d_face_above), f(c.d_face_below), c.T_amb,
    new Float64Array(n), new Float64Array(n), f(c.u_inlet),
    f(c.k_wall_ghost), f(c.eps_wall_ghost), c.beta_p, c.beta_d,
    { nx: c.nx, ny: c.ny, nz: c.nz });
  compareFields(`kEpsilon.${c.name}`,
    [['k', kT, c.k_out], ['eps', eT, c.eps_out], ['nu_t', nuT, c.nu_t_out]],
    XLANG_TOL, `${c.nz}x${c.ny}x${c.nx}, ${c.n_canopy} canopy cells`);
}

// ── DOMRadiationSolver.solve ──────────────────────────────────────────────
for (const c of G.dom.cases) {
  const f = (a) => Float64Array.from(a);
  const n = c.nz * c.ny * c.nx;
  const sol = new DOMRadiationSolver({
    nz: c.nz, ny: c.ny, nx: c.nx, dx: c.dx, dy: c.dy, dzArr: f(c.dz_arr),
  });
  const qs = new Float64Array(n), qg = new Float64Array(n);
  const qsoil = new Float64Array(c.ny * c.nx);
  sol.solve({
    Ts: f(c.T_s), Tg: f(c.T_g), alphaS: f(c.alpha_s),
    omegaComb: f(c.omega_comb), sigmaSav: c.sigma_sav, Tamb: c.T_amb,
    qRadSolidOut: qs, qRadGasOut: qg,
    TsoilSurface: f(c.T_soil), qInSoilOut: qsoil,
    YH2O: c.Y_H2O ? f(c.Y_H2O) : null,
    rho: c.rho ? f(c.rho) : null,
    bedMoisturePerCell: f(c.bed_moisture),
  });
  compareFields(`dom.solve.${c.name}`,
    [['q_solid', qs, c.q_rad_solid], ['q_gas', qg, c.q_rad_gas],
     ['q_soil', qsoil, c.q_in_soil]],
    XLANG_TOL, `${c.nz}x${c.ny}x${c.nx}, S4/24 ordinates, source-iterated`);
}

// ── Loop-support kernels ──────────────────────────────────────────────────
for (const c of G.support.cases) {
  const f = (a) => Float64Array.from(a);
  const shape = { nx: c.nx, ny: c.ny, nz: c.nz };
  const n = c.nz * c.ny * c.nx;

  const uSp = f(c.sponge_u_in);
  applyOutflowSponge(uSp, f(c.sponge_u_target), f(c.sponge_sigma_x),
    f(c.sponge_Y_F), c.sponge_Y_F_skip, c.sponge_dt, shape);
  compareFields(`support.sponge.${c.name}`,
    [['u', uSp, c.sponge_u_out]], XLANG_TOL,
    `${c.n_sponge_skipped} fuel-bearing cells skipped in the sponge zone`);

  const kGh = new Float64Array(c.ny * c.nx);
  const epsGh = new Float64Array(c.ny * c.nx);
  applyWallFunction(f(c.wf_u), f(c.wf_v), f(c.rho), f(c.wf_alpha_s),
    f(c.dz_arr), kGh, epsGh,
    { ...shape, kMin: c.wf_k_min, epsMin: c.wf_eps_min });
  compareFields(`support.wallFunction.${c.name}`,
    [['k_ghost', kGh, c.wf_k_out], ['eps_ghost', epsGh, c.wf_eps_out]],
    XLANG_TOL,
    `${c.n_wf_bed} bed columns skipped, ${c.n_wf_loglaw} through the log law`);

  const om = new Float64Array(n).fill(1.0e30);
  stepO2SupplyRate(f(c.rho), f(c.u), f(c.v), f(c.w), f(c.Y_O2),
    c.dx, c.dy, f(c.dz_arr), om, shape);
  compareFields(`support.o2Supply.${c.name}`,
    [['omega_O2', om, c.o2_out]], XLANG_TOL,
    c.n_o2_written === 0
      ? 'Ny=1: interior loop is EMPTY, every cell keeps the 1e30 fill'
      : `${c.n_o2_written} interior cells written`);

  // Soil grid is rebuilt rather than read back, so buildSoilGrid is checked too.
  const sg = buildSoilGrid();
  compareFields(`support.soilGrid.${c.name}`,
    [['dz', sg.soilDz, c.soil_dz], ['d_above', sg.dAbove, c.soil_d_above],
     ['d_below', sg.dBelow, c.soil_d_below]],
    XLANG_TOL, `${c.n_soil} layers, ${(c.soil_depth * 1000).toFixed(1)} mm deep`);

  const Tsoil = f(c.soil_T_in);
  stepSoilConduction(Tsoil, f(c.soil_q_in), c.soil_dt, sg.soilDz,
    sg.dAbove, sg.dBelow,
    { nx: c.nx, ny: c.ny, nSoil: c.n_soil, Tamb: 300.0 });
  compareFields(`support.soilConduction.${c.name}`,
    [['T_soil', Tsoil, c.soil_T_out]], XLANG_TOL,
    'T^4 surface loss against conduction');

  const Tg = f(c.Tg_in);
  advGasEnergy(Tg, f(c.u), f(c.v), f(c.w), c.gas_dt, c.dx, c.dy, f(c.dz_arr),
    f(c.d_face_above), f(c.d_face_below), 2.0e-5, 300.0, shape);
  compareFields(`support.gasEnergy.${c.name}`,
    [['T_g', Tg, c.Tg_out]], XLANG_TOL, 'MUSCL advection + FV diffusion');
}

// ── Level-set front + flame geometry ──────────────────────────────────────
for (const c of G.flame_front.cases) {
  const f = (a) => Float64Array.from(a);
  const u8 = (a) => Uint8Array.from(a);
  // null round-trips through JSON as the marker for +inf, which JSON cannot
  // carry. Only cell_age uses it.
  const fInf = (a) => Float64Array.from(a, (v) => (v === null ? Infinity : v));
  const shape = { nx: c.nx, ny: c.ny, nz: c.nz };
  const n = c.nz * c.ny * c.nx;

  // Albini flame tilt, across a wind sweep including the U=0 degenerate case.
  {
    let bad = 0, worst = 0;
    for (const [u, want] of c.tilt_probe) {
      const got = flameTiltBandM(u);
      const d = want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want);
      if (d > worst) worst = d;
      if (d > XLANG_TOL) bad++;
    }
    check(`flameFront.tiltBand.${c.name}`, bad === 0,
      `${c.tilt_probe.length} wind probes, max rel ${worst.toExponential(1)}`);
  }

  // Source-patch initialisation, then the two Godunov gradients.
  const lset = new LevelSetFront3D({
    nz: c.nz, ny: c.ny, nx: c.nx, dx: c.dx, dy: c.dy, dzArr: f(c.dz_arr),
  });
  lset.initializeSourcePatch(1, 4, c.n_z_bed - 1, f(c.x_mid));
  compareFields(`flameFront.initPatch.${c.name}`,
    [['phi', lset.phi, c.phi_init]], XLANG_TOL, 'signed distance to patch edge');

  const grad = new Float64Array(n);
  godunovGradNorm(lset.phi, c.dx, c.dy, f(c.dz_arr), grad, shape);
  compareFields(`flameFront.godunov.${c.name}`,
    [['grad', grad, c.grad_out]], XLANG_TOL, 'upwind |grad phi|, v_n > 0');

  const gradR = new Float64Array(n);
  reinitGodunovGrad(f(c.phi_dist), f(c.phi_init), c.dx, c.dy, f(c.dz_arr),
                    gradR, shape);
  compareFields(`flameFront.reinitGrad.${c.name}`,
    [['grad', gradR, c.grad_reinit]], XLANG_TOL, 'sign-aware, both branches');

  // Evolve then reinit -- the full per-step level-set update.
  lset.evolve(c.dt, f(c.v_n_in));
  compareFields(`flameFront.evolve.${c.name}`,
    [['phi', lset.phi, c.phi_evolved]], XLANG_TOL, 'z-varying v_n');
  lset.reinitialize();
  compareFields(`flameFront.reinit.${c.name}`,
    [['phi', lset.phi, c.phi_reinit]], XLANG_TOL,
    `${5} Sussman substeps`);

  // Masks and front position.
  const ahead = lset.aheadBandMask(c.band_m);
  for (let k = c.n_z_bed; k < c.nz; k++) {
    ahead.fill(0, k * c.ny * c.nx, (k + 1) * c.ny * c.nx);   // bed-only
  }
  compareFields(`flameFront.masks.${c.name}`,
    [['ahead', ahead, c.ahead_mask],
     ['flame_body', lset.flameBodyMask(), c.flame_body_mask],
     ['burned', lset.burnedMask(), c.burned_mask]],
    XLANG_TOL, `${c.n_ahead} cells in the ahead-band`);
  {
    const got = lset.frontX(1, Math.floor(c.ny / 2));
    const want = c.front_x;
    const ok = want === null ? !Number.isFinite(got)
      : Math.abs(got - want) / Math.abs(want) <= XLANG_TOL;
    check(`flameFront.frontX.${c.name}`, ok, `x = ${got}`);
  }

  // phi_flame: the exact separable EDT against scipy's exact EDT.
  const phiFlame = computePhiFlameFromState(
    f(c.omega), f(c.T_g), f(c.Y_fuel), c.dx, c.dy, f(c.dz_arr), shape);
  compareFields(`flameFront.phiFlame.${c.name}`,
    [['phi_flame', phiFlame, c.phi_flame],
     ['body_mask', flameBodyMaskFromPhiFlame(phiFlame, 0.0), c.fb_from_phi]],
    XLANG_TOL,
    `exact EDT vs scipy, ${c.n_active} active cells (reaction + plume tail)`);

  // Forward flux into the band, and the v_n it drives, dry and wet.
  const qIn = computeQInAtFront3d(new Float64Array(n), f(c.q_dom_fwd),
                                  ahead, f(c.q_burst), shape);
  const vnDry = computeVn3d(qIn, 1.07, 1850.0, 0.25, 600.0, 300.0, null);
  const vnWet = computeVn3d(qIn, 1.07, 1850.0, 0.25, 600.0, 300.0, f(c.M_local));
  compareFields(`flameFront.vn.${c.name}`,
    [['q_in', qIn, c.q_in], ['v_n_dry', vnDry, c.v_n_dry],
     ['v_n_wet', vnWet, c.v_n_wet]],
    XLANG_TOL, 'latent term dominates at M >= 0.1');

  // cell_age across ignite / continue / reset in one call.
  const age = fInf(c.cell_age_in);
  updateCellAge(age, u8(c.flame_body_mask), c.dt);
  const wantAge = fInf(c.cell_age_out);
  let bad = 0;
  for (let i = 0; i < n; i++) {
    if (!(age[i] === wantAge[i] || (!Number.isFinite(age[i]) && !Number.isFinite(wantAge[i])))) bad++;
  }
  check(`flameFront.cellAge.${c.name}`, bad === 0,
    bad === 0 ? 'ignite / continue / reset all bit-exact' : `${bad} cells differ`);
}

// ── Pressure projection ───────────────────────────────────────────────────
// Three checks at two different standards -- see vec_projection() for why the
// projection result is judged on residual rather than elementwise.
for (const c of G.projection.cases) {
  const f = (a) => Float64Array.from(a);
  const sol = new ProjectionSolver3D({
    nz: c.nz, ny: c.ny, nx: c.nx, dx: c.dx, dy: c.dy,
    dzArr: f(c.dz_arr), dFaceAbove: f(c.d_face_above),
    dFaceBelow: f(c.d_face_below), method: 'fft_pcg', cgRtol: c.cg_rtol,
  });
  sol.setInletBC(f(c.u_inlet));
  sol.rebuildForRho(f(c.rho));

  // 1. The operator. Seven stencil arrays here against a scipy CSR matrix
  //    there -- same star, different storage.
  const probe = f(c.probe);
  const mv = new Float64Array(probe.length);
  sol.matvec(probe, mv);
  compareFields(`projection.matvec.${c.name}`,
    [['A_x', mv, c.matvec_out]], XLANG_TOL,
    '7-point variable-density operator vs scipy CSR');

  // 2. The divergence stencil, with its mirror ghosts at inlet and wall.
  const u = f(c.u_in), v = f(c.v_in), w = f(c.w_in);
  const div = sol.divergence(u, v, w).slice();
  compareFields(`projection.divergence.${c.name}`,
    [['div', div, c.div_out]], XLANG_TOL, 'FV divergence, mirror ghost BCs');

  // 3. Did the projection do its job? Both codes must drive
  //    max|div(u_new) - div_target| to the same smallness. They will not
  //    agree on p -- two Krylov solves stopped at the same relative residual
  //    sit in the same ball, not on the same point.
  const dTarget = f(c.div_target);
  sol.project(u, v, w, f(c.rho), c.dt, dTarget);
  const after = sol.divergence(u, v, w);
  let resid = 0;
  for (let i = 0; i < after.length; i++) {
    resid = Math.max(resid, Math.abs(after[i] - dTarget[i]));
  }
  // Allow 10x the reference residual: a different Krylov path lands at a
  // different point inside the same tolerance ball, and the div residual is
  // the amplified image of that. An order of magnitude is generous enough not
  // to be flaky and tight enough that a real ordering bug -- which would move
  // this by many orders -- still fails.
  const budget = Math.max(10.0 * c.proj_resid, 1e-9);
  check(`projection.project.${c.name}`, resid <= budget,
    `max|div-target| ${resid.toExponential(2)} vs ref ` +
    `${c.proj_resid.toExponential(2)} (budget ${budget.toExponential(2)}), ` +
    `${sol.lastIters} BiCGSTAB iters vs ref ${c.n_iters}`);
}

// ── Lagrangian bed particles ──────────────────────────────────────────────
// Four kernels, checked in the order the solver calls them. The vectors carry
// flat per-slot particle arrays rather than (Nz,Ny,Nx) fields, so this block
// builds its state differently from every other one above.
for (const c of G.lagrangian_bed.cases) {
  const f = (a) => Float64Array.from(a);
  const i32 = (a) => Int32Array.from(a);
  const shape = { nx: c.nx, ny: c.ny, nz: c.nz };
  const n = c.nz * c.ny * c.nx;

  // 1. The initialiser. Deterministic coprime-mod packing, no RNG anywhere,
  //    so this one genuinely should be bit-exact — it is pure arithmetic on
  //    small integers.
  {
    const buf = allocateBedParticleBuffers(c.n_max);
    const nAlloc = initializeBedParticlesFromAlphaS(
      buf, f(c.alpha_s), c.rho_b_dry, c.moisture_frac, c.T_amb,
      c.dx, c.dy, f(c.dz_arr), c.n_z_bed, c.n_per_cell,
      { ...shape, sav: c.sav },
    );
    compareFields(`bed.init.${c.name}`,
      [['x', buf.x, c.init_x], ['y', buf.y, c.init_y], ['z', buf.z, c.init_z],
       ['m_solid', buf.m_solid, c.init_m_solid],
       ['m_water', buf.m_water, c.init_m_water],
       ['alive', buf.alive, c.init_alive]],
      XLANG_TOL, `${nAlloc}/${c.n_alloc} particles, ${c.n_per_cell}/cell`);
  }

  // 2. The step itself. Builds the perturbed mid-fire population from the
  //    recorded inputs, not from the initialiser, so the two are independent.
  const mkState = () => ({
    x: f(c.in_x), y: f(c.in_y), z: f(c.in_z), alive: i32(c.in_alive),
    m_solid: f(c.in_m_solid), m_water: f(c.in_m_water), m_char: f(c.in_m_char),
    T_s: f(c.in_T_s), m_water_0: f(c.in_m_water_0), sav: f(c.in_sav),
    m_char_max: f(c.in_m_char_max),
  });
  const mkOut = () => ({
    S_pyro: new Float64Array(n), S_drying: new Float64Array(n),
    Q_pyro: new Float64Array(n), Q_drying: new Float64Array(n),
    Y_F_source: new Float64Array(n), Q_char: new Float64Array(n),
    Q_smold: new Float64Array(n), Q_g_conv: new Float64Array(n),
    nAliveOut: new Int32Array(1), nBurnedOut: new Int32Array(1),
    diagMaxOut: new Float64Array(16),
  });
  const par = {
    dx: c.dx, dy: c.dy, dzArr: f(c.dz_arr), zFace: f(c.z_face),
    hConv: c.h_conv, rhoSolidTrue: c.rho_solid_true, cpSolid: c.cp_solid,
    epsSolid: c.eps_solid, tAmb: c.T_amb, viewFactor: c.view_factor,
    viewFactorGeometric: c.view_factor_geometric, hBed: c.h_bed,
    kappaBedEff: c.kappa_bed_eff, dt: c.dt,
    doDrying: true, doPyrolysis: true, doCharOx: true, doSmolder: true,
    dryingMode: c.drying_mode, charOxFluxCapWm2: c.char_ox_flux_cap,
    charOxAshExp: c.char_ox_ash_exp, nPerCellForSplit: c.n_per_cell,
  };
  const gas = { ...shape, T_g: f(c.T_g), Y_O2: f(c.Y_O2), Q_solid_ext: f(c.Q_solid_ext) };

  const s = mkState();
  const out = mkOut();
  stepBedParticles(s, gas, out, par);

  compareFields(`bed.step.${c.name}`,
    [['T_s', s.T_s, c.out_T_s], ['m_solid', s.m_solid, c.out_m_solid],
     ['m_water', s.m_water, c.out_m_water], ['m_char', s.m_char, c.out_m_char],
     ['m_char_max', s.m_char_max, c.out_m_char_max],
     ['alive', s.alive, c.out_alive],
     ['S_pyro', out.S_pyro, c.out_S_pyro],
     ['S_drying', out.S_drying, c.out_S_drying],
     ['Q_pyro', out.Q_pyro, c.out_Q_pyro],
     ['Q_drying', out.Q_drying, c.out_Q_drying],
     ['Y_F_source', out.Y_F_source, c.out_Y_F_source],
     ['Q_char', out.Q_char, c.out_Q_char],
     ['Q_smold', out.Q_smold, c.out_Q_smold],
     ['Q_g_conv', out.Q_g_conv, c.out_Q_g_conv],
     ['diag', out.diagMaxOut, c.out_diag]],
    XLANG_TOL,
    `${c.n_alloc} particles, dry_mode=${c.drying_mode}, ` +
    `geom_vf=${c.view_factor_geometric}, ash_exp=${c.char_ox_ash_exp}`);

  // The alive/burned tally is an integer count — no tolerance applies, it is
  // either the same number or the port took a different branch somewhere.
  check(`bed.counts.${c.name}`,
    out.nAliveOut[0] === c.out_n_alive && out.nBurnedOut[0] === c.out_n_burned,
    `alive ${out.nAliveOut[0]}/${c.out_n_alive}, ` +
    `burned ${out.nBurnedOut[0]}/${c.out_n_burned}`);

  // 3. The two aggregators, on the post-step particle state — same order the
  //    solver uses, since both feed DOM.
  const TsGrid = new Float64Array(n).fill(c.T_amb);
  aggregateParticlesToTsGrid(s.x, s.y, s.z, s.alive, s.m_solid, s.m_water,
    s.m_char, s.T_s, c.dx, c.dy, f(c.z_face), TsGrid, c.T_amb, shape);
  const MGrid = new Float64Array(n);
  aggregateParticlesToMLocalGrid(s.x, s.y, s.z, s.alive, s.m_solid, s.m_water,
    c.dx, c.dy, f(c.z_face), MGrid, shape);
  compareFields(`bed.aggregate.${c.name}`,
    [['T_s_grid', TsGrid, c.out_T_s_grid], ['M_local', MGrid, c.out_M_local]],
    XLANG_TOL, 'mass-weighted T_s, ratio-of-sums M_local');

  // 4. Horizontal conduction, from the same pinned grid the Python used.
  const condT = f(c.cond_T_in);
  const condPartT = f(c.out_T_s);
  stepHorizontalSolidConductionScatter(
    s.x, s.y, s.z, s.alive, s.m_solid, s.m_water, s.m_char, condPartT,
    condT, f(c.alpha_s), c.dx, c.dy, f(c.z_face),
    c.k_solid, c.rho_solid_true, c.cp_solid, c.n_z_bed, c.dt, shape);
  compareFields(`bed.conduction.${c.name}`,
    [['T_s_grid', condT, c.cond_T_out],
     ['part_T_s', condPartT, c.cond_part_T_out]],
    XLANG_TOL, `k_solid=${c.k_solid} W/m/K, ${c.n_z_bed} bed layers`);
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

  const dm = G.dom.cases[0];
  checkDeterminism('dom.solve', () => {
    const n = dm.nz * dm.ny * dm.nx;
    const sol = new DOMRadiationSolver({
      nz: dm.nz, ny: dm.ny, nx: dm.nx, dx: dm.dx, dy: dm.dy,
      dzArr: f(dm.dz_arr),
    });
    const qs = new Float64Array(n), qg = new Float64Array(n);
    sol.solve({
      Ts: f(dm.T_s), Tg: f(dm.T_g), alphaS: f(dm.alpha_s),
      omegaComb: f(dm.omega_comb), sigmaSav: dm.sigma_sav, Tamb: dm.T_amb,
      qRadSolidOut: qs, qRadGasOut: qg, TsoilSurface: f(dm.T_soil),
      bedMoisturePerCell: f(dm.bed_moisture),
    });
    return [qs, qg];
  });

  const ke = G.kepsilon.cases[0];
  checkDeterminism('kEpsilon', () => {
    const n = ke.nz * ke.ny * ke.nx;
    const kT = f(ke.k_in), eT = f(ke.eps_in), nuT = new Float64Array(n);
    stepKEpsilon(kT, eT, nuT, f(ke.u), f(ke.v), f(ke.w), f(ke.T_g), f(ke.rho),
      f(ke.alpha_s), ke.sigma_sav, ke.dt, ke.dx, ke.dy, f(ke.dz_arr),
      f(ke.d_face_above), f(ke.d_face_below), ke.T_amb,
      new Float64Array(n), new Float64Array(n), f(ke.u_inlet),
      f(ke.k_wall_ghost), f(ke.eps_wall_ghost), ke.beta_p, ke.beta_d,
      { nx: ke.nx, ny: ke.ny, nz: ke.nz });
    return [kT, eT, nuT];
  });

  const td = G.turb_diff.cases[0];
  checkDeterminism('turbDiff', () => {
    const fld = f(td.field_in);
    applyTurbulentDiffusion(fld, f(td.nu_t), td.sc_t, td.dt, td.dx, td.dy,
      f(td.dz_arr), f(td.d_face_above), f(td.d_face_below),
      { nx: td.nx, ny: td.ny, nz: td.nz });
    return [fld];
  });

  const po = G.poisson.cases[0];
  checkDeterminism('poisson.solve', () => {
    const sol = new SeparableLaplacian3D({
      nz: po.nz, ny: 1, nx: po.nx, dx: po.dx, dy: po.dy,
      dzArr: f(po.dz_arr), dFaceAbove: f(po.d_face_above),
      dFaceBelow: f(po.d_face_below), epsReg: po.eps_reg,
    });
    return [sol.solve(f(po.rhs))];
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

  const sup = G.support.cases[0];
  checkDeterminism('support.gasEnergy', () => {
    const Tg = f(sup.Tg_in);
    advGasEnergy(Tg, f(sup.u), f(sup.v), f(sup.w), sup.gas_dt, sup.dx, sup.dy,
      f(sup.dz_arr), f(sup.d_face_above), f(sup.d_face_below), 2.0e-5, 300.0,
      { nx: sup.nx, ny: sup.ny, nz: sup.nz });
    return [Tg];
  });

  const ls = G.flame_front.cases[0];
  checkDeterminism('flameFront.evolveReinit', () => {
    const lset = new LevelSetFront3D({
      nz: ls.nz, ny: ls.ny, nx: ls.nx, dx: ls.dx, dy: ls.dy,
      dzArr: f(ls.dz_arr),
    });
    lset.initializeSourcePatch(1, 4, ls.n_z_bed - 1, f(ls.x_mid));
    lset.evolve(ls.dt, f(ls.v_n_in));
    lset.reinitialize();
    return [lset.phi.slice()];
  });
  checkDeterminism('flameFront.phiFlame', () => [
    computePhiFlameFromState(f(ls.omega), f(ls.T_g), f(ls.Y_fuel),
      ls.dx, ls.dy, f(ls.dz_arr), { nx: ls.nx, ny: ls.ny, nz: ls.nz }),
  ]);

  const pj = G.projection.cases[0];
  checkDeterminism('projection.project', () => {
    const sol = new ProjectionSolver3D({
      nz: pj.nz, ny: pj.ny, nx: pj.nx, dx: pj.dx, dy: pj.dy,
      dzArr: f(pj.dz_arr), dFaceAbove: f(pj.d_face_above),
      dFaceBelow: f(pj.d_face_below), method: 'fft_pcg', cgRtol: pj.cg_rtol,
    });
    sol.setInletBC(f(pj.u_inlet));
    sol.rebuildForRho(f(pj.rho));
    const u = f(pj.u_in), v = f(pj.v_in), w = f(pj.w_in);
    const p = sol.project(u, v, w, f(pj.rho), pj.dt, f(pj.div_target));
    return [u, v, w, p.slice()];
  });

  const bd = G.lagrangian_bed.cases[0];
  checkDeterminism('bed.stepBedParticles', () => {
    const n = bd.nz * bd.ny * bd.nx;
    const s = {
      x: f(bd.in_x), y: f(bd.in_y), z: f(bd.in_z),
      alive: Int32Array.from(bd.in_alive),
      m_solid: f(bd.in_m_solid), m_water: f(bd.in_m_water),
      m_char: f(bd.in_m_char), T_s: f(bd.in_T_s),
      m_water_0: f(bd.in_m_water_0), sav: f(bd.in_sav),
      m_char_max: f(bd.in_m_char_max),
    };
    const out = {
      S_pyro: new Float64Array(n), S_drying: new Float64Array(n),
      Q_pyro: new Float64Array(n), Q_drying: new Float64Array(n),
      Y_F_source: new Float64Array(n), Q_char: new Float64Array(n),
      Q_smold: new Float64Array(n), Q_g_conv: new Float64Array(n),
      nAliveOut: new Int32Array(1), nBurnedOut: new Int32Array(1),
      diagMaxOut: new Float64Array(16),
    };
    stepBedParticles(s,
      { nx: bd.nx, ny: bd.ny, nz: bd.nz, T_g: f(bd.T_g), Y_O2: f(bd.Y_O2),
        Q_solid_ext: f(bd.Q_solid_ext) },
      out,
      { dx: bd.dx, dy: bd.dy, dzArr: f(bd.dz_arr), zFace: f(bd.z_face),
        hConv: bd.h_conv, rhoSolidTrue: bd.rho_solid_true,
        cpSolid: bd.cp_solid, epsSolid: bd.eps_solid, tAmb: bd.T_amb,
        viewFactor: bd.view_factor,
        viewFactorGeometric: bd.view_factor_geometric,
        hBed: bd.h_bed, kappaBedEff: bd.kappa_bed_eff, dt: bd.dt,
        doDrying: true, doPyrolysis: true, doCharOx: true, doSmolder: true,
        dryingMode: bd.drying_mode, charOxFluxCapWm2: bd.char_ox_flux_cap,
        charOxAshExp: bd.char_ox_ash_exp, nPerCellForSplit: bd.n_per_cell });
    return [s.T_s, s.m_solid, s.m_water, s.m_char, out.S_pyro, out.Q_g_conv,
            out.diagMaxOut];
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
