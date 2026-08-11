/**
 * The outdoor 3D fire-spread solver — JS port of `run_3d_spread` in
 * model_outdoor/spread_3d.py.
 *
 * This is the piece that makes the other twelve modules a solver rather than a
 * pile of kernels. Everything here is ORDERING, and the ordering is
 * load-bearing: an operator-split fractional step (Chorin 1967) whose stages
 * were arrived at by fixing specific failures, several of which are recorded
 * in comments below because the code alone does not explain them.
 *
 * SCOPE. This runs the configuration the production Cheney decks specify:
 * Lagrangian bed, k-epsilon turbulence, EDC combustion, DOM radiation,
 * FFT-PCG projection, periodic y, log-law inlet. Deck options outside that set
 * — Smagorinsky LES, P1 radiation, the Eulerian bed, the FSD/PaSR closures,
 * SEM inlet turbulence, the Finney tendril and Lagrangian closures, cup-burner
 * boundaries, the Phase 24 moisture jump — are NOT implemented. `runSpread3D`
 * checks for them up front and throws with the offending option named. It does
 * not silently fall back, because a silent fallback here would produce a
 * plausible ROS from the wrong physics.
 *
 * THE STEP, in order:
 *
 *    0.  adaptive dt from CFL + diffusion, throttled
 *    1.  rho from the equation of state, BEFORE momentum sees it
 *    2.  bed particles -> S_pyro, Q_pyro, T_s mirror, horizontal conduction
 *    3.  drag
 *    4.  tentative momentum
 *    5.  projection, iterated to a divergence tolerance
 *    6.  outflow sponge
 *    7.  k-epsilon (+ wall function), tau_mix
 *    8.  DOM radiation (sub-cycled), soil conduction
 *    9.  ignition pulse
 *   10.  level-set masks, phi_flame, DOM forward flux
 *   11.  gas-energy advection
 *   12.  N_SUB x [chemistry -> species transport -> coupling -> conduction]
 *   13.  level-set evolution and reinit
 *   14.  EoS again, front tracking, exit checks
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i.
 */
import { buildGrid3D } from './mesh.js';
import { stepDragForce } from './drag.js';
import { stepTentativeVelocity } from './momentum.js';
import { ProjectionSolver3D } from './projection.js';
import {
  stepKEpsilon, C_MU, K_MIN, EPS_MIN,
  BETA_P_CANOPY_DEFAULT, BETA_D_CANOPY_DEFAULT,
} from './kepsilon.js';
import { applyTurbulentDiffusion, SC_T, PR_T } from './turbulentDiffusion.js';
import { DOMRadiationSolver } from './dom.js';
import { stepChemistryOdeEdc } from './edc.js';
import { stepSpeciesTransport } from './species.js';
import { stepSolidConductionVertical, K_SOLID_GRASS } from './solidConduction.js';
import {
  allocateBedParticleBuffers, initializeBedParticlesFromAlphaS,
  stepBedParticles, aggregateParticlesToTsGrid,
  aggregateParticlesToMLocalGrid, stepHorizontalSolidConductionScatter,
  DRY_MODE_ARRHENIUS, DRY_MODE_EQUILIBRIUM, DRY_MODE_COMBINED,
  RHO_SOLID_TRUE_GRASS, CP_SOLID_GRASS, SAV_GRASS_DEFAULT,
  EPS_SOLID_DEFAULT, T_BOIL_WATER, ALIVE_FALSE, ALIVE_TRUE,
} from './lagrangianBed.js';
import {
  LevelSetFront3D, computePhiFlameFromState, flameBodyMaskFromPhiFlame,
  computeQDomFwdAtBand, computeQInAtFront3d, computeVn3d, updateCellAge,
  L_BURNOUT_M, DX_VN_BAND_M,
} from './flameFront.js';
import { rosFromU10, blendResolvedEmpirical, A_CH } from '../cheney.js';
import {
  applyOutflowSponge, applyWallFunction,
  buildSoilGrid, stepSoilConduction, advGasEnergy,
  updateFrontTracking, computeSteadyRos,
} from './support.js';

// ── Constants (spread_3d module scope) ────────────────────────────────

const R_UNIV = 8.314;
const M_AIR = 0.02897;
const R_AIR = R_UNIV / M_AIR;
const P0 = 101325.0;             // [Pa] thermodynamic pressure, low-Mach
const G = 9.81;
const RHO_PARTICLE = 500.0;      // [kg/m^3] dry biomass particle density
const CP_SOLID = 1300.0;         // [J/kg/K] dry biomass heat capacity
const Z_0_INLET = 0.01;          // [m] short stubble, Monteith & Unsworth T4.1
const KAPPA_LOGLAW = 0.40;
const WAF = { open: 0.90, shrub: 0.60 };
const CHI_RAD_GRASS = 0.34;      // NIST TN 2314 (Sung 2025)

const N_SPONGE_SIGMA_MAX = 5.0;  // [1/s] ~0.2 s e-folding at the outlet
const Y_F_SPONGE_SKIP = 1.0e-3;
const N_SOIL = 6;

const CP_GAS_DRY = 1100.0;       // [J/kg/K] dry air at flame T
const CP_VAPOR = 2000.0;         // [J/kg/K] water vapour, NIST 1000-2000 K
const T_FLAME_AD = 10000.0;      // numerical safety only -- see note in step 12
const T_SURF_MAX = 10000.0;

const Q_DRIP_PER_AREA = 30000.0;      // [W/m^2] gas-side drip torch
const Q_DRIP_PER_AREA_BED = 30000.0;  // [W/m^2] solid-side, particle path
const F_DRIP_TO_SOLID = 0.80;
const Q_RAD_MAX_BED_DEFAULT = 1.0e5;  // [W/m^3] clamp on q_rad into particles
const Q_IGNITION_PULSE = 240000.0;    // [W/m^2] Phase 15L kick intensity

/** Midflame wind from the 10 m reference (Rothermel 1972 convention). */
export function midflameWindSpeed(u10, terrain = 'open') {
  return u10 * (WAF[String(terrain).toLowerCase()] ?? WAF.open);
}

/**
 * Log-law wind over a rough surface: u(z) = (u_tau/kappa) ln((z+z_0)/z_0),
 * with u_tau set so u(z_ref) = U_ref. Zero at z = 0 exactly.
 *
 * This is the profile at the INLET, which sits upstream of the bed over bare
 * ground — so it is an atmospheric BL, not an in-canopy profile. Imposing a
 * canopy-equilibrium profile here (which has u(0) far from zero) artificially
 * raised the in-bed wind at the bed leading edge and contributed to advective
 * washout in sparse Nat-pasture cases. Inside the bed, porous drag attenuates
 * the wind organically.
 */
export function windProfileLogLaw(z, URef, zRef = 10.0, z0 = Z_0_INLET) {
  if (z <= 0.0 || URef <= 0.0 || zRef <= 0.0) return 0.0;
  const uTauOverKappa = URef / Math.log((zRef + z0) / z0);
  return Math.max(0.0, uTauOverKappa * Math.log((z + z0) / z0));
}

const UNSUPPORTED = [
  ['turbulence_model', (v) => v !== undefined && v !== 'k_epsilon',
   "only 'k_epsilon' is ported; Smagorinsky LES is not"],
  ['radiation_solver', (v) => v !== undefined && v !== 'dom',
   "only 'dom' is ported; the P1 solver is not"],
  ['combustion_closure', (v) => v !== undefined && v !== 'edc',
   "only 'edc' is ported; FSD / PaSR / ebu_bootstrap / 2-step are not"],
  ['projection_method', (v) => v !== undefined && v !== 'fft_pcg',
   "only 'fft_pcg' is ported; PARDISO and AMG are not"],
  ['boundary_condition_kind', (v) => v !== undefined && v !== 'outdoor_wind',
   "only 'outdoor_wind' is ported; the cup burner is not"],
  ['lagrangian_bed_enable', (v) => v === false,
   'the Eulerian bed path is not ported (and gives -19% ROS upstream)'],
  ['sem_enable', (v) => v === true, 'SEM inlet turbulence is not ported'],
  ['finney_tendril_enable', (v) => v === true, 'the Finney tendril closure is not ported'],
  ['finney_lagrangian_enable', (v) => v === true, 'the Finney Lagrangian closure is not ported'],
  ['finney_burst_enable', (v) => v === true, 'the Finney burst closure is not ported'],
  ['moisture_jump_enable', (v) => v === true, 'the Phase 24 moisture jump is not ported'],
  ['volume_weighted_projection', (v) => v === true,
   'the volume-weighted projection is not ported (opt-in upstream, off by default)'],
];

function rejectUnsupported(cfg) {
  const bad = [];
  for (const [key, isBad, why] of UNSUPPORTED) {
    if (isBad(cfg[key])) bad.push(`  ${key} = ${JSON.stringify(cfg[key])} — ${why}`);
  }
  if (bad.length) {
    throw new Error(
      'runSpread3D: the deck asks for options this port does not implement.\n' +
      bad.join('\n') +
      '\nNothing falls back silently: a fallback here would produce a ' +
      'plausible ROS from the wrong physics.');
  }
}

/**
 * Run a spread case to completion.
 *
 * @param {object} cfg deck-equivalent configuration
 * @param {function} [onStep] called as (info) once per step, for the applet's
 *        render loop. Return false to stop early.
 * @returns {object} {rosMs, rosMMin, frontT, frontX, grid, state, steps, t}
 */
export function runSpread3D(cfg, onStep = null) {
  rejectUnsupported(cfg);

  const {
    Lx, Ly, Lz, dx, dy = null, nZBed,
    hBed, rhoB, sigmaSav = 6562.0, canopyCd = 0.30,
    initialMoistureFrac = 0.0,
    windSpeedMs, terrain = 'open',
    bedXStart = 0.0, bedXEnd = null,
    TAmb = 300.0,
    cflFactor = 0.40, minDtS = 1.0e-4, maxWallTimeS = 15.0,
    ignitionDurationS = 30.0, ignitionQMult = 1.0, ignitionWidthMult = 1.0,
    ignitionTPinEnable = false, ignitionTPinK = 1500.0,
    ignitionTPinHeightM = 0.30, ignitionTPinRampS = 0.5,
    ignitionTPinInBed = false,
    solidPhaseIgnitionEnable = false, solidPhaseIgnitionTsK = 1000.0,
    lagrangianBedNPerCell = 20,
    lagrangianBedDryingMode = 'combined',
    lagrangianBedHConv = 25.0,
    lagrangianBedRhoSolidTrue = RHO_SOLID_TRUE_GRASS,
    lagrangianBedCpSolid = CP_SOLID_GRASS,
    lagrangianBedEpsSolid = EPS_SOLID_DEFAULT,
    lagrangianBedViewFactor = 1.0,
    lagrangianBedViewFactorGeometric = false,
    lagrangianBedDoDrying = true, lagrangianBedDoPyrolysis = true,
    lagrangianBedDoCharOx = true, lagrangianBedDoSmolder = true,
    charOxFluxCapWm2 = 1.0e5, charOxAshExp = 0.0,
    wallFunction = false,
    domSubcycleEvery = 5,
    projectionCgRtol = 1.0e-6, projMaxIter = 50, projDivTol = 1.0e-3,
    levelSetPassive = false,
    nSub = 10,
    chiRad = CHI_RAD_GRASS,
    hConvMult = 1.0,
    canopyBetaP = BETA_P_CANOPY_DEFAULT,
    canopyBetaD = BETA_D_CANOPY_DEFAULT,
    // Phase 19/20 empirical-ROS hybrid. Off by default, matching upstream.
    empiricalRosEnable = false,
    empiricalRosACh = A_CH.natural,
    empiricalRosUThresholdMs = 3.5,
    empiricalRosBlendWidthMs = 1.0,
    // Cap on radiant power delivered into bed particles [W/m^3].
    //
    // Upstream hardcodes 1e5 with a NUMERICAL justification ("the rate the
    // particles can absorb without coupling-rate instability"). Measured, it
    // binds on 8-16% of bed cells at up to 283x over -- exactly the cells at
    // the front, where radiation is strongest.
    //
    // Integrated over the bed depth, 1e5 W/m^3 x 0.10 m caps absorbed flux at
    // 10 kW/m^2. Measured flux at a grass-fire front is 20-100 kW/m^2 (Anderson
    // 1969; Frankman 2013) and piloted ignition of cured grass needs roughly
    // 10-20 kW/m^2 sustained. So the cap pins the bed at or below the ignition
    // threshold by construction -- which is survivable above U~4 where
    // convection makes up the difference, and fatal below it where radiation is
    // the only forward mechanism.
    //
    // Exposed so that hypothesis is testable. Default unchanged.
    qRadMaxBedWm3 = Q_RAD_MAX_BED_DEFAULT,
    // Diagnostic only -- see edc.js. 0 disables (reference behaviour).
    laminarFloorSL = 0.0,
    // Opt-in corrections to two energy-balance inconsistencies in the
    // radiation/bed coupling (SOLVER_PORT.md 7.8, defects 3 and 4):
    //   - feed the bed the ABSORPTION-only channel, not the net, since the
    //     particle kernel applies its own Stefan-Boltzmann loss on top
    //   - weight each particle's share of that absorption by its own f_geom,
    //     normalised per cell, so absorption and emission carry the same
    //     geometric factor (Kirchhoff reciprocity)
    // Default off: reference behaviour is preserved bit for bit.
    radiationFixes = false,
  } = cfg;

  // ── Grid + state ────────────────────────────────────────────────────
  const grid = buildGrid3D({
    Lx, Ly, Lz, dx, dy, hBed, nZBed,
    dzExpansion: cfg.dzExpansion ?? 1.0,
    dzFirst: cfg.dzFirst ?? null,
    blGrowth: cfg.blGrowth ?? 1.0,
    wallBlN: cfg.wallBlN ?? 0,
    wallBlFirstDz: cfg.wallBlFirstDz ?? 0.0,
    wallBlGrowth: cfg.wallBlGrowth ?? 1.3,
    atmMaxDz: cfg.atmMaxDz ?? null,
    atmGrowth: cfg.atmGrowth ?? 1.3,
    atmUniformDz: cfg.atmUniformDz ?? null,
  });
  const { nx, ny, nz, nZBed: nZB } = grid;
  const n = nz * ny * nx;
  const nxy = ny * nx;
  const shape = { nx, ny, nz };
  const zero = () => new Float64Array(n);

  const state = {
    u: zero(), v: zero(), w: zero(),
    rho: new Float64Array(n).fill(P0 / (R_AIR * TAmb)),
    T_g: new Float64Array(n).fill(TAmb),
    T_s: new Float64Array(n).fill(TAmb),
    Y_fuel: zero(), Y_O2: new Float64Array(n).fill(0.232), Y_H2O: zero(),
    alpha_s: zero(),
  };

  // ── Bed initialisation ──────────────────────────────────────────────
  const bedXEndUse = bedXEnd === null ? Lx : bedXEnd;
  let iBedStart = 0;
  while (iBedStart < nx && grid.xMid[iBedStart] < bedXStart) iBedStart++;
  let iBedEnd = 0;
  while (iBedEnd < nx && grid.xMid[iBedEnd] <= bedXEndUse) iBedEnd++;
  if (iBedStart >= iBedEnd) {
    throw new Error(
      `bed_x_start=${bedXStart} >= bed_x_end=${bedXEndUse} after grid snapping ` +
      `(i_bed_start=${iBedStart}, i_bed_end=${iBedEnd})`);
  }
  // alpha_s is the bed-volume-averaged SOLID FRACTION. rho_b is a BULK
  // density that already accounts for porosity, so the two are related by
  // the particle density and must not be multiplied together downstream.
  const alphaSAvg = rhoB / RHO_PARTICLE;
  for (let k = 0; k < nZB; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = iBedStart; i < iBedEnd; i++) state.alpha_s[k * nxy + j * nx + i] = alphaSAvg;
    }
  }

  // ── Lagrangian bed particles ────────────────────────────────────────
  const dryModeMap = {
    arrhenius: DRY_MODE_ARRHENIUS,
    equilibrium: DRY_MODE_EQUILIBRIUM,
    combined: DRY_MODE_COMBINED,
  };
  const dryingMode = dryModeMap[lagrangianBedDryingMode];
  if (dryingMode === undefined) {
    throw new Error(`lagrangian_bed_drying_mode=${lagrangianBedDryingMode} not in ` +
                    `{arrhenius, equilibrium, combined}`);
  }
  const nMaxParticles = nZB * ny * (iBedEnd - iBedStart) * lagrangianBedNPerCell;
  const bed = allocateBedParticleBuffers(nMaxParticles);
  const nAlloc = initializeBedParticlesFromAlphaS(
    bed, state.alpha_s, rhoB, initialMoistureFrac, TAmb,
    grid.dx, grid.dy, grid.dzArr, nZB, lagrangianBedNPerCell,
    { ...shape, sav: sigmaSav, iLo: iBedStart, iHi: iBedEnd });

  // ── Ignition source strip ───────────────────────────────────────────
  const srcWidthM = 0.5 * ignitionWidthMult;
  const nSrc = Math.max(1, Math.round(srcWidthM / grid.dx));
  const iSrcStart = iBedStart;
  const iSrcEnd = Math.min(iBedEnd, iSrcStart + nSrc);

  // Solid-phase ignition: pre-heat source-patch PARTICLES rather than pinning
  // the gas. The 1500 K gas pin injects ~250 kJ/m^2 of artificial gas-side
  // energy, enough to drive a plume that REVERSES the bed wind at low U —
  // that was the low-U plume-reversal artifact. Seeding the solid lets the
  // gas warm through particle convection and volatile injection instead,
  // which is how a torch-flash igniter actually works.
  if (solidPhaseIgnitionEnable) {
    const xLo = grid.xMid[iSrcStart] - 0.5 * grid.dx;
    const xHi = grid.xMid[iSrcEnd - 1] + 0.5 * grid.dx;
    for (let p = 0; p < nAlloc; p++) {
      if (bed.alive[p] === ALIVE_FALSE) continue;
      if (bed.x[p] >= xLo && bed.x[p] < xHi) bed.T_s[p] = solidPhaseIgnitionTsK;
    }
  }

  // ── Inlet profile ───────────────────────────────────────────────────
  const uInlet = new Float64Array(nz * ny);
  // v and w inlet ghosts are zero (no cross-stream or vertical inflow). They
  // exist as arrays rather than a null because the momentum kernel reads them
  // unconditionally at i = 0 -- SEM inlet turbulence writes into them upstream.
  const vInlet = new Float64Array(nz * ny);
  const wInlet = new Float64Array(nz * ny);
  for (let k = 0; k < nz; k++) {
    const uz = windProfileLogLaw(grid.zMid[k], windSpeedMs);
    for (let j = 0; j < ny; j++) uInlet[k * ny + j] = uz;
  }
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      const uz = uInlet[k * ny + j];
      for (let i = 0; i < nx; i++) state.u[k * nxy + j * nx + i] = uz;
    }
  }

  // ── Sponge profile ──────────────────────────────────────────────────
  // Quadratic ramp over the last cells (Israeli & Orszag 1981 — a quadratic
  // ramp avoids the reflection a step change causes). Capped at Nx/4 so small
  // domains keep a usable interior.
  const nSponge = Math.max(3, Math.min(Math.round(0.5 / grid.dx), Math.floor(nx / 4)));
  const sigmaXSponge = new Float64Array(nx);
  for (let i = nx - nSponge; i < nx; i++) {
    const frac = (i - (nx - nSponge)) / nSponge;
    sigmaXSponge[i] = N_SPONGE_SIGMA_MAX * frac * frac;
  }

  // ── Turbulence, radiation, soil, level set ──────────────────────────
  const UMf = midflameWindSpeed(windSpeedMs, terrain);
  const delta = Math.max(grid.dx, Math.max(grid.dy, grid.dz));
  const tauMixBuoy = Math.sqrt((2.0 * delta) / G);
  const tauMix = new Float64Array(n).fill(tauMixBuoy);
  // k and eps back-calculated from nu_t = 0.01 with k = 1.5*(I_t*U)^2 at
  // I_t = 0.10, the atmospheric turbulence intensity (Garratt 1992).
  const kInit = Math.max(1.5 * Math.pow(0.10 * Math.max(UMf, 0.5), 2), 1.0e-4);
  const epsInit = (C_MU * kInit * kInit) / 0.01;
  const kTurb = new Float64Array(n).fill(kInit);
  const epsTurb = new Float64Array(n).fill(epsInit);
  const nuT = new Float64Array(n).fill(0.01);
  const SMag2 = zero(), OmegaMag2 = zero();
  const kWallGhost = new Float64Array(nxy).fill(K_MIN);
  const epsWallGhost = new Float64Array(nxy).fill(EPS_MIN);

  const radSolver = new DOMRadiationSolver({
    nz, ny, nx, dx: grid.dx, dy: grid.dy, dzArr: grid.dzArr,
  });
  const qRad = zero(), qRadGas = zero();
  const qRadAbs = radiationFixes ? zero() : null;
  const soil = buildSoilGrid(N_SOIL);
  const TSoil = new Float64Array(N_SOIL * nxy).fill(TAmb);
  const qInSoil = new Float64Array(nxy);
  const TSoilSurface = new Float64Array(nxy).fill(TAmb);

  const lset = new LevelSetFront3D({
    nz, ny, nx, dx: grid.dx, dy: grid.dy, dzArr: grid.dzArr,
    LBurnout: L_BURNOUT_M,
  });
  lset.initializeSourcePatch(iSrcStart, iSrcEnd, nZB - 1, grid.xMid);
  // Fixed ahead-band, floored at dx. Plume tilt with wind is captured by the
  // state-derived phi_flame, NOT by an external Albini geometric formula —
  // that is why flameTiltBandM exists but is not called here.
  const bandMTilt = Math.max(grid.dx, DX_VN_BAND_M);

  const proj = new ProjectionSolver3D({
    nz, ny, nx, dx: grid.dx, dy: grid.dy, dzArr: grid.dzArr,
    dFaceAbove: grid.dFaceAbove, dFaceBelow: grid.dFaceBelow,
    method: 'fft_pcg', cgRtol: projectionCgRtol,
  });
  proj.setInletBC(uInlet);

  // ── Work arrays ─────────────────────────────────────────────────────
  const Fx = zero(), Fy = zero(), Fz = zero();
  const SPyro = zero(), QPyro = zero(), QComb = zero(), omega = zero();
  const bedSp = zero(), bedSd = zero(), bedQp = zero(), bedQd = zero();
  const bedYFs = zero(), bedQch = zero(), bedQsm = zero(), bedQgc = zero();
  const bedQsx = zero(), bedMLocal = zero();
  const nAliveOut = new Int32Array(1), nBurnedOut = new Int32Array(1);
  const diagMax = new Float64Array(16);
  const rhoPrev = Float64Array.from(state.rho);
  const divTarget = zero();
  const qFrankman = zero(), qDomFwd = zero();
  const vnField = zero();
  const cellAge = new Float64Array(n).fill(Infinity);
  const SFeff = zero(), SO2eff = zero(), SH2Oeff = zero();
  const TIgn = TAmb + 300.0;

  // Seed the front history at the SOURCE-PATCH front, not at (0, 0).
  // Starting from zero makes computeSteadyRos take its slope from the origin,
  // which counts the source patch itself as fire spread -- on step 1 x jumps
  // from 0 to the patch edge and that whole jump enters the numerator. On this
  // case it inflates ROS by 2.2x. (Phase 17b ROS fix.)
  const initialFrontX = (nZB > 0 && iSrcEnd > iSrcStart)
    ? grid.xMid[iSrcEnd - 1] + 0.5 * grid.dx
    : 0.0;
  const frontT = [0.0], frontX = [initialFrontX];

  // ── ROS_Ts: the truthful spread metric ──────────────────────────────
  // The level-set front is NOT the answer when the level set is passive --
  // which is the canonical high-wind configuration. There v_n is zero by
  // design, the marker never moves, and a level-set-derived ROS reads ~0
  // whether the fire is spreading or extinct. Measured on 2026-08-10: 0.31
  // m/min from the level set while the bed front advanced 5.5 m at 27.78.
  //
  // So track the SOLID front too, by the same rule the validation workers use
  // (_cheney_phase16_worker.py): the furthest column with any cell at or above
  // 600 K, over the full z-column -- not bed-only, and no alpha_s filter.
  const T_IGN_TS = 600.0;
  const tsT = [], tsX = [];
  const tsFrontX = () => {
    let iMax = -1;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = iMax + 1; i < nx; i++) {
          if (state.T_s[k * nxy + j * nx + i] >= T_IGN_TS && i > iMax) iMax = i;
        }
      }
    }
    return iMax < 0 ? NaN : grid.xMid[iMax];
  };

  // ── Phase 19/20 empirical-ROS hybrid ────────────────────────────────
  // Scalars, computed once: the inflow wind is constant in these cases. A
  // dynamic-wind extension would move this inside the loop.
  const empiricalRosMs = empiricalRosEnable
    ? rosFromU10(windSpeedMs, initialMoistureFrac, empiricalRosACh) : 0.0;
  const empiricalBlendW = empiricalRosEnable
    ? blendResolvedEmpirical(windSpeedMs, empiricalRosUThresholdMs,
                             empiricalRosBlendWidthMs) : 0.0;
  let empiricalSeedCounter = 0;
  let dzMin = grid.dzArr[0];
  for (let k = 1; k < nz; k++) if (grid.dzArr[k] < dzMin) dzMin = grid.dzArr[k];

  const UChar = Math.max(UMf, Math.max(Math.sqrt(G * hBed * 0.5), 0.1));
  let dt = Math.min(
    (cflFactor * Math.min(grid.dx, Math.min(grid.dy, dzMin))) / UChar,
    Math.min((0.25 * dzMin * dzMin) / Math.max(0.01, 1.0e-3), 0.05));
  if (dt <= 0.0) throw new Error(`computed dt=${dt} <= 0 (check CFL params)`);

  // Optional per-stage timing. Off by default -- the clock calls are cheap but
  // not free, and this is a diagnostic, not part of the solve. Enabled with
  // cfg.profile = true; the totals come back on the result as `timings`.
  const prof = cfg.profile ? Object.create(null) : null;
  const now = (typeof performance !== 'undefined' && performance.now)
    ? () => performance.now() : () => Number(process.hrtime.bigint()) / 1e6;
  let _tm = 0;
  const tic = prof ? () => { _tm = now(); } : () => {};
  const toc = prof ? (k) => { prof[k] = (prof[k] || 0) + (now() - _tm); } : () => {};

  let t = 0.0;
  let step = 0;
  let vnExtinctCount = 0;
  let lastFrontX = 0.0;
  const bedXEndActual = grid.xMid[iBedEnd - 1];

  // ── Time loop ───────────────────────────────────────────────────────
  while (t < maxWallTimeS) {
    step += 1;

    // 0. Adaptive dt, recomputed from CURRENT state. Fixing dt once from
    //    cold-flow state under-counts CFL badly once combustion fires: plume
    //    buoyancy spikes |u| 3-5x and nu_t by 10x or more.
    let uNow = 0.0;
    for (let c = 0; c < n; c++) {
      const a = Math.abs(state.u[c]); if (a > uNow) uNow = a;
      const b = Math.abs(state.v[c]); if (b > uNow) uNow = b;
      const d = Math.abs(state.w[c]); if (d > uNow) uNow = d;
    }
    uNow = Math.max(uNow, Math.max(UMf, Math.max(Math.sqrt(G * hBed * 0.5), 0.1)));
    let nuNow = 1.0e-3;
    for (let c = 0; c < n; c++) if (nuT[c] > nuNow) nuNow = nuT[c];
    let dtNew = Math.min(
      (cflFactor * Math.min(grid.dx, Math.min(grid.dy, dzMin))) / uNow,
      Math.min((0.25 * dzMin * dzMin) / nuNow, 0.05));
    // Never grow by more than 1.5x (avoids oscillation), but shrink freely —
    // ignition needs to be able to drop dt fast.
    dtNew = Math.min(dtNew, dt * 1.5);
    if (dtNew < minDtS) dtNew = minDtS;   // floor: CFL is violated here, but
                                          // that beats chasing microsecond
                                          // transients for hours. Real
                                          // explosions are caught below.
    let TgMax = 0.0;
    for (let c = 0; c < n; c++) if (state.T_g[c] > TgMax) TgMax = state.T_g[c];
    if (!Number.isFinite(TgMax)) {
      throw new Error(
        `State NaN/Inf at t=${t.toFixed(3)}s — simulation exploded. ` +
        `(adaptive dt: ${(dtNew * 1e6).toFixed(2)} us, u_max=${uNow.toFixed(2)} m/s, ` +
        `nu_t_max=${nuNow.toExponential(3)} m2/s)`);
    }
    dt = dtNew;

    rhoPrev.set(state.rho);

    // Diagnostic gas pin, opt-in and off by default. Ramped rather than
    // stepped: an instantaneous cold-to-1500 K jump is a 5x density shock in
    // one cell. np.maximum semantics — it only RAISES T_g, so real combustion
    // can exceed it without being clamped back down.
    if (ignitionTPinEnable && t < ignitionDurationS) {
      let kLo, kHi;
      if (ignitionTPinInBed) { kLo = 0; kHi = nZB; }
      else {
        kLo = nZB;
        const zTop = hBed + ignitionTPinHeightM;
        kHi = kLo;
        for (let k = kLo; k < nz; k++) {
          if (grid.zFace[k + 1] > zTop) break;
          kHi = k + 1;
        }
      }
      if (kHi > kLo) {
        const fRamp = Math.min(1.0, t / Math.max(ignitionTPinRampS, 1e-6));
        const TPin = TAmb + fRamp * (ignitionTPinK - TAmb);
        for (let k = kLo; k < kHi; k++) {
          for (let j = 0; j < ny; j++) {
            for (let i = iSrcStart; i < iSrcEnd; i++) {
              const c = k * nxy + j * nx + i;
              if (state.T_g[c] < TPin) state.T_g[c] = TPin;
            }
          }
        }
      }
    }

    // 1. EoS BEFORE momentum and projection. If rho still reflects the old
    //    T_g, the projection enforces continuity against a stale density and
    //    mass conservation breaks for large T jumps. The chemistry's own T_g
    //    changes land later in the step and are picked up by the NEXT step's
    //    drho_dt — a one-step lag, acceptable for slow combustion changes.
    for (let c = 0; c < n; c++) {
      state.rho[c] = P0 / (R_AIR * Math.max(state.T_g[c], TAmb));
    }
    // Fresh air at the inlet. Without this the upwind scheme lets
    // combustion-vitiated air back-flow in and the open boundary stops
    // conserving mass.
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) state.Y_O2[k * nxy + j * nx] = 0.232;
    }

    // 2. Bed particles. Replaces the Eulerian drying + pyrolysis + char-ox +
    //    smoulder kernels with one per-particle pass emitting cell sources.
    bedQsx.fill(0.0);
    if (t < ignitionDurationS) {
      const qDripBedVol = Q_DRIP_PER_AREA_BED / Math.max(hBed, 1e-3);
      for (let k = 0; k < nZB; k++) {
        for (let j = 0; j < ny; j++) {
          for (let i = iSrcStart; i < iSrcEnd; i++) {
            bedQsx[k * nxy + j * nx + i] += F_DRIP_TO_SOLID * qDripBedVol;
          }
        }
      }
    }
    // Radiation into the bed, from the PREVIOUS step's DOM. This is the
    // forward-spread mechanism: hot bed radiates, cells ahead absorb,
    // particles heat, pyrolyse, release Y_F, the front advances.
    // The clamp is not cosmetic — q_rad reaches 2.4e7 W/m^3 at the bed top
    // and uncapped it produces a step-2 spike the coupling cannot integrate.
    const qRadForBed = radiationFixes ? qRadAbs : qRad;
    for (let k = 0; k < nZB; k++) {
      const invDz = 1.0 / grid.dzArr[k];
      for (let s = 0; s < nxy; s++) {
        let v = qRadForBed[k * nxy + s] * invDz;
        if (v > qRadMaxBedWm3) v = qRadMaxBedWm3;
        else if (v < -qRadMaxBedWm3) v = -qRadMaxBedWm3;
        bedQsx[k * nxy + s] += v;
      }
    }

    tic();
    stepBedParticles(bed,
      { ...shape, T_g: state.T_g, Y_O2: state.Y_O2, Q_solid_ext: bedQsx },
      { S_pyro: bedSp, S_drying: bedSd, Q_pyro: bedQp, Q_drying: bedQd,
        Y_F_source: bedYFs, Q_char: bedQch, Q_smold: bedQsm, Q_g_conv: bedQgc,
        nAliveOut, nBurnedOut, diagMaxOut: diagMax },
      { dx: grid.dx, dy: grid.dy, dzArr: grid.dzArr, zFace: grid.zFace,
        hConv: lagrangianBedHConv, rhoSolidTrue: lagrangianBedRhoSolidTrue,
        cpSolid: lagrangianBedCpSolid, epsSolid: lagrangianBedEpsSolid,
        tAmb: TAmb, viewFactor: lagrangianBedViewFactor,
        viewFactorGeometric: lagrangianBedViewFactorGeometric,
        hBed,
        // kappa ~ sav * alpha_s with alpha_s = rho_b/rho_solid_true. For
        // grass that is 2000 * 1.07/380 ~ 5.6 1/m, so kappa*h_bed ~ 2.1 and
        // the deepest particles emit ~12% of what surface particles do.
        kappaBedEff: (sigmaSav * rhoB) / lagrangianBedRhoSolidTrue,
        dt,
        doDrying: lagrangianBedDoDrying, doPyrolysis: lagrangianBedDoPyrolysis,
        doCharOx: lagrangianBedDoCharOx, doSmolder: lagrangianBedDoSmolder,
        dryingMode, charOxFluxCapWm2, charOxAshExp,
        absorbGeometric: radiationFixes,
        nPerCellForSplit: lagrangianBedNPerCell });
    toc('bed');

    // S_pyro drives the projection's mass source, so it is volatile + vapour.
    // Y_fuel's source is volatile ONLY -- see step 12.
    for (let c = 0; c < n; c++) SPyro[c] = bedSp[c] + bedSd[c];
    // Net solid heat sink, Eulerian sign convention: endothermic pyrolysis and
    // drying positive, exothermic char-ox and smoulder negative.
    for (let c = 0; c < n; c++) QPyro[c] = bedQp[c] + bedQd[c] - bedQch[c] - bedQsm[c];

    // Mirror particle T_s into the grid. DOM reads state.T_s for its
    // K_emit*sigma*T_s^4 emission, so without this the bed never radiates
    // forward and the fire stalls at the ignition patch.
    aggregateParticlesToTsGrid(bed.x, bed.y, bed.z, bed.alive,
      bed.m_solid, bed.m_water, bed.m_char, bed.T_s,
      grid.dx, grid.dy, grid.zFace, state.T_s, TAmb, shape);
    stepHorizontalSolidConductionScatter(bed.x, bed.y, bed.z, bed.alive,
      bed.m_solid, bed.m_water, bed.m_char, bed.T_s,
      state.T_s, state.alpha_s, grid.dx, grid.dy, grid.zFace,
      K_SOLID_GRASS, lagrangianBedRhoSolidTrue, lagrangianBedCpSolid,
      nZB, dt, shape);

    // 3. Drag
    tic();
    stepDragForce(state.u, state.v, state.w, state.rho, state.alpha_s,
      sigmaSav, Fx, Fy, Fz, canopyCd);
    toc('drag');

    // 4. Tentative momentum
    tic();
    stepTentativeVelocity(state.u, state.v, state.w, state.rho, state.T_g,
      Fx, Fy, Fz, dt, grid.dx, grid.dy, grid.dzArr,
      grid.dFaceAbove, grid.dFaceBelow, TAmb, uInlet, vInlet, wInlet, shape);
    toc('momentum');

    // 5. Projection. div_target = (S_pyro - drho_dt)/rho: the low-Mach mass
    //    source. Without the S_pyro term a strict-incompressible projection
    //    silently discards the gas mass pyrolysis added, and fuel accumulates
    //    locally instead of expanding outward. The drho_dt term carries gas
    //    expansion from heating; without it, sudden T jumps NaN'd the
    //    cold-bed cases once the projection was cleaned up (the residual had
    //    been acting as accidental damping).
    for (let c = 0; c < n; c++) {
      const drhoDt = (state.rho[c] - rhoPrev[c]) / dt;
      divTarget[c] = (SPyro[c] - drhoDt) / Math.max(state.rho[c], 0.1);
    }
    tic();
    proj.rebuildForRho(state.rho);
    let projDivMax = 0.0;
    let projNIter = 0;
    for (let it = 0; it < projMaxIter; it++) {
      proj.project(state.u, state.v, state.w, state.rho, dt, divTarget);
      // Iterated because the BCs are NOT consistent with the projection's
      // discrete operator, so one solve leaves boundary divergence that
      // advection then carries inward. Iterating drives it under tolerance
      // regardless of which BC is responsible (FDS Tech Ref §6.3).
      const divNow = proj.divergence(state.u, state.v, state.w);
      projDivMax = 0.0;
      for (let c = 0; c < n; c++) {
        const d = Math.abs(divNow[c] - divTarget[c]);
        if (d > projDivMax) projDivMax = d;
      }
      projNIter = it + 1;
      if (projDivMax < projDivTol) break;
    }
    toc('projection');

    // 6. Outflow sponge
    applyOutflowSponge(state.u, uInlet, sigmaXSponge, state.Y_fuel,
      Y_F_SPONGE_SKIP, dt, shape);

    // 7. k-epsilon. nu_t feeds tau_mix = k/eps in the combustion closure and
    //    adds turbulent diffusion to the species, which is what lifts
    //    volatiles out of the bed into the flame body.
    if (wallFunction) {
      applyWallFunction(state.u, state.v, state.rho, state.alpha_s, grid.dzArr,
        kWallGhost, epsWallGhost, { ...shape, kMin: K_MIN, epsMin: EPS_MIN });
    }
    tic();
    stepKEpsilon(kTurb, epsTurb, nuT, state.u, state.v, state.w, state.T_g,
      state.rho, state.alpha_s, sigmaSav, dt, grid.dx, grid.dy, grid.dzArr,
      grid.dFaceAbove, grid.dFaceBelow, TAmb, SMag2, OmegaMag2, uInlet,
      kWallGhost, epsWallGhost, canopyBetaP, canopyBetaD, shape);
    toc('kepsilon');
    // tau_mix has NO upper bound, deliberately. Capping it (the old 1.0 s
    // limit) is non-standard against M&H 1977 / FDS / FireFOAM, which let it
    // grow without limit in laminar zones so omega_EBU falls to zero there and
    // Arrhenius chemistry takes over — which is the right limiter when there
    // is no turbulence. The LOWER clamp at the buoyancy timescale keeps
    // omega_EBU finite while k-eps is still spinning up.
    for (let c = 0; c < n; c++) {
      const tm = kTurb[c] / Math.max(epsTurb[c], 1e-12);
      tauMix[c] = tm > tauMixBuoy ? tm : tauMixBuoy;
    }

    // 8. DOM, sub-cycled. The radiation field changes slowly next to
    //    advection, so solving every K steps and reusing the cached arrays is
    //    standard (FDS Tech Ref Vol.1 §6.2; Howell 2010 §17). At K=5 radiation
    //    drops from 25% of the loop to about 5%.
    tic();
    if (step % Math.max(domSubcycleEvery, 1) === 0) {
      aggregateParticlesToMLocalGrid(bed.x, bed.y, bed.z, bed.alive,
        bed.m_solid, bed.m_water, grid.dx, grid.dy, grid.zFace, bedMLocal, shape);
      for (let s = 0; s < nxy; s++) TSoilSurface[s] = TSoil[s];
      radSolver.solve({
        Ts: state.T_s, Tg: state.T_g, alphaS: state.alpha_s,
        omegaComb: omega, sigmaSav, Tamb: TAmb,
        qRadSolidOut: qRad, qRadGasOut: qRadGas, qRadSolidAbsOut: qRadAbs,
        TsoilSurface: TSoilSurface, qInSoilOut: qInSoil,
        YH2O: state.Y_H2O, rho: state.rho, bedMoisturePerCell: bedMLocal,
      });
    }
    stepSoilConduction(TSoil, qInSoil, dt, soil.soilDz, soil.dAbove, soil.dBelow,
      { nx, ny, nSoil: N_SOIL, Tamb: TAmb });
    toc('radiation+soil');

    // 9. Ignition pulse as an external FLUX on the top bed layer, not a T_s
    //    clamp. Quintiere (2006) §7.4: piloted ignition is a fixed external
    //    heat flux, and the coupling then computes dT_s from the full energy
    //    balance. Clamping T_s to T_ign instead drove pyrolysis runaway.
    if (t < ignitionDurationS) {
      const kTopBed = nZB - 1;
      for (let j = 0; j < ny; j++) {
        for (let i = iSrcStart; i < iSrcEnd; i++) {
          qRad[kTopBed * nxy + j * nx + i] += Q_IGNITION_PULSE * ignitionQMult;
        }
      }
    }

    // 10. The two level sets.
    const aheadBand = lset.aheadBandMask(bandMTilt);
    // phi_bed is z-uniform by construction, so its 3D ahead-band is also true
    // in the atmosphere above the bed. Column sums downstream would then count
    // atmospheric absorption as if it heated the bed, which is mesh-runaway
    // (more atm cells per column at finer meshes = more contributions). Mask
    // above the bed top so q_in stays a bed-only integral.
    for (let k = nZB; k < nz; k++) aheadBand.fill(0, k * nxy, (k + 1) * nxy);
    tic();
    const phiFlame = computePhiFlameFromState(omega, state.T_g, state.Y_fuel,
      grid.dx, grid.dy, grid.dzArr, shape);
    const flameBody = flameBodyMaskFromPhiFlame(phiFlame, 0.0);
    // Frankman flame-tip convection stays zeroed: a phenomenological non-local
    // shortcut that double-counted the ordinary gas-solid coupling. v_n is
    // driven by DOM forward intensity alone.
    qFrankman.fill(0.0);
    computeQDomFwdAtBand(radSolver, aheadBand, qDomFwd);
    toc('levelset_masks');

    // 11. Gas-energy advection. The coupling kernel only applies point-wise
    //     sources to T_g; heat has to convect downstream separately.
    tic();
    advGasEnergy(state.T_g, state.u, state.v, state.w, dt, grid.dx, grid.dy,
      grid.dzArr, grid.dFaceAbove, grid.dFaceBelow, 2.0e-5, TAmb, shape);
    toc('gas_advection');

    // 12. Sub-stepped chemistry + coupling (Strang 1968 splitting). The
    //     combustion source is stiff: at hot T_g with fuel present, Q_comb
    //     reaches hundreds of MW/m^3, which would move T_g hundreds of K in
    //     one outer step. Pyrolysis is frozen across the sub-loop (S_pyro is a
    //     rate, already integrated over dt).
    const dtSub = dt / nSub;
    tic();
    for (let sub = 0; sub < nSub; sub++) {
      // NOT COMPUTED HERE, and that is not an omission.
      //
      // Upstream this sub-step begins by filling omega_O2 via
      // step_o2_supply_rate and building omega_max_T (the Damkohler
      // turbulent-flame-speed cap), then passes them plus tau_mix into
      // chemistry_closures.run(). The EDC closure's signature ends in
      // `**_unused`, and tau_mix / omega_O2 / omega_max_T are not among its
      // named parameters -- it derives its own timescale from k and eps. So
      // all three are computed and DISCARDED on every one of the 10 sub-steps.
      //
      // They are live for the FSD and PaSR closures, which this port does not
      // implement. Skipping them here cannot change any number, because the
      // values never reached a consumer. That is a different thing from the
      // faithful-reproduction rule elsewhere in this port: those cases were
      // behaviours that affect results, this is arithmetic whose output is
      // provably dropped on the floor. See SOLVER_PORT.md 7.7 -- upstream is
      // spending ~20 full-field passes per step on it.
      stepChemistryOdeEdc(state.rho, state.T_g, state.Y_fuel, state.Y_O2,
        kTurb, epsTurb, chiRad, CP_GAS_DRY, dtSub, 1, omega, state.Y_H2O,
        { ...shape, laminarFloorSL, dxCell: grid.dx });
      // Chemistry already updated T_g, so Q_comb starts clean.
      QComb.fill(0.0);

      // Drip torch, gas side only: the 80% solid share already went to the
      // particles through Q_solid_ext above.
      if (t < ignitionDurationS) {
        const qDrip = Q_DRIP_PER_AREA / Math.max(hBed, 1e-3);
        const gasShare = (1.0 - F_DRIP_TO_SOLID) * qDrip;
        for (let k = 0; k < nZB; k++) {
          for (let j = 0; j < ny; j++) {
            for (let i = iSrcStart; i < iSrcEnd; i++) QComb[k * nxy + j * nx + i] += gasShare;
          }
        }
      }

      // Species. Y_fuel's source is TRUE PYROLYSIS VOLATILE ONLY. Using
      // S_pyro (volatile + drying vapour) counted water as fuel and produced
      // fake extra combustion at higher moisture, which masked the real Cheney
      // moisture penalty. Drying vapour goes to Y_H2O as its own species.
      //
      // The -Y_i*S_total term is DILUTION: injecting one species must reduce
      // the mass fractions of the others.
      for (let c = 0; c < n; c++) {
        const sTotal = bedSp[c] + bedSd[c];
        SFeff[c] = bedSp[c] - state.Y_fuel[c] * sTotal;
        SO2eff[c] = -state.Y_O2[c] * sTotal;
        SH2Oeff[c] = bedSd[c] - state.Y_H2O[c] * sTotal;
      }
      stepSpeciesTransport(state.Y_fuel, state.rho, state.u, state.v, state.w,
        SFeff, dtSub, grid.dx, grid.dy, grid.dzArr, grid.dFaceAbove,
        grid.dFaceBelow, 1.0e-5, 0.0, shape);
      stepSpeciesTransport(state.Y_O2, state.rho, state.u, state.v, state.w,
        SO2eff, dtSub, grid.dx, grid.dy, grid.dzArr, grid.dFaceAbove,
        grid.dFaceBelow, 1.0e-5, 0.232, shape);
      stepSpeciesTransport(state.Y_H2O, state.rho, state.u, state.v, state.w,
        SH2Oeff, dtSub, grid.dx, grid.dy, grid.dzArr, grid.dFaceAbove,
        grid.dFaceBelow, 1.0e-5, 0.0, shape);
      for (let c = 0; c < n; c++) {
        if (state.Y_O2[c] < 0.0) state.Y_O2[c] = 0.0;
        else if (state.Y_O2[c] > 0.232) state.Y_O2[c] = 0.232;
        if (state.Y_H2O[c] < 0.0) state.Y_H2O[c] = 0.0;
        else if (state.Y_H2O[c] > 1.0) state.Y_H2O[c] = 1.0;
      }

      applyTurbulentDiffusion(state.Y_fuel, nuT, SC_T, dtSub, grid.dx, grid.dy,
        grid.dzArr, grid.dFaceAbove, grid.dFaceBelow, shape);
      applyTurbulentDiffusion(state.Y_O2, nuT, SC_T, dtSub, grid.dx, grid.dy,
        grid.dzArr, grid.dFaceAbove, grid.dFaceBelow, shape);
      applyTurbulentDiffusion(state.Y_H2O, nuT, SC_T, dtSub, grid.dx, grid.dy,
        grid.dzArr, grid.dFaceAbove, grid.dFaceBelow, shape);
      for (let c = 0; c < n; c++) {
        if (state.Y_O2[c] < 0.0) state.Y_O2[c] = 0.0;
        else if (state.Y_O2[c] > 0.232) state.Y_O2[c] = 0.232;
        if (state.Y_fuel[c] < 0.0) state.Y_fuel[c] = 0.0;
        else if (state.Y_fuel[c] > 1.0) state.Y_fuel[c] = 1.0;
        if (state.Y_H2O[c] < 0.0) state.Y_H2O[c] = 0.0;
        else if (state.Y_H2O[c] > 1.0) state.Y_H2O[c] = 1.0;
      }

      // Gas-phase radiation absorption. DOM writes the gas share to q_rad_gas
      // in W/m^2; it was computed from Phase 14a onward but the caller never
      // read the array, so gas absorption was silently discarded. Plume gases
      // can absorb 30-50% of a hot cell's total.
      for (let k = 0; k < nz; k++) {
        const invDz = 1.0 / grid.dzArr[k];
        for (let s = 0; s < nxy; s++) QComb[k * nxy + s] += qRadGas[k * nxy + s] * invDz;
      }

      // Gas energy on the particle path. The bed kernel already updated
      // particle T_s and emitted Q_g_conv, so the Eulerian coupling kernel is
      // skipped entirely and the balance is applied here:
      //   rho*cp_mix*dT_g/dt = Q_comb - Q_g_conv - Q_vapour_debit
      //
      // cp_mix is composition-dependent because water vapour is about 2000
      // J/kg/K against dry air's 1100 at flame temperature. Using the dry
      // constant under-estimated thermal inertia by up to 25% in
      // moisture-laden cells, warming the gas faster than it should and
      // erasing part of the Cheney moisture feedback.
      //
      // The vapour debit is separate from cp_mix: every kg injected arrives at
      // T_inject and must be warmed to T_g before it can carry heat.
      for (let c = 0; c < n; c++) {
        const cpMix = (1.0 - state.Y_H2O[c]) * CP_GAS_DRY + state.Y_H2O[c] * CP_VAPOR;
        const dTdry = state.T_g[c] - state.T_s[c];
        const dTwet = state.T_g[c] - T_BOIL_WATER;
        const qVapour = (bedSp[c] * (dTdry > 0.0 ? dTdry : 0.0)
                       + bedSd[c] * (dTwet > 0.0 ? dTwet : 0.0)) * CP_VAPOR;
        const gasInv = dtSub / (Math.max(state.rho[c], 1.0e-3) * cpMix);
        let T = state.T_g[c] + (QComb[c] - bedQgc[c] - qVapour) * gasInv;
        if (T < TAmb) T = TAmb;   // transient over-cooling must not go
                                  // nonphysical; matches the coupling kernel's
                                  // effective floor
        state.T_g[c] = T;
      }

      // Vertical solid conduction: the grass blade is one continuous solid
      // spanning the bed cells. Heat absorbed at the bed top conducts down and
      // warms the base. Without it T_s stays strictly cell-local, only the top
      // bed cell heats, and pyrolysis cannot propagate down through the depth.
      stepSolidConductionVertical(state.T_s, state.alpha_s, grid.dzArr,
        grid.dFaceAbove, grid.dFaceBelow, K_SOLID_GRASS, 500.0, CP_SOLID,
        dtSub, shape);

      applyTurbulentDiffusion(state.T_g, nuT, PR_T, dtSub, grid.dx, grid.dy,
        grid.dzArr, grid.dFaceAbove, grid.dFaceBelow, shape);

      // Caps applied EVERY sub-step so nothing overshoots mid-loop. These are
      // numerical safety only. The old 1900 K value (Drysdale grass adiabatic)
      // was acting as a PHYSICAL clip: at peak burn M=4% reaches a higher T_g
      // than M=8%, but both got pinned at 1900 K, so both radiated forward
      // identically and moisture had no effect on ROS at all.
      for (let c = 0; c < n; c++) {
        if (state.T_g[c] > T_FLAME_AD) state.T_g[c] = T_FLAME_AD;
        else if (state.T_g[c] < TAmb) state.T_g[c] = TAmb;
        if (state.T_s[c] > T_SURF_MAX) state.T_s[c] = T_SURF_MAX;
        else if (state.T_s[c] < TAmb) state.T_s[c] = TAmb;
      }
    }

    toc('substep_loop');

    // 13. Level-set evolution.
    tic();
    updateCellAge(cellAge, flameBody, dt);
    const qIn3d = computeQInAtFront3d(qFrankman, qDomFwd, aheadBand, null, shape);
    if (levelSetPassive) {
      // Zero forcing: the level set stays put and the bed must self-ignite
      // ahead through CFD advection, DOM radiation and bed coupling. Tests
      // whether the kinematic v_n was masking working CFD physics or
      // supplying essential closure.
      vnField.fill(0.0);
    } else {
      computeVn3d(qIn3d, rhoB, CP_SOLID, hBed, TIgn, TAmb, bedMLocal,
        undefined, 1.0, vnField);
    }
    // WRF-Fire / CAWFE pattern: blend the resolved front speed toward the
    // empirical rate at low wind, where the resolved closure cannot propagate.
    if (empiricalRosEnable && empiricalBlendW > 0.0) {
      const w = empiricalBlendW;
      for (let c = 0; c < n; c++) {
        vnField[c] = (1.0 - w) * vnField[c] + w * empiricalRosMs;
      }
    }
    lset.evolve(dt, vnField);
    // Reinit runs in BOTH modes. Skipping it when passive looked safe and was
    // a regression: without periodic reinit the float precision drifts even at
    // v_n = 0.
    lset.maybeReinitialize();
    toc('levelset_evolve');

    // 14. EoS, front tracking, exits.
    for (let c = 0; c < n; c++) {
      state.rho[c] = P0 / (R_AIR * Math.max(state.T_g[c], TAmb));
    }
    // The T_s tracker is diagnostic-only while a level set is running: "any
    // cell >= T_ign" is mesh-runaway by construction. append=false.
    const frontTs = updateFrontTracking(state.T_s, state.alpha_s, grid.xMid,
      TIgn, t, frontT, frontX, nZB === 0, { nx, ny, nZBed: nZB });
    let newFront;
    if (nZB > 0) {
      const fl = lset.frontX(Math.floor(nZB / 2), Math.floor(ny / 2));
      if (Number.isFinite(fl)) {
        const last = frontX.length ? frontX[frontX.length - 1] : 0.0;
        if (fl > last) { frontT.push(t); frontX.push(fl); }
        newFront = fl;      // NOT max() -- the level set wins outright
      } else {
        newFront = frontTs;
      }
    } else {
      newFront = frontTs;
    }
    if (newFront > lastFrontX) lastFrontX = newFront;

    // Sample the T_s front on the same ~1 s cadence the workers snapshot at.
    if (tsT.length === 0 || t - tsT[tsT.length - 1] >= 1.0) {
      const xTs = tsFrontX();
      if (!Number.isNaN(xTs)) { tsT.push(t); tsX.push(xTs); }
    }

    t += dt;

    if (onStep && onStep({
      step, t, dt, frontX: newFront, projDivMax, projNIter,
      nAlive: nAliveOut[0], nBurned: nBurnedOut[0], qRad,
      TgMax, TsMax: diagMax[0], grid, state, lset, phiFlame,
    }) === false) break;

    // Extinction: no flame body for 50 consecutive steps, once past ignition
    // and the bootstrap window. The old criterion (v_n < 1e-3) never fired,
    // because residual DOM-only propagation keeps v_n near 0.05 m/s even with
    // no flame at all.
    let flameCells = 0;
    for (let c = 0; c < n; c++) if (flameBody[c]) flameCells++;
    if (flameCells === 0 && t > ignitionDurationS + 2.0) {
      vnExtinctCount += 1;
      if (vnExtinctCount > 50) break;
    } else {
      vnExtinctCount = 0;
    }
    // Stop before the front reaches the outlet, or the open-boundary
    // treatment contaminates the ROS measurement.
    const frontLset = lset.frontX(1, Math.floor(ny / 2));
    if (frontLset > bedXEndActual - grid.dx) break;
  }

  // source_x = n_src*dx and domain_m = Lx, matching the reference exactly.
  // Note source_x is the source WIDTH measured from x=0, not the source's
  // position in the domain -- with bed_x_start > 0 those differ. Faithful.
  const rosMs = computeSteadyRos(frontT, frontX, t,
    (iSrcEnd - iSrcStart) * grid.dx, grid.Lx);

  // Least-squares slope of the T_s front, skipping t < 1 s so the ignition
  // transient does not enter the fit -- the workers' rule, and the reason
  // Phase 17f caught a "PASS" that was really the source-patch burst.
  let rosTsMs = NaN;
  {
    const idx = [];
    for (let i = 0; i < tsT.length; i++) if (tsT[i] >= 1.0) idx.push(i);
    if (idx.length >= 3) {
      let sx = 0, sy = 0;
      for (const i of idx) { sx += tsT[i]; sy += tsX[i]; }
      const mx = sx / idx.length, my = sy / idx.length;
      let num = 0, den = 0;
      for (const i of idx) { num += (tsT[i] - mx) * (tsX[i] - my); den += (tsT[i] - mx) ** 2; }
      if (den > 0) rosTsMs = num / den;
    }
  }

  // The headline number follows the same switch the physics does. When the
  // level set is passive it is inert, so reporting its position as the spread
  // rate is meaningless; the solid front is the fire.
  const rosReportedMs = levelSetPassive && Number.isFinite(rosTsMs) ? rosTsMs : rosMs;
  return {
    rosMs: rosReportedMs, rosMMin: rosReportedMs * 60.0,
    rosLsetMs: rosMs, rosLsetMMin: rosMs * 60.0,
    rosTsMs, rosTsMMin: rosTsMs * 60.0,
    tsT, tsX, timings: prof,
    empiricalRosMs, empiricalBlendW,
    frontT, frontX, grid, state, lset, bed,
    steps: step, t, nAlloc,
  };
}
