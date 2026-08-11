/**
 * Lagrangian sub-grid bed particles — JS port of
 * model_outdoor/physics_3d/lagrangian_bed_3d.py.
 *
 * The bed is a population of discrete particles rather than Eulerian
 * (Nz,Ny,Nx) m_solid/m_water/m_char fields, so bed-scale physics (drying,
 * pyrolysis, char oxidation, smoldering) runs at sub-grid resolution
 * independent of dx. This is the bed half of the WFDS Lagrangian-vegetation
 * approach (Mell et al. 2007, IJWF 19:238).
 *
 * WHY IT IS REQUIRED, not optional: the Eulerian bed alternative was measured
 * at ROS 30.854 vs 38.130 m/min (-19%) on the same case, and it is no faster.
 * A single lumped pyrolysis state per cell cannot carry the front.
 *
 * PORTING NOTES — the places this deviates from a naive transcription, and why:
 *
 *   1. `T_s**4` in the radiative-loss diagnostic is a float**INT in Python.
 *      numba lowers integer powers to repeated multiplication while JS `**`
 *      and CPython both call pow(); they disagree in the last ulp. Written
 *      here as Ts2*Ts2. Same gotcha as coupling.js. The Newton loop already
 *      used explicit multiplies upstream, so it needed no change.
 *
 *   2. `Y_O2 ** N_O2_OP` and `_burn_frac ** char_ox_ash_exp` are float**float
 *      and DO go through pow in numba — Math.pow is correct for those two.
 *
 *   3. `int(x/dx)` truncates toward zero in Python, so int(-0.5) is 0, NOT -1.
 *      Math.trunc, never Math.floor. The scatter and aggregate kernels rely on
 *      this: they index with int() first and only afterwards test i < 0, so a
 *      particle at small negative x lands in column 0 rather than being
 *      rejected. Faithful, deliberate, and probably an upstream latent bug —
 *      logged in SOLVER_PORT.md rather than silently fixed here.
 *
 *   4. Grids are flat Float64Array, idx = (k*Ny + j)*Nx + i, matching every
 *      other kernel in this directory. Particle arrays are flat and indexed
 *      by slot; `alive` is Int32Array with 0 = dead.
 *
 * Determinism: every loop is sequential, exactly as the Python is. Rule #17
 * (bit-exact reproducibility) is a property of the ported code too — the
 * kernel-vector harness asserts it via checkDeterminism().
 */

// ── Shared kinetics constants ─────────────────────────────────────────
// Values mirror model_outdoor/physics_3d/pyrolysis_3d.py so the particle and
// grid paths remain physically interchangeable.

export const R_GAS = 8.314;              // [J/mol/K]

export const A_DRY = 4290.0;             // [1/s]   Lautenberger 2009 white pine
export const E_DRY = 43800.0;            // [J/mol]
export const L_VAP_WATER = 2260000.0;    // [J/kg]

export const A_MD2004 = 36280.0;         // [1/s]   thermal pyrolysis
export const E_MD2004 = 58600.0;         // [J/mol]
export const A_OP_MD2004 = 72600.0;      // [1/s]   oxidative pyrolysis
export const E_OP_MD2004 = 55000.0;      // [J/mol]
export const N_O2_OP = 1.0;
export const Y_O2_MIN_OP = 0.001;
export const ETA_MD2004 = 0.85;          // fuel-gas fraction of pyrolysate
export const CHAR_YIELD_MD2004 = 0.15;
export const HEAT_OF_PYROLYSIS = 400000.0;   // [J/kg] +ve = endothermic
export const HOR_OP_MD2004 = -1000000.0;     // [J/kg] -ve = exothermic

export const A_CHAR = 709000.0;          // [1/s]
export const E_CHAR = 94000.0;           // [J/mol]
export const HOC_CHAR = 32000000.0;      // [J/kg]
export const T_CHAR_ONSET = 600.0;       // [K]
export const Y_O2_MIN_CHAR = 0.001;

export const A_SMOLD = 1000000.0;        // [1/s]
export const E_SMOLD = 80000.0;          // [J/mol]
export const HOC_SMOLD = 28000000.0;     // [J/kg]
export const T_SMOLD_ONSET = 473.0;      // [K]
export const Y_O2_MIN_SMOLD = 0.001;

// ── Bed-particle constants ────────────────────────────────────────────

export const RHO_SOLID_TRUE_GRASS = 380.0;  // [kg/m^3] Susott 1982
export const CP_SOLID_GRASS = 1500.0;       // [J/kg/K] Mell 2007 §3.4
export const H_CONV_DEFAULT = 25.0;         // [W/m^2/K] grass blade, Mell 2007
export const SAV_GRASS_DEFAULT = 2000.0;    // [1/m] Cheney 1993 fine fuel
export const M_PARTICLE_BURNOUT = 1.0e-8;   // [kg] retire below this total mass
export const SIGMA_SB = 5.67e-8;            // [W/m^2/K^4]
export const EPS_SOLID_DEFAULT = 0.9;
export const T_BOIL_WATER = 373.15;         // [K]

// Grass-tuned Arrhenius drying for DRY_MODE_COMBINED. E = 30 kJ/mol sits
// between Sano & Hasegawa 1995 (rice straw capillary water, 20 kJ/mol) and
// Lautenberger 2009 (white-pine bound water, 44 kJ/mol).
export const A_DRY_GRASS = 4.29e3;
export const E_DRY_GRASS = 30000.0;

export const DRY_MODE_ARRHENIUS = 0;    // Lautenberger only
export const DRY_MODE_EQUILIBRIUM = 1;  // FIRETEC heat-rate-limited only
export const DRY_MODE_COMBINED = 2;     // grass Arrhenius + equilibrium above boil

export const ALIVE_FALSE = 0;
export const ALIVE_TRUE = 1;

// Smolder flux cap [W/m^2] — gentler than char-ox by the 2/5 volumetric ratio.
const Q_SMOLD_FLUX_MAX = 4.0e4;

// ── Cell location ─────────────────────────────────────────────────────

/**
 * k-index of the cell containing z, or -1 if outside the domain.
 * z_face has length Nz+1; cell k spans z_face[k]..z_face[k+1].
 */
export function locateKFromZ(z, zFace, nz) {
  if (z < zFace[0] || z >= zFace[nz]) return -1;
  for (let k = 0; k < nz; k++) {
    if (z < zFace[k + 1]) return k;
  }
  return nz - 1;
}

/**
 * (i, j, k) of the cell containing (x, y, z); any index -1 means the particle
 * left the domain along that axis. Caller treats that as a domain-exit event.
 */
export function locateCell(x, y, z, dx, dy, zFace, nz, nx, ny) {
  let i;
  if (x < 0.0) i = -1;
  else {
    i = Math.trunc(x / dx);
    if (i >= nx) i = -1;
  }
  let j;
  if (y < 0.0) j = -1;
  else {
    j = Math.trunc(y / dy);
    if (j >= ny) j = -1;
  }
  const k = locateKFromZ(z, zFace, nz);
  return [i, j, k];
}

// ── Buffer allocation + initialisation ────────────────────────────────

/**
 * Allocate the particle state arrays for N_max slots.
 * Kinematic (x,y,z,u,v,w,age,alive) plus bed-specific mass/temperature state.
 */
export function allocateBedParticleBuffers(nMax) {
  if (nMax < 0) throw new Error(`N_max must be >= 0; got ${nMax}`);
  const f = () => new Float64Array(nMax);
  return {
    x: f(), y: f(), z: f(),
    u: f(), v: f(), w: f(),
    age: f(),
    alive: new Int32Array(nMax),
    m_solid: f(), m_water: f(), m_char: f(),
    T_s: f(),
    m_solid_0: f(), m_water_0: f(),
    sav: f(),
    // Peak char mass ever reached, ratcheted during pyrolysis. Reference for
    // the ash-coverage penalty in char oxidation (Phase 20 C).
    m_char_max: f(),
  };
}

/**
 * Populate `buf` with particles across every bed cell holding fuel.
 *
 * One `nPerCell` group per (k,j,i) with k < nZBed and alpha_s > 0. Positions
 * use a coprime-mod sub-cell packing — deterministic and low-discrepancy, with
 * no RNG anywhere, so repeat initialisation is bit-exact (Rule #17).
 *
 * rho_b_dry is the BULK density (kg solid per m^3 of cell volume; porosity is
 * already baked in), so cell solid mass is rho_b*V_cell and is NOT multiplied
 * by alpha_s again. Matches the Eulerian init in spread_3d.
 *
 * @returns {number} particles allocated
 */
export function initializeBedParticlesFromAlphaS(
  buf, alphaS, rhoBDry, moistureFrac, tAmb,
  dx, dy, dzArr, nZBed, nPerCell,
  { nx, ny, nz, sav = SAV_GRASS_DEFAULT,
    iLo = null, iHi = null, jLo = null, jHi = null } = {},
) {
  const nMax = buf.alive.length;
  const i0 = iLo === null ? 0 : iLo;
  const i1 = iHi === null ? nx : iHi;
  const j0 = jLo === null ? 0 : jLo;
  const j1 = jHi === null ? ny : jHi;
  const kTop = Math.min(nZBed, nz);

  let nBedCells = 0;
  for (let k = 0; k < kTop; k++) {
    for (let j = j0; j < Math.min(j1, ny); j++) {
      for (let i = i0; i < Math.min(i1, nx); i++) {
        if (alphaS[(k * ny + j) * nx + i] > 0.0) nBedCells++;
      }
    }
  }
  const nRequired = nBedCells * nPerCell;
  if (nRequired > nMax) {
    throw new Error(
      `Buffer too small: need ${nRequired} slots (${nBedCells} bed cells ` +
      `x ${nPerCell} particles); have ${nMax}`,
    );
  }

  let slot = 0;
  for (let k = 0; k < kTop; k++) {
    const vCell = dx * dy * dzArr[k];
    // Sum of dz below this cell. numpy's pairwise sum is a plain sequential
    // loop below 8 elements, and n_z_bed is 8 in production, so a straight
    // accumulation matches it. If n_z_bed ever exceeds 8 this needs numpy's
    // 8-accumulator unrolled form to stay bit-comparable.
    let zFaceK = 0.0;
    for (let kk = 0; kk < k; kk++) zFaceK += dzArr[kk];

    for (let j = j0; j < Math.min(j1, ny); j++) {
      for (let i = i0; i < Math.min(i1, nx); i++) {
        const aS = alphaS[(k * ny + j) * nx + i];
        if (aS <= 0.0) continue;

        const mSolidCell = rhoBDry * vCell;
        const mSolidPerP = mSolidCell / nPerCell;
        const mWaterPerP = moistureFrac * mSolidPerP;

        for (let p = 0; p < nPerCell; p++) {
          const fx = (((p * 13) % nPerCell) + 0.5) / nPerCell;
          const fy = (((p * 7) % nPerCell) + 0.5) / nPerCell;
          const fz = (p + 0.5) / nPerCell;

          buf.x[slot] = (i + fx) * dx;
          buf.y[slot] = (j + fy) * dy;
          buf.z[slot] = zFaceK + fz * dzArr[k];
          buf.u[slot] = 0.0;
          buf.v[slot] = 0.0;
          buf.w[slot] = 0.0;
          buf.alive[slot] = ALIVE_TRUE;
          buf.age[slot] = 0.0;

          buf.m_solid[slot] = mSolidPerP;
          buf.m_water[slot] = mWaterPerP;
          buf.m_char[slot] = 0.0;
          buf.T_s[slot] = tAmb;
          buf.m_solid_0[slot] = mSolidPerP;
          buf.m_water_0[slot] = mWaterPerP;
          buf.sav[slot] = sav;

          slot++;
        }
      }
    }
  }
  return slot;
}

// ── Horizontal solid conduction ───────────────────────────────────────

/**
 * Horizontal (x,y) conduction on the per-cell T_s grid, with the resulting
 * delta scattered back onto the particles.
 *
 * Grass conduction along the bed plane is genuinely small (alpha_solid about
 * 1.4e-7 m^2/s, penetration sqrt(alpha*t) about 12 um in 1 ms) but it is the
 * only forward-spread pathway besides gas-mediated radiation feedback, so it
 * is retained. k_solid and rho*cp match the vertical-conduction kernel.
 *
 * Neighbours read from a snapshot of the old T_s, so there is no in-place
 * hazard. x uses zero-flux Neumann at the edges and at non-fuel neighbours;
 * y is periodic.
 */
export function stepHorizontalSolidConductionScatter(
  partX, partY, partZ, partAlive,
  partMSolid, partMWater, partMChar, partTs,
  TsGrid, alphaSGrid,
  dx, dy, zFace,
  kSolid, rhoSolidTrue, cpSolid, nZBed, dt,
  { nx, ny, nz } = {},
) {
  const diff = kSolid / (rhoSolidTrue * cpSolid);   // m^2/s
  const invDx2 = 1.0 / (dx * dx);
  const invDy2 = 1.0 / (dy * dy);
  const nxy = ny * nx;

  const TsOld = TsGrid.slice();
  const deltaT = new Float64Array(TsGrid.length);

  const kTop = Math.min(nZBed, nz);
  for (let k = 0; k < kTop; k++) {
    for (let j = 0; j < ny; j++) {
      const jp1 = j + 1 < ny ? j + 1 : 0;
      const jm1 = j - 1 >= 0 ? j - 1 : ny - 1;
      for (let i = 0; i < nx; i++) {
        const c = (k * ny + j) * nx + i;
        if (alphaSGrid[c] <= 0.0) continue;
        const Tc = TsOld[c];

        const TxP = (i + 1 < nx && alphaSGrid[c + 1] > 0.0) ? TsOld[c + 1] : Tc;
        const TxM = (i - 1 >= 0 && alphaSGrid[c - 1] > 0.0) ? TsOld[c - 1] : Tc;
        const cjp = k * nxy + jp1 * nx + i;
        const cjm = k * nxy + jm1 * nx + i;
        const TyP = alphaSGrid[cjp] > 0.0 ? TsOld[cjp] : Tc;
        const TyM = alphaSGrid[cjm] > 0.0 ? TsOld[cjm] : Tc;

        const lap = (TxP - 2.0 * Tc + TxM) * invDx2
                  + (TyP - 2.0 * Tc + TyM) * invDy2;
        const dT = diff * lap * dt;
        deltaT[c] = dT;
        TsGrid[c] = Tc + dT;
      }
    }
  }

  const nMax = partAlive.length;
  for (let p = 0; p < nMax; p++) {
    if (partAlive[p] === ALIVE_FALSE) continue;
    const i = Math.trunc(partX[p] / dx);
    const j = Math.trunc(partY[p] / dy);
    if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
    const k = locateKFromZ(partZ[p], zFace, nz);
    if (k < 0 || k >= nZBed) continue;
    partTs[p] += deltaT[(k * ny + j) * nx + i];
  }
}

// ── Particle → grid aggregation ───────────────────────────────────────

/**
 * Mirror per-particle T_s into the Eulerian T_s grid as a mass-weighted mean.
 * Cells with no particles are LEFT UNCHANGED — the downstream DOM kernel
 * expects T_s >= T_amb, and non-bed cells already hold T_amb.
 */
export function aggregateParticlesToTsGrid(
  partX, partY, partZ, partAlive,
  partMSolid, partMWater, partMChar, partTs,
  dx, dy, zFace, TsGrid, tAmb,
  { nx, ny, nz } = {},
) {
  const n = TsGrid.length;
  const num = new Float64Array(n);
  const den = new Float64Array(n);
  const nMax = partAlive.length;

  for (let p = 0; p < nMax; p++) {
    if (partAlive[p] === ALIVE_FALSE) continue;
    const xi = Math.trunc(partX[p] / dx);
    const yj = Math.trunc(partY[p] / dy);
    if (xi < 0 || xi >= nx || yj < 0 || yj >= ny) continue;
    const zk = locateKFromZ(partZ[p], zFace, nz);
    if (zk < 0) continue;
    const mT = partMSolid[p] + partMWater[p] + partMChar[p];
    if (mT <= 0.0) continue;
    const c = (zk * ny + yj) * nx + xi;
    num[c] += partTs[p] * mT;
    den[c] += mT;
  }
  for (let c = 0; c < n; c++) {
    if (den[c] > 0.0) TsGrid[c] = num[c] / den[c];
  }
}

/**
 * Aggregate per-particle M = m_water/m_solid into a per-cell grid.
 *
 * Water and dry mass are summed independently, then divided — a cell-level
 * ratio of sums, not a mean of ratios. Cells without particles get 0.
 *
 * Consumed by the DOM solver, which scales kappa_solid by (1 + beta*M_local):
 * a wet bed absorbs more IR per kg of solid (Mell 2007 WFDS / Linn 2002).
 */
export function aggregateParticlesToMLocalGrid(
  partX, partY, partZ, partAlive,
  partMSolid, partMWater,
  dx, dy, zFace, MLocalGrid,
  { nx, ny, nz } = {},
) {
  const n = MLocalGrid.length;
  const num = new Float64Array(n);
  const den = new Float64Array(n);
  const nMax = partAlive.length;

  for (let p = 0; p < nMax; p++) {
    if (partAlive[p] === ALIVE_FALSE) continue;
    const xi = Math.trunc(partX[p] / dx);
    const yj = Math.trunc(partY[p] / dy);
    if (xi < 0 || xi >= nx || yj < 0 || yj >= ny) continue;
    const zk = locateKFromZ(partZ[p], zFace, nz);
    if (zk < 0) continue;
    const c = (zk * ny + yj) * nx + xi;
    num[c] += partMWater[p];
    den[c] += partMSolid[p];
  }
  for (let c = 0; c < n; c++) {
    MLocalGrid[c] = den[c] > 0.0 ? num[c] / den[c] : 0.0;
  }
}

// ── The main particle step ────────────────────────────────────────────

/**
 * One step of drying + pyrolysis + char-ox + smolder + T_s update + aggregation.
 *
 * Per alive particle:
 *   1. locate cell (retire if it left the domain)
 *   2. drying — Arrhenius, mode-dependent
 *   3. pyrolysis — MD2004 thermal + oxidative, with a linear moisture gate
 *   4. char oxidation, surface-flux capped
 *   5. smoldering, surface-flux capped
 *   6. aggregate volumetric sources to the gas cell
 *   7. T_s via Newton-implicit convection + reaction + radiative loss
 *   8. burnout check
 *
 * The eight per-cell aggregation arrays are ZEROED at entry — the caller does
 * not need to clear them. This matches the Eulerian kernels' convention.
 *
 * @param {object} s   particle state arrays (mutated in place)
 * @param {object} g   gas + grid fields
 * @param {object} out per-cell source arrays (overwritten) + diagnostics
 * @param {object} par physics parameters and toggles
 */
export function stepBedParticles(s, g, out, par) {
  // Kirchhoff-symmetric absorption (opt-in via par.absorbGeometric).
  //
  // Emission carries a depth attenuation, f_geom = exp(-kappa*(h_bed - z_p)),
  // which is ~12% for deep particles. Absorption carries none -- Q_solid_ext is
  // split uniformly across the n_per_cell particles in a cell. So a deep
  // particle absorbs its full share and emits 12% of what it should, and cannot
  // radiatively self-limit. Reciprocity requires the same geometric factor on
  // both sides (SOLVER_PORT.md 7.8, defect 3).
  //
  // Fixed by weighting each particle's share of the cell's external heat by its
  // own absorption cross-section, A_p * f_geom, NORMALISED per cell so the
  // total delivered to the cell is unchanged. Redistribution, not attenuation.
  //
  // The A_p term matters as much as f_geom. Absorption was split by particle
  // COUNT, independent of size, while emission is eps*sigma*A_p*T^4. So as a
  // particle burns down, A_p -> 0 while its share of the absorbed radiation
  // stays fixed, and the only way to balance is T -> infinity. That is the
  // 4601 K runaway. Kirchhoff again: absorptivity and emissivity belong to the
  // SAME surface, so both must carry A_p and both must carry f_geom.
  const { nx, ny, nz } = g;
  const nxy = ny * nx;
  const nMax = s.alive.length;

  const {
    dx, dy, dzArr, zFace,
    hConv, rhoSolidTrue, cpSolid, epsSolid, tAmb,
    viewFactor, viewFactorGeometric, hBed, kappaBedEff,
    dt, doDrying, doPyrolysis, doCharOx, doSmolder, dryingMode,
    charOxFluxCapWm2, charOxAshExp, nPerCellForSplit,
  } = par;

  const {
    S_pyro, S_drying, Q_pyro, Q_drying, Y_F_source,
    Q_char, Q_smold, Q_g_conv, nAliveOut, nBurnedOut, diagMaxOut,
  } = out;

  const nCells = nz * ny * nx;
  for (let c = 0; c < nCells; c++) {
    S_pyro[c] = 0.0;
    S_drying[c] = 0.0;
    Q_pyro[c] = 0.0;
    Q_drying[c] = 0.0;
    Y_F_source[c] = 0.0;
    Q_char[c] = 0.0;
    Q_smold[c] = 0.0;
    Q_g_conv[c] = 0.0;
  }

  const invDt = dt > 0.0 ? 1.0 / dt : 0.0;

  // Pre-pass: per-cell sum of f_geom over live particles, for the normalised
  // absorption split. Only needed when both geometric flags are on.
  const absorbGeometric = par.absorbGeometric === true && viewFactorGeometric;
  let fGeomSum = null;
  if (absorbGeometric) {
    fGeomSum = new Float64Array(nCells);
    for (let p = 0; p < nMax; p++) {
      if (s.alive[p] === ALIVE_FALSE) continue;
      const [i2, j2, k2] = locateCell(s.x[p], s.y[p], s.z[p], dx, dy, zFace, nz, nx, ny);
      if (i2 < 0 || j2 < 0 || k2 < 0) continue;
      let hA = hBed - s.z[p];
      if (hA < 0.0) hA = 0.0;
      // Absorption cross-section: same A_p and same f_geom the emission uses.
      const apP = (s.sav[p] * (s.m_solid[p] + s.m_char[p])) / rhoSolidTrue;
      fGeomSum[k2 * nxy + j2 * nx + i2] += apP * Math.exp(-kappaBedEff * hA);
    }
  }

  for (let d = 0; d < diagMaxOut.length; d++) diagMaxOut[d] = 0.0;
  let TsMaxSeen = 0.0;
  let newtonMaxFGlobal = 0.0;

  let nAlive = 0;
  let nBurned = 0;

  for (let p = 0; p < nMax; p++) {
    if (s.alive[p] === ALIVE_FALSE) continue;

    const [i, j, k] = locateCell(s.x[p], s.y[p], s.z[p], dx, dy, zFace, nz, nx, ny);
    if (i < 0 || j < 0 || k < 0) {
      s.alive[p] = ALIVE_FALSE;
      continue;
    }
    const c = k * nxy + j * nx + i;

    const vCell = dx * dy * dzArr[k];
    let Ts = s.T_s[p];
    const Tg = g.T_g[c];
    const YO2 = g.Y_O2[c];

    let mSolidP = s.m_solid[p];
    let mWaterP = s.m_water[p];
    let mCharP = s.m_char[p];
    const mWater0 = s.m_water_0[p];
    const savP = s.sav[p];

    // ── (1) Drying ────────────────────────────────────────────────────
    // ARRHENIUS  : Lautenberger 2009 white-pine bound-water kinetics.
    // EQUILIBRIUM: skipped here; step (6.5) does all evaporation.
    // COMBINED   : grass Arrhenius below boil + equilibrium above.
    let dmEvap = 0.0;
    if (doDrying && mWaterP > 0.0 && Ts > 0.0) {
      if (dryingMode === DRY_MODE_ARRHENIUS) {
        const kDry = A_DRY * Math.exp(-E_DRY / (R_GAS * Ts));
        const mwNew = mWaterP * Math.exp(-kDry * dt);
        dmEvap = mWaterP - mwNew;
        mWaterP = mwNew;
      } else if (dryingMode === DRY_MODE_COMBINED) {
        const kDry = A_DRY_GRASS * Math.exp(-E_DRY_GRASS / (R_GAS * Ts));
        const mwNew = mWaterP * Math.exp(-kDry * dt);
        dmEvap = mWaterP - mwNew;
        mWaterP = mwNew;
      }
    }

    // ── (2) Pyrolysis — MD2004 thermal + oxidative, moisture-gated ────
    let dmPyro = 0.0;
    let ratePyro = 0.0;
    let fThermal = 1.0;
    let fOp = 0.0;
    if (doPyrolysis && mSolidP > 0.0 && Ts > 0.0) {
      // Moisture gate: water vapour competes for cell-wall sites and cools
      // the particle (Mell 2007 WFDS pattern). LINEAR (1 - wet), not the
      // original (1 - 100*wet) hard cutoff — that blocked pyrolysis entirely
      // above wet = 1% and produced a discontinuous ignition delay (M=0%
      // igniting in 0.4 s against M=5% taking 28 s at cone density). The
      // linear gate matches cone-calorimeter t_ig(M) across the full range.
      let moistGate;
      if (mWater0 > 0.0) {
        const wet = mWaterP / mWater0;
        moistGate = 1.0 - wet;
        if (moistGate < 0.0) moistGate = 0.0;
      } else {
        moistGate = 1.0;
      }

      if (moistGate > 0.0) {
        const kThermal = A_MD2004 * Math.exp(-E_MD2004 / (R_GAS * Ts));
        let kOp;
        if (YO2 > Y_O2_MIN_OP) {
          // float**float -> pow in numba, so Math.pow is the faithful lowering
          kOp = A_OP_MD2004 * Math.exp(-E_OP_MD2004 / (R_GAS * Ts))
              * Math.pow(YO2, N_O2_OP);
        } else {
          kOp = 0.0;
        }
        const kTotal = kThermal + kOp;
        if (kTotal > 0.0) {
          const mSolidNew = mSolidP * Math.exp(-kTotal * dt);
          const dmFull = mSolidP - mSolidNew;
          dmPyro = dmFull * moistGate;
          ratePyro = dmPyro * invDt;
          fThermal = kThermal / kTotal;
          fOp = kOp / kTotal;
          mCharP += CHAR_YIELD_MD2004 * dmPyro;
          mSolidP -= dmPyro;
          if (mCharP > s.m_char_max[p]) s.m_char_max[p] = mCharP;
        }
      }
    }

    // Condensed-phase surface area after pyrolysis. m_char is INCLUDED —
    // char keeps the geometric surface that oxidises and radiates. Excluding
    // it zeroed A_p once pyrolysis completed, which killed both the
    // Stefan-Boltzmann loss and Q_conv and let char heat unboundedly.
    const APNow = savP * (mSolidP + mCharP) / rhoSolidTrue;

    // ── (3) Char oxidation ────────────────────────────────────────────
    // Exothermic; heats the particle and the gas cell. No mass goes to the
    // gas (CO2 is not tracked separately — matches the Eulerian convention).
    // The cap is a SURFACE FLUX, not a volumetric rate: as char depletes A_p
    // shrinks and the cap must shrink with it, otherwise burnt-out particles
    // climb to 10,000 K under a fixed volumetric cap.
    let dmCharOx = 0.0;
    let QCharOxP = 0.0;
    if (doCharOx && mCharP > 1.0e-12 && Ts >= T_CHAR_ONSET && YO2 >= Y_O2_MIN_CHAR) {
      const kCh = A_CHAR * Math.exp(-E_CHAR / (R_GAS * Ts));
      const mDotCh = kCh * mCharP * YO2;
      let mConsCh = mDotCh * dt;
      if (mConsCh > 0.5 * mCharP) mConsCh = 0.5 * mCharP;

      let AReactive = APNow;
      if (charOxAshExp > 0.0 && s.m_char_max[p] > 1.0e-15) {
        // Ash-coverage penalty: as char burns below its peak, ash blocks O2
        // access and effective area falls faster than mass.
        let burnFrac = mCharP / s.m_char_max[p];
        if (burnFrac < 0.0) burnFrac = 0.0;
        AReactive *= Math.pow(burnFrac, charOxAshExp);
      }
      const QCapPart = charOxFluxCapWm2 * AReactive;
      const QArrhPart = (mConsCh * HOC_CHAR) * invDt;
      if (QArrhPart > QCapPart) mConsCh = QCapPart / (HOC_CHAR * invDt);
      dmCharOx = mConsCh;
      QCharOxP = (dmCharOx * HOC_CHAR) * invDt;
      mCharP -= dmCharOx;
    }

    // ── (4) Smoldering ────────────────────────────────────────────────
    // Slow low-T surface oxidation, same Arrhenius form at lower E. Consumes
    // m_solid AND m_char: in the sub-grid view both surfaces are exposed.
    // (The Eulerian path consumes m_solid only.)
    let QSmoldP = 0.0;
    if (doSmolder && Ts >= T_SMOLD_ONSET && YO2 >= Y_O2_MIN_SMOLD) {
      const mAvailSm = mSolidP + mCharP;
      if (mAvailSm > 1.0e-12) {
        const kSm = A_SMOLD * Math.exp(-E_SMOLD / (R_GAS * Ts));
        const mDotSm = kSm * mAvailSm * YO2;
        let mConsSm = mDotSm * dt;
        if (mConsSm > 0.5 * mAvailSm) mConsSm = 0.5 * mAvailSm;

        const QCapSmPart = Q_SMOLD_FLUX_MAX * APNow;
        const QArrhSm = (mConsSm * HOC_SMOLD) * invDt;
        if (QArrhSm > QCapSmPart) mConsSm = QCapSmPart / (HOC_SMOLD * invDt);

        if (mAvailSm > 0.0) {
          const fSolidSm = mSolidP / mAvailSm;
          mSolidP -= fSolidSm * mConsSm;
          mCharP -= (1.0 - fSolidSm) * mConsSm;
        }
        QSmoldP = (mConsSm * HOC_SMOLD) * invDt;   // W per particle
      }
    }

    // ── (5) Aggregate to the gas cell as volumetric sources ───────────
    const invV = 1.0 / vCell;
    S_pyro[c] += ETA_MD2004 * ratePyro * invV;
    S_drying[c] += dmEvap * invDt * invV;
    // Heat of reaction: thermal is endothermic (positive = solid heat sink),
    // oxidative has a negative HOR so it releases. Sign matches the Eulerian
    // step_pyrolysis_md2004 convention.
    Q_pyro[c] += (ratePyro * (fThermal * HEAT_OF_PYROLYSIS + fOp * HOR_OP_MD2004)) * invV;
    Q_drying[c] += dmEvap * L_VAP_WATER * invDt * invV;
    Y_F_source[c] += ETA_MD2004 * ratePyro * invV;
    // Char-ox and smolder heat release: POSITIVE = released to gas, matching
    // the Eulerian step_char_oxidation / step_smoldering_oxidation sign.
    Q_char[c] += QCharOxP * invV;
    Q_smold[c] += QSmoldP * invV;

    // ── (6) T_s update ────────────────────────────────────────────────
    const mTotalP = mSolidP + mWaterP + mCharP;
    let QConv = 0.0;
    if (mTotalP > 0.0) {
      const AP = savP * (mSolidP + mCharP) / rhoSolidTrue;
      QConv = hConv * AP * (Tg - Ts);
      // HEAT_OF_PYROLYSIS positive -> endothermic -> cools the particle;
      // HOR_OP_MD2004 negative -> exothermic -> heats it.
      const QRxn = -ratePyro * (fThermal * HEAT_OF_PYROLYSIS + fOp * HOR_OP_MD2004);
      const QDryPart = -dmEvap * L_VAP_WATER * invDt;

      // External cell heat (drip torch, bootstrap, radiation absorption)
      // split uniformly across the n_per_cell particles in the cell.
      let QExtP = 0.0;
      if (absorbGeometric && fGeomSum[c] > 0.0) {
        // Share proportional to this particle's own view of the radiation
        // field, normalised so the cell total is preserved exactly.
        let hA0 = hBed - s.z[p];
        if (hA0 < 0.0) hA0 = 0.0;
        // NOTE: uses the PRE-step masses, matching the pre-pass sum, so the
        // per-cell weights are consistent and the normalisation is exact.
        const apAbs = (savP * (s.m_solid[p] + s.m_char[p])) / rhoSolidTrue;
        QExtP = g.Q_solid_ext[c] * vCell
              * ((apAbs * Math.exp(-kappaBedEff * hA0)) / fGeomSum[c]);
      } else if (nPerCellForSplit > 0) {
        QExtP = g.Q_solid_ext[c] * vCell / nPerCellForSplit;
      }

      // Radiative loss, scaled by an effective view factor. The scalar
      // viewFactor always applies; with viewFactorGeometric it is ALSO
      // multiplied by a per-particle Beer-Lambert term in depth below the bed
      // top, so surface particles emit freely and deep particles have their
      // emission reabsorbed by neighbours (handled by DOM instead).
      let fGeom = 1.0;
      if (viewFactorGeometric) {
        let hAbove = hBed - s.z[p];
        if (hAbove < 0.0) hAbove = 0.0;
        fGeom = Math.exp(-kappaBedEff * hAbove);
      }
      const CRad = epsSolid * SIGMA_SB * AP * viewFactor * fGeom;

      // Newton-implicit Stefan-Boltzmann. Solves
      //   m*cp*(T_new - T_old)/dt = Q_other - C*(T_new^4 - T_amb^4)
      // in 5 iterations. A single linearisation (FIRESTAR-style) is too weak
      // when 4*C*T^3 << m*cp/dt, which is always true at our particle sizes;
      // Newton converges quadratically onto the true plateau where
      // Q_other = C*(T_plateau^4 - T_amb^4).
      const mc = mTotalP * cpSolid;
      const tAmb2 = tAmb * tAmb;
      const TAmb4 = tAmb2 * tAmb2;
      const QOther = QConv + QRxn + QDryPart + QCharOxP + QSmoldP + QExtP;
      const mcInvDt = mc / dt;
      const TOldForNewton = Ts;
      let TIter = Ts;
      let F = 0.0;
      for (let it = 0; it < 5; it++) {
        const T2 = TIter * TIter;
        const T3 = T2 * TIter;
        const T4 = T3 * TIter;
        F = mcInvDt * (TIter - TOldForNewton) - QOther + CRad * (T4 - TAmb4);
        const Fp = mcInvDt + 4.0 * CRad * T3;
        TIter = TIter - F / Fp;
        if (TIter < tAmb) TIter = tAmb;   // keep the iterate physical
      }
      Ts = TIter;
      // T_s**4 is float**INT in the Python — numba lowers it to multiplies,
      // so `** 4` here would differ in the last ulp. See header note 1.
      const Ts2 = Ts * Ts;
      const QRadLossP = CRad * (Ts2 * Ts2 - TAmb4);

      const absF = F >= 0.0 ? F : -F;
      if (absF > newtonMaxFGlobal) newtonMaxFGlobal = absF;

      if (Ts > TsMaxSeen) {
        TsMaxSeen = Ts;
        diagMaxOut[0] = Ts;
        diagMaxOut[1] = QConv;
        diagMaxOut[2] = QRxn;
        diagMaxOut[3] = QDryPart;
        diagMaxOut[4] = QCharOxP;
        diagMaxOut[5] = QSmoldP;
        diagMaxOut[6] = QExtP;
        diagMaxOut[7] = QRadLossP;
        diagMaxOut[8] = QOther;
        diagMaxOut[9] = Tg;
        diagMaxOut[10] = mcInvDt;
        diagMaxOut[11] = CRad;
        diagMaxOut[12] = absF;
        diagMaxOut[14] = AP;
        diagMaxOut[15] = mTotalP;
      }

      // Backstop only — should never bind if Newton converged.
      if (Ts > 1.0e4) Ts = 1.0e4;

      // ── (6.5) FIRETEC heat-rate-limited equilibrium drying ──────────
      // Pin T_s at boiling while water remains and divert the excess energy
      // to latent evaporation (Linn 2002 / Pimont & Linn 2009). Once water is
      // exhausted the residual keeps heating the particle. This is what
      // recovers the Cheney 1993 moisture penalty that first-order Arrhenius
      // drying alone cannot reproduce.
      if (doDrying
          && (dryingMode === DRY_MODE_EQUILIBRIUM || dryingMode === DRY_MODE_COMBINED)
          && mWaterP > 0.0 && Ts > T_BOIL_WATER) {
        const excessJ = mc * (Ts - T_BOIL_WATER);
        let dmEq = excessJ / L_VAP_WATER;
        if (dmEq >= mWaterP) {
          const residualJ = excessJ - mWaterP * L_VAP_WATER;
          dmEq = mWaterP;
          mWaterP = 0.0;
          Ts = T_BOIL_WATER + (mc > 0.0 ? residualJ / mc : 0.0);
        } else {
          mWaterP -= dmEq;
          Ts = T_BOIL_WATER;
        }
        S_drying[c] += dmEq * invDt * invV;
        Q_drying[c] += dmEq * L_VAP_WATER * invDt * invV;
        dmEvap += dmEq;
      }
    }

    // Gas-side convective sink: positive means the gas loses this heat to
    // the particle.
    Q_g_conv[c] += QConv * invV;

    s.T_s[p] = Ts;
    s.m_solid[p] = mSolidP;
    s.m_water[p] = mWaterP;
    s.m_char[p] = mCharP;

    // ── (7) Burnout ───────────────────────────────────────────────────
    // Tests m_total_p as computed BEFORE step 6.5 removed water. Faithful to
    // the reference; the difference only matters for a particle within one
    // burnout threshold of retiring.
    if (mTotalP < M_PARTICLE_BURNOUT) {
      s.alive[p] = ALIVE_FALSE;
      nBurned++;
    } else {
      nAlive++;
    }
  }

  nAliveOut[0] = nAlive;
  nBurnedOut[0] = nBurned;
  diagMaxOut[13] = newtonMaxFGlobal;
}
