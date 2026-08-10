/**
 * The small pieces the main loop calls that no profile entry ever named.
 *
 * Each is a few percent of runtime or less, which is exactly why a
 * profile-driven scope survey missed them and why they all ended up here
 * together rather than in files of their own. They are still load-bearing:
 * without the O2 supply limit combustion is unbounded, without the sponge the
 * outlet admits backflow, without the wall function the near-wall k-e is wrong.
 *
 *   applyOutflowSponge          momentum_3d.apply_outflow_sponge
 *   applyWallFunction           turbulence_3d.apply_wall_function
 *   stepO2SupplyRate            combustion_3d.step_o2_supply_rate
 *   buildSoilGrid               soil_3d.build_soil_grid
 *   stepSoilConduction          soil_3d.step_soil_conduction
 *   advGasEnergy                spread_3d._adv_gas_energy
 *   updateFrontTracking         spread_3d._update_front_tracking
 *
 * `_apply_velocity_bcs` is deliberately absent. Upstream it is a documented
 * no-op: every BC is enforced by on-the-fly ghost computation inside the
 * operators ("Way B"), because real cells must never be overwritten mid-run.
 * Porting an empty function would only invite someone to fill it in.
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i.
 */
import { advect3dScalarMuscl } from './muscl.js';

// ── Outflow sponge ────────────────────────────────────────────────────

/**
 * Relax u toward the inlet log-law profile in the last few cells before the
 * outlet, damping the backflow the open (p = 0) boundary otherwise admits.
 *
 * FLAME-AWARE, and that matters: cells carrying fuel above Y_F_skip are
 * skipped entirely. An unconditional sponge kills the flame whenever the front
 * reaches the outlet zone, which it routinely does at short Lx — the Cheney
 * sweep at Lx = 10 m measured Nat 4% U=2 falling from 1.14 (PASS) to 0.13.
 */
export function applyOutflowSponge(u, uTarget2d, sigmaX, YF, YFskip, dt,
                                   { nx, ny, nz }) {
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      const uT = uTarget2d[k * ny + j];
      const row = (k * ny + j) * nx;
      for (let i = 0; i < nx; i++) {
        const sig = sigmaX[i];
        if (sig > 0.0 && YF[row + i] < YFskip) {
          u[row + i] += sig * dt * (uT - u[row + i]);
        }
      }
    }
  }
}

// ── k-epsilon wall function ───────────────────────────────────────────

export const KAPPA_VK = 0.41;        // von Karman
export const B_LOGLAW = 5.0;
export const Y_PLUS_TRANSITION = 11.0;
export const C_MU = 0.09;
export const NU_GAS_WALL = 1.5e-5;   // molecular nu -- the log law wants laminar

/**
 * Solve u_p = u_tau * ((1/kappa) ln(z_p u_tau / nu) + B) for u_tau, by Newton.
 *
 * Below y+ = Y_PLUS_TRANSITION it returns the viscous-sublayer result
 * sqrt(u_p nu / z_p) instead — the log law has no meaning there.
 */
export function uTauLogLaw(uP, zP, nu, kMinIters = 15) {
  if (uP <= 0.0 || zP <= 0.0 || nu <= 0.0) return 0.0;
  const uPabs = uP > 0.0 ? uP : -uP;
  // Start from cf ~ 5e-3, a turbulent BL at Re_x of 1e5-1e6.
  let uTau = uPabs * 0.05;
  if (uTau < 1.0e-6) uTau = 1.0e-6;
  for (let it = 0; it < kMinIters; it++) {
    const zPlus = (zP * uTau) / nu;
    if (zPlus < Y_PLUS_TRANSITION) return Math.sqrt((uPabs * nu) / zP);
    const logTerm = Math.log(zPlus) / KAPPA_VK + B_LOGLAW;
    const f = uPabs - uTau * logTerm;
    const df = -logTerm - 1.0 / KAPPA_VK;
    let uTauNew = uTau + -f / df;
    if (uTauNew <= 0.0) uTauNew = 0.5 * uTau;   // underrelax rather than flip sign
    if (Math.abs(uTauNew - uTau) < 1.0e-7 * uTauNew) { uTau = uTauNew; break; }
    uTau = uTauNew;
  }
  return uTau;
}

/**
 * Launder & Spalding (1974) wall function, written as GHOST values.
 *
 *   k_w   = u_tau^2 / sqrt(C_mu)
 *   eps_w = u_tau^3 / (kappa * z_p)
 *
 * No real cell is written — the k-e kernel reads these at its k=0 ghost slot.
 * That is the "Way B" rule: only ghosts are modifiable during a run.
 *
 * SKIPPED inside the porous bed (alpha_s[0] > 0). A smooth-wall log law is the
 * wrong physics there; the bed is a rough-wall canopy and porous-media drag
 * already carries the friction. Those cells get the k/eps floors instead.
 *
 * Reads u, v at k=1 from the PREVIOUS step's projection — an explicit lag the
 * reference accepts.
 */
export function applyWallFunction(u, v, rho, alphaS, dzArr,
                                  kWallGhost, epsWallGhost,
                                  { nx, ny, nz, kMin, epsMin, nu = NU_GAS_WALL }) {
  if (nz < 2) return;
  const zPabove = dzArr[0] + 0.5 * dzArr[1];
  const sqrtCmu = Math.sqrt(C_MU);
  const zPfirst = 0.5 * dzArr[0];
  const nxy = ny * nx;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const s = j * nx + i;
      if (alphaS[s] > 0.0) {                 // inside the bed
        kWallGhost[s] = kMin;
        epsWallGhost[s] = epsMin;
        continue;
      }
      const c1 = nxy + s;                    // k = 1
      const u1 = u[c1], v1 = v[c1];
      const uPabove = Math.sqrt(u1 * u1 + v1 * v1);
      if (uPabove < 1.0e-12) {
        kWallGhost[s] = kMin;
        epsWallGhost[s] = epsMin;
        continue;
      }
      const uTau = uTauLogLaw(uPabove, zPabove, nu);
      let kw = (uTau * uTau) / sqrtCmu;
      if (kw < kMin) kw = kMin;
      // u_tau**3 is float**INT -- numba lowers it to multiplies, so it is
      // written out here rather than as Math.pow. See lagrangianBed.js note 1.
      let epsw = (uTau * uTau * uTau) / (KAPPA_VK * zPfirst);
      if (epsw < epsMin) epsw = epsMin;
      kWallGhost[s] = kw;
      epsWallGhost[s] = epsw;
    }
  }
}

// ── O2 supply rate ────────────────────────────────────────────────────

export const S_STOICH_BIOMASS = 1.3;   // kg O2 per kg biomass volatile,
                                       // combustion_3d.S_STOICH

/**
 * Per-cell combustion limit set by how fast O2 is actually delivered [kg
 * fuel/m^3/s]. First-order upwind over the six faces, counting inflow only:
 *
 *   m_O2_in = sum over faces of max(0, rho_up * u_face) * Y_O2_up / delta
 *
 * A cell with no inflow cannot burn no matter how hot it is. A well-supplied
 * cell gets a number far above omega_chem, so this never throttles it.
 *
 * INTERIOR ONLY — the loops start at 1 and stop at N-1 in all three axes.
 * Boundary cells keep whatever the caller pre-filled, which is 1e30, i.e.
 * "infinite supply", so pilot and inlet zones stay governed by chemistry.
 *
 * That has a consequence worth stating: at Ny = 1 the j-loop range is EMPTY,
 * so this kernel writes nothing at all and every cell keeps the 1e30 fill. The
 * 2D slab therefore runs with no O2-supply limit. Faithful to the reference,
 * and arguably fine because a 2D slab has no cross-stream entrainment to
 * model, but it is a real difference between the 2D and 3D configurations
 * rather than an approximation the model makes deliberately.
 *
 * Spalding (1971) Combust. Sci. Tech. 4:43; Pruyn et al. (2018) 187:182.
 */
export function stepO2SupplyRate(rho, u, v, w, YO2, dx, dy, dzArr, omegaO2Out,
                                 { nx, ny, nz, sStoich = S_STOICH_BIOMASS }) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const nxy = ny * nx;
  for (let k = 1; k < nz - 1; k++) {
    const invDz = 1.0 / dzArr[k];
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const c = k * nxy + j * nx + i;
        let mIn = 0.0;

        let uf = 0.5 * (u[c - 1] + u[c]);
        if (uf > 0.0) mIn += rho[c - 1] * uf * YO2[c - 1] * invDx;
        uf = 0.5 * (u[c] + u[c + 1]);
        if (uf < 0.0) mIn += rho[c + 1] * -uf * YO2[c + 1] * invDx;

        let vf = 0.5 * (v[c - nx] + v[c]);
        if (vf > 0.0) mIn += rho[c - nx] * vf * YO2[c - nx] * invDy;
        vf = 0.5 * (v[c] + v[c + nx]);
        if (vf < 0.0) mIn += rho[c + nx] * -vf * YO2[c + nx] * invDy;

        let wf = 0.5 * (w[c - nxy] + w[c]);
        if (wf > 0.0) mIn += rho[c - nxy] * wf * YO2[c - nxy] * invDz;
        wf = 0.5 * (w[c] + w[c + nxy]);
        if (wf < 0.0) mIn += rho[c + nxy] * -wf * YO2[c + nxy] * invDz;

        omegaO2Out[c] = mIn / sStoich;
      }
    }
  }
}

// ── Soil ──────────────────────────────────────────────────────────────

export const RHO_SOIL_DEFAULT = 1500.0;   // [kg/m^3] dry topsoil, Hahn 1981
export const CP_SOIL_DEFAULT = 800.0;     // [J/kg/K]
export const K_SOIL_DEFAULT = 0.48;       // [W/m/K]
export const EPS_SOIL_DEFAULT = 0.85;     // [-] IR emissivity
export const SIGMA_SB_SOIL = 5.67e-8;

/**
 * Geometrically stretched 1D soil grid, 1 mm at the surface growing downward.
 *
 * ~30 mm total covers the thermal penetration depth at fire timescales —
 * sqrt(alpha * 30 s) is about 4 mm — with a large safety factor. 5-8 cells is
 * the FIRESTAR / FIRETEC standard.
 */
export function buildSoilGrid(nSoil = 6, dzFirst = 0.001, growth = 1.5) {
  const dz = new Float64Array(nSoil);
  for (let k = 0; k < nSoil; k++) dz[k] = dzFirst * Math.pow(growth, k);
  const dAbove = new Float64Array(nSoil);
  const dBelow = new Float64Array(nSoil);
  dAbove[0] = dz[0] / 2.0;                     // half-cell up to the surface BC
  for (let k = 1; k < nSoil; k++) dAbove[k] = 0.5 * (dz[k - 1] + dz[k]);
  for (let k = 0; k < nSoil - 1; k++) dBelow[k] = 0.5 * (dz[k] + dz[k + 1]);
  dBelow[nSoil - 1] = dz[nSoil - 1] / 2.0;     // half-cell down to the deep BC
  let total = 0.0;
  for (let k = 0; k < nSoil; k++) total += dz[k];
  return { soilDz: dz, dAbove, dBelow, depthTotal: total };
}

/**
 * One explicit-Euler step of 1D vertical soil conduction under every (j, i).
 *
 *   top    (z = 0):   net flux = q_in - eps*sigma*T[0]^4
 *   bottom (z = -d):  T = T_amb, the semi-infinite assumption
 *
 * This is what stops the ground behaving as a cold blackbody. As the soil
 * heats to 500-700 K under the fire, net radiation loss into it drops from
 * nearly all-incident to about half, and that fraction returns to the gas.
 *
 * Explicit is safe here: dt_max = dz_first^2/(2*alpha) is around 1.25 s at
 * 1 mm and 4e-7 m^2/s, three orders above the gas-phase dt, so no substepping.
 *
 * Carslaw & Jaeger (1959) §2; Morvan & Dupuy (2004) FIRESTAR; Pimont et al.
 * (2006) Combust. Sci. Tech. 178:1389 §2.4; Hahn (1981) J. Atmos. Sci. 38:1601.
 */
export function stepSoilConduction(Tsoil, qInSurface, dt, soilDz, dAbove, dBelow,
                                   { nx, ny, nSoil,
                                     kS = K_SOIL_DEFAULT,
                                     rhoS = RHO_SOIL_DEFAULT,
                                     cpS = CP_SOIL_DEFAULT,
                                     epsS = EPS_SOIL_DEFAULT,
                                     Tamb }) {
  const rhoCpInv = 1.0 / (rhoS * cpS);
  const nxy = ny * nx;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const s = j * nx + i;
      for (let k = 0; k < nSoil; k++) {
        const c = k * nxy + s;
        const Tk = Tsoil[c];
        let qTop;
        if (k === 0) {
          // T**4 is float**INT in the reference -- multiplies, not Math.pow.
          const T2 = Tk * Tk;
          qTop = qInSurface[s] - epsS * SIGMA_SB_SOIL * (T2 * T2);
        } else {
          qTop = (kS * (Tsoil[c - nxy] - Tk)) / dAbove[k];
        }
        const qBot = k === nSoil - 1
          ? (kS * (Tk - Tamb)) / dBelow[k]
          : (kS * (Tk - Tsoil[c + nxy])) / dBelow[k];
        Tsoil[c] = Tk + (dt * (qTop - qBot) * rhoCpInv) / soilDz[k];
      }
    }
  }
}

// ── Gas-energy advection ──────────────────────────────────────────────

/**
 * Advect and diffuse T_g in place: dT/dt + u.grad(T) = alpha_th * lap(T).
 *
 * Advection is MUSCL with the minmod limiter (shared with the species
 * transport), diffusion is central in finite-volume form so the stretched dz
 * is handled correctly. No source terms — Q_comb and q_conv belong to the
 * coupling step.
 *
 * The gas is advected SEPARATELY from the coupling for a reason: the coupling
 * kernel only applies point-wise sources to T_g and has no transport in it, so
 * without this call the heat a cell gains never moves downstream.
 *
 * Ghosts: x inlet is T_amb (cold air arriving), x outlet is zero-gradient,
 * y is periodic, z wall and top are zero-flux Neumann.
 *
 * The final floor at T_amb and cap at 10,000 K are numerical safety only. The
 * cap used to be 1900 K (Drysdale grass adiabatic) and was acting as a
 * PHYSICAL clip — it erased the T_g difference between M=4% and M=8% at peak
 * burn, so both radiated forward identically and moisture had no effect on
 * ROS. Raising it let the energy balance set T_g on its own.
 */
export function advGasEnergy(Tg, u, v, w, dt, dx, dy, dzArr,
                             dFaceAbove, dFaceBelow, alphaTh, Tamb,
                             { nx, ny, nz }) {
  const n = nz * ny * nx;
  const nxy = ny * nx;
  const dT = new Float64Array(n);

  advect3dScalarMuscl(Tg, u, v, w, dx, dy, dFaceAbove, dFaceBelow, dT, Tamb,
                      { nx, ny, nz });

  const invDx2 = 1.0 / (dx * dx);
  const invDy2 = 1.0 / (dy * dy);
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      const jp = (j + 1) % ny;
      const jm = (j - 1 + ny) % ny;
      const row = k * nxy + j * nx;
      for (let i = 0; i < nx; i++) {
        const c = row + i;
        // x
        if (i === 0) {
          dT[c] += alphaTh * (Tg[c + 1] - 2.0 * Tg[c] + Tamb) * invDx2;
        } else if (i === nx - 1) {
          dT[c] += alphaTh * (Tg[c - 1] - Tg[c]) * invDx2;
        } else {
          dT[c] += alphaTh * (Tg[c + 1] - 2.0 * Tg[c] + Tg[c - 1]) * invDx2;
        }
        // y, periodic
        dT[c] += alphaTh * (Tg[k * nxy + jp * nx + i] - 2.0 * Tg[c]
                          + Tg[k * nxy + jm * nx + i]) * invDy2;
        // z, finite-volume
        if (k === 0) {
          dT[c] += (alphaTh * ((Tg[c + nxy] - Tg[c]) / dFaceAbove[0])) / dzArr[0];
        } else if (k === nz - 1) {
          dT[c] += (alphaTh * (-(Tg[c] - Tg[c - nxy]) / dFaceBelow[nz - 1])) / dzArr[nz - 1];
        } else {
          dT[c] += (alphaTh * ((Tg[c + nxy] - Tg[c]) / dFaceAbove[k]
                             - (Tg[c] - Tg[c - nxy]) / dFaceBelow[k])) / dzArr[k];
        }
      }
    }
  }

  for (let c = 0; c < n; c++) {
    let T = Tg[c] + dT[c] * dt;
    if (T < Tamb) T = Tamb;
    if (T > 10000.0) T = 10000.0;   // numerical backstop, never binds
    Tg[c] = T;
  }
}

// ── Front tracking ────────────────────────────────────────────────────

/**
 * Front x = the x of the FURTHEST COLUMN holding any burning bed cell [m].
 *
 * A column counts as burning if any (k, j) in it has T_s >= T_ign and fuel.
 * Returns the previous front position (or 0) when nothing is burning, so the
 * front never moves backward through this path.
 *
 * DIAGNOSTIC ONLY when a level set is running, and the caller enforces that by
 * passing append = false. The "any cell >= T_ign" heuristic is mesh-runaway by
 * construction — a finer mesh puts less thermal mass behind each cell, so
 * cells cross T_ign sooner and the apparent front races ahead. It used to
 * override the level set through a max() and had to be untangled (Phase 15F).
 * Cold-flow runs with no bed keep it as the only front source.
 */
export function updateFrontTracking(Ts, alphaS, xMid, TIgn, t,
                                    frontT, frontX, append,
                                    { nx, ny, nZBed }) {
  const nxy = ny * nx;
  const last = frontX.length ? frontX[frontX.length - 1] : 0.0;
  let iMax = -1;
  for (let k = 0; k < nZBed; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = k * nxy + j * nx + i;
        if (Ts[c] >= TIgn && alphaS[c] > 0.0 && i > iMax) iMax = i;
      }
    }
  }
  if (iMax < 0) return last;
  const xFront = xMid[iMax];
  if (append && xFront > last) {
    frontT.push(t);
    frontX.push(xFront);
  }
  return xFront;
}

/** Current ROS from the last few front samples [m/s]. */
export function estimateRos(frontT, frontX) {
  if (frontT.length < 2) return 0.0;
  const n = Math.min(frontT.length, 5);
  const dt = frontT[frontT.length - 1] - frontT[frontT.length - n];
  const dx = frontX[frontX.length - 1] - frontX[frontX.length - n];
  return dt <= 0 ? 0.0 : dx / dt;
}

/**
 * Steady-state ROS from the front history [m/s], with stall rejection.
 *
 * Returns 0 rather than a number when the run did not actually produce a
 * spread: less than 0.1 m past the source, or no advance for 30 s without the
 * fire having reached 70% of the domain. Reporting a small positive ROS for a
 * fire that went out would be worse than reporting nothing.
 *
 * The slope is taken over the FULL history, front[0] to front[-1] against
 * t_end - t[0] — not over a trailing window. A trailing window over-estimates
 * badly whenever the front stalls early: a wet bed that stopped at t = 1.7 s
 * of a 3 s run got a 1.7 s denominator instead of 3 s, inflating ROS by ~75%
 * (Phase 17b).
 */
export function computeSteadyRos(frontT, frontX, tEnd, sourceX, domainM) {
  if (frontT.length < 2) return 0.0;
  const lastX = frontX[frontX.length - 1];
  if (lastX - sourceX < 0.1) return 0.0;
  const tSince = tEnd - frontT[frontT.length - 1];
  const reachedFar = lastX > domainM * 0.7;
  if (tSince > 30.0 && !reachedFar) return 0.0;
  const t0 = frontT[0];
  return tEnd > t0 ? (lastX - frontX[0]) / (tEnd - t0) : 0.0;
}
