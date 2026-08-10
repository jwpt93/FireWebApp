/**
 * Realizable k-epsilon turbulence — JS port of
 * model_outdoor/physics_3d/turbulence_3d.py ::
 * _strain_and_vorticity_squared + step_k_epsilon.
 *
 * Shih et al. (1995) realizable C_mu, Henkes et al. (1991) buoyancy
 * correction, Yakhot-Orszag (1986) RNG correction on C_2eps, Sanz (2003)
 * canopy production/dissipation, Menter (2003) production limiter, Rodi
 * (1987) clamp on the buoyancy term in the k equation.
 *
 *   C_mu = 1 / (A_0 + A_S U* k/eps),  U* = sqrt((|S|^2 + |Omega|^2)/2)
 *   nu_t = C_mu k^2 / eps
 *
 * WHY REALIZABLE. Standard k-epsilon over-predicts nu_t in high-strain and
 * high-rotation regions; the realizable C_mu self-limits there, so no
 * artificial viscosity cap is needed.
 *
 * DOUBLE-BUFFERED, and that is load-bearing rather than tidy. The kernel
 * reads k and eps at k+/-1 and k+/-2 for MUSCL z-advection, so writing back
 * in place under a prange-over-k would let one thread read a neighbour
 * another has already updated. The Python hit exactly that race (Phase
 * 14q-D1) and fixed it this way; the port keeps the same structure.
 *
 * BOUNDARY FIX. The buoyancy gradient dT/dz used to read T_g[k+1] and
 * T_g[k-1] unconditionally. numba does not bounds-check, so at k=0 the k-1
 * read wrapped to the domain TOP and at k=Nz-1 the k+1 read went past the end
 * of the array into adjacent memory. Found while writing this port and fixed
 * upstream on 2026-08-10: one-sided differences at both boundaries, matching
 * the convention _strain_and_vorticity_squared already used. Effect on the
 * validated case: ROS_Ts 34.359 -> 33.818 m/min (-1.57%), band verdict
 * unchanged. This port implements the FIXED behaviour.
 *
 * THREE PATHS ARE NOT PORTED because they are dead in this configuration:
 * BVG buoyant vorticity generation (bvg_factor is hardcoded 0.0 at the call
 * site), the L-min realizability cap and the Durbin time-scale bound (both
 * default 0.0 and neither deck nor worker sets them). They throw if enabled
 * rather than silently doing nothing, so a future caller cannot switch them
 * on and get wrong answers quietly.
 */
import { musclFaceValue } from './muscl.js';

// ── closure constants ──────────────────────────────────────────────────────
const NU_GAS = 1.5e-5;      // [m^2/s] at 300 K
const G = 9.81;
const C_MU = 0.09;
const C_1EPS = 1.44;
const C_2EPS = 1.92;
const SIGMA_K = 1.0;
const SIGMA_EPS = 1.3;
const PR_T = 0.85;
const K_MIN = 1.0e-8;
const EPS_MIN = 1.0e-8;
const NU_T_MIN = 0.0;
const A_0_REAL = 4.04;
const A_S_REAL = 4.5;
const C_LIM_P = 10.0;       // Menter production limiter
const U_TINY_HENKES = 0.01; // [m/s]
const C_D_DRAG = 1.0;
const C_EPS4_CANOPY = 0.9;
const C_EPS5_CANOPY = 0.9;
const ETA0 = 4.38;
const BETA_RNG = 0.012;

/**
 * |S|^2 = 2 S_ij S_ij and |Omega|^2 = 2 Omega_ij Omega_ij at cell centres.
 * Both are needed for the realizable formulation.
 *
 * Ghosts: x inlet from u_inlet with v = w = 0; x outlet zero-gradient;
 * z wall all-zero (no-slip); z top zero-gradient; y periodic. z uses a
 * one-sided distance at both boundaries.
 */
export function strainAndVorticitySquared(
  u, v, w, dx, dy, dzArr, dFaceAbove, dFaceBelow, Smag2, Omag2, uInlet,
  { nx, ny, nz },
) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const nxy = ny * nx;

  for (let k = 0; k < nz; k++) {
    let invDzCentral;
    if (k === 0) invDzCentral = 1.0 / dFaceAbove[0];
    else if (k === nz - 1) invDzCentral = 1.0 / dFaceBelow[nz - 1];
    else invDzCentral = 1.0 / (dFaceBelow[k] + dFaceAbove[k]);
    const kBase = k * nxy;

    for (let j = 0; j < ny; j++) {
      const jm1 = (((j - 1) % ny) + ny) % ny;
      const jp1 = (j + 1) % ny;
      const row = kBase + j * nx;
      const rjm1 = kBase + jm1 * nx;
      const rjp1 = kBase + jp1 * nx;
      const inl = k * ny + j;

      for (let i = 0; i < nx; i++) {
        const c = row + i;
        const ui = u[c], vi = v[c], wi = w[c];

        let uxL, vxL, wxL;
        if (i === 0) { uxL = uInlet[inl]; vxL = 0.0; wxL = 0.0; }
        else { uxL = u[c - 1]; vxL = v[c - 1]; wxL = w[c - 1]; }
        let uxR, vxR, wxR;
        if (i === nx - 1) { uxR = ui; vxR = vi; wxR = wi; }
        else { uxR = u[c + 1]; vxR = v[c + 1]; wxR = w[c + 1]; }
        let uzL, vzL, wzL;
        if (k === 0) { uzL = 0.0; vzL = 0.0; wzL = 0.0; }
        else { uzL = u[c - nxy]; vzL = v[c - nxy]; wzL = w[c - nxy]; }
        let uzR, vzR, wzR;
        if (k === nz - 1) { uzR = ui; vzR = vi; wzR = wi; }
        else { uzR = u[c + nxy]; vzR = v[c + nxy]; wzR = w[c + nxy]; }

        const dudx = (uxR - uxL) * 0.5 * invDx;
        const dudy = (u[rjp1 + i] - u[rjm1 + i]) * 0.5 * invDy;
        const dudz = (uzR - uzL) * invDzCentral;
        const dvdx = (vxR - vxL) * 0.5 * invDx;
        const dvdy = (v[rjp1 + i] - v[rjm1 + i]) * 0.5 * invDy;
        const dvdz = (vzR - vzL) * invDzCentral;
        const dwdx = (wxR - wxL) * 0.5 * invDx;
        const dwdy = (w[rjp1 + i] - w[rjm1 + i]) * 0.5 * invDy;
        const dwdz = (wzR - wzL) * invDzCentral;

        const S11 = dudx, S22 = dvdy, S33 = dwdz;
        const S12 = 0.5 * (dudy + dvdx);
        const S13 = 0.5 * (dudz + dwdx);
        const S23 = 0.5 * (dvdz + dwdy);
        Smag2[c] = 2.0 * (S11 * S11 + S22 * S22 + S33 * S33)
                 + 4.0 * (S12 * S12 + S13 * S13 + S23 * S23);

        const O12 = 0.5 * (dudy - dvdx);
        const O13 = 0.5 * (dudz - dwdx);
        const O23 = 0.5 * (dvdz - dwdy);
        Omag2[c] = 4.0 * (O12 * O12 + O13 * O13 + O23 * O23);
      }
    }
  }
}

/**
 * Advance k and epsilon by one step and write nu_t. All three are updated.
 *
 * @param {object} opts {nx, ny, nz, bvgFactor?, epsRealizLMin?,
 *                       epsRealizDurbinAlpha?}
 */
export function stepKEpsilon(
  kTurb, epsTurb, nuTOut, u, v, w, Tg, rho, alphaS, sigmaSav, dt, dx, dy,
  dzArr, dFaceAbove, dFaceBelow, Tamb, Smag2Work, Omag2Work, uInlet,
  kWallGhost, epsWallGhost, betaPCanopy, betaDCanopy,
  { nx, ny, nz, bvgFactor = 0.0, epsRealizLMin = 0.0,
    epsRealizDurbinAlpha = 0.0 } = {},
) {
  if (bvgFactor !== 0.0) {
    throw new Error('stepKEpsilon: bvgFactor is not ported (dead at the ' +
      'call site, hardcoded 0.0). Port the Sandia SAND2005-6273 BVG term ' +
      'before enabling it.');
  }
  if (epsRealizLMin !== 0.0 || epsRealizDurbinAlpha !== 0.0) {
    throw new Error('stepKEpsilon: the Phase 15E realizability caps are not ' +
      'ported (both default 0.0 and unset in this configuration).');
  }

  strainAndVorticitySquared(u, v, w, dx, dy, dzArr, dFaceAbove, dFaceBelow,
                            Smag2Work, Omag2Work, uInlet, { nx, ny, nz });

  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const invDx2 = invDx * invDx;
  const invDy2 = invDy * invDy;
  const nxy = ny * nx;

  const kNewBuf = new Float64Array(kTurb.length);
  const eNewBuf = new Float64Array(epsTurb.length);

  for (let k = 0; k < nz; k++) {
    const invDzK = 1.0 / dzArr[k];
    const invDAbove = 1.0 / dFaceAbove[k];
    const invDBelow = 1.0 / dFaceBelow[k];
    // One-sided at the boundaries — see the BOUNDARY FIX note above.
    let invDzCentral;
    if (k === 0) invDzCentral = 1.0 / dFaceAbove[0];
    else if (k === nz - 1) invDzCentral = 1.0 / dFaceBelow[nz - 1];
    else invDzCentral = 1.0 / (dFaceAbove[k] + dFaceBelow[k]);
    const kBase = k * nxy;

    for (let j = 0; j < ny; j++) {
      const jm2 = (((j - 2) % ny) + ny) % ny;
      const jm1 = (((j - 1) % ny) + ny) % ny;
      const jp1 = (j + 1) % ny;
      const jp2 = (j + 2) % ny;
      const row = kBase + j * nx;
      const rjm2 = kBase + jm2 * nx;
      const rjm1 = kBase + jm1 * nx;
      const rjp1 = kBase + jp1 * nx;
      const rjp2 = kBase + jp2 * nx;
      const wall = j * nx;

      for (let i = 0; i < nx; i++) {
        const c = row + i;
        let kLoc = kTurb[c];
        let eLoc = epsTurb[c];
        if (kLoc < K_MIN) kLoc = K_MIN;
        if (eLoc < EPS_MIN) eLoc = EPS_MIN;

        // Way B ghosts: inlet is quiescent (MIN), outlet/top zero-gradient,
        // wall uses the Launder-Spalding equilibrium ghosts.
        const kxL = i === 0 ? K_MIN : kTurb[c - 1];
        const exL = i === 0 ? EPS_MIN : epsTurb[c - 1];
        const kxR = i === nx - 1 ? kLoc : kTurb[c + 1];
        const exR = i === nx - 1 ? eLoc : epsTurb[c + 1];
        const kzL = k === 0 ? kWallGhost[wall + i] : kTurb[c - nxy];
        const ezL = k === 0 ? epsWallGhost[wall + i] : epsTurb[c - nxy];
        const kzR = k === nz - 1 ? kLoc : kTurb[c + nxy];
        const ezR = k === nz - 1 ? eLoc : epsTurb[c + nxy];

        const Smag2 = Smag2Work[c];
        const Omag2 = Omag2Work[c];
        const Ustar = Math.sqrt(0.5 * (Smag2 + Omag2));
        const CmuReal = 1.0 / (A_0_REAL + (A_S_REAL * Ustar * kLoc) / eLoc);

        let nuT = (CmuReal * kLoc * kLoc) / eLoc;
        if (nuT < NU_T_MIN) nuT = NU_T_MIN;

        const Pk = nuT * Smag2;
        // Menter limiter, applied to BOTH the k and eps sources. A no-op at
        // equilibrium; it only bites during transients such as
        // combustion-driven momentum spikes.
        let PkLim = Pk;
        if (PkLim > C_LIM_P * eLoc) PkLim = C_LIM_P * eLoc;

        let dTdz;
        if (k === 0) dTdz = (Tg[c + nxy] - Tg[c]) * invDzCentral;
        else if (k === nz - 1) dTdz = (Tg[c] - Tg[c - nxy]) * invDzCentral;
        else dTdz = (Tg[c + nxy] - Tg[c - nxy]) * invDzCentral;

        let TForBuoy = Tg[c];
        if (TForBuoy < Tamb) TForBuoy = Tamb;
        let Gk = (nuT / PR_T) * (G / TForBuoy) * dTdz;
        if (Gk < 0.0) Gk = 0.0;      // stably stratified: no k production
        // Rodi clamp, k equation only. Without it, in horizontal shear where
        // Henkes C_3eps ~ 0, G_k drives k unbounded while eps cannot catch up.
        let GkForK = Gk;
        if (GkForK > PkLim) GkForK = PkLim;

        const ui = u[c], vi = v[c], wi = w[c];
        let uH = Math.sqrt(ui * ui + vi * vi);
        if (uH < U_TINY_HENKES) uH = U_TINY_HENKES;
        const C3eps = Math.tanh(Math.abs(wi) / uH);

        // Sanz canopy: drag work both produces and dissipates TKE.
        const aS = alphaS[c];
        let PkCanopy = 0.0;
        let DkCanopy = 0.0;
        if (aS > 0.0) {
          const aV = sigmaSav * aS;
          const speed = Math.sqrt(ui * ui + vi * vi + wi * wi);
          const cdAvSpeed = C_D_DRAG * aV * speed;
          PkCanopy = betaPCanopy * cdAvSpeed * speed * speed;
          DkCanopy = betaDCanopy * cdAvSpeed * kLoc;
        }

        // ── k transport ──────────────────────────────────────────────────
        let advKx;
        if (i >= 2 && i <= nx - 3) {
          const fp = musclFaceValue(kTurb[c - 1], kLoc, kTurb[c + 1], kTurb[c + 2], ui);
          const fm = musclFaceValue(kTurb[c - 2], kTurb[c - 1], kLoc, kTurb[c + 1], ui);
          advKx = ui * (fp - fm) * invDx;
        } else if (ui >= 0.0) advKx = ui * (kLoc - kxL) * invDx;
        else advKx = ui * (kxR - kLoc) * invDx;

        const kfp = musclFaceValue(kTurb[rjm1 + i], kLoc, kTurb[rjp1 + i], kTurb[rjp2 + i], vi);
        const kfm = musclFaceValue(kTurb[rjm2 + i], kTurb[rjm1 + i], kLoc, kTurb[rjp1 + i], vi);
        const advKy = vi * (kfp - kfm) * invDy;

        let advKz;
        if (k >= 2 && k <= nz - 3) {
          const fp = musclFaceValue(kTurb[c - nxy], kLoc, kTurb[c + nxy], kTurb[c + 2 * nxy], wi);
          const fm = musclFaceValue(kTurb[c - 2 * nxy], kTurb[c - nxy], kLoc, kTurb[c + nxy], wi);
          advKz = (wi * (fp - fm)) / (0.5 * (dFaceAbove[k] + dFaceBelow[k]));
        } else if (wi >= 0.0) advKz = wi * (kLoc - kzL) * invDBelow;
        else advKz = wi * (kzR - kLoc) * invDAbove;

        const advK = advKx + advKy + advKz;
        const alphaK = NU_GAS + nuT / SIGMA_K;
        const d2kx = (kxR - 2.0 * kLoc + kxL) * invDx2;
        const d2ky = (kTurb[rjp1 + i] - 2.0 * kLoc + kTurb[rjm1 + i]) * invDy2;
        const d2kz = ((kzR - kLoc) * invDAbove - (kLoc - kzL) * invDBelow) * invDzK;
        const diffK = alphaK * (d2kx + d2ky + d2kz);

        // Implicit destruction keeps this stable under stiff transient sources.
        const SposK = PkLim + GkForK + PkCanopy;   // + G_B, which is 0 here
        const SnegK = eLoc + DkCanopy;
        let kNew = (kLoc + (-advK + diffK + SposK) * dt) / (1.0 + (SnegK * dt) / kLoc);
        if (kNew < K_MIN) kNew = K_MIN;

        // ── epsilon transport ────────────────────────────────────────────
        let advEx;
        if (i >= 2 && i <= nx - 3) {
          const fp = musclFaceValue(epsTurb[c - 1], eLoc, epsTurb[c + 1], epsTurb[c + 2], ui);
          const fm = musclFaceValue(epsTurb[c - 2], epsTurb[c - 1], eLoc, epsTurb[c + 1], ui);
          advEx = ui * (fp - fm) * invDx;
        } else if (ui >= 0.0) advEx = ui * (eLoc - exL) * invDx;
        else advEx = ui * (exR - eLoc) * invDx;

        const efp = musclFaceValue(epsTurb[rjm1 + i], eLoc, epsTurb[rjp1 + i], epsTurb[rjp2 + i], vi);
        const efm = musclFaceValue(epsTurb[rjm2 + i], epsTurb[rjm1 + i], eLoc, epsTurb[rjp1 + i], vi);
        const advEy = vi * (efp - efm) * invDy;

        let advEz;
        if (k >= 2 && k <= nz - 3) {
          const fp = musclFaceValue(epsTurb[c - nxy], eLoc, epsTurb[c + nxy], epsTurb[c + 2 * nxy], wi);
          const fm = musclFaceValue(epsTurb[c - 2 * nxy], epsTurb[c - nxy], eLoc, epsTurb[c + nxy], wi);
          advEz = (wi * (fp - fm)) / (0.5 * (dFaceAbove[k] + dFaceBelow[k]));
        } else if (wi >= 0.0) advEz = wi * (eLoc - ezL) * invDBelow;
        else advEz = wi * (ezR - eLoc) * invDAbove;

        const advE = advEx + advEy + advEz;
        const alphaEps = NU_GAS + nuT / SIGMA_EPS;
        const d2ex = (exR - 2.0 * eLoc + exL) * invDx2;
        const d2ey = (epsTurb[rjp1 + i] - 2.0 * eLoc + epsTurb[rjm1 + i]) * invDy2;
        const d2ez = ((ezR - eLoc) * invDAbove - (eLoc - ezL) * invDBelow) * invDzK;
        const diffE = alphaEps * (d2ex + d2ey + d2ez);

        // RNG correction to C_2eps. Standard C_MU here by the original RNG
        // derivation — the realizable C_mu above affects only nu_t.
        const Smag = Math.sqrt(Smag2);
        const etaRng = (Smag * kLoc) / eLoc;
        const eta3 = etaRng * etaRng * etaRng;   // numba lowers **3 to multiplies
        let rngCorr = (C_MU * eta3 * (1.0 - etaRng / ETA0)) / (1.0 + BETA_RNG * eta3);
        if (rngCorr < 0.0) rngCorr = 0.0;        // only ADD dissipation
        const C2eff = C_2EPS + rngCorr;
        const ekRatio = eLoc / kLoc;

        // Henkes C_3eps weights buoyancy in the eps equation, using the
        // UNCLAMPED G_k so vertical-plume buoyancy still reaches eps.
        const SposEps = C_1EPS * (PkLim + C3eps * Gk) * ekRatio
                      + C_EPS4_CANOPY * PkCanopy * ekRatio;
        const SnegEps = C2eff * eLoc * ekRatio + C_EPS5_CANOPY * DkCanopy * ekRatio;
        let eNew = (eLoc + (-advE + diffE + SposEps) * dt) / (1.0 + (SnegEps * dt) / eLoc);
        if (eNew < EPS_MIN) eNew = EPS_MIN;

        kNewBuf[c] = kNew;
        eNewBuf[c] = eNew;

        const CmuNew = 1.0 / (A_0_REAL + (A_S_REAL * Ustar * kNew) / eNew);
        let nuTNew = (CmuNew * kNew * kNew) / eNew;
        if (nuTNew < NU_T_MIN) nuTNew = NU_T_MIN;
        nuTOut[c] = nuTNew;
      }
    }
  }

  kTurb.set(kNewBuf);
  epsTurb.set(eNewBuf);
}
