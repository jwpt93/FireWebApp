/**
 * DOM radiation — JS port of model_outdoor/physics_3d/dom_3d.py.
 *
 * Discrete Ordinates solution of the RTE on an S4 level-symmetric quadrature
 * (24 ordinates), with step differencing:
 *
 *   I[c] = (kappa*B + |xi|/dx I_x + |eta|/dy I_y + |mu|/dz I_z)
 *          / (kappa + |xi|/dx + |eta|/dy + |mu|/dz)
 *
 * THE ONLY NON-LOCAL KERNEL IN THE PORT. Every other one is a stencil; this
 * sweeps the whole domain in upwind order per ordinate, so each cell depends
 * on the one already computed behind it. That is why the Python marks it
 * `parallel=False` — the sweep ordering IS the algorithm.
 *
 * SOURCE ITERATION. For a purely absorbing-emitting medium one sweep per
 * ordinate is exact. The ground is a diffuse-grey wall with eps_w = 0.85, so
 * its reflected contribution depends on the incident flux, which depends on
 * the intensities — hence iterate, under-relaxed at omega = 0.7, to a
 * relative tolerance on G.
 *
 * ABSORPTION has three contributions, and the second and third are the ones
 * that matter for this model:
 *   kappa_solid  = sigma*alpha_s, scaled by (1 + 5*M) in wet bed — a wet bed
 *                  absorbs more per kg because H2O bands supplement cellulose
 *                  (Mell 2007 WFDS; Linn 2002 FIRETEC).
 *   kappa_gas    = ramped by EITHER active combustion OR hot post-combustion
 *                  gas. The `max` of the two matters: a binary cutoff on
 *                  omega alone used to kill emission from the buoyant plume
 *                  the moment its local omega fell below threshold, even at
 *                  T_g >> 1000 K.
 *   kappa_h2o    = 30 * rho * Y_H2O — water vapour ahead of the front
 *                  absorbing forward radiation that would otherwise preheat
 *                  the bed. This is what closes the Cheney moisture gap.
 *
 * Ny = 1 is assumed periodic in y, which for a single cell means the y
 * upwind neighbour is the cell itself.
 */

const SIGMA_SB = 5.67e-8;
const KAPPA_SOOT_HOT = 0.5;
const OMEGA_COMB_THRESH = 1.0e-3;
const T_GAS_RAD_REF = 600.0;
const T_GAS_RAD_DT = 400.0;
const EPS_W_GROUND = 0.85;
const KAPPA_FLOOR = 1.0e-3;
const A_H2O_RAD = 30.0;          // [m^2/kg] mass-specific H2O extinction
const BETA_KSOLID_WATER = 5.0;   // wet-bed kappa_solid multiplier

/**
 * S4 level-symmetric ordinates (Lathrop & Carlson 1968 Table III), reflected
 * into all eight octants. The level-symmetric constraint 2*mu1^2 + mu2^2 = 1
 * holds: 2(0.295876)^2 + (0.908248)^2 = 1.
 *
 * Only S4 is implemented, matching the Python — S6/S8 need level-index-triplet
 * enumeration and the Python raises NotImplementedError for them.
 */
export function generateSnOrdinates(N = 4) {
  if (N !== 4) {
    throw new Error(`S${N} not implemented; only S4 is supported (matches ` +
      `dom_3d.py, which raises NotImplementedError for S6/S8).`);
  }
  const mu1 = 0.295876;
  const mu2 = 0.908248;
  const perOctant = [[mu1, mu1, mu2], [mu1, mu2, mu1], [mu2, mu1, mu1]];
  const octants = [
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
    [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
  ];
  const om = [];
  for (const [sx, sy, sz] of octants) {
    for (const o of perOctant) om.push([sx * o[0], sy * o[1], sz * o[2]]);
  }
  const M = om.length;
  const w = new Float64Array(M).fill((4.0 * Math.PI) / M);
  return { omega: om, weights: w, M };
}

/**
 * One upwind sweep for a single ordinate. Writes I in place.
 * Sweep direction follows the sign of each direction cosine.
 */
export function sweepOneOrdinate(
  I, kappa, B, xi, eta, mu, dx, dy, dzArr,
  Ileft, Iright, Iback, Ifront, Itop, IgroundIn, yPeriodic,
  { nx, ny, nz },
) {
  const aix = Math.abs(xi);
  const aiy = Math.abs(eta);
  const aiz = Math.abs(mu);
  const invDx = aix / dx;
  const invDy = aiy / dy;
  const nxy = ny * nx;

  const [iS, iE, iD] = xi >= 0 ? [0, nx, 1] : [nx - 1, -1, -1];
  const [jS, jE, jD] = eta >= 0 ? [0, ny, 1] : [ny - 1, -1, -1];
  const [kS, kE, kD] = mu >= 0 ? [0, nz, 1] : [nz - 1, -1, -1];

  for (let k = kS; k !== kE; k += kD) {
    const invDzK = aiz / dzArr[k];
    const kBase = k * nxy;
    for (let j = jS; j !== jE; j += jD) {
      const row = kBase + j * nx;
      for (let i = iS; i !== iE; i += iD) {
        const c = row + i;

        let Ix;
        if (xi >= 0) Ix = i > 0 ? I[c - 1] : Ileft;
        else Ix = i < nx - 1 ? I[c + 1] : Iright;

        let Iy;
        if (eta >= 0) {
          if (j > 0) Iy = I[c - nx];
          else if (yPeriodic) Iy = I[kBase + (ny - 1) * nx + i];
          else Iy = Iback;
        } else if (j < ny - 1) Iy = I[c + nx];
        else if (yPeriodic) Iy = I[kBase + i];
        else Iy = Ifront;

        let Iz;
        if (mu >= 0) Iz = k > 0 ? I[c - nxy] : IgroundIn[j * nx + i];
        else Iz = k < nz - 1 ? I[c + nxy] : Itop;

        const num = kappa[c] * B[c] + invDx * Ix + invDy * Iy + invDzK * Iz;
        const den = kappa[c] + invDx + invDy + invDzK;
        I[c] = num / den;
      }
    }
  }
}

export class DOMRadiationSolver {
  constructor({ nz, ny, nx, dx, dy, dzArr, yBc = 'periodic', nQuadrature = 4,
                epsWGround = EPS_W_GROUND, maxSourceIter = 30,
                tolSourceIter = 1.0e-3, omegaRelax = 0.7,
                kappaGasMax = KAPPA_SOOT_HOT }) {
    this.nz = nz; this.ny = ny; this.nx = nx;
    this.dx = dx; this.dy = dy;
    this.dzArr = Float64Array.from(dzArr);
    this.yPeriodic = yBc === 'periodic';
    this.epsWGround = epsWGround;
    this.maxIter = maxSourceIter;
    this.tol = tolSourceIter;
    this.omegaRelax = omegaRelax;
    this.kappaGasMax = kappaGasMax;

    const q = generateSnOrdinates(nQuadrature);
    this.Omega = q.omega;
    this.weights = q.weights;
    this.M = q.M;

    const n = nz * ny * nx;
    this.Iset = Array.from({ length: this.M }, () => new Float64Array(n));
    this.G = new Float64Array(n);
    this.Gprev = new Float64Array(n);
  }

  /**
   * Fill qRadSolidOut and qRadGasOut [W/m^2] for the current state.
   *
   * @param {object} a  state and outputs; see dom_3d.py :: solve
   */
  solve({
    Ts, Tg, alphaS, omegaComb, sigmaSav, Tamb,
    qRadSolidOut, qRadGasOut, qRadSolidAbsOut = null,
    TsoilSurface = null, qInSoilOut = null,
    YH2O = null, rho = null, bedMoisturePerCell = null,
  }) {
    // qRadSolidAbsOut (optional): the ABSORPTION-only solid channel,
    // kappa_solid * G * dz, WITHOUT the -4*pi*B emission term that
    // qRadSolidOut carries. The bed-particle kernel applies its own
    // Stefan-Boltzmann loss, so handing it the net value double-counts
    // emission. See SOLVER_PORT.md 7.8 defect 4.
    const { nz, ny, nx } = this;
    const n = nz * ny * nx;
    const nxy = ny * nx;

    // ── absorption coefficients ──────────────────────────────────────────
    const kappaSolid = new Float64Array(n);
    const kappaGas = new Float64Array(n);
    const kappa = new Float64Array(n);
    const kappaSafe = new Float64Array(n);
    const B = new Float64Array(n);

    for (let c = 0; c < n; c++) {
      let ks = sigmaSav * alphaS[c];
      if (bedMoisturePerCell) ks *= 1.0 + BETA_KSOLID_WATER * bedMoisturePerCell[c];
      kappaSolid[c] = ks;

      // Either active combustion OR hot gas ramps kappa_gas — the max, not
      // the omega alone, or the plume stops emitting the moment omega dips.
      let of = omegaComb[c] / OMEGA_COMB_THRESH;
      if (of < 0.0) of = 0.0; else if (of > 1.0) of = 1.0;
      let tf = (Tg[c] - T_GAS_RAD_REF) / T_GAS_RAD_DT;
      if (tf < 0.0) tf = 0.0; else if (tf > 1.0) tf = 1.0;
      let kg = this.kappaGasMax * Math.max(of, tf);
      if (YH2O && rho) kg += A_H2O_RAD * rho[c] * YH2O[c];
      kappaGas[c] = kg;

      const kt = ks + kg;
      kappa[c] = kt;
      kappaSafe[c] = kt > KAPPA_FLOOR ? kt : KAPPA_FLOOR;

      // Blackbody source, blending solid and gas temperatures by their
      // absorption contribution.
      let Trad4;
      if (kt > 1.0e-9) {
        const Ts2 = Ts[c] * Ts[c];
        const Tg2 = Tg[c] * Tg[c];
        Trad4 = (ks * Ts2 * Ts2 + kg * Tg2 * Tg2) / kappaSafe[c];
      } else {
        const Ta2 = Tamb * Tamb;
        Trad4 = Ta2 * Ta2;
      }
      B[c] = (SIGMA_SB * Trad4) / Math.PI;
    }

    const Tamb2 = Tamb * Tamb;
    const Iamb = (SIGMA_SB * Tamb2 * Tamb2) / Math.PI;

    const IsoilEmit = new Float64Array(nxy);
    for (let s = 0; s < nxy; s++) {
      IsoilEmit[s] = TsoilSurface
        ? (this.epsWGround * SIGMA_SB * TsoilSurface[s] ** 4) / Math.PI
        : this.epsWGround * Iamb;
    }

    let IgroundOut = Float64Array.from(IsoilEmit);
    const zeroGround = new Float64Array(nxy);
    const qInGround = new Float64Array(nxy);

    for (let it = 0; it < this.maxIter; it++) {
      this.Gprev.set(this.G);
      this.G.fill(0.0);

      for (let m = 0; m < this.M; m++) {
        const [xi, eta, mu] = this.Omega[m];
        sweepOneOrdinate(
          this.Iset[m], kappaSafe, B, xi, eta, mu, this.dx, this.dy, this.dzArr,
          Iamb, Iamb, Iamb, Iamb, Iamb,
          mu > 0 ? IgroundOut : zeroGround, this.yPeriodic,
          { nx, ny, nz });
      }

      // G = sum_n w_n I_n
      for (let c = 0; c < n; c++) {
        let s = 0.0;
        for (let m = 0; m < this.M; m++) s += this.weights[m] * this.Iset[m][c];
        this.G[c] = s;
      }

      // Downward flux at the ground, for the diffuse-grey wall BC.
      qInGround.fill(0.0);
      for (let m = 0; m < this.M; m++) {
        const mu = this.Omega[m][2];
        if (mu < 0) {
          const wa = this.weights[m] * Math.abs(mu);
          for (let s = 0; s < nxy; s++) qInGround[s] += wa * this.Iset[m][s];
        }
      }

      for (let s = 0; s < nxy; s++) {
        const newI = IsoilEmit[s] + ((1.0 - this.epsWGround) / Math.PI) * qInGround[s];
        IgroundOut[s] = this.omegaRelax * newI + (1.0 - this.omegaRelax) * IgroundOut[s];
      }

      if (it > 0) {
        let gMax = 1e-12;
        for (let c = 0; c < n; c++) if (this.G[c] > gMax) gMax = this.G[c];
        let err = 0.0;
        for (let c = 0; c < n; c++) {
          const d = Math.abs(this.G[c] - this.Gprev[c]);
          if (d > err) err = d;
        }
        if (err / gMax < this.tol) break;
      }
    }

    // div.q_rad = kappa (4 pi B - G), per unit horizontal area -> x dz
    for (let k = 0; k < nz; k++) {
      const dz = this.dzArr[k];
      for (let s = 0; s < nxy; s++) {
        const c = k * nxy + s;
        const netVol = kappa[c] * (this.G[c] - 4.0 * Math.PI * B[c]);
        const netPerHoriz = netVol * dz;
        const fSolid = kappa[c] > 1.0e-9 ? kappaSolid[c] / kappaSafe[c] : 0.0;
        const fGas = kappa[c] > 1.0e-9 ? kappaGas[c] / kappaSafe[c] : 0.0;
        qRadSolidOut[c] = netPerHoriz * fSolid;
        qRadGasOut[c] = netPerHoriz * fGas;
        if (qRadSolidAbsOut) {
          // Absorption only: kappa_solid * G * dz. No -4*pi*B term.
          qRadSolidAbsOut[c] = kappa[c] * this.G[c] * dz * fSolid;
        }
      }
    }

    if (qInSoilOut) {
      for (let s = 0; s < nxy; s++) qInSoilOut[s] = this.epsWGround * qInGround[s];
    }
  }
}
