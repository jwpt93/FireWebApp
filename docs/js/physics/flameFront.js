/**
 * Level-set fire front + flame geometry — JS port of
 * model_outdoor/physics_3d/flame_front_3d.py.
 *
 * Two DIFFERENT level sets live here and they are not interchangeable. Keeping
 * them separate was a Phase 14y fix; before that a single mask served both and
 * it broke at high wind, where the plume tilts far downstream of the bed front.
 *
 *   phi (LevelSetFront3D) — KINEMATIC, tracks the bed-pyrolysis front. Driven
 *     by v_n from the forward heat flux. Its ahead-of-front band is the
 *     DESTINATION for forward radiation.
 *
 *   phi_flame — STATE-DERIVED, a signed distance to wherever the gas is
 *     actually reacting right now. Its interior is the flame body, the SOURCE.
 *     Bed can pyrolyse with no flame above it, and flame can sit over bed that
 *     already burned. Different objects.
 *
 * WHAT IS PORTED. The Frankman flame-tip convection kernel is NOT: it is
 * disabled upstream (`q_frankman_3d.fill(0.0)` in the main loop) as a
 * phenomenological non-local shortcut that double-counted the ordinary
 * gas-solid coupling. The legacy 2D `compute_q_in_at_front` / `compute_v_n`
 * pair is also skipped — the loop calls the 3D versions. Both are left out
 * rather than ported dead.
 *
 * THE EUCLIDEAN DISTANCE TRANSFORM. `compute_phi_flame_from_state` leans on
 * scipy.ndimage.distance_transform_edt, which has no browser equivalent, so
 * this file carries its own. It uses the Felzenszwalb & Huttenlocher (2012)
 * separable lower-envelope algorithm: one 1D pass per axis over squared
 * distance, each parabola contributing (sampling_axis * (p - q))^2. That is
 * EXACT Euclidean, the same guarantee scipy gives, not an approximation like
 * a chamfer or two-pass mask — so the two agree to floating point rather than
 * to a few percent. It is also O(N) per axis.
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i. Masks are Uint8Array.
 */

// ── Constants ─────────────────────────────────────────────────────────

/** [m] Albini 1985 grass burning-zone length. */
export const L_BURNOUT_M = 0.50;

/** [m] WFDS Mell 2007 §3.4 preheating band. Legacy fixed value — the loop
 *  uses the wind-dependent flameTiltBandM instead. */
export const DX_VN_BAND_M = 0.20;

export const G_ACCEL = 9.81;

/** Cheney 1993: U_10 -> U_1.5 midflame wind reduction. */
export const WIND_MIDFLAME_FRAC = 0.723;

export const LEVELSET_REINIT_INTERVAL = 10;  // outer steps between reinits
export const REINIT_SUBSTEPS = 5;            // Sussman substeps per reinit
export const DT_REINIT_FRAC = 0.5;           // tau-step as a fraction of h_min
export const PHI_INIT_UNBURNED = 100.0;      // far outside any narrow band

/** Active-flame criterion thresholds. */
export const OMEGA_MIN_FLAME = 1.0e-3;   // [kg/m^3/s] reaction zone
export const T_PLUME_MIN = 1000.0;       // [K] plume tail
export const Y_F_MIN_PLUME = 1.0e-3;     // [-] fuel-bearing plume

/** Bootstrap heating, used only by the legacy ebu_bootstrap closure. */
export const Q_BOOTSTRAP_W_M3 = 500000.0;   // Pyne 1993 §11.3 drip-torch scale
export const T_BOOTSTRAP_S = 2.0;

export const L_VAP_WATER = 2.26e6;   // [J/kg]

/**
 * Albini 1981 flame-tilt projected band length [m].
 *
 * A buoyant flame leaning downwind projects onto the unburned bed over a
 * horizontal extent L_flame*sin(theta), where tan(theta) = U_mid/sqrt(2 g L).
 * At zero wind the flame stands up and the band vanishes; at high wind theta
 * approaches 90 degrees and the band saturates at the full flame length.
 *
 * U_mid is the wind at half-flame height — the deck's U_10 times Cheney's
 * 0.723 reduction, NOT U_10 itself.
 *
 * Albini, F.A. (1981) Combustion and Flame 43:155. Nelson (2002) IJWF 11:153
 * gives the same form. Cheney et al. (1993) IJWF 3:31 for the wind reduction.
 *
 * Math.pow rather than Math.sqrt on purpose: the reference is plain Python,
 * not numba, so `** 0.5` goes through C pow() there too.
 */
export function flameTiltBandM(u10, LFlame = L_BURNOUT_M, g = G_ACCEL,
                               midflameFrac = WIND_MIDFLAME_FRAC) {
  const uBuoy = Math.pow(2.0 * g * LFlame, 0.5);   // ~3.13 m/s at L=0.5
  const uMid = Math.max(u10, 0.0) * midflameFrac;
  const tanTh = uMid / uBuoy;
  const sinTh = tanTh / Math.pow(1.0 + tanTh * tanTh, 0.5);
  return LFlame * sinTh;
}

// ── Godunov |grad phi| ────────────────────────────────────────────────

/** Per-k vertical inverse spacings, one-sided at the two ends. */
function dzInv(k, nz, dzArr) {
  if (k === 0) {
    return [1.0 / dzArr[0],
            nz > 1 ? 2.0 / (dzArr[0] + dzArr[1]) : 1.0 / dzArr[0]];
  }
  if (k === nz - 1) {
    return [2.0 / (dzArr[nz - 1] + dzArr[nz - 2]), 1.0 / dzArr[nz - 1]];
  }
  return [2.0 / (dzArr[k] + dzArr[k - 1]), 2.0 / (dzArr[k] + dzArr[k + 1])];
}

/**
 * |grad phi| by Godunov upwind, for advection in the normal direction
 * (Sethian 1999 §6.4).
 *
 * For v_n > 0 the front advances and phi decreases through it, so the stable
 * choice is  |grad phi|^2 = sum over axes of max(D-,0)^2 + min(D+,0)^2.
 *
 * x is one-sided at the domain edges (D = 0 there), y is periodic, z is
 * one-sided at top and bottom.
 */
export function godunovGradNorm(phi, dx, dy, dzArr, gradOut, { nx, ny, nz }) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const nxy = ny * nx;

  for (let k = 0; k < nz; k++) {
    const [invDzM, invDzP] = dzInv(k, nz, dzArr);
    for (let j = 0; j < ny; j++) {
      const jm = j > 0 ? j - 1 : ny - 1;
      const jp = j < ny - 1 ? j + 1 : 0;
      for (let i = 0; i < nx; i++) {
        const c = k * nxy + j * nx + i;
        const phiC = phi[c];

        const dmX = i > 0 ? (phiC - phi[c - 1]) * invDx : 0.0;
        const dpX = i < nx - 1 ? (phi[c + 1] - phiC) * invDx : 0.0;
        const dmY = (phiC - phi[k * nxy + jm * nx + i]) * invDy;
        const dpY = (phi[k * nxy + jp * nx + i] - phiC) * invDy;
        const dmZ = k > 0 ? (phiC - phi[c - nxy]) * invDzM : 0.0;
        const dpZ = k < nz - 1 ? (phi[c + nxy] - phiC) * invDzP : 0.0;

        const gxp = Math.max(dmX, 0.0), gxm = Math.min(dpX, 0.0);
        const gyp = Math.max(dmY, 0.0), gym = Math.min(dpY, 0.0);
        const gzp = Math.max(dmZ, 0.0), gzm = Math.min(dpZ, 0.0);

        // `** 0.5` -- numba lowers that to sqrt, so Math.sqrt is right here
        // (unlike flameTiltBandM above, which is plain Python).
        gradOut[c] = Math.sqrt(gxp * gxp + gxm * gxm
                             + gyp * gyp + gym * gym
                             + gzp * gzp + gzm * gzm);
      }
    }
  }
}

/**
 * |grad phi| with sign-aware Godunov upwind — the Sussman (1994) reinit step.
 *
 * The characteristic direction flips with the side of the interface, and that
 * flip is what drives |grad phi| toward 1:
 *   sign > 0 (unburned): max(D-,0)^2 + min(D+,0)^2
 *   sign < 0 (burned):   min(D-,0)^2 + max(D+,0)^2
 */
export function reinitGodunovGrad(phi, phi0, dx, dy, dzArr, gradOut,
                                  { nx, ny, nz }) {
  const invDx = 1.0 / dx;
  const invDy = 1.0 / dy;
  const nxy = ny * nx;

  for (let k = 0; k < nz; k++) {
    const [invDzM, invDzP] = dzInv(k, nz, dzArr);
    for (let j = 0; j < ny; j++) {
      const jm = j > 0 ? j - 1 : ny - 1;
      const jp = j < ny - 1 ? j + 1 : 0;
      for (let i = 0; i < nx; i++) {
        const c = k * nxy + j * nx + i;
        const phiC = phi[c];
        const s = phi0[c] > 0.0 ? 1.0 : (phi0[c] < 0.0 ? -1.0 : 0.0);

        const dmX = i > 0 ? (phiC - phi[c - 1]) * invDx : 0.0;
        const dpX = i < nx - 1 ? (phi[c + 1] - phiC) * invDx : 0.0;
        const dmY = (phiC - phi[k * nxy + jm * nx + i]) * invDy;
        const dpY = (phi[k * nxy + jp * nx + i] - phiC) * invDy;
        const dmZ = k > 0 ? (phiC - phi[c - nxy]) * invDzM : 0.0;
        const dpZ = k < nz - 1 ? (phi[c + nxy] - phiC) * invDzP : 0.0;

        let gxp, gxm, gyp, gym, gzp, gzm;
        if (s > 0.0) {
          gxp = Math.max(dmX, 0.0); gxm = Math.min(dpX, 0.0);
          gyp = Math.max(dmY, 0.0); gym = Math.min(dpY, 0.0);
          gzp = Math.max(dmZ, 0.0); gzm = Math.min(dpZ, 0.0);
        } else {
          gxp = Math.min(dmX, 0.0); gxm = Math.max(dpX, 0.0);
          gyp = Math.min(dmY, 0.0); gym = Math.max(dpY, 0.0);
          gzp = Math.min(dmZ, 0.0); gzm = Math.max(dpZ, 0.0);
        }

        gradOut[c] = Math.sqrt(gxp * gxp + gxm * gxm
                             + gyp * gyp + gym * gym
                             + gzp * gzp + gzm * gzm);
      }
    }
  }
}

// ── The kinematic bed front ───────────────────────────────────────────

export class LevelSetFront3D {
  constructor({ nz, ny, nx, dx, dy, dzArr, LBurnout = L_BURNOUT_M }) {
    Object.assign(this, { nz, ny, nx, dx, dy });
    this.dzArr = Float64Array.from(dzArr);
    this.LBurnout = LBurnout;
    this.n = nz * ny * nx;
    this.phi = new Float64Array(this.n).fill(PHI_INIT_UNBURNED);
    this._grad = new Float64Array(this.n);
    this._phi0 = new Float64Array(this.n);
    this._step = 0;
  }

  /**
   * Signed-distance field for an x-strip source patch.
   *
   * Source bed cells start pre-burned at -L_burnout/2, well inside the burned
   * region. Everything else takes the signed distance to the patch's leading
   * edge, floored at -L_burnout/2 so cells behind the source do not report an
   * arbitrarily large negative distance.
   */
  initializeSourcePatch(iStart, iEnd, kTopBed, xMid) {
    const { nz, ny, nx, dx } = this;
    const xFront = xMid[iEnd - 1] + 0.5 * dx;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const c = (k * ny + j) * nx + i;
          if (i >= iStart && i < iEnd && k <= kTopBed) {
            this.phi[c] = -this.LBurnout / 2.0;
          } else {
            const d = xMid[i] - xFront;
            this.phi[c] = d > 0.0 ? d : Math.max(d, -this.LBurnout / 2.0);
          }
        }
      }
    }
  }

  /** One step of  d(phi)/dt + v_n |grad phi| = 0,  with v_n >= 0. */
  evolve(dt, vnField) {
    const { nx, ny, nz } = this;
    godunovGradNorm(this.phi, this.dx, this.dy, this.dzArr, this._grad,
                    { nx, ny, nz });
    for (let c = 0; c < this.n; c++) {
      this.phi[c] -= dt * vnField[c] * this._grad[c];
    }
    this._step += 1;
  }

  /**
   * Restore |grad phi| = 1 in the narrow band, Sussman (1994):
   * d(phi)/d(tau) + sign(phi_0)(|grad phi| - 1) = 0, five substeps.
   */
  reinitialize() {
    const { nx, ny, nz } = this;
    this._phi0.set(this.phi);
    let dzMin = this.dzArr[0];
    for (let k = 1; k < nz; k++) if (this.dzArr[k] < dzMin) dzMin = this.dzArr[k];
    const dtau = DT_REINIT_FRAC * Math.min(this.dx, Math.min(this.dy, dzMin));
    for (let s = 0; s < REINIT_SUBSTEPS; s++) {
      reinitGodunovGrad(this.phi, this._phi0, this.dx, this.dy, this.dzArr,
                        this._grad, { nx, ny, nz });
      for (let c = 0; c < this.n; c++) {
        const sg = this._phi0[c] > 0.0 ? 1.0 : (this._phi0[c] < 0.0 ? -1.0 : 0.0);
        this.phi[c] -= dtau * sg * (this._grad[c] - 1.0);
      }
    }
  }

  /** Reinit only every LEVELSET_REINIT_INTERVAL steps — it is not cheap. */
  maybeReinitialize() {
    if (this._step % LEVELSET_REINIT_INTERVAL === 0 && this._step > 0) {
      this.reinitialize();
    }
  }

  /** Cells already burned: phi <= 0. */
  burnedMask(out = null) {
    const m = out ?? new Uint8Array(this.n);
    for (let c = 0; c < this.n; c++) m[c] = this.phi[c] <= 0.0 ? 1 : 0;
    return m;
  }

  /** Active burning zone: -L_burnout <= phi <= 0. */
  flameBodyMask(out = null) {
    const m = out ?? new Uint8Array(this.n);
    for (let c = 0; c < this.n; c++) {
      m[c] = (this.phi[c] >= -this.LBurnout && this.phi[c] <= 0.0) ? 1 : 0;
    }
    return m;
  }

  /** Preheating band ahead of the front: 0 < phi <= band_m. */
  aheadBandMask(bandM = DX_VN_BAND_M, out = null) {
    const m = out ?? new Uint8Array(this.n);
    for (let c = 0; c < this.n; c++) {
      m[c] = (this.phi[c] > 0.0 && this.phi[c] <= bandM) ? 1 : 0;
    }
    return m;
  }

  /**
   * Front x-coordinate at row (k, j), linearly interpolated across the first
   * sign change. Infinity when the row holds no front.
   */
  frontX(k = 0, j = 0) {
    const { nx, ny, dx } = this;
    const row = (k * ny + j) * nx;
    for (let i = 0; i < nx - 1; i++) {
      const p0 = this.phi[row + i];
      const p1 = this.phi[row + i + 1];
      if ((p0 < 0.0 && p1 > 0.0) || (p1 < 0.0 && p0 > 0.0)) {
        const frac = -p0 / (p1 - p0);
        return (i + frac + 0.5) * dx;
      }
    }
    return Infinity;
  }
}

// ── Forward flux and the normal velocity it drives ────────────────────

/**
 * Per-cell forward heat flux into the ahead-of-front band [W/m^2].
 *
 * Kept z-resolved rather than column-summed, which is what lets v_n vary with
 * height: top-of-bed cells see forward IR first and advance fastest, bottom-
 * of-bed cells lag. That z-variation is physics, not the numerical drift the
 * old enforce_z_uniformity() was written to suppress (Phase 17b).
 *
 * The Finney burst flux is 2D — flame-finger contact at the bed top — so it
 * is added to the topmost in-band cell of each column only.
 */
export function computeQInAtFront3d(qFrankman, qDomFwd, aheadBandMask,
                                    qBurstConv2d = null, { nx, ny, nz } = {}) {
  const n = qFrankman.length;
  const out = new Float64Array(n);
  for (let c = 0; c < n; c++) {
    out[c] = aheadBandMask[c] ? qFrankman[c] + qDomFwd[c] : 0.0;
  }
  if (qBurstConv2d !== null) {
    const nxy = ny * nx;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        // Topmost in-band cell in this column, matching numpy's argmax on the
        // z-reversed mask (which finds the LAST True from the bottom).
        let kTop = -1;
        for (let k = nz - 1; k >= 0; k--) {
          if (aheadBandMask[k * nxy + j * nx + i]) { kTop = k; break; }
        }
        if (kTop >= 0) out[kTop * nxy + j * nx + i] += qBurstConv2d[j * nx + i];
      }
    }
  }
  return out;
}

/**
 * Per-cell front-normal velocity [m/s]:  v_n = q_in / E_ign.
 *
 *   E_ign = rho_b*cp_s*h_bed*(T_ign - T_amb)          sensible
 *         + rho_b*M_local*h_bed*L_vap*f_dry_to_ignite  latent
 *
 * The latent term is not a correction — at field-density grass with M >= 0.1
 * it dominates, and at M = 0.30 it is roughly 5x the sensible term. That is
 * the Cheney moisture penalty entering the front speed directly.
 * (Drysdale 2011 §3.5; Mell 2007 WFDS §3.4; Linn 2002 FIRETEC.)
 *
 * With M_local null this reduces exactly to the dry formula.
 */
export function computeVn3d(qIn3d, rhoB, cpS, hBed, TIgn, TAmb,
                            MLocal = null, LVap = L_VAP_WATER,
                            fDryToIgnite = 1.0, out = null) {
  const n = qIn3d.length;
  const vn = out ?? new Float64Array(n);
  const ESens = rhoB * cpS * hBed * (TIgn - TAmb);   // [J/m^2]
  for (let c = 0; c < n; c++) {
    let E = MLocal === null
      ? ESens
      : ESens + rhoB * hBed * LVap * fDryToIgnite * MLocal[c];
    if (E < 1.0) E = 1.0;      // divide guard; E_sens > 0 for any real input
    const v = qIn3d[c] / E;
    vn[c] = v > 0.0 ? v : 0.0;
  }
  return vn;
}

/**
 * DOM forward-pointing radiative flux into the ahead-of-front band [W/m^2].
 *
 * sum over ordinates with xi > 0 of  w_n * |xi_n| * I_n  — for weights summing
 * to 4*pi that sum IS the +x hemispheric flux, already in W/m^2.
 *
 * It is NOT multiplied by dz. A spurious dz multiply here (fixed upstream at
 * Phase 15M) turned the flux into a line integrand in W/m, which the
 * downstream column-sum then relabelled W/m^2 — giving v_n about 10x too
 * small and stalling the front. Worth stating because the shapes still work
 * out either way and nothing crashes.
 */
export function computeQDomFwdAtBand(radSolver, aheadBandMask, qDomFwdOut) {
  qDomFwdOut.fill(0.0);
  for (let nOrd = 0; nOrd < radSolver.M; nOrd++) {
    const xi = radSolver.Omega[nOrd][0];
    if (xi <= 0.0) continue;              // forward-pointing ordinates only
    const w = radSolver.weights[nOrd];
    const I = radSolver.Iset[nOrd];
    const wxi = w * Math.abs(xi);
    for (let c = 0; c < qDomFwdOut.length; c++) {
      if (aheadBandMask[c]) qDomFwdOut[c] += wxi * I[c];
    }
  }
}

// ── State-derived flame geometry ──────────────────────────────────────

/**
 * Exact anisotropic Euclidean distance transform, squared, in place.
 *
 * Felzenszwalb & Huttenlocher (2012), Theory of Computing 8:415: the squared
 * EDT is separable, and each 1D pass is the lower envelope of parabolas
 * f(q) + (scale*(p-q))^2. Computing that envelope by the standard hull scan is
 * O(N) per row and EXACT — the same guarantee scipy.ndimage gives, so the two
 * agree to floating point rather than approximately.
 *
 * `f` starts as 0 at seed cells and +inf elsewhere; it ends as squared
 * distance to the nearest seed.
 */
function edtSquaredInPlace(f, dims, scales) {
  const [nz, ny, nx] = dims;
  const nxy = ny * nx;
  const maxLen = Math.max(nz, ny, nx);
  const d = new Float64Array(maxLen);
  const vArr = new Int32Array(maxLen);
  const zArr = new Float64Array(maxLen + 1);
  const out = new Float64Array(maxLen);

  const pass = (len, scale, get, set) => {
    const s2 = scale * scale;
    for (let q = 0; q < len; q++) d[q] = get(q);
    let kk = 0;
    vArr[0] = 0;
    zArr[0] = -Infinity;
    zArr[1] = Infinity;
    for (let q = 1; q < len; q++) {
      let s;
      for (;;) {
        const v = vArr[kk];
        s = ((d[q] + s2 * q * q) - (d[v] + s2 * v * v)) / (2.0 * s2 * (q - v));
        if (s > zArr[kk]) break;
        kk--;
        if (kk < 0) { kk = 0; break; }
      }
      kk++;
      vArr[kk] = q;
      zArr[kk] = s;
      zArr[kk + 1] = Infinity;
    }
    kk = 0;
    for (let q = 0; q < len; q++) {
      while (zArr[kk + 1] < q) kk++;
      const v = vArr[kk];
      out[q] = s2 * (q - v) * (q - v) + d[v];
    }
    for (let q = 0; q < len; q++) set(q, out[q]);
  };

  // x
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      const base = k * nxy + j * nx;
      pass(nx, scales[2], (q) => f[base + q], (q, val) => { f[base + q] = val; });
    }
  }
  // y
  for (let k = 0; k < nz; k++) {
    for (let i = 0; i < nx; i++) {
      const base = k * nxy + i;
      pass(ny, scales[1], (q) => f[base + q * nx], (q, val) => { f[base + q * nx] = val; });
    }
  }
  // z
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const base = j * nx + i;
      pass(nz, scales[0], (q) => f[base + q * nxy], (q, val) => { f[base + q * nxy] = val; });
    }
  }
}

/** Distance from every cell to the nearest cell where `seed` is 0. */
function distanceTransform(seed, dims, scales) {
  const n = seed.length;
  const f = new Float64Array(n);
  for (let c = 0; c < n; c++) f[c] = seed[c] ? Infinity : 0.0;
  edtSquaredInPlace(f, dims, scales);
  for (let c = 0; c < n; c++) f[c] = Math.sqrt(f[c]);
  return f;
}

/**
 * Signed distance to the active-flame region, negative inside [m].
 *
 * A cell is actively flaming when it is reacting, or when it is hot AND still
 * carrying fuel:
 *     active = omega > OMEGA_MIN_FLAME  OR  (T_g > T_PLUME_MIN AND Y_F > Y_F_MIN_PLUME)
 * The second clause is what captures the plume tail, where reaction has
 * finished but the gas is still hot and fuel-bearing.
 *
 * Sampling uses a single representative dz (the mean), matching scipy's
 * per-axis uniform sampling — the reference cannot feed a per-cell dz into
 * the EDT either, and on these grids dz varies modestly.
 */
export function computePhiFlameFromState(omega, Tg, Yfuel, dx, dy, dzArr,
                                         { nx, ny, nz }) {
  const n = omega.length;
  const active = new Uint8Array(n);
  let nActive = 0;
  for (let c = 0; c < n; c++) {
    const a = (omega[c] > OMEGA_MIN_FLAME)
           || (Tg[c] > T_PLUME_MIN && Yfuel[c] > Y_F_MIN_PLUME);
    active[c] = a ? 1 : 0;
    if (a) nActive++;
  }

  const phi = new Float64Array(n);
  if (nActive === 0) { phi.fill(1.0e6); return phi; }
  if (nActive === n) { phi.fill(-1.0e6); return phi; }

  let dzSum = 0.0;
  for (let k = 0; k < nz; k++) dzSum += dzArr[k];
  const dzEff = dzSum / nz;
  const dims = [nz, ny, nx];
  const scales = [dzEff, dy, dx];

  const inv = new Uint8Array(n);
  for (let c = 0; c < n; c++) inv[c] = active[c] ? 0 : 1;
  const distOutside = distanceTransform(inv, dims, scales);   // seeds = active
  const distInside = distanceTransform(active, dims, scales); // seeds = inactive

  for (let c = 0; c < n; c++) {
    phi[c] = active[c] ? -distInside[c] : distOutside[c];
  }
  return phi;
}

/** Cells inside the flame (band_m = 0) or within band_m outside it. */
export function flameBodyMaskFromPhiFlame(phiFlame, bandM = 0.0, out = null) {
  const m = out ?? new Uint8Array(phiFlame.length);
  for (let c = 0; c < phiFlame.length; c++) m[c] = phiFlame[c] <= bandM ? 1 : 0;
  return m;
}

/**
 * Per-cell time since ignition, for the bootstrap window.
 *
 * Newly ignited (in the flame body, age was inf) resets to 0; continuing cells
 * age by dt; cells outside go back to inf so they can re-ignite later.
 */
export function updateCellAge(cellAge, flameBodyMask, dt) {
  for (let c = 0; c < cellAge.length; c++) {
    if (flameBodyMask[c]) {
      cellAge[c] = Number.isFinite(cellAge[c]) ? cellAge[c] + dt : 0.0;
    } else {
      cellAge[c] = Infinity;
    }
  }
}

/**
 * Bootstrap heat in newly-burning cells [W/m^3], added to Q_comb.
 *
 * Only the legacy `ebu_bootstrap` closure uses this. EDC and PaSR self-ignite
 * and the loop gates it off for them.
 */
export function applyBootstrapHeat(Qcomb, flameBodyMask, cellAge,
                                   Qbootstrap = Q_BOOTSTRAP_W_M3,
                                   tBootstrap = T_BOOTSTRAP_S) {
  for (let c = 0; c < Qcomb.length; c++) {
    if (flameBodyMask[c] && cellAge[c] < tBootstrap) Qcomb[c] += Qbootstrap;
  }
}
