/**
 * Resolved-solver rate of spread, and the Phase 19 / Phase 20 hybrid blend.
 *
 * The applet cannot run the 3D solver — one case is 10 minutes to an hour.
 * It carries the solver's ANSWERS instead, exported by
 * scripts/export_resolved_ros.py from the parent project's Cheney sweeps, and
 * reproduces the blend that project validated:
 *
 *     U_10 <  2.5    pure Cheney regression
 *     2.5 .. 3.5     linear ramp between regression and solver
 *     U_10 >= 3.5    pure resolved solver
 *
 * The threshold is not a free knob. It is the wind below which the resolved
 * closure stops propagating correctly: Phase 19 put it at 1.4 and left a hole
 * — Nat 4% at U_10 = 2 resolved to 6.56 m/min against Cheney's 26.42, ratio
 * 0.248, the one failure in an otherwise 19/20 sweep. Phase 20 "Option B"
 * raised it to 3.5, above that hole. See Cheney 1998 §3.2 and Finney 2015
 * (PNAS 112:9833) for why mean-field closures fail at low wind: spread there
 * is mediated by intermittent flame contact that a RANS average removes.
 *
 * WHAT IS STORED IS A RATIO, not a rate. Resolved ROS spans 6-86 m/min across
 * the sweep; the ratio to Cheney spans 0.58-0.94. Interpolating the near-flat
 * ratio off a handful of samples is well conditioned; interpolating the steep
 * rate is not. It is also the honest description of what a resolved run adds:
 * a measured correction to the regression.
 */
import { rosFromU10, U2_PER_U10 } from './cheney.js';

/**
 * Weight on the EMPIRICAL side of the blend.
 *
 * Exact port of blend_resolved_empirical() in
 * src/model_outdoor/empirical_ros.py — 1.0 below (threshold - width),
 * 0.0 at or above threshold, linear ramp between.
 *
 * @param {number} U10_m_s
 * @param {number} threshold_m_s
 * @param {number} width_m_s
 * @returns {number} in [0, 1]
 */
export function empiricalWeight(U10_m_s, threshold_m_s, width_m_s) {
  if (U10_m_s >= threshold_m_s) return 0;
  if (width_m_s <= 0) return 1;
  const uLo = threshold_m_s - width_m_s;
  if (U10_m_s <= uLo) return 1;
  return (threshold_m_s - U10_m_s) / width_m_s;
}

/** Linear interpolation over a sorted [x, y] list, clamped outside its range. */
function lerpTable(points, x) {
  if (!points || !points.length) return NaN;
  if (x <= points[0][0]) return points[0][1];
  const last = points[points.length - 1];
  if (x >= last[0]) return last[1];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    if (x <= x1) return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
  }
  return last[1];
}

/** Moisture levels the solver was actually run at. */
const M_LO = 4;
const M_HI = 8;

export class ResolvedROS {
  /** @param {object} payload parsed docs/data/resolved.json */
  constructor(payload) {
    this.table = payload.table;
    this.threshold = payload._meta.blend.u_threshold_U10_m_s;
    this.width = payload._meta.blend.blend_width_m_s;
  }

  /** Key into the table, e.g. ('natural', 4) -> 'Nat4'. */
  static key(fuelKey, mfPct) {
    return `${fuelKey === 'cut' ? 'Cut' : 'Nat'}${mfPct}`;
  }

  /**
   * Resolved/Cheney ratio at (fuel, moisture, wind).
   *
   * Bilinear-ish: interpolate in wind within each sampled moisture, then
   * between the two moistures. The solver was run at 4% and 8% ONLY, so
   * outside that span this clamps rather than extrapolating a two-point
   * trend — an extrapolated ratio would be a guess wearing the solver's
   * authority.
   */
  ratio(fuelKey, moistureFrac, U10_m_s) {
    const lo = this.table[ResolvedROS.key(fuelKey, M_LO)];
    const hi = this.table[ResolvedROS.key(fuelKey, M_HI)];
    if (!lo || !hi) return NaN;
    const rLo = lerpTable(lo, U10_m_s);
    const rHi = lerpTable(hi, U10_m_s);
    const mfPct = Math.min(M_HI, Math.max(M_LO, moistureFrac * 100));
    const f = (mfPct - M_LO) / (M_HI - M_LO);
    return rLo + (rHi - rLo) * f;
  }

  /** Resolved-solver ROS [m/s]: the stored ratio applied to Cheney. */
  ros(fuelKey, moistureFrac, U10_m_s, aCh) {
    const r = this.ratio(fuelKey, moistureFrac, U10_m_s);
    if (!Number.isFinite(r)) return NaN;
    return r * rosFromU10(U10_m_s, moistureFrac, aCh);
  }

  /**
   * The hybrid head ROS [m/s] — what the applet actually propagates.
   *
   * Because the resolved side is stored as a ratio, the blend collapses to a
   * single multiplier on the Cheney rate:
   *
   *     ROS = Cheney x ( w + (1 - w) * ratio )
   *
   * which is continuous across the blend window by construction.
   */
  hybridRos(fuelKey, moistureFrac, U10_m_s, aCh) {
    const cheney = rosFromU10(U10_m_s, moistureFrac, aCh);
    const w = empiricalWeight(U10_m_s, this.threshold, this.width);
    if (w >= 1) return cheney;
    const r = this.ratio(fuelKey, moistureFrac, U10_m_s);
    if (!Number.isFinite(r)) return cheney;
    return cheney * (w + (1 - w) * r);
  }

  /**
   * Which side of the blend a given wind sits on, for the UI to state
   * plainly rather than leaving the viewer to guess.
   *
   * @returns {{regime: 'fit'|'blend'|'resolved', w_emp: number,
   *            U10_m_s: number, U2_m_s: number}}
   */
  regime(U2_m_s) {
    const U10 = U2_m_s / U2_PER_U10;
    const w = empiricalWeight(U10, this.threshold, this.width);
    return {
      regime: w >= 1 ? 'fit' : w <= 0 ? 'resolved' : 'blend',
      w_emp: w,
      U10_m_s: U10,
      U2_m_s,
    };
  }

  /** Threshold expressed in the applet's slider units (2 m wind). */
  get thresholdU2() {
    return this.threshold * U2_PER_U10;
  }

  /** Start of the blend window in slider units. */
  get blendStartU2() {
    return (this.threshold - this.width) * U2_PER_U10;
  }
}
