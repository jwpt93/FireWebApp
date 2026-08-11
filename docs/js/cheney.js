/**
 * Cheney, Gould & Catchpole (1993) grassland rate-of-spread — JavaScript port.
 *
 * Reference implementation: src/model_outdoor/empirical_ros.py
 * Cross-checked against it by docs/test.html using docs/data/golden.json.
 *
 * THE LAW
 * -------
 * Cheney 1993 IJWF 3(1):31-44 gives it directly in the Figure 8 caption:
 *
 *     R = a · U_2^0.987 · exp(-0.0707 · M_f)
 *
 *   R    rate of spread                     [m/s]
 *   U_2  wind speed at 2 m                  [m/s]
 *   M_f  dead fuel moisture content         [PERCENT, not fraction]
 *   a    fuel coefficient: 0.406 natural sward, 0.343 cut grass
 *
 * WIND REFERENCE HEIGHT — read this before touching the formula
 * -------------------------------------------------------------
 * U_2 is the paper's native variable.  Table 2 defines it as "Wind speed at
 * 2 m (m s-1)", and the printed x-axis of Figure 8 is "Wind speed at 2 m
 * (ms-1)".  The digitised Fig 8 data in data/cheney_experimental/ is
 * therefore in U_2, despite its _meta.columns saying "U_10_m_s".
 *
 * The Python reference function `cheney_eq6_ros_m_per_s()` takes U_10 and
 * applies the 0.723 factor internally.  Mixing the two conventions costs
 * 27-38%, so the two entry points below are kept separate and explicit
 * rather than sharing a defaulted argument.  Prefer rosFromU2() — it is
 * the paper's own variable and needs no conversion to sit on the Fig 8 axis.
 *
 * VALIDITY
 * --------
 * The regression was fitted to experimental fires at Annaburroo, N.T.,
 * spanning roughly U_2 = 2-7 m/s and M_f = 2-12%.  Outside that box this
 * is extrapolation, and callers should say so on screen.
 */

/** Exponent on U_2. Cheney 1993 Fig 8 caption. */
export const CHENEY_U_EXP = 0.987;

/** Moisture damping coefficient [1/percent]. Cheney 1993 Fig 8 caption. */
export const CHENEY_B_MF = 0.0707;

/**
 * U_2 = 0.723 · U_10, the 10 m -> 2 m factor used throughout the parent
 * project (model_outdoor/empirical_ros.py: CHENEY_EQ6_U2_RATIO).
 */
export const U2_PER_U10 = 0.723;

/** Fuel coefficient a. Cheney 1993 Fig 8 caption. */
export const A_CH = Object.freeze({
  natural: 0.406,
  cut: 0.343,
});

/** Fitted range of the regression — used to flag extrapolation in the UI. */
export const VALID_RANGE = Object.freeze({
  U2_m_s: [2.0, 7.0],
  moisture_pct: [2.0, 12.0],
});

/**
 * Cheney 1993 rate of spread from the 2 m wind — the paper's native form.
 *
 * @param {number} U2_m_s        wind speed at 2 m [m/s]
 * @param {number} moistureFrac  dead fuel moisture as a FRACTION (0.04 = 4%)
 * @param {number} aCh           fuel coefficient (see A_CH)
 * @returns {number} rate of spread [m/s], never negative
 */
export function rosFromU2(U2_m_s, moistureFrac, aCh) {
  if (!(U2_m_s > 0)) return 0;
  const mfPct = moistureFrac * 100;
  const ros = aCh * Math.pow(U2_m_s, CHENEY_U_EXP) * Math.exp(-CHENEY_B_MF * mfPct);
  return ros > 0 ? ros : 0;
}

/**
 * Cheney 1993 rate of spread from the 10 m wind.
 *
 * Applies the 0.723 factor, so this is the exact analogue of the Python
 * `cheney_eq6_ros_m_per_s(U_m_s, moisture_frac, a_ch)`.
 *
 * @param {number} U10_m_s       wind speed at 10 m [m/s]
 * @param {number} moistureFrac  dead fuel moisture as a FRACTION
 * @param {number} aCh           fuel coefficient (see A_CH)
 * @returns {number} rate of spread [m/s]
 */
export function rosFromU10(U10_m_s, moistureFrac, aCh) {
  if (!(U10_m_s > 0)) return 0;
  return rosFromU2(U2_PER_U10 * U10_m_s, moistureFrac, aCh);
}

/** True when (U_2, moisture) falls outside the fitted range. */
export function isExtrapolating(U2_m_s, moistureFrac) {
  const mfPct = moistureFrac * 100;
  const [uLo, uHi] = VALID_RANGE.U2_m_s;
  const [mLo, mHi] = VALID_RANGE.moisture_pct;
  return U2_m_s < uLo || U2_m_s > uHi || mfPct < mLo || mfPct > mHi;
}

// ---------------------------------------------------------------------------
// Derived fire-behaviour quantities
//
// These are energy-budget and correlation relations layered on top of the
// ROS.  They are what turn a spread rate into something a reader can picture.
// ---------------------------------------------------------------------------

/**
 * Low heat of combustion for dry grass fuel [kJ/kg].
 *
 * Byram (1959) conventional value for wildland fuels.  Exposed as a named
 * constant rather than buried in a formula so it can be swapped per fuel.
 */
export const HEAT_OF_COMBUSTION_KJ_KG = 18600.0;

/**
 * Byram (1959) fireline intensity.
 *
 *     I_B = H · w_0 · R
 *
 * This is an energy budget, not a fit: the rate at which the flaming front
 * releases energy per unit length of fire edge.
 *
 * @param {number} ros_m_s   rate of spread [m/s]
 * @param {number} w0_kg_m2  oven-dry fuel load consumed [kg/m^2]
 * @param {number} [H_kJ_kg] low heat of combustion [kJ/kg]
 * @returns {number} fireline intensity [kW/m]
 */
export function firelineIntensity(ros_m_s, w0_kg_m2, H_kJ_kg = HEAT_OF_COMBUSTION_KJ_KG) {
  const I = H_kJ_kg * w0_kg_m2 * ros_m_s;
  return I > 0 ? I : 0;
}

/**
 * Byram (1959) flame length, metric form (see Alexander 1982, Can. J. For.
 * Res. 12:245 for the SI restatement).
 *
 *     L = 0.0775 · I_B^0.46      I_B in kW/m, L in m
 *
 * This one IS a correlation, not a derivation — flagged as such because the
 * applet distinguishes fitted quantities from conserved ones.
 *
 * @param {number} I_kW_m fireline intensity [kW/m]
 * @returns {number} flame length [m]
 */
export function flameLength(I_kW_m) {
  if (!(I_kW_m > 0)) return 0;
  return 0.0775 * Math.pow(I_kW_m, 0.46);
}

/** Gravitational acceleration [m/s^2]. boundary.py: _G */
const G = 9.81;

/**
 * Wind-adjustment factor, 10 m -> midflame. boundary.py: _WAF
 * 0.90 open, 0.60 shrub.
 */
export const WAF = Object.freeze({ open: 0.90, shrub: 0.60 });

/**
 * Byram (1959) flame tilt angle from vertical [rad].
 *
 *     Fr    = U_mf^2 / (g · L_f)        convective Froude number
 *     tan θ = 0.88 · sqrt(Fr)
 *
 * Port of `flame_tilt_angle()` in src/model_outdoor/boundary.py, which takes
 * the 10 m wind and converts to midflame internally. This is what makes the
 * side view show a flame that lies over as the wind rises: a stronger wind
 * at fixed flame length raises Fr, and the flame's vertical reach falls as
 * cos θ even though its LENGTH is unchanged.
 *
 * @param {number} U10_m_s  wind at 10 m [m/s]
 * @param {number} L_f_m    flame length [m]
 * @param {string} [terrain] 'open' or 'shrub'
 * @returns {number} tilt from vertical [rad]; 0 with no wind or no flame
 */
export function flameTilt(U10_m_s, L_f_m, terrain = 'open') {
  if (!(U10_m_s > 0) || !(L_f_m > 0)) return 0;
  const U_mf = U10_m_s * (WAF[terrain] ?? WAF.open);
  const Fr = (U_mf * U_mf) / (G * L_f_m);
  return Math.atan(0.88 * Math.sqrt(Fr));
}

/**
 * Vertical reach of a tilted flame [m].
 *
 *     H = L_f · cos θ
 *
 * The number people mean by "flame height", as distinct from Byram's flame
 * LENGTH measured along the tilted flame axis.
 */
export function flameHeight(L_f_m, tiltRad) {
  return L_f_m * Math.cos(tiltRad);
}

// ---------------------------------------------------------------------------
// Front normal speed models
//
// Cheney 1993 gives a HEAD-fire rate of spread: one number, for the fastest
// point of the fire, spreading with the wind.  Turning that into a 2D front
// speed v_n(theta) requires an extra assumption that the paper does not
// supply — its own "fire shape" variable is a CATEGORICAL head-fire shape
// (pointed vs parabolic, Table 2 "HFS"), used as a regression covariate.
// There is no length-to-breadth or elliptical relation in the 1993 paper.
//
// So the shape model is an explicit, swappable choice, and the UI must say
// which one is active.  Both below reduce to exactly the Cheney head ROS
// when the front normal points downwind.
// ---------------------------------------------------------------------------

/**
 * Isotropic: every point of the front advances at the head ROS.
 *
 * Physically wrong in wind (it makes flanks and the backing edge run as fast
 * as the head, so the fire stays circular) but it is the assumption-free
 * baseline, and useful as a reference for what the shape model is doing.
 */
export function isotropicSpeed(headRos_m_s) {
  return () => headRos_m_s;
}

/**
 * Wind-projected: evaluate the Cheney law at the component of wind along
 * the front normal.
 *
 *     v_n(n) = max( R( U_2 · max(n · w, 0) ),  R_back )
 *
 * where w is the unit wind vector.  Uses only the fitted law and no new
 * constants, at the cost of extrapolating that law to off-axis directions
 * it was never fitted against.  The small isotropic R_back keeps the flanks
 * and rear moving, which is what stops the front pinching to a cusp.
 *
 * R_back is a FLOOR, not an offset added on top.  Adding it would make the
 * head run at (1 + backingFrac) × the Cheney rate, breaking the one thing
 * this mode guarantees: at the head, where n is aligned with the wind,
 * v_n is exactly the published rate of spread.
 *
 * @param {number} U2_m_s        2 m wind speed [m/s]
 * @param {number} moistureFrac  moisture as a fraction
 * @param {number} aCh           fuel coefficient
 * @param {number} windDirRad    direction the wind blows TOWARDS [rad]
 * @param {number} [backingFrac] backing rate as a fraction of head ROS
 * @returns {(nx: number, ny: number) => number}
 */
export function windProjectedSpeed(U2_m_s, moistureFrac, aCh, windDirRad, backingFrac = 0.05) {
  const wx = Math.cos(windDirRad);
  const wy = Math.sin(windDirRad);
  const head = rosFromU2(U2_m_s, moistureFrac, aCh);
  const back = backingFrac * head;
  return (nx, ny) => {
    const proj = nx * wx + ny * wy;
    if (!(proj > 0)) return back;
    const v = rosFromU2(U2_m_s * proj, moistureFrac, aCh);
    return v > back ? v : back;
  };
}

/**
 * Weight applied to the EMPIRICAL rate in the blended front speed.
 *
 * 1.0 below (threshold - width): empirical only.
 * 0.0 at or above threshold:     resolved solver only.
 * Linear ramp between.  width = 0 gives a hard step.
 *
 * Port of model_outdoor/empirical_ros.py :: blend_resolved_empirical.
 *
 * The threshold is not a free knob. It is the wind below which the resolved
 * closure stops propagating: Phase 19 set it at 1.4 and left a hole -- Nat 4%
 * at U_10 = 2 resolved to 6.56 m/min against Cheney's 26.42, ratio 0.248, the
 * single failure in an otherwise 19/20 sweep. Phase 20 "Option B" raised it to
 * 3.5, clear of that hole. Upstream decks still carry the older 1.4/0.5.
 *
 * Why a mean-field closure fails at low wind at all: spread there is carried by
 * intermittent flame contact, which a RANS average removes by construction
 * (Finney 2015, PNAS 112:9833; Cheney 1998 section 3.2).
 */
export function blendResolvedEmpirical(U_m_s, uThreshold, blendWidth) {
  if (U_m_s >= uThreshold) return 0.0;
  if (blendWidth <= 0.0) return 1.0;
  const uLo = uThreshold - blendWidth;
  if (U_m_s <= uLo) return 1.0;
  return (uThreshold - U_m_s) / blendWidth;
}
