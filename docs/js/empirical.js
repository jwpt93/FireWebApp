/**
 * empirical.js — Tier-1 empirical grass-fire rate-of-spread models.
 *
 * Hand port of src/model_outdoor/empirical_ros.py (Cheney 1998 Eq. 6 and
 * Marsden-Smedley & Catchpole 1995).  Keep the two implementations in sync;
 * scripts/web_export/check_js_port.mjs verifies numerical agreement against
 * the Python reference table.
 *
 * References:
 *   Cheney, N.P., Gould, J.S., Catchpole, W.R. (1998) "Prediction of fire
 *     spread in grasslands," Int. J. Wildland Fire 8:1-13.  Eq. 6 power-law.
 *   Marsden-Smedley, J.B. & Catchpole, W.R. (1995) "Fire behaviour modelling
 *     in Tasmanian buttongrass moorlands," Int. J. Wildland Fire 5(4):215.
 */

// Cheney 1998 Eq. 6 default exponents (calibrated against Annaburroo grass).
export const CHENEY_EQ6_U_EXP = 0.987;
export const CHENEY_EQ6_B_MF = 0.0707;
export const CHENEY_EQ6_U2_RATIO = 0.723; // U_2 = 0.723 · U_10 (Cheney 1993 convention)

// Fuel-dependent coefficient a_ch (Cheney 1998).
export const A_CH_NATURAL = 0.406; // natural / undisturbed grass
export const A_CH_CUT = 0.343;     // cut / mown grass

// Marsden-Smedley & Catchpole 1995 buttongrass moorland fit.
const MS_1995_CONST = 0.678 / 60.0; // m/s per (km/h)^1.312 unit-of-U-M-age
const MS_1995_U_EXP = 1.312;
const MS_1995_B_MF = 0.0243;
const MS_1995_AGE_LAMBDA = 0.116;   // age-asymptote build-up rate (1/yr)

/**
 * Cheney 1998 Eq. 6 grass-fire rate of spread.
 *   U_m_s        mean wind speed at standard reference height [m/s]
 *   moisture_frac fuel moisture, mass fraction (0.04 = 4%)
 *   a_ch         fuel coefficient: 0.406 natural, 0.343 cut
 * Returns ROS [m/s], non-negative.
 */
export function cheneyEq6Ros(U_m_s, moisture_frac, a_ch) {
  if (U_m_s <= 0.0) return 0.0;
  const mf_pct = moisture_frac * 100.0;
  const u2 = Math.max(0.0, CHENEY_EQ6_U2_RATIO * U_m_s);
  const ros_m_per_min =
    a_ch * Math.pow(u2, CHENEY_EQ6_U_EXP) * Math.exp(-CHENEY_EQ6_B_MF * mf_pct) * 60.0;
  return Math.max(0.0, ros_m_per_min / 60.0);
}

/**
 * Marsden-Smedley 1995 buttongrass moorland head-fire ROS regression.
 *   U_1p7_m_s    wind at 1.7 m above ground [m/s]  (U_1.7 ≈ U_10 / 1.44)
 *   moisture_frac dead-fuel moisture, mass fraction
 *   age_yr       stand age since last fire [yr]
 * Returns ROS [m/s], non-negative.
 */
export function marsdenSmedleyRos(U_1p7_m_s, moisture_frac, age_yr) {
  if (U_1p7_m_s <= 0.0 || age_yr <= 0.0) return 0.0;
  const U_kmh = 3.6 * U_1p7_m_s;
  const mf_pct = moisture_frac * 100.0;
  const ros =
    MS_1995_CONST *
    Math.pow(U_kmh, MS_1995_U_EXP) *
    Math.exp(-MS_1995_B_MF * mf_pct) *
    (1.0 - Math.exp(-MS_1995_AGE_LAMBDA * age_yr));
  return Math.max(0.0, ros);
}

/**
 * Marsden-Smedley 2001 (IJWF 10(2):255, Part IV) sustaining logistic.
 * Returns probability that a buttongrass fire sustains propagation.
 *   productivity: 1 = low (quartzite), 2 = medium (dolerite/limestone/till)
 */
export function marsdenSmedleyPSustain(U_1p7_m_s, moisture_frac, productivity = 1) {
  const U_kmh = 3.6 * U_1p7_m_s;
  const mf_pct = moisture_frac * 100.0;
  const a =
    -1.0 +
    0.68 * U_kmh -
    0.07 * mf_pct -
    0.0037 * U_kmh * mf_pct +
    2.1 * productivity;
  return 1.0 / (1.0 + Math.exp(-a));
}
