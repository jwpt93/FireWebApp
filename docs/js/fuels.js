/**
 * Fuel-bed properties for the Cheney 1993 grass types.
 *
 * Every value carries its source, per the parent project's convention that a
 * non-trivial parameter without provenance does not belong in a deck.
 *
 * Bulk density and depth are the Cheney 1993 Table 3 field measurements as
 * used by the parent project's validation decks; SAV is Tyrer (1986) IJWF as
 * cited in unitiedmodel2's boundary.py canopy-roughness notes.
 */

/**
 * Surface-area-to-volume ratio [1/m].
 *
 * Cheney 1993 measured this directly: "The surface area-to-volume ratio (σ)
 * for undisturbed pasture was: Eriachne 97.7 cm⁻¹; Themeda 122.4 cm⁻¹" —
 * i.e. 9770 and 12240 m⁻¹. Eriachne is used here as the more conservative
 * (coarser) of the two.
 *
 * The SAME value applies to both treatments, because the paper says so: "We
 * did not attempt to measure σ for the treatments where the fuel was
 * harvested and removed ... we assigned these treatments the σ value for the
 * respective species." Cutting changed height and bulk density, not the
 * fineness of the grass itself.
 *
 * NOT the σ = 2000 / 3500 that appears in the parent project's
 * boundary.py canopy-roughness notes — those are drag-related effective
 * values for the frontal-area index, and using them here inflated residence
 * time about fivefold (38 s instead of 8 s), which showed up as an
 * implausible 39 m deep flaming zone in the side view.
 */
const SAV_1_M = 9770;

export const FUELS = Object.freeze({
  natural: {
    label: 'Natural pasture',
    a_ch: 0.406,                 // Cheney 1993 Fig 8 caption
    bulk_density_kg_m3: 1.07,    // Cheney 1993 Table 3 (undisturbed sward)
    depth_m: 0.37,               // Cheney 1993 Table 3
    sav_1_m: SAV_1_M,            // Cheney 1993, Eriachne 97.7 cm^-1
  },
  cut: {
    label: 'Cut grass',
    a_ch: 0.343,                 // Cheney 1993 Fig 8 caption
    bulk_density_kg_m3: 2.95,    // Cheney 1993 Table 3 (cut and returned)
    depth_m: 0.15,               // Cheney 1993 Table 3
    sav_1_m: SAV_1_M,            // same species; see SAV_1_M note
  },
});

/**
 * Oven-dry fuel load available to the flaming front [kg/m^2].
 *
 *     w_0 = rho_b * delta
 *
 * Both Cheney types land near 0.4 kg/m^2, which is where grass fuel loads
 * normally sit -- a useful sanity anchor if these numbers are ever edited.
 */
export function fuelLoad(fuel) {
  return fuel.bulk_density_kg_m3 * fuel.depth_m;
}

/**
 * Flame residence time [s] -- how long a cell stays in the flaming front
 * once the fire arrives.
 *
 *     t_r [min] = 384 / sigma [1/ft]
 *
 * Anderson (1969) USDA FS INT-69, as used by Rothermel (1972) INT-115.
 * Fine grass (high SAV) burns out in tens of seconds; this is what sets the
 * width of the bright band trailing the front in the applet, and it is the
 * only reason the burnt area behind the fire ever stops glowing.
 */
export function residenceTime_s(fuel) {
  const sigma_ft = fuel.sav_1_m / 3.28084;
  return (384 / sigma_ft) * 60;
}
