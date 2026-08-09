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

export const FUELS = Object.freeze({
  natural: {
    label: 'Natural pasture',
    a_ch: 0.406,                 // Cheney 1993 Fig 8 caption
    bulk_density_kg_m3: 1.07,    // Cheney 1993 Table 3 (undisturbed sward)
    depth_m: 0.37,               // Cheney 1993 Table 3
    sav_1_m: 2000,               // Tyrer 1986 IJWF, Themeda/Eriachne analog
  },
  cut: {
    label: 'Cut grass',
    a_ch: 0.343,                 // Cheney 1993 Fig 8 caption
    bulk_density_kg_m3: 2.95,    // Cheney 1993 Table 3 (cut and returned)
    depth_m: 0.15,               // Cheney 1993 Table 3
    sav_1_m: 3500,               // Tyrer 1986 IJWF
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
