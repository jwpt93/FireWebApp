/**
 * Frame extraction for the live-solver page.
 *
 * Kept separate from the worker so it can be unit-tested in node — a worker
 * needs a browser, this does not.
 *
 * The solver's domain is 8 m tall and the fire lives in the bottom ~1 m, so
 * shipping the whole field to the main thread every frame would be mostly
 * empty sky. `visibleRows` picks the k-range worth drawing once, at startup.
 */

/** Index of the first cell whose bottom face is at or above `zTop`. */
export function visibleRows(zFace, nz, zTop) {
  let k = 0;
  while (k < nz && zFace[k] < zTop) k++;
  return Math.max(1, Math.min(k, nz));
}

/**
 * Pack a (kVis, nx) window of T_g and the bed's T_s into one transferable
 * Float32Array.
 *
 * Float32, not Float64: this is display data. Halving the bytes matters more
 * than the last 29 bits of mantissa on a value that becomes a pixel colour.
 *
 * Layout: [ T_g row-major (kVis*nx) | T_s of the bed rows (nZBed*nx) ]
 */
export function packFrame(Tg, Ts, { nx, ny, nz, kVis, nZBed, jSlice = 0 }) {
  const nxy = ny * nx;
  const out = new Float32Array(kVis * nx + nZBed * nx);
  let o = 0;
  for (let k = 0; k < kVis; k++) {
    const row = k * nxy + jSlice * nx;
    for (let i = 0; i < nx; i++) out[o++] = Tg[row + i];
  }
  for (let k = 0; k < nZBed; k++) {
    const row = k * nxy + jSlice * nx;
    for (let i = 0; i < nx; i++) out[o++] = Ts[row + i];
  }
  return out;
}

/**
 * Fire palette: 300 K to `tMax`, black-body-ish through red and orange to a
 * pale yellow. Returns [r, g, b] in 0-255.
 *
 * Deliberately NOT a rainbow. A rainbow implies ordering the eye does not
 * actually read monotonically, and for a temperature field the whole point is
 * that hotter looks hotter.
 */
export function fireColour(T, tMin = 300, tMax = 1500) {
  const f = Math.max(0, Math.min(1, (T - tMin) / (tMax - tMin)));
  if (f <= 0.002) return [0, 0, 0];
  // Three-stop ramp: near-black -> deep red -> orange -> pale yellow.
  const stops = [
    [0.00, 12, 6, 22],
    [0.30, 150, 26, 12],
    [0.65, 236, 116, 20],
    [1.00, 252, 232, 176],
  ];
  for (let s = 0; s < stops.length - 1; s++) {
    const [p0, r0, g0, b0] = stops[s];
    const [p1, r1, g1, b1] = stops[s + 1];
    if (f <= p1) {
      const u = (f - p0) / (p1 - p0);
      return [Math.round(r0 + u * (r1 - r0)),
              Math.round(g0 + u * (g1 - g0)),
              Math.round(b0 + u * (b1 - b0))];
    }
  }
  return [252, 232, 176];
}

/** Precomputed 256-entry palette — one lookup per pixel instead of a branch. */
export function buildPalette(tMin = 300, tMax = 1500) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = fireColour(tMin + (i / 255) * (tMax - tMin), tMin, tMax);
    lut[i * 3] = r; lut[i * 3 + 1] = g; lut[i * 3 + 2] = b;
  }
  return lut;
}
