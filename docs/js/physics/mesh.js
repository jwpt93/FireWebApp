/**
 * Vertical mesh construction — JS port of model_outdoor/mesh.py and the
 * new-kernel branch of Grid3D.build in model_outdoor/spread_3d.py.
 *
 * The z-axis is composed from named segments rather than one formula, the way
 * a mesh generator does it:
 *
 *     [wall BL] [bulk bed] [inner-solid BL] | [outer-air BL] [bulk atmosphere]
 *                                      z = h_bed
 *
 * Everything except the bulk bed is optional. n_z_bed is the TOTAL bed cell
 * count across the first three, so the bulk count is what is left over.
 *
 * WHY THIS IS NOT A ONE-LINER. The near-wall spacing is not a free parameter.
 * Refining `wall_bl_first_dz` below about 5 mm at U = 4 m/s pushes the first
 * cell out of the log-law layer and into the buffer/viscous sublayer, where
 * the k-e wall function does not apply, and ROS comes out ~30% high. The
 * production decks set `first_dz` proportional to 1/U precisely so every wind
 * speed runs at a comparable y+. Getting the segment stack right is what makes
 * the wall function legitimate.
 *
 * BOTH PATHS ARE PORTED, and which one runs is decided exactly as upstream
 * decides it: the segment stack is used only when at least one BL cell count
 * is greater than zero, otherwise the legacy uniform-bed branch runs.
 *
 * That dispatch has a consequence worth knowing before reading further. Every
 * production deck sets `wall_bl_N = 0`, so they ALL take the legacy branch —
 * and the legacy branch ignores `atm_growth` and `atm_max_dz` entirely, using
 * `dz_expansion` (default 1.0) instead. The Cheney deck asks for an
 * atmosphere growing to a 1 m cap and gets 320 uniform 25 mm cells to z = 8 m.
 * See SOLVER_PORT.md §7.6. The port reproduces this rather than fixing it.
 *
 * A NOTE ON CELL-TO-CELL STRETCH. Project rules cap the ratio at 1.2 (target
 * 1.1) and the aspect ratio at 100:1. The defaults here use growth = 1.3,
 * which VIOLATES that cap — it is inherited from the current production decks
 * and is a known outstanding item upstream, not something this port
 * introduced or should quietly correct.
 */

// ── Segments ──────────────────────────────────────────────────────────

/** N cells of uniform size L/N. */
export function uniformSegment(L, N) {
  if (N <= 0 || L <= 0.0) return new Float64Array(0);
  return new Float64Array(N).fill(L / N);
}

/**
 * N geometrically growing cells: dz_k = first_dz * growth^k.
 *
 * `reverse` puts the THIN cell at the far end instead of the near end — used
 * at internal interfaces where the fine cells belong against the next segment
 * rather than the previous one.
 */
export function inflationSegment(N, firstDz, growth = 1.2, reverse = false) {
  if (N <= 0 || firstDz <= 0.0) return new Float64Array(0);
  const dzs = new Float64Array(N);
  if (Math.abs(growth - 1.0) < 1e-12) {
    dzs.fill(firstDz);
  } else {
    for (let k = 0; k < N; k++) dzs[k] = firstDz * Math.pow(growth, k);
  }
  if (reverse) dzs.reverse();
  return dzs;
}

/**
 * Fill length L with cells growing from interface_dz by `growth`, capped at
 * max_dz.
 *
 * TERMINATION IS DELIBERATE: the sequence stops when one more cell would
 * overshoot L by more than half a cell, so the segment ends slightly SHORT of
 * L rather than ending in a remainder cell whose ratio to its predecessor
 * breaks the growth rule. Lz drifts a few percent below what was requested.
 * Stretch-rule compliance beats exact length matching.
 */
export function bulkSegment(L, interfaceDz, maxDz, growth = 1.3) {
  if (L <= 0.0) return new Float64Array(0);
  const cells = [];
  let cum = 0.0;
  let nextDz = Math.min(interfaceDz * growth, maxDz);
  while (cum < L) {
    const dz = Math.min(nextDz, maxDz);
    if (cum + dz > L + 0.5 * dz) break;
    cells.push(dz);
    cum += dz;
    nextDz = dz * growth;
  }
  return Float64Array.from(cells);
}

function concatSegments(arrays) {
  const kept = arrays.filter((a) => a.length > 0);
  let n = 0;
  for (const a of kept) n += a.length;
  const out = new Float64Array(n);
  let o = 0;
  for (const a of kept) { out.set(a, o); o += a.length; }
  return out;
}

const sum = (a) => { let s = 0.0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };

/**
 * Assemble the full z-axis.
 *
 * @returns {{dzArr: Float64Array, nZBed: number}}
 */
export function buildZAxisBedAtm({
  hBed, Lz, nZBed,
  wallBlN = 0, wallBlFirstDz = 0.0, wallBlGrowth = 1.3,
  bedTopInnerBlN = 0, bedTopInnerBlFirstDz = 0.0, bedTopInnerBlGrowth = 1.3,
  bedTopOuterBlN = 0, bedTopOuterBlFirstDz = 0.0, bedTopOuterBlGrowth = 1.3,
  atmMaxDz = null, atmGrowth = 1.3, atmUniformDz = null,
}) {
  if (hBed < 0 || Lz <= hBed || nZBed <= 0) {
    throw new Error(
      `h_bed=${hBed}, Lz=${Lz}, n_z_bed=${nZBed} must satisfy ` +
      `h_bed >= 0, Lz > h_bed, n_z_bed > 0`);
  }

  const bedSegs = [];
  let bedUsed = 0.0;
  if (wallBlN > 0 && wallBlFirstDz > 0.0) {
    const seg = inflationSegment(wallBlN, wallBlFirstDz, wallBlGrowth, false);
    bedSegs.push(seg);
    bedUsed = sum(seg);
  }

  let topInner = null;
  let topInnerThickness = 0.0;
  if (bedTopInnerBlN > 0 && bedTopInnerBlFirstDz > 0.0) {
    topInner = inflationSegment(bedTopInnerBlN, bedTopInnerBlFirstDz,
                                bedTopInnerBlGrowth, true);
    topInnerThickness = sum(topInner);
  }

  const nBulkBed = nZBed - wallBlN - bedTopInnerBlN;
  if (nBulkBed < 0) {
    throw new Error(
      `n_z_bed=${nZBed} smaller than wall_bl_N + bed_top_inner_bl_N ` +
      `(${wallBlN + bedTopInnerBlN}); not enough cells for bulk`);
  }
  const LBulkBed = hBed - bedUsed - topInnerThickness;
  if (LBulkBed < 0) {
    throw new Error(
      `Wall BL + top-inner BL thickness (${bedUsed + topInnerThickness}) ` +
      `exceeds h_bed (${hBed})`);
  }
  if (nBulkBed > 0) {
    bedSegs.push(uniformSegment(LBulkBed, nBulkBed));
  } else if (LBulkBed > 1e-3 * hBed) {
    throw new Error(
      `n_bulk_bed=0 but L_bulk_bed=${LBulkBed.toFixed(4)} > 0 — increase ` +
      `n_z_bed or reduce BL cell counts`);
  }
  if (topInner !== null) bedSegs.push(topInner);

  let bedDz = concatSegments(bedSegs);

  // Geometric sums leave a millimetre-scale mismatch at growth^N. Rescale the
  // whole bed proportionally so it lands on h_bed exactly — a residual here
  // would shift where the bed top sits relative to the air-side BL.
  const bedTotal = sum(bedDz);
  const residual = hBed - bedTotal;
  if (Math.abs(residual) > 1e-6 * hBed) {
    if (Math.abs(residual) > 0.05 * hBed) {
      throw new Error(
        `Bed segments sum to ${bedTotal} but h_bed=${hBed} — residual >5% ` +
        `of h_bed indicates a real geometry error`);
    }
    const scale = hBed / bedTotal;
    bedDz = bedDz.map((v) => v * scale);
  }

  const airSegs = [];
  let interfaceDz = bedDz[bedDz.length - 1];
  if (bedTopOuterBlN > 0 && bedTopOuterBlFirstDz > 0.0) {
    const outer = inflationSegment(bedTopOuterBlN, bedTopOuterBlFirstDz,
                                   bedTopOuterBlGrowth, false);
    airSegs.push(outer);
    interfaceDz = outer[outer.length - 1];
  }

  let airThickness = 0.0;
  for (const s of airSegs) airThickness += sum(s);
  const LAtm = Lz - hBed - airThickness;
  if (atmUniformDz !== null && atmUniformDz > 0 && LAtm > 0) {
    // Forced-uniform atmosphere: keeps the atm mesh identical across
    // BL-placement experiments, so a change in ROS is attributable to the BL
    // rather than to the atmosphere silently re-meshing with it.
    const nUniform = Math.max(1, Math.round(LAtm / atmUniformDz));
    airSegs.push(uniformSegment(nUniform * atmUniformDz, nUniform));
  } else if (LAtm > 0) {
    const cap = atmMaxDz === null ? Math.max(0.05, 4.0 * interfaceDz) : atmMaxDz;
    airSegs.push(bulkSegment(LAtm, interfaceDz, cap, atmGrowth));
  }

  const airDz = concatSegments(airSegs);
  const dzArr = airDz.length > 0 ? concatSegments([bedDz, airDz]) : bedDz.slice();
  return { dzArr, nZBed: bedDz.length };
}

/**
 * The legacy z-axis: uniform (or wall-stretched) bed, then buffer cells that
 * expand by `dz_expansion`.
 *
 * `atm_growth` and `atm_max_dz` play NO PART here — that is the whole point of
 * §7.6. With the default dz_expansion = 1.0 the buffer is uniform at the bed
 * cell size all the way to Lz.
 *
 * When dz_first and bl_growth are both set, n_z_bed becomes a SOFT TARGET: the
 * geometric stack is built until it would overshoot h_bed, and the final cell
 * absorbs the remainder so the bed is exactly h_bed thick. The actual cell
 * count is whatever that produced.
 */
export function buildZAxisLegacy({
  hBed, Lz, dx, nZBed, dzExpansion = 1.0, dzFirst = null, blGrowth = 1.0,
  dzFirstAbove = null, blGrowthAbove = 1.3, bedRefineTop = false,
}) {
  let bedDz;
  let dzBed;
  if (nZBed > 0 && dzFirst !== null && dzFirst > 0.0 && blGrowth > 1.0 + 1e-12) {
    const list = [dzFirst];
    let cum = dzFirst;
    for (let i = 0; i < nZBed - 1; i++) {
      const next = list[list.length - 1] * blGrowth;
      if (cum + next >= hBed) break;
      list.push(next);
      cum += next;
    }
    const tail = hBed - cum;
    if (tail > 0.0) list.push(tail);
    bedDz = Float64Array.from(list);
    // Flip so the thin cells sit at the TOP of the bed instead of the bottom
    // — same cells, same h_bed, used upstream to isolate which end of the bed
    // drives the resolution gain (top = where downward DOM lands; bottom =
    // where wind shears against the ground).
    if (bedRefineTop) bedDz.reverse();
    nZBed = bedDz.length;
    dzBed = bedDz[bedDz.length - 1];
  } else {
    dzBed = hBed / Math.max(nZBed, 1);
    bedDz = new Float64Array(nZBed).fill(dzBed);
  }

  // Cold-flow-only wall BL, for n_z_bed = 0. With a bed present there is no
  // gas-phase BL at z=0 to resolve — the bed sits on the wall.
  let blDz = new Float64Array(0);
  let dzBedForBuffer;
  if (nZBed === 0 && dzFirst !== null && dzFirst > 0.0) {
    if (blGrowth <= 1.0 + 1e-12) {
      blDz = Float64Array.from([dzFirst]);   // capped at one; growth>1 is normal
    } else {
      const list = [];
      let dz = dzFirst;
      while (dz < dx && list.length < 200) { list.push(dz); dz *= blGrowth; }
      blDz = Float64Array.from(list);
    }
    dzBedForBuffer = dx;
  } else {
    dzBedForBuffer = dzBed;
  }

  // BL above the bed: resolves the steep T_g gradient at the bed-top/plume
  // interface. Without it the first cell above the bed is a full dz_bed thick,
  // too coarse for the layer between flame body and bed surface.
  let blAboveDz = new Float64Array(0);
  if (nZBed > 0 && dzFirstAbove !== null && dzFirstAbove > 0.0
      && dzFirstAbove < dzBedForBuffer && blGrowthAbove > 1.0 + 1e-12) {
    const list = [dzFirstAbove];
    while (list[list.length - 1] * blGrowthAbove < dzBedForBuffer) {
      list.push(list[list.length - 1] * blGrowthAbove);
      if (list.length >= 200) break;
    }
    blAboveDz = Float64Array.from(list);
  }

  const targetBuffer = Lz - hBed - sum(blDz) - sum(blAboveDz);
  let bufDz;
  if (dzExpansion <= 1.0 + 1e-12) {
    const nBuf = Math.max(1, Math.round(targetBuffer / dzBedForBuffer));
    bufDz = new Float64Array(nBuf).fill(dzBedForBuffer);
  } else {
    const ratio = (targetBuffer * (dzExpansion - 1.0)) / dzBedForBuffer + 1.0;
    const nBuf = Math.max(1, Math.ceil(Math.log(ratio) / Math.log(dzExpansion)));
    bufDz = new Float64Array(nBuf);
    for (let j = 0; j < nBuf; j++) bufDz[j] = dzBedForBuffer * Math.pow(dzExpansion, j);
  }

  return {
    dzArr: concatSegments([bedDz, blDz, blAboveDz, bufDz]),
    nZBedActual: nZBed,
    dzBed,
  };
}

// ── Grid ──────────────────────────────────────────────────────────────

/**
 * Full grid metadata: extents, cell centres, and the precomputed vertical
 * face distances every kernel reads.
 *
 * d_face_above[k] and d_face_below[k] are CENTRE-TO-CENTRE distances, and at
 * the two ends they are the half-cell to the ghost, not the full cell. Several
 * kernels divide by these, so the end values are load-bearing rather than
 * placeholders.
 */
export function buildGrid3D({
  Lx, Ly, Lz, dx, dy = null, hBed, nZBed,
  dzExpansion = 1.0, dzFirst = null, blGrowth = 1.0,
  dzFirstAbove = null, blGrowthAbove = 1.3, bedRefineTop = false,
  wallBlN = 0, wallBlFirstDz = 0.0, wallBlGrowth = 1.3,
  bedTopInnerBlN = 0, bedTopInnerBlFirstDz = 0.0, bedTopInnerBlGrowth = 1.3,
  bedTopOuterBlN = 0, bedTopOuterBlFirstDz = 0.0, bedTopOuterBlGrowth = 1.3,
  atmMaxDz = null, atmGrowth = 1.3, atmUniformDz = null,
}) {
  if (Lz < hBed) throw new Error(`Lz=${Lz} must exceed h_bed=${hBed}`);
  if (dzExpansion < 1.0) throw new Error(`dz_expansion=${dzExpansion} must be >= 1.0`);
  if (blGrowth < 1.0) throw new Error(`bl_growth=${blGrowth} must be >= 1.0`);

  const nx = Math.max(2, Math.round(Lx / dx));
  const dyTarget = dy === null ? dx : dy;
  const ny = Math.max(1, Math.round(Ly / dyTarget));
  const dyOut = ny > 0 ? Ly / ny : 1.0;

  // Same dispatch as upstream: the segment stack only when a BL is asked for.
  const useSegmentStack = wallBlN > 0 || bedTopInnerBlN > 0 || bedTopOuterBlN > 0;
  let dzArr, nZBedActual, dzBed;
  if (useSegmentStack) {
    ({ dzArr, nZBed: nZBedActual } = buildZAxisBedAtm({
      hBed, Lz, nZBed,
      wallBlN, wallBlFirstDz, wallBlGrowth,
      bedTopInnerBlN, bedTopInnerBlFirstDz, bedTopInnerBlGrowth,
      bedTopOuterBlN, bedTopOuterBlFirstDz, bedTopOuterBlGrowth,
      atmMaxDz, atmGrowth, atmUniformDz,
    }));
    dzBed = hBed / Math.max(nZBed, 1);
  } else {
    ({ dzArr, nZBedActual, dzBed } = buildZAxisLegacy({
      hBed, Lz, dx, nZBed, dzExpansion, dzFirst, blGrowth,
      dzFirstAbove, blGrowthAbove, bedRefineTop,
    }));
  }
  const nz = dzArr.length;

  const zFace = new Float64Array(nz + 1);
  for (let k = 0; k < nz; k++) zFace[k + 1] = zFace[k] + dzArr[k];
  const zMid = new Float64Array(nz);
  for (let k = 0; k < nz; k++) zMid[k] = zFace[k] + 0.5 * dzArr[k];

  const dFaceAbove = new Float64Array(nz);
  const dFaceBelow = new Float64Array(nz);
  for (let k = 0; k < nz; k++) {
    dFaceAbove[k] = k + 1 < nz ? 0.5 * (dzArr[k] + dzArr[k + 1]) : 0.5 * dzArr[k];
    dFaceBelow[k] = k - 1 >= 0 ? 0.5 * (dzArr[k] + dzArr[k - 1]) : 0.5 * dzArr[k];
  }

  const dxOut = Lx / nx;
  const xMid = new Float64Array(nx);
  for (let i = 0; i < nx; i++) xMid[i] = (i + 0.5) * dxOut;
  const yMid = new Float64Array(ny);
  for (let j = 0; j < ny; j++) yMid[j] = (j + 0.5) * dyOut;

  const invDzArr = new Float64Array(nz);
  for (let k = 0; k < nz; k++) invDzArr[k] = 1.0 / dzArr[k];

  return {
    nx, ny, nz,
    dx: dxOut, dy: dyOut, dz: hBed / Math.max(nZBed, 1),
    Lx, Ly, Lz: zFace[nz],
    nZBed: nZBedActual,
    xMid, yMid, zMid,
    dzArr, invDzArr, zFace, dFaceAbove, dFaceBelow,
  };
}
