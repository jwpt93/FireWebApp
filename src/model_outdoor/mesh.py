"""1D mesh axis construction kernel — Phase 14ag.

Composes an axis (cells in z, x, or y) from an ordered list of segments.
Each segment specifies its own length-and-cell-count behavior; the kernel
just concatenates segment cell arrays.  This lets you build complex
non-uniform meshes (fuel bed + BL inflation at the bed-top interface +
bulk atmosphere with growing cells) with a clear, composable structure
instead of a single function with nested ``if`` branches.

Segment types
-------------
* :class:`UniformSegment` — N cells of dz = L/N (uniform fill).
* :class:`InflationSegment` — N cells geometrically growing from a face;
  ``first_dz`` at the *near* end, ``growth`` ratio per cell.  ``reverse=True``
  puts the thin cell at the *far* end (useful at internal interfaces where
  you want refinement on one side of the segment boundary).
* :class:`BulkSegment` — fills a prescribed length ``L`` with cells growing
  geometrically from ``interface_dz`` up to ``max_dz`` then uniform.  Used
  for atmosphere bulk above the bed — keeps cell count moderate while
  avoiding the over-refined uniform stack we had before.

Wrapper
-------
:func:`build_z_axis_bed_atm` is the high-level convenience for the
outdoor 3-D solver.  Takes physical parameters (h_bed, n_z_bed, Lz, BL
configs at top-air / top-solid / wall) and assembles segments correctly.
The legacy ``Grid3D.build`` delegates here.

References
----------
None — this is a pure-numerical utility.  Consistent with conventional
finite-volume mesh-generation approaches (e.g., Pointwise, Gmsh) where
non-uniform 1-D distributions are composed from named regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class UniformSegment:
    """N cells of uniform size = L/N covering length L."""
    L: float
    N: int

    def cells(self) -> np.ndarray:
        if self.N <= 0 or self.L <= 0.0:
            return np.empty(0, dtype=np.float64)
        return np.full(self.N, self.L / self.N, dtype=np.float64)

    def thickness(self) -> float:
        return float(self.L)


@dataclass
class InflationSegment:
    """N geometrically-growing cells.

    Cell sizes (from near end to far end):
        dz_k = first_dz * growth^k    for k = 0, 1, ..., N-1

    Total thickness:
        L = first_dz * (growth^N - 1) / (growth - 1)     (growth > 1)
        L = N * first_dz                                  (growth == 1)

    ``reverse=True`` reverses the resulting dz array so the thin cell sits
    at the FAR end (useful at internal interfaces where you want fine
    cells next to the segment-after's interface, not the segment-before's).
    """
    N: int
    first_dz: float
    growth: float = 1.2
    reverse: bool = False

    def cells(self) -> np.ndarray:
        if self.N <= 0 or self.first_dz <= 0.0:
            return np.empty(0, dtype=np.float64)
        if abs(self.growth - 1.0) < 1e-12:
            dzs = np.full(self.N, self.first_dz, dtype=np.float64)
        else:
            dzs = self.first_dz * (self.growth ** np.arange(self.N, dtype=np.float64))
        if self.reverse:
            dzs = dzs[::-1].copy()
        return dzs

    def thickness(self) -> float:
        return float(self.cells().sum())


@dataclass
class BulkSegment:
    """Fill length L with cells geometrically growing from interface_dz to
    max_dz, then uniform at max_dz.

    Starts with the first cell of size ``interface_dz * growth`` (so cells
    are continuous in size across the segment boundary, growing into the
    bulk by ``growth`` each cell, capped at ``max_dz``).

    Termination strategy (2026-06-01 rule-compliant): the geometric
    sequence is built strictly with ratio = ``growth`` between cells.
    The sequence stops when adding another cell would overshoot ``L`` by
    more than half a cell.  This lets the actual segment length drift
    slightly below the requested ``L`` (typically a few percent) rather
    than create a "remainder" final cell that violates the growth ratio.
    Stretch-rule compliance is preferred over exact ``L`` matching.
    """
    L: float
    interface_dz: float
    max_dz: float
    growth: float = 1.3

    def cells(self) -> np.ndarray:
        if self.L <= 0.0:
            return np.empty(0, dtype=np.float64)
        cells = []
        cum = 0.0
        next_dz = self.interface_dz * self.growth
        # Cap at max_dz from the start so the bulk region doesn't get a
        # cell larger than max_dz on the first step.
        next_dz = min(next_dz, self.max_dz)
        while cum < self.L:
            dz = min(next_dz, self.max_dz)
            # Rule-compliant termination: stop when adding ``dz`` would
            # overshoot ``L`` by more than half a cell.  Lz then drifts
            # slightly below requested rather than admitting a final
            # cell whose ratio to its predecessor violates ``growth``.
            if cum + dz > self.L + 0.5 * dz:
                break
            cells.append(dz)
            cum += dz
            next_dz = dz * self.growth   # geometric growth, will be clipped at max_dz on next iter
        return np.array(cells, dtype=np.float64)

    def thickness(self) -> float:
        return float(self.L)


def build_axis(segments: List) -> np.ndarray:
    """Concatenate segment cell arrays into a single dz_arr."""
    arrays = [seg.cells() for seg in segments]
    arrays = [a for a in arrays if a.size > 0]
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays)


# ───────────────────────── z-axis convenience wrapper ─────────────────────────

def build_z_axis_bed_atm(
    *,
    h_bed: float,
    Lz: float,
    n_z_bed: int,
    # Wall BL inside bed near z=0 (config 4):
    wall_bl_N: int = 0,
    wall_bl_first_dz: float = 0.0,
    wall_bl_growth: float = 1.3,
    # Solid-side BL at top of bed (config 3): thin cells just below z=h_bed
    bed_top_inner_bl_N: int = 0,
    bed_top_inner_bl_first_dz: float = 0.0,
    bed_top_inner_bl_growth: float = 1.3,
    # Air-side BL above bed (config 2): thin cells just above z=h_bed
    bed_top_outer_bl_N: int = 0,
    bed_top_outer_bl_first_dz: float = 0.0,
    bed_top_outer_bl_growth: float = 1.3,
    # Bulk atmosphere above the outer BL
    atm_max_dz: float | None = None,
    atm_growth: float = 1.3,
    atm_uniform_dz: float | None = None,   # if set: atm is uniform at this size
) -> tuple[np.ndarray, int]:
    """Build the z-axis as: [bed] + [outer-air BL] + [bulk atmosphere].

    The bed segment is composed of (in order, all optional):
        1. wall BL (thin cells from z=0 up)              [config 4]
        2. uniform bulk-bed cells (fills middle)
        3. solid-side top BL (thin cells reaching z=h_bed) [config 3]

    Above the bed, optionally:
        4. air-side BL (thin cells from z=h_bed up)       [config 2]
        5. bulk atmosphere (cells grow to atm_max_dz, capped)

    ``n_z_bed`` is the TOTAL number of cells in the bed (wall BL + bulk +
    inner top BL).  The bulk-bed cell count is computed as
    ``n_z_bed - wall_bl_N - bed_top_inner_bl_N`` and must be ≥ 0.

    Returns
    -------
    dz_arr : np.ndarray
        Per-cell vertical spacings (in order from z=0 upward).
    n_z_bed_actual : int
        Number of cells with z_mid < h_bed (the bed cells, used to index
        bed-only fields).
    """
    if h_bed < 0 or Lz <= h_bed or n_z_bed <= 0:
        raise ValueError(
            f"h_bed={h_bed}, Lz={Lz}, n_z_bed={n_z_bed} must satisfy "
            f"h_bed >= 0, Lz > h_bed, n_z_bed > 0"
        )

    # ── Build the bed segments ───────────────────────────────────────────
    bed_segs: List = []

    # 1. Wall BL inside bed (z=0 → z=wall_bl_thickness)
    if wall_bl_N > 0 and wall_bl_first_dz > 0.0:
        wall_seg = InflationSegment(
            N=wall_bl_N, first_dz=wall_bl_first_dz,
            growth=wall_bl_growth, reverse=False,  # thin cell at NEAR end (z=0)
        )
        bed_segs.append(wall_seg)
        bed_used = wall_seg.thickness()
    else:
        bed_used = 0.0

    # 3. Top-inner BL inside bed (z=h_bed - top_thickness → z=h_bed)
    if bed_top_inner_bl_N > 0 and bed_top_inner_bl_first_dz > 0.0:
        top_inner_seg = InflationSegment(
            N=bed_top_inner_bl_N, first_dz=bed_top_inner_bl_first_dz,
            growth=bed_top_inner_bl_growth, reverse=True,  # thin cell at FAR end (z=h_bed)
        )
        top_inner_thickness = top_inner_seg.thickness()
    else:
        top_inner_seg = None
        top_inner_thickness = 0.0

    # 2. Uniform bulk-bed (fills h_bed - wall - top_inner)
    n_bulk_bed = n_z_bed - wall_bl_N - bed_top_inner_bl_N
    if n_bulk_bed < 0:
        raise ValueError(
            f"n_z_bed={n_z_bed} smaller than wall_bl_N + bed_top_inner_bl_N "
            f"({wall_bl_N + bed_top_inner_bl_N}); not enough cells for bulk"
        )
    L_bulk_bed = h_bed - bed_used - top_inner_thickness
    if L_bulk_bed < 0:
        raise ValueError(
            f"Wall BL + top-inner BL thickness ({bed_used + top_inner_thickness}) "
            f"exceeds h_bed ({h_bed})"
        )
    if n_bulk_bed > 0:
        bulk_seg = UniformSegment(L=L_bulk_bed, N=n_bulk_bed)
        bed_segs.append(bulk_seg)
    elif L_bulk_bed > 1e-3 * h_bed:
        # > 0.1% of h_bed → genuine geometry mismatch
        raise ValueError(
            f"n_bulk_bed=0 but L_bulk_bed={L_bulk_bed:.4f} > 0 — increase "
            f"n_z_bed or reduce BL cell counts"
        )
    elif L_bulk_bed > 1e-9:
        # Tiny FP residual; absorb into the top of the wall BL stack to
        # preserve h_bed exactly without adding a degenerate bulk cell.
        if bed_segs:
            last_cells = bed_segs[-1].cells()
            if isinstance(bed_segs[-1], InflationSegment):
                # Adjust by recasting as fixed cells with absorbed remainder
                pass   # tolerated; FP residual will appear in final sum
        # No raise — sub-mm residual is fine

    if top_inner_seg is not None:
        bed_segs.append(top_inner_seg)

    bed_dz = build_axis(bed_segs)

    # Adjust last cell to absorb any FP residual so bed total = h_bed exactly.
    # The geometric InflationSegment sums can leave a tiny mismatch (~mm at
    # the ratio of growth^N).  Absorb into the largest cell which is the
    # least sensitive to the small change.
    bed_total = bed_dz.sum()
    residual = h_bed - bed_total
    if abs(residual) > 1e-6 * h_bed:
        if abs(residual) > 0.05 * h_bed:
            raise RuntimeError(
                f"Bed segments sum to {bed_total} but h_bed={h_bed} — "
                f"residual >5% of h_bed indicates a real geometry error"
            )
        # Spread the residual proportionally across all bed cells
        bed_dz = bed_dz * (h_bed / bed_total)

    # ── Air segments above the bed ───────────────────────────────────────
    air_segs: List = []
    interface_dz = bed_dz[-1]  # top cell of bed (where air meets bed)

    # 4. Air-side BL above bed (z=h_bed → z=h_bed + outer_thickness)
    if bed_top_outer_bl_N > 0 and bed_top_outer_bl_first_dz > 0.0:
        outer_seg = InflationSegment(
            N=bed_top_outer_bl_N, first_dz=bed_top_outer_bl_first_dz,
            growth=bed_top_outer_bl_growth, reverse=False,  # thin cell at NEAR end (z=h_bed)
        )
        air_segs.append(outer_seg)
        interface_dz = outer_seg.cells()[-1]

    # 5. Bulk atmosphere (z=h_bed + outer → z=Lz)
    L_atm = Lz - h_bed - sum(s.thickness() for s in air_segs)
    if atm_uniform_dz is not None and atm_uniform_dz > 0 and L_atm > 0:
        # Forced-uniform atm cells (e.g. h_bed/8 to match baseline bulksize
        # regardless of bed BL structure).  Independent of interface_dz so
        # the atm mesh stays identical across BL-location experiments.
        n_uniform = max(1, int(round(L_atm / atm_uniform_dz)))
        air_segs.append(UniformSegment(
            L=n_uniform * atm_uniform_dz, N=n_uniform,
        ))
    elif L_atm > 0:
        if atm_max_dz is None:
            # Default: cap at 4× the bed-top cell (modest growth)
            atm_max_dz = max(0.05, 4.0 * interface_dz)
        air_segs.append(BulkSegment(
            L=L_atm, interface_dz=interface_dz,
            max_dz=atm_max_dz, growth=atm_growth,
        ))

    air_dz = build_axis(air_segs)
    dz_arr = np.concatenate([bed_dz, air_dz]) if air_dz.size > 0 else bed_dz.copy()

    n_z_bed_actual = bed_dz.size
    return dz_arr, n_z_bed_actual
