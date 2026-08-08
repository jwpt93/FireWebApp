"""Unit tests for model_outdoor/mesh.py — Phase 14ag mesh kernel.

Tests segment behavior, composition, axis assembly, and the
build_z_axis_bed_atm convenience wrapper.

Per Rule #18 (unit tests required for every new module), these tests must
pass before any sweep is run.
"""
from __future__ import annotations

import numpy as np
import pytest

from model_outdoor.mesh import (
    UniformSegment,
    InflationSegment,
    BulkSegment,
    build_axis,
    build_z_axis_bed_atm,
)


# ─── UniformSegment ──────────────────────────────────────────────────────────

def test_uniform_segment_cells_match_L_over_N():
    seg = UniformSegment(L=0.5, N=10)
    dzs = seg.cells()
    assert dzs.shape == (10,)
    assert np.allclose(dzs, 0.05)
    assert abs(dzs.sum() - 0.5) < 1e-15


def test_uniform_segment_zero_N_returns_empty():
    seg = UniformSegment(L=1.0, N=0)
    assert seg.cells().size == 0


def test_uniform_segment_zero_L_returns_empty():
    seg = UniformSegment(L=0.0, N=5)
    assert seg.cells().size == 0


# ─── InflationSegment ────────────────────────────────────────────────────────

def test_inflation_segment_geometric_growth():
    seg = InflationSegment(N=5, first_dz=0.01, growth=2.0)
    dzs = seg.cells()
    expected = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    assert np.allclose(dzs, expected)
    # Thickness: 0.01 * (2^5 - 1) / (2 - 1) = 0.31
    assert abs(seg.thickness() - 0.31) < 1e-15


def test_inflation_segment_growth_one_yields_uniform():
    seg = InflationSegment(N=4, first_dz=0.025, growth=1.0)
    dzs = seg.cells()
    assert np.allclose(dzs, 0.025)


def test_inflation_segment_reverse_flips_array():
    seg_fwd = InflationSegment(N=5, first_dz=0.01, growth=1.5, reverse=False)
    seg_rev = InflationSegment(N=5, first_dz=0.01, growth=1.5, reverse=True)
    assert np.allclose(seg_fwd.cells(), seg_rev.cells()[::-1])
    # Same total thickness
    assert abs(seg_fwd.thickness() - seg_rev.thickness()) < 1e-15


# ─── BulkSegment ─────────────────────────────────────────────────────────────

def test_bulk_segment_total_matches_L_exactly():
    seg = BulkSegment(L=2.5, interface_dz=0.05, max_dz=0.30, growth=1.3)
    dzs = seg.cells()
    assert abs(dzs.sum() - 2.5) < 1e-12, (
        f"BulkSegment total {dzs.sum()} != L=2.5"
    )


def test_bulk_segment_first_cell_grows_from_interface():
    """First cell should be interface_dz * growth (or max_dz if smaller)."""
    seg = BulkSegment(L=5.0, interface_dz=0.020, max_dz=0.30, growth=1.5)
    dzs = seg.cells()
    # First cell = 0.020 * 1.5 = 0.030
    assert abs(dzs[0] - 0.030) < 1e-12


def test_bulk_segment_caps_at_max_dz():
    """Cells should plateau at max_dz once growth would exceed it."""
    seg = BulkSegment(L=10.0, interface_dz=0.020, max_dz=0.30, growth=1.5)
    dzs = seg.cells()
    # Most cells should be at max_dz=0.30 (the last is the remainder)
    interior = dzs[1:-1]
    assert (interior <= 0.30 + 1e-12).all()
    # Once cells plateau, they should equal max_dz
    plateau_count = np.sum(np.abs(interior - 0.30) < 1e-10)
    assert plateau_count > 5, f"Expected plateau at max_dz, got cells: {interior[:15]}"


# ─── build_axis composition ──────────────────────────────────────────────────

def test_build_axis_concatenates_segments():
    segs = [
        UniformSegment(L=1.0, N=4),     # 4 cells of 0.25
        InflationSegment(N=3, first_dz=0.1, growth=1.5),  # 0.1, 0.15, 0.225
    ]
    dz_arr = build_axis(segs)
    assert dz_arr.size == 7
    assert np.allclose(dz_arr[:4], 0.25)
    assert np.allclose(dz_arr[4:], [0.1, 0.15, 0.225])


def test_build_axis_empty_list_returns_empty():
    assert build_axis([]).size == 0


# ─── build_z_axis_bed_atm convenience wrapper ────────────────────────────────

def test_z_axis_uniform_bed_no_BL_matches_legacy():
    """Default settings should give: uniform bed cells + bulk atm (growing)."""
    dz_arr, n_z_bed = build_z_axis_bed_atm(
        h_bed=0.37, Lz=8.0, n_z_bed=4,
    )
    # First 4 cells = bed, uniform at h_bed/4 = 0.0925
    assert n_z_bed == 4
    assert np.allclose(dz_arr[:4], 0.37 / 4)
    # Total sums to Lz
    assert abs(dz_arr.sum() - 8.0) < 1e-9
    # Atmosphere should NOT be uniform (it's a BulkSegment that grows)
    # So there should be at least one cell > h_bed/4
    assert dz_arr[4:].max() > 0.0925


def test_z_axis_config_2_outer_BL_only():
    """Air-side BL above bed: thin cells appear just above z=h_bed."""
    dz_arr, n_z_bed = build_z_axis_bed_atm(
        h_bed=0.37, Lz=8.0, n_z_bed=4,
        bed_top_outer_bl_N=5,
        bed_top_outer_bl_first_dz=0.010,
        bed_top_outer_bl_growth=1.3,
    )
    # Bed cells unchanged: 4 of 0.0925
    assert n_z_bed == 4
    assert np.allclose(dz_arr[:4], 0.0925)
    # Cells 4..8 should be the outer BL: 0.010, 0.013, 0.0169, 0.02197, 0.02856
    expected_bl = 0.010 * 1.3 ** np.arange(5)
    assert np.allclose(dz_arr[4:9], expected_bl)
    # Total sums to Lz
    assert abs(dz_arr.sum() - 8.0) < 1e-9


def test_z_axis_config_3_solid_top_BL_only():
    """Solid-side BL at top of bed: thin cells just below z=h_bed."""
    dz_arr, n_z_bed = build_z_axis_bed_atm(
        h_bed=0.37, Lz=8.0, n_z_bed=8,
        bed_top_inner_bl_N=4,
        bed_top_inner_bl_first_dz=0.005,
        bed_top_inner_bl_growth=1.3,
    )
    # n_z_bed=8 cells; 4 are bulk + 4 are top-inner BL.
    assert n_z_bed == 8
    # Top-inner BL has thin cell at the FAR end (highest z, at z=h_bed)
    # So inside bed, dz_arr[7] (top of bed) should be thinnest (= 0.005)
    bed = dz_arr[:8]
    assert abs(bed[-1] - 0.005) < 1e-12, f"Top cell of bed: {bed[-1]}"
    # Bulk part should be uniform
    bulk_thickness = 0.005 * (1.3**4 - 1) / 0.3  # = 0.0309
    bulk_L = 0.37 - bulk_thickness
    assert abs(bed[:4].sum() - bulk_L) < 1e-9
    # Total bed = h_bed
    assert abs(bed.sum() - 0.37) < 1e-9


def test_z_axis_config_4_wall_BL_only():
    """Wall BL inside bed near z=0: thin cells just above z=0."""
    dz_arr, n_z_bed = build_z_axis_bed_atm(
        h_bed=0.37, Lz=8.0, n_z_bed=8,
        wall_bl_N=4,
        wall_bl_first_dz=0.005,
        wall_bl_growth=1.3,
    )
    assert n_z_bed == 8
    # Bottom cell (at z=0) should be the thinnest (= 0.005)
    assert abs(dz_arr[0] - 0.005) < 1e-12
    # Should grow geometrically for first 4 cells
    for k in range(4):
        assert abs(dz_arr[k] - 0.005 * 1.3**k) < 1e-12
    # Total bed = h_bed
    assert abs(dz_arr[:8].sum() - 0.37) < 1e-9


def test_z_axis_config_1_all_three_BLs():
    """All three BL locations active simultaneously."""
    dz_arr, n_z_bed = build_z_axis_bed_atm(
        h_bed=0.37, Lz=8.0, n_z_bed=12,
        wall_bl_N=4, wall_bl_first_dz=0.005, wall_bl_growth=1.3,
        bed_top_inner_bl_N=4, bed_top_inner_bl_first_dz=0.005,
        bed_top_inner_bl_growth=1.3,
        bed_top_outer_bl_N=5, bed_top_outer_bl_first_dz=0.010,
        bed_top_outer_bl_growth=1.3,
    )
    assert n_z_bed == 12
    # Bottom of bed: thin (wall BL)
    assert abs(dz_arr[0] - 0.005) < 1e-12
    # Top of bed (z=h_bed-eps): thin (inner BL, reversed)
    assert abs(dz_arr[11] - 0.005) < 1e-12
    # First air cell above bed: thin (outer BL)
    assert abs(dz_arr[12] - 0.010) < 1e-12
    # Total bed = h_bed
    assert abs(dz_arr[:12].sum() - 0.37) < 1e-9
    # Total = Lz
    assert abs(dz_arr.sum() - 8.0) < 1e-9


def test_z_axis_raises_when_bed_too_small_for_BLs():
    """Should raise if n_z_bed is smaller than wall_bl_N + bed_top_inner_bl_N."""
    with pytest.raises(ValueError, match="bulk"):
        build_z_axis_bed_atm(
            h_bed=0.37, Lz=8.0, n_z_bed=4,
            wall_bl_N=3, wall_bl_first_dz=0.005,
            bed_top_inner_bl_N=3, bed_top_inner_bl_first_dz=0.005,
        )


def test_z_axis_raises_when_BL_thickness_exceeds_h_bed():
    """Should raise if combined BL thickness > h_bed."""
    with pytest.raises(ValueError, match="exceeds h_bed"):
        build_z_axis_bed_atm(
            h_bed=0.10, Lz=8.0, n_z_bed=12,
            wall_bl_N=4, wall_bl_first_dz=0.020, wall_bl_growth=1.5,
            bed_top_inner_bl_N=4, bed_top_inner_bl_first_dz=0.020,
            bed_top_inner_bl_growth=1.5,
        )


def test_z_axis_total_sums_to_Lz_with_BLs():
    """Whatever BL config you pick, the total mesh sums exactly to Lz."""
    configs = [
        dict(),  # default
        dict(bed_top_outer_bl_N=5, bed_top_outer_bl_first_dz=0.01),
        dict(bed_top_inner_bl_N=3, bed_top_inner_bl_first_dz=0.005, n_z_bed=6),
        dict(wall_bl_N=4, wall_bl_first_dz=0.005, n_z_bed=8),
    ]
    for cfg in configs:
        defaults = dict(h_bed=0.37, Lz=8.0, n_z_bed=4)
        defaults.update(cfg)
        dz_arr, _ = build_z_axis_bed_atm(**defaults)
        assert abs(dz_arr.sum() - 8.0) < 1e-9, (
            f"Config {cfg}: sum = {dz_arr.sum()}, expected 8.0"
        )
