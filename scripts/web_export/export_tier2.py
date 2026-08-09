#!/usr/bin/env python3
"""Precompute Tier-2 (1D line spread) results for the website applet.

Runs ``run_1d_spread`` over the Cheney case matrix and writes one JSON per
run into ``docs/data/tier2/`` plus an ``index.json`` catalogue.  The static
site loads these in the Tier-2 result viewer — no backend needed.

Case matrix
-----------
  * cheney_nat4  — Cheney 1993 natural pasture, M=4 %  (web export deck)
  * cheney_cut4  — Cheney 1993 cut pasture,     M=4 %  (web export deck)
    each at U_10 ∈ {0.5, 1, 2, 3, 4, 6, 8} m/s (wind applied via the
    ``wind_speed_m_s`` kwarg, overriding the deck).
  * gr1_free_burn — the vendored POC deck as-is (Anderson GR1, U=0,
    free-burning pulse; documents the no-spread outcome).

Usage (from repo root):
    OMP_NUM_THREADS=8 .venv/bin/python scripts/web_export/export_tier2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))  # allow running without install

from model_outdoor.spread import SpreadConfig, run_1d_spread  # noqa: E402

DECKS = REPO / "scripts" / "web_export" / "decks"
OUT_DIR = REPO / "docs" / "data" / "tier2"

WIND_SWEEP_M_S = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
MAX_WALL_TIME_S = 300.0
MAX_TRACE_POINTS = 400  # downsample per-cell traces to keep JSON small

# (case_id, deck, human label, fuel params for the Tier-1 cross-link)
CASES = [
    (
        "cheney_nat4",
        DECKS / "Cheney_Nat4_tier2.txt",
        "Cheney natural pasture, M = 4%",
        {"a_ch": 0.406, "moisture_frac": 0.04, "fuel": "natural"},
    ),
    (
        "cheney_cut4",
        DECKS / "Cheney_Cut4_tier2.txt",
        "Cheney cut pasture, M = 4%",
        {"a_ch": 0.343, "moisture_frac": 0.04, "fuel": "cut"},
    ),
    (
        "gr1_free_burn",
        REPO / "data" / "validation_cases" / "Outdoor_Grass_GR1__free_burn.txt",
        "Anderson GR1 short grass, M = 5%, free-burning",
        {"a_ch": 0.406, "moisture_frac": 0.05, "fuel": "natural"},
    ),
]


def _downsample(t: np.ndarray, y: np.ndarray, n: int = MAX_TRACE_POINTS):
    """Evenly-spaced index downsample; always keeps the endpoints."""
    if len(t) <= n:
        return t.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(t) - 1, n).astype(int))
    return t[idx].tolist(), y[idx].tolist()


def export_run(case_id: str, deck: Path, wind: float) -> dict:
    # SpreadConfig defaults (20 cells, dx=0.30 m, chi_rad=0.25) plus the
    # calibrated convective preheat alpha_conv_preheat=0.010 — calibrated for
    # GR1 (h_bed=0.30 m, Anderson 1982) per the SpreadConfig docstring
    # ("scan from 0.005→0.10; pick minimum α giving 6/6 PASS" across the
    # U=0.5..8 wind cases).  Radiation-only (alpha=0) stalls after cell 1.
    cfg = SpreadConfig(alpha_conv_preheat=0.010)
    t0 = time.time()
    res = run_1d_spread(
        deck, cfg, wind_speed_m_s=wind, max_wall_time_s=MAX_WALL_TIME_S
    )
    elapsed = time.time() - t0

    cells = []
    for t, hrrpua in zip(res.cell_t, res.cell_hrrpua):
        t_ds, h_ds = _downsample(np.asarray(t, float), np.asarray(hrrpua, float))
        cells.append({"t_s": t_ds, "hrrpua_kW_m2": h_ds})

    payload = {
        "case_id": case_id,
        "deck": deck.name,
        "wind_speed_m_s": wind,
        "max_wall_time_s": MAX_WALL_TIME_S,
        "ros_m_s": res.ros_m_s,
        "n_cells_ignited": res.n_cells_ignited,
        "t_ignition_s": res.t_ignition,
        "spread_config": {
            "max_cells": cfg.max_cells,
            "dx_m": cfg.dx_m,
            "chi_rad_spread": cfg.chi_rad_spread,
            "hrrpua_ign_kW_m2": cfg.hrrpua_ign_kW_m2,
            "kappa_flame_m": cfg.kappa_flame_m,
            "alpha_conv_preheat": cfg.alpha_conv_preheat,
        },
        "cells": cells,
    }
    print(
        f"  {case_id} U={wind:>4}: {res.n_cells_ignited} cells ignited, "
        f"ROS={res.ros_m_s:.4f} m/s  ({elapsed:.1f}s)",
        flush=True,
    )
    return payload


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []

    for case_id, deck, label, fuel in CASES:
        winds = (0.0,) if case_id == "gr1_free_burn" else WIND_SWEEP_M_S
        for wind in winds:
            # g-format matches JS String(w): 4.0 -> "U4", 0.5 -> "U0p5", 0.0 -> "U0"
            run_id = f"{case_id}__U{f'{wind:g}'.replace('.', 'p')}"
            payload = export_run(run_id, deck, wind)
            payload.update(label=label, **fuel)
            (OUT_DIR / f"{run_id}.json").write_text(json.dumps(payload))
            index.append(
                {
                    k: payload[k]
                    for k in (
                        "case_id",
                        "label",
                        "fuel",
                        "a_ch",
                        "moisture_frac",
                        "wind_speed_m_s",
                        "ros_m_s",
                        "n_cells_ignited",
                        "t_ignition_s",
                    )
                }
            )

    (OUT_DIR / "index.json").write_text(json.dumps({"runs": index}, indent=1))
    print(f"\nwrote {len(index)} runs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
