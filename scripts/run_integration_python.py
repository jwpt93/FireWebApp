"""Run the integration case through the REAL Python solver.

Writes the result to docs/data/integration_reference.json for
scripts/integration_test.mjs to compare against.

This is the whole-solver counterpart to the kernel vectors. The vectors prove
each kernel in isolation; this proves the ORDERING — that the operator split,
the sub-step structure and the state threading were transcribed correctly. A
kernel-level bug shows up in the vectors; a swapped stage or a missing state
update only shows up here.

WHAT IS COMPARED, and at what standard: ROS and a handful of scalar state
summaries, within a BAND, not elementwise and not bit-exact. Per the
verification contract in SOLVER_PORT.md §4, the two codes are two valid
solutions of the same model, not one replaying the other. EDC's extinction
gates are discontinuous, so a 1-ulp difference in omega can cross a threshold
and move a cell by degrees; over hundreds of steps those diverge in detail
while the bulk answer stays put.

Run:
    OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
      /home/jw/.venvs/unitiedmodel2/bin/python scripts/run_integration_python.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = Path("/home/jw/projects/unitiedmodel2")
sys.path.insert(0, str(PARENT))

DECK = ROOT / "scripts" / "integration_case.txt"
OUT = ROOT / "docs" / "data" / "integration_reference.json"


def main() -> None:
    from model_outdoor.spread_3d import run_3d_spread

    t0 = time.perf_counter()
    res = run_3d_spread(
        DECK,
        wind_speed_m_s=4.0,
        max_wall_time_s=0.6,
        turbulence_model="k_epsilon",
        radiation_solver="dom",
    )
    wall = time.perf_counter() - t0

    grid = res.grid if hasattr(res, "grid") else None
    payload = {
        "_meta": {
            "purpose": "Whole-solver cross-check: the JS port's ORDERING "
                       "against the Python reference.",
            "standard": "ROS within a band, NOT elementwise. See "
                        "SOLVER_PORT.md section 4.",
            "deck": str(DECK.relative_to(ROOT)),
            "wall_s": round(wall, 2),
        },
        "ros_m_s": float(res.ros_m_s) if hasattr(res, "ros_m_s") else None,
        "front_t": [float(x) for x in getattr(res, "front_t", [])],
        "front_x": [float(x) for x in getattr(res, "front_x", [])],
        "n_steps": len(res.diag_t),
        # Per-step maxima: a trajectory, not just an endpoint. If the two codes
        # agree at t_end but diverged in between, that shows up here and
        # nowhere else.
        "diag_t": [float(x) for x in res.diag_t],
        "diag_Tg_max": [float(x) for x in res.diag_Tg_max],
        "diag_Ts_max": [float(x) for x in res.diag_Ts_max],
        "diag_Y_max": [float(x) for x in res.diag_Y_max],
        "diag_omega_max": [float(x) for x in res.diag_omega_max],
    }
    # Scalar state summaries: cheap, and they localise a disagreement to a
    # stage. If T_g agrees but T_s does not, the bed step is the suspect; if
    # both agree but the front does not, it is the level set.
    for name in ("T_g", "T_s", "Y_fuel", "Y_O2", "rho", "u"):
        arr = getattr(res.state_final, name, None)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        payload[f"{name}_max"] = float(a.max())
        payload[f"{name}_min"] = float(a.min())
        payload[f"{name}_mean"] = float(a.mean())
    if grid is not None:
        payload["grid"] = {"nz": int(grid.Nz), "ny": int(grid.Ny),
                           "nx": int(grid.Nx), "n_z_bed": int(grid.n_z_bed),
                           "dx": float(grid.dx), "dy": float(grid.dy),
                           "Lz": float(grid.Lz)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}  ({wall:.1f}s wall)")
    for k, v in payload.items():
        if k not in ("_meta", "front_t", "front_x"):
            print(f"  {k}: {v}")
    print(f"  front samples: {len(payload['front_t'])}")


if __name__ == "__main__":
    main()
