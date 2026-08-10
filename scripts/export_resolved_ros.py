"""Export resolved-solver rate-of-spread as a lookup table for the applet.

WHAT THIS IS
------------
The applet cannot run the 3D solver — a single case is 10 minutes to an hour.
But the solver has already been run, so the applet can carry its ANSWERS and
reproduce the parent project's validated Phase 19 / Phase 20 hybrid:

    U_10 <  threshold - width   pure Cheney regression
    U_10 in the blend window    linear ramp between the two
    U_10 >= threshold           pure resolved 3D solver

Phase 20 "Option B" settings (threshold 3.5 m/s, width 1.0) are used here.
The threshold is not a tuning knob: it is the wind speed below which the
resolved closure stops propagating correctly. Phase 19 used 1.4 and left a
hole — Nat 4% at U_10 = 2 resolved to 6.56 m/min against Cheney's 26.42
(ratio 0.248, the single failure in an otherwise 19/20 sweep). Option B
raises the threshold above that hole.

WHAT IS STORED, AND WHY IT IS A RATIO
-------------------------------------
Not raw ROS: the ratio ROS_resolved / ROS_Cheney at each grid point.

Raw resolved ROS spans 6 to 86 m/min over the sweep, so interpolating it
means interpolating a steep power law off two or three samples. The ratio
spans 0.58 to 0.94 — nearly flat — so linear interpolation is well
conditioned and clamping outside the sampled range is defensible. It is also
the honest description of what a resolved run contributes: a correction to
the regression, measured.

The applet reconstructs  ROS = ratio(fuel, M, U) x Cheney(U, M, a_ch).

WHICH POINTS QUALIFY AS "RESOLVED"
----------------------------------
Only cases whose own run had the empirical weight at zero. A case that was
itself blended or fully empirical would be circular — it would be the fit
wearing the solver's name. Under Phase 19 (threshold 1.4, width 0.5) that
means U_10 >= 1.4; under Phase 20 Option B (3.5, width 1.0), U_10 >= 3.5.

Run:
    .venv/bin/python scripts/export_resolved_ros.py [--src <diagnostics dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "resolved.json"

DEFAULT_SRC = Path("/home/jw/projects/unitiedmodel2/local/diagnostics")

# Phase 20 Option B — the blend the applet reproduces.
THRESHOLD_U10 = 3.5
BLEND_WIDTH = 1.0

# (sweep dir, threshold, width) of the run that PRODUCED each case, used to
# decide whether that case was resolved rather than blended or empirical.
SWEEPS = [
    ("cheney_phase19_sweep", 1.4, 0.5),
    ("phase20_blend_optB", 3.5, 1.0),
]


def empirical_weight(U, threshold, width):
    """Port of blend_resolved_empirical() in model_outdoor/empirical_ros.py."""
    if U >= threshold:
        return 0.0
    if width <= 0.0:
        return 1.0
    u_lo = threshold - width
    if U <= u_lo:
        return 1.0
    return (threshold - U) / width


def collect(src: Path):
    """Gather every case, tagged with how much of it was actually resolved."""
    out = []
    for sweep, thr, wid in SWEEPS:
        d = src / sweep
        if not d.is_dir():
            print(f"  ! missing sweep dir, skipped: {d}")
            continue
        for case in sorted(d.iterdir()):
            f = case / "result.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            U = r.get("U_m_s")
            ros = r.get("ROS_Ts_m_min")
            ref = r.get("eq6_ref_m_min")
            if U is None or not ros or not ref:
                continue
            w = empirical_weight(U, thr, wid)
            out.append({
                "sweep": sweep,
                "case": case.name,
                "fuel": r.get("fuel"),
                "mf_pct": r.get("mf_pct"),
                "U10_m_s": U,
                "ros_m_min": ros,
                "eq6_m_min": ref,
                "ratio": ros / ref,
                "w_emp_in_source_run": w,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="parent project's local/diagnostics directory")
    args = ap.parse_args()

    allcases = collect(args.src)
    if not allcases:
        raise SystemExit(f"no sweep results found under {args.src}")

    # Keep only genuinely resolved points, and only those the Option B blend
    # can actually reach (it never queries the table below the blend window).
    u_min = THRESHOLD_U10 - BLEND_WIDTH
    resolved = [c for c in allcases
                if c["w_emp_in_source_run"] == 0.0 and c["U10_m_s"] >= u_min]

    # Index by fuel + moisture, sorted in wind.
    table = {}
    for c in resolved:
        key = f"{c['fuel']}{c['mf_pct']:.0f}"
        table.setdefault(key, []).append([c["U10_m_s"], c["ratio"]])
    for k in table:
        table[k] = sorted(table[k])
        # Two runs can land on the same wind; keep the finer sweep's value by
        # averaging duplicates rather than silently dropping one.
        merged, i = [], 0
        while i < len(table[k]):
            u = table[k][i][0]
            same = [p[1] for p in table[k] if p[0] == u]
            merged.append([u, sum(same) / len(same)])
            i += len(same)
        table[k] = merged

    payload = {
        "_meta": {
            "purpose": "Resolved 3D-solver ROS as a correction ratio to the "
                       "Cheney regression, for the applet's hybrid mode.",
            "blend": {"model": "phase20_option_b",
                      "u_threshold_U10_m_s": THRESHOLD_U10,
                      "blend_width_m_s": BLEND_WIDTH,
                      "rule": "w_emp = 1 below (threshold-width); linear ramp; "
                              "0 at or above threshold"},
            "stored": "ratio = ROS_resolved / ROS_cheney_eq6, indexed by "
                      "[U_10 m/s, ratio]; reconstruct as ratio x Cheney",
            "wind_convention": "U_10. The applet slider is U_2; convert with "
                               "U_10 = U_2 / 0.723 before querying.",
            "provenance": "unitiedmodel2 local/diagnostics: "
                          "cheney_phase19_sweep + phase20_blend_optB. "
                          "ROS_Ts_m_min (level-set front tracker).",
            "limits": "Moisture sampled at 4% and 8% only; anything outside "
                      "is extrapolation. Wind coverage is thin for Nat8/Cut4/"
                      "Cut8 (two points each). The applet interpolates the "
                      "solver's ANSWERS -- it does not run the solver.",
        },
        "table": table,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(allcases)} cases scanned, {len(resolved)} genuinely resolved "
          f"and at/above the blend window (U_10 >= {u_min})")
    for k, pts in sorted(table.items()):
        rng = ", ".join(f"U{u:g}->{r:.3f}" for u, r in pts)
        print(f"  {k:6s} {len(pts)} pts: {rng}")
    dropped = [c for c in allcases if c["w_emp_in_source_run"] > 0.0]
    print(f"  dropped {len(dropped)} case(s) whose source run was blended or "
          f"empirical (would be circular)")


if __name__ == "__main__":
    main()
