#!/usr/bin/env python3
"""Export a Tier-1 reference table from the Python models.

The website's docs/js/empirical.js is a hand port of
src/model_outdoor/empirical_ros.py.  This script dumps a grid of
(input -> output) pairs computed by the *Python* implementation so that
scripts/web_export/check_js_port.mjs can assert the two agree to float
precision.  Run both after touching either implementation.

Usage (from repo root):
    .venv/bin/python scripts/web_export/export_tier1_reference.py
    node scripts/web_export/check_js_port.mjs
"""

from __future__ import annotations

import json
from pathlib import Path

from model_outdoor.empirical_ros import CheneyEq6, MarsdenSmedley

OUT = Path(__file__).resolve().parent / "tier1_reference.json"


def main() -> None:
    cheney = CheneyEq6()
    ms = MarsdenSmedley()

    cheney_cases = []
    for U in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        for M in (0.0, 0.02, 0.04, 0.08, 0.12, 0.20, 0.30):
            for a_ch in (0.406, 0.343):
                cheney_cases.append(
                    {
                        "U_m_s": U,
                        "moisture_frac": M,
                        "a_ch": a_ch,
                        "ros_m_s": cheney.ros(U_m_s=U, moisture_frac=M, a_ch=a_ch),
                    }
                )

    ms_cases = []
    for U in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
        for M in (0.0, 0.10, 0.20, 0.30, 0.45):
            for age in (1.0, 5.0, 10.0, 25.0, 50.0):
                ms_cases.append(
                    {
                        "U_m_s": U,
                        "moisture_frac": M,
                        "age_yr": age,
                        "ros_m_s": ms.ros(U_m_s=U, moisture_frac=M, age_yr=age),
                        "p_sustain": ms.p_sustain(U_m_s=U, moisture_frac=M),
                    }
                )

    OUT.write_text(
        json.dumps({"cheney_eq6": cheney_cases, "marsden_smedley": ms_cases}, indent=1)
    )
    print(f"wrote {OUT}  ({len(cheney_cases)} cheney, {len(ms_cases)} ms cases)")


if __name__ == "__main__":
    main()
