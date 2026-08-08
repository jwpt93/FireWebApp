# Fuel-Only Reduced-Order Model

This package contains a minimal two-node fuel model (surface + interior) with
surface moisture and a simple Arrhenius pyrolysis closure. It is designed to be
extended later with flame feedback and spray forcing.

Quick start:
- `python -m model_fuel.runner`

Outputs:
- Prints final state and pyrolysis flux stats.
- Writes `outputs/model_fuel/model_fuel_demo.csv` under the repository root.

Notes:
- Physics closures are intentionally simple; search for `TODO` markers.
- Only `numpy` and `scipy` are required.
