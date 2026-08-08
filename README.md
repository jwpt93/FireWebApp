# cheney-web

Standalone Cheney-case grass-fire spread models, packaged for use as
the compute backend of a website applet.

Vendored from [`unitiedmodel2`](https://github.com/jw/unitiedmodel2) —
this directory contains only the code paths and validation data
relevant to reproducing the **Cheney et al. 1993** (Australian grass
fire) validation series, with three levels of physical detail.

---

## Three tiers of model

The same input conditions (wind speed, moisture, fuel bulk density,
fuel depth) can be pushed through three progressively richer
representations of the physics:

### Tier 1 — Empirical Cheney Eq.6 (milliseconds)

**Source:** [`src/model_outdoor/empirical_ros.py`](src/model_outdoor/empirical_ros.py)

Closed-form regression from Cheney, Gould & Catchpole 1993 (Fig 8):

```
ROS [m/s] = (a_ch / 60) · U_2^0.987 · exp(−0.0707 · M_pct)
U_2 = U_10 · 0.723   (10 m to 2 m log-law extrapolation)
a_ch = 0.406 (natural pasture) or 0.343 (cut grass)
```

- Zero compute cost per call → suitable for interactive sliders in a
  browser (embed via Pyodide, or reimplement in JS in <20 lines).
- No spatial resolution. Returns a scalar `ROS` for a given `(U, M)`.
- Also includes the Marsden-Smedley 1995 buttongrass regression as
  an alternate closed form for extinction-axis validation.

### Tier 2 — 1D flame-line spread (seconds)

**Source:** [`src/model_outdoor/spread.py`](src/model_outdoor/spread.py)

Sequential 1D cascade of fuel elements. Each element runs the
project's `fuel_element.py` pyrolysis ROM, and downwind elements
receive radiation from all upstream burning cells (Albini 1981/1985
line-source view factor with Beer-Lambert attenuation).

- ~seconds per simulation.
- Produces a per-cell HRRPUA(t) trace and a derived line ROS.
- No transverse (y) or vertical (z) resolution. Fire is treated as a
  cross-line-averaged 1D wave.

### Tier 3 — 3D reactive flow (minutes to hours)

**Source:** [`src/model_outdoor/spread_3d.py`](src/model_outdoor/spread_3d.py)
+ [`src/model_outdoor/physics_3d/`](src/model_outdoor/physics_3d/)

Full 3D k-ε RANS + EDC combustion + DOM radiation + Lagrangian bed
particles + level-set front. Same solver stack as the parent
`unitiedmodel2` research project.

- Order 10 minutes to 1 hour per case at OMP=8, dx=0.1 m.
- Produces 3D snapshots (`.npz`) with resolved velocity, temperature,
  species, and per-particle bed state.
- Rendered by the animation scripts in [`scripts/animations/`](scripts/animations/).

---

## What's included

```
cheney-web/
├── src/
│   ├── model/                     ← 1D fuel/flame ROM (Tier 2 dep) + I/O
│   └── model_outdoor/             ← wildfire spread physics
│       ├── empirical_ros.py       ← Tier 1
│       ├── spread.py              ← Tier 2 (1D line spread)
│       ├── spread_3d.py           ← Tier 3 (3D reactive flow)
│       └── physics_3d/            ← Tier 3 physics modules
├── data/
│   ├── cheney_experimental/       ← Cheney 1993 paper PDF + digitised Fig 8
│   └── validation_cases/          ← deck files for 12 Cheney conditions
├── scripts/
│   ├── animations/                ← 3 Cheney animation scripts (matplotlib)
│   ├── plots/                     ← Cheney sweep-vs-EXP plotters
│   └── workers/                   ← per-case runners (spawn a single Cheney sim)
├── docs/                          ← (empty — bring in from parent as needed)
└── references/                    ← (empty)
```

**Vendored as-is.** No pruning of Tier 3 physics module dependencies has
been done — the whole solver stack ships together so that any deck can
run. You can safely delete `physics_3d/` if you only need Tiers 1-2.

---

## Running each tier

Set up a Python venv (recommend Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then from the repo root:

### Tier 1 — instant

```python
from model_outdoor.empirical_ros import CheneyEq6

model = CheneyEq6()
ros_m_s = model.ros(
    U_m_s=4.0,          # wind at 2 m reference height
    moisture_frac=0.04, # 4% (mass fraction)
    a_ch=0.406,         # 0.406 natural pasture, 0.343 cut grass
)
print(f"ROS = {ros_m_s * 60:.2f} m/min")   # → 52.37 m/min
```

The `CheneyEq6` class subclasses the shared `FuelModel` protocol —
`MarsdenSmedley` (buttongrass) is a drop-in alternate with the same
call signature (plus an `age_yr` param).

### Tier 2 — 1D line spread

```bash
python -m model_outdoor.spread data/validation_cases/Outdoor_Grass_GR1__free_burn.txt
```

(Deck path → CLI arg. See `spread.py` docstring for parameters.)

### Tier 3 — 3D physics

```bash
OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python scripts/workers/_cheney_phase16_Nat4U4_worker.py
```

Then animate:

```bash
python scripts/animations/animate_cheney_phase16_4case.py
```

Outputs go to `local/diagnostics/` (created on first run).

---

## Validation data — Cheney 1993 Fig 8

`data/cheney_experimental/`:
- `cheney1993.pdf` — the paper.
- `cheney1993_fig8_data_v2.json` — digitised Fig 8 (ROS vs U_10 for
  natural / cut grass at various moisture bins). This is the target
  the Tier-3 sweep validates against, and the empirical Tier 1 is
  the closed-form fit to.

Plotting scripts in `scripts/plots/` overlay model results on the
Fig 8 envelope.

---

## Attribution

Reduced-order fire spread models developed in the `unitiedmodel2`
research project (jw / jwpt93@gmail.com). Cheney experimental data
digitised from:

> Cheney NP, Gould JS, Catchpole WR (1993). "The influence of fuel,
> weather and fire shape variables on fire-spread in grasslands."
> *International Journal of Wildland Fire* 3(1): 31–44.

---

## License

TBD — inherit from parent project or set independently.
