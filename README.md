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
ROS [m/s] = a_ch · U_2^0.987 · exp(−0.0707 · M_pct)
U_2 = U_10 · 0.723   (10 m to 2 m log-law extrapolation)
a_ch = 0.406 (natural pasture) or 0.343 (cut grass)
```

`U_2` is the paper's native variable — Table 2 defines it as "Wind speed at
2 m", and the printed x-axis of Fig 8 is "Wind speed at 2 m (ms⁻¹)".
`cheney_eq6_ros_m_per_s()` takes **U_10** and applies the 0.723 factor
internally, so do not pre-convert. Mixing the two conventions costs 27–38%.

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
│   ├── workers/                   ← per-case runners (spawn a single Cheney sim)
│   ├── web_export/                ← Tier-2 / Tier-3 precompute for the site
│   └── gen_golden_vectors.py      ← golden vectors + Fig 8 data for the site
├── docs/                          ← the published applet (GitHub Pages root)
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
    U_m_s=4.0,          # wind at 10 m; the 0.723 factor is applied inside
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
- `cheney1993_fig8_data_v2.json` — digitised Fig 8 (ROS vs **U_2** for
  natural / cut grass at various moisture bins). This is the target
  the Tier-3 sweep validates against, and the empirical Tier 1 is
  the closed-form fit to.

  **Known defect:** the file's `_meta.columns` says `U_10_m_s`, but the
  x-values are U_2 — Fig 8's printed axis is the 2 m wind. Confirmed
  statistically: read as U_2 the implied `a·exp(−0.0707·M)` lands between
  the M=4% and M=8% curves (natural median 0.2705 vs the [0.2306, 0.3060]
  band); read as U_10 it sits above both, implying M < 4% and contradicting
  the figure caption. The label is wrong, not the data — do not "fix" it by
  rescaling the values.

Plotting scripts in `scripts/plots/` overlay model results on the
Fig 8 envelope.

---

## Website (docs/)

A static site — plain HTML/JS, no build step, no backend — in [`docs/`](docs/),
deployed via GitHub Pages (**Settings → Pages → Deploy from a branch →
`main` / `/docs`**). The repo root carries a redirect stub so the site still
resolves if Pages is ever pointed at the root instead.

The applet runs the **2D level-set fire spread live in the browser**: a front
propagating across a fuel bed, driven by the Cheney regression, with sliders
for wind, moisture, wind direction, fuel type, ignition pattern and shape
model. Readouts give rate of spread, Byram fireline intensity, flame length,
fuel load, residence time, burnt area and perimeter, and a Fig 8 panel shows
where the current settings sit inside the real experimental scatter.

```
docs/
├── index.html          ← the applet
├── js/cheney.js        ← Cheney 1993 law + Byram derived quantities
├── js/levelset.js      ← 2D Godunov level-set front
├── js/fuels.js         ← Cheney Table 3 bed properties, with provenance
├── js/sim.js           ← front + arrival times + derived metrics
├── js/firemap.js       ← fuel / burning / burnt canvas renderer
├── js/fig8panel.js     ← Fig 8 overlay with a live marker
├── js/app.js           ← control wiring and animation loop
├── data/golden.json    ← generated; pins the JS to the Python reference
├── data/fig8.json      ← generated; Fig 8 scatter
├── data/tier2/         ← precomputed Tier-2 runs (15 cases)
├── assets/tier3/       ← Tier-3 animation gallery manifest
├── test.html           ← kernel cross-check, in-browser
├── test.mjs            ← same checks, CLI
├── simtest.mjs         ← simulation behaviour checks
└── smoketest.mjs       ← headless boot test of the whole page
```

Local preview (`fetch()` and ES modules both need http, not `file://`):

```bash
python3 -m http.server -d docs 8000   # → http://localhost:8000
```

Regenerating site data after model changes:

```bash
# Tier-1 golden vectors + Fig 8 scatter (writes docs/data/)
.venv/bin/python scripts/gen_golden_vectors.py

# Tier-2 precompute (seconds; writes docs/data/tier2/)
OMP_NUM_THREADS=8 .venv/bin/python scripts/web_export/export_tier2.py

# Tier-3 gallery (after running a Tier-3 case + animation script)
.venv/bin/python scripts/web_export/export_tier3_gallery.py
```

The Tier-2 export decks live in `scripts/web_export/decks/` — they reuse the
GR1 kinetics with Cheney bed parameters and a sustained 50 kW/m² source flux
(required for the cascade; see `run_1d_spread` docstring). They are export
artifacts, not validation cases.

### History

An earlier `docs/` applet (Tier-1 sliders + precomputed Tier-2 plots, by
Timothy LaPlaca) was a demonstration of the deployment pattern rather than a
final state, and has been replaced by the live simulation above. Kept from it,
because they are additive rather than duplicated:

- `scripts/web_export/export_tier2.py` and the 15 precomputed Tier-2 runs
- `scripts/web_export/export_tier3_gallery.py` and the gallery manifest
- the headless DOM-stub harness, now `docs/smoketest.mjs`

Removed as superseded: `docs/js/empirical.js` (duplicate Tier-1 port —
`docs/js/cheney.js` covers it and is pinned by golden vectors),
`scripts/web_export/check_js_port.mjs` + `tier1_reference.json` +
`export_tier1_reference.py` (superseded by `gen_golden_vectors.py`, which
pins the level set too), and the old `docs/index.html` / `js/app.js` /
`js/plot.js` / `css/style.css`.

**Note for the earlier applet's author:** it plotted the Fig 8 scatter at its
raw x-values against a U₁₀ model curve. Fig 8's x-axis is the **2 m** wind, so
the scatter sat ~38% too far left and appeared to lie above the Eq. 6 curve it
should straddle. Same root cause as the upstream bug fixed in `unitiedmodel2`
on 2026-08-09; see the Fig 8 note above.

---

## Tests

The full outdoor test suite is vendored in `tests/outdoor/`. Run:

```bash
OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/outdoor/ -q --tb=line
```

Expected: **354 passed, 41 failed** (~3 min).

The 41 failures are pre-existing test-infra drift inherited from the
parent `unitiedmodel2` repo — test helpers haven't kept up with
recent kernel signature changes, so tests calling
`step_bed_particles` and similar fail with `not enough arguments`.
These are NOT regressions from the extraction; running the same tests
in the parent gives the same 41 failures.

**Cheney-critical subset** (132 tests, ~6 s, all pass):

```bash
python -m pytest \
  tests/outdoor/test_empirical_ros_marsden_smedley.py \
  tests/outdoor/test_boundary_condition_registry.py \
  tests/outdoor/test_chemistry_family_override.py \
  tests/outdoor/test_edc_0d_adiabatic_validation.py \
  tests/outdoor/test_fft_poisson_3d.py \
  tests/outdoor/test_projection_3d_determinism.py \
  tests/outdoor/test_projection_3d_fft_pcg.py \
  tests/outdoor/test_dom_moisture.py \
  tests/outdoor/test_moisture_jump_bc.py \
  tests/outdoor/test_level_set_fsd.py \
  tests/outdoor/test_flame_front_3d_forcing.py \
  -q
```

These cover: Tier 1 (empirical Cheney/M-S), deck-loading + BC
registry, chemistry family dispatch, EDC closure 0D, Rule #17
bit-exact projection determinism, DOM moisture radiation, moisture-jump
BC (10 unit tests), level-set FSD closure, and 3D flame-front forcing.

---

## Applet — verification and future work

### Future work — exploit the steady state instead of re-solving it

Once wind, moisture and fuel stop changing, the applet is solving the same
problem every frame. With `v_n = V·(n·ŵ)` the level-set equation reduces to

```
φ_t + V ŵ·∇φ = 0
```

which is **pure translation** — the reason the planar-front test reproduces the
Cheney rate to 0.000%. So once the front is quasi-steady, the field could be
scrolled rather than stepped.

Two things fall out of that:

- **Cycle.** Detect the quasi-steady regime and translate the rendered field at
  the analytic rate instead of integrating. Nearly free per frame.
- **Cycle and refine.** Spend the reclaimed budget on resolution: precompute a
  converged front once at `dx = 0.25 m` (1.8% head-rate error) instead of
  running `dx = 1.0 m` live (5.9%), then translate it at the exact rate. That
  recovers both the accuracy and the frame rate currently traded against each
  other.

Caveat to check first: the backing floor makes flanks and rear expand at
`0.05·V`, so the fire still grows and the shape is not strictly invariant. It
should approach a steady form in a co-moving frame after the transient — that
needs measuring before the optimisation is safe.

### Verifying the port

The JS is a hand port, so it ships with golden vectors proving it still
agrees with the research code:

```bash
.venv/bin/python scripts/gen_golden_vectors.py   # regenerate golden.json + fig8.json
node docs/test.mjs        # kernels vs Python golden vectors  — 10/10
node docs/simtest.mjs     # simulation behaviour              — 16/16
node docs/smoketest.mjs   # headless boot of the whole page
python3 -m http.server -d docs 8000              # then open / or /test.html
```

`docs/test.mjs` and `docs/test.html` both call `runChecks()` in `docs/js/selftest.js`, so
they cannot drift.

`docs/simtest.mjs` checks the layer built on top of the kernels — that the front
actually advances at the rate the Cheney law specifies, that wind steers and
elongates it, that burnt area grows monotonically, and that the whole loop is
bit-exact across runs.

**A known, measured limitation.** Readouts and the Fig 8 marker are evaluated
analytically from the law and are exact. The *animation* is a level set on a
1 m grid, and first-order upwinding lags a **curved** front by an O(dx)
amount. Measured for a point ignition at U₂ = 4 m/s, M = 6%:

| dx [m] | head-rate error |
|---|---|
| 2.0 | −10.4% |
| 1.0 | −5.9% ← shipped |
| 0.5 | −3.4% |
| 0.25 | −1.8% |

Clean first order. A **planar** front is exact (0.000%), so the scheme is
right, just coarse. The grid stays at 1 m because cost scales as 1/dx³ —
38 ms/frame at dx = 0.5 against a 16.7 ms budget. `docs/simtest.mjs` pins the
convergence rather than a magic tolerance, so a real regression cannot hide
behind it. Nudging the speed to make the picture match the readout would stop
the level set from solving the equation it claims to; see the future-work note
above for the fix that gets both.

The level-set cases must be **bit-exact** — their speed fields use only
IEEE-exact operations, so any difference means the discretisation diverged.
The Cheney and Byram cases allow 1e-12 relative: `cheney_eq6_ros_m_per_s()`
multiplies by 60.0 and divides by 60.0 again, a no-op round-trip that costs
up to 1 ulp, and `pow`/`exp` may differ by an ulp across libm builds.

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
