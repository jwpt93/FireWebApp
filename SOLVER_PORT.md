# Porting the 3D solver to the browser — state of play

Working notes for the effort to run `unitiedmodel2`'s actual fire solver in a
web page, rather than replaying precomputed answers. Written to be picked up
cold, months later, by someone who does not remember any of this.

Last updated 2026-08-10.

---

## 1. Why this is being attempted at all

The applet already runs a **level-set front** live at 60 fps — the fire shape,
merging, arrival times, flame geometry are all genuinely computed in the
browser. What it does *not* compute is the **spread rate**, which comes from a
lookup table of precomputed solver results (`docs/data/resolved.json`).

That is architecturally defensible: the rate is one scalar per condition, so
running CFD live to produce a number we already have would burn a hundred-fold
speedup for nothing. It is also, deliberately, not what was asked for. The
goal is the real solver running client-side, no server.

---

## 2. Is it fast enough? Yes, for "watchable"

Every number below is measured, not estimated. The chain, on a Cheney
Nat 4% / U₁₀ = 4 case:

| step | ×real time | note |
|---|---|---|
| 3D production (Ny=5) | 64.7 | the validated configuration |
| → true 2D (Ny=1) | 17.5 | **−3.2% ROS**, still PASSes the band |
| → window to Lx = 10 m | 9.7 | −3.2% ROS |
| → coarsen dx to 0.30 | 6.5 | +1.3% ROS |
| → species serial dispatch | 5.3 | bit-exact |
| → N_SUB 10 → 1 | **2.26** | −0.20% ROS |
| → in a browser (~2.2× JS) | **~5** | estimate, not measured |

**29× total.** At ~5× a 15 s fire runs in about 75 seconds with the front
visibly advancing — watchable, which was the bar. Real-time at 60 fps is not
reachable and was correctly abandoned.

### What each lever cost to find

- **True 2D**: `Ly = dy` gives `Ny = max(1, round(Ly/dy)) = 1`. The wrap
  pattern `jm = j-1 if j>0 else Ny-1` degrades correctly. Worker:
  `unitiedmodel2/scripts/diagnostics/_cheney_2d_worker.py`.
- **Windowing bottoms out at ~10 m.** At 7 m ROS collapses 24% — the window
  must contain the flaming zone plus radiative preheat. Cost does not scale
  with domain size: 4× smaller bought only 1.93×.
- **Thread count does nothing.** 1 thread vs 12 differed by 1%, ROS
  bit-identical. A micro-benchmark suggesting 4× was wrong — see §6.
- **`n_z_bed = 8` is a hard floor.** 6 gives −25%, 4 gives −61%. And `dt` is
  entirely bed-limited (0.0048 s at *every* dx from 0.10 to 0.30), so
  coarsening streamwise never raises the timestep.

### A validation finding, not a porting one

`dx = 0.10` is **not grid-converged**. The 0.10–0.30 range forms a tight
plateau (+1.3% across it) sitting ~29% ABOVE `dx = 0.05`. Probably by design
— Phase 17f caps `dx = clip(0.025U, 0.025, 0.10)` to hold cell residence time
against the EDC timescale, so halving dx changes the answer through the
chemistry closure. At `dx = 0.05` the Cheney ratio is 0.509, not 0.656.
Still inside [1/3, 3], but the 19/20 result is plateau-specific.

Plot: `unitiedmodel2/plots/cheney_2d_grid_convergence.png`.

---

## 3. What has to be ported (~6,000 lines)

Scope is set by what the 2D Cheney configuration **actually calls**, taken
from the run's own timing profile — not by what exists in `physics_3d/`.

| module | lines | status |
|---|---|---|
| `muscl_3d` | 171 | **done** — bit-exact |
| `species_3d` | 224 | **done** — bit-exact (21.9% of loop) |
| `momentum_3d` | 317 | **done** — bit-exact |
| `coupling_3d` | 192 | **done** — bit-exact |
| `drag_3d` | 144 | **done** — bit-exact |
| `solid_conduction_3d` | 127 | **done** — bit-exact |
| `chemistry_closures/edc` | 376 | **done** — within tolerance |
| `fft_poisson_3d` | 228 | **done** — 9.3e-13, own eigensolver |
| `turbulence_3d` (k-ε subset) | 560 | **done** — bit-exact / 1 ulp |
| ~~`pyrolysis_3d`~~ | ~~614~~ | **NOT NEEDED** — see below |
| `dom_3d` (DOM radiation) | 473 | **done** — ~1e-13 |
| `lagrangian_bed_3d` | 1,070 | **done** — init/conduction bit-exact, step ~1e-15 |
| `projection_3d` (fft_pcg path) | 450 of 948 | **done** — operator 1e-14, same residual |
| `flame_front_3d` (level set + v_n) | ~500 of 835 | **done** — bit-exact; own exact EDT |
| `soil_3d` | 153 | **done** — bit-exact |
| `combustion_3d` (O2 supply only) | 66 of 251 | **done** — bit-exact |
| `momentum_3d.apply_outflow_sponge` | 30 | **done** — bit-exact |
| `turbulence_3d.apply_wall_function` | 60 | **done** — bit-exact |
| `spread_3d` BC + advection helpers | 130 | **done** — bit-exact |
| `mesh` + `Grid3D.build` (both branches) | 300 | **done** — bit-exact |
| `spread_3d` main loop | ~800 | **done** — 17/17 integration, ROS 0.00% |

**~6,400 of ~6,400 done.** 106/106 kernel checks and 17/17 integration checks
pass. **The port is complete for the production Cheney configuration.**

Whole-solver agreement on the integration case (24x1x30, 217 steps, 0.6 s):

| | JS | Python | diff |
|---|---|---|---|
| **ROS** | 0.0998999 m/s | 0.0998999 m/s | **0.00%** |
| steps | 218 | 217 | 0.46% |
| T_g max | 1343.39 K | 1343.14 K | 0.02% |
| T_s max | 1540.95 K | 1541.09 K | 0.01% |
| Y_fuel max | 0.611979 | 0.611494 | 0.08% |
| rho mean | 0.812344 | 0.813250 | 0.11% |
| wall time | **2.9 s** | 3.1 s | — |

Worst per-step trajectory deviation over all 218 samples: 5.6% on T_g max,
4.6% on T_s max. That is the expected signature of EDC's discontinuous
extinction gates — the two runs diverge transiently in the detail and land in
the same place — not of a transcription error, which would show a monotonic
drift.

The wall-time result is the one I would not have predicted: **JS is level with
numba here**, not the 2.2x slower the earlier micro-benchmarks suggested and
nowhere near the 15-25x my first estimate claimed. See §6.

### The ~4,900 estimate was wrong — it is ~6,100

The original scope count enumerated the modules with *kernels I had already
identified from the profile*. Reading the main loop end to end surfaced six
more it calls that no profile entry named, because they are cheap per call:
the projection wrapper around the FFT solver, the level-set front and its v_n
driver, soil conduction, the O2-supply rate, the outflow sponge, and the k-e
wall function. Plus three helpers that live in `spread_3d.py` itself rather
than in `physics_3d/` — velocity BCs, gas-energy advection, front tracking —
so they never appeared in a module listing.

None of it changes the approach; it is about 1,200 lines more than budgeted.
Worth recording as a lesson: **profile weight is not a scope estimate.** A
kernel that is 0.3% of runtime is still 100% required, and the cheap ones are
exactly the ones a profile-driven survey misses.

### `pyrolysis_3d` drops out entirely (614 lines)

The four Eulerian pyrolysis kernels (`step_drying`, `step_pyrolysis_md2004`,
`step_char_oxidation`, `step_smoldering_oxidation`) sit in the **`else`**
branch of `lagrangian_bed_enable`. The Cheney run has the Lagrangian bed ON,
so none of them ever execute — the profile confirms it: zero `pyrolysis:*`
timing entries, only `lagrangian_bed`. That work happens per-particle instead.

Also out: `radiation_3d` (578) — the run uses `dom_3d` (473). And of
`turbulence_3d`'s 949 lines only ~560 are needed, since `step_smagorinsky_les`
and `apply_wall_function` are unused at `k_epsilon` + `wall_function=False`.

Total scope is therefore **~4,900**, not ~6,000.

### Two scope probes, both worth the hour

- **Projection: 948 → 228 lines.** `proj_iter = 1` on *every* logged step in
  2D with `divmax ~5e-7`, and `fft_poisson_3d.py` documents its separable FFT
  preconditioner as EXACT for near-constant coefficients. The BiCGSTAB
  wrapper never iterates here, so only the FFT solve is needed.
- **Lagrangian bed cannot be dropped.** `lagrangian_bed_enable` defaults to
  `False` and there is an Eulerian path, but it gives ROS 30.854 against
  38.130 — **−19%** — and is no faster. Phase 16's per-particle
  drying/pyrolysis/char-ox/smouldering is not something a bulk bed
  reproduces.

**Not needed** (1,960 lines): `finney_*`, `lagrangian_particles_3d`,
`level_set_fsd_3d`. Nor the unused closures (842 lines): `pasr`,
`ebu_bootstrap`, `edc_2step_methane`, `level_set_fsd` — the run uses
`combustion_closure="edc"`.

---

## 4. The verification contract

Two different standards, deliberately.

**Between codes — a tolerance, not identical bits.** Bit-exactness across
languages is unattainable for any kernel touching `pow()` or `exp()`: IEEE
does not require them to be correctly rounded, and V8's libm disagrees with
glibc's by ~2 ulp (`pow(1.743e-4, 0.25)` is `…469729` in V8 against `…469727`
in glibc). Chasing it is wasted effort. The actual max relative difference is
**reported per field** so drift stays visible, and bit-exactness is called out
where it happens to hold. Observed: omega ~7e-16, mass fractions ~2e-16.

**Within the JS — bit-exact, no exceptions.** Every kernel runs twice on
identical inputs and must match to the last bit. Rule #17 applied to the port.
This is what makes the cross-language tolerance meaningful: a reproducible
port a few ulp from the reference is a different thing from one that wanders.

### Consequence for the eventual whole-solver check

EDC's extinction gates are **discontinuous** — a 1-ulp `omega` can cross the
`< 0.5` wet-bulb threshold and move a cell's T_g by degrees (measured: 2 ulp
became 4.8 K). Divergences compound over a run. **The browser solver will be a
valid solution of the same model, not a bit-identical replay of the Python
trajectory.** So the integration test must be "agrees on ROS within a band",
never "reproduces the reference".

### The harness

```bash
.venv/bin/python scripts/gen_kernel_vectors.py   # calls the REAL Python
node docs/kerneltest.mjs                         # replays through the JS
```

Golden vectors are recorded input→output pairs from the reference. They verify
**faithfulness, not correctness** — if the Python has a bug, the vectors
encode it and the port must reproduce it. That is the right tool here: the
physics was already validated upstream at 19/20 against Cheney, so the only
new risk is transcription.

**Build vectors that fire the awkward branches.** A branch that never executes
is not tested. Current vectors deliberately include: wind that reverses (both
upwind paths), a sharp gradient (limiter engages), T_g spanning ambient (60
and 96 cells buoyant *downward*), an empty upper domain (128 and 240 no-fuel
early-returns), pre-loaded accumulators (catches assign-vs-accumulate),
pre-filled outputs (catches the reverse), and water content tuned so the
evaporation cap binds — at the first attempt **zero** cells dried and half
that branch was silently untested.

---

### A constant I assumed instead of checking

`S_STOICH` is **1.3**, not 1.35. I wrote 1.35 from memory into the O2-supply
port and it came back as a systematic 3.7% error on every written cell — not
one cell, all 96, which is the signature of a wrong constant rather than a
wrong index.

Cheap to catch here because the vectors exist. The lesson is narrower than "be
careful": a plausible-looking physical constant is exactly the kind of thing
that survives review, because 1.35 is not obviously wrong for a biomass
stoichiometric ratio. Read the constant out of the module; never type it from
memory.

---

## 5. Porting gotchas, in the order they cost time

1. **Match what numba EMITS, not what the Python says.** The reference is
   compiled. Verified case by case:
   `x**3 → x*x*x`, `x**0.5 → sqrt(x)`, `x**0.25 → pow(x, 0.25)`
   (NOT `sqrt(sqrt(x))`). Getting `T_s**4` wrong was a 1-ulp error; getting
   `**0.25` wrong put a 1-ulp error into `gamma*` that propagated into omega
   and on into T_g.

2. **Probe the values the kernel ACTUALLY sees.** Twice a small probe set
   where `pow` and `sqrt(sqrt)` happened to agree sent the port the wrong way.

3. **JS `%` keeps the dividend's sign; Python's does not.** `(j-2) % Ny` needs
   normalising. Only a 3D vector catches this — at Ny=1 every wrap is the cell
   itself.

4. **Python scoping leaks across loop iterations.** See §7.1.

5. **Indexing**: flat `Float64Array`, `idx = (k*Ny + j)*Nx + i`, matching NumPy
   C order so vectors transfer without reshaping. Inlet arrays are `(Nz*Ny)`,
   `idx = k*Ny + j`.

---

## 6. Performance lessons

- **Do not extrapolate solver performance from micro-benchmarks.** A single
  stencil kernel said 12 threads was 4.2× *slower* than 1 at 14k cells. In the
  real solver, thread count made no measurable difference — time is spread
  over ~20 kernels, many not `parallel=True`. The same mistake predicted JS as
  15–25× slower than numba when the measured figure was 2.2×.

- **The 2D loop was dominated by parallel-region overhead, not physics.**
  `species:transport` cost ~1 ms/call *regardless of grid size* (940 µs at
  14,000 cells, 1019 µs at 1,650). `prange` runs over `Nz≈50`, so a windowed
  2D grid hands each of 12 threads ~137 cells and the setup swamps the work.
  A size-aware serial dispatch gave **5× on that kernel, 1.27× overall,
  bit-exact**. 70% of the loop sat in that pattern; the remaining substepped
  kernels have not been given the same treatment yet.

---

## 7. Findings that belong to `unitiedmodel2`, not to this port

These are bugs and gaps discovered while porting. None are fixed upstream.

### 7.1 A flag leaks across cells in `edc.py`

`_h2o_quench_substantial` is assigned only inside the `else` branch of the
substep loop. A fuel-starved cell (`Yf <= 1e-9`) never resets it and inherits
whatever the **previous cell** left — numba scopes it to the function, not the
loop body.

Deterministic (`prange` splits over `k` while the leak travels along `i`, so
the predecessor is always in the same chunk; verified identical at 1/2/4/12
threads), so not a Rule #17 violation. But a cell's wet-bulb cooling being
triggered by its neighbour's moisture is not intended physics. Reproduced in
`docs/js/physics/edc.js` rather than silently fixed.

### 7.2 The `n_z_bed` kwarg is silently ignored

`spread_3d.py:1041` — `n_z_bed = int(_deck_first("n_z_bed", n_z_bed))`, and
`_deck_first` prefers the **deck** over the kwarg. `Outdoor_Grass_GR1__free_burn.txt`
pins `outdoor.n_z_bed = 8`, so passing `n_z_bed=4` to `run_3d_spread()` does
nothing. A first convergence sweep returned byte-identical ROS for 4/6/8/12
and would have "proved" the bed converged at 4 cells.

Force it through `ri.outdoor_overrides["n_z_bed"]`, and verify by checking
that `bed_cells=` and `dt=` in the log actually move. `dx` is *not*
deck-first, so the kwarg works there.

### 7.3 `N_SUB = 10` appears to be unnecessary

Hardcoded in `spread_3d.py`, justified by operator-splitting theory (Strang
1968), never measured. ROS is flat to **0.08% across N_SUB = 1 … 40**.
Validation against N_SUB=10 at production 2D mesh: **6 of 8 pairs pass** with
worst deviation 1.91%, and ignition fires everywhere (peak T_g 1354–1797 K
against a 1000 K floor — the Phase 14ah-4 failure signature is T_g stranded
at ~330 K and never appears). N_SUB=1 is 1.36–1.54× faster.

The historical failure was an *adaptive early-exit* bailing mid-loop during
pre-ignition; uniformly fewer substeps is a different operation.

**Do not read the two U=2 "failures" as evidence.** Cheney's reference at
U₁₀ = 2 is ~26 m/min; those runs give 1.573 and **−0.425** m/min. A negative
ROS is unphysical — neither propagates at either N_SUB, so the percentage is
noise. U=2 is the known hole: the single failure in the 19/20 Phase 19 sweep,
and the reason Option B moved the blend threshold 1.4 → 3.5. **Any future
N_SUB verdict must exclude non-propagating baselines** — suggested gate:
`ROS_Ts(N_SUB=10) > 0.2 × Cheney Eq.6` before a case may vote.

Remaining before it could become a default: 12 case-pairs (~12.6 h at
production fidelity, `Cut4_U1` alone is 2.6 h), a Rule #16 regression, and a
3D spot-check. Script is resumable:
`unitiedmodel2/scripts/run_2d_nsub_validation.py`.

### 7.5 Out-of-domain particles are placed, not rejected

`step_horizontal_solid_conduction_scatter`, `aggregate_particles_to_T_s_grid`
and `aggregate_particles_to_M_local_grid` all compute the column index as
`int(part_x[p] / dx)` and only afterwards test `i < 0`. Python's `int()`
truncates toward zero, so a particle at `x = -0.4*dx` gets `i = 0` and passes
the test — its mass and temperature are deposited into the west edge column
rather than being skipped.

`locate_cell`, used by `step_bed_particles` itself, guards `x < 0.0` first and
does not have the problem, so the main step is fine. Only the three scatter /
aggregate kernels are affected, and only for particles within one cell width
outside the west or south face.

Impact is probably nil in production — bed particles are stationary and are
initialised strictly inside the domain, so nothing ever reaches negative x.
It would bite the moment bed particles are given motion, or if a moisture-jump
zone ever places one off-grid. A one-line `if (part_x[p] < 0.0) continue`
in each of the three would close it.

### 7.6 `atm_growth` / `atm_max_dz` are inert in every production deck

`Grid3D.build` dispatches on whether any boundary-layer cell count is nonzero:

```python
use_new_kernel = (wall_bl_N > 0 or bed_top_inner_bl_N > 0 or bed_top_outer_bl_N > 0)
```

Only the new-kernel branch reads `atm_growth` and `atm_max_dz`. The legacy
branch builds its buffer from `dz_expansion` (default **1.0** — uniform) and
never looks at either.

**All 22 decks that set `atm_growth`/`atm_max_dz` also set `wall_bl_N = 0`**,
so in every one of them those two parameters do nothing.

For `Outdoor_Cheney_Cut4_U0p5__scout.txt`, which asks for an atmosphere
growing at 1.20 to a 1.0 m cap:

| | Nz | atmosphere aloft | cells |
|---|---|---|---|
| what the deck gets | **320** | uniform 25 mm to z = 8 m | 384,000 |
| what its own atm settings would give | **26** | grows to 1.0 m | 31,200 |

The bed is identical either way — 4 cells at 25 mm. The entire difference is
resolution in the air above the fire, where 25 mm cells at 7 m altitude are
not resolving anything. That is a **12.3x** factor on Nz and therefore on
every kernel in the loop.

I have NOT changed this, and it should not be changed without a regression run
— a mesh change is exactly the kind of thing CLAUDE.md Rule #16 exists for
(re-run the recent validation set before committing physics changes, because
shared-kernel edits routinely move cases beyond the one being fixed). It is
possible the fine uniform mesh is load-bearing for plume behaviour in a way
the deck author discovered and the parameters are vestigial. But it is equally
possible this is 12x of wasted compute across every outdoor sweep ever run,
and nobody has looked.

It is also directly relevant to this port: **384,000 cells is not going to run
in a browser.** The applet will need either the growing atmosphere or a lower
Lz, and that choice needs the regression run behind it.

Related in spirit to CLAUDE.md Rule #11 (no silent parameter effects — any
parameter that materially affects physics must be explicitly set in the deck).
This is the mirror image: parameters explicitly set in the deck that silently
affect nothing. The rule protects against relying on defaults; nothing
currently protects against a deck line that is quietly ignored.

### 7.7 The O2-supply rate and Damköhler cap are computed and discarded

Every chemistry sub-step — ten per outer step — the loop does this:

```python
omega_O2.fill(1.0e30)
combustion_3d.step_o2_supply_rate(...)          # 6-face upwind over the interior
_u_prime = np.sqrt(2.0 * k_turb / 3.0)          # full-field
_omega_max_T = state.rho * (S_L_GRASS + _u_prime) / grid.dx
chemistry_closures.run(combustion_closure, ..., omega_O2=omega_O2,
                       omega_max_T=_omega_max_T, tau_mix=tau_mix, ...)
```

The EDC closure's signature ends in `**_unused`, and none of `tau_mix`,
`omega_O2` or `omega_max_T` are among its named parameters — it derives its own
timescale from `k` and `eps`. All three are silently swallowed.

They are genuinely live for the FSD and PaSR closures. But **EDC is the
production closure**, so on every production run this is dead work. The
profiler on the integration case puts `combustion:o2_supply` at **2.1% of loop
time** on its own, plus two full-field temporaries per sub-step for
`_omega_max_T` that land in the unprofiled remainder.

The port omits it. That is a deliberate exception to the
reproduce-faithfully rule used everywhere else here, and the distinction is
worth being precise about: elsewhere the port reproduces upstream *behaviours*
that affect results (the EDC flag leak, the `int()` truncation, the Ny=1 empty
loop). This is arithmetic whose output provably never reaches a consumer.
Skipping it cannot change a number — and the integration test agreeing to
0.00% on ROS is the evidence.

Cheap fix upstream: hoist the three computations inside a
`if combustion_closure in ("level_set_fsd", "pasr"):` guard.

### 7.8 Four defects in the radiation/reaction coupling (low-wind failure)

Full write-up and measurements in the project memory note
`lowu_four_defects_radiation_reaction_coupling`. Summary, because these are
upstream physics issues surfaced by the port and shouldn't live only in a
memory file:

1. **EDC has no chemistry limb.** Its own docstring says so — *"Magnussen 1981
   EDC closure (no Arrhenius / cell-T gate, no bootstrap)"*. The rate is
   `gamma*·rho·Y/tau*` with `gamma* ~ eps^0.75` and `1/tau* ~ eps^0.5`, so
   `omega ~ u'^0.75` and vanishes with turbulence. The physical limit is not
   zero, it is the laminar flame at S_L ~ 0.4 m/s. Measured at U=0.5:
   **zero cells above 1000 K** — no flame exists at all.

2. ~~**The char-ox cap limits mass loss.**~~ **RETRACTED.** The cap reduces
   `m_cons_ch` and the heat release *together*, consistently, and
   `char_ox_flux_cap_W_m2` is literature-grounded (Williams 1985 char surface
   flux). Char oxidation genuinely is surface-flux limited. Acting on this
   would have broken correct physics.

3. **View-factor attenuation is one-sided.** Emission carries
   `f_geom = exp(-kappa*(h_bed - z_p))` (~12% for deep particles); absorption
   carries no depth term at all. Deep particles absorb fully and emit at 12%,
   so they cannot radiatively self-limit. Kirchhoff reciprocity wants the same
   factor on both sides.

4. **Emission is double-counted.** `dom_3d` returns
   `kappa*(G - 4*pi*B)`, which already nets off emission; the bed-particle
   kernel then applies its own Stefan-Boltzmann loss on top.

And the cap that started the investigation, `Q_RAD_MAX = 1e5 W/m^3`: measured
q_rad reaches 2.83e7, so it binds at up to **283x** on 8-16% of bed cells. It
is NOT the cause of the low-wind failure — raising it 100x leaves ROS negative
and degrades high wind. **And it is legitimate, not a band-aid**: at 1e7 W/m^3 a
surface particle takes ~250 W over ~3.5e-5 m^2 = 7 MW/m^2, which equilibrates
near 3400 K by sigma*T^4 alone. That flux is not physical for a bed cell.

None of this overturns the Phase 18 closure-class conclusion. Those seven
variants were all heat-TRANSPORT interventions and so was the radiation cap;
all eight failed because the missing piece is the reaction, not the transport.
Supplying the reaction (a laminar-propagation floor) does create a flame and
still does not create spread.

**Outcome.** The two Kirchhoff fixes (3 and 4) are implemented behind
`radiationFixes`, default off. Scored on the project's eq6 ratio 1/3..3:

| variant | U | ROS_Ts | ratio |
|---|---|---|---|
| reference | 4 | 27.04 | 0.516 pass |
| radiationFixes | 4 | **30.56** | **0.584 pass, closer to Cheney** |
| reference | 0.5 | -5.48 | fail (receding) |
| radiationFixes | 0.5 | 0.00 | fail (stationary, no longer receding) |

Worth keeping on their own merits. They do NOT unlock low-wind spread.

A second hypothesis is also **retracted**: I argued the 4601 K runaway came
from particles burning down while keeping a fixed share of absorbed radiation.
Within-cell mass spread is only 2.7-14.2% and drops to 0.2% with the fixes on,
and the fixes made the peak temperature WORSE (3304 -> 4581 K) because
weighting by `A_p*f_geom` concentrates the cell's radiation onto the surface
particles that can see it. Nine interventions have now failed across Phase 18
and this session; the Finney intermittency argument is the only explanation
consistent with all of them.

### 7.4 Validation fidelity ≠ applet fidelity

Worth remembering when a run seems inexplicably slow. Production mesh follows
`dx = clip(0.025U, 0.025, 0.10)`, so **U = 0.5 runs at dx = 0.025 m** — 12×
finer than the applet's 0.30 — over 25–30 s of sim instead of 5. For `Cut4_U1`
that is ~90× more work than the applet configuration.

**Open gap:** the grid-convergence study was run at **U=4 only**, where
production dx is already at its 0.10 cap. The applet's dx=0.30 at low wind
would be a 12× coarsening and is **untested**. It does not currently matter,
because below U₁₀ = 3.5 the applet uses the Cheney fit rather than the solver
— but the coarse mesh is only validated for U₁₀ ≳ 3.5.

---

## 8. Suggested order for the rest

1. ~~`fft_poisson_3d`~~ — **done.** Needed its own symmetric-tridiagonal
   eigensolver (implicit QL, ~90 lines) rather than shipping a precomputed
   eigenbasis, so the grid stays changeable. Ported for **Ny = 1 only**: the
   y-transform is an FFT over the periodic direction and is the identity for
   a single cell. It throws for Ny > 1 rather than silently returning
   something wrong. Verified on the SOLUTION, not on eigenvectors — those are
   defined only up to sign and ordering, so LAPACK and implicit QL
   legitimately disagree on them while agreeing on p to 9.3e-13.
2. **`turbulence_3d`** (949) — k-ε, feeds `k` and `eps` into EDC.
3. **`pyrolysis_3d`** (614) — feeds `Q_pyro` into coupling.
4. **`radiation_3d`** (578) — DOM; the only one with a non-local stencil, so
   expect it to be the awkward one.
5. ~~`lagrangian_bed_3d`~~ — **done.** Largest of them, and the only
   particle-based one, so the vector format grew a flat per-slot array
   convention alongside the (Nz,Ny,Nx) fields. Four kernels: the initialiser,
   `step_bed_particles`, the two grid aggregators, and horizontal conduction.
   The initialiser and the conduction scatter came out **bit-exact** — they
   are pure arithmetic with no transcendentals. The step lands at ~1e-15,
   which is `exp`/`pow` in the Arrhenius terms and nothing more.

   Three things needed care:
   - `T_s**4` in the radiative-loss diagnostic is float**INT, which numba
     lowers to multiplies. Written out as `Ts2*Ts2`. The Newton loop already
     used explicit multiplies and needed no change. Same gotcha as §5.
   - `Y_O2 ** N_O2_OP` and the ash-coverage `burn_frac ** exp` are
     float**float and *do* go through `pow` — `Math.pow` is right for those.
     The rule is per-expression, not per-file.
   - `int(x/dx)` truncates toward zero, so `int(-0.5)` is `0`, not `-1`.
     `Math.trunc`, never `Math.floor`. It matters: the scatter and aggregate
     kernels index with `int()` *before* testing `i < 0`, so a particle just
     outside the west edge lands in column 0 instead of being rejected. That
     is very likely an upstream latent bug — logged at §7.5, reproduced here
     rather than silently fixed, per the faithfulness rule in §4.
6. **Main loop** — operator-split ordering from `spread_3d.py`. Worth doing
   last and worth reading `spread_3d.py` end to end first; the ordering of the
   substep block is load-bearing.

Then the integration test: run the JS solver and the Python on the same case
and compare **ROS**, not fields, per §4.
