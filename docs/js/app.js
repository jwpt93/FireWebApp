/**
 * app.js — wiring for the Cheney grass-fire applet.
 *
 * Tier 1 runs live here (empirical.js); Tier 2/3 load precomputed JSON/mp4
 * produced by scripts/web_export/.  No backend, no build step.
 */

import {
  cheneyEq6Ros,
  marsdenSmedleyRos,
  marsdenSmedleyPSustain,
  A_CH_NATURAL,
  A_CH_CUT,
} from "./empirical.js";
import { makePlot } from "./plot.js";

const A_CH = { natural: A_CH_NATURAL, cut: A_CH_CUT };

// ── Shared UI state ─────────────────────────────────────────────────────────
const state = {
  model: "cheney_eq6",   // or "marsden_smedley"
  fuel: "natural",       // or "cut"
  U: 4.0,                // wind slider [m/s]  (U_10 for Cheney, U_1.7 for M-S)
  M_pct: 4.0,            // moisture slider [%]
  age_yr: 10.0,          // M-S stand age [yr]
};

// ── Data loading ────────────────────────────────────────────────────────────
async function fetchJson(url, fallback) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn(`fetch ${url} failed (${e.message}); serving the page over ` +
      `http(s) is required — try: python -m http.server -d docs`);
    return fallback;
  }
}

const [fig8, tier2Index, tier3Gallery] = await Promise.all([
  fetchJson("data/cheney1993_fig8.json", { natural: [], cut: [] }),
  fetchJson("data/tier2/index.json", { runs: [] }),
  fetchJson("assets/tier3/gallery.json", { items: [] }),
]);

const tier2RunCache = new Map();
async function loadTier2Run(id) {
  if (!tier2RunCache.has(id)) {
    tier2RunCache.set(id, fetchJson(`data/tier2/${id}.json`, null));
  }
  return tier2RunCache.get(id);
}

// ── Tier 1 ──────────────────────────────────────────────────────────────────
const tier1Plot = makePlot(document.getElementById("tier1-plot"), {
  xLabel: "wind speed (m/s)",
  yLabel: "rate of spread (m/s)",
  xMin: 0, xMax: 8, yMin: 0, yMax: 2.5,
});

const el = (id) => document.getElementById(id);
const fmt = (x, d) => x.toFixed(d);

function tier1Ros(U) {
  if (state.model === "cheney_eq6") {
    return cheneyEq6Ros(U, state.M_pct / 100, A_CH[state.fuel]);
  }
  return marsdenSmedleyRos(U, state.M_pct / 100, state.age_yr);
}

function redrawTier1() {
  // model curve
  const xs = [], ys = [];
  for (let U = 0; U <= 8.0 + 1e-9; U += 0.05) { xs.push(U); ys.push(tier1Ros(U)); }

  const series = [{ kind: "line", xs, ys, color: "#b8402a", width: 2 }];
  let yMax = Math.max(...ys);

  if (state.model === "cheney_eq6") {
    const pts = fig8[state.fuel] || [];
    series.push({
      kind: "points",
      xs: pts.map((p) => p[0]),
      ys: pts.map((p) => p[1]),
      color: "#333", radius: 3, hollow: true,
    });
    yMax = Math.max(yMax, ...pts.map((p) => p[1]));
  }
  tier1Plot.setLimits(0, 8, 0, Math.max(0.25, 1.15 * yMax));
  tier1Plot.setSeries(series);

  const ros = tier1Ros(state.U);
  tier1Plot.setMarker(state.U, ros);
  tier1Plot.redraw();

  // readout
  const units = `${fmt(ros, 3)} m/s = ${fmt(ros * 60, 1)} m/min`;
  let extra = "";
  if (state.model === "marsden_smedley") {
    const p = marsdenSmedleyPSustain(state.U, state.M_pct / 100);
    extra = `<dt>P(sustain)</dt><dd>${fmt(100 * p, 0)} %` +
      (p < 0.5 ? ` — <span class="warn">likely self-extinguishes</span>` : "") + `</dd>`;
  }
  el("tier1-readout").innerHTML =
    `<span class="big">${units}</span>` +
    `<dl><dt>Model</dt><dd>${state.model === "cheney_eq6" ? "Cheney Eq. 6" : "Marsden-Smedley 1995"}</dd>` +
    `<dt>Conditions</dt><dd>U = ${fmt(state.U, 1)} m/s, M = ${fmt(state.M_pct, 1)} %` +
    (state.model === "marsden_smedley" ? `, age = ${state.age_yr} yr` :
      `, ${state.fuel === "natural" ? "natural pasture" : "cut grass"}`) + `</dd>${extra}</dl>`;

  // caption + control visibility
  const isCheney = state.model === "cheney_eq6";
  el("age-control").hidden = isCheney;
  el("fuel-picker").style.display = isCheney ? "" : "none";
  el("wind-symbol").innerHTML = isCheney ? "U<sub>10</sub>" : "U<sub>1.7</sub>";
  el("tier1-caption").textContent = isCheney
    ? "Solid line: Cheney Eq. 6 for the current moisture and fuel. " +
      "Open circles: digitized Cheney 1993 Fig. 8 field measurements " +
      `(${state.fuel === "natural" ? "30 natural-pasture" : "51 cut-grass"} points). ` +
      "Crosshair: current slider setting."
    : "Marsden-Smedley 1995 buttongrass moorland regression (wind at 1.7 m " +
      "reference height; fuel build-up with stand age). P(sustain) is the " +
      "Marsden-Smedley 2001 probability that the fire keeps propagating.";
}

function bindSlider(id, key, fmtDigits) {
  el(id).addEventListener("input", (e) => {
    state[key] = parseFloat(e.target.value);
    el(`${id}-value`).textContent = fmt(state[key], fmtDigits);
    redrawTier1();
  });
}
bindSlider("wind", "U", 1);
bindSlider("moisture", "M_pct", 1);
bindSlider("age", "age_yr", 0);
el("wind-value").textContent = fmt(state.U, 1);
el("moisture-value").textContent = fmt(state.M_pct, 1);
el("age-value").textContent = fmt(state.age_yr, 0);

document.querySelectorAll('input[name="fuel"]').forEach((r) =>
  r.addEventListener("change", () => { state.fuel = r.value; redrawTier1(); }));
document.querySelectorAll('input[name="model"]').forEach((r) =>
  r.addEventListener("change", () => { state.model = r.value; redrawTier1(); }));

// ── Tier 2 ──────────────────────────────────────────────────────────────────
const TIER2_CASES = [
  { caseId: "cheney_nat4", button: "Natural pasture, M = 4%" },
  { caseId: "cheney_cut4", button: "Cut grass, M = 4%" },
  { caseId: "gr1_free_burn", button: "GR1 free burn (no wind)" },
];

const tier2 = { caseId: "cheney_nat4", wind: 4.0 };

const tier2Traces = makePlot(el("tier2-traces"), {
  xLabel: "time (s)", yLabel: "HRRPUA (kW/m²)",
  xMin: 0, xMax: 300, yMin: 0, yMax: 100,
});
const tier2RosPlot = makePlot(el("tier2-ros"), {
  xLabel: "U₁₀ (m/s)", yLabel: "ROS (m/s)",
  xMin: 0, xMax: 8, yMin: 0, yMax: 1.6,
});

function tier2RunsFor(caseId) {
  return tier2Index.runs
    .filter((r) => r.case_id.startsWith(caseId))
    .sort((a, b) => a.wind_speed_m_s - b.wind_speed_m_s);
}

function buildTier2Buttons() {
  const fuelSpan = el("tier2-fuel-buttons");
  fuelSpan.innerHTML = "";
  for (const c of TIER2_CASES) {
    const b = document.createElement("button");
    b.className = "chip" + (c.caseId === tier2.caseId ? " active" : "");
    b.textContent = c.button;
    b.addEventListener("click", () => {
      tier2.caseId = c.caseId;
      const winds = tier2RunsFor(c.caseId).map((r) => r.wind_speed_m_s);
      if (!winds.includes(tier2.wind)) tier2.wind = winds[winds.length - 1];
      buildTier2Buttons();
      redrawTier2();
    });
    fuelSpan.appendChild(b);
  }

  const windSpan = el("tier2-wind-buttons");
  windSpan.innerHTML = "";
  for (const r of tier2RunsFor(tier2.caseId)) {
    const w = r.wind_speed_m_s;
    const b = document.createElement("button");
    b.className = "chip" + (w === tier2.wind ? " active" : "");
    b.textContent = w === 0 ? "0 (free burn)" : String(w);
    b.addEventListener("click", () => {
      tier2.wind = w;
      buildTier2Buttons();
      redrawTier2();
    });
    windSpan.appendChild(b);
  }
}

async function redrawTier2() {
  const runId = `${tier2.caseId}__U${String(tier2.wind).replace(".", "p")}`;
  const run = await loadTier2Run(runId);
  if (!run) { el("tier2-readout").textContent = "run data unavailable"; return; }

  // ── left panel: per-cell HRRPUA(t) traces ──
  const colors = ["#b8402a", "#2a5db8", "#4d4d4d", "#999"];
  const traceSeries = run.cells.map((c, i) => ({
    kind: "line", xs: c.t_s, ys: c.hrrpua_kW_m2,
    color: colors[Math.min(i, colors.length - 1)], width: i === 0 ? 2 : 1.5,
  }));
  const peakAll = Math.max(...run.cells.flatMap((c) => c.hrrpua_kW_m2), 50);
  tier2Traces.setLimits(0, run.max_wall_time_s, 0, 1.1 * peakAll);
  tier2Traces.setSeries(traceSeries);
  tier2Traces.setMarker(null);
  tier2Traces.redraw();

  // ── right panel: ROS vs U sweep for this fuel, vs Tier-1 curve ──
  const sweep = tier2RunsFor(tier2.caseId);
  const t1xs = [], t1ys = [];
  for (let U = 0; U <= 8.0 + 1e-9; U += 0.05) {
    t1xs.push(U);
    t1ys.push(cheneyEq6Ros(U, run.moisture_frac, run.a_ch));
  }
  tier2RosPlot.setLimits(0, 8, 0, Math.max(0.2, 1.15 * Math.max(...t1ys)));
  tier2RosPlot.setSeries([
    { kind: "line", xs: t1xs, ys: t1ys, color: "#b8402a", width: 2 },
    {
      kind: "points",
      xs: sweep.map((r) => r.wind_speed_m_s),
      ys: sweep.map((r) => r.ros_m_s),
      color: "#2a5db8", radius: 4,
    },
  ]);
  tier2RosPlot.setMarker(tier2.wind, run.ros_m_s);
  tier2RosPlot.redraw();

  // ── readout ──
  const rosEmp = cheneyEq6Ros(run.wind_speed_m_s, run.moisture_frac, run.a_ch);
  const spread = run.n_cells_ignited >= 2;
  const delay = spread ? run.t_ignition_s[1] : null;
  el("tier2-readout").innerHTML =
    `<span class="big">${spread ? `${fmt(run.ros_m_s, 4)} m/s` : "no spread"}</span>` +
    (spread ? ` = ${fmt(run.ros_m_s * 60, 2)} m/min (front crossing)` : "") +
    `<dl>` +
    `<dt>Case</dt><dd>${run.label}; U = ${fmt(run.wind_speed_m_s, 1)} m/s</dd>` +
    (spread
      ? `<dt>Ignition delay</dt><dd>cell 1 ignited ${fmt(delay, 1)} s after the source</dd>` +
        `<dt>Cells ignited</dt><dd>${run.n_cells_ignited} (one fire-front depth — see caption)</dd>` +
        `<dt>Tier 1 same conditions</dt><dd>${fmt(rosEmp, 3)} m/s = ${fmt(rosEmp * 60, 1)} m/min ` +
        `(${fmt(rosEmp / run.ros_m_s, 0)}× faster)</dd>`
      : `<dt>Outcome</dt><dd><span class="warn">cascade did not propagate</span> — ` +
        `only the driven source cell burned</dd>`) +
    `</dl>`;

  el("tier2-caption").textContent =
    "Left: per-cell heat-release-rate traces (red = driven source cell, blue = first " +
    "downstream cell ignited by spread flux). Right: Tier-2 front-crossing ROS " +
    "(blue points) against the Tier-1 empirical curve for the same fuel and moisture " +
    "(red line). The resolved 1D cascade reproduces the wind trend but sits one to two " +
    "orders of magnitude below the field regression — the structural gap discussed in " +
    "Limitations below. Runs stop after one fire-front depth (fuel_depth / dx) by design.";
}

// ── Presets ─────────────────────────────────────────────────────────────────
const PRESETS = [
  { label: "Nat 4%, U = 0.5", fuel: "natural", U: 0.5, M: 4, caseId: "cheney_nat4" },
  { label: "Nat 4%, U = 4", fuel: "natural", U: 4.0, M: 4, caseId: "cheney_nat4" },
  { label: "Cut 4%, U = 0.5", fuel: "cut", U: 0.5, M: 4, caseId: "cheney_cut4" },
  { label: "Cut 4%, U = 4", fuel: "cut", U: 4.0, M: 4, caseId: "cheney_cut4" },
  { label: "GR1 free burn", fuel: "natural", U: 0.0, M: 5, caseId: "gr1_free_burn", wind: 0.0 },
];

function buildPresets() {
  const span = el("preset-buttons");
  for (const p of PRESETS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = p.label;
    b.addEventListener("click", () => {
      state.model = "cheney_eq6";
      state.fuel = p.fuel;
      state.U = p.U;
      state.M_pct = p.M;
      el("wind").value = p.U;
      el("moisture").value = p.M;
      el("wind-value").textContent = fmt(p.U, 1);
      el("moisture-value").textContent = fmt(p.M, 1);
      document.querySelector('input[name="model"][value="cheney_eq6"]').checked = true;
      document.querySelector(`input[name="fuel"][value="${p.fuel}"]`).checked = true;
      tier2.caseId = p.caseId;
      tier2.wind = p.wind ?? p.U;
      span.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      b.classList.add("active");
      buildTier2Buttons();
      redrawTier1();
      redrawTier2();
    });
    span.appendChild(b);
  }
}

// ── Tier 3 gallery ──────────────────────────────────────────────────────────
function buildGallery() {
  const div = el("tier3-gallery");
  if (!tier3Gallery.items || tier3Gallery.items.length === 0) {
    div.innerHTML =
      `<div class="empty">No 3D runs published yet — each case takes 10–60 minutes ` +
      `of compute and is added by <code>scripts/web_export/export_tier3_gallery.py</code>.</div>`;
    return;
  }
  div.innerHTML = "";
  for (const item of tier3Gallery.items) {
    const fig = document.createElement("figure");
    const media = item.still
      ? `<img src="assets/tier3/${item.still}" alt="${item.title}">`
      : `<video src="assets/tier3/${item.video}" controls loop muted playsinline></video>`;
    fig.innerHTML = `${media}<figcaption><strong>${item.title}</strong> — ${item.caption}</figcaption>`;
    div.appendChild(fig);
  }
}

// ── Boot ────────────────────────────────────────────────────────────────────
buildPresets();
buildTier2Buttons();
redrawTier1();
redrawTier2();
buildGallery();
