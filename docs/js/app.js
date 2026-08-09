/**
 * Mode A applet — UI wiring and the animation loop.
 *
 * Physics lives in sim.js / cheney.js / levelset.js; this file only moves
 * values between the DOM and the simulation, and drives the two canvases.
 */
import { FireSim, DOMAIN } from './sim.js';
import { FireMap } from './firemap.js';
import { Fig8Panel } from './fig8panel.js';
import { SideView } from './sideview.js';
import { FUELS, fuelLoad } from './fuels.js';
import { isExtrapolating, VALID_RANGE } from './cheney.js';

const $ = (id) => document.getElementById(id);

const sim = new FireSim();
const map = new FireMap($('map'));
const side = new SideView($('side'));
let panel = null;

/** Simulated seconds per real second. A real grass fire is slow to watch. */
let timeScale = 20;
let running = true;
let lastFrame = 0;

// ── controls ───────────────────────────────────────────────────────────────

const CONTROLS = [
  ['wind', 'U2_m_s', (v) => parseFloat(v)],
  ['moisture', 'moistureFrac', (v) => parseFloat(v) / 100],
  ['winddir', 'windDirDeg', (v) => parseFloat(v)],
];

function readControls() {
  for (const [id, key, parse] of CONTROLS) {
    sim.params[key] = parse($(id).value);
  }
  sim.params.fuelKey = document.querySelector('input[name=fuel]:checked').value;
  sim.params.shape = document.querySelector('input[name=shape]:checked').value;
  sim.params.ignition = document.querySelector('input[name=ignition]:checked').value;
  timeScale = parseFloat($('speed').value);
}

function syncReadouts() {
  const f = sim.fuel;
  const ros = sim.headRos_m_s;

  $('windval').textContent = `${sim.params.U2_m_s.toFixed(1)} m/s`;
  $('moistureval').textContent = `${(sim.params.moistureFrac * 100).toFixed(0)} %`;
  $('winddirval').textContent = `${sim.params.windDirDeg.toFixed(0)}°`;
  $('speedval').textContent = `${timeScale.toFixed(0)}×`;
  $('speedval2').textContent = `${timeScale.toFixed(0)}×`;

  $('out-ros').textContent = (ros * 60).toFixed(1);
  $('out-ros-ms').textContent = ros.toFixed(3);
  $('out-intensity').textContent = Math.round(sim.intensity_kW_m).toLocaleString();
  $('out-flame').textContent = sim.flameLength_m.toFixed(2);
  $('out-flameh').textContent = sim.flameHeight_m.toFixed(2);
  $('out-tilt').textContent = ((sim.flameTilt_rad * 180) / Math.PI).toFixed(0);
  $('out-flamed').textContent = sim.flameDepth_m.toFixed(1);
  $('out-load').textContent = fuelLoad(f).toFixed(2);
  $('out-residence').textContent = sim.residence_s.toFixed(0);
  $('out-area').textContent = sim.burntArea_ha.toFixed(2);
  $('out-perimeter').textContent = Math.round(sim.perimeter_m).toLocaleString();
  $('out-time').textContent = formatTime(sim.t);

  // Extrapolation is stated, not hidden. Cheney's regression was fitted over
  // a specific box; outside it the number is still computed but is no longer
  // supported by the experiment.
  const out = isExtrapolating(sim.params.U2_m_s, sim.params.moistureFrac);
  const banner = $('extrap');
  banner.hidden = !out;
  if (out) {
    const [uLo, uHi] = VALID_RANGE.U2_m_s;
    const [mLo, mHi] = VALID_RANGE.moisture_pct;
    const why = [];
    if (sim.params.U2_m_s < uLo || sim.params.U2_m_s > uHi) {
      why.push(`U₂ outside ${uLo}–${uHi} m/s`);
    }
    const mf = sim.params.moistureFrac * 100;
    if (mf < mLo || mf > mHi) why.push(`moisture outside ${mLo}–${mHi}%`);
    $('extrap-why').textContent = why.join(' and ');
  }
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

// ── loop ───────────────────────────────────────────────────────────────────

function frame(ts) {
  const dtReal = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0;
  lastFrame = ts;

  if (running && !sim.hasReachedEdge()) {
    sim.advance(dtReal * timeScale);
  } else if (running && sim.hasReachedEdge()) {
    running = false;
    $('playpause').textContent = 'Restart';
    $('edgenote').hidden = false;
  }

  map.draw(sim);
  side.draw(sim);
  panel?.draw({
    fuelKey: sim.params.fuelKey,
    a_ch: sim.fuel.a_ch,
    U2_m_s: sim.params.U2_m_s,
    moistureFrac: sim.params.moistureFrac,
    ros_m_s: sim.headRos_m_s,
  });

  $('out-area').textContent = sim.burntArea_ha.toFixed(2);
  $('out-perimeter').textContent = Math.round(sim.perimeter_m).toLocaleString();
  $('out-time').textContent = formatTime(sim.t);

  requestAnimationFrame(frame);
}

// ── wiring ─────────────────────────────────────────────────────────────────

function restart() {
  sim.reset();
  running = true;
  $('playpause').textContent = 'Pause';
  $('edgenote').hidden = true;
  syncReadouts();
}

for (const [id] of CONTROLS) {
  $(id).addEventListener('input', () => { readControls(); syncReadouts(); });
}
$('speed').addEventListener('input', () => { readControls(); syncReadouts(); });

// Fuel, shape model and ignition pattern all change the burn from t=0, so
// changing them restarts rather than leaving a scar from the old settings.
for (const name of ['fuel', 'shape', 'ignition']) {
  for (const el of document.querySelectorAll(`input[name=${name}]`)) {
    el.addEventListener('change', () => { readControls(); restart(); });
  }
}

$('playpause').addEventListener('click', () => {
  if (sim.hasReachedEdge()) { restart(); return; }
  running = !running;
  $('playpause').textContent = running ? 'Pause' : 'Play';
});
$('reset').addEventListener('click', restart);

$('domain').textContent = `${DOMAIN.Lx} × ${DOMAIN.Ly} m at ${DOMAIN.dx} m`;
for (const key of Object.keys(FUELS)) {
  const el = $(`fuel-${key}-label`);
  if (el) el.textContent = FUELS[key].label;
}

const fig8 = await (await fetch('./data/fig8.json')).json();
panel = new Fig8Panel($('fig8'), fig8);

readControls();
restart();
requestAnimationFrame(frame);
