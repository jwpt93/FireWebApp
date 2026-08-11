/**
 * Main-thread UI for the live-solver page.
 *
 * Speed is reported as simulated seconds per wall-clock second, which is the
 * meaningful measure for a solver -- not a frame rate, which would say nothing
 * about how much physics happened.
 */
import { buildPalette } from './solverframe.js';
import { blendResolvedEmpirical, isExtrapolating, U2_PER_U10 } from './cheney.js';
import { CFG, CFG_NOTE, U_THRESH, U_BLEND_W } from './solverconfig.js';

const $ = (id) => document.getElementById(id);
const PALETTE = buildPalette(300, 1500);

let worker = null;
let meta = null;
let img = null;
let ctx = null;
let running = false;
let firstFrameAt = 0;

function setRunning(on) {
  running = on;
  $('run').textContent = on ? 'Stop' : 'Run solver';
  $('run').dataset.on = on ? '1' : '';
  for (const el of [$('wind'), $('moist')]) el.disabled = on;
}

function draw(data) {
  const { nx, kVis } = meta;
  if (!img || img.width !== nx || img.height !== kVis) {
    const cv = $('field');
    cv.width = nx; cv.height = kVis;
    ctx = cv.getContext('2d', { alpha: false });
    img = ctx.createImageData(nx, kVis);
  }
  const px = img.data;
  // Row 0 of the packed frame is the GROUND; canvas row 0 is the top of the
  // image, so k is flipped on the way in.
  for (let k = 0; k < kVis; k++) {
    const srcRow = (kVis - 1 - k) * nx;
    const dstRow = k * nx * 4;
    for (let i = 0; i < nx; i++) {
      const T = data[srcRow + i];
      let idx = Math.round(((T - 300) / 1200) * 255);
      if (idx < 0) idx = 0; else if (idx > 255) idx = 255;
      const p = dstRow + i * 4;
      px[p] = PALETTE[idx * 3];
      px[p + 1] = PALETTE[idx * 3 + 1];
      px[p + 2] = PALETTE[idx * 3 + 2];
      px[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function fmt(x, d = 2) { return Number.isFinite(x) ? x.toFixed(d) : '—'; }

function stop() {
  if (worker) { worker.postMessage({ type: 'stop' }); }
  setRunning(false);
}

function start() {
  const cfg = {
    ...CFG,
    windSpeedMs: Number($('wind').value),
    initialMoistureFrac: Number($('moist').value) / 100,
    maxWallTimeS: 30.0,
  };

  if (worker) worker.terminate();
  worker = new Worker(new URL('./solverworker.js', import.meta.url), { type: 'module' });
  meta = null; img = null; firstFrameAt = 0;
  $('err').hidden = true;
  $('done').hidden = true;
  setRunning(true);

  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'meta') {
      meta = m;
      $('meshinfo').textContent =
        `${m.nz}×${m.nx} = ${m.cells.toLocaleString()} cells · `
        + `showing lowest ${fmt(m.zVis, 2)} m of ${fmt(m.Lz, 1)} m`;
    } else if (m.type === 'frame') {
      if (!firstFrameAt) firstFrameAt = performance.now();
      draw(m.data);
      $('step').textContent = m.step.toLocaleString();
      $('simt').textContent = `${fmt(m.t, 3)} s`;
      $('msstep').textContent = `${fmt(m.msPerStep, 1)} ms`;
      $('xrt').textContent = `${fmt(m.xRealtime, 4)}×`;
      $('slower').textContent = 'simulated seconds per second';
      $('tg').textContent = `${fmt(m.TgMax, 0)} K`;
      $('ts').textContent = `${fmt(m.TsMax, 0)} K`;
      $('alive').textContent = m.nAlive != null ? m.nAlive.toLocaleString() : '—';
    } else if (m.type === 'done') {
      setRunning(false);
      $('done').hidden = false;
      $('donetext').textContent = m.stopped
        ? `Stopped after ${m.steps.toLocaleString()} steps `
          + `(${fmt(m.t, 2)} s simulated in ${fmt(m.wall, 1)} s wall, `
          + `${fmt(m.msPerStep, 1)} ms/step).`
        : `Finished: ${m.steps.toLocaleString()} steps, ${fmt(m.t, 2)} s simulated `
          + `in ${fmt(m.wall, 1)} s wall (${fmt(m.msPerStep, 1)} ms/step). `
          + `ROS ${fmt(m.rosMMin, 2)} m/min.`;
    } else if (m.type === 'error') {
      setRunning(false);
      $('err').hidden = false;
      $('errtext').textContent = m.message;
    }
  };
  worker.onerror = (e) => {
    setRunning(false);
    $('err').hidden = false;
    $('errtext').textContent = e.message || 'worker failed to start';
  };

  worker.postMessage({ type: 'run', cfg, frameEvery: 3, zTop: 2.7 });
}

function syncWindNote() {
  const u10 = Number($('wind').value);
  const w = blendResolvedEmpirical(u10, U_THRESH, U_BLEND_W);
  const u2 = u10 * U2_PER_U10;
  const extrap = isExtrapolating(u2, Number($('moist').value) / 100);
  const driver = w >= 1 ? 'Cheney fit only'
    : (w <= 0 ? 'resolved solver only'
              : `${(w * 100).toFixed(0)}% fit / ${((1 - w) * 100).toFixed(0)}% solver`);
  $('driver').textContent = driver;
  $('windnote').innerHTML =
    `Front driven by <strong>${driver}</strong> (U₂ = ${u2.toFixed(2)} m/s).`
    + (extrap
      ? ' <span style="color:var(--warn-ink)">Outside the range Cheney 1993'
        + ' fitted (U₂ 2–7 m/s, moisture 2–12%) — the fit is being'
        + ' extrapolated here.</span>'
      : '');
}

$('run').addEventListener('click', () => (running ? stop() : start()));
$('wind').addEventListener('input', () => {
  $('windval').textContent = `${Number($('wind').value).toFixed(1)} m/s`;
  syncWindNote();
});
$('moist').addEventListener('input', () => {
  $('moistval').textContent = `${Number($('moist').value).toFixed(0)}%`;
  syncWindNote();
});
$('cfgnote').textContent = CFG_NOTE;
syncWindNote();
