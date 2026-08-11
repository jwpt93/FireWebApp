/**
 * Main-thread UI for the live-solver page.
 *
 * The point of this page is to answer one question honestly: how slow is the
 * real solver in a browser? So the speed readout is the headline, not a
 * footnote, and it reports simulated-seconds per wall-second rather than
 * anything that could be mistaken for a frame rate.
 */
import { buildPalette } from './solverframe.js';

const $ = (id) => document.getElementById(id);
const PALETTE = buildPalette(300, 1500);

// Three meshes, chosen to make the cost of the mesh decision visible rather
// than hidden behind a single "quality" slider.
const PRESETS = {
  window2m: {
    label: '2 m window · dx 0.10 · growing atmosphere',
    note: 'Smallest useful domain. Fast, but the outlet sits almost on the '
        + 'flame — the solver normally refuses to measure ROS this close.',
    cfg: { Lx: 2.0, dx: 0.10, Lz: 8.0, bedXStart: 0.3, bedXEnd: 1.7,
           wallBlN: 1, wallBlFirstDz: 0.025, wallBlGrowth: 1.0 },
  },
  grow10: {
    label: '12 m · dx 0.10 · growing atmosphere',
    note: 'Full Cheney domain on the mesh the deck actually asks for. This is '
        + 'the realistic target.',
    cfg: { Lx: 12.0, dx: 0.10, Lz: 8.0, bedXStart: 1.0, bedXEnd: 9.0,
           wallBlN: 1, wallBlFirstDz: 0.025, wallBlGrowth: 1.0 },
  },
  grow05: {
    label: '12 m · dx 0.05 · growing atmosphere',
    note: 'Production horizontal resolution, corrected vertical mesh.',
    cfg: { Lx: 12.0, dx: 0.05, Lz: 8.0, bedXStart: 1.0, bedXEnd: 9.0,
           wallBlN: 1, wallBlFirstDz: 0.025, wallBlGrowth: 1.0 },
  },
  production: {
    label: '12 m · dx 0.05 · PRODUCTION mesh (320 uniform z-cells)',
    note: 'What the deck currently gets: atm_growth and atm_max_dz are ignored '
        + 'on the legacy mesh path, so the atmosphere is 25 mm cells all the '
        + 'way to 8 m. 76,800 cells. Included so the cost is visible — expect '
        + 'roughly 1.5 s per step.',
    cfg: { Lx: 12.0, dx: 0.05, Lz: 8.0, bedXStart: 1.0, bedXEnd: 9.0 },
  },
};

const BASE = {
  Ly: 0.10, dy: 0.10, nZBed: 4, hBed: 0.10, rhoB: 1.07,
  sigmaSav: 2000.0, canopyCd: 0.30, initialMoistureFrac: 0.04,
  atmGrowth: 1.20, atmMaxDz: 1.0,
  cflFactor: 0.40, minDtS: 1.0e-4,
  ignitionDurationS: 3.0, ignitionQMult: 3.0, ignitionWidthMult: 3.0,
  ignitionTPinEnable: false,
  solidPhaseIgnitionEnable: true, solidPhaseIgnitionTsK: 1000.0,
  lagrangianBedNPerCell: 4, lagrangianBedDryingMode: 'combined',
  lagrangianBedHConv: 250.0, lagrangianBedViewFactorGeometric: true,
  domSubcycleEvery: 5, levelSetPassive: true, wallFunction: false,
  // N_SUB = 1, not the upstream default of 10.
  //
  // Upstream notes that N_SUB "has never had a convergence study -- it is a
  // hardcoded constant justified by splitting theory, not by measurement".
  // The study was run (scripts/run_2d_nsub_validation.py, 2D production mesh):
  // six Cheney cases at N_SUB 10 vs 1, worst ROS deviation 1.9% against a 5%
  // band, all pass. Reproduced independently in this port at -0.01% on a 6 s
  // run. It is worth 1.78x here -- the chemistry sub-loop drops from 48% of
  // step time to 8%.
  //
  // The SOLVER's own default stays at 10, faithful to upstream. This is the
  // applet making an explicit, measured choice.
  nSub: 1,
  // Projection inner tolerance 1e-4, not the upstream 1e-6.
  //
  // The Krylov solve feeds an OUTER loop that iterates on the actual
  // divergence residual to projDivTol = 1e-3. An inner tolerance three orders
  // tighter than the thing consuming it is resolving detail that gets thrown
  // away. Measured, 12 m / dx 0.10, ROS identical to 4 decimals throughout:
  //
  //   rtol    ms/step   projection   proj iters   div residual
  //   1e-6     16.4       10.06         1.00        6.1e-6
  //   1e-5     13.9        7.58         1.00        5.9e-5
  //   1e-4     11.7        5.23         1.00        5.7e-4
  //   1e-3     34.2       21.68         1.71        1.0e-3   <- cliff
  //
  // There is a cliff, not a gradient. Past ~3e-4 the divergence residual
  // reaches projDivTol and the outer loop needs a SECOND projection, which
  // costs far more than the loosened inner tolerance saved. 1e-4 keeps about
  // 2x margin to it. If a stiffer case ever crosses anyway the failure is
  // graceful -- an extra outer iteration, so slower, not wrong.
  //
  // As with nSub, the SOLVER's own default stays at the upstream 1e-6.
  projectionCgRtol: 1.0e-4,
};

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
  for (const el of [$('preset'), $('wind'), $('moist')]) el.disabled = on;
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
  const preset = PRESETS[$('preset').value];
  const cfg = {
    ...BASE, ...preset.cfg,
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
      $('slower').textContent = m.xRealtime > 0
        ? `${fmt(1 / m.xRealtime, 0)}× slower than real time` : '—';
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

  worker.postMessage({ type: 'run', cfg, frameEvery: 3, zTop: 1.2 });
}

function syncPresetNote() {
  const p = PRESETS[$('preset').value];
  $('presetnote').textContent = p.note;
}

$('run').addEventListener('click', () => (running ? stop() : start()));
$('preset').addEventListener('change', syncPresetNote);
$('wind').addEventListener('input', () => {
  $('windval').textContent = `${Number($('wind').value).toFixed(1)} m/s`;
});
$('moist').addEventListener('input', () => {
  $('moistval').textContent = `${Number($('moist').value).toFixed(0)}%`;
});
syncPresetNote();
