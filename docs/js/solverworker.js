/**
 * Web Worker running the REAL ported solver.
 *
 * A worker is not optional. `runSpread3D` is a synchronous loop that runs for
 * minutes; on the main thread the page would be frozen the entire time and the
 * browser would offer to kill the tab. In here it can grind while the UI stays
 * responsive.
 *
 * Protocol:
 *   in   {type:'run', cfg, frameEvery, zTop}
 *   in   {type:'stop'}
 *   out  {type:'meta',  grid, kVis, nZBed, cells}
 *   out  {type:'frame', data(Float32Array, transferred), step, t, dt,
 *                       msPerStep, xRealtime, TgMax, TsMax, frontX, projIter}
 *   out  {type:'done',  rosMs, rosMMin, steps, t, wall, msPerStep}
 *   out  {type:'error', message}
 */
import { runSpread3D } from './physics/solver.js';
import { visibleRows, packFrame } from './solverframe.js';

let stopRequested = false;

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'stop') { stopRequested = true; return; }
  if (msg.type !== 'run') return;

  stopRequested = false;
  const { cfg, frameEvery = 4, zTop = 1.2 } = msg;

  let kVis = 0;
  let meta = null;
  const t0 = performance.now();
  let lastFrameWall = t0;

  try {
    const res = runSpread3D(cfg, (info) => {
      if (stopRequested) return false;
      const { grid, state } = info;

      if (meta === null) {
        kVis = visibleRows(grid.zFace, grid.nz, zTop);
        meta = {
          nx: grid.nx, ny: grid.ny, nz: grid.nz, nZBed: grid.nZBed,
          dx: grid.dx, Lx: grid.Lx, Lz: grid.Lz,
          zVis: grid.zFace[kVis],
          cells: grid.nz * grid.ny * grid.nx,
        };
        self.postMessage({ type: 'meta', ...meta, kVis });
      }

      if (info.step % frameEvery !== 0) return true;

      const wall = (performance.now() - t0) / 1000;
      const data = packFrame(state.T_g, state.T_s, {
        nx: grid.nx, ny: grid.ny, nz: grid.nz,
        kVis, nZBed: grid.nZBed, jSlice: Math.floor(grid.ny / 2),
      });
      self.postMessage({
        type: 'frame', data,
        step: info.step, t: info.t, dt: info.dt,
        msPerStep: (performance.now() - t0) / info.step,
        // The honest headline: simulated seconds per wall-clock second.
        // Below 1 means slower than real time.
        xRealtime: info.t / wall,
        TgMax: info.TgMax, TsMax: info.TsMax,
        frontX: info.frontX, projIter: info.projNIter,
        nAlive: info.nAlive,
      }, [data.buffer]);
      lastFrameWall = performance.now();
      return true;
    });

    const wall = (performance.now() - t0) / 1000;
    self.postMessage({
      type: 'done', rosMs: res.rosMs, rosMMin: res.rosMMin,
      steps: res.steps, t: res.t, wall,
      msPerStep: (wall * 1000) / Math.max(res.steps, 1),
      stopped: stopRequested,
    });
  } catch (e) {
    self.postMessage({ type: 'error', message: e && e.message ? e.message : String(e) });
  }
};
