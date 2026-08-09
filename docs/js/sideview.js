/**
 * Side elevation of the fire front.
 *
 * A vertical slice along the wind axis through the ignition point, showing
 * the fuel bed, the flaming zone, and the flame standing on it.
 *
 * DRAWN AT TRUE SCALE.  The window is 60 m wide and 10 m tall, so one metre
 * across is one metre up: flame height can be read honestly against the
 * ground and against the fuel bed. Vertical exaggeration would make the
 * flame look impressive and mean nothing. The window tracks the head, so the
 * fire stays in frame as it runs the length of the plan view above.
 *
 * WHAT IS ACTUALLY DRAWN
 * ----------------------
 *   bed depth      Cheney 1993 Table 3 (0.37 m natural, 0.15 m cut)
 *   flame depth    D = ROS · t_r, the along-wind width of the flaming zone,
 *                  read out of the arrival-time field rather than assumed
 *   flame length   Byram (1959) L_f = 0.0775 · I_B^0.46
 *   tilt           Byram (1959) tan θ = 0.88 · sqrt(U_mf² / (g L_f))
 *   flame height   H = L_f · cos θ
 *
 * So the flame lies over as the wind rises, and the flaming zone deepens as
 * the fire runs faster — both emerge from quantities computed elsewhere,
 * neither is drawn in by hand.
 */
import { DOMAIN } from './sim.js';

/** Window dimensions [m]. 6:1, matching the canvas aspect, so scale is 1:1. */
const WIN_W = 60;
const WIN_H = 10;

/** Where the head sits in the window: mostly burnt behind, some fuel ahead. */
const HEAD_FRAC = 0.72;

const THEME = {
  light: {
    sky: '#eef1f4', skyLow: '#e4e8ea',
    ground: '#6b6455', bedUnburnt: '#a9ad7e', bedBurnt: '#3a3632',
    ink: '#0b0b0b', inkMuted: '#52514e', grid: 'rgba(11,11,11,0.10)',
    rule: 'rgba(11,11,11,0.35)',
  },
  dark: {
    sky: '#191b1d', skyLow: '#121314',
    ground: '#3a352c', bedUnburnt: '#5c6442', bedBurnt: '#1e1c1a',
    ink: '#ffffff', inkMuted: '#c3c2b7', grid: 'rgba(255,255,255,0.12)',
    rule: 'rgba(255,255,255,0.40)',
  },
};

/** Hot ramp, flame base -> tip. Matches the plan view's fire colours. */
const FLAME = ['#fff7d6', '#ffd666', '#f59426', '#d64a1e', '#8a261a'];

export class SideView {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  static isDark() {
    const stamped = document.documentElement.getAttribute('data-theme');
    if (stamped === 'dark') return true;
    if (stamped === 'light') return false;
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.w = r.width;
    this.h = r.height;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** @param {import('./sim.js').FireSim} sim */
  draw(sim) {
    this.resize();
    const { ctx, w, h } = this;
    const t = SideView.isDark() ? THEME.dark : THEME.light;

    // Window in metres along the wind axis, tracking the head.
    const head = sim.headPosition_m();
    const sHead = Number.isFinite(head) ? head : 0;
    const s0 = sHead - WIN_W * HEAD_FRAC;
    const s1 = s0 + WIN_W;

    // 1:1 scale. Vertical scale is derived from the horizontal one so the
    // aspect is honest even if CSS gives the canvas a different box than 6:1;
    // any leftover height simply shows more sky.
    const mPerPx = WIN_W / w;
    const groundY = h - 18;
    const zToY = (z) => groundY - z / mPerPx;
    const sToX = (s) => ((s - s0) / WIN_W) * w;

    // ── sky ────────────────────────────────────────────────────────────────
    const sky = ctx.createLinearGradient(0, 0, 0, groundY);
    sky.addColorStop(0, t.skyLow);
    sky.addColorStop(1, t.sky);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    // ── height gridlines every 2 m ─────────────────────────────────────────
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = t.inkMuted;
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let z = 2; z <= WIN_H; z += 2) {
      const y = zToY(z);
      if (y < 2) break;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillText(`${z} m`, 4, y - 6);
    }

    // ── ground + fuel bed ──────────────────────────────────────────────────
    const bedTopY = zToY(sim.fuel.depth_m);
    ctx.fillStyle = t.ground;
    ctx.fillRect(0, groundY, w, h - groundY);

    const n = Math.max(2, Math.round(w));
    const slice = sim.sliceAlongWind(s0, s1, n);
    const colW = w / (n - 1) + 1;
    for (let k = 0; k < n; k++) {
      const st = slice[k].state;
      if (st === 'outside') continue;
      ctx.fillStyle = st === 'unburnt' ? t.bedUnburnt : t.bedBurnt;
      ctx.fillRect(sToX(slice[k].s) - 0.5, bedTopY, colW, groundY - bedTopY);
    }

    // ── flame sheets over every contiguous burning run ─────────────────────
    const L_f = sim.flameLength_m;
    const tilt = sim.flameTilt_rad;
    const H = sim.flameHeight_m;
    const reach = L_f * Math.sin(tilt);      // downwind lean [m]

    for (const [a, b] of runsOf(slice, 'burning')) {
      this._flame(sToX(slice[a].s), sToX(slice[b].s), bedTopY, H, reach, mPerPx);
    }

    // ── ground rule + labels ───────────────────────────────────────────────
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(w, groundY);
    ctx.stroke();

    ctx.fillStyle = t.inkMuted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${WIN_W} m of ground, drawn to scale`, w - 5, h - 4);

    // Flame-height dimension, only once there is a flame to measure.
    if (H > 0.05 && Number.isFinite(head)) {
      const xh = sToX(sHead) + 6;
      if (xh > 40 && xh < w - 60) {
        const yTop = zToY(H);
        ctx.strokeStyle = t.rule;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(xh, yTop);
        ctx.lineTo(xh, bedTopY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = t.ink;
        ctx.font = '600 11px ui-monospace, Menlo, Consolas, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${H.toFixed(1)} m`, xh + 5, yTop + 7);
      }
    }
  }

  /**
   * One flame sheet: a leaning quadrilateral from the bed top, with a
   * deterministic ragged tip so it reads as flame rather than a solid block.
   * Deterministic because a random tip would make redraws non-reproducible
   * for no visual gain.
   */
  _flame(x0, x1, baseY, H_m, reach_m, mPerPx) {
    const { ctx } = this;
    if (!(H_m > 0.02)) return;
    const wpx = Math.max(2, x1 - x0);
    const hpx = H_m / mPerPx;
    const rpx = reach_m / mPerPx;

    const g = ctx.createLinearGradient(0, baseY, 0, baseY - hpx);
    FLAME.forEach((c, i) => g.addColorStop(i / (FLAME.length - 1), c));
    ctx.fillStyle = g;
    ctx.globalAlpha = 0.92;

    const STEPS = Math.max(6, Math.round(wpx / 5));
    ctx.beginPath();
    ctx.moveTo(x0, baseY);
    for (let i = 0; i <= STEPS; i++) {
      const f = i / STEPS;
      // Deterministic mottle keyed to position — stable across redraws.
      const seed = Math.sin((x0 + f * wpx) * 12.9898) * 43758.5453;
      const jitter = 0.82 + 0.18 * (seed - Math.floor(seed));
      // Taper toward the run's edges so the sheet has shoulders, not corners.
      const taper = Math.sin(Math.PI * Math.min(1, Math.max(0, f))) ** 0.35;
      const hh = hpx * jitter * taper;
      ctx.lineTo(x0 + f * wpx + (rpx * hh) / hpx, baseY - hh);
    }
    ctx.lineTo(x1, baseY);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

/** Contiguous [startIndex, endIndex] runs of samples in a given state. */
function runsOf(slice, state) {
  const runs = [];
  let start = -1;
  for (let k = 0; k < slice.length; k++) {
    if (slice[k].state === state) {
      if (start < 0) start = k;
    } else if (start >= 0) {
      runs.push([start, k - 1]);
      start = -1;
    }
  }
  if (start >= 0) runs.push([start, slice.length - 1]);
  return runs;
}
