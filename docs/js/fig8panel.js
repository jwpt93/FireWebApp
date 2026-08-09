/**
 * Cheney 1993 Fig 8 overlay panel.
 *
 * Draws the experimental scatter for the selected fuel, the Cheney Eq. 6
 * reference curves at the caption's two moistures, and a live marker for the
 * applet's current slider state. The marker is the point of the panel: it
 * shows, continuously, whether the toy is sitting inside real data or has
 * wandered outside it.
 *
 * FORM.  One scatter series (context) + two reference curves (ordinal in
 * moisture, so one hue at two steps, separated by dash pattern and direct
 * labels) + one highlighted marker. Only one fuel is shown at a time, which
 * keeps this to two chromatic roles and well clear of the all-pairs series
 * cap that a two-fuel scatter would run into.
 *
 * COLOR.  Reference curves blue, live marker orange -- validated as a
 * categorical pair in both modes (worst-pair CVD deltaE 24.7 light /
 * 26.8 dark against a target of 8; contrast >= 3:1 on both surfaces).
 * The scatter is deliberately achromatic: it is context, not a series.
 *
 * Canvas rather than SVG because it redraws every animation frame alongside
 * the fire map. Identity is never carried by color alone -- both curves are
 * directly labelled and the marker is labelled with its own value.
 */
import { rosFromU2, isExtrapolating, VALID_RANGE } from './cheney.js';

const PAD = { l: 52, r: 14, t: 14, b: 34 };

/** Axis maxima. Fixed so the panel does not rescale under the slider. */
const U_MAX = 10.0;
const R_MAX = 2.6;

const THEME = {
  light: {
    surface: '#fcfcfb',
    ink: '#0b0b0b',
    inkMuted: '#52514e',
    grid: 'rgba(11,11,11,0.10)',
    scatter: 'rgba(82,81,78,0.42)',
    ref: '#2a78d6',
    live: '#eb6834',
  },
  dark: {
    surface: '#1a1a19',
    ink: '#ffffff',
    inkMuted: '#c3c2b7',
    grid: 'rgba(255,255,255,0.12)',
    scatter: 'rgba(195,194,183,0.38)',
    ref: '#3987e5',
    live: '#d95926',
  },
};

export class Fig8Panel {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} fig8  parsed docs/data/fig8.json
   */
  constructor(canvas, fig8) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.fig8 = fig8;
  }

  get theme() {
    const root = document.documentElement;
    const stamped = root.getAttribute('data-theme');
    if (stamped === 'dark') return THEME.dark;
    if (stamped === 'light') return THEME.light;
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? THEME.dark : THEME.light;
  }

  /** Size the backing store to the CSS box at device pixel ratio. */
  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.w = r.width;
    this.h = r.height;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  _x(u) {
    return PAD.l + (u / U_MAX) * (this.w - PAD.l - PAD.r);
  }

  _y(r) {
    return this.h - PAD.b - (r / R_MAX) * (this.h - PAD.t - PAD.b);
  }

  /**
   * @param {object} state  {fuelKey, a_ch, U2_m_s, moistureFrac, ros_m_s}
   */
  draw(state) {
    this.resize();
    const { ctx, w, h } = this;
    const t = this.theme;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = t.surface;
    ctx.fillRect(0, 0, w, h);

    // ── grid + axes (recessive) ───────────────────────────────────────────
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = t.inkMuted;
    ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let u = 0; u <= U_MAX; u += 2) {
      const x = this._x(u);
      ctx.beginPath();
      ctx.moveTo(x, PAD.t);
      ctx.lineTo(x, h - PAD.b);
      ctx.stroke();
      ctx.fillText(String(u), x, h - PAD.b + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let r = 0; r <= R_MAX; r += 0.5) {
      const y = this._y(r);
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(w - PAD.r, y);
      ctx.stroke();
      ctx.fillText(r.toFixed(1), PAD.l - 7, y);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('wind at 2 m  U₂  [m/s]', (PAD.l + w - PAD.r) / 2, h - 2);
    ctx.save();
    ctx.translate(11, (PAD.t + h - PAD.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('rate of spread  [m/s]', 0, 0);
    ctx.restore();

    // ── shade the fitted range; everything outside it is extrapolation ────
    const [uLo, uHi] = VALID_RANGE.U2_m_s;
    ctx.fillStyle = t.grid;
    ctx.fillRect(this._x(0), PAD.t, this._x(uLo) - this._x(0), h - PAD.t - PAD.b);
    ctx.fillRect(this._x(uHi), PAD.t, this._x(U_MAX) - this._x(uHi), h - PAD.t - PAD.b);

    // ── experimental scatter (context, achromatic) ────────────────────────
    const pts = this.fig8[state.fuelKey] || [];
    ctx.fillStyle = t.scatter;
    for (const [u, r] of pts) {
      const x = this._x(u);
      const y = this._y(r);
      if (x < PAD.l || x > w - PAD.r || y < PAD.t || y > h - PAD.b) continue;
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── Cheney Eq. 6 reference curves at the caption's two moistures ──────
    // Ordinal in moisture, so one hue: the drier curve solid, the wetter
    // dashed, each directly labelled.
    ctx.strokeStyle = t.ref;
    ctx.lineWidth = 2;
    for (const [mfPct, dash] of [[4, []], [8, [5, 4]]]) {
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const u = (i / 120) * U_MAX;
        const r = rosFromU2(u, mfPct / 100, state.a_ch);
        const x = this._x(u);
        const y = this._y(r);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const rEnd = rosFromU2(U_MAX, mfPct / 100, state.a_ch);
      const yEnd = Math.max(PAD.t + 6, this._y(rEnd));
      ctx.setLineDash([]);
      ctx.fillStyle = t.ref;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`M=${mfPct}%`, w - PAD.r - 2, yEnd - 3);
    }
    ctx.setLineDash([]);

    // ── live marker: where the sliders currently sit ──────────────────────
    const lx = this._x(state.U2_m_s);
    const ly = this._y(state.ros_m_s);
    const outside = isExtrapolating(state.U2_m_s, state.moistureFrac);

    ctx.strokeStyle = t.surface;          // 2px surface ring, per mark spec
    ctx.lineWidth = 2;
    ctx.fillStyle = t.live;
    ctx.beginPath();
    ctx.arc(lx, ly, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (outside) {                        // hollow centre = extrapolating
      ctx.fillStyle = t.surface;
      ctx.beginPath();
      ctx.arc(lx, ly, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Direct label on the marker — identity and value, never color alone.
    const label = `${(state.ros_m_s * 60).toFixed(1)} m/min`;
    ctx.font = '600 11px ui-monospace, Menlo, Consolas, monospace';
    const tw = ctx.measureText(label).width;
    let tx = lx + 11;
    let ta = 'left';
    if (tx + tw > w - PAD.r) { tx = lx - 11; ta = 'right'; }
    ctx.textAlign = ta;
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 3;
    ctx.strokeStyle = t.surface;
    ctx.strokeText(label, tx, ly);        // halo so it survives over scatter
    ctx.fillStyle = t.ink;
    ctx.fillText(label, tx, ly);
  }
}
