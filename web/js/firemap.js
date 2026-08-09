/**
 * Fire-map renderer — paints the level-set state onto a canvas.
 *
 * Three regions, distinguished by state rather than by an arbitrary palette:
 *
 *   unburnt   the fuel bed, tinted by fuel type
 *   burning   cells within one flaming residence time of arrival, on a hot
 *             ramp from ignition to burnout
 *   burnt     everything older
 *
 * The width of the burning band is not decorative: it is
 * ROS x residence-time, so a fast dry run visibly carries a deeper flaming
 * zone than a slow damp one. That is the Byram flame-depth relation showing
 * up for free from the arrival-time field.
 *
 * Painted through a single ImageData per frame -- per-cell fillRect at
 * 38,400 cells would not hold frame rate.
 */

/** Hot ramp, ignition -> burnout. Sequential in burn age, one direction. */
const FIRE_RAMP = [
  [255, 247, 214],
  [255, 214, 102],
  [245, 148, 38],
  [214, 74, 30],
  [138, 38, 26],
];

const SKIN = {
  light: {
    unburnt: { natural: [186, 190, 138], cut: [203, 200, 158] },
    burnt: [58, 54, 50],
    burntEdge: [38, 35, 32],
  },
  dark: {
    unburnt: { natural: [92, 100, 66], cut: [104, 104, 78] },
    burnt: [30, 28, 26],
    burntEdge: [20, 19, 18],
  },
};

function rampAt(u) {
  const x = Math.min(0.999999, Math.max(0, u)) * (FIRE_RAMP.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = FIRE_RAMP[i];
  const b = FIRE_RAMP[Math.min(FIRE_RAMP.length - 1, i + 1)];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

export class FireMap {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this._img = null;
  }

  static isDark() {
    const stamped = document.documentElement.getAttribute('data-theme');
    if (stamped === 'dark') return true;
    if (stamped === 'light') return false;
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  /**
   * @param {import('./sim.js').FireSim} sim
   */
  draw(sim) {
    const { nx, ny } = sim;
    const { ctx } = this;

    // The backing store is one pixel per cell; CSS scales it to the layout
    // box. imageSmoothingEnabled=false keeps cell edges crisp rather than
    // blurring the front into a gradient.
    if (this.canvas.width !== nx || this.canvas.height !== ny) {
      this.canvas.width = nx;
      this.canvas.height = ny;
      this._img = ctx.createImageData(nx, ny);
    }
    ctx.imageSmoothingEnabled = false;

    const skin = FireMap.isDark() ? SKIN.dark : SKIN.light;
    const unburnt = skin.unburnt[sim.params.fuelKey] || skin.unburnt.natural;
    const tau = sim.residence_s;
    const { phi } = sim.ls;
    const arrival = sim.arrival;
    const now = sim.t;
    const data = this._img.data;

    for (let k = 0; k < phi.length; k++) {
      const p = k * 4;
      const a = arrival[k];
      let r, g, b;

      if (a < 0) {
        // Unburnt. A faint deterministic mottle so the bed reads as a
        // surface rather than a flat fill -- no RNG, so redraws are stable.
        const m = ((k * 2654435761) % 23) / 23 - 0.5;
        r = unburnt[0] + m * 10;
        g = unburnt[1] + m * 10;
        b = unburnt[2] + m * 8;
      } else {
        const age = now - a;
        if (age <= tau) {
          const c = rampAt(age / tau);
          r = c[0]; g = c[1]; b = c[2];
        } else {
          // Burnt. Cells just past burnout keep a little heat so the
          // transition does not read as a hard edge.
          const cool = Math.min(1, (age - tau) / (tau * 1.5));
          const e = skin.burnt;
          const hot = rampAt(1);
          r = hot[0] + (e[0] - hot[0]) * cool;
          g = hot[1] + (e[1] - hot[1]) * cool;
          b = hot[2] + (e[2] - hot[2]) * cool;
        }
      }

      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 255;
    }

    ctx.putImageData(this._img, 0, 0);
  }
}
