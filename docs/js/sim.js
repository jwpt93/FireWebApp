/**
 * Mode A simulation — real front geometry, fitted spread rate.
 *
 * Owns a LevelSet2D plus a per-cell arrival-time field, and exposes the
 * derived fire-behaviour quantities the readouts need.
 *
 * WHAT IS REAL AND WHAT IS FITTED
 * -------------------------------
 * The front propagation is exact geometry: an arbitrary ignition evolves into
 * a fire shape, separate fronts merge, the perimeter wraps an obstacle -- none
 * of it prescribed. Fuel consumption and fireline intensity are conservation
 * statements. The local spread RATE, and the flame-length correlation, are
 * regressions. The applet says so on screen; see the README's Mode A / Mode C
 * split.
 */
import { LevelSet2D } from './levelset.js';
import { rosFromU2, firelineIntensity, flameLength, windProjectedSpeed,
         isotropicSpeed, flameTilt, flameHeight, U2_PER_U10 } from './cheney.js';
import { FUELS, fuelLoad, residenceTime_s } from './fuels.js';

/** Domain, in metres. Wide enough that a fast case still takes a while. */
export const DOMAIN = Object.freeze({ Lx: 240, Ly: 160, dx: 1.0 });

/** Level-set reinitialisation cadence, matching the parent project's default. */
const REINIT_EVERY = 10;

export class FireSim {
  constructor() {
    this.nx = Math.round(DOMAIN.Lx / DOMAIN.dx);
    this.ny = Math.round(DOMAIN.Ly / DOMAIN.dx);
    this.ls = new LevelSet2D({
      nx: this.nx, ny: this.ny, dx: DOMAIN.dx, dy: DOMAIN.dx,
    });
    this.arrival = new Float64Array(this.nx * this.ny);
    this.vn = new Float64Array(this.nx * this.ny);
    this.t = 0;
    this.steps = 0;
    this.params = {
      fuelKey: 'natural',
      U2_m_s: 4.0,
      moistureFrac: 0.06,
      windDirDeg: 0,
      shape: 'windProjected',
      ignition: 'point',
      model: 'hybrid',        // 'hybrid' (resolved + fit) or 'cheney' (fit only)
    };
    /** Set by the app once docs/data/resolved.json has loaded. */
    this.resolved = null;
    this.reset();
  }

  get fuel() {
    return FUELS[this.params.fuelKey];
  }

  /** Head-fire rate of spread [m/s] from the Cheney regression alone. */
  get cheneyRos_m_s() {
    return rosFromU2(this.params.U2_m_s, this.params.moistureFrac, this.fuel.a_ch);
  }

  /**
   * Head-fire rate of spread [m/s] actually propagated.
   *
   * In 'hybrid' mode this is the parent project's validated Phase 20
   * Option B blend: the Cheney regression below U_10 = 2.5, the resolved 3D
   * solver at or above 3.5, a linear ramp between. Falls back to the
   * regression if the resolved table has not loaded yet, so a slow fetch
   * degrades to Mode A rather than to a stalled fire.
   */
  get headRos_m_s() {
    const { model, U2_m_s, moistureFrac, fuelKey } = this.params;
    if (model === 'hybrid' && this.resolved) {
      const U10 = U2_m_s / U2_PER_U10;
      const r = this.resolved.hybridRos(fuelKey, moistureFrac, U10, this.fuel.a_ch);
      if (Number.isFinite(r)) return r;
    }
    return this.cheneyRos_m_s;
  }

  /** Which side of the blend the current wind sits on, or null in fit mode. */
  get regime() {
    if (this.params.model !== 'hybrid' || !this.resolved) return null;
    return this.resolved.regime(this.params.U2_m_s);
  }

  /** Byram fireline intensity [kW/m] at the head. */
  get intensity_kW_m() {
    return firelineIntensity(this.headRos_m_s, fuelLoad(this.fuel));
  }

  /** Byram flame length [m] at the head. */
  get flameLength_m() {
    return flameLength(this.intensity_kW_m);
  }

  /** Flaming residence time [s] — sets the width of the burning band. */
  get residence_s() {
    return residenceTime_s(this.fuel);
  }

  /** Burnt area [ha]. */
  get burntArea_ha() {
    return this.ls.burntArea() / 10000;
  }

  /** Perimeter length [m], from the count of cells straddling phi = 0. */
  get perimeter_m() {
    const { phi } = this.ls;
    const { nx, ny } = this;
    let n = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (phi[k] >= 0) continue;
        if (i > 0 && phi[k - 1] >= 0) { n++; continue; }
        if (i < nx - 1 && phi[k + 1] >= 0) { n++; continue; }
        if (j > 0 && phi[k - nx] >= 0) { n++; continue; }
        if (j < ny - 1 && phi[k + nx] >= 0) { n++; }
      }
    }
    return n * DOMAIN.dx;
  }

  /** The direction-dependent normal speed for the current parameters. */
  speedFn() {
    const { U2_m_s, moistureFrac, windDirDeg, shape } = this.params;
    const a = this.fuel.a_ch;
    const head = this.headRos_m_s;
    if (shape === 'isotropic') return isotropicSpeed(head);

    // windProjectedSpeed is built on the Cheney law, so in hybrid mode scale
    // its whole directional profile by the hybrid/Cheney ratio. That keeps
    // the head at exactly headRos_m_s while preserving the shape model --
    // rather than letting the map and the readouts disagree.
    const base = windProjectedSpeed(U2_m_s, moistureFrac, a,
                                    (windDirDeg * Math.PI) / 180);
    const cheney = this.cheneyRos_m_s;
    if (!(cheney > 0)) return base;
    const k = head / cheney;
    if (Math.abs(k - 1) < 1e-12) return base;
    return (nx, ny) => k * base(nx, ny);
  }

  /** Clear the burn and re-seed the ignition. */
  reset() {
    const { Lx, Ly, dx } = DOMAIN;
    this.ls = new LevelSet2D({
      nx: this.nx, ny: this.ny, dx, dy: dx,
    });
    this.arrival.fill(-1);
    this.t = 0;
    this.steps = 0;

    // Ignite upwind of centre so a wind-driven run has room to develop.
    const cx = Lx * 0.28;
    const cy = Ly * 0.5;
    if (this.params.ignition === 'line') {
      // Across-wind ignition line, the Cheney experimental configuration
      // (fires were lit from a line and run downwind).
      const dir = (this.params.windDirDeg * Math.PI) / 180;
      const half = 30;
      const nxv = -Math.sin(dir);
      const nyv = Math.cos(dir);
      this.ls.seedLine(cx - half * nxv, cy - half * nyv,
                       cx + half * nxv, cy + half * nyv, 1.5);
    } else {
      this.ls.seedCircle(cx, cy, 3.0);
    }
    this._recordArrivals();
  }

  /** Stamp the arrival time of every cell that has just burnt. */
  _recordArrivals() {
    const { phi } = this.ls;
    const a = this.arrival;
    for (let k = 0; k < phi.length; k++) {
      if (a[k] < 0 && phi[k] < 0) a[k] = this.t;
    }
  }

  /** Largest stable step for the current speed field. */
  maxDt() {
    const head = this.headRos_m_s;
    return this.ls.maxDt(Math.max(head * 1.05, 1e-6));
  }

  /**
   * Advance by up to `dtWanted` seconds of simulated time, sub-stepping to
   * respect CFL.
   *
   * @returns {number} seconds actually advanced
   */
  advance(dtWanted) {
    if (!(dtWanted > 0)) return 0;
    const fn = this.speedFn();
    const dtMax = this.maxDt();
    if (!isFinite(dtMax)) return 0;

    let remaining = dtWanted;
    let advanced = 0;
    let guard = 0;
    while (remaining > 1e-9 && guard++ < 64) {
      const dt = Math.min(dtMax, remaining);
      this.ls.fillNormalSpeed(fn, this.vn);
      this.ls.step(dt, this.vn);
      this.t += dt;
      this.steps++;
      if (this.steps % REINIT_EVERY === 0) this.ls.reinitialize();
      this._recordArrivals();
      remaining -= dt;
      advanced += dt;
    }
    return advanced;
  }

  /** Byram flame tilt from vertical [rad] at the head. */
  get flameTilt_rad() {
    // flameTilt takes the 10 m wind; the slider is U_2, so undo the 0.723.
    const U10 = this.params.U2_m_s / U2_PER_U10;
    return flameTilt(U10, this.flameLength_m);
  }

  /** Vertical reach of the tilted flame [m] — "flame height". */
  get flameHeight_m() {
    return flameHeight(this.flameLength_m, this.flameTilt_rad);
  }

  /**
   * Flame depth [m] — the along-wind width of the flaming zone.
   *
   *     D = ROS · t_r
   *
   * The band of fuel that is alight at any instant: everything the front
   * passed within one residence time. Falls straight out of the arrival-time
   * field, and is what the side view draws the flame sheet on top of.
   */
  get flameDepth_m() {
    return this.headRos_m_s * this.residence_s;
  }

  /** Ignition origin in metres — where the seed was placed. */
  get origin() {
    return { x: DOMAIN.Lx * 0.28, y: DOMAIN.Ly * 0.5 };
  }

  /** Nearest-cell index for a point in metres, or -1 if outside. */
  _cellAt(x_m, y_m) {
    const i = Math.floor(x_m / DOMAIN.dx);
    const j = Math.floor(y_m / DOMAIN.dx);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return -1;
    return j * this.nx + i;
  }

  /**
   * Sample the burn state along the wind axis through the ignition origin.
   *
   * Returns one entry per sample position `s` (metres along the wind
   * direction, signed, zero at the origin), each tagged with what the side
   * view needs to paint that column.
   *
   * @param {number} s0     start of the window [m along wind]
   * @param {number} s1     end of the window [m along wind]
   * @param {number} n      number of samples
   */
  sliceAlongWind(s0, s1, n) {
    const dir = (this.params.windDirDeg * Math.PI) / 180;
    const wx = Math.cos(dir);
    const wy = Math.sin(dir);
    const { x: ox, y: oy } = this.origin;
    const tau = this.residence_s;
    const out = new Array(n);

    for (let k = 0; k < n; k++) {
      const s = s0 + ((s1 - s0) * k) / (n - 1);
      const cell = this._cellAt(ox + s * wx, oy + s * wy);
      if (cell < 0) {
        out[k] = { s, state: 'outside', age: -1 };
        continue;
      }
      const a = this.arrival[cell];
      if (a < 0) {
        out[k] = { s, state: 'unburnt', age: -1 };
      } else {
        const age = this.t - a;
        out[k] = { s, state: age <= tau ? 'burning' : 'burnt', age };
      }
    }
    return out;
  }

  /**
   * Distance from the ignition origin to the head of the fire, along the
   * wind axis [m]. NaN before anything has burnt.
   */
  headPosition_m() {
    const dir = (this.params.windDirDeg * Math.PI) / 180;
    const wx = Math.cos(dir);
    const wy = Math.sin(dir);
    const { x: ox, y: oy } = this.origin;
    const step = DOMAIN.dx * 0.5;
    const maxS = Math.max(DOMAIN.Lx, DOMAIN.Ly);
    let last = NaN;
    for (let s = 0; s <= maxS; s += step) {
      const cell = this._cellAt(ox + s * wx, oy + s * wy);
      if (cell < 0) break;
      if (this.ls.phi[cell] < 0) last = s;
    }
    return last;
  }

  /** True once the front has reached any domain edge. */
  hasReachedEdge() {
    const { phi } = this.ls;
    const { nx, ny } = this;
    for (let i = 0; i < nx; i++) {
      if (phi[i] < 0 || phi[(ny - 1) * nx + i] < 0) return true;
    }
    for (let j = 0; j < ny; j++) {
      if (phi[j * nx] < 0 || phi[j * nx + nx - 1] < 0) return true;
    }
    return false;
  }
}
