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
         isotropicSpeed } from './cheney.js';
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
    };
    this.reset();
  }

  get fuel() {
    return FUELS[this.params.fuelKey];
  }

  /** Head-fire rate of spread [m/s] — the Cheney regression. */
  get headRos_m_s() {
    return rosFromU2(this.params.U2_m_s, this.params.moistureFrac, this.fuel.a_ch);
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
    if (shape === 'isotropic') return isotropicSpeed(rosFromU2(U2_m_s, moistureFrac, a));
    return windProjectedSpeed(U2_m_s, moistureFrac, a, (windDirDeg * Math.PI) / 180);
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
