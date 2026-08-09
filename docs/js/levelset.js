/**
 * 2D level-set fire front — JavaScript port.
 *
 * Mirrors the discretisation of the parent project's 3D tracker in
 * src/model_outdoor/physics_3d/flame_front_3d.py, reduced to two dimensions.
 * Cross-checked against a NumPy reference by docs/test.html using
 * docs/data/golden.json.
 *
 * GOVERNING EQUATION
 * ------------------
 *     dphi/dt + v_n · |grad phi| = 0
 *
 * Sign convention (same as the parent project):
 *     phi < 0   burnt / burning
 *     phi ~ 0   the front
 *     phi > 0   unburnt, ahead of the front
 *
 * This is exact front geometry, not a fit.  It is what lets an arbitrary
 * ignition pattern evolve into a fire shape, lets separate fronts merge,
 * and lets the perimeter wrap a firebreak — none of which is prescribed.
 * The physics content all lives in v_n, supplied by the caller.
 *
 * DISCRETISATION
 * --------------
 * First-order Godunov upwind (Sethian 1999 §6.4).  For v_n > 0 the front
 * advances and phi decreases, so the stable one-sided choice is
 *
 *     |grad phi|^2 = max(D-, 0)^2 + min(D+, 0)^2       per axis
 *
 * with D- = (phi_i - phi_{i-1})/dx and D+ = (phi_{i+1} - phi_i)/dx.
 *
 * BOUNDARIES
 * ----------
 * One-sided differences are set to zero at every domain edge.  The 3D parent
 * uses periodic y (it models an infinite fire line); a bounded 2D applet
 * domain does not, so both axes here get the parent's x-treatment.  The
 * practical effect is a zero-gradient edge: the front stops cleanly at the
 * domain wall rather than wrapping around.
 *
 * DETERMINISM
 * -----------
 * Every kernel reads the old buffer and writes a separate new buffer, then
 * swaps — the double-buffer pattern CLAUDE.md Rule #17 requires of the parent
 * project's kernels.  There is no accumulation order that depends on
 * iteration order, so repeated runs on identical input are bit-exact.
 */

export class LevelSet2D {
  /**
   * @param {object} opts
   * @param {number} opts.nx  cells in x
   * @param {number} opts.ny  cells in y
   * @param {number} opts.dx  cell size in x [m]
   * @param {number} opts.dy  cell size in y [m]
   */
  constructor({ nx, ny, dx, dy }) {
    this.nx = nx;
    this.ny = ny;
    this.dx = dx;
    this.dy = dy;
    const n = nx * ny;
    this.phi = new Float64Array(n);
    this._phiNew = new Float64Array(n);
    this._phi0 = new Float64Array(n);
    this._grad = new Float64Array(n);
    // Start fully unburnt. A large positive value, not Infinity, so that
    // differences stay finite before the first reinitialisation.
    this.phi.fill(Math.max(nx * dx, ny * dy));
  }

  /** Linear index of cell (i, j). */
  idx(i, j) {
    return j * this.nx + i;
  }

  /**
   * Seed a circular ignition as an exact signed distance.
   * Cell centres are at ((i + 0.5)·dx, (j + 0.5)·dy).
   */
  seedCircle(cx_m, cy_m, r_m) {
    const { nx, ny, dx, dy, phi } = this;
    for (let j = 0; j < ny; j++) {
      const y = (j + 0.5) * dy - cy_m;
      for (let i = 0; i < nx; i++) {
        const x = (i + 0.5) * dx - cx_m;
        const d = Math.sqrt(x * x + y * y) - r_m;
        const k = j * nx + i;
        if (d < phi[k]) phi[k] = d;
      }
    }
  }

  /**
   * Seed a straight ignition line segment as an exact signed distance,
   * with `halfWidth_m` of initial burnt depth either side.
   */
  seedLine(x0_m, y0_m, x1_m, y1_m, halfWidth_m) {
    const { nx, ny, dx, dy, phi } = this;
    const ex = x1_m - x0_m;
    const ey = y1_m - y0_m;
    const len2 = ex * ex + ey * ey;
    for (let j = 0; j < ny; j++) {
      const y = (j + 0.5) * dy;
      for (let i = 0; i < nx; i++) {
        const x = (i + 0.5) * dx;
        let t = len2 > 0 ? ((x - x0_m) * ex + (y - y0_m) * ey) / len2 : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const px = x0_m + t * ex - x;
        const py = y0_m + t * ey - y;
        const d = Math.sqrt(px * px + py * py) - halfWidth_m;
        const k = j * nx + i;
        if (d < phi[k]) phi[k] = d;
      }
    }
  }

  /**
   * |grad phi| by Godunov upwind for v_n > 0 advection.
   * Mirrors godunov_grad_norm() in flame_front_3d.py.
   *
   * @param {Float64Array} src
   * @param {Float64Array} out
   */
  gradNorm(src, out) {
    const { nx, ny, dx, dy } = this;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const c = src[k];

        const dmx = i > 0 ? (c - src[k - 1]) * invDx : 0.0;
        const dpx = i < nx - 1 ? (src[k + 1] - c) * invDx : 0.0;
        const dmy = j > 0 ? (c - src[k - nx]) * invDy : 0.0;
        const dpy = j < ny - 1 ? (src[k + nx] - c) * invDy : 0.0;

        const gxp = dmx > 0 ? dmx : 0.0;
        const gxm = dpx < 0 ? dpx : 0.0;
        const gyp = dmy > 0 ? dmy : 0.0;
        const gym = dpy < 0 ? dpy : 0.0;

        out[k] = Math.sqrt(gxp * gxp + gxm * gxm + gyp * gyp + gym * gym);
      }
    }
  }

  /**
   * |grad phi| by sign-aware Godunov upwind, for the Sussman (1994)
   * reinitialisation step.  Mirrors reinit_godunov_grad() in
   * flame_front_3d.py.
   *
   * @param {Float64Array} src
   * @param {Float64Array} sign0  pre-reinit phi, supplies the sign
   * @param {Float64Array} out
   */
  reinitGradNorm(src, sign0, out) {
    const { nx, ny, dx, dy } = this;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const c = src[k];
        const s = sign0[k] > 0 ? 1.0 : sign0[k] < 0 ? -1.0 : 0.0;

        const dmx = i > 0 ? (c - src[k - 1]) * invDx : 0.0;
        const dpx = i < nx - 1 ? (src[k + 1] - c) * invDx : 0.0;
        const dmy = j > 0 ? (c - src[k - nx]) * invDy : 0.0;
        const dpy = j < ny - 1 ? (src[k + nx] - c) * invDy : 0.0;

        let gxp, gxm, gyp, gym;
        if (s > 0) {
          gxp = dmx > 0 ? dmx : 0.0;
          gxm = dpx < 0 ? dpx : 0.0;
          gyp = dmy > 0 ? dmy : 0.0;
          gym = dpy < 0 ? dpy : 0.0;
        } else {
          gxp = dmx < 0 ? dmx : 0.0;
          gxm = dpx > 0 ? dpx : 0.0;
          gyp = dmy < 0 ? dmy : 0.0;
          gym = dpy > 0 ? dpy : 0.0;
        }

        out[k] = Math.sqrt(gxp * gxp + gxm * gxm + gyp * gyp + gym * gym);
      }
    }
  }

  /**
   * Advance the front by dt with a per-cell normal speed.
   *
   *     phi_new = phi - dt · v_n · |grad phi|
   *
   * @param {number} dt              timestep [s]
   * @param {Float64Array} vnField   normal speed per cell [m/s], v_n >= 0
   */
  step(dt, vnField) {
    const { phi, _phiNew, _grad } = this;
    this.gradNorm(phi, _grad);
    for (let k = 0; k < phi.length; k++) {
      _phiNew[k] = phi[k] - dt * vnField[k] * _grad[k];
    }
    this.phi = _phiNew;
    this._phiNew = phi;
  }

  /**
   * Restore |grad phi| = 1 by iterating the Sussman (1994) reinitialisation
   *
   *     dphi/dtau = sign(phi_0) · (1 - |grad phi|)
   *
   * The front (the phi = 0 contour) is held fixed; only the distance
   * property of the surrounding field is repaired.
   *
   * @param {number} [substeps] pseudo-time iterations
   * @param {number} [cfl]      pseudo-timestep as a fraction of min(dx, dy)
   */
  reinitialize(substeps = 5, cfl = 0.5) {
    const { phi, _phi0, _grad, dx, dy } = this;
    _phi0.set(phi);
    const dtau = cfl * Math.min(dx, dy);
    for (let s = 0; s < substeps; s++) {
      const src = this.phi;
      const dst = this._phiNew;
      this.reinitGradNorm(src, _phi0, _grad);
      for (let k = 0; k < src.length; k++) {
        const sgn = _phi0[k] > 0 ? 1.0 : _phi0[k] < 0 ? -1.0 : 0.0;
        dst[k] = src[k] + dtau * sgn * (1.0 - _grad[k]);
      }
      this.phi = dst;
      this._phiNew = src;
    }
  }

  /**
   * Largest stable timestep for a given peak normal speed.
   * CFL for first-order upwind Hamilton-Jacobi in 2D.
   *
   * @param {number} vnMax    peak normal speed [m/s]
   * @param {number} [safety] CFL safety factor
   */
  maxDt(vnMax, safety = 0.4) {
    if (!(vnMax > 0)) return Infinity;
    return (safety * Math.min(this.dx, this.dy)) / vnMax;
  }

  /**
   * Fill a normal-speed field from a direction-dependent speed function.
   * The front normal is grad phi / |grad phi|, evaluated by central
   * differences and pointing from burnt toward unburnt.
   *
   * @param {(nx: number, ny: number) => number} speedFn
   * @param {Float64Array} out
   */
  fillNormalSpeed(speedFn, out) {
    const { nx, ny, dx, dy, phi } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        // Central difference in the interior, one-sided at the edges.
        // spanX/spanY count how many cells the difference actually spans.
        const im = i > 0 ? k - 1 : k;
        const ip = i < nx - 1 ? k + 1 : k;
        const jm = j > 0 ? k - nx : k;
        const jp = j < ny - 1 ? k + nx : k;
        const spanX = (i > 0 ? 1 : 0) + (i < nx - 1 ? 1 : 0);
        const spanY = (j > 0 ? 1 : 0) + (j < ny - 1 ? 1 : 0);
        const gx = spanX > 0 ? (phi[ip] - phi[im]) / (spanX * dx) : 0;
        const gy = spanY > 0 ? (phi[jp] - phi[jm]) / (spanY * dy) : 0;
        const mag = Math.sqrt(gx * gx + gy * gy);
        if (mag > 0) {
          out[k] = speedFn(gx / mag, gy / mag);
        } else {
          out[k] = speedFn(0, 0);
        }
      }
    }
  }

  /** Burnt area [m^2] — cells with phi < 0. */
  burntArea() {
    let n = 0;
    for (let k = 0; k < this.phi.length; k++) if (this.phi[k] < 0) n++;
    return n * this.dx * this.dy;
  }
}
