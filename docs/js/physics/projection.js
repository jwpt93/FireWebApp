/**
 * Variable-density pressure projection — JS port of
 * model_outdoor/physics_3d/projection_3d.py :: ProjectionSolver3D,
 * `fft_pcg` method only.
 *
 * Chorin (1967) fractional step at low Mach: solve
 *
 *     div( (alpha_g/rho) grad p ) = div(u*) / dt  -  div_target/dt
 *
 * then correct u -= dt * grad(p) / rho so the new velocity carries the
 * prescribed divergence rather than zero. The non-zero target is what lets
 * pyrolysis add gas mass: with strict incompressibility the projection
 * silently discards it and fuel piles up locally instead of expanding.
 *
 * WHAT IS PORTED, AND WHAT IS NOT
 * The Python class carries three solvers — PARDISO direct LU, AMG-CG, and
 * FFT-PCG. Only FFT-PCG is ported, because that is what the deck selects
 * (`projection_method='fft_pcg'`, the Phase 14ax default) and it is the one
 * with no library dependency: a constant-coefficient separable-FFT solve is a
 * *mathematical* preconditioner, built once, with no per-step hierarchy setup.
 * `method` is checked in the constructor and anything else throws rather than
 * silently falling back.
 *
 * NO CSR. The Python assembles a scipy sparse matrix and carries a cached
 * COO-to-CSR permutation to avoid re-sorting each step. None of that is
 * physics — it is scipy storage bookkeeping. The stencil is a fixed 7-point
 * star, so this port keeps seven coefficient arrays and applies them directly.
 * Same operator, no permutation to get wrong, and the matvec is a straight
 * pass over flat arrays.
 *
 * BiCGSTAB, not CG. The regularisation term eps*I makes the operator very
 * slightly indefinite — a single eigenvalue near -eps — and CG breaks down on
 * that. BiCGSTAB costs two matvecs per iteration instead of one and does not
 * care. At the density contrast a fire produces (~4x) the FFT preconditioner
 * is close enough to exact that it converges in a handful of iterations.
 *
 * HOW THIS IS VERIFIED. Not by comparing p against the reference elementwise —
 * two Krylov solvers stopped at the same relative residual legitimately return
 * different vectors within that tolerance, and scipy's BiCGSTAB and this one
 * do not take identical steps. The vectors instead check what the projection
 * is *for*: that the corrected velocity's divergence matches div_target, and
 * that the operator itself (the matvec) agrees with the reference matrix.
 * Same reasoning as the eigenvector-vs-solution split in poisson.js.
 *
 * Indexing: flat Float64Array, idx = (k*Ny + j)*Nx + i.
 */
import { SeparableLaplacian3D } from './poisson.js';

/** Diagonal regularisation, matching projection_3d.py. */
const EPS_REG = 1.0e-6;

/** Floor on rho before inverting — guards a cold-start divide, never binds. */
const RHO_FLOOR = 0.01;

export class ProjectionSolver3D {
  /**
   * @param {object} o
   * @param {number} o.nz, o.ny, o.nx
   * @param {number} o.dx, o.dy
   * @param {Float64Array} o.dzArr, o.dFaceAbove, o.dFaceBelow
   * @param {string} [o.yBc='periodic']
   * @param {string} [o.method='fft_pcg']
   * @param {number} [o.cgRtol=1e-6]
   */
  constructor({
    nz, ny, nx, dx, dy, dzArr, dFaceAbove, dFaceBelow,
    yBc = 'periodic', method = 'fft_pcg', cgRtol = 1.0e-6,
  }) {
    if (method !== 'fft_pcg') {
      throw new Error(
        `ProjectionSolver3D: only method='fft_pcg' is ported; got '${method}'. ` +
        `The PARDISO and AMG paths need LAPACK and pyamg respectively.`,
      );
    }
    if (yBc !== 'periodic') {
      throw new Error(`ProjectionSolver3D: only y_bc='periodic' is ported; got '${yBc}'.`);
    }
    Object.assign(this, { nz, ny, nx, dx, dy, dzArr, dFaceAbove, dFaceBelow,
                          yBc, method, cgRtol });
    this.n = nz * ny * nx;
    this.nxy = ny * nx;

    // Constant-coefficient preconditioner. Built once — it is a fixed
    // mathematical inverse, not something that adapts to the current rho.
    this.fft = new SeparableLaplacian3D({
      nz, ny, nx, dx, dy, dzArr, dFaceAbove, dFaceBelow,
    });

    // Seven stencil coefficient arrays, refilled by rebuildForRho.
    const a = () => new Float64Array(this.n);
    this.cDiag = a();
    this.cXp = a(); this.cXm = a();
    this.cYp = a(); this.cYm = a();
    this.cZp = a(); this.cZm = a();

    // alpha_g = 1 - alpha_s. null keeps the pure-gas operator
    // div((1/rho) grad p) with a plain div(u) divergence.
    this.alphaG = null;

    // Face-Dirichlet ghost values used by the divergence's mirror reflection.
    this.uInlet = new Float64Array(nz * ny);       // face at x = -dx/2
    this.wInletZmin = new Float64Array(ny * nx);   // face at k = -0.5

    this.pPrev = null;          // warm start
    this.lastIters = 0;

    // Krylov scratch, allocated once — this runs every projection iteration
    // of every step, and per-call allocation showed up in the profile.
    this._r = a(); this._rh = a(); this._p = a(); this._v = a();
    this._s = a(); this._tv = a(); this._y = a(); this._z = a();
    this._x = a(); this._div = a();
  }

  setAlphaG(alphaG) { this.alphaG = alphaG; }
  setInletBC(uInlet) { this.uInlet = uInlet; }
  setBottomInletBC(wInletZmin) { this.wInletZmin = wInletZmin; }

  /**
   * Refill the seven stencil coefficients for the current density.
   *
   * Discretises div((alpha_g/rho) grad p) with SEPARATELY face-averaged
   * alpha_g and 1/rho: face_coef = avg(alpha_g) * avg(1/rho). The split is
   * what makes the discrete identity div_vw(grad_correction) = A*p hold
   * exactly against the div(alpha_g*u) divergence operator below. With
   * alpha_g all ones this reduces to face-averaged 1/rho — the pre-14aw
   * baseline, unchanged.
   *
   * Boundaries: x has Neumann at the inlet and Dirichlet pressure at the
   * outlet; y is periodic; z has Neumann at the wall and Dirichlet pressure
   * at the top. Dirichlet faces take alpha_g from the boundary cell.
   */
  rebuildForRho(rho) {
    const { nz, ny, nx, nxy, dx, dy, dzArr, dFaceAbove, dFaceBelow } = this;
    const dx2 = dx * dx;
    const dy2 = dy * dy;
    const ag = this.alphaG;
    const { cDiag, cXp, cXm, cYp, cYm, cZp, cZm } = this;

    // inv_rho inline rather than materialised — one pass, one less array.
    const ir = (idx) => 1.0 / Math.max(rho[idx], RHO_FLOOR);
    const agAt = ag === null ? () => 1.0 : (idx) => ag[idx];

    for (let k = 0; k < nz; k++) {
      const invDzK = 1.0 / dzArr[k];
      for (let j = 0; j < ny; j++) {
        const jp = (j + 1) % ny;
        const jm = (j - 1 + ny) % ny;
        for (let i = 0; i < nx; i++) {
          const p = (k * ny + j) * nx + i;
          let diag = EPS_REG;
          let xp = 0, xm = 0, yp = 0, ym = 0, zp = 0, zm = 0;

          if (nx > 1) {
            if (i >= 1 && i <= nx - 2) {
              const irXp = 0.5 * (ir(p) + ir(p + 1));
              const irXm = 0.5 * (ir(p - 1) + ir(p));
              const agXp = 0.5 * (agAt(p) + agAt(p + 1));
              const agXm = 0.5 * (agAt(p - 1) + agAt(p));
              xp = (agXp * irXp) / dx2;
              xm = (agXm * irXm) / dx2;
              diag += -xp - xm;
            } else if (i === 0) {
              const irXp = 0.5 * (ir(p) + ir(p + 1));
              const agXp = 0.5 * (agAt(p) + agAt(p + 1));
              xp = (agXp * irXp) / dx2;
              diag += -xp;
            } else {
              // Outlet: Dirichlet p = 0 half a cell beyond, so the outward
              // face coefficient enters the diagonal at double weight and
              // has no neighbour to couple to.
              const irXm = 0.5 * (ir(p - 1) + ir(p));
              const agXm = 0.5 * (agAt(p - 1) + agAt(p));
              xm = (agXm * irXm) / dx2;
              const xpFace = (2.0 * agAt(p) * ir(p)) / dx2;
              diag += -xm - xpFace;
            }
          }

          if (ny > 1) {
            const pjp = k * nxy + jp * nx + i;
            const pjm = k * nxy + jm * nx + i;
            const irYp = 0.5 * (ir(p) + ir(pjp));
            const irYm = 0.5 * (ir(pjm) + ir(p));
            const agYp = 0.5 * (agAt(p) + agAt(pjp));
            const agYm = 0.5 * (agAt(pjm) + agAt(p));
            yp = (agYp * irYp) / dy2;
            ym = (agYm * irYm) / dy2;
            diag += -yp - ym;
          }

          if (nz > 1) {
            if (k >= 1 && k <= nz - 2) {
              const irZp = 0.5 * (ir(p) + ir(p + nxy));
              const irZm = 0.5 * (ir(p - nxy) + ir(p));
              const agZp = 0.5 * (agAt(p) + agAt(p + nxy));
              const agZm = 0.5 * (agAt(p - nxy) + agAt(p));
              zp = (agZp * irZp * invDzK) / dFaceAbove[k];
              zm = (agZm * irZm * invDzK) / dFaceBelow[k];
              diag += -zp - zm;
            } else if (k === 0) {
              const irZp = 0.5 * (ir(p) + ir(p + nxy));
              const agZp = 0.5 * (agAt(p) + agAt(p + nxy));
              zp = (agZp * irZp * invDzK) / dFaceAbove[0];
              diag += -zp;
            } else {
              const irZm = 0.5 * (ir(p - nxy) + ir(p));
              const agZm = 0.5 * (agAt(p - nxy) + agAt(p));
              zm = (agZm * irZm * invDzK) / dFaceBelow[nz - 1];
              const zpFace = 2.0 * agAt(p) * ir(p) * invDzK * invDzK;
              diag += -zm - zpFace;
            }
          }

          cDiag[p] = diag;
          cXp[p] = xp; cXm[p] = xm;
          cYp[p] = yp; cYm[p] = ym;
          cZp[p] = zp; cZm[p] = zm;
        }
      }
    }
  }

  /**
   * y = A*x for the 7-point operator. Terms are summed in a fixed order, so
   * repeated calls on the same input are bit-identical (Rule #17).
   */
  matvec(x, out) {
    const { nz, ny, nx, nxy } = this;
    const { cDiag, cXp, cXm, cYp, cYm, cZp, cZm } = this;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        const jp = (j + 1) % ny;
        const jm = (j - 1 + ny) % ny;
        for (let i = 0; i < nx; i++) {
          const p = (k * ny + j) * nx + i;
          let s = cDiag[p] * x[p];
          if (cXp[p] !== 0.0) s += cXp[p] * x[p + 1];
          if (cXm[p] !== 0.0) s += cXm[p] * x[p - 1];
          if (cYp[p] !== 0.0) s += cYp[p] * x[k * nxy + jp * nx + i];
          if (cYm[p] !== 0.0) s += cYm[p] * x[k * nxy + jm * nx + i];
          if (cZp[p] !== 0.0) s += cZp[p] * x[p + nxy];
          if (cZm[p] !== 0.0) s += cZm[p] * x[p - nxy];
          out[p] = s;
        }
      }
    }
  }

  /**
   * Preconditioned BiCGSTAB. Van der Vorst (1992) SIAM J. Sci. Stat. Comput.
   * 13:631, in the right-preconditioned form scipy uses.
   *
   * Breakdown (rho or omega underflowing to zero) returns what it has rather
   * than throwing: the caller runs several projection iterations per step and
   * a partial solve on one of them is recoverable, whereas an exception is
   * not. It is reported through `lastIters` and `lastResidual`.
   */
  _bicgstab(b, x0) {
    const n = this.n;
    const { _r: r, _rh: rh, _p: p, _v: v, _s: s, _tv: tv, _y: y, _z: z, _x: x } = this;
    x.set(x0);

    let bnorm = 0.0;
    for (let i = 0; i < n; i++) bnorm += b[i] * b[i];
    bnorm = Math.sqrt(bnorm);
    if (bnorm === 0.0) { x.fill(0.0); this.lastIters = 0; this.lastResidual = 0; return x; }
    const tol = this.cgRtol * bnorm;

    this.matvec(x, tv);
    for (let i = 0; i < n; i++) { r[i] = b[i] - tv[i]; rh[i] = r[i]; p[i] = r[i]; }

    let rho1 = 0.0;
    for (let i = 0; i < n; i++) rho1 += rh[i] * r[i];

    let iters = 0;
    let resid = Infinity;
    for (let it = 0; it < 200; it++) {
      iters = it + 1;
      // y = M^-1 p
      y.set(this.fft.solve(p));
      this.matvec(y, v);
      let rhv = 0.0;
      for (let i = 0; i < n; i++) rhv += rh[i] * v[i];
      if (rhv === 0.0) break;
      const alpha = rho1 / rhv;

      for (let i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
      let snorm = 0.0;
      for (let i = 0; i < n; i++) snorm += s[i] * s[i];
      snorm = Math.sqrt(snorm);
      if (snorm < tol) {
        for (let i = 0; i < n; i++) x[i] += alpha * y[i];
        resid = snorm;
        break;
      }

      z.set(this.fft.solve(s));
      this.matvec(z, tv);
      let tt = 0.0, ts = 0.0;
      for (let i = 0; i < n; i++) { tt += tv[i] * tv[i]; ts += tv[i] * s[i]; }
      if (tt === 0.0) break;
      const omega = ts / tt;

      for (let i = 0; i < n; i++) {
        x[i] += alpha * y[i] + omega * z[i];
        r[i] = s[i] - omega * tv[i];
      }

      let rnorm = 0.0;
      for (let i = 0; i < n; i++) rnorm += r[i] * r[i];
      resid = Math.sqrt(rnorm);
      if (resid < tol) break;
      if (omega === 0.0) break;

      let rho2 = 0.0;
      for (let i = 0; i < n; i++) rho2 += rh[i] * r[i];
      if (rho2 === 0.0) break;
      const beta = (rho2 / rho1) * (alpha / omega);
      for (let i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);
      rho1 = rho2;
    }
    this.lastIters = iters;
    this.lastResidual = resid / bnorm;
    return x;
  }

  /**
   * Finite-volume divergence, consistent with the matrix discretisation.
   *
   * At Dirichlet velocity boundaries — the inlet at i=0 and the no-slip wall
   * at k=0 — this uses MIRROR ghost reflection rather than a one-sided
   * difference, so div[0] = (u[0] - u_inlet)/dx and div_z[0] = w[0]/dz[0].
   * Those capture the deviation from the prescribed boundary value, which is
   * exactly what the pressure gradient at face 0.5 then corrects.
   *
   * Normalisation is by dz_arr[k], the cell height, NOT d_below[k], the
   * cell-centre distance. That is what makes it the finite-volume operator
   * the matrix was built for.
   */
  divergence(u, v, w, out = null) {
    const { nz, ny, nx, nxy, dx, dy, dzArr } = this;
    const div = out ?? this._div;
    div.fill(0.0);
    const ag = this.alphaG;

    if (ag === null) {
      for (let k = 0; k < nz; k++) {
        const invDz = 1.0 / dzArr[k];
        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx; i++) {
            const p = (k * ny + j) * nx + i;
            const uL = i === 0 ? this.uInlet[k * ny + j] : u[p - 1];
            let d = (u[p] - uL) / dx;
            if (ny > 1) {
              const jm = (j - 1 + ny) % ny;
              d += (v[p] - v[k * nxy + jm * nx + i]) / dy;
            }
            const wL = k === 0 ? this.wInletZmin[j * nx + i] : w[p - nxy];
            d += (w[p] - wL) * invDz;
            div[p] = d;
          }
        }
      }
      return div;
    }

    // Volume-weighted div(alpha_g * u). Ghost faces at the open inlet and
    // top vent take alpha_g = 1 (pure gas — no fuel outside the bed); the
    // outlet and top faces extrapolate alpha_g from the boundary cell.
    for (let k = 0; k < nz; k++) {
      const invDz = 1.0 / dzArr[k];
      for (let j = 0; j < ny; j++) {
        const jm = (j - 1 + ny) % ny;
        for (let i = 0; i < nx; i++) {
          const p = (k * ny + j) * nx + i;
          const agR = i === nx - 1 ? ag[p] : 0.5 * (ag[p] + ag[p + 1]);
          let d;
          if (i === 0) {
            d = (agR * u[p] - this.uInlet[k * ny + j]) / dx;
          } else {
            const agL = 0.5 * (ag[p - 1] + ag[p]);
            d = (agR * u[p] - agL * u[p - 1]) / dx;
          }
          if (ny > 1) {
            const pjp = k * nxy + ((j + 1) % ny) * nx + i;
            const pjm = k * nxy + jm * nx + i;
            const agYf = 0.5 * (ag[p] + ag[pjp]);
            const agYfL = 0.5 * (ag[pjm] + ag[p]);
            d += (agYf * v[p] - agYfL * v[pjm]) / dy;
          }
          const agZR = k === nz - 1 ? ag[p] : 0.5 * (ag[p] + ag[p + nxy]);
          if (k === 0) {
            d += (agZR * w[p] - this.wInletZmin[j * nx + i]) * invDz;
          } else {
            const agZL = 0.5 * (ag[p - nxy] + ag[p]);
            d += (agZR * w[p] - agZL * w[p - nxy]) * invDz;
          }
          div[p] = d;
        }
      }
    }
    return div;
  }

  /**
   * Project u, v, w in place so div(u_new) equals divTarget (or zero).
   *
   * The divergence uses backward differences and the gradient correction uses
   * forward differences — a "compatible" pair, so that the composition
   * div(grad p) reproduces exactly the 7-point Laplacian in the matrix. Mixing
   * the two conventions makes the projection inconsistent and divergence grows
   * step over step instead of being removed.
   *
   * @returns {Float64Array} the pressure field (borrowed, not copied)
   */
  project(u, v, w, rho, dt, divTarget = null) {
    const { nz, ny, nx, nxy, dx, dy, dzArr, dFaceAbove, n } = this;

    const div = this.divergence(u, v, w);
    const rhs = new Float64Array(n);
    if (divTarget === null) {
      for (let i = 0; i < n; i++) rhs[i] = div[i] / dt;
    } else {
      for (let i = 0; i < n; i++) rhs[i] = (div[i] - divTarget[i]) / dt;
    }

    const x0 = this.pPrev ?? new Float64Array(n);
    const p = this._bicgstab(rhs, x0);
    if (this.pPrev === null) this.pPrev = new Float64Array(n);
    this.pPrev.set(p);

    // u_new = u* - dt * grad(p) / rho, forward differences.
    // Cells i=0..Nx-2 use the face at i+0.5. The outlet cell sees a Dirichlet
    // p = 0 half a cell out, giving +2*dt*p/(dx*rho).
    // Note k=0 and i=0 ARE corrected. They used to be skipped on the
    // assumption that the BC pass would re-pin them, but that pin hid the
    // divergence at sourced boundary cells — burning bed cells at k=0 have a
    // real S_pyro. The face BC is now carried by the ghost reflection in
    // divergence() instead.
    if (nx > 2) {
      for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) {
          const row = (k * ny + j) * nx;
          for (let i = 0; i < nx - 1; i++) {
            const c = row + i;
            const irF = 0.5 * (1.0 / Math.max(rho[c], RHO_FLOOR)
                             + 1.0 / Math.max(rho[c + 1], RHO_FLOOR));
            u[c] -= ((dt * (p[c + 1] - p[c])) / dx) * irF;
          }
          const cE = row + nx - 1;
          u[cE] += ((2.0 * dt * p[cE]) / dx) * (1.0 / Math.max(rho[cE], RHO_FLOOR));
        }
      }
    }

    if (ny > 1) {
      for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) {
          const jp = (j + 1) % ny;
          for (let i = 0; i < nx; i++) {
            const c = k * nxy + j * nx + i;
            const cp = k * nxy + jp * nx + i;
            const irF = 0.5 * (1.0 / Math.max(rho[c], RHO_FLOOR)
                             + 1.0 / Math.max(rho[cp], RHO_FLOOR));
            v[c] -= ((dt * (p[cp] - p[c])) / dy) * irF;
          }
        }
      }
    }

    if (nz > 2) {
      for (let k = 0; k < nz - 1; k++) {
        const dA = dFaceAbove[k];
        for (let c = k * nxy; c < (k + 1) * nxy; c++) {
          const irF = 0.5 * (1.0 / Math.max(rho[c], RHO_FLOOR)
                           + 1.0 / Math.max(rho[c + nxy], RHO_FLOOR));
          w[c] -= ((dt * (p[c + nxy] - p[c])) / dA) * irF;
        }
      }
      const kT = nz - 1;
      for (let c = kT * nxy; c < n; c++) {
        w[c] += ((2.0 * dt * p[c]) / dzArr[kT]) * (1.0 / Math.max(rho[c], RHO_FLOOR));
      }
    }

    return p;
  }
}
