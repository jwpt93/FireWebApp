/**
 * Separable Poisson solver — JS port of
 * model_outdoor/physics_3d/fft_poisson_3d.py :: SeparableLaplacian3D.
 *
 * Solves lap(p) = rhs by tensor-product eigendecomposition:
 *
 *     lap = L_x (x) I_y (x) I_z  +  I_x (x) L_y (x) I_z  +  I_x (x) I_y (x) L_z
 *
 * Boundary conditions match the projection operator:
 *     x   Neumann at i=0 (inlet),  Dirichlet at i=Nx-1 (outlet, p_face = 0)
 *     y   periodic
 *     z   Neumann at k=0 (ground), Dirichlet at k=Nz-1 (top, p_face = 0)
 *
 * WHY THIS AND NOT projection_3d.py.  In the parent project this is a
 * PRECONDITIONER for BiCGSTAB on the variable-coefficient operator
 * div((alpha_g/rho) grad p). But its own docstring notes it is EXACT when the
 * coefficient is near-constant, and in the 2D configuration the solver
 * reports proj_iter = 1 on every logged step with divmax ~5e-7 — the Krylov
 * wrapper never iterates. So the browser port needs this 228-line solve, not
 * the 948-line wrapper around it.
 *
 * Ny = 1 ONLY. The y-transform in the Python is an FFT over the periodic
 * direction. At Ny = 1 it is the identity and lambda_y = 0, which is the case
 * the applet runs (a true 2D slab). Supporting Ny > 1 means carrying complex
 * arithmetic through the transform for no benefit here, so it throws instead
 * of silently returning something wrong.
 *
 * Eigendecomposition: both 1D operators are symmetric TRIDIAGONAL, so the
 * implicit-QL algorithm below is enough — no dense eigensolver needed. L_z is
 * asymmetric as built (non-uniform dz) and is symmetrised by the diagonal
 * similarity D = diag(sqrt(dz)), which preserves eigenvalues and
 * tridiagonality. That similarity is exact rather than approximate because
 * d_below[k+1] == d_above[k], which makes a_k*dz_k == b_{k+1}*dz_{k+1}.
 *
 * NOTE ON VERIFYING THIS ONE. Eigenvectors are defined only up to sign and
 * ordering, so comparing them against LAPACK is meaningless. The test
 * compares the SOLUTION p for a given rhs, which is invariant to both.
 *
 * References: Swarztrauber (1974) SIAM J. Numer. Anal.; Rehm & Baum (1978).
 */

/**
 * Eigen-decompose a symmetric tridiagonal matrix in place (implicit QL with
 * Wilkinson shifts — the classic tql2).
 *
 * @param {Float64Array} d  (n) diagonal; overwritten with eigenvalues
 * @param {Float64Array} e  (n) sub-diagonal in e[1..n-1]; destroyed
 * @param {Float64Array} Z  (n*n) row-major; overwritten with eigenvectors in
 *                          COLUMNS, i.e. Z[i*n + j] is component i of
 *                          eigenvector j. Pass the identity.
 */
export function tql2(d, e, Z, n) {
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0.0;

  for (let l = 0; l < n; l++) {
    let iter = 0;
    let m;
    do {
      // Find a small sub-diagonal element to split on.
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= Number.EPSILON * dd) break;
      }
      if (m !== l) {
        if (iter++ === 50) throw new Error('tql2: no convergence in 50 iterations');
        let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        let r = Math.hypot(g, 1.0);
        g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1.0;
        let c = 1.0;
        let p = 0.0;
        for (let i = m - 1; i >= l; i--) {
          let f = s * e[i];
          const b = c * e[i];
          r = Math.hypot(f, g);
          e[i + 1] = r;
          if (r === 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          for (let k = 0; k < n; k++) {
            f = Z[k * n + i + 1];
            Z[k * n + i + 1] = s * Z[k * n + i] + c * f;
            Z[k * n + i] = c * Z[k * n + i] - s * f;
          }
        }
        if (r === 0.0 && m - 1 >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m !== l);
  }

  // Ascending order, carrying the eigenvectors along — matches the ordering
  // scipy.linalg.eigh returns, so lambda arrays line up.
  for (let i = 0; i < n - 1; i++) {
    let k = i;
    let p = d[i];
    for (let j = i + 1; j < n; j++) if (d[j] < p) { k = j; p = d[j]; }
    if (k !== i) {
      d[k] = d[i];
      d[i] = p;
      for (let j = 0; j < n; j++) {
        const t = Z[j * n + i];
        Z[j * n + i] = Z[j * n + k];
        Z[j * n + k] = t;
      }
    }
  }
}

export class SeparableLaplacian3D {
  /**
   * @param {object} g
   * @param {number} g.nz
   * @param {number} g.ny  must be 1
   * @param {number} g.nx
   * @param {number} g.dx
   * @param {number} g.dy
   * @param {Float64Array} g.dzArr       (nz)
   * @param {Float64Array} g.dFaceAbove  (nz)
   * @param {Float64Array} g.dFaceBelow  (nz)
   * @param {number} [g.epsReg]  diagonal regularisation, matching the
   *                             projection operator's
   */
  constructor({ nz, ny, nx, dx, dy, dzArr, dFaceAbove, dFaceBelow,
                epsReg = 1.0e-6 }) {
    if (ny !== 1) {
      throw new Error(
        `SeparableLaplacian3D: ny must be 1 (got ${ny}). The y-transform is ` +
        `an FFT over the periodic direction; only the Ny=1 slab is ported.`);
    }
    this.nz = nz; this.ny = ny; this.nx = nx;
    this.dzArr = dzArr;

    // ── x: Neumann at 0, Dirichlet face beyond Nx-1 ───────────────────────
    const invDx2 = 1.0 / (dx * dx);
    const dxDiag = new Float64Array(nx);
    const dxOff = new Float64Array(nx);      // sub-diagonal, index 1..nx-1
    for (let i = 0; i < nx; i++) {
      if (i === 0) {
        dxDiag[0] = -1.0 * invDx2;            // half-stencil (Neumann)
      } else if (i === nx - 1) {
        dxDiag[i] = -3.0 * invDx2;            // extra -2/dx^2 for the face
        dxOff[i] = invDx2;
      } else {
        dxDiag[i] = -2.0 * invDx2;
        dxOff[i] = invDx2;
      }
    }

    // ── z: Neumann at 0, Dirichlet face above Nz-1, non-uniform dz ────────
    // Built directly in symmetrised form. The raw operator has
    //   above(k) = (1/dz_k)/d_above[k],  below(k) = (1/dz_k)/d_below[k]
    // and D = diag(sqrt(dz)) makes the off-diagonals equal because
    // d_below[k+1] == d_above[k].
    const dzDiag = new Float64Array(nz);
    const dzOff = new Float64Array(nz);
    for (let k = 0; k < nz; k++) {
      const invDzK = 1.0 / dzArr[k];
      if (k === 0) {
        dzDiag[0] = -(invDzK / dFaceAbove[0]);
      } else if (k === nz - 1) {
        const coefBelow = invDzK / dFaceBelow[nz - 1];
        const faceCoef = 2.0 * invDzK * invDzK;
        dzDiag[k] = -faceCoef - coefBelow;
      } else {
        dzDiag[k] = -(invDzK / dFaceAbove[k]) - (invDzK / dFaceBelow[k]);
      }
      if (k > 0) {
        // symmetric off-diagonal: sqrt(above(k-1) * below(k))
        const above = (1.0 / dzArr[k - 1]) / dFaceAbove[k - 1];
        const below = invDzK / dFaceBelow[k];
        dzOff[k] = Math.sqrt(above * below);
      }
    }

    this.Ux = identity(nx);
    this.lambdaX = Float64Array.from(dxDiag);
    tql2(this.lambdaX, Float64Array.from(dxOff), this.Ux, nx);

    this.Uz = identity(nz);
    this.lambdaZ = Float64Array.from(dzDiag);
    tql2(this.lambdaZ, Float64Array.from(dzOff), this.Uz, nz);

    // y periodic, analytic. At ny = 1 this is exactly zero.
    const lambdaY0 = (2.0 / (dy * dy)) * (Math.cos(0.0) - 1.0);

    // Inverse eigenvalue grid. All eigenvalues are <= 0 and the Dirichlet
    // faces anchor the solution, so there is no exact null mode; epsReg only
    // guards the numerically-near-zero corner.
    this.lambdaInv = new Float64Array(nz * nx);
    for (let k = 0; k < nz; k++) {
      for (let i = 0; i < nx; i++) {
        this.lambdaInv[k * nx + i] =
          1.0 / (this.lambdaZ[k] + lambdaY0 + this.lambdaX[i] - epsReg);
      }
    }

    this.Dz = new Float64Array(nz);
    this.DzInv = new Float64Array(nz);
    for (let k = 0; k < nz; k++) {
      this.Dz[k] = Math.sqrt(dzArr[k]);
      this.DzInv[k] = 1.0 / this.Dz[k];
    }

    this._q = new Float64Array(nz * nx);
    this._t = new Float64Array(nz * nx);
  }

  /**
   * Solve lap(p) = rhs.
   *
   * @param {Float64Array} rhs (nz*nx), row-major with k outer
   * @param {Float64Array} [out] optional destination
   * @returns {Float64Array} p
   */
  solve(rhs, out = null) {
    const { nz, nx } = this;
    const q = this._q;
    const t = this._t;
    const p = out ?? new Float64Array(nz * nx);

    // 1. symmetrise along z
    for (let k = 0; k < nz; k++) {
      const s = this.Dz[k];
      for (let i = 0; i < nx; i++) q[k * nx + i] = rhs[k * nx + i] * s;
    }

    // 2a. forward x:  q @ U_x
    for (let k = 0; k < nz; k++) {
      for (let c = 0; c < nx; c++) {
        let acc = 0.0;
        for (let i = 0; i < nx; i++) acc += q[k * nx + i] * this.Ux[i * nx + c];
        t[k * nx + c] = acc;
      }
    }
    // 2b. y-transform is the identity at ny = 1.
    // 2c. forward z:  U_z^T @ t
    for (let r = 0; r < nz; r++) {
      for (let c = 0; c < nx; c++) {
        let acc = 0.0;
        for (let k = 0; k < nz; k++) acc += this.Uz[k * nz + r] * t[k * nx + c];
        q[r * nx + c] = acc;
      }
    }

    // 3. spectral divide
    for (let n = 0; n < nz * nx; n++) q[n] *= this.lambdaInv[n];

    // 4a. inverse z:  U_z @ q
    for (let r = 0; r < nz; r++) {
      for (let c = 0; c < nx; c++) {
        let acc = 0.0;
        for (let k = 0; k < nz; k++) acc += this.Uz[r * nz + k] * q[k * nx + c];
        t[r * nx + c] = acc;
      }
    }
    // 4b. inverse x:  t @ U_x^T
    for (let k = 0; k < nz; k++) {
      for (let c = 0; c < nx; c++) {
        let acc = 0.0;
        for (let i = 0; i < nx; i++) acc += t[k * nx + i] * this.Ux[c * nx + i];
        q[k * nx + c] = acc;
      }
    }

    // 5. de-symmetrise
    for (let k = 0; k < nz; k++) {
      const s = this.DzInv[k];
      for (let i = 0; i < nx; i++) p[k * nx + i] = q[k * nx + i] * s;
    }
    return p;
  }
}

function identity(n) {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1.0;
  return I;
}
