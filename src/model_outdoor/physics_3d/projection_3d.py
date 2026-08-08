"""Chorin (1967) pressure projection for 3D variable-density flow.

Given a tentative velocity field u* (from momentum_3d), enforce a
target divergence ∇·u^{n+1} = D_target by solving a pressure Poisson
equation:

    ∇²p = (1/dt) (∇·u* - D_target)
    u^{n+1} = u* - dt · ∇p / ρ

For pure incompressible flow D_target = 0 (Chorin 1967).  For
low-Mach flow with gas-phase mass sources (e.g. pyrolysis releasing
volatile gas into the bed), mass conservation requires:

    ∂ρ/∂t + ∇·(ρu) = S_mass

Under the low-Mach approximation (∂ρ/∂t ≈ 0, ρ ≈ const along streamlines):

    D_target = S_mass / ρ

This allows pyrolysis-driven gas expansion to push fuel out of cells
correctly — without it, fuel mass added by pyrolysis is silently
discarded by the strict-incompressible projection (Y_fuel grows but
no corresponding outflow), and Y_fuel can saturate locally.

Reference: Pember et al. (1998) JCP 142:1 — low-Mach projection with
gas-phase sources; FDS Tech Ref Vol.1 §3.2 (McDermott et al. 2011)
applies the same formulation to wildland-fire pyrolysis.

BC convention (Ferziger & Perić 2002 § 7.7.4):

  All velocity-equation BCs translate to ∂p/∂n = 0 (Neumann)
  via the identity n·u^{n+1} = n·u*, since the projection step
  cannot change the normal component on a wall.

  Compatibility: pure-Neumann Laplacian has a constant null mode,
  so the Poisson RHS must integrate to zero.  We enforce this
  discretely (rhs -= rhs.mean()).  When D_target ≠ 0 (e.g. mass
  source), the mean correction redistributes the global mass
  imbalance over the boundary — appropriate for our zero-gradient
  outlet which acts as a passive open boundary.

Implementation: build a sparse 7-point Laplacian once, factorize with
PyPardiso (Intel MKL PARDISO), reuse the factorization each step.

Solver choice rationale (CLAUDE.md Rule #17 — bit-exact determinism):
scipy.sparse.linalg.splu uses OpenBLAS for level-3 BLAS within SuperLU
supernodes, whose multi-thread reductions are not bit-deterministic
across thread counts.  PyPardiso (Intel MKL PARDISO) provides documented
Conditional Numerical Reproducibility (CBWR) — bit-identical results
across runs at fixed thread count and CBWR=AVX2 (or higher).  Worker
sets MKL_CBWR=AVX2, MKL_DYNAMIC=FALSE, OMP_DYNAMIC=FALSE, MKL_NUM_THREADS
to enforce the determinism contract.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from numba import njit
from pypardiso import PyPardisoSolver


@njit(cache=True)
def _fill_var_density_data(
    inv_rho, alpha_g, dz_arr, d_above, d_below,
    dx2, dy2, eps_reg, y_periodic,
    rows, cols, data,
):
    """Phase 14u-C / 14aw-1: variable-density Poisson kernel with optional
    α_g volume weighting.

    Discretizes ∇·((α_g/ρ)·∇p) using SEPARATE face-averaged α_g and
    face-averaged 1/ρ: face_coef = avg(α_g_face) × avg(inv_rho_face).
    This split matches the divergence operator ∇·(α_g·u), so that the
    discrete identity div_vw(grad_correction) = matrix·p holds exactly.

    When alpha_g ≡ 1.0 everywhere (caller passes ones), the face
    coefficient reduces to face-averaged 1/ρ (pre-14aw baseline,
    bit-exact preserved).

    Boundary stencils:
      x: i=0 Neumann (inlet), i=Nx-1 Dirichlet pressure (outlet)
      y: periodic (currently only mode supported in this Numba path)
      z: k=0 Neumann (wall), k=Nz-1 Dirichlet pressure (top)

    Dirichlet-pressure boundary faces use α_g at the boundary cell
    (cell-center extrapolation to the open face).

    Returns the index past the last filled entry.  Caller pre-allocates
    arrays large enough for the maximum possible entries (7·N).
    """
    Nz, Ny, Nx = inv_rho.shape
    idx = 0
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                p = (k * Ny + j) * Nx + i
                diag = eps_reg

                # x-direction
                if Nx > 1:
                    if 1 <= i <= Nx - 2:
                        ir_xp = 0.5 * (inv_rho[k,j,i] + inv_rho[k,j,i+1])
                        ir_xm = 0.5 * (inv_rho[k,j,i-1] + inv_rho[k,j,i])
                        ag_xp = 0.5 * (alpha_g[k,j,i] + alpha_g[k,j,i+1])
                        ag_xm = 0.5 * (alpha_g[k,j,i-1] + alpha_g[k,j,i])
                        cxp = ag_xp * ir_xp / dx2
                        cxm = ag_xm * ir_xm / dx2
                        diag += -cxp - cxm
                        rows[idx] = p; cols[idx] = (k*Ny+j)*Nx + (i+1); data[idx] = cxp; idx += 1
                        rows[idx] = p; cols[idx] = (k*Ny+j)*Nx + (i-1); data[idx] = cxm; idx += 1
                    elif i == 0:
                        ir_xp = 0.5 * (inv_rho[k,j,0] + inv_rho[k,j,1])
                        ag_xp = 0.5 * (alpha_g[k,j,0] + alpha_g[k,j,1])
                        cxp = ag_xp * ir_xp / dx2
                        diag += -cxp
                        rows[idx] = p; cols[idx] = (k*Ny+j)*Nx + 1; data[idx] = cxp; idx += 1
                    else:
                        ir_xm = 0.5 * (inv_rho[k,j,Nx-2] + inv_rho[k,j,Nx-1])
                        ir_xp_face = inv_rho[k,j,Nx-1]
                        ag_xm = 0.5 * (alpha_g[k,j,Nx-2] + alpha_g[k,j,Nx-1])
                        ag_xp_face = alpha_g[k,j,Nx-1]
                        cxm = ag_xm * ir_xm / dx2
                        cxp_face = 2.0 * ag_xp_face * ir_xp_face / dx2
                        diag += -cxm - cxp_face
                        rows[idx] = p; cols[idx] = (k*Ny+j)*Nx + (Nx-2); data[idx] = cxm; idx += 1

                # y-direction (periodic)
                if Ny > 1 and y_periodic:
                    jp = (j + 1) % Ny
                    jm = (j - 1) % Ny
                    ir_yp = 0.5 * (inv_rho[k,j,i] + inv_rho[k,jp,i])
                    ir_ym = 0.5 * (inv_rho[k,jm,i] + inv_rho[k,j,i])
                    ag_yp = 0.5 * (alpha_g[k,j,i] + alpha_g[k,jp,i])
                    ag_ym = 0.5 * (alpha_g[k,jm,i] + alpha_g[k,j,i])
                    cyp = ag_yp * ir_yp / dy2
                    cym = ag_ym * ir_ym / dy2
                    diag += -cyp - cym
                    rows[idx] = p; cols[idx] = (k*Ny+jp)*Nx + i; data[idx] = cyp; idx += 1
                    rows[idx] = p; cols[idx] = (k*Ny+jm)*Nx + i; data[idx] = cym; idx += 1

                # z-direction
                if Nz > 1:
                    inv_dz_k = 1.0 / dz_arr[k]
                    if 1 <= k <= Nz - 2:
                        ir_zp = 0.5 * (inv_rho[k,j,i] + inv_rho[k+1,j,i])
                        ir_zm = 0.5 * (inv_rho[k-1,j,i] + inv_rho[k,j,i])
                        ag_zp = 0.5 * (alpha_g[k,j,i] + alpha_g[k+1,j,i])
                        ag_zm = 0.5 * (alpha_g[k-1,j,i] + alpha_g[k,j,i])
                        czp = ag_zp * ir_zp * inv_dz_k / d_above[k]
                        czm = ag_zm * ir_zm * inv_dz_k / d_below[k]
                        diag += -czp - czm
                        rows[idx] = p; cols[idx] = ((k+1)*Ny+j)*Nx + i; data[idx] = czp; idx += 1
                        rows[idx] = p; cols[idx] = ((k-1)*Ny+j)*Nx + i; data[idx] = czm; idx += 1
                    elif k == 0:
                        ir_zp = 0.5 * (inv_rho[0,j,i] + inv_rho[1,j,i])
                        ag_zp = 0.5 * (alpha_g[0,j,i] + alpha_g[1,j,i])
                        czp = ag_zp * ir_zp * inv_dz_k / d_above[0]
                        diag += -czp
                        rows[idx] = p; cols[idx] = (1*Ny+j)*Nx + i; data[idx] = czp; idx += 1
                    else:
                        ir_zm = 0.5 * (inv_rho[Nz-2,j,i] + inv_rho[Nz-1,j,i])
                        ir_zp_face = inv_rho[Nz-1,j,i]
                        ag_zm = 0.5 * (alpha_g[Nz-2,j,i] + alpha_g[Nz-1,j,i])
                        ag_zp_face = alpha_g[Nz-1,j,i]
                        czm = ag_zm * ir_zm * inv_dz_k / d_below[Nz-1]
                        czp_face = 2.0 * ag_zp_face * ir_zp_face * inv_dz_k * inv_dz_k
                        diag += -czm - czp_face
                        rows[idx] = p; cols[idx] = ((Nz-2)*Ny+j)*Nx + i; data[idx] = czm; idx += 1

                # diagonal
                rows[idx] = p; cols[idx] = p; data[idx] = diag; idx += 1
    return idx


class ProjectionSolver3D:
    """Cached LU factorization of the pressure-Poisson operator.

    KNOWN LIMITATION (collocated-grid BC inconsistency):
    The matrix-solve is exact (residual ~1e-12), but the velocity update
    and divergence operators are inconsistent at velocity-Dirichlet
    boundaries (inlet i=0, no-slip wall k=0).  The pressure gradient at
    face 0.5 (between the boundary cell and its first interior neighbor)
    is never applied to any cell-centered velocity in the collocated
    scheme — the boundary cell's update is skipped (Dirichlet u known),
    and the interior cell uses gradient at face 1.5.  Consequence:
    div_new(u) != div_target at burning wall cells that have S_pyro != 0
    (Phase 14v-conv diagnostic: median residual ~1-3 1/s for fire steps).
    Iterating doesn't help — the residual is structural, not iterative.

    Mitigations:
      • Move bed away from inlet via outdoor.bed_x_start > 0 (removes
        S_pyro at i=0; ratifies inlet Neumann pressure BC).
      • Wall (k=0 with bed cells producing S_pyro) cannot be fixed
        within the collocated scheme without Rhie-Chow interpolation
        or a switch to a staggered (MAC) grid — both significant
        restructures, deferred.
    """

    def __init__(
        self,
        Nz: int, Ny: int, Nx: int,
        dy: float, dx: float,
        dz_arr: np.ndarray,        # (Nz,) [m] per-cell vertical spacing
        d_face_above: np.ndarray,  # (Nz,) [m] cell-center distance to k+1
        d_face_below: np.ndarray,  # (Nz,) [m] cell-center distance to k-1
        y_bc: str = "periodic",
        method: str = "pardiso",   # "pardiso" | "amg_cg" | "fft_pcg"
        cg_rtol: float = 1.0e-6,   # CG relative-residual tolerance
        amg_rebuild_every: int = 100,  # rebuild AMG hierarchy every N steps
    ) -> None:
        self.Nz = Nz; self.Ny = Ny; self.Nx = Nx
        self.dy = dy; self.dx = dx
        self.dz_arr = dz_arr
        self.d_face_above = d_face_above
        self.d_face_below = d_face_below
        # Backward-compat scalar (= bed-cell dz) for code that still reads .dz
        self.dz = float(dz_arr[0])
        self.y_bc = y_bc
        self.method = method
        self.cg_rtol = cg_rtol
        self.amg_rebuild_every = amg_rebuild_every
        self._pp_solver: Optional[PyPardisoSolver] = None
        # AMG-CG state (Phase 14ah: 2.7× speedup vs PARDISO refactor)
        self._ml = None                # pyamg.MultilevelSolver
        self._p_prev: Optional[np.ndarray] = None    # warm-start cache
        self._steps_since_amg_build: int = 0
        self._A_pos = None             # cached -A as SPD matrix
        # Phase 14ax-2: FFT-preconditioned-CG state.  When method='fft_pcg',
        # `_fft_solver` is the constant-coefficient ∇²p solver used as M⁻¹
        # in BiCGSTAB on the true variable-coefficient operator A.  Built
        # once at __init__; never rebuilt (no per-step setup overhead).
        self._fft_solver = None        # SeparableLaplacian3D | None
        if method == "fft_pcg":
            from model_outdoor.physics_3d.fft_poisson_3d import (
                SeparableLaplacian3D,
            )
            self._fft_solver = SeparableLaplacian3D(
                Nz=Nz, Ny=Ny, Nx=Nx,
                dx=dx, dy=dy,
                dz_arr=dz_arr,
                d_face_above=d_face_above,
                d_face_below=d_face_below,
            )
        # Phase 14aw-1: gas volume fraction α_g = 1 - α_s.  When None,
        # operator is ∇·((1/ρ)·∇p) (pre-14aw, pure-gas).  When set, operator
        # becomes ∇·((α_g/ρ)·∇p) and divergence becomes ∇·(α_g·u).  Default
        # None preserves bit-exact baseline behavior.
        self._alpha_g: Optional[np.ndarray] = None
        # Phase 14v-bc: face-Dirichlet BC values for divergence ghost reflection.
        # Set externally via set_inlet_BC().  Defaults to zero (no flow).
        self._u_inlet = np.zeros((Nz, Ny), dtype=np.float64)   # face -0.5 (x=0)
        # Phase 23 Refactor 2C: face-Dirichlet BC values at z=0 (bottom).
        # Set externally via set_bottom_inlet_BC().  When zero everywhere
        # (default), z=0 remains a no-slip wall — bit-exact-invariant for
        # the outdoor cases which never touch this.  Non-zero values
        # activate the z-min inlet (e.g. cup burner fuel-jet + coflow).
        self._w_inlet_zmin = np.zeros((Ny, Nx), dtype=np.float64)   # face k=-0.5
        # Phase 14u-opt2: skip legacy _build() (uniform-rho LU was 2 sec
        # startup overhead and never used — variable-density rebuild
        # happens per-step via rebuild_for_rho).

    # ─── Index helpers ───────────────────────────────────────────────────
    def _idx(self, k: int, j: int, i: int) -> int:
        return (k * self.Ny + j) * self.Nx + i

    def set_alpha_g(self, alpha_g: Optional[np.ndarray]) -> None:
        """Phase 14aw-1: set the gas volume fraction field α_g = 1 - α_s.

        When set (array shape (Nz, Ny, Nx)), the Poisson operator becomes
        ∇·((α_g/ρ)·∇p) (volume-weighted), the divergence operator becomes
        ∇·(α_g·u), and the velocity correction stays u_new = u - (dt/ρ)·∇p
        (the α_g factor cancels in the face-collocated form; see derivation
        in module docstring).  This is the canonical low-Mach E-E
        projection (Anderson & Jackson 1967; Pember et al. 1998 JCP 142:1).

        When alpha_g=None (default), the operator reverts to the standard
        ∇·((1/ρ)·∇p) variable-density Poisson and divergence is plain ∇·u.

        Must be called BEFORE rebuild_for_rho on any step where the value
        has changed, so the matrix coefficients reflect the current α_g.
        """
        if alpha_g is None:
            self._alpha_g = None
        else:
            self._alpha_g = np.ascontiguousarray(alpha_g, dtype=np.float64)

    def set_inlet_BC(self, u_inlet: np.ndarray) -> None:
        """Store the Dirichlet velocity BC at the inlet face (x=0).

        ``u_inlet`` shape (Nz, Ny).  Used by ``_divergence_compatible`` as the
        face flux at face -0.5 (mirror reflection: u_ghost = u_inlet),
        making div_x[0] = (u[0] - u_inlet)/dx — capturing deviations from
        the BC that the projection then corrects via pressure gradient
        at face 0.5.  The wall (k=0) BC is hardcoded as w_face = 0
        (no-slip / no-penetration); not stored.
        """
        self._u_inlet = np.ascontiguousarray(u_inlet, dtype=np.float64)

    def set_bottom_inlet_BC(self, w_inlet_zmin: np.ndarray) -> None:
        """Store the Dirichlet velocity BC at the bottom face (z=0).

        ``w_inlet_zmin`` shape (Ny, Nx).  Used by ``_divergence_compatible``
        as the face flux at face k=-0.5, making div_z[0] =
        (w[0] - w_inlet_zmin) / dz_arr[0] — capturing deviations that
        the projection corrects via pressure gradient at face 0.5.
        Zero-fill everywhere (default) reproduces the pre-Phase-23
        no-slip wall.  Cup burner installs a spatially-varying array:
        U_fuel in the fuel-jet band, U_coflow in the annulus, 0 in
        wall cells outside the chimney.
        """
        self._w_inlet_zmin = np.ascontiguousarray(w_inlet_zmin, dtype=np.float64)

    # ─── Public rebuild API (call once per outer time step) ──────────────
    def rebuild_for_rho(self, rho: np.ndarray) -> None:
        """Rebuild the Poisson matrix for current ρ field.

        Phase 14z-det: PyPardiso factorize for bit-determinism via MKL
        CBWR.  Replaces scipy splu (which used non-deterministic OpenBLAS
        reductions inside SuperLU supernodes).  Comparable speed (~30ms
        factorize, ~1ms solve at 10k cells).

        2026-05-13 optimization: cache CSR sparsity pattern after first
        rebuild.  Subsequent rebuilds:
          - Numba kernel fills _buf_data (fast)
          - scatter into existing CSR via cached _csr_map
          - skip the scipy COO→CSR conversion + sum_duplicates() entirely
        Saves ~50-100ms per step on large grids (no behavior change; the
        Numba kernel emits unique (row, col) pairs by construction, so
        the scatter is bijective and bit-exact reproduces the original
        coo.tocsr() result).
        """
        if self.method == "amg_cg":
            self._rebuild_for_rho_amg(rho)
            return

        if self.method == "fft_pcg":
            # Phase 14ax-2: refresh the variable-coefficient matrix data;
            # the FFT preconditioner is constant-coefficient and was built
            # once at __init__ (no per-step setup).
            if not hasattr(self, "_csr_map"):
                self._build_variable_density(rho)
                self._compute_csr_map()
            else:
                n_filled = self._fill_buffers_for_rho(rho)
                assert n_filled == self._n_filled, (
                    f"CSR pattern changed: {self._n_filled} → {n_filled}")
                self._A.data[self._csr_map] = self._buf_data[:n_filled]
            return

        # ── PARDISO path (legacy default) ──────────────────────────────
        if self._pp_solver is None or not hasattr(self, "_csr_map"):
            # First call: full build (COO → CSR) and cache the map.
            # Also call PARDISO phase 12 (analysis + numeric factorization)
            # so subsequent calls can skip phase 11 (analysis).
            self._build_variable_density(rho)
            self._compute_csr_map()
            if self._pp_solver is None:
                self._pp_solver = PyPardisoSolver()
            self._pp_solver.factorize(self._A)   # phase 12: analyze + factor
        else:
            # Subsequent calls (Opt 1 + Opt 2):
            #   - Numba kernel refills _buf_data
            #   - Scatter via cached CSR map (no COO→CSR, no sum_duplicates)
            #   - PARDISO phase 22: numerical factorization only, reuses
            #     phase-11 analysis from first call (pt handle persists).
            n_filled = self._fill_buffers_for_rho(rho)
            assert n_filled == self._n_filled, (
                f"CSR pattern changed: n_filled went from {self._n_filled} "
                f"to {n_filled} — should be constant after init."
            )
            self._A.data[self._csr_map] = self._buf_data[:n_filled]
            self._numeric_factorize_only(self._A)

    def _rebuild_for_rho_amg(self, rho: np.ndarray) -> None:
        """AMG-CG path: refill CSR data, periodically rebuild hierarchy.

        Phase 14ah: PyAMG smoothed_aggregation_solver as preconditioner
        for CG, replacing PARDISO direct LU (refactor 700ms + solve 350ms
        per step at 364k cells).  AMG hierarchy setup is ~830ms one-shot
        and reusable across many steps as a frozen preconditioner.  When
        ρ drifts far from the build-time field, the preconditioner
        degrades and CG iter count grows — we periodically rebuild every
        ``amg_rebuild_every`` steps (default 100).  Amortized cost
        ~8 ms/step.

        Our discrete operator ∇·((1/ρ)·∇p) is negative-semi-definite;
        with small ε·I regularization it's indefinite — CG requires SPD,
        so we negate (_A_pos = -_A) and solve A_pos·p = -rhs instead.
        """
        import pyamg
        if not hasattr(self, "_csr_map"):
            # First call: full build (COO → CSR) and cache the map.
            self._build_variable_density(rho)
            self._compute_csr_map()
            # Negated copy with shared sparsity (data buffer is independent).
            self._A_pos = (-self._A).tocsr()
            # Phase 14ah-2: ruge_stuben_solver (classical AMG) is 2× faster
            # than smoothed_aggregation on this 7-point Poisson with
            # periodic-y wraparound — bench at 96k cells:
            #   SA default (Gauss-Seidel):  106 ms / 3 iters
            #   RS default (classical):      55 ms / 2 iters
            # Both deliver same relative error vs PARDISO (~5e-3).
            self._ml = pyamg.ruge_stuben_solver(self._A_pos)
            self._steps_since_amg_build = 0
        else:
            # Subsequent calls: refill _A.data and mirror to _A_pos.data.
            n_filled = self._fill_buffers_for_rho(rho)
            assert n_filled == self._n_filled, (
                f"CSR pattern changed: n_filled went from {self._n_filled} "
                f"to {n_filled} — should be constant after init."
            )
            self._A.data[self._csr_map] = self._buf_data[:n_filled]
            # Mirror sign-flipped values into the SPD copy without
            # reallocating.  Both share the same CSR sparsity.
            self._A_pos.data[:] = -self._A.data
            # Periodic hierarchy rebuild for accuracy as ρ drifts.
            # Phase 14aw-3: when α_g weighting is active, force per-step
            # rebuild.  The α_g modulation makes the coefficient field
            # less smooth at bed-air interfaces, so the AMG hierarchy
            # built once at step 0 becomes a poor preconditioner within
            # tens of steps.  Symptom otherwise: BiCGSTAB declares
            # convergence at rtol=1e-3 but solution is garbage,
            # producing spurious velocities ~67 m/s that collapse dt
            # to ~0.6 ms (sweep abort 2026-05-27).  PARDISO is immune
            # because it refactors fresh every step.
            effective_rebuild_every = (1 if self._alpha_g is not None
                                       else self.amg_rebuild_every)
            self._steps_since_amg_build += 1
            if self._steps_since_amg_build >= effective_rebuild_every:
                self._ml = pyamg.ruge_stuben_solver(self._A_pos)
                self._steps_since_amg_build = 0

    def _numeric_factorize_only(self, A) -> None:
        """PARDISO phase 22 (numerical refactor) reusing phase-11 analysis.

        Replicates PyPardiso.factorize() but with phase=22 instead of 12.
        The pt (handle) array persists across calls, retaining the
        permutation + symbolic factorization computed in the first
        factorize().  Requires the matrix sparsity pattern to be
        unchanged from that initial call.

        Also updates self._pp_solver.factorized_A so subsequent solve()
        calls see the matrix as "already factorized" and use phase 33
        (solve only).
        """
        import numpy as _np
        solver = self._pp_solver
        # Update factorized_A cache (same logic as factorize()).
        if A.nnz > solver.size_limit_storage:
            solver.factorized_A = solver._hash_csr_matrix(A)
        else:
            solver.factorized_A = A.copy()
        solver.set_phase(22)
        b = _np.zeros((A.shape[0], 1))
        solver._call_pardiso(A, b)

    def _fill_buffers_for_rho(self, rho: np.ndarray) -> int:
        """Just run the Numba kernel to refill _buf_data (and _buf_rows,
        _buf_cols — those are ignored after first build).  Returns n_filled.

        Phase 14aw-1: kernel takes α_g separately so face coefficients
        are face-avg(α_g) × face-avg(1/ρ).  When self._alpha_g is None,
        an all-ones array is passed (bit-exact-equivalent to pre-14aw).
        """
        inv_rho = 1.0 / np.maximum(rho, 0.01)
        if self._alpha_g is not None:
            alpha_g = self._alpha_g
        else:
            alpha_g = np.ones_like(inv_rho)
        eps_reg = 1.0e-6
        y_periodic = (self.y_bc == "periodic")
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        n_filled = _fill_var_density_data(
            inv_rho, alpha_g, self.dz_arr, self.d_face_above, self.d_face_below,
            dx2, dy2, eps_reg, y_periodic,
            self._buf_rows, self._buf_cols, self._buf_data,
        )
        return n_filled

    def _compute_csr_map(self) -> None:
        """Compute permutation: csr_map[output_idx] = CSR position.

        Called ONCE after first _build_variable_density.  The CSR data
        array is filled by scipy's coo→csr conversion in (row, col)-sorted
        order.  We replicate the same ordering by lexsort, then invert.
        Subsequent rebuilds do data-only updates: A.data[csr_map] = buf_data.
        """
        n_filled = self._n_filled
        rows = self._buf_rows[:n_filled]
        cols = self._buf_cols[:n_filled]
        # CSR positions correspond to (row, col) sorted ascending by row,
        # then by col within each row.
        sort_order = np.lexsort((cols, rows))    # sort_order[k] = output idx going to CSR slot k
        csr_map = np.empty(n_filled, dtype=np.int64)
        csr_map[sort_order] = np.arange(n_filled)
        # Bit-exact safety: verify scatter produces the same CSR data we
        # already have in self._A.data.
        reconstructed = np.empty(n_filled, dtype=np.float64)
        reconstructed[csr_map] = self._buf_data[:n_filled]
        if not np.array_equal(reconstructed, self._A.data):
            raise RuntimeError(
                "CSR map self-check failed: scatter does not reproduce "
                "scipy's COO→CSR data layout.  Suggests the Numba kernel "
                "emits duplicate (row, col) pairs."
            )
        self._csr_map = csr_map

    # ─── Build & factorize ───────────────────────────────────────────────
    def _build_variable_density(self, rho: np.ndarray) -> None:
        """Phase 14u-C: variable-density Poisson matrix (Numba-accelerated).

        Discretize ∇·((1/ρ)·∇p) instead of ∇²p.  Face-averaged 1/ρ via
        arithmetic mean (= harmonic mean of ρ, Patankar 1980 §4.2.5).
        Numba kernel `_fill_var_density_data` fills (rows,cols,data) in
        ~1ms; scipy COO→CSC + LU factor adds ~50ms.  Total: ~50ms/step.
        """
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        N = Nz * Ny * Nx

        inv_rho = 1.0 / np.maximum(rho, 0.01)
        if self._alpha_g is not None:
            alpha_g = self._alpha_g
        else:
            alpha_g = np.ones_like(inv_rho)
        eps_reg = 1.0e-6
        y_periodic = (self.y_bc == "periodic")

        # Pre-allocate buffers (max possible: 7·N entries)
        max_entries = 7 * N
        if not hasattr(self, "_buf_rows") or self._buf_rows.size != max_entries:
            self._buf_rows = np.empty(max_entries, dtype=np.int64)
            self._buf_cols = np.empty(max_entries, dtype=np.int64)
            self._buf_data = np.empty(max_entries, dtype=np.float64)

        n_filled = _fill_var_density_data(
            inv_rho, alpha_g, self.dz_arr, self.d_face_above, self.d_face_below,
            dx2, dy2, eps_reg, y_periodic,
            self._buf_rows, self._buf_cols, self._buf_data,
        )
        self._n_filled = n_filled

        A = sp.coo_matrix(
            (self._buf_data[:n_filled],
             (self._buf_rows[:n_filled], self._buf_cols[:n_filled])),
            shape=(N, N),
        ).tocsr()   # CSR is faster for matvec (used in CG)
        A.sum_duplicates()
        self._A = A
        # No splu — was 2 sec for our periodic-y matrix due to high fill-in
        # from wraparound entries.  CG with diagonal preconditioner is
        # ~30ms for the same matrix.

    def _build(self) -> None:
        """Build the sparse Laplacian compatible with backward-div +
        forward-grad on a collocated Cartesian grid.

        Discrete identity: div_backward(grad_forward(p))[i] = standard
        7-point Laplacian for interior cells, half-stencil at boundaries.

        At cell (k=0, j=0, i=0) we pin p=0 to remove the constant null
        mode of the pure-Neumann Laplacian.
        """
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        dz_arr = self.dz_arr
        d_above = self.d_face_above
        d_below = self.d_face_below
        N = Nz * Ny * Nx

        rows = []; cols = []; data = []

        # Regularization: small ε·I added to the diagonal.  Removes the
        # constant null mode of the pure-Neumann Laplacian without
        # creating a sharp boundary like a single-cell pin would.  ε
        # must be << |λ_smallest_nonzero|; for our scales 1e-6 is safe.
        eps_reg = 1.0e-6

        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    p = self._idx(k, j, i)

                    diag = eps_reg

                    # ── x-direction (forward grad → backward div) ──────
                    # Phase 14u Option B: BC-consistent matrix.
                    # Interior i (1 ≤ i ≤ Nx-2): full stencil
                    #   ∇²p = (p[i+1] - 2p[i] + p[i-1]) / dx²
                    # Inlet i=0 (Dirichlet velocity ⇒ Neumann pressure):
                    #   half-stencil — diag = -1/dx² (FV with ∂p/∂n=0 at face)
                    # Outlet i=Nx-1 (Open vent ⇒ Dirichlet pressure p_face=0):
                    #   FV with face_grad = (0 - p[Nx-1])/(dx/2) = -2p/dx
                    #   ∇²p[Nx-1] = (-2·p[Nx-1]/dx - (p[Nx-1]-p[Nx-2])/dx)/dx
                    #            = (-3·p[Nx-1] + p[Nx-2])/dx²
                    #   diag = -3/dx², off-diag = +1/dx²
                    if Nx > 1:
                        if 1 <= i <= Nx - 2:
                            diag += -2.0 / dx2
                            rows.append(p); cols.append(self._idx(k, j, i + 1)); data.append(1.0 / dx2)
                            rows.append(p); cols.append(self._idx(k, j, i - 1)); data.append(1.0 / dx2)
                        elif i == 0:
                            # Inlet: Neumann pressure (Dirichlet velocity)
                            diag += -1.0 / dx2
                            rows.append(p); cols.append(self._idx(k, j, 1)); data.append(1.0 / dx2)
                        else:  # i == Nx - 1
                            # Outlet: Dirichlet pressure (open vent)
                            diag += -3.0 / dx2
                            rows.append(p); cols.append(self._idx(k, j, Nx - 2)); data.append(1.0 / dx2)

                    # ── y-direction ────────────────────────────────────
                    if Ny > 1:
                        if self.y_bc == "periodic":
                            jp = (j + 1) % Ny
                            jm = (j - 1) % Ny
                            diag += -2.0 / dy2
                            rows.append(p); cols.append(self._idx(k, jp, i)); data.append(1.0 / dy2)
                            rows.append(p); cols.append(self._idx(k, jm, i)); data.append(1.0 / dy2)
                        else:  # edge_loss → Neumann (half-stencil at boundaries)
                            if 1 <= j <= Ny - 2:
                                diag += -2.0 / dy2
                                rows.append(p); cols.append(self._idx(k, j + 1, i)); data.append(1.0 / dy2)
                                rows.append(p); cols.append(self._idx(k, j - 1, i)); data.append(1.0 / dy2)
                            elif j == 0:
                                diag += -1.0 / dy2
                                rows.append(p); cols.append(self._idx(k, 1, i)); data.append(1.0 / dy2)
                            else:
                                diag += -1.0 / dy2
                                rows.append(p); cols.append(self._idx(k, Ny - 2, i)); data.append(1.0 / dy2)

                    # ── z-direction ────────────────────────────────────
                    # Phase 14u Option B: BC-consistent matrix.
                    # Wall k=0 (Dirichlet velocity ⇒ Neumann pressure):
                    #   half-stencil — diag = -coef_above
                    # Top k=Nz-1 (Open vent ⇒ Dirichlet pressure p_face=0):
                    #   FV with face_grad = (0 - p[Nz-1])/(dz_arr[Nz-1]/2)
                    #                     = -2·p[Nz-1]/dz_arr[Nz-1]
                    #   ∇²p[Nz-1] = (-2·p[Nz-1]/dz_arr[Nz-1] - (p[Nz-1]-p[Nz-2])/d_below[Nz-1])
                    #                / dz_arr[Nz-1]
                    #             = -2/dz_arr[Nz-1]² · p[Nz-1] - coef_below·(p[Nz-1]-p[Nz-2])
                    if Nz > 1:
                        inv_dz_k = 1.0 / dz_arr[k]
                        coef_above = inv_dz_k / d_above[k]   # = 1/(dz·d_above)
                        coef_below = inv_dz_k / d_below[k]   # = 1/(dz·d_below)
                        if 1 <= k <= Nz - 2:
                            diag += -coef_above - coef_below
                            rows.append(p); cols.append(self._idx(k + 1, j, i)); data.append(coef_above)
                            rows.append(p); cols.append(self._idx(k - 1, j, i)); data.append(coef_below)
                        elif k == 0:
                            # Wall: Neumann pressure (Dirichlet velocity)
                            diag += -coef_above
                            rows.append(p); cols.append(self._idx(1, j, i)); data.append(coef_above)
                        else:  # k == Nz - 1
                            # Top: Dirichlet pressure (open vent), face at z=Lz
                            # face_coef = 2 / (dz_arr[k]·dz_arr[k]) for the half-cell
                            # to face contribution
                            face_coef = 2.0 * inv_dz_k * inv_dz_k
                            diag += -face_coef - coef_below
                            rows.append(p); cols.append(self._idx(Nz - 2, j, i)); data.append(coef_below)

                    rows.append(p); cols.append(p); data.append(diag)

        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A.sum_duplicates()
        self._A = A
        if self._pp_solver is None:
            self._pp_solver = PyPardisoSolver()
        self._pp_solver.factorize(A)

    # ─── Project ─────────────────────────────────────────────────────────
    def project(
        self,
        u: np.ndarray, v: np.ndarray, w: np.ndarray,
        rho: np.ndarray,
        dt: float,
        div_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """In-place project u, v, w so that ∇·u^{n+1} = div_target.

        ``div_target`` (default None → strict incompressible ∇·u = 0).
        When provided as an array shape (Nz, Ny, Nx), the projection
        produces a velocity field with that divergence.  Used for
        low-Mach gas-phase mass sources: ``div_target = S_mass / ρ``
        where S_mass [kg/m³/s] is the pyrolysis volatile production
        rate (or any other gas-phase mass source).

        Uses backward-difference divergence + forward-difference gradient
        (a "compatible" pair) so that the discrete identity
            div(grad p) = (1/dx²) (p[i+1] - 2p[i] + p[i-1])
        matches the standard 7-point Laplacian used in the matrix.
        Otherwise the projection is inconsistent and divergence grows.

        At domain boundaries, ghost-cell reflection
        (u_ghost = u_boundary, p_ghost = p_boundary) is applied so that
        normal derivatives at walls are zero — consistent with the
        Neumann BC on pressure.
        """
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        dx, dy = self.dx, self.dy
        d_above = self.d_face_above
        dz_arr = self.dz_arr

        div = self._divergence_compatible(u, v, w)
        if div_target is not None:
            # Low-Mach: target is S_mass/ρ, project so ∇·u^{n+1} = div_target
            div = div - div_target
        # Phase 14u-C: variable-density Poisson.  Caller calls
        # rebuild_for_rho() at start of each outer step.  Solver: CG with
        # diagonal (Jacobi) preconditioner.  Direct LU was 2s/call due to
        # high fill-in from periodic-y wraparound; CG converges in
        # ~30-50 iters at ~30ms total.
        if not hasattr(self, "_A") or self._A is None:
            self.rebuild_for_rho(rho)
        rhs = (div / dt).reshape(-1)

        if self.method == "amg_cg":
            from scipy.sparse.linalg import bicgstab
            # Solve A_pos · p = -rhs (equivalent to A · p = rhs since
            # A_pos = -A).  Note A_pos is NOT strictly SPD: the ε·I
            # regularization added to A becomes -ε·I on the diagonal of
            # -A, giving a single tiny negative eigenvalue near -ε.  CG
            # breaks down on such indefinite systems, so we use BiCGSTAB
            # (handles non-SPD, slightly more cost per iter — typically
            # 2 matvecs vs 1).  AMG V-cycle remains a strong
            # preconditioner.  Warm-start from previous step's solution.
            x0 = self._p_prev if self._p_prev is not None else np.zeros_like(rhs)
            M = self._ml.aspreconditioner(cycle='V')
            p_flat, info = bicgstab(
                self._A_pos, -rhs, x0=x0, M=M,
                rtol=self.cg_rtol, maxiter=200,
            )
            if info != 0:
                # Didn't converge — force AMG rebuild and retry once.
                import pyamg
                self._ml = pyamg.ruge_stuben_solver(self._A_pos)
                self._steps_since_amg_build = 0
                M = self._ml.aspreconditioner(cycle='V')
                p_flat, info = bicgstab(
                    self._A_pos, -rhs, x0=x0, M=M,
                    rtol=self.cg_rtol, maxiter=200,
                )
                if info != 0:
                    raise RuntimeError(
                        f"AMG-BiCGSTAB failed to converge: info={info} "
                        f"(positive=maxiter, negative=breakdown).  "
                        f"Cells={self._A.shape[0]}, rtol={self.cg_rtol}, "
                        f"after AMG rebuild."
                    )
            self._p_prev = p_flat.copy()
        elif self.method == "fft_pcg":
            # Phase 14ax-2: BiCGSTAB on the true variable-coefficient matrix
            # A with a constant-coefficient ∇²p preconditioner (SeparableLaplacian3D).
            # At α_g/ρ ≈ const, the preconditioner is exact and convergence
            # happens in 1 iter.  At typical fire ρ-contrast (~4×) we expect
            # 2-5 iters.  No per-step hierarchy setup (preconditioner is
            # mathematical, not learned).
            from scipy.sparse.linalg import bicgstab, LinearOperator
            shape3d = (Nz, Ny, Nx)
            fft_solver = self._fft_solver

            def _apply_M_inv(r):
                # r is 1D; reshape, solve ∇²p̃ = r, return 1D.
                # Sign convention: matrix A has negative diagonal (Laplacian);
                # FFT solver inverts the same operator with the same sign.
                return fft_solver.solve(r.reshape(shape3d)).reshape(-1)

            M = LinearOperator((rhs.size, rhs.size), matvec=_apply_M_inv)
            x0 = self._p_prev if self._p_prev is not None else np.zeros_like(rhs)
            # Phase 14ax-3: count BiCGSTAB iterations for performance triage
            _n_iter = [0]
            def _cb(xk):
                _n_iter[0] += 1
            p_flat, info = bicgstab(
                self._A, rhs, x0=x0, M=M,
                rtol=self.cg_rtol, maxiter=200,
                callback=_cb,
            )
            self._last_fft_pcg_iters = _n_iter[0]
            if info != 0:
                raise RuntimeError(
                    f"FFT-PCG BiCGSTAB failed to converge: info={info} "
                    f"(positive=maxiter at 200, negative=breakdown).  "
                    f"Cells={self._A.shape[0]}, rtol={self.cg_rtol}."
                )
            self._p_prev = p_flat.copy()
        else:
            # Direct LU solve via PyPardiso (Phase 14z-det).  MKL-CBWR
            # bit-deterministic at fixed thread count.
            p_flat = self._pp_solver.solve(self._A, rhs)
        p = p_flat.reshape((Nz, Ny, Nx))

        # u_new = u* - dt · ∇p / ρ  using FORWARD differences so that
        # subsequent backward-div gives the 7-point Laplacian.
        # Phase 14v-bc: include u[0] and w[0] in the correction.  These
        # were previously skipped under the assumption that BC application
        # would re-pin them; that pin masked the divergence at sourced
        # boundary cells (S_pyro at burning bed cells, k=0).  Now u[0]
        # and w[0] evolve, and the FACE BC is enforced via mirror ghost
        # reflection in _divergence_compatible (u_ghost = u_inlet at face
        # -0.5, w_ghost = 0 at face k=-0.5).
        #
        # Boundary-cell velocity update at i=0 (cell adjacent to inlet
        # face -0.5): apply gradient at face 0.5 (between cells 0 and 1).
        # Matches matrix's row 0 stencil (cxp coupling only, Neumann
        # pressure at face -0.5).
        #
        # Outlet (i=Nx-1, Dirichlet pressure p_face=0):
        #   forward grad at face Nx-0.5 = (0 - p[Nx-1])/(dx/2) = -2 p[Nx-1]/dx
        #   u_new[Nx-1] = u[Nx-1] + 2·dt·p[Nx-1]/(dx·ρ)
        # Phase 14u-C: velocity correction uses face-averaged 1/ρ
        # (consistent with matrix's variable-density operator).
        inv_rho = 1.0 / np.maximum(rho, 0.01)
        if Nx > 2:
            # Cells i=0..Nx-2 all use forward gradient at face i+0.5.
            inv_rho_face_x = 0.5 * (inv_rho[:, :, :-1] + inv_rho[:, :, 1:])
            u[:, :, :-1] -= dt * (p[:, :, 1:] - p[:, :, :-1]) / dx * inv_rho_face_x
            # Outlet (i=Nx-1): face at x=Lx, p_face=0 ⇒ +2·dt·p[Nx-1]/dx·inv_rho.
            u[:, :, -1] += 2.0 * dt * p[:, :, -1] / dx * inv_rho[:, :, -1]

        # y: forward, with periodic (no inlet/outlet in y).
        if self.y_bc == "periodic" and Ny > 1:
            inv_rho_face_y = 0.5 * (inv_rho + np.roll(inv_rho, -1, axis=1))
            v -= dt * (np.roll(p, -1, axis=1) - p) / dy * inv_rho_face_y
        elif Ny > 1:
            inv_rho_face_y = 0.5 * (inv_rho[:, :-1, :] + inv_rho[:, 1:, :])
            v[:, :-1, :] -= dt * (p[:, 1:, :] - p[:, :-1, :]) / dy * inv_rho_face_y

        # z: include k=0 (was skipped — same fix as inlet).  All cells
        # k=0..Nz-2 use forward gradient at face k+0.5.  Top (k=Nz-1) uses
        # Dirichlet pressure formula.
        if Nz > 2:
            d_above_all = d_above[:-1].reshape(-1, 1, 1)  # k=0..Nz-2
            inv_rho_face_z = 0.5 * (inv_rho[:-1, :, :] + inv_rho[1:, :, :])
            w[:-1, :, :] -= dt * (p[1:, :, :] - p[:-1, :, :]) / d_above_all * inv_rho_face_z
            # Top: face at z=Lz, ρ_face = ρ[Nz-1]
            w[-1, :, :] += 2.0 * dt * p[-1, :, :] / dz_arr[-1] * inv_rho[-1, :, :]

        return p

    # ─── FV divergence with face-Dirichlet BC ghost reflection ─────────
    def _divergence_compatible(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    ) -> np.ndarray:
        """Finite-volume divergence consistent with the matrix discretization.

        At Dirichlet velocity boundaries (inlet i=0, no-slip wall k=0),
        uses MIRROR ghost reflection: u_ghost = u_inlet (face value),
        w_ghost = 0.  This makes div[0] = (u[0] − u_inlet)/dx and
        div_z[0] = w[0]/dz_arr[0] — capturing deviations the projection
        then corrects via pressure gradient at face 0.5.

        Normalization uses dz_arr[k] (cell volume) instead of d_below[k]
        (cell-center distance) to match the matrix's FV operator.

        Phase 14aw-1: when self._alpha_g is set, returns the volume-
        weighted divergence ∇·(α_g·u) using face-averaged α_g.  Domain
        boundary ghost faces use α_g_inlet = α_g_top = 1 (pure-gas open
        boundaries).  When self._alpha_g is None, returns plain ∇·u
        (bit-exact preserved).
        """
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        dx, dy = self.dx, self.dy
        dz_arr = self.dz_arr
        u_inlet = self._u_inlet
        w_inlet_zmin = self._w_inlet_zmin
        div = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        dz_arr_k = dz_arr.reshape(-1, 1, 1)

        if self._alpha_g is None:
            # Pure-gas (bit-exact pre-14aw behavior)
            # x: ∂u/∂x at cell i = (u[i] - u[i-1])/dx, with u_ghost = u_inlet at i=0.
            div[:, :, 1:] += (u[:, :, 1:] - u[:, :, :-1]) / dx
            div[:, :, 0]  += (u[:, :, 0] - u_inlet) / dx
            # y (periodic or zero-grad)
            if self.y_bc == "periodic" and Ny > 1:
                div += (v - np.roll(v, 1, axis=1)) / dy
            elif Ny > 1:
                div[:, 1:, :] += (v[:, 1:, :] - v[:, :-1, :]) / dy
            # z: FV form uses dz_arr[k].  Ghost w_ghost at k=-0.5 face:
            # for the pre-Phase-23 outdoor cases w_inlet_zmin is zero
            # everywhere (no-slip wall, bit-exact-invariant); cup burner
            # sets it to U_fuel in the jet, U_coflow in the annulus,
            # 0 in cells outside the chimney.
            div[1:, :, :] += (w[1:, :, :] - w[:-1, :, :]) / dz_arr_k[1:]
            div[0, :, :]  += (w[0, :, :] - w_inlet_zmin) / dz_arr_k[0]
            return div

        # Phase 14aw-1: volume-weighted divergence ∇·(α_g u)
        ag = self._alpha_g
        # x-faces: face_{i+0.5} = 0.5*(α_g[i] + α_g[i+1]) for i=0..Nx-2.
        # In backward-diff convention, u[i] is the flux at face_{i+0.5}.
        # Inlet face_{-0.5} α_g_ghost = 1 (pure-gas inlet).
        # Outlet face_{Nx-1+0.5} α_g_ghost = α_g[Nx-1] (cell-center extrap).
        ag_xf = 0.5 * (ag[:, :, :-1] + ag[:, :, 1:])   # shape (Nz,Ny,Nx-1)
        # div[0] = (ag_xf[0] × u[0] - 1.0 × u_inlet) / dx
        div[:, :, 0] += (ag_xf[:, :, 0] * u[:, :, 0] - u_inlet) / dx
        # div[i] for i=1..Nx-1: (ag_xface_R × u[i] - ag_xface_L × u[i-1]) / dx
        # ag_xface_R = ag_xf[i] for i<Nx-1, else ag[Nx-1] (outlet extrap)
        # ag_xface_L = ag_xf[i-1]
        if Nx > 2:
            div[:, :, 1:-1] += (ag_xf[:, :, 1:] * u[:, :, 1:-1]
                                - ag_xf[:, :, :-1] * u[:, :, :-2]) / dx
        if Nx > 1:
            div[:, :, -1] += (ag[:, :, -1] * u[:, :, -1]
                              - ag_xf[:, :, -1] * u[:, :, -2]) / dx

        # y: periodic — face_{j+0.5} = 0.5*(α_g[j] + α_g[j+1]) (or wrapped)
        if self.y_bc == "periodic" and Ny > 1:
            ag_yf = 0.5 * (ag + np.roll(ag, -1, axis=1))      # face between j and j+1
            ag_yf_L = np.roll(ag_yf, 1, axis=1)                # face between j-1 and j
            v_L = np.roll(v, 1, axis=1)                        # v at face j-0.5 (= v[j-1])
            div += (ag_yf * v - ag_yf_L * v_L) / dy
        elif Ny > 1:
            ag_yf = 0.5 * (ag[:, :-1, :] + ag[:, 1:, :])
            div[:, 1:, :] += (ag_yf * v[:, 1:, :]
                              - np.concatenate([ag[:, :1, :] * 0 + 0,
                                                ag_yf[:, :-1, :]], axis=1)
                              * np.concatenate([v[:, :1, :] * 0,
                                                v[:, :-1, :]], axis=1)) / dy

        # z: k=0 face treats ghost per _w_inlet_zmin (zero for outdoor
        # wall — bit-exact invariant; U_fuel/U_coflow for cup burner).
        # Face α_g_ghost = 1 (pure-gas inlet — no fuel below the cup rim).
        # Top face at k=Nz-1+0.5 (open vent): α_g_ghost = α_g[Nz-1].
        if Nz > 1:
            ag_zf = 0.5 * (ag[:-1, :, :] + ag[1:, :, :])      # shape (Nz-1,Ny,Nx)
            # div[0] += (ag_zf[0] × w[0] - 1.0 × w_inlet_zmin) / dz_arr[0]
            div[0, :, :] += (ag_zf[0, :, :] * w[0, :, :] - w_inlet_zmin) / dz_arr_k[0, :, :]
            if Nz > 2:
                div[1:-1, :, :] += (ag_zf[1:, :, :] * w[1:-1, :, :]
                                    - ag_zf[:-1, :, :] * w[:-2, :, :]) / dz_arr_k[1:-1, :, :]
            # div[Nz-1]: (α_g[Nz-1] × w[Nz-1] - ag_zf[Nz-2] × w[Nz-2]) / dz_arr[Nz-1]
            div[-1, :, :] += (ag[-1, :, :] * w[-1, :, :]
                              - ag_zf[-1, :, :] * w[-2, :, :]) / dz_arr_k[-1, :, :]
        return div

    # ─── Diagnostic (alias for the compatible div) ───────────────────────
    def divergence(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    ) -> np.ndarray:
        """Compute ∇·u for diagnostics (after project should be ~0)."""
        return self._divergence_compatible(u, v, w)
