"""Phase 14ax: constant-coefficient 3D Poisson solver via separable 1D
eigendecompositions.

Solves ∇²p = RHS on a Cartesian grid with the boundary conditions
matching ProjectionSolver3D's variable-density Poisson matrix:

    x: Neumann at i=0 (inlet),       Dirichlet at i=Nx-1 (outlet,  p_face=0)
    y: periodic
    z: Neumann at k=0 (wall, ground), Dirichlet at k=Nz-1 (top,    p_face=0)

with arbitrary (possibly non-uniform) ``dz_arr`` in z.

Used as a preconditioner ``M`` for BiCGSTAB on the variable-coefficient
operator A = ∇·((α_g/ρ)·∇p) in ProjectionSolver3D.  At constant
coefficient (α_g/ρ ≈ const everywhere) this preconditioner is EXACT
and BiCGSTAB converges in 1 iteration.  At variable coefficient
(typical fire: ρ ranges 0.3-1.2, contrast ≈ 4×) it remains a high-
quality preconditioner whose iteration count grows mildly with
coefficient contrast.

Algorithm: separable 3D Poisson via tensor-product 1D eigendecompositions.
   ∇²  =  L_x ⊗ I_y ⊗ I_z  +  I_x ⊗ L_y ⊗ I_z  +  I_x ⊗ I_y ⊗ L_z

Build once (at construction): eigendecompositions of L_x and L_z,
analytic eigenvalues for periodic L_y.  Each solve is:
  1. Forward transform: x along axis=2, y FFT along axis=1, z along axis=0
  2. Pointwise divide by the eigenvalue grid Λ[k,j,i] = λ_z[k]+λ_y[j]+λ_x[i]
  3. Inverse transform in the same three directions

Setup cost: O(Nx³ + Nz³) eigendecomp (~5-50 ms one-shot).
Per-solve cost: O(N · (Nx + log Ny + Nz)) ~ tens of ms for typical
grids; no per-step setup overhead.

References:
- Swarztrauber (1974) SIAM J. Numer. Anal. — separable Poisson solvers
- Rehm & Baum (1978) — pure-gas variant for constant-density FDS use
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


class SeparableLaplacian3D:
    """Solve ∇²p = RHS by tensor-product eigendecompositions.

    Parameters
    ----------
    Nz, Ny, Nx : int
        Grid dimensions (z is index 0, y is 1, x is 2 in the (Nz,Ny,Nx)
        array convention used by ProjectionSolver3D).
    dx, dy : float
        Uniform x and y cell spacing [m].
    dz_arr : np.ndarray of shape (Nz,)
        Per-cell vertical spacing [m].  May be non-uniform.
    d_face_above : np.ndarray of shape (Nz,)
        Distance from cell k center to cell k+1 center [m].  For k=Nz-1
        this should equal dz_arr[Nz-1] (distance to ghost above the
        Dirichlet face — convention matches ProjectionSolver3D).
    d_face_below : np.ndarray of shape (Nz,)
        Distance from cell k to cell k-1 center [m].  For k=0 unused
        (ghost is reflective Neumann face).
    eps_reg : float, default 1.0e-6
        Diagonal regularization added to the operator before inversion.
        Matches ProjectionSolver3D's ``eps_reg`` so the null mode
        treatment is consistent across the two operators.
    """

    def __init__(
        self,
        Nz: int, Ny: int, Nx: int,
        dx: float, dy: float,
        dz_arr: np.ndarray,
        d_face_above: np.ndarray,
        d_face_below: np.ndarray,
        eps_reg: float = 1.0e-6,
    ) -> None:
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy = dx, dy
        self.dz_arr = np.asarray(dz_arr, dtype=np.float64)
        self.eps_reg = eps_reg

        # ─── Build 1D operators ─────────────────────────────────────────
        L_x = self._build_L_x(Nx, dx)
        L_z_asym = self._build_L_z(Nz, dz_arr, d_face_above, d_face_below)
        # z is asymmetric (non-uniform dz).  Symmetrize via D·L·D^{-1}
        # where D = diag(sqrt(dz_arr)).  This preserves eigenvalues and
        # gives orthogonal eigenvectors in q-space (q = D·p).
        D = np.sqrt(self.dz_arr)
        D_inv = 1.0 / D
        L_z_sym = (D[:, None] * L_z_asym) * D_inv[None, :]

        # ─── Eigendecompose 1D operators ────────────────────────────────
        # Both are real symmetric ⇒ scipy.linalg.eigh.
        # Negative semi-definite (Laplacian convention with sign here is
        # such that diag is negative).
        lambda_x, self.U_x = eigh(L_x)
        lambda_z, self.U_z = eigh(L_z_sym)

        # y: periodic ⇒ analytic eigenvalues, FFT does the transform
        j = np.arange(Ny)
        lambda_y = (2.0 / (dy * dy)) * (np.cos(2.0 * np.pi * j / Ny) - 1.0)

        # ─── Precompute inverse eigenvalue grid ─────────────────────────
        # Λ_total[k, j, i] = λ_z[k] + λ_y[j] + λ_x[i]
        # All ≤ 0; with mixed Neumann + Dirichlet BCs in x and z, no
        # exact null mode exists (Dirichlet anchors the solution).  ε
        # regularization handles numerical safety for the y-periodic
        # j=0 mode coupled to a near-zero x or z mode.
        lambda_total = (lambda_z[:, None, None]
                      + lambda_y[None, :, None]
                      + lambda_x[None, None, :])
        # Subtract ε to match the matrix's diagonal regularization
        # (matrix has +ε on diag; here we work with operator L − εI, so
        # eigenvalues are λ_i − ε).  For inversion we negate the sign:
        # since L is negative semi-definite, we solve −L · p = −rhs
        # equivalently and use |λ_total − ε|.
        lambda_total = lambda_total - eps_reg
        # Inverse (safe — no zero crossing because λ ≤ 0 and ε > 0).
        self.lambda_total_inv = 1.0 / lambda_total

        # Cache D for the z-axis similarity transform per solve
        self._D_z = D                # shape (Nz,)
        self._D_z_inv = D_inv        # shape (Nz,)

    # ─── 1D operator builders (match ProjectionSolver3D stencils) ─────
    @staticmethod
    def _build_L_x(Nx: int, dx: float) -> np.ndarray:
        """Discrete 1D Laplacian with Neumann at i=0 and Dirichlet at
        i=Nx-1 (p_face=0 at face Nx-0.5).  Matches the x-direction
        stencil in ``_fill_var_density_data`` for inv_rho ≡ 1.
        """
        L = np.zeros((Nx, Nx), dtype=np.float64)
        inv_dx2 = 1.0 / (dx * dx)
        for i in range(Nx):
            if 1 <= i <= Nx - 2:
                L[i, i] = -2.0 * inv_dx2
                L[i, i + 1] = inv_dx2
                L[i, i - 1] = inv_dx2
            elif i == 0:
                # Neumann: half-stencil
                L[0, 0] = -1.0 * inv_dx2
                L[0, 1] = 1.0 * inv_dx2
            else:  # i == Nx - 1
                # Dirichlet at face Nx-0.5: diag picks up an extra
                # -2/dx² for the face contribution.
                L[Nx - 1, Nx - 1] = -3.0 * inv_dx2
                L[Nx - 1, Nx - 2] = 1.0 * inv_dx2
        return L

    @staticmethod
    def _build_L_z(
        Nz: int, dz_arr: np.ndarray,
        d_above: np.ndarray, d_below: np.ndarray,
    ) -> np.ndarray:
        """Discrete 1D Laplacian with Neumann at k=0 and Dirichlet at
        k=Nz-1.  Non-uniform spacing makes this matrix asymmetric in
        its raw form; the constructor symmetrizes via diagonal scaling.
        """
        L = np.zeros((Nz, Nz), dtype=np.float64)
        for k in range(Nz):
            inv_dz_k = 1.0 / dz_arr[k]
            if 1 <= k <= Nz - 2:
                coef_above = inv_dz_k / d_above[k]
                coef_below = inv_dz_k / d_below[k]
                L[k, k] = -coef_above - coef_below
                L[k, k + 1] = coef_above
                L[k, k - 1] = coef_below
            elif k == 0:
                # Neumann at wall
                coef_above = inv_dz_k / d_above[0]
                L[0, 0] = -coef_above
                L[0, 1] = coef_above
            else:  # k == Nz - 1
                # Dirichlet at top face z=Lz
                coef_below = inv_dz_k / d_below[Nz - 1]
                face_coef = 2.0 * inv_dz_k * inv_dz_k
                L[Nz - 1, Nz - 1] = -face_coef - coef_below
                L[Nz - 1, Nz - 2] = coef_below
        return L

    # ─── Solve ───────────────────────────────────────────────────────────
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ∇²p = rhs.

        Parameters
        ----------
        rhs : np.ndarray of shape (Nz, Ny, Nx)

        Returns
        -------
        p : np.ndarray of shape (Nz, Ny, Nx)
        """
        if rhs.shape != (self.Nz, self.Ny, self.Nx):
            raise ValueError(
                f"rhs shape {rhs.shape} != expected "
                f"({self.Nz}, {self.Ny}, {self.Nx})")
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx

        # 1. Symmetrize: multiply rhs by D along z (q-space transform)
        q = rhs * self._D_z[:, None, None]

        # 2. Forward transforms
        # x: reshape to 2D so matmul hits BLAS gemm directly (avoids
        #    the 3D-batched overhead in numpy@ on small middle axes).
        #    Cost: 86×5×400² FLOPS ≈ 1 ms vs ~5 ms for batched matmul.
        q_hat = (q.reshape(Nz * Ny, Nx) @ self.U_x).reshape(Nz, Ny, Nx)
        # y: FFT along axis=1
        q_hat = np.fft.fft(q_hat, axis=1)
        # z: U_z.T @ q along axis=0.  Reshape (Nz, Ny*Nx) so this is a
        #    single 2D matmul instead of einsum — einsum was ~34 ms;
        #    BLAS gemm here is ~1 ms.
        q_hat = (self.U_z.T @ q_hat.reshape(Nz, Ny * Nx)).reshape(Nz, Ny, Nx)

        # 3. Spectral divide
        q_hat = q_hat * self.lambda_total_inv

        # 4. Inverse transforms (reverse order; U is orthogonal so U^{-1}=U^T)
        # z back: U_z @ q
        q = (self.U_z @ q_hat.reshape(Nz, Ny * Nx)).reshape(Nz, Ny, Nx)
        # y back: IFFT
        q = np.fft.ifft(q, axis=1).real
        # x back: reshape→matmul→reshape (same gain as forward x).
        q = (q.reshape(Nz * Ny, Nx) @ self.U_x.T).reshape(Nz, Ny, Nx)

        # 5. De-symmetrize: p = D^{-1} · q
        return q * self._D_z_inv[:, None, None]
