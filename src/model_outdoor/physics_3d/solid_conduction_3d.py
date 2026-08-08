"""Vertical solid-side conduction in the outdoor 3D porous-media model.

Physical motivation
-------------------
A single grass blade is a continuous solid spanning many vertical Eulerian
cells (base at z=0, tip at z=h_bed).  Heat absorbed at the tip (radiation
from the overhead flame, hot-gas advection arriving at the bed top)
conducts down the blade and warms the base.  Without solid-side conduction
each cell's T_s is independent and the heating at the tip cannot
propagate down — leaving the bulk of the bed cold while the top crust
heats and pyrolyzes alone.

We model this as 1-D vertical conduction along the blade axis:

    ρ_s · cp_s · α_s · ∂T_s/∂t = ∂/∂z ( k_s · α_s · ∂T_s/∂z )

where α_s is the local solid volume fraction (which is also the
cross-sectional area fraction of solid through a horizontal slice of the
cell, because grass blades are tall thin prisms).  Horizontal conduction
between cells is neglected — adjacent grass plants are not thermally
connected.

References
----------
* Petrich (2008) Bioresour. Tech. 99:8093 — biomass thermal conductivity
  k_s ≈ 0.10–0.30 W/m/K for dry plant material at moderate T.
* Fons (1946) J. Agric. Res. 72:93 — first analytical "stick" model of
  vertical conduction in burning grass blade.
* Spalding (1963) — fin-conduction grass-blade closure.
* Drysdale (2011) §2.3 — biomass cp ≈ 1300–1500 J/kg/K.

The 3-D model is a fin-conduction surrogate at the Eulerian cell level:
T_s for the cell aggregates many blades, k_s · α_s is the effective
cross-sectional conductance.

Determinism (Rule #17)
----------------------
Double-buffer pattern: read from ``T_s`` (old buffer), write to
``T_s_new`` (new buffer), copy back at end.  Each prange iteration writes
to a unique (k,j,i) cell — no cross-thread reductions.  Bit-exact at any
thread count.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# Grass-blade thermal conductivity [W/m/K].  Cured pasture grass (Petrich
# 2008 fits, Drysdale 2011 §2.3 plant-material survey).  Constant along
# the blade; not temperature-dependent in this ROM.
K_SOLID_GRASS = 0.20


@njit(cache=True, parallel=True)
def step_solid_conduction_vertical(
    T_s: np.ndarray,        # (Nz, Ny, Nx) IN-PLACE update
    alpha_s: np.ndarray,    # (Nz, Ny, Nx)
    dz_arr: np.ndarray,     # (Nz,) cell thickness
    d_face_above: np.ndarray,   # (Nz,) face-to-face dz, k → k+1
    d_face_below: np.ndarray,   # (Nz,) face-to-face dz, k → k-1
    k_solid: float,
    rho_solid: float,
    cp_solid: float,
    dt: float,
) -> None:
    """Explicit-Euler vertical conduction for the solid.

    Discretization (per cell (k,j,i) where α_s>0):

        flux_up   = k_s · α_s_face_up   · (T_s[k+1] − T_s[k]) / d_face_above[k]
        flux_down = k_s · α_s_face_down · (T_s[k-1] − T_s[k]) / d_face_below[k]
        ΔT_s/Δt = (flux_up + flux_down) / (dz_arr[k] · ρ_s · cp_s · α_s[k])

    Face α_s is the harmonic mean of the two adjacent cell values, so
    cells with α_s=0 (gas) decouple from the solid stack.  The first
    cell above the bed (α_s=0) becomes a no-flux boundary automatically.

    Stability: explicit Euler ⇒ Fourier number F_z = k·dt/(ρ·cp·dz²) ≤ 0.5.
    For k=0.2, ρ=500, cp=1300, dz=0.0925 ⇒ Fmax dt ≈ 144 s.  Outer dt
    from CFL is ≪ this, so explicit is stable.
    """
    Nz, Ny, Nx = T_s.shape
    T_s_new = T_s.copy()
    # Inverse heat capacity factor per cell: 1/(ρ·cp)
    inv_rho_cp = 1.0 / (rho_solid * cp_solid)
    for j in prange(Ny):
        for i in range(Nx):
            for k in range(Nz):
                a_s_c = alpha_s[k, j, i]
                if a_s_c <= 0.0:
                    continue
                # Face above (k → k+1)
                if k < Nz - 1:
                    a_s_a = alpha_s[k + 1, j, i]
                    if a_s_a > 0.0:
                        # harmonic-mean α_s at the face
                        a_eff_up = 2.0 * a_s_c * a_s_a / (a_s_c + a_s_a)
                        flux_up = (
                            k_solid * a_eff_up
                            * (T_s[k + 1, j, i] - T_s[k, j, i])
                            / d_face_above[k]
                        )
                    else:
                        flux_up = 0.0
                else:
                    flux_up = 0.0
                # Face below (k → k-1)
                if k > 0:
                    a_s_b = alpha_s[k - 1, j, i]
                    if a_s_b > 0.0:
                        a_eff_dn = 2.0 * a_s_c * a_s_b / (a_s_c + a_s_b)
                        flux_dn = (
                            k_solid * a_eff_dn
                            * (T_s[k - 1, j, i] - T_s[k, j, i])
                            / d_face_below[k]
                        )
                    else:
                        flux_dn = 0.0
                else:
                    flux_dn = 0.0
                # Volumetric source [W/m³] divided by (ρ·cp·α_s) → K/s
                src = (flux_up + flux_dn) / dz_arr[k]
                dT = dt * src * inv_rho_cp / a_s_c
                T_s_new[k, j, i] = T_s[k, j, i] + dT
    # Copy new buffer back to T_s (Rule #17 double-buffer)
    T_s[:] = T_s_new
