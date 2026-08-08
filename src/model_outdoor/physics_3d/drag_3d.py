"""Porous-medium drag for cylindrical fuel elements (Ergun two-term).

For cylindrical fuel elements (grass blades, twigs) the full Ergun
(1952) drag has TWO components — viscous (linear in u, Darcy regime)
and Forchheimer (quadratic in u, inertial regime):

    F_drag = -K_visc · μ · σ² · α_s² / (1-α_s)³ · u        (Darcy/viscous)
             -C_D · σ · α_s · 0.5 · ρ · |u| · u            (Forchheimer/quadratic)

where σ [1/m] is the fuel surface-area-to-volume ratio, α_s [-] the
local solid volume fraction, ρ [kg/m³] the gas density, μ the gas
dynamic viscosity, and u the gas velocity vector.

The viscous term was previously omitted (Phase 13.B–13.S).  This left
low-velocity flow in dense beds (e.g. Cheney 1993 Cut grass at U=0.5)
essentially undamped — the quadratic term scales as |u|² and vanishes
faster than linear at small u, so combustion-driven recirculation in
the bed found no resistance.  Real Ergun-Carman analysis of fiber
beds (Tomadakis & Robertson 2005 J. Compos. Mater. 39:163; Pruyn et
al. 2018 Combust. Flame 187:182) shows the viscous term dominates
below Re ~ 10 and is what physically throttles low-Re flow through
compacted vegetation.

For an *isolated* cylinder at moderate Re (10 < Re < 10⁴), C_D ≈ 1.0
(Schlichting Boundary Layer Theory).  But for a *canopy* of cylinders
mutual sheltering and wake interaction reduce the effective form-drag
coefficient to 0.20–0.50.  Phase 14w-E correction: previously we used
the single-cylinder C_D = 1.0 here, mis-citing Morvan & Dupuy 2001;
the canopy values cited in the wildland-fire CFD literature are:
    Wilson & Shaw (1977) Bound.-Layer Meteorol. 13:419   C_D = 0.30
    Massman (1997) Bound.-Layer Meteorol. 83:407          0.25–0.50
    Lalic (2004) Bound.-Layer Meteorol. 113:99            0.20–0.30
    Morvan & Dupuy (2004) Combust. Flame 138:199 §3.1     0.40
    Morvan (2009) Fire Tech. 45:447 §2.3.2                0.40
    FIRETEC (Linn 2002), WFDS (Mell 2007)                ~0.3–0.5
We adopt C_D = 0.30 (Wilson & Shaw 1977 grass canopy; matches Lalic
2004 lower bound; below Massman / MD2004 / FIRETEC mid-range).
Effect: in-bed drag at U=2 m/s drops ~3.3× relative to C=1.0,
restoring the canopy-flow regime that Phase 14w propagation diagnostic
showed was being over-pinned (u_mid_bed ≈ 0 instead of physical
~0.4·U from Cionco-style attenuation).

Pressure drop over a bed of length L for uniform inlet velocity U:
    ΔP = [K_visc·μ·σ²·α_s²/(1-α_s)³ + ½·C_D·σ·α_s·ρ·U] · U · L
At Re ≫ 1 the viscous term is negligible and the test in
test_3d_components.py:test_b2_drag_pressure_drop continues to pass
unchanged (its inlet velocity puts it in the Forchheimer regime).

References:
- Ergun, S. (1952) Chem. Eng. Prog. 48:89 — original two-term drag
- Wilson & Shaw (1977) Bound.-Layer Meteorol. 13:419 — canopy C_D = 0.30
- Massman (1997) Bound.-Layer Meteorol. 83:407 — canopy drag survey
- Lalic (2004) Bound.-Layer Meteorol. 113:99 — vegetation drag survey
- Morvan & Dupuy (2004) Combust. Flame 138:199 §3.1 — FIRESTAR canopy C_D
- Morvan (2009) Fire Tech. 45:447 — FIRESTAR
- Tomadakis & Robertson (2005) J. Compos. Mater. 39:163 — fiber-bed Darcy
- Linn et al. (2002) IJWF 11:233 — vegetation drag in FIRETEC
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# Canopy form-drag coefficient — DEFAULT.  Used by callers that don't
# pass an explicit value (legacy + tests).  Production callers pass
# `outdoor.canopy_C_d` from the deck (Phase 14at re-added 2026-05-30).
#
# Literature values:
#   Wilson & Shaw (1977) Bound.-Layer Meteorol. 13:419   C_D = 0.30
#     (canonical dense-canopy default)
#   Lalic et al. (2004)                                  C_D = 0.20
#   Massman (1997) Bound.-Layer Meteorol. 83:407         C_D = 0.10–0.30
#   Mueller (2021) Agric. For. Meteorol. 311:108691      C_D = 0.15
#     (specifically calibrated for pasture grass)
#   Morvan & Dupuy (2004) Combust. Flame 138:199         C_D = 0.50
#     (FIRESTAR mixed shrub canopy)
C_D_DEFAULT = 0.30
# Backward-compat alias for legacy imports (tests reading the symbol
# directly).  New code should use `C_D_DEFAULT` and pass C_D explicitly
# to step_drag_force().
C_D = C_D_DEFAULT
# Gas dynamic viscosity (air at ~600 K, Drysdale 2011 Table 2.4).
MU_GAS = 3.0e-5     # [Pa·s]
# Ergun viscous prefactor for fiber/grass beds.  Standard Ergun (1952)
# packed-sphere coefficient is 150; converting to SAV-based form for
# cylindrical particles (σ=4/d_p) gives 150/16 = 9.375.  Tomadakis &
# Robertson (2005) measured fiber beds and found 5–10× higher viscous
# drag than Carman-Kozeny predicts due to anisotropic packing — we use
# 50.0 as a midpoint of their fiber-bed regression (Table 2 values
# converted to σ-based form).  This makes the Darcy term comparable
# to the Forchheimer term at u ~ 0.05 m/s in dense Cut grass, the
# regime where compaction-induced flow throttling is observed in
# Cheney (1993) low-wind cut-grass burns.
ERGUN_VISC_K = 50.0


@njit(cache=True, parallel=True)
def step_drag_force(
    u: np.ndarray,             # (Nz, Ny, Nx) [m/s]
    v: np.ndarray,
    w: np.ndarray,
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    alpha_s: np.ndarray,       # (Nz, Ny, Nx) [-]
    sigma_sav: float,          # [1/m] fuel SAV
    Fx_out: np.ndarray,        # (Nz, Ny, Nx) [N/m³] (overwritten)
    Fy_out: np.ndarray,
    Fz_out: np.ndarray,
    C_D: float,                # form-drag coefficient (deck: canopy_C_d)
) -> None:
    """Compute volumetric drag force vector per cell (Ergun two-term).

    Output arrays Fx_out, Fy_out, Fz_out are overwritten.  Values are
    zero where alpha_s == 0 (no fuel, no drag).

    C_D is now a function argument (Phase 14at re-added 2026-05-30) —
    callers should pass `outdoor.canopy_C_d` from the deck.  See the
    `C_D_DEFAULT` module constant for the literature default (0.30).
    """
    Nz, Ny, Nx = u.shape
    sigma2 = sigma_sav * sigma_sav
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                a = alpha_s[k, j, i]
                if a <= 0.0:
                    Fx_out[k, j, i] = 0.0
                    Fy_out[k, j, i] = 0.0
                    Fz_out[k, j, i] = 0.0
                    continue
                ui = u[k, j, i]; vi = v[k, j, i]; wi = w[k, j, i]
                speed = (ui * ui + vi * vi + wi * wi) ** 0.5
                # Viscous (Darcy/Ergun) coefficient: K·μ·σ²·α²/(1-α)³
                one_m_a = 1.0 - a
                K_visc = (ERGUN_VISC_K * MU_GAS * sigma2 * a * a
                          / (one_m_a * one_m_a * one_m_a))
                # Forchheimer (quadratic) coefficient: ½·C_D·σ·α·ρ·|u|
                K_quad = C_D * sigma_sav * a * 0.5 * rho[k, j, i] * speed
                # F_drag = -(K_visc + K_quad) · u
                coeff = -(K_visc + K_quad)
                Fx_out[k, j, i] = coeff * ui
                Fy_out[k, j, i] = coeff * vi
                Fz_out[k, j, i] = coeff * wi
