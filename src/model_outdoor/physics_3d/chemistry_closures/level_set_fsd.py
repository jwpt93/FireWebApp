"""Phase 15D — per-cell hybrid EDC + FSD chemistry closure.

Closure interface for the hybrid kernel in
:mod:`model_outdoor.physics_3d.level_set_fsd_3d`.  See that module's
docstring for kernel physics, lit refs, and Rule #10 notes on ``s_L``.

History
-------
- Phase 15C (initial FSD landing): pure FSD with a state-aware three-mode
  c-builder (no_fire / ignition floating-threshold T_g / established
  phi_flame).  Mesh L0→E spread 24.4% (WARN) — most of the spread came
  from the ignition-mode T_g-based c-field which is itself mesh-dependent.
- Phase 15D (this revision): replace the three-mode c-builder with a
  per-cell hybrid kernel.  c is always phi_flame; the kernel decides
  per-cell whether to apply FSD's surface-integral rate (where
  ``phi_flame > 0``) or EDC's T-independent Magnussen rate (where
  ``phi_flame == 0``).  EDC handles cold-start ignition cleanly without
  the chicken-egg problem since Magnussen has no T_g threshold.

Per-cell hybrid
~~~~~~~~~~~~~~~

::

    if phi_flame[cell] > 0:
        ω = ρ · s_L · |∇c_smooth| · f_avail        (FSD branch — mesh-stable)
    else:
        ω = γ* · ρ · min(Y_F, Y_O2/s) / τ*          (EDC Magnussen branch)

with the same downstream ODE update for Y_F, Y_O2, T_g.  ``c_smooth`` is
the 3D box-filtered ``phi_flame``; ``|∇c_smooth|`` is the FSD surface
density Σ.

Why this resolves the ignition chicken-egg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EDC's Magnussen rate fires whenever ``Y_F > 0 AND Y_O2 > 0 AND k > 0``,
regardless of T_g.  At cold start (phi_flame = 0 everywhere):

  drip-torch → solid pyrolysis → Y_F injected into gas → EDC branch fires
  → T_g rises via chemistry self-heating → eventually crosses 1000K
  → phi_flame triggers → those cells switch to FSD branch (mesh-stable).

Leading edge of the propagating front always has ``phi_flame = 0`` for
one cell layer (just-ignited cells before they meet the OR-of-conditions).
EDC fires there for one outer step; next step phi_flame triggers and FSD
takes over.  Mesh-dependence of the leading-edge EDC layer is bounded
because the layer is 1-cell thick at any given time.

F-option (Phase 15D-F): compute c_grad_norm once per outer step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production main loop pre-computes ``c_grad_norm`` once per outer step
via :func:`level_set_fsd_3d.compute_c_grad_norm_from_phi_flame` and
passes the array as the ``c_grad_norm`` kwarg to the closure.  The
closure skips its internal smoothing + gradient (~10× perf saving
without changing physics).  If ``c_grad_norm`` is NOT supplied, the
closure builds it internally (so unit tests and one-off uses still work).
"""
from __future__ import annotations

import numpy as np

from .. import level_set_fsd_3d


# Laminar flame speed default.  Rule #10 — calibration to Cheney NOT
# allowed.  Acceptable to document Rule #4 limitation if the lit value
# gives off-band ROS.
S_L_DEFAULT = 0.4    # [m/s]  Williams 1985 hydrocarbon-air, ϕ≈1


def run(
    *,
    rho: np.ndarray,
    T_g: np.ndarray,
    Y_fuel: np.ndarray,
    Y_O2: np.ndarray,
    phi_flame: np.ndarray,
    k_turb: np.ndarray,
    eps_turb: np.ndarray,
    dx: float,
    dy: float,
    dz_arr: np.ndarray,
    chi_rad: float,
    cp_g: float,
    dt: float,
    n_substeps: int,
    omega_out: np.ndarray,
    c_grad_norm: np.ndarray | None = None,
    s_L: float = S_L_DEFAULT,
    smoothing_iters: int = level_set_fsd_3d.SMOOTHING_ITERS_DEFAULT,
    Y_F_unb: float = level_set_fsd_3d.Y_F_UNB_DEFAULT,
    Y_O2_unb: float = level_set_fsd_3d.Y_O2_UNB_DEFAULT,
    use_turbulent_s_T: bool = False,   # Phase 15G — Damköhler 1
    s_T_cap_factor: float = 5.0,
    tfm_xi: float = 1.0,               # Phase 15H — Charlette 2002 wrinkling Ξ
    inner_body_edc: bool = False,      # Phase 15J — Linn 2002 FIRETEC-style
    **_unused,
) -> None:
    """Pluggable-closure entry point for the per-cell hybrid EDC + FSD closure.

    Steps each call:
      1. If ``c_grad_norm`` was not supplied: build ``c = phi_flame``,
         smooth, and compute |∇c| (single per-step cost; main loop should
         provide ``c_grad_norm`` precomputed once per outer step — see
         module docstring §F-option).
      2. Apply the hybrid kernel: per cell selects FSD or EDC branch
         from ``phi_flame``; both branches share the same ODE update of
         Y_F / Y_O2 / T_g.

    Required kwargs (all closures):
        rho, T_g, Y_fuel, Y_O2, dx, dy, dz_arr, chi_rad, cp_g, dt,
        n_substeps, omega_out

    FSD-specific kwargs:
        phi_flame, k_turb, eps_turb

    Optional / overrides:
        c_grad_norm, s_L, smoothing_iters, Y_F_unb, Y_O2_unb

    Phase 15G optional:
        use_turbulent_s_T, s_T_cap_factor — Damköhler 1: s_T = s_L·(1+u'/s_L)

    Phase 15H optional:
        tfm_xi — Charlette 2002 sub-grid wrinkling factor:
                 ω_TFM = Ξ · ρ · s_L · |∇c| · f_av  (FSD branch only)

    Phase 15J optional:
        inner_body_edc — when True, route inner-body cells through EDC
                          rather than FSD.  Matches Linn 2002 FIRETEC /
                          Mell 2007 WFDS mixing-limited fast-chemistry
                          practice; level-set v_n still tracks the front.

    Extra kwargs not consumed (e.g., tau_mix, omega_O2, omega_max_T)
    are silently ignored — see :mod:`chemistry_closures._interface`.
    """
    if c_grad_norm is None:
        c_grad_norm = level_set_fsd_3d.compute_c_grad_norm_from_phi_flame(
            phi_flame, dx, dy, dz_arr, smoothing_iters=smoothing_iters,
        )
    level_set_fsd_3d.step_hybrid_edc_fsd_chemistry(
        rho, T_g, Y_fuel, Y_O2,
        c_grad_norm,
        phi_flame.astype(np.float64) if phi_flame.dtype != np.float64 else phi_flame,
        k_turb, eps_turb,
        chi_rad, cp_g,
        s_L, Y_F_unb, Y_O2_unb,
        dt, n_substeps, omega_out,
        use_turbulent_s_T, s_T_cap_factor,
        tfm_xi,
        inner_body_edc,
    )
