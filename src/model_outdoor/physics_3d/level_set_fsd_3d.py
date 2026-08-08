"""Phase 15C — Flame Surface Density (FSD) level-set chemistry kernels.

Reaction rate is driven by flame surface area density Σ ≈ |∇c|, where c
is a smoothed progress variable derived from `phi_flame` (the existing
binary flame-zone marker from :mod:`flame_front_3d`).  Cell-averaged
fuel-consumption rate:

    ω_cell = ρ · s_L · |∇c_smooth| · f_avail(Y_F, Y_O2)        [kg/m³/s]

where ``f_avail`` is a fuel-and-oxidizer availability gate that masks
out plume-back, top, and side surfaces (where the smoothed phi_flame
still has gradient but no fuel left to burn):

    f_avail = min( Y_F / Y_F_unb,  Y_O2 / Y_O2_unb,  1 )

with reference unburnt-stoichiometric values

    Y_F_unb  = Z_st · Y_F_stream                 (≈ 0.1514)
    Y_O2_unb = (1 - Z_st) · Y_O2_amb             (≈ 0.1969)
    Z_st     = Y_O2_amb / (S_STOICH + Y_O2_amb)  (≈ 0.1514)

This formulation is mesh-stable by the coarea identity
∫ |∇c| dV ≈ flame surface area (m²), independent of cell size.  Offline
verification on the L0→E refinement series (commit prior; see
``memory/phase15a_day3_kill_test_results.md``): FSD ratio range
1.066 – 1.196 across 48 (snapshot-time × smoothing-iters × band-width)
parameter combinations.  EDC's ratio range was 2.9 – 3.1×.

Literature
----------
- Boger, M., Veynante, D., Boughanem, H. & Trouvé, A. (1998)
  "Direct numerical simulation analysis of flame surface density
  concept for large-eddy simulation of turbulent premixed combustion"
  Symp. (Int.) Combust. 27:917-925.
- Cant, R.S. & Pope, S.B. (1990) "Modelling of flamelet surface-to-volume
  ratio in turbulent premixed combustion" Symp. Combust. 23:809.
- Williams, F.A. (1985) "Turbulent combustion" Combust. Sci. Tech.
  41:235 — laminar flame speed s_L for hydrocarbon-air.
- Drysdale (2011) Fire Dynamics 3rd ed. §1.2.3 + Tab 1.13 — grass-flame
  temperature ~1500-1800 K (for inferred s_L of cellulose volatile).
- Magnussen (1981) AIAA-81-0042 — the EDC closure being replaced.

Rule #10 note: ``s_L = 0.4 m/s`` (Williams 1985 hydrocarbon-air) is the
literature default exposed via deck flag ``s_L_volatile_m_s``.  Cellulose
volatile flames at canopy density may sustain an effectively lower
turbulent flame speed; if Cheney calibration requires substantially
lower s_L, this is documented as a Rule #4 model limitation, NOT a
free-parameter tuning.

Phase 15-0 closure-registry framework integration: this kernel is wrapped
by :mod:`chemistry_closures.level_set_fsd`.run; do not call directly
from the main loop.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .chemistry_closures._constants import S_STOICH, Y_O2_AIR, HOC_J
# EDC Magnussen constants re-used for the per-cell hybrid kernel below
# (Phase 15D-C — EDC fallback where phi_flame is 0; FSD elsewhere).
from .chemistry_closures.edc import (
    C_GAMMA_EDC,
    C_TAU_EDC,
    NU_GAS_EDC,
    K_TURB_FLOOR_EDC,
    EPS_TURB_FLOOR_EDC,
)


# Reference unburnt-stoichiometric concentrations for the availability
# gate.  Derived from the project's canonical mixture-fraction Z_st.
# Used as defaults; deck can override via closure kwargs.
Z_ST                = Y_O2_AIR / (S_STOICH + Y_O2_AIR)      # 0.1514
Y_F_UNB_DEFAULT     = Z_ST * 1.0                            # Y_F_stream = 1.0 (lumped)
Y_O2_UNB_DEFAULT    = (1.0 - Z_ST) * Y_O2_AIR               # 0.1969

# Box-filter smoothing iterations (default).  Offline robustness sweep
# showed near-zero sensitivity to this in [1, 7]; pick 3 for moderate
# regularization.
SMOOTHING_ITERS_DEFAULT = 3


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing — three 1D 1-2-1 box-filter passes, n_iters times each.
# Boundary cells use zero-gradient (replicate) BC.
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def _box_pass_x(src: np.ndarray, dst: np.ndarray) -> None:
    """1-2-1 box-filter pass along x.  dst[k,j,i] = (src[k,j,i-1] + 2·src[k,j,i] + src[k,j,i+1])/4."""
    Nz, Ny, Nx = src.shape
    for k in prange(Nz):
        for j in range(Ny):
            dst[k, j, 0]      = src[k, j, 0]   # replicate
            dst[k, j, Nx - 1] = src[k, j, Nx - 1]
            for i in range(1, Nx - 1):
                dst[k, j, i] = 0.25 * (
                    src[k, j, i - 1] + 2.0 * src[k, j, i] + src[k, j, i + 1]
                )


@njit(cache=True, parallel=True)
def _box_pass_y(src: np.ndarray, dst: np.ndarray) -> None:
    """1-2-1 box-filter pass along y."""
    Nz, Ny, Nx = src.shape
    for k in prange(Nz):
        for i in range(Nx):
            dst[k, 0, i]      = src[k, 0, i]
            dst[k, Ny - 1, i] = src[k, Ny - 1, i]
            for j in range(1, Ny - 1):
                dst[k, j, i] = 0.25 * (
                    src[k, j - 1, i] + 2.0 * src[k, j, i] + src[k, j + 1, i]
                )


@njit(cache=True, parallel=True)
def _box_pass_z(src: np.ndarray, dst: np.ndarray) -> None:
    """1-2-1 box-filter pass along z (uniform weights — see module docstring
    note on non-uniform-dz handling: smoothing is regularization, not
    conservation, so uniform stencil is acceptable)."""
    Nz, Ny, Nx = src.shape
    for j in prange(Ny):
        for i in range(Nx):
            dst[0, j, i]      = src[0, j, i]
            dst[Nz - 1, j, i] = src[Nz - 1, j, i]
            for k in range(1, Nz - 1):
                dst[k, j, i] = 0.25 * (
                    src[k - 1, j, i] + 2.0 * src[k, j, i] + src[k + 1, j, i]
                )


def smooth_phi_flame(
    phi: np.ndarray,
    n_iters: int = SMOOTHING_ITERS_DEFAULT,
    scratch_a: np.ndarray | None = None,
    scratch_b: np.ndarray | None = None,
) -> np.ndarray:
    """Return a smoothed copy of ``phi`` (3D box-filter, ``n_iters`` passes).

    Allocates two scratch buffers internally if not supplied.  The pinned
    contiguity is required for the @njit kernels.
    """
    if scratch_a is None:
        scratch_a = np.empty_like(phi, dtype=np.float64)
    if scratch_b is None:
        scratch_b = np.empty_like(phi, dtype=np.float64)
    if not np.issubdtype(phi.dtype, np.floating):
        phi_f = phi.astype(np.float64)
    else:
        phi_f = phi.astype(np.float64, copy=True)

    src = phi_f
    dst = scratch_a
    other = scratch_b
    for _ in range(n_iters):
        _box_pass_x(src, dst)
        _box_pass_y(dst, other)
        _box_pass_z(other, dst)
        src = dst
        dst = other
        other = src if src is scratch_a else scratch_a
    # `src` holds the final result.
    if src is scratch_a:
        return scratch_a
    if src is scratch_b:
        return scratch_b
    return src


# ─────────────────────────────────────────────────────────────────────────────
# Non-uniform-dz |∇c| via central differences.
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def compute_grad_norm_nonuniform(
    field: np.ndarray,
    dx: float,
    dy: float,
    dz_arr: np.ndarray,
    grad_norm_out: np.ndarray,
) -> None:
    """Write |∇field| into ``grad_norm_out`` using non-uniform-dz central
    differences in z and uniform central differences in x, y.  Boundaries
    fall back to one-sided differences.

    Mirrors the Python-level reference implementation used in the
    Phase 15A day-3 offline kill test (``scripts/diagnostics/
    run_chi_st_offline_test.py``).
    """
    Nz, Ny, Nx = field.shape
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_dx  = 1.0 / dx
    inv_dy  = 1.0 / dy

    for k in prange(Nz):
        # z spacing for central diff at this k
        if k == 0:
            dz_use = dz_arr[0]
            use_one_sided_z = True
        elif k == Nz - 1:
            dz_use = dz_arr[Nz - 1]
            use_one_sided_z = True
        else:
            dz_use = dz_arr[k] + 0.5 * (dz_arr[k - 1] + dz_arr[k + 1])
            use_one_sided_z = False
        inv_dz_z = 1.0 / dz_use

        for j in range(Ny):
            for i in range(Nx):
                # x
                if i == 0:
                    dfdx = (field[k, j, 1] - field[k, j, 0]) * inv_dx
                elif i == Nx - 1:
                    dfdx = (field[k, j, Nx - 1] - field[k, j, Nx - 2]) * inv_dx
                else:
                    dfdx = (field[k, j, i + 1] - field[k, j, i - 1]) * inv_2dx
                # y
                if j == 0:
                    dfdy = (field[k, 1, i] - field[k, 0, i]) * inv_dy
                elif j == Ny - 1:
                    dfdy = (field[k, Ny - 1, i] - field[k, Ny - 2, i]) * inv_dy
                else:
                    dfdy = (field[k, j + 1, i] - field[k, j - 1, i]) * inv_2dy
                # z
                if use_one_sided_z:
                    if k == 0:
                        dfdz = (field[1, j, i] - field[0, j, i]) * inv_dz_z
                    else:
                        dfdz = (field[Nz - 1, j, i] - field[Nz - 2, j, i]) * inv_dz_z
                else:
                    dfdz = (field[k + 1, j, i] - field[k - 1, j, i]) * inv_dz_z

                grad_norm_out[k, j, i] = math.sqrt(
                    dfdx * dfdx + dfdy * dfdy + dfdz * dfdz
                )


# ─────────────────────────────────────────────────────────────────────────────
# FSD reaction-rate ODE — explicit Euler operator-split (same structure as EDC).
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def step_fsd_chemistry(
    rho: np.ndarray,           # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,           # (Nz, Ny, Nx) [K]   updated in place
    Y_fuel: np.ndarray,        # (Nz, Ny, Nx) [-]   updated in place
    Y_O2: np.ndarray,          # (Nz, Ny, Nx) [-]   updated in place
    c_grad_norm: np.ndarray,   # (Nz, Ny, Nx) [1/m] |∇c_smooth|
    chi_rad: float,
    cp_g: float,
    s_L: float,                # [m/s] laminar flame speed
    Y_F_unb: float,            # [-] stoichiometric unburnt fuel mass fraction
    Y_O2_unb: float,           # [-] stoichiometric unburnt O₂ mass fraction
    dt: float,
    n_substeps: int,
    omega_int_out: np.ndarray, # (Nz, Ny, Nx) [kg/m³/s] time-averaged ω
) -> None:
    """FSD source applied via explicit Euler operator split.

    ω = ρ · s_L · |∇c_smooth| · min(Y_F/Y_F_unb, Y_O2/Y_O2_unb, 1)

    The availability gate `min(Y_F/Y_F_unb, Y_O2/Y_O2_unb, 1)` ensures
    ω → 0 in burnt/vitiated cells (where smoothed-c gradient may still
    be non-zero from the plume body) and in air cells (no fuel).

    Updates Y_F, Y_O2, T_g exactly as the EDC kernel does:
        ΔY_F  = -ω · h / ρ
        ΔY_O2 = -s · ω · h / ρ        (s = S_STOICH)
        ΔT_g  = +ω · HoC_eff · h / (ρ · cp_g)
    where HoC_eff = HoC · (1 − χ_rad) for the radiation budget.

    T_g capped at 2400 K to match EDC's hard cap.

    Rule #17: kernel uses no parallel reductions; each `(k,j,i)` cell
    updates are independent and bit-exact under thread re-scheduling.
    """
    Nz, Ny, Nx = rho.shape
    h = dt / max(n_substeps, 1)
    HoC_eff = HOC_J * (1.0 - chi_rad)
    inv_Y_F_unb  = 1.0 / Y_F_unb  if Y_F_unb  > 0.0 else 0.0
    inv_Y_O2_unb = 1.0 / Y_O2_unb if Y_O2_unb > 0.0 else 0.0

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Yf    = Y_fuel[k, j, i]
                YO2   = Y_O2[k, j, i]
                Tg    = T_g[k, j, i]
                rho_i = rho[k, j, i]
                grad  = c_grad_norm[k, j, i]

                if rho_i <= 0.0 or grad <= 0.0 or Yf <= 1e-9 or YO2 <= 1e-9:
                    omega_int_out[k, j, i] = 0.0
                    continue

                omega_acc = 0.0
                for _ in range(max(n_substeps, 1)):
                    if Yf <= 1e-9 or YO2 <= 1e-9:
                        omega = 0.0
                    else:
                        # Availability gate: min(Y_F/Y_F_unb, Y_O2/Y_O2_unb, 1)
                        a_f  = Yf  * inv_Y_F_unb
                        a_o  = YO2 * inv_Y_O2_unb
                        f_av = a_f if a_f < a_o else a_o
                        if f_av > 1.0:
                            f_av = 1.0
                        omega = rho_i * s_L * grad * f_av

                    dY  = -omega * h / rho_i
                    Yf  = Yf + dY
                    if Yf < 0.0:
                        Yf = 0.0
                    YO2 = YO2 + S_STOICH * dY
                    if YO2 < 0.0:
                        YO2 = 0.0
                    Tg  = Tg + omega * HoC_eff * h / (rho_i * cp_g)
                    if Tg > 2400.0:
                        Tg = 2400.0

                    omega_acc += omega * h

                Y_fuel[k, j, i] = Yf
                Y_O2[k, j, i]   = YO2
                T_g[k, j, i]    = Tg
                omega_int_out[k, j, i] = omega_acc / dt if dt > 0.0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 15D-F (precompute helper) — build c_grad_norm once per outer step
# so chemistry sub-step loop reuses the same array.  Saves ~N_SUB× chemistry
# kernel work without changing physics.
# ─────────────────────────────────────────────────────────────────────────────


def compute_c_grad_norm_from_phi_flame(
    phi_flame: np.ndarray,
    dx: float,
    dy: float,
    dz_arr: np.ndarray,
    smoothing_iters: int = SMOOTHING_ITERS_DEFAULT,
    scratch_a: np.ndarray | None = None,
    scratch_b: np.ndarray | None = None,
) -> np.ndarray:
    """Build |∇c_smooth| from phi_flame in a single call.

    Phase 15D bugfix (2026-06-06): ``phi_flame`` is a signed-distance
    field, not a [0,1] progress variable.  Convention from
    :func:`flame_front_3d.compute_phi_flame_from_state`:

      phi_flame ≤ 0  inside flame body  (active reaction zone)
      phi_flame > 0  outside flame body (positive distance to boundary)
      phi_flame = ±1e6 sentinel when there is no flame anywhere

    Smoothing the raw signed distance is meaningless (saturated ±1e6 cells
    swamp the gradient).  We binarize first into a [0,1] indicator, smooth
    that, then take the gradient.  The result is a proper flame-surface
    density estimate: ∫|∇c_smooth| dV ≈ flame surface area.

    Wraps :func:`smooth_phi_flame` + :func:`compute_grad_norm_nonuniform`
    for use from the outer time loop (Phase 15D-F: compute once per outer
    step, reuse across N_SUB chemistry sub-steps).

    Returns a fresh ``(Nz, Ny, Nx)`` float64 array of |∇c_smooth|.
    """
    # Binarize the signed-distance field into a [0,1] inside/outside
    # indicator BEFORE smoothing.  See docstring.
    c_binary = (phi_flame <= 0.0).astype(np.float64)
    c_smooth = smooth_phi_flame(
        c_binary, n_iters=smoothing_iters,
        scratch_a=scratch_a, scratch_b=scratch_b,
    )
    grad = np.empty_like(c_smooth, dtype=np.float64)
    compute_grad_norm_nonuniform(c_smooth, dx, dy, dz_arr, grad)
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Phase 15D-C — per-cell hybrid EDC + FSD chemistry step.
#
# Each cell selects its rate formula from phi_flame:
#   phi_flame > 0  →  FSD rate  ω = ρ · s_L · |∇c| · f_avail
#   phi_flame == 0 →  EDC Magnussen 1981 rate
#                     ω = γ* · ρ · min(Y_F, Y_O2/s) / τ*
#                     γ* = (C_γ · (ν·ε/k²)^¼)³, clamped at 1
#                     τ* = C_τ · (ν/ε)^½
#
# Rationale
# ~~~~~~~~~
# Pure FSD has a chicken-egg ignition problem (phi_flame requires high T
# AND/OR firing chemistry; FSD's omega depends on phi_flame).  The
# three-mode T_g-based bootstrap (Phase 15C, pre-15D) solves ignition but
# introduces 24% L0→E mesh-spread because the floating-threshold T_g-c
# formulation is mesh-dependent in shape.
#
# The per-cell hybrid avoids both:
#   • Cells without phi_flame (cold air / pre-ignition / leading edge):
#     EDC's T-independent Magnussen rate kickstarts and propagates.
#     Mesh-dependence of EDC localized to 1-cell-thick leading-edge layer.
#   • Cells with phi_flame (post-ignition flame body):
#     FSD's surface-integral rate (mesh-stable per coarea identity).
#
# This is the "EDC for ignition, FSD for steady propagation" design,
# implemented as a per-cell branch rather than a global mode switch.
# ─────────────────────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def step_hybrid_edc_fsd_chemistry(
    rho: np.ndarray,            # (Nz, Ny, Nx) [kg/m³]
    T_g: np.ndarray,            # (Nz, Ny, Nx) [K]    updated in place
    Y_fuel: np.ndarray,         # (Nz, Ny, Nx) [-]    updated in place
    Y_O2: np.ndarray,           # (Nz, Ny, Nx) [-]    updated in place
    c_grad_norm: np.ndarray,    # (Nz, Ny, Nx) [1/m]  |∇c_smooth| (from phi_flame)
    phi_flame: np.ndarray,      # (Nz, Ny, Nx) [-]    per-cell branch indicator
    k_turb: np.ndarray,         # (Nz, Ny, Nx) [m²/s²] for EDC branch
    eps_turb: np.ndarray,       # (Nz, Ny, Nx) [m²/s³] for EDC branch
    chi_rad: float,
    cp_g: float,                # [J/kg/K]
    s_L: float,                 # [m/s] FSD laminar flame speed
    Y_F_unb: float,             # [-]   FSD stoichiometric reference fuel mass fraction
    Y_O2_unb: float,            # [-]   FSD stoichiometric reference O₂ mass fraction
    dt: float,                  # [s]
    n_substeps: int,
    omega_int_out: np.ndarray,  # (Nz, Ny, Nx) [kg/m³/s] time-averaged ω
    use_turbulent_s_T: bool = False,  # Phase 15G — Damköhler 1: s_T = s_L·(1+u'/s_L)
    s_T_cap_factor: float = 5.0,       # Cap s_T at this multiple of s_L
    tfm_xi: float = 1.0,               # Phase 15H — Charlette 2002 sub-grid
                                        # wrinkling factor Ξ for the FSD rate.
                                        # ω_TFM = Ξ · ρ · s_L · |∇c| · f_av.
                                        # 1.0 (default) = unmodified back-compat.
                                        # Charlette 2002 Eq 35: Ξ ≈ (Δ/δ_L)^β,
                                        # β ≈ 0.3-0.5; for Δ=0.1 m, δ_L=3 mm
                                        # this gives Ξ ≈ 3-6 (covers our 5×
                                        # under-prediction).
    inner_body_edc: bool = False,      # Phase 15J — route inner-body cells
                                        # (phi_flame ≤ 0) through the EDC
                                        # branch instead of FSD.  Effective
                                        # chemistry then matches Linn 2002
                                        # FIRETEC / Mell 2007 WFDS practice:
                                        # mixing-limited fast chemistry
                                        # everywhere; the level-set front
                                        # tracker provides v_n kinematics.
                                        # Default False = current Phase 15D
                                        # hybrid behavior (FSD inside body).
) -> None:
    """Per-cell hybrid: EDC where phi_flame=0, FSD where phi_flame>0.

    Both branches share the same explicit-Euler ODE update for
    Y_F / Y_O2 / T_g.  Per-cell branch is deterministic and bit-exact
    under thread re-scheduling (Rule #17).

    EDC branch (phi_flame == 0):
        γ* = (C_γ · (ν·ε/k²)^¼)³ ≤ 1
        τ* = C_τ · (ν/ε)^½
        ω  = γ* · ρ · min(Y_F, Y_O2/s) / τ*

    FSD branch (phi_flame > 0):
        ω = ρ · s_L · |∇c_smooth| · min(Y_F/Y_F_unb, Y_O2/Y_O2_unb, 1)
    """
    Nz, Ny, Nx = rho.shape
    h = dt / max(n_substeps, 1)
    HoC_eff = HOC_J * (1.0 - chi_rad)
    inv_Y_F_unb  = 1.0 / Y_F_unb  if Y_F_unb  > 0.0 else 0.0
    inv_Y_O2_unb = 1.0 / Y_O2_unb if Y_O2_unb > 0.0 else 0.0

    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Yf    = Y_fuel[k, j, i]
                YO2   = Y_O2[k, j, i]
                Tg    = T_g[k, j, i]
                rho_i = rho[k, j, i]
                phi   = phi_flame[k, j, i]

                if rho_i <= 0.0 or Yf <= 1e-9 or YO2 <= 1e-9:
                    omega_int_out[k, j, i] = 0.0
                    continue

                # Phase 15D bugfix (2026-06-06): phi_flame is a signed
                # distance field per flame_front_3d.compute_phi_flame_from_state:
                #   phi_flame ≤ 0  → inside flame body  → use FSD
                #   phi_flame > 0  → outside flame body → use EDC kickstart
                # The prior phi > 0.0 test was inverted, routing every
                # cold-start cell (where phi_flame=+1e6 sentinel) to FSD,
                # which then gave ω=0 because c_grad_norm was 0 over the
                # uniform sentinel field.  See memory note
                # phase15d_phi_flame_sign_inversion_bug.md for the diagnosis.
                #
                # Phase 15J (2026-06-08): inner_body_edc forces EDC for ALL
                # cells (Linn 2002 / Mell 2007 mixing-limited practice);
                # level-set still tracks front kinematics separately.
                use_fsd = (phi <= 0.0) and (not inner_body_edc)

                if use_fsd:
                    grad = c_grad_norm[k, j, i]
                    # EDC vars unused; set to dummies to satisfy Numba
                    # type inference (kept inside the branch for clarity)
                    gamma_star = 0.0
                    tau_star = 1.0
                else:
                    # EDC Magnussen 1981 fine-structure rate
                    k_t = k_turb[k, j, i]
                    if k_t < K_TURB_FLOOR_EDC:
                        k_t = K_TURB_FLOOR_EDC
                    e_t = eps_turb[k, j, i]
                    if e_t < EPS_TURB_FLOOR_EDC:
                        e_t = EPS_TURB_FLOOR_EDC
                    ratio = NU_GAS_EDC * e_t / (k_t * k_t)
                    if ratio < 1e-30:
                        ratio = 1e-30
                    gamma_star = (C_GAMMA_EDC * ratio ** 0.25) ** 3
                    if gamma_star > 1.0:
                        gamma_star = 1.0
                    tau_star = C_TAU_EDC * (NU_GAS_EDC / e_t) ** 0.5
                    grad = 0.0

                omega_acc = 0.0
                for _ in range(max(n_substeps, 1)):
                    if Yf <= 1e-9 or YO2 <= 1e-9:
                        omega = 0.0
                    elif use_fsd:
                        # FSD rate.  Phase 15G optional Damköhler 1
                        # turbulent flame speed s_T = s_L·(1 + u'/s_L)
                        # with u' = √(2k/3) (RMS velocity fluctuation).
                        # Capped at s_T_cap_factor × s_L to avoid runaway
                        # in high-k cells.  When False, falls back to
                        # laminar s_L (back-compat).
                        a_f  = Yf  * inv_Y_F_unb
                        a_o  = YO2 * inv_Y_O2_unb
                        f_av = a_f if a_f < a_o else a_o
                        if f_av > 1.0:
                            f_av = 1.0
                        if use_turbulent_s_T:
                            k_local = k_turb[k, j, i]
                            if k_local < 0.0:
                                k_local = 0.0
                            u_prime = math.sqrt(2.0 * k_local / 3.0)
                            s_eff = s_L + u_prime
                            s_cap = s_L * s_T_cap_factor
                            if s_eff > s_cap:
                                s_eff = s_cap
                        else:
                            s_eff = s_L
                        omega = rho_i * s_eff * grad * f_av * tfm_xi
                    else:
                        # EDC rate
                        Y_lim = Yf if Yf < (YO2 / S_STOICH) else (YO2 / S_STOICH)
                        omega_fine = rho_i * Y_lim / tau_star
                        omega = gamma_star * omega_fine

                    # Shared ODE update
                    dY = -omega * h / rho_i
                    Yf = Yf + dY
                    if Yf < 0.0:
                        Yf = 0.0
                    YO2 = YO2 + S_STOICH * dY
                    if YO2 < 0.0:
                        YO2 = 0.0
                    Tg = Tg + omega * HoC_eff * h / (rho_i * cp_g)
                    if Tg > 2400.0:
                        Tg = 2400.0

                    omega_acc += omega * h

                Y_fuel[k, j, i] = Yf
                Y_O2[k, j, i]   = YO2
                T_g[k, j, i]    = Tg
                omega_int_out[k, j, i] = omega_acc / dt if dt > 0.0 else 0.0
