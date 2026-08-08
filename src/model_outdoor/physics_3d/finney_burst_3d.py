"""Phase 15N — Finney 2015 parameterized burst-convective preheat closure.

Adds a sub-grid forward convective heat-flux mechanism to ahead-band cells,
parameterized after Finney et al. 2015 PNAS 112(32):9833 "Role of buoyant
flame dynamics in wildfire spread."

PHYSICAL BASIS
~~~~~~~~~~~~~~

Finney 2015 PNAS measured intermittent flame-tongue contact on fine fuel
ahead of a propagating flame front.  Key measurements (Finney 2015 §Results):

  - Steady radiation: ≤ 30 kW/m² preheating fine particles
  - Intermittent flame contact: fine-particle heating rate ≈ 2,500 °C/s
  - Ratio: ~500× the radiation-only heating rate
  - Mechanism: buoyancy instability (Strouhal-Froude scaling holds across
    lab cardboard cribs and Texas grass field tests)

The instability frequency is 1-10 Hz with forward reach scaling as ~0.3-1.0
× Byram flame length L_F.  RANS k-ε cannot resolve this directly (time-
averaging window > 1/f); we parameterize the time-averaged forward flux
as a smooth exponential decay from the front:

    q_burst_conv(d) = q_0 · exp(-d / L_burst) · gate(I_fire)

where d is the +x distance from the cell center to the nearest flame-body
edge (phi_flame ≤ 0) in that y-row.  The gate enforces zero contribution
when there is no flame body upstream.

PARAMETER VALUES (Phase 15N initial commit, Rule #1 lit-anchored)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Q_0_DEFAULT     = 100_000.0  W/m²   — Finney 2015 §Results, conservative
                                        end of measured fluxes for fine
                                        grass-scale fuels at 2,500 °C/s
                                        (back-converted via ρ_p·c_p·δ).
  L_BURST_DEFAULT = 0.30        m     — Strouhal-Froude scaling for
                                        Cheney Nat 4% U=4 m/s with Byram
                                        L_F ≈ 1 m: L_burst ≈ 0.3 m.
  D_MAX_CUTOFF    = 1.0         m     — beyond ~3·L_burst the parameterized
                                        contribution is negligible
                                        (numerical floor).

These values are committed BEFORE any production runs.  Rule #2 forbids
fishing them into the calibration window; if Cheney does not match with
these defaults, the result is a documented Rule #4 limitation, not a
re-tuning trigger.

USAGE
~~~~~

The closure is OFF by default in production decks (Rule #10).  To enable
on a per-run basis, pass ``finney_burst_enable=True`` to
``run_3d_spread``.  The deck must explicitly opt in; defaults stay
zero-effect.

References
~~~~~~~~~~

- Finney, M.A., Cohen, J.D., Forthofer, J.M., McAllister, S.S., Gollner,
  M.J., Gorham, D.J., Saito, K., Akafuah, N.K., Adam, B.A., English, J.D.
  (2015) "Role of buoyant flame dynamics in wildfire spread,"
  Proc. Nat. Acad. Sci. 112(32):9833-9838.
- Cohen, J.D., Finney, M.A. (2010) "An examination of flame shape related
  to convection heat transfer in deep-fuel beds," USFS RMRS proceedings.
- Byram, G.M. (1959) "Combustion of forest fuels," in Davis, K.P. (ed.),
  Forest Fire: Control and Use, McGraw-Hill.
"""
from __future__ import annotations

import numpy as np


# Phase 15N defaults — committed values per the Phase 15N plan.  Do NOT
# adjust without re-running the verification sequence and updating the
# memory + docs to match.
Q_0_DEFAULT     = 100_000.0   # [W/m²] peak forward burst convective flux
L_BURST_DEFAULT = 0.30         # [m]    e-folding decay length
D_MAX_CUTOFF    = 1.0          # [m]    above this distance, flux is numerically 0

# Local-intensity gate.  Below this fireline intensity, the buoyancy
# instability does not establish (Finney 2015 Fig 3 broom data).  We use
# a soft saturating gate so the closure ramps off smoothly when chemistry
# is weak (e.g. quiescent inter-burst intervals).
I_FIRE_THRESH = 100_000.0      # [W/m fireline] above this → gate = 1


def compute_finney_burst_q_at_band(
    phi_flame: np.ndarray,       # (Nz, Ny, Nx) signed-distance to flame body
    ahead_band_mask: np.ndarray, # (Nz, Ny, Nx) bool — Phase 15F bed-only
    dx: float,
    x_mid: np.ndarray,           # (Nx,) cell-center x
    I_fire_per_y: np.ndarray | None = None,  # (Ny,) [W/m fireline] per-y-row fire intensity
    q_0: float = Q_0_DEFAULT,
    L_burst: float = L_BURST_DEFAULT,
    d_max: float = D_MAX_CUTOFF,
    I_thresh: float = I_FIRE_THRESH,
) -> np.ndarray:
    """Compute Finney 2015 parameterized burst-convective surface flux [W/m²].

    For each (j, i) horizontal position, find the upstream flame-edge
    position x_flame_edge[j] = max{x_mid[i'] : phi_flame[k, j, i'] ≤ 0 for some k},
    and apply:

        q_burst[j, i] = q_0 · exp(-d / L_burst) · gate(I_fire[j])

    where d = max(0, x_mid[i] - x_flame_edge[j]).  Returns (Ny, Nx) W/m².

    The output is meant to be ADDED to the q_in_at_front per-(j,i) flux
    so the level-set v_n picks up the burst contribution as a forward
    surface heat flux.

    Parameters
    ----------
    phi_flame : (Nz, Ny, Nx) float
        Signed-distance field to the flame body. phi_flame ≤ 0 indicates
        a flame-body cell.
    ahead_band_mask : (Nz, Ny, Nx) bool
        Ahead-band mask. Phase 15F restricts to k < n_z_bed (bed-only).
        Only (j, i) cells with at least one True entry in z are eligible.
    dx : float
        Cell size in x [m].
    x_mid : (Nx,) float
        Cell-center x positions [m].
    I_fire_per_y : (Ny,) float, optional
        Fireline intensity per y-row [W/m fireline]. If supplied, gates
        the burst flux by a soft saturating function above ``I_thresh``.
        If None, gate is 1 wherever flame body is present.
    q_0 : float
        Peak burst convective flux at the front [W/m²].
    L_burst : float
        e-folding decay length [m].
    d_max : float
        Distance cutoff beyond which q_burst is set to zero (numerical
        floor; ~3·L_burst).
    I_thresh : float
        Soft-saturation scale for the I_fire gate [W/m fireline].
    """
    Nz, Ny, Nx = phi_flame.shape
    q_out = np.zeros((Ny, Nx), dtype=np.float64)

    # x position of any flame-body cell at each (k, i) ─ pre-collapse over z:
    # has_flame[j, i] is True if any z-cell at (j, i) has phi_flame ≤ 0
    has_flame = (phi_flame <= 0.0).any(axis=0)   # (Ny, Nx) bool

    # x_flame_edge[j] = max x of any flame-body cell in y-row j
    # Sentinel for "no flame body in this row": -inf
    x_flame_edge = np.full(Ny, -np.inf, dtype=np.float64)
    for j in range(Ny):
        if has_flame[j].any():
            # rightmost (largest x) flame body cell in this row
            x_flame_edge[j] = x_mid[has_flame[j]].max()

    # Per-row I_fire gate (soft saturating)
    if I_fire_per_y is not None:
        gate = np.clip(I_fire_per_y / I_thresh, 0.0, 1.0)
    else:
        gate = np.where(np.isfinite(x_flame_edge), 1.0, 0.0)

    # Cell-in-band mask collapsed to (Ny, Nx): True if any z is in band
    any_in_band = ahead_band_mask.any(axis=0)

    for j in range(Ny):
        if not np.isfinite(x_flame_edge[j]) or gate[j] <= 0.0:
            continue
        for i in range(Nx):
            if not any_in_band[j, i]:
                continue
            d_ahead = x_mid[i] - x_flame_edge[j]
            if d_ahead <= 0.0 or d_ahead > d_max:
                continue
            q_out[j, i] = q_0 * np.exp(-d_ahead / L_burst) * gate[j]
    return q_out


def compute_I_fire_per_y(
    omega: np.ndarray,           # (Nz, Ny, Nx) [kg/m³/s] combustion rate
    dx: float,
    dz_arr: np.ndarray,
    H_c: float = 17.5e6,         # [J/kg] cellulose volatiles HOC
) -> np.ndarray:
    """Per-y-row fireline intensity [W/m fireline].

    I_fire(j) = ∫∫ ω · H_c · dx · dz integrated over (x, z) for that y-row.
    Used as the soft-saturation gate input for compute_finney_burst_q_at_band.
    """
    dz_col = dz_arr.reshape(-1, 1, 1)
    return (omega * H_c * dx * dz_col).sum(axis=(0, 2))   # (Ny,)
