"""OutdoorWindBC — the wind-driven inflow used by all pre-Phase-23 cases.

x-min face  = wind inlet (log-law or Raupach-canopy-BL profile)
x-max face  = outflow (zero-grad)
y-faces     = periodic (thin-slab convention)
z-min face  = solid wall (bed base) — no-slip
z-max face  = outflow (free)

Behaviour-preserving refactor of the inline u_inlet computation
previously in :func:`model_outdoor.spread_3d.run_3d_spread`
(lines 1337-1372).  Same math, wrapped as a class so future BCs
(cup burner, water-mist injection, etc.) can be added alongside
without touching this code path.

Rule #17 (bit-exact preservation): the numerical output of
``build_u_inlet(grid)`` must be identical to the pre-refactor inline
code.  Verified by Cheney Nat4_U4 regression check after Refactor 2A.
"""
from __future__ import annotations

import numpy as np

from model_outdoor.boundary import (
    wind_profile_log_law, wind_profile_canopy_bl, raupach_d_z0,
)
from .base import BoundaryCondition


# Same constant as in spread_3d.py — bare-ground short-stubble roughness
# for the upstream fetch.  Duplicated here (not imported) to keep the BC
# module self-contained.  Kept in sync via unit test.
Z_0_INLET = 0.01    # [m]


class OutdoorWindBC(BoundaryCondition):
    """x-face wind inlet + all other faces standard outdoor pattern."""

    kind = "outdoor_wind"

    def __init__(
        self,
        wind_speed_m_s: float,
        wind_profile_type: str = "log_law",
        h_bed: float = 0.0,
        sigma_sav: float = 2000.0,
        alpha_s_avg: float = 0.0,
    ):
        """
        Parameters
        ----------
        wind_speed_m_s : float
            Reference wind speed at 10 m (Rothermel convention).
        wind_profile_type : str
            "log_law" (bare-ground BL) or "canopy_bl" (Raupach-adjusted
            displaced log-law for vegetated fetch).
        h_bed : float
            Canopy height [m].  Only used when wind_profile_type=
            "canopy_bl".
        sigma_sav : float
            Fuel surface-to-volume [1/m].  Only used for canopy_bl (to
            derive the Raupach frontal-area index).
        alpha_s_avg : float
            Cell-averaged solid volume fraction in the bed.  Only used
            for canopy_bl.
        """
        if wind_profile_type not in ("log_law", "canopy_bl"):
            raise ValueError(
                f"wind_profile_type={wind_profile_type!r} not in "
                f"{{'log_law', 'canopy_bl'}}"
            )
        self.wind_speed_m_s = float(wind_speed_m_s)
        self.wind_profile_type = str(wind_profile_type)
        self.h_bed = float(h_bed)
        self.sigma_sav = float(sigma_sav)
        self.alpha_s_avg = float(alpha_s_avg)
        # Populated by build_u_inlet() at solver setup time and cached
        # for downstream consumers (SEM inlet perturbation, diagnostics).
        self.u_inlet: np.ndarray | None = None

    def build_u_inlet(self, grid) -> np.ndarray:
        """Return the (Nz, Ny) inflow-face u profile.

        Bit-exact reproduction of the pre-Refactor-2A inline code in
        spread_3d.run_3d_spread.
        """
        u_inlet = np.zeros((grid.Nz, grid.Ny))
        if self.wind_profile_type == "log_law":
            for k in range(grid.Nz):
                z = grid.z_mid[k]
                u_inlet[k, :] = wind_profile_log_law(
                    z, self.wind_speed_m_s,
                    z_ref=10.0, z_0=Z_0_INLET,
                )
        else:  # canopy_bl
            lambda_F_bed = 0.5 * self.sigma_sav * self.alpha_s_avg * self.h_bed
            d_canopy, z0_canopy = raupach_d_z0(self.h_bed, lambda_F_bed)
            for k in range(grid.Nz):
                z = grid.z_mid[k]
                u_inlet[k, :] = wind_profile_canopy_bl(
                    z, self.wind_speed_m_s,
                    z_ref=10.0, h_canopy=self.h_bed,
                    alpha_cionco=1.0,
                    d_canopy=d_canopy, z_0_canopy=z0_canopy,
                )
        self.u_inlet = u_inlet
        return u_inlet

    def configure(self, proj_solver, grid, state) -> None:
        """Compute u_inlet, install into projection solver, seed interior."""
        u_inlet = self.build_u_inlet(grid)
        proj_solver.set_inlet_BC(u_inlet)
        state.u[:, :, 0] = u_inlet
        # Interior spin-up: initialize u to inlet profile everywhere.
        state.u[:, :, :] = u_inlet[:, :, np.newaxis]
