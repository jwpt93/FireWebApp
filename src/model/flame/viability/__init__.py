"""FuelViability protocol and standard fuel_outputs interface.

Standard fuel_outputs dict
--------------------------
Required keys (used by core flame module):
    "HRRPUA_W_m2"  float  total heat release rate per unit area [W/m²]
    "T_surf_K"     float  surface temperature [K]

Optional keys (used by specific viability adapters):
    "m_py"         float  pyrolysis mass flux [kg/m²/s]   (solid fuels)
    "alpha_bar"    float  mean char fraction [-]           (solid fuels)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FuelViability(Protocol):
    """Fuel-type-specific ignition and extinction criteria.

    Implementations receive the standard fuel_outputs dict and return bool.
    The core flame state machine accepts any object satisfying this protocol.
    """

    def should_ignite(self, fuel_outputs: dict) -> bool:
        """Return True if conditions are sufficient to ignite a flame."""
        ...

    def should_extinguish(self, fuel_outputs: dict) -> bool:
        """Return True if conditions require the flame to extinguish."""
        ...


__all__ = ["FuelViability"]
