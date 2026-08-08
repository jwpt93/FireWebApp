"""SolidFuelViability — ignition/extinction criteria for solid pyrolysing fuels.

Uses pyrolysis mass flux (m_py) and surface temperature (T_surf_K) from the
standard fuel_outputs dict.  These are the same criteria previously baked into
model/flame/viability.py, now expressed as a FuelViability adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolidFuelViability:
    """Viability adapter for solid pyrolysing fuels (wood, PMMA, particle board, etc.)."""

    m_py_ignite: float = 0.005
    """Pyrolysis flux threshold for ignition [kg/m²/s]."""

    m_py_crit: float = 0.001
    """Pyrolysis flux threshold for extinction [kg/m²/s]."""

    T_ignite: float = 600.0
    """Surface temperature threshold for ignition [K]."""

    T_py: float = 500.0
    """Minimum surface temperature to sustain pyrolysis [K]."""

    def should_ignite(self, fuel_outputs: dict) -> bool:
        """Return True if m_py or T_surf exceeds ignition threshold."""
        return (
            float(fuel_outputs.get("m_py", 0.0)) > self.m_py_ignite
            or float(fuel_outputs.get("T_surf_K", 0.0)) > self.T_ignite
        )

    def should_extinguish(self, fuel_outputs: dict) -> bool:
        """Return True if m_py OR T_surf is below sustain threshold."""
        return (
            float(fuel_outputs.get("m_py", 0.0)) < self.m_py_crit
            or float(fuel_outputs.get("T_surf_K", 0.0)) < self.T_py
        )
