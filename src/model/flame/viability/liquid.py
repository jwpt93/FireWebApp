"""LiquidFuelViability — stub for liquid pool fire ignition/extinction.

Future implementation: ignition based on vapor concentration exceeding LFL,
extinction based on vapor flux dropping below critical value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiquidFuelViability:
    """Stub viability adapter for liquid pool fires. Not yet implemented."""

    def should_ignite(self, fuel_outputs: dict) -> bool:
        raise NotImplementedError("LiquidFuelViability is not yet implemented.")

    def should_extinguish(self, fuel_outputs: dict) -> bool:
        raise NotImplementedError("LiquidFuelViability is not yet implemented.")
