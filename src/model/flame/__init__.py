"""Flame submodule — fuel-agnostic flame state machine and radiative feedback.

Public API
----------
FlameConfig          dataclass — radiation + state-machine parameters
FlameState           enum      — UNLIT / BURNING / OUT
FlameInternalState   dataclass — mutable state carried between timesteps
flame_step()         function  — advance state, return q_fb [W/m²]
FuelViability        protocol  — interface for fuel-type-specific viability checks
SolidFuelViability   class     — concrete adapter for solid pyrolysing fuels
"""

from .config import FlameConfig
from .state_machine import FlameState, FlameInternalState, flame_step, make_legacy_viability
from .viability import FuelViability
from .viability.solid import SolidFuelViability
from .feedback import flame_feedback
from .geometry import bed_equivalent_diameter, heskestad_flame_height, pool_fire_view_factor
from .plume import mccaffrey_plume

__all__ = [
    "FlameConfig",
    "FlameState",
    "FlameInternalState",
    "flame_step",
    "make_legacy_viability",
    "FuelViability",
    "SolidFuelViability",
    "flame_feedback",
    "bed_equivalent_diameter",
    "heskestad_flame_height",
    "pool_fire_view_factor",
    "mccaffrey_plume",
]
