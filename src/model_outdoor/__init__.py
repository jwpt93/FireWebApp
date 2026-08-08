"""Outdoor small-fire / brush-fire extension for the ROM.

Single-element outdoor physics (wind convection, dead fuel moisture,
spray suppression) that wraps the existing lumped pyrolysis ROM.

See docs/user_guide/ch_outdoor_wip.tex for governing equations and
deck parameter reference.

Planned modules:
  config.py        -- OutdoorEnvConfig, SprayConfig dataclasses
  moisture.py      -- Nelson (2000) dead fuel moisture lag ODE
  boundary.py      -- wind_h_conv(), flame_tilt_angle()
  suppression.py   -- spray_heat_sink(), w_critical()
  fuel_element.py  -- run_outdoor_element() wrapper
  spread.py        -- future 1-D line spread (stub)
"""

from model_outdoor.config import OutdoorEnvConfig, SprayConfig
from model_outdoor.fuel_element import run_outdoor_element

__all__ = ["OutdoorEnvConfig", "SprayConfig", "run_outdoor_element"]
