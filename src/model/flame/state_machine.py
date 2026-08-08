from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .config import FlameConfig
from .feedback import flame_feedback
from .viability import FuelViability


class FlameState(Enum):
    UNLIT = "UNLIT"
    BURNING = "BURNING"
    OUT = "OUT"


@dataclass
class FlameInternalState:
    """Internal flame state tracker."""

    state: FlameState = FlameState.UNLIT
    t_out: Optional[float] = None  # [s] time when flame first transitioned to OUT


def flame_step(
    t: float,
    fuel_outputs: dict,
    flame_cfg: FlameConfig,
    viability: FuelViability,
    internal_state: FlameInternalState,
    spray_terms: Optional[dict] = None,
) -> Tuple[float, FlameState, FlameInternalState]:
    """Advance flame state and return feedback heat flux.

    Parameters
    ----------
    t : float
        Current simulation time [s].
    fuel_outputs : dict
        Standard fuel outputs dict with at minimum:
            "HRRPUA_W_m2"  [W/m²]   total heat release rate per unit area
            "T_surf_K"     [K]      surface temperature
        Additional keys may be used by the viability adapter.
    flame_cfg : FlameConfig
        Radiative feedback parameters and state-machine settings.
    viability : FuelViability
        Fuel-type-specific ignition/extinction criteria (any FuelViability implementer).
    internal_state : FlameInternalState
        Mutable state carried between timesteps.  Modified in-place and returned.
    spray_terms : dict, optional
        Spray suppression factors passed to flame_feedback().

    Returns
    -------
    q_fb : float
        Flame radiation feedback heat flux [W/m²].
    state : FlameState
        Updated flame state.
    internal_state : FlameInternalState
        Updated internal state.
    """
    state = internal_state.state

    if state == FlameState.UNLIT:
        if viability.should_ignite(fuel_outputs):
            state = FlameState.BURNING
            internal_state.t_out = None

    elif state == FlameState.BURNING:
        if viability.should_extinguish(fuel_outputs):
            state = FlameState.OUT
            if internal_state.t_out is None:
                internal_state.t_out = t

    elif state == FlameState.OUT:
        # Persistence window: allow re-ignition only within persistence_s after going OUT.
        # After the window expires the flame stays OUT permanently (until next ignition
        # event from UNLIT, which cannot happen once OUT — so OUT is a terminal state
        # for a single burn event).
        elapsed = t - (internal_state.t_out if internal_state.t_out is not None else t)
        if elapsed < flame_cfg.persistence_s:
            # Within window: re-ignite if conditions recover
            if viability.should_ignite(fuel_outputs):
                state = FlameState.BURNING
                internal_state.t_out = None
        # else: past persistence window — remain OUT

    internal_state.state = state
    q_fb = flame_feedback(
        fuel_outputs, flame_cfg, is_burning=(state == FlameState.BURNING),
        spray_terms=spray_terms,
    )
    return q_fb, state, internal_state


def make_legacy_viability(thresholds) -> FuelViability:
    """Build a SolidFuelViability from the legacy Thresholds dataclass.

    Allows model/runner.py (spray path) to keep using Thresholds without
    modification while flame_step() uses the new FuelViability protocol.
    """
    from .viability.solid import SolidFuelViability
    return SolidFuelViability(
        m_py_ignite=float(getattr(thresholds, "m_py_ignite", 0.005)),
        m_py_crit=float(getattr(thresholds, "m_py_crit", 0.001)),
        T_ignite=float(getattr(thresholds, "T_ignite", 600.0)),
        T_py=float(getattr(thresholds, "T_py", 500.0)),
    )
