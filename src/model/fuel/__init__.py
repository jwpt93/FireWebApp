"""Fuel submodels."""

from .two_node import fuel_rhs, integrate_fuel
from .heat_transfer import (
    air_props,
    h_forced_flat_plate,
    h_natural_flat_plate,
    h_conv,
    q_radiation,
    open_face_loss_flux,
    heat_losses,
)
from .pyrolysis import compute_m_dot_kinetics, pyrolysis_flux, pyrolysis_margin, moisture_factor
from .moisture import moisture_loss_rate, evap_mass_flux, evap_heat_sink
from .depletion import apply_depletion

__all__ = [
    "fuel_rhs",
    "integrate_fuel",
    "compute_m_dot_kinetics",
    "pyrolysis_flux",
    "pyrolysis_margin",
    "moisture_factor",
    "moisture_loss_rate",
    "evap_mass_flux",
    "evap_heat_sink",
    "air_props",
    "h_forced_flat_plate",
    "h_natural_flat_plate",
    "h_conv",
    "q_radiation",
    "open_face_loss_flux",
    "heat_losses",
    "apply_depletion",
]
