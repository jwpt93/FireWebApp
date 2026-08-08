from __future__ import annotations

from ..config.schemas import FuelConfig
from .properties import clamp01, safe_exp


def moisture_loss_rate(T1: float, M1: float, fuel_cfg: FuelConfig) -> float:
    """Moisture loss rate [1/s]."""

    if T1 <= fuel_cfg.T_evap_onset:
        return 0.0
    theta = (T1 - fuel_cfg.T_evap_onset) / max(fuel_cfg.T_evap_onset, 1.0)
    return fuel_cfg.k_evap0 * safe_exp(theta)


def evap_mass_flux(T1: float, M1: float, fuel_cfg: FuelConfig) -> float:
    """Evaporation mass flux [kg/m^2/s]."""

    k_ev = moisture_loss_rate(T1, M1, fuel_cfg)
    m1 = clamp01(M1)
    return k_ev * m1 * fuel_cfg.m1_max_kg_m2


def evap_heat_sink(T1: float, M1: float, fuel_cfg: FuelConfig) -> float:
    """Evaporative heat sink [W/m^2]."""

    m_evap = evap_mass_flux(T1, M1, fuel_cfg)
    # TODO: incorporate air flow and spray cooling in evaporation.
    return m_evap * fuel_cfg.h_fg
