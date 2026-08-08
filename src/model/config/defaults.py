from __future__ import annotations

from .schemas import FuelConfig, EnvConfig, SimConfig, Thresholds


def default_fuel_config() -> FuelConfig:
    return FuelConfig()


def default_env_config() -> EnvConfig:
    return EnvConfig()


def default_sim_config() -> SimConfig:
    return SimConfig()


def default_thresholds() -> Thresholds:
    return Thresholds()


def default_configs() -> tuple[FuelConfig, EnvConfig, SimConfig, Thresholds]:
    return (
        default_fuel_config(),
        default_env_config(),
        default_sim_config(),
        default_thresholds(),
    )
