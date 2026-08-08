from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class AirProps:
    k: float  # [W/m/K]
    nu: float  # [m^2/s]
    alpha: float  # [m^2/s]
    Pr: float  # [-]


def air_props(T_f_K: float) -> AirProps:
    """Return simple/constant air properties at film temperature."""
    k = 0.026
    nu = 1.5e-5
    alpha = 2.2e-5
    Pr = nu / alpha
    return AirProps(k=k, nu=nu, alpha=alpha, Pr=Pr)


def h_forced_flat_plate(T_s: float, T_inf: float, u_inf: float, L: float, props: AirProps) -> float:
    if u_inf <= 1.0e-9 or L <= 1.0e-9:
        return 0.0
    Re_L = u_inf * L / props.nu
    Pr = props.Pr
    if Re_L <= 5.0e5:
        Nu_L = 0.664 * np.sqrt(Re_L) * Pr ** (1.0 / 3.0)
    else:
        Nu_L = (0.037 * Re_L ** 0.8 - 871.0) * Pr ** (1.0 / 3.0)
        if Nu_L < 0.0:
            Nu_L = 0.0
    return props.k * Nu_L / L


def h_natural_flat_plate(
    T_s: float,
    T_inf: float,
    L: float,
    props: AirProps,
    orientation: str = "vertical",
) -> float:
    if L <= 1.0e-9:
        return 0.0
    delta_T = abs(T_s - T_inf)
    if delta_T <= 1.0e-9:
        return 0.0
    T_f = 0.5 * (T_s + T_inf)
    beta = 1.0 / max(T_f, 1.0)
    g = 9.81
    Gr_L = g * beta * delta_T * L**3 / (props.nu**2)
    Ra_L = Gr_L * props.Pr

    orientation = orientation.lower()
    if orientation == "horizontal_up":
        if 1.0e5 < Ra_L < 1.0e10:
            Nu_L = 0.54 * Ra_L ** 0.25
        elif 1.0e10 <= Ra_L < 1.0e13:
            Nu_L = 0.15 * Ra_L ** (1.0 / 3.0)
        else:
            Nu_L = 0.68 + (0.670 * Ra_L ** 0.25) / ((1.0 + (0.492 / props.Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))
    elif orientation == "horizontal_down":
        Nu_L = 0.27 * Ra_L ** 0.25
    else:
        Nu_L = 0.68 + (0.670 * Ra_L ** 0.25) / ((1.0 + (0.492 / props.Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))

    return props.k * Nu_L / L


def h_conv(
    T_s: float,
    T_inf: float,
    u_inf: float,
    L: float,
    props: AirProps,
    mode: str = "auto",
    orientation: str = "vertical",
    n: float = 3.0,
) -> float:
    mode_l = mode.lower()
    if mode_l == "auto":
        mode_l = "forced" if u_inf > 1.0e-9 else "natural"

    h_forced = h_forced_flat_plate(T_s, T_inf, u_inf, L, props)
    h_nat = h_natural_flat_plate(T_s, T_inf, L, props, orientation=orientation)

    if mode_l == "forced":
        return h_forced
    if mode_l == "natural":
        return h_nat
    if mode_l == "mixed":
        return (h_forced**n + h_nat**n) ** (1.0 / n)
    return h_nat


def q_radiation(
    T_s: float,
    T_sur: float,
    eps: float,
    C_eps: float = 1.0,
    sigma: float = 5.670374419e-8,
    T_max_K: float = 5000.0,
):
    # Allow eps_eff=0.0 so radiation losses can be explicitly disabled in sweeps.
    eps_eff = float(np.clip(C_eps * eps, 0.0, 0.99))
    # Guard against overflow for extreme temperatures
    T_s_c = float(np.clip(T_s, 0.0, T_max_K))
    T_sur_c = float(np.clip(T_sur, 0.0, T_max_K))
    q_rad = eps_eff * sigma * (T_s_c**4 - T_sur_c**4)
    h_rad_equiv = eps_eff * sigma * (T_s_c + T_sur_c) * (T_s_c**2 + T_sur_c**2)
    return q_rad, h_rad_equiv, eps_eff


def open_face_loss_flux(
    T2: float,
    h_open: float,
    eps_open: float,
    T_inf: float,
    T_sur: float,
    sigma: float = 5.670374419e-8,
    T_max_K: float = 5000.0,
) -> float:
    """Open-backside loss flux [W/m^2] for node-2.

    Positive values represent heat loss from the fuel.
    """

    T2_c = float(np.clip(T2, 0.0, T_max_K))
    T_inf_c = float(np.clip(T_inf, 0.0, T_max_K))
    T_sur_c = float(np.clip(T_sur, 0.0, T_max_K))
    h_eff = max(float(h_open), 0.0)
    eps_eff = float(np.clip(eps_open, 0.0, 1.0))
    return h_eff * (T2_c - T_inf_c) + eps_eff * sigma * (T2_c**4 - T_sur_c**4)


def heat_losses(
    T_s: float,
    T_inf: float,
    T_sur: float,
    eps: float,
    L: float,
    u_inf: float,
    mode: str,
    orientation: str,
    C_h_conv: float = 1.0,
    C_eps: float = 1.0,
) -> Dict[str, float]:
    props = air_props(0.5 * (T_s + T_inf))
    h_base = h_conv(T_s, T_inf, u_inf, L, props, mode=mode, orientation=orientation)
    C_h = float(np.clip(C_h_conv, 0.0, 2.0))
    h_eff = C_h * h_base
    q_conv = h_eff * (T_s - T_inf)
    q_rad, h_rad_equiv, eps_eff = q_radiation(T_s, T_sur, eps, C_eps=C_eps)

    return {
        "h_conv": h_eff,
        "q_conv": q_conv,
        "q_rad": q_rad,
        "h_rad_equiv": h_rad_equiv,
        "eps_eff": eps_eff,
    }
