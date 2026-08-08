"""Dead fuel moisture dynamics — Nelson (2000) lag-time model.

Governing ODE:
    dM/dt = (EMC(T, RH) - M) / tau

where:
    M    = fuel moisture content [kg water / kg dry fuel]
    EMC  = equilibrium moisture content [-], Van Wagner & Pickett (1985)
    tau  = lag time [s] (1-hr fine fuel: 3600 s, 10-hr: 36000 s)

EMC formula (Van Wagner & Pickett 1985, as used in NFDRS):
    EMC_20 = 0.942 * RH^1.104 + 8.27 * RH^3.112 / (1 - RH)   [at 20 °C ref]
    Temperature correction:
        EMC(T) = EMC_20 * (1 - 0.003 * (T_C - 20))

Typical dead fuel moisture values:
    Cured grass (1-hr):   M = 0.03–0.08   extinction M_x = 0.12
    Shrub litter (10-hr): M = 0.08–0.15   extinction M_x = 0.20

Usage::
    from model_outdoor.moisture import solve_dead_fuel_moisture, equilibrium_mc

    # Pre-ignition drying over 3600 s at T=300 K, RH=30%, 1-hr fuel
    M_final = solve_dead_fuel_moisture(
        t_dry_s=3600.0,
        M0=0.10,
        T_K=300.0,
        RH_frac=0.30,
        tau_s=3600.0,
    )

References:
    Nelson, R.M. (2000). Prediction of diurnal change in 10-h fuel stick moisture
        content. Canadian Journal of Forest Research, 30, 1071–1087.
    Van Wagner, C.E. & Pickett, T.L. (1985). Equations and FORTRAN program for
        the Canadian Forest Fire Weather Index System. Can. For. Serv. Tech. Rep. 33.
    Viney, N.R. (1991). A review of fine fuel moisture modelling.
        International Journal of Wildland Fire, 1(3), 215–224.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def equilibrium_mc(T_K: float, RH_frac: float) -> float:
    """Equilibrium moisture content [-] via Van Wagner & Pickett (1985).

    Parameters
    ----------
    T_K : float
        Air temperature [K].
    RH_frac : float
        Relative humidity [0–1].

    Returns
    -------
    float
        EMC [kg water / kg dry fuel], clamped to [0, 0.40].
    """
    # Clamp RH to avoid division by zero and numerical instability at high RH
    RH = float(np.clip(RH_frac, 1e-4, 0.999))
    T_C = T_K - 273.15

    # Van Wagner & Pickett (1985) formula at reference T = 20 °C
    EMC_20 = 0.942 * RH**1.104 + 8.27 * RH**3.112 / (1.0 - RH)

    # Temperature correction factor (Nelson 2000 eq. 4)
    EMC = EMC_20 * (1.0 - 0.003 * (T_C - 20.0))

    # Convert from percent to fraction (formula gives %)
    EMC_frac = EMC / 100.0

    return float(np.clip(EMC_frac, 0.0, 0.40))


def solve_dead_fuel_moisture(
    t_dry_s: float,
    M0: float,
    T_K: float,
    RH_frac: float,
    tau_s: float = 3600.0,
) -> float:
    """Solve the Nelson (2000) lag-time ODE and return final moisture content.

    Integrates dM/dt = (EMC - M) / tau over [0, t_dry_s] using constant
    atmospheric conditions (T_K, RH_frac).  For variable conditions, use
    ``solve_dead_fuel_moisture_trajectory``.

    Parameters
    ----------
    t_dry_s : float
        Pre-ignition drying duration [s].
    M0 : float
        Initial moisture content [kg water / kg dry fuel].
    T_K : float
        Constant air temperature [K].
    RH_frac : float
        Constant relative humidity [0–1].
    tau_s : float
        Lag time [s].  Default 3600 s (1-hr fine fuel, Nelson 2000).

    Returns
    -------
    float
        Final moisture content M(t_dry_s) [kg/kg].
    """
    if t_dry_s <= 0.0:
        return float(M0)

    EMC = equilibrium_mc(T_K, RH_frac)

    # Analytical solution: M(t) = EMC + (M0 - EMC) * exp(-t / tau)
    M_final = EMC + (M0 - EMC) * np.exp(-t_dry_s / tau_s)
    return float(np.clip(M_final, 0.0, 1.0))


def solve_dead_fuel_moisture_trajectory(
    t_span_s: tuple[float, float],
    M0: float,
    T_K_callable,
    RH_frac_callable,
    tau_s: float = 3600.0,
    n_out: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Nelson ODE with time-varying T and RH.

    Parameters
    ----------
    t_span_s : (t0, t1)
        Integration interval [s].
    M0 : float
        Initial moisture content [kg/kg].
    T_K_callable : callable
        T_K(t) [K] as a function of time.
    RH_frac_callable : callable
        RH(t) [-] as a function of time.
    tau_s : float
        Lag time [s].
    n_out : int
        Number of output time points.

    Returns
    -------
    t : np.ndarray shape (n_out,)
        Time [s].
    M : np.ndarray shape (n_out,)
        Moisture content [-].
    """

    def rhs(t, y):
        EMC = equilibrium_mc(float(T_K_callable(t)), float(RH_frac_callable(t)))
        return [(EMC - y[0]) / tau_s]

    t_eval = np.linspace(t_span_s[0], t_span_s[1], n_out)
    sol = solve_ivp(rhs, t_span_s, [M0], t_eval=t_eval, method="RK45", dense_output=False)
    return sol.t, np.clip(sol.y[0], 0.0, 1.0)
