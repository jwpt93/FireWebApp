"""Spray suppression physics.

Three suppression mechanisms (Rasbash 1962):
  1. Thermal quench — water absorbs heat as it heats and vaporises
  2. Steam expansion — displaces O2, suppresses flaming combustion
  3. Foam blanket — shields fuel surface from incoming radiation + flame feedback

Thermal heat sink applied to the ROM surface boundary condition:
    Q_water [W/m²] = m_dot_w * (c_p_w * dT_w + eta * L_v)

Foam blanket scales both incident flux and flame feedback:
    q_eff = (q_in + q_fb) * (1 - foam_cover_frac)

Critical water application rate (Johansson et al. 2018, Fire 2(1):3):
    W_crit [kg/m²/s] = I_B_W_m / Q_water_per_kg
where I_B_W_m = fireline intensity [W/m] and Q_water_per_kg is the denominator above.

Usage::
    from model_outdoor.suppression import spray_heat_sink_W_m2, w_critical
    from model_outdoor.config import SprayConfig

    spray = SprayConfig(enable=True, m_dot_water_kg_m2_s=0.025, eta_evap=0.70)
    Q_sink = spray_heat_sink_W_m2(spray)   # [W/m²] to subtract from surface BC
    W_c = w_critical(I_B_kW_m=50.0, spray=spray)  # [kg/m²/s]

References:
    Rasbash, D.J. (1962). The extinction of fires by water sprays.
        Fire Research Abstracts and Reviews, Vols. 4–5.
    Johansson, N., van Hees, P., & Särdqvist, S. (2018). Calculation of critical
        water flow rates for wildfire suppression.  Fire, 2(1), 3.
        https://doi.org/10.3390/fire2010003
"""

from __future__ import annotations

from model_outdoor.config import SprayConfig


def spray_heat_sink_W_m2(spray: SprayConfig) -> float:
    """Thermal heat sink from water spray [W/m²].

    Computes the rate at which applied water removes energy from the fuel surface
    per unit area, via sensible heating and (partial) evaporation.

        Q_water = m_dot_w * (c_p_w * dT_w + eta * L_v)

    Parameters
    ----------
    spray : SprayConfig
        Spray device configuration.

    Returns
    -------
    float
        Q_water [W/m²] — positive value to be *subtracted* from the net surface
        heat flux in the ROM boundary condition.  Returns 0.0 if spray disabled
        or m_dot_water is zero.
    """
    if not spray.enable or spray.m_dot_water_kg_m2_s <= 0.0:
        return 0.0
    Q = spray.m_dot_water_kg_m2_s * (
        spray.c_p_water_J_kg_K * spray.delta_T_water_K
        + spray.eta_evap * spray.L_v_J_kg
    )
    return Q


def foam_flux_factor(spray: SprayConfig) -> float:
    """Multiplicative factor on (q_in + q_fb) due to foam blanket shielding.

    Returns (1 - foam_cover_frac), so a full coverage foam (frac=1) drives
    incoming flux to zero.

    Parameters
    ----------
    spray : SprayConfig

    Returns
    -------
    float
        Scale factor in [0, 1].
    """
    if not spray.enable:
        return 1.0
    return max(0.0, 1.0 - float(spray.foam_cover_frac))


def w_critical(I_B_kW_m: float, spray: SprayConfig) -> float:
    """Critical water application rate required to suppress a given fire [kg/m²/s].

    From Johansson et al. (2018, Fire 2(1):3):
        W_crit = I_B [W/m] / (eta * (c_p_w * dT_w + L_v))

    This is a diagnostic output — it tells the user whether their applied flow
    rate (spray.m_dot_water_kg_m2_s * fuel_depth) is sufficient to suppress I_B.

    Parameters
    ----------
    I_B_kW_m : float
        Byram fireline intensity [kW/m].
    spray : SprayConfig
        Spray configuration (uses eta_evap, c_p_water_J_kg_K, delta_T_water_K, L_v_J_kg).

    Returns
    -------
    float
        W_crit [kg/m²/s].  For comparison: Johansson (2018) reports
        0.016–0.042 kg/m²/s for light outdoor fuels (I_B ~50–100 kW/m).
    """
    I_B_W_m = I_B_kW_m * 1000.0  # kW/m → W/m
    denominator = spray.eta_evap * (
        spray.c_p_water_J_kg_K * spray.delta_T_water_K + spray.L_v_J_kg
    )
    if denominator <= 0.0:
        return float("inf")
    return I_B_W_m / denominator


def suppression_summary(
    I_B_kW_m: float,
    spray: SprayConfig,
    fuel_depth_m: float = 1.0,
) -> dict:
    """Return a summary dict of spray suppression diagnostics.

    Parameters
    ----------
    I_B_kW_m : float
        Byram fireline intensity [kW/m].
    spray : SprayConfig
    fuel_depth_m : float
        Fuel bed depth [m] — used to convert area-flux to line-flux.

    Returns
    -------
    dict with keys:
        Q_sink_W_m2        thermal heat sink [W/m²]
        foam_factor        flux scale factor [-]
        W_crit_kg_m2_s     critical flow rate [kg/m²/s]
        W_applied_kg_m2_s  applied flow rate
        suppression_ratio  W_applied / W_crit (>1 → expected suppression)
    """
    Q_sink = spray_heat_sink_W_m2(spray)
    f_foam = foam_flux_factor(spray)
    W_c = w_critical(I_B_kW_m, spray)
    W_app = spray.m_dot_water_kg_m2_s if spray.enable else 0.0
    ratio = W_app / W_c if W_c > 0.0 else 0.0

    return {
        "Q_sink_W_m2": Q_sink,
        "foam_factor": f_foam,
        "W_crit_kg_m2_s": W_c,
        "W_applied_kg_m2_s": W_app,
        "suppression_ratio": ratio,
    }
