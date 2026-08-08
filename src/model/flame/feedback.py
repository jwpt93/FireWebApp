from __future__ import annotations

from typing import Optional

from .config import FlameConfig


def flame_feedback(
    fuel_outputs: dict,
    flame_cfg: FlameConfig,
    is_burning: bool,
    spray_terms: Optional[dict] = None,
) -> float:
    """Return flame radiation feedback heat flux [W/m²].

    Physics basis (De Ris / Tewarson):
        q_fb = chi_rad × view_factor × HRRPUA

    where HRRPUA [W/m²] is the total combustion heat release rate per unit
    area from the fuel model.  This formulation is fuel-type agnostic: it
    does not matter whether HRRPUA came from solid pyrolysis, liquid
    evaporation, or gas combustion.

    Spray suppression (optional):
        q_fb *= (1 - beta_blank) × (1 - beta_peel)

    Parameters
    ----------
    fuel_outputs : dict
        Standard fuel outputs dict.  Must contain "HRRPUA_W_m2" [W/m²].
    flame_cfg : FlameConfig
        chi_rad and view_factor parameters.
    is_burning : bool
        Whether the flame is currently in the BURNING state.
    spray_terms : dict, optional
        Spray suppression factors.  Keys: "beta_blank", "beta_peel" (both in [0, 1]).
        Defaults to no suppression when None or keys absent.

    Returns
    -------
    float
        Flame feedback heat flux [W/m²].  Zero when not burning.
    """
    if not is_burning:
        return 0.0

    HRRPUA = max(float(fuel_outputs.get("HRRPUA_W_m2", 0.0)), 0.0)
    q_fb = flame_cfg.chi_rad * flame_cfg.view_factor * HRRPUA

    if spray_terms:
        beta_blank = float(spray_terms.get("beta_blank", 0.0))
        beta_peel = float(spray_terms.get("beta_peel", 0.0))
        q_fb *= (1.0 - beta_blank) * (1.0 - beta_peel)

    return q_fb
