from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlameConfig:
    """Flame radiation feedback and state-machine parameters.

    These parameters are fuel-type agnostic. Material-specific values
    (chi_rad, view_factor) are set per calibration target.
    """

    # --- Radiative feedback ---
    chi_rad: float = 0.35
    """Radiative fraction of HRRPUA [-].
    Typical values: wood ~0.35 (Tewarson), PMMA ~0.40, charcoal ~0.30."""

    view_factor: float = 0.40
    """Geometric view factor from flame to surface [-].
    Cone calorimeter: ~0.35–0.45 depending on flame height; calibration parameter."""

    # --- State machine ---
    persistence_s: float = 5.0
    """Extinction persistence window [s].
    Flame must remain below extinction thresholds for this duration before
    transitioning OUT → confirmed-out (no re-ignition within window)."""

    # --- Coupling ---
    enable: bool = False
    """Activate flame coupling in rom_adapter chunked loop.
    False = single-shot integration (existing behavior, q_fb=0)."""

    dt_chunk_s: float = 1.0
    """Coupling chunk size [s] for the rom_adapter chunked integration loop.
    Smaller = more frequent flame-state updates; 0.5–2.0 s is typical."""

    hoc_eff_J_kg: float = 15.5e6
    """Effective heat of combustion [J/kg] used to convert m_py → HRRPUA when the
    fuel model does not provide HRRPUA_W_m2 directly (e.g. model/runner.py spray path).
    Typical values: wood ~15.5 MJ/kg, PMMA ~24.9 MJ/kg."""

    # --- Finite-bed geometry (sub-1 m³ free-burning) ---
    area_m2: float | None = None
    """Fuel bed plan area [m²]. When set and geometry_mode = 'heskestad', the view
    factor is computed per time-step from the Heskestad (1983) flame height and the
    Drysdale (1999) pool-fire radiation configuration factor, replacing the static
    deck value. None = deck view_factor used as-is (backward compatible)."""

    geometry_mode: str = "deck"
    """View factor source: 'deck' = use static view_factor field (default, backward
    compatible); 'heskestad' = compute F(t) from bed area + HRRPUA at each coupling
    step via Heskestad (1983) and Drysdale (1999) Eq. 4.16."""

    plume_heights_m: list | None = None
    """Heights [m] at which to evaluate the McCaffrey (1979) centreline plume as
    diagnostic output. None = no plume output. Set via deck:
    ``fuel.flame_plume_heights_m = 0.25; 0.50; 1.00``"""
