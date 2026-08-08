"""Outdoor boundary condition helpers.

Wind convection and flame tilt physics for outdoor fire elements.

Wind-enhanced convective heat transfer coefficient:
    h_c(U_mf) = 12.5 * U_mf^0.8   [W/m²/K]
    Source: Rothermel (1972) INT-115, empirical correlation for surface fuel beds.

Midflame wind speed from 10-m reference:
    U_mf = WAF * U_10m
    WAF = 0.9 (open grassland);  WAF = 0.6 (shrubland)
    Source: Rothermel (1972); Albini & Baughman (1979) USDA FS INT-221.

Byram (1959) flame tilt angle from convective Froude number:
    Fr  = U_mf² / (g * L_f)
    tan(θ) = 0.88 * Fr^0.5
    Source: Byram (1959) in Davis (ed.), Forest Fire: Control and Use, pp. 61–89.

View factor enhancement due to flame tilt toward unburned fuel:
    F_tilt / F_0 ≈ 1 + 0.4 * sin(θ)
    (~30–50% increase in forward radiant flux; Albini 1981, Comb. Flame 43:155)

References:
    Rothermel, R.C. (1972). A mathematical model for predicting fire spread in
        wildland fuels.  USDA Forest Service Research Paper INT-115.
    Albini, F.A. & Baughman, R.G. (1979). Estimating windspeeds for predicting
        wildland fire behavior.  USDA FS INT-RP-221.
    Byram, G.M. (1959). Combustion of forest fuels.  In Davis (ed.), Forest Fire:
        Control and Use.  McGraw-Hill, pp. 61–89.
    Albini, F.A. (1981). A model for wind-blown flame from a line fire.
        Combustion and Flame, 43, 155–174.
"""

from __future__ import annotations

import math

# Wind adjustment factors for midflame wind speed from 10-m reference
# Rothermel (1972) / Albini & Baughman (1979)
_WAF = {
    "open": 0.9,    # open grassland
    "shrub": 0.6,   # shrubland with partial canopy
}
_WAF_DEFAULT = 0.9

_G = 9.81  # gravitational acceleration [m/s²]


def midflame_wind_speed(U_10m: float, terrain: str = "open") -> float:
    """Convert 10-m reference wind speed to midflame wind speed.

    Parameters
    ----------
    U_10m : float
        10-m reference wind speed [m/s] (Rothermel 1972 convention).
    terrain : str
        Terrain/vegetation type: 'open' (WAF=0.90) or 'shrub' (WAF=0.60).

    Returns
    -------
    float
        Midflame wind speed U_mf [m/s].
    """
    waf = _WAF.get(terrain.lower(), _WAF_DEFAULT)
    return U_10m * waf


def wind_profile_log_law(z: float, U_ref: float, z_ref: float = 10.0,
                         z_0: float = 0.01) -> float:
    """Log-law boundary-layer wind profile over a rough surface.

    Pre-developed atmospheric boundary layer pinned by no-slip at z=0
    and free-stream U_ref at z=z_ref.  Inside the BL:

        u(z) = (u_τ / κ) · ln((z + z_0) / z_0)

    with κ = 0.40 (von Kármán) and u_τ chosen so that u(z_ref) = U_ref.
    At z = 0 the formula gives u = 0 exactly (no-slip).

    Reference roughness lengths (Monteith & Unsworth 2013 Table 4.1):
        bare soil / desert     z_0 ≈ 0.001 m
        short grass            z_0 ≈ 0.01 m
        long grass / pasture   z_0 ≈ 0.05 m
        crops / shrubs         z_0 ≈ 0.10 m

    For our upstream (bare or short stubble before the bed leading edge),
    z_0 = 0.01 m is a reasonable default; the bed itself adds its own
    drag downstream of bed_x_start.

    Parameters
    ----------
    z : float       Height above ground [m].  Must be ≥ 0.
    U_ref : float   Wind speed at the reference height [m/s].
    z_ref : float   Reference height [m] (default 10 m, Rothermel convention).
    z_0 : float     Surface roughness length [m].

    Returns
    -------
    float
        Wind speed at height z [m/s].  Zero at z=0; rises log-shaped above.

    References
    ----------
    Monin & Obukhov (1954) similarity theory.
    Monteith & Unsworth (2013) "Principles of Environmental Physics" §4.
    """
    if z <= 0.0 or U_ref <= 0.0 or z_ref <= 0.0:
        return 0.0
    KAPPA = 0.40
    u_tau_over_kappa = U_ref / math.log((z_ref + z_0) / z_0)
    return max(0.0, u_tau_over_kappa * math.log((z + z_0) / z_0))


def raupach_d_z0(h_canopy: float, lambda_F: float,
                 c_d2: float = 7.5, C_S: float = 0.003, C_R: float = 0.3) -> tuple:
    """Raupach 1994 (BLM 71:211) sparse-canopy d, z_0 from frontal area index.

    For a vegetation canopy of frontal-area index λ_F (= projected
    silhouette area per unit ground area), the displacement height d and
    roughness z_0 are NOT simply 0.7·h and 0.1·h (which is Raupach 1992's
    bulk asymptote, valid only for densely-packed canopies).  Raupach 1994
    derived sparse-canopy corrections from wind-tunnel measurements over
    arrays of cylinders/plates with varying λ_F:

        d / h     = 1 − (1 − exp(−√(c_d2·λ_F))) / √(c_d2·λ_F)
        z_0 / h   = (1 − d/h) · exp(−κ · u_h/u_*)
        u_h / u_* = (C_S + C_R · λ_F)^(−1/2)

    Limits:
    • λ_F → ∞ (dense canopy):  d/h → 1,  z_0/h → 0.05 (the bulk Raupach 1992 asymptote)
    • λ_F → 0 (bare ground):    d/h → 0,  z_0/h → exp(−κ·(C_S)^(−1/2)) → tiny
                                                  matching bare-ground log-law

    For grass beds, frontal area index of vertically-oriented cylindrical
    blades is approximately:

        λ_F ≈ (σ_sav · α_s · h_bed) / 2

    Typical pasture values:
    • Nat (σ=2000, α_s=7.5e-4, h=0.37):  λ_F = 0.28 → d/h = 0.47, z_0/h ≈ 0.136
    • Cut (σ=3500, α_s=2.5e-3, h=0.15):  λ_F = 0.66 → d/h = 0.60, z_0/h ≈ 0.16
    • Bare ground:                       λ_F = 0    → d/h = 0,   z_0 ≈ z_0_ground

    Parameters
    ----------
    h_canopy : float    Canopy top height [m].
    lambda_F : float    Frontal area index λ_F (= σ·α_s·h/2 for cylindrical
                        blades).  Dimensionless.
    c_d2 : float        Raupach 1994 fit constant (≈7.5 for cylinder arrays).
    C_S, C_R : float    Smooth-surface and canopy element drag coefficients
                        (Raupach 1994 defaults).

    Returns
    -------
    (d, z_0) : tuple of float [m, m]

    References
    ----------
    Raupach (1994) Bound.-Layer Meteorol. 71:211 — sparse-canopy d/z_0.
    Massman (1997) Bound.-Layer Meteorol. 83:407 — review and extension.
    """
    if h_canopy <= 0.0:
        return 0.0, 0.001
    if lambda_F <= 1e-6:
        # Effectively bare ground; tiny z_0
        return 0.0, max(1e-4, 0.001 * h_canopy)
    KAPPA = 0.40
    arg = c_d2 * lambda_F
    sqrt_arg = math.sqrt(arg)
    d_over_h = 1.0 - (1.0 - math.exp(-sqrt_arg)) / sqrt_arg
    d_over_h = max(0.0, min(1.0, d_over_h))
    u_h_over_u_star = (C_S + C_R * lambda_F) ** (-0.5)
    z0_over_h = (1.0 - d_over_h) * math.exp(-KAPPA * u_h_over_u_star)
    z0_over_h = max(1e-6, z0_over_h)
    return d_over_h * h_canopy, z0_over_h * h_canopy


def wind_profile_canopy_bl(z: float, U_ref: float, z_ref: float = 10.0,
                            h_canopy: float = 0.37,
                            alpha_cionco: float = 1.0,
                            z_0_canopy: float = None,
                            d_canopy: float = None) -> float:
    """Atmospheric boundary layer over vegetated canopy — composite profile.

    Above the canopy (z > h_canopy): displaced log-law

        u(z) = (u_τ / κ) · ln((z - d) / z_0)        [z > h_canopy]

    Inside the canopy (0 ≤ z ≤ h_canopy): Cionco 1965 exponential decay

        u(z) = u(h_canopy) · exp(α · (z / h_canopy − 1))   [0 ≤ z ≤ h_canopy]

    Anchored so u(z_ref) = U_ref.  Continuous at z = h_canopy.  At z = 0
    inside the canopy: u = u(h_canopy) · exp(−α) → low but non-zero (Cionco's
    in-canopy bottom-velocity, not strict no-slip; the discrete cell's
    averaging + the no-slip BC at the actual z=0 wall handle the rest).

    Defaults follow Raupach 1992 *Bound.-Layer Meteorol.* 60:375 generic
    vegetation surface scaling:
        z_0 = 0.1 × h_canopy
        d   = 0.7 × h_canopy

    For pasture grass (h_canopy = 0.37 m):
        z_0 = 0.037 m,  d = 0.26 m (Monteith & Unsworth 2013 Table 4.1
        gives z_0 = 0.05 m for "long grass / pasture" — agrees within
        factor 1.3 of Raupach rule-of-thumb).

    Parameters
    ----------
    z : float           Height above ground [m].
    U_ref : float       Wind speed at reference height [m/s].
    z_ref : float       Reference height [m] (default 10 m, Rothermel convention).
    h_canopy : float    Canopy top height [m] (= h_bed for our beds).
    alpha_cionco : float  Cionco in-canopy decay coefficient (1.0 sparse, 2.0 dense).
    z_0_canopy : float  Roughness length above canopy [m].  Default 0.1·h_canopy.
    d_canopy : float    Displacement height [m].  Default 0.7·h_canopy.

    Returns
    -------
    float
        Wind speed at height z [m/s].

    References
    ----------
    Cionco, R.M. (1965) J. Appl. Meteorol. 4:517 — in-canopy decay.
    Raupach, M.R. (1992) Bound.-Layer Meteorol. 60:375 — z_0 and d from h.
    Monteith & Unsworth (2013) *Principles of Environmental Physics* §4.
    """
    if h_canopy <= 0.0 or U_ref <= 0.0 or z_ref <= 0.0 or z < 0.0:
        return 0.0
    if z_0_canopy is None:
        z_0_canopy = 0.1 * h_canopy
    if d_canopy is None:
        d_canopy = 0.7 * h_canopy
    if z_ref <= d_canopy + z_0_canopy:
        # Reference too close to canopy — fallback to pure log-law over bare ground.
        return wind_profile_log_law(z, U_ref, z_ref, z_0_canopy)

    KAPPA = 0.40
    # Anchor u_τ/κ so that u(z_ref) = U_ref via above-canopy log-law.
    u_tau_over_kappa = U_ref / math.log((z_ref - d_canopy) / z_0_canopy)

    if z > h_canopy:
        # Above canopy: displaced log-law.
        arg = (z - d_canopy) / z_0_canopy
        if arg <= 0.0:
            return 0.0
        return max(0.0, u_tau_over_kappa * math.log(arg))

    # Inside canopy: Cionco exponential, anchored to u(h_canopy) from log-law.
    u_top = u_tau_over_kappa * math.log((h_canopy - d_canopy) / z_0_canopy)
    if u_top <= 0.0:
        return 0.0
    z_ratio = max(0.0, min(z / h_canopy, 1.0))
    return u_top * math.exp(alpha_cionco * (z_ratio - 1.0))


def wind_profile_in_bed(z: float, h_bed: float, U_mf: float,
                        alpha_cionco: float = 1.0) -> float:
    """Wind speed at height z within a porous fuel bed.

    Cionco (1965) exponential attenuation within canopy:
        U(z) = U_mf × exp(α × (z/h - 1))

    At z = h_bed (top of bed): U = U_mf (midflame wind speed).
    At z = 0 (ground): U = U_mf × exp(-α).

    Phase 14w-F correction: was 3.0 (Cionco "closed forest"), now 1.0
    (Cionco "sparse grass") which matches Cheney 1993 grassland canopy.
    Cionco 1965 Table 1:
        Open grass field        α = 0.5
        Sparse grass / pasture  α = 1.0
        Dense grass             α = 2.0
        Closed forest           α = 3.0–4.0
    The 3.0 default was over-attenuating: at z=h/2 it gives
    u/U_mf = exp(-1.5) ≈ 0.22, vs the grass value exp(-0.5) ≈ 0.61.
    Combined with the porous-medium drag in interior cells this had
    been pinning in-bed wind to ≈0 in low-wind grass-fire validation.

    Reference: Cionco (1965) "A mathematical model for air flow in a
    vegetative canopy" J. Appl. Meteorol. 4:517-522.

    Parameters
    ----------
    z : float           Height above ground [m].
    h_bed : float       Fuel bed depth [m].
    U_mf : float        Midflame wind speed at bed top [m/s].
    alpha_cionco : float  Attenuation coefficient [-]; 1.0 for sparse
                          grass / pasture (Cionco 1965 Table 1).
                          Range 0.5–4.0 (open grass to closed forest).
    """
    if h_bed <= 0.0 or U_mf <= 0.0:
        return 0.0
    z_ratio = min(z / h_bed, 1.0)
    return U_mf * math.exp(alpha_cionco * (z_ratio - 1.0))


def wind_h_conv(U_10m: float, terrain: str = "open") -> float:
    """Wind-enhanced convective heat transfer coefficient [W/m²/K].

    Empirical correlation for surface fuel beds from Rothermel (1972):
        h_c = 12.5 * U_mf^0.8

    Parameters
    ----------
    U_10m : float
        10-m reference wind speed [m/s].
    terrain : str
        'open' or 'shrub' — sets midflame wind speed conversion.

    Returns
    -------
    float
        h_c [W/m²/K].  Returns 0.0 when U_10m = 0 (calm).
    """
    U_mf = midflame_wind_speed(U_10m, terrain)
    if U_mf <= 0.0:
        return 0.0
    return 12.5 * U_mf**0.8


def byram_flame_length(HRRPUA_W_m2: float, fuel_depth_m: float) -> float:
    """Byram (1959) flame length from HRRPUA and fuel bed depth.

    Fireline intensity: I_B = HRRPUA * fuel_depth  [kW/m]
    Flame length:      L_f = 0.0475 * I_B^0.493   [m]

    Parameters
    ----------
    HRRPUA_W_m2 : float
        Heat release rate per unit area [W/m²].
    fuel_depth_m : float
        Fuel bed depth [m].

    Returns
    -------
    float
        Flame length L_f [m].  Returns 0.0 for zero or negative HRRPUA.
    """
    if HRRPUA_W_m2 <= 0.0 or fuel_depth_m <= 0.0:
        return 0.0
    I_B_kW_m = (HRRPUA_W_m2 / 1000.0) * fuel_depth_m  # kW/m
    return 0.0475 * I_B_kW_m**0.493


def fireline_intensity(HRRPUA_W_m2: float, fuel_depth_m: float) -> float:
    """Byram (1959) fireline intensity [kW/m].

    I_B = HRRPUA [kW/m²] * fuel_depth [m]

    For a stationary single element this approximates:
        I_B = HRRPUA * depth  (assumes unit rate of spread = 1 m/s as placeholder)

    Parameters
    ----------
    HRRPUA_W_m2 : float
        Heat release rate per unit area [W/m²].
    fuel_depth_m : float
        Fuel bed depth [m].

    Returns
    -------
    float
        I_B [kW/m].
    """
    return (HRRPUA_W_m2 / 1000.0) * fuel_depth_m


def flame_tilt_angle(U_10m: float, L_f_m: float, terrain: str = "open") -> float:
    """Byram (1959) flame tilt angle from vertical [rad].

    Convective Froude number:  Fr = U_mf² / (g * L_f)
    Tilt:                      tan(θ) = 0.88 * Fr^0.5

    Parameters
    ----------
    U_10m : float
        10-m reference wind speed [m/s].
    L_f_m : float
        Flame length [m] (from byram_flame_length).
    terrain : str
        'open' or 'shrub'.

    Returns
    -------
    float
        Tilt angle θ [rad].  0.0 if no wind or no flame.
    """
    if U_10m <= 0.0 or L_f_m <= 0.0:
        return 0.0
    U_mf = midflame_wind_speed(U_10m, terrain)
    Fr = U_mf**2 / (_G * L_f_m)
    theta = math.atan(0.88 * Fr**0.5)
    return theta


def view_factor_tilt_enhancement(theta_rad: float) -> float:
    """View factor enhancement due to flame tilt toward unburned fuel.

    Approximate increase in forward radiant flux from a tilted flame:
        F_tilt / F_0 ≈ 1 + 0.4 * sin(θ)

    Albini (1981, Combustion and Flame 43:155) shows ~30–50% increase at
    moderate tilt angles.  This factor applies to the flame feedback q_fb
    directed toward an adjacent unburned fuel element in 1-D spread.

    Parameters
    ----------
    theta_rad : float
        Flame tilt angle from vertical [rad].

    Returns
    -------
    float
        Dimensionless view factor multiplier (≥ 1.0).
    """
    return 1.0 + 0.4 * math.sin(theta_rad)
