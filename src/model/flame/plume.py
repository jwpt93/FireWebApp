"""McCaffrey (1979) centreline plume correlation for free-burning fires.

Three-zone empirical model for the centreline temperature excess and vertical
velocity above a pool fire, using the similarity variable ζ = z / Q^{2/5}.

Reference
---------
McCaffrey, B.J. (1979). "Purely Buoyant Diffusion Flames: Some Experimental
  Results." NBSIR 79-1910, National Bureau of Standards (now NIST).

Zone boundaries (Table 2 of McCaffrey 1979):
  ζ < 0.08  m·kW^{-2/5}   — continuous flame
  0.08–0.20               — intermittent zone
  ζ > 0.20                — buoyant plume

Usage
-----
This module provides diagnostic output only. Results are NOT coupled back
into the ODE. Use to assess plume conditions at heights relevant to device
placement (e.g. sprinkler, nozzle, sensor locations).
"""

from __future__ import annotations


def mccaffrey_plume(
    Q_kW: float,
    z_m: float,
    T_amb_K: float = 293.0,
) -> tuple[float, float]:
    """McCaffrey (1979) three-zone centreline plume.

    Returns centerline temperature excess ΔT [K above ambient] and
    centerline velocity u [m/s].

    Parameters
    ----------
    Q_kW    : float  Total fire HRR [kW]. Clamped to ≥ 0.01.
    z_m     : float  Height above fire base [m]. Must be > 0.
    T_amb_K : float  Ambient temperature [K]. Default 293 K (20 °C).

    Returns
    -------
    (dT_K, u_m_s) : tuple[float, float]
        dT_K   — centerline temperature excess above ambient [K]
        u_m_s  — centerline upward velocity [m/s]

    Notes
    -----
    Similarity variable: ζ = z / Q^{2/5}   [m · kW^{-2/5}]

    Temperature: ΔT = C_T × T_amb × ζ^{n_T}
    Velocity:    u  = C_u × Q^{1/5} × ζ^{n_u}

    The velocity formula expands to (substituting ζ = z/Q^{0.4}):
      flame zone    (n_u = 1/2): u = C_u × z^{1/2}              (Q-independent)
      intermittent  (n_u = 0):   u = C_u × Q^{1/5}
      plume         (n_u =-1/3): u = C_u × Q^{1/3} × z^{-1/3}  (classical Rouse scaling)

    The Q^{1/3} × z^{-1/3} scaling in the plume zone matches the classical result
    of Morton, Taylor & Turner (1956) for a buoyant point-source plume.

    McCaffrey's coefficients are purely empirical; continuity at zone boundaries
    is approximate (~2 % for ΔT; ~50 % step in velocity at ζ = 0.08). This
    discontinuity is an accepted feature of the three-zone fit (see Drysdale 2011,
    §4.3). For a diagnostic-only output it does not affect model predictions.
    """
    Q = max(float(Q_kW), 0.01)
    z = max(float(z_m), 1e-6)
    T_a = max(float(T_amb_K), 200.0)

    zeta = z / (Q ** 0.4)

    if zeta < 0.08:               # continuous flame zone
        C_T, n_T = 6.8,  0.5
        C_u, n_u = 3.47, 0.5
    elif zeta < 0.20:             # intermittent zone
        C_T, n_T = 1.9,  0.0
        C_u, n_u = 1.90, 0.0
    else:                          # buoyant plume zone
        C_T, n_T = 1.1, -1.0 / 3.0
        C_u, n_u = 1.10, -1.0 / 3.0

    dT_K  = C_T * T_a * (zeta ** n_T)
    u_m_s = C_u * (Q ** 0.2) * (zeta ** n_u)
    return dT_K, u_m_s
