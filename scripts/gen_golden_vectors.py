"""Generate golden vectors pinning the JS port to the Python reference.

The web applet reimplements two things in JavaScript:

  1. the Cheney 1993 rate-of-spread law   -> src/model_outdoor/empirical_ros.py
  2. a 2D Godunov level-set front         -> src/model_outdoor/physics_3d/
                                              flame_front_3d.py (3D, reduced)

Both are hand ports, so both need proof they still agree with the research
code.  This script emits docs/data/golden.json; docs/test.html loads it, runs
the JS, and reports any disagreement.

For (1) we call the actual project function, so the vectors cannot drift
from the model by construction.  For (2) there is no 2D level set in the
parent project to call, so this file carries a NumPy reference that mirrors
godunov_grad_norm() / reinit_godunov_grad() reduced to two dimensions, with
the parent's x-axis boundary treatment applied to both axes (see the
BOUNDARIES note in web/js/levelset.js).

Run:
    .venv/bin/python scripts/gen_golden_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_outdoor.empirical_ros import (  # noqa: E402
    CHENEY_EQ6_U2_RATIO,
    blend_resolved_empirical,
    cheney_eq6_ros_m_per_s,
)

OUT = ROOT / "docs" / "data" / "golden.json"
OUT_FIG8 = ROOT / "docs" / "data" / "fig8.json"
SRC_FIG8 = ROOT / "data" / "cheney_experimental"


# ---------------------------------------------------------------------------
# 1. Cheney 1993 rate of spread
# ---------------------------------------------------------------------------

def cheney_vectors() -> dict:
    """Vectors for both wind conventions.

    `from_u10` calls the project function directly.  `from_u2` divides out
    the internal 0.723 factor so the vector is expressed in the paper's own
    variable -- the one the applet slider uses and the one the digitised
    Fig 8 x-axis is in.
    """
    winds_u10 = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    moistures = [0.02, 0.04, 0.06, 0.08, 0.12, 0.20]
    a_chs = {"natural": 0.406, "cut": 0.343}

    from_u10, from_u2 = [], []
    for name, a_ch in a_chs.items():
        for u10 in winds_u10:
            for mf in moistures:
                ros = cheney_eq6_ros_m_per_s(u10, mf, a_ch)
                from_u10.append(
                    {"fuel": name, "U10_m_s": u10, "moisture_frac": mf,
                     "a_ch": a_ch, "ros_m_s": ros}
                )
                # Same physical case, expressed at 2 m.
                from_u2.append(
                    {"fuel": name, "U2_m_s": CHENEY_EQ6_U2_RATIO * u10,
                     "moisture_frac": mf, "a_ch": a_ch, "ros_m_s": ros}
                )

    # Edge cases the JS must reproduce rather than crash on.
    edge = [
        {"fuel": "natural", "U10_m_s": 0.0, "moisture_frac": 0.04,
         "a_ch": 0.406, "ros_m_s": cheney_eq6_ros_m_per_s(0.0, 0.04, 0.406)},
        {"fuel": "natural", "U10_m_s": -1.0, "moisture_frac": 0.04,
         "a_ch": 0.406, "ros_m_s": cheney_eq6_ros_m_per_s(-1.0, 0.04, 0.406)},
        {"fuel": "cut", "U10_m_s": 4.0, "moisture_frac": 0.0,
         "a_ch": 0.343, "ros_m_s": cheney_eq6_ros_m_per_s(4.0, 0.0, 0.343)},
    ]

    return {"u2_per_u10": CHENEY_EQ6_U2_RATIO,
            "from_u10": from_u10, "from_u2": from_u2, "edge": edge}


# ---------------------------------------------------------------------------
# 2. Byram derived quantities
# ---------------------------------------------------------------------------

H_KJ_KG = 18600.0


def byram_vectors() -> dict:
    """Byram (1959) fireline intensity and flame length.

    Pure arithmetic plus one power, so these are here to catch a transcribed
    coefficient rather than a numerical subtlety.
    """
    out = []
    for ros in (0.05, 0.2, 0.5, 1.0, 2.0):
        for w0 in (0.2, 0.45, 0.8):
            I = H_KJ_KG * w0 * ros
            out.append({"ros_m_s": ros, "w0_kg_m2": w0, "H_kJ_kg": H_KJ_KG,
                        "I_kW_m": I, "L_f_m": 0.0775 * I ** 0.46})
    return {"cases": out}


# ---------------------------------------------------------------------------
# 3. 2D level set — NumPy reference
# ---------------------------------------------------------------------------

def _grad_norm(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Godunov upwind |grad phi| for v_n > 0.

    Mirrors godunov_grad_norm() in physics_3d/flame_front_3d.py.  One-sided
    differences are zero at every edge (the parent's x-treatment, applied to
    both axes because a bounded 2D domain is not periodic).
    """
    ny, nx = phi.shape
    d_minus_x = np.zeros_like(phi)
    d_plus_x = np.zeros_like(phi)
    d_minus_y = np.zeros_like(phi)
    d_plus_y = np.zeros_like(phi)

    d_minus_x[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
    d_plus_x[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx
    d_minus_y[1:, :] = (phi[1:, :] - phi[:-1, :]) / dy
    d_plus_y[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dy

    gxp = np.maximum(d_minus_x, 0.0)
    gxm = np.minimum(d_plus_x, 0.0)
    gyp = np.maximum(d_minus_y, 0.0)
    gym = np.minimum(d_plus_y, 0.0)
    return np.sqrt(gxp * gxp + gxm * gxm + gyp * gyp + gym * gym)


def _reinit_grad_norm(phi: np.ndarray, phi0: np.ndarray,
                      dx: float, dy: float) -> np.ndarray:
    """Sign-aware Godunov |grad phi| for the Sussman reinit step.

    Mirrors reinit_godunov_grad() in physics_3d/flame_front_3d.py.
    """
    d_minus_x = np.zeros_like(phi)
    d_plus_x = np.zeros_like(phi)
    d_minus_y = np.zeros_like(phi)
    d_plus_y = np.zeros_like(phi)

    d_minus_x[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
    d_plus_x[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx
    d_minus_y[1:, :] = (phi[1:, :] - phi[:-1, :]) / dy
    d_plus_y[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dy

    pos = phi0 > 0.0
    gxp = np.where(pos, np.maximum(d_minus_x, 0.0), np.minimum(d_minus_x, 0.0))
    gxm = np.where(pos, np.minimum(d_plus_x, 0.0), np.maximum(d_plus_x, 0.0))
    gyp = np.where(pos, np.maximum(d_minus_y, 0.0), np.minimum(d_minus_y, 0.0))
    gym = np.where(pos, np.minimum(d_plus_y, 0.0), np.maximum(d_plus_y, 0.0))
    return np.sqrt(gxp * gxp + gxm * gxm + gyp * gyp + gym * gym)


def _seed_circle(nx, ny, dx, dy, cx, cy, r) -> np.ndarray:
    big = max(nx * dx, ny * dy)
    phi = np.full((ny, nx), big, dtype=np.float64)
    xs = (np.arange(nx) + 0.5) * dx - cx
    ys = (np.arange(ny) + 0.5) * dy - cy
    d = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) - r
    return np.minimum(phi, d)


def levelset_vectors() -> dict:
    """Two advection cases plus one reinitialisation case.

    Speeds are chosen to be exactly representable (constant, and a linear
    function of the coordinate) so that the comparison is bit-exact: only
    +, -, *, / and sqrt are involved, all of which are IEEE-754 exact
    operations.  A speed built from exp()/pow() would leave the JS and
    NumPy results free to differ in the last ulp for reasons that have
    nothing to do with the port being correct.
    """
    nx, ny, dx, dy = 24, 20, 0.5, 0.5
    cases = []

    # (a) uniform speed — the front stays circular
    phi = _seed_circle(nx, ny, dx, dy, 6.0, 5.0, 1.5)
    vn = np.ones((ny, nx), dtype=np.float64)
    dt = 0.1
    for _ in range(12):
        phi = phi - dt * vn * _grad_norm(phi, dx, dy)
    cases.append({
        "name": "uniform_speed",
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "seed": {"type": "circle", "cx": 6.0, "cy": 5.0, "r": 1.5},
        "vn": {"type": "uniform", "value": 1.0},
        "dt": dt, "steps": 12,
        "phi": phi.ravel().tolist(),
    })

    # (b) speed varying linearly in x — the front stretches downwind
    phi = _seed_circle(nx, ny, dx, dy, 6.0, 5.0, 1.5)
    xs = (np.arange(nx) + 0.5) * dx
    vn = np.repeat((0.25 + 0.125 * xs)[None, :], ny, axis=0).copy()
    dt = 0.05
    for _ in range(20):
        phi = phi - dt * vn * _grad_norm(phi, dx, dy)
    cases.append({
        "name": "linear_x_speed",
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "seed": {"type": "circle", "cx": 6.0, "cy": 5.0, "r": 1.5},
        "vn": {"type": "linear_x", "a": 0.25, "b": 0.125},
        "dt": dt, "steps": 20,
        "phi": phi.ravel().tolist(),
    })

    # (c) reinitialisation restores |grad phi| = 1 from a deliberately
    #     distorted field (a circle scaled by 3, so |grad phi| = 3)
    phi = 3.0 * _seed_circle(nx, ny, dx, dy, 6.0, 5.0, 1.5)
    phi0 = phi.copy()
    dtau = 0.5 * min(dx, dy)
    sgn = np.sign(phi0)
    for _ in range(5):
        phi = phi + dtau * sgn * (1.0 - _reinit_grad_norm(phi, phi0, dx, dy))
    cases.append({
        "name": "reinit_scaled_circle",
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "seed": {"type": "circle_scaled", "cx": 6.0, "cy": 5.0,
                 "r": 1.5, "scale": 3.0},
        "substeps": 5, "cfl": 0.5,
        "phi": phi.ravel().tolist(),
    })

    return {"cases": cases}


# ---------------------------------------------------------------------------
# 4. Cheney Fig 8 experimental scatter, for the applet's overlay panel
# ---------------------------------------------------------------------------

def fig8_for_web() -> dict:
    """Merge the v1 + v2 digitisations into one payload docs/ can fetch.

    Emitted into docs/ so the published page is self-contained -- it can be
    served from web/ alone, with no path escaping the deploy root.

    x is U_2, the paper's own variable: Cheney 1993 Table 2 defines u2 as
    "Wind speed at 2 m", and the printed Fig 8 axis reads the same.  The
    applet's wind slider is U_2 for exactly this reason -- the model dot and
    the experimental scatter then share an axis with no conversion anywhere.

    KNOWN: the v1 `cut` digitisation runs ~17% low against v2 over
    U_2 in [3, 6] (implied Cheney coefficient 0.1744 vs 0.2071).  That ratio,
    0.84, is not the 0.723 wind-convention factor, so it is a digitisation
    disagreement rather than a unit error.  Both are shipped and tagged by
    source so the applet can show or hide either.
    """
    out = {"_meta": {
        "columns": ["U_2_m_s", "ROS_m_s"],
        "source": "Cheney, Gould & Catchpole (1993) IJWF 3(1):31-44, Fig 8, "
                  "user-digitised. x is the 2-m wind (Table 2; printed axis).",
        "caption_moistures_pct": [4, 8],
        "note": "v1 `cut` runs ~17% low vs v2 over U_2 in [3,6] -- a "
                "digitisation disagreement, not a unit error.",
    }}
    for fuel in ("natural", "cut"):
        rows = []
        for fname, tag in (("cheney1993_fig8_data_v2.json", "v2"),
                           ("cheney1993_fig8_data.json", "v1")):
            doc = json.loads((SRC_FIG8 / fname).read_text())
            rows += [[U, R, tag] for U, R in doc.get(fuel, [])]
        out[fuel] = rows
    return out


def blend_vectors() -> dict:
    """blend_resolved_empirical — the weight on the empirical side.

    Probed densely across and beyond the Option B window (threshold 3.5,
    width 1.0), including the exact boundaries where the branch conditions
    flip: at u_lo the weight must still be 1.0, at the threshold exactly 0.0.
    Also covers width = 0 (hard step) and a wind above threshold, so every
    branch of the function is represented.
    """
    out = []
    for thr, wid in ((3.5, 1.0), (1.4, 0.5), (3.5, 0.0)):
        for U in (0.0, 0.5, 1.0, 1.4, 2.0, 2.4, 2.5, 2.6, 3.0, 3.4, 3.5,
                  3.6, 4.0, 8.0, 20.0):
            out.append({"U10_m_s": U, "threshold": thr, "width": wid,
                        "w_emp": blend_resolved_empirical(U, thr, wid)})
    return {"cases": out}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "purpose": "Pin the JS port in web/js/ to the Python reference "
                       "in src/model_outdoor/.  Regenerate with "
                       "scripts/gen_golden_vectors.py; verify with "
                       "web/test.html.",
            "cheney_source": "Cheney, Gould & Catchpole (1993) IJWF 3(1):31-44, "
                             "Fig 8 caption: R = a·U_2^0.987·exp(-0.0707·M_f)",
            "byram_source": "Byram (1959); metric flame length per "
                            "Alexander (1982) Can. J. For. Res. 12:245",
            "levelset_source": "Godunov upwind, Sethian (1999) §6.4; mirrors "
                               "physics_3d/flame_front_3d.py reduced to 2D",
            "wind_convention_note":
                "U_2 is Cheney's native variable (Table 2; Fig 8 x-axis). "
                "cheney_eq6_ros_m_per_s() takes U_10 and applies 0.723 "
                "internally. Both conventions are pinned below so the port "
                "cannot silently double-convert.",
        },
        "cheney": cheney_vectors(),
        "blend": blend_vectors(),
        "byram": byram_vectors(),
        "levelset": levelset_vectors(),
    }
    OUT.write_text(json.dumps(payload))
    n_ls = sum(len(c["phi"]) for c in payload["levelset"]["cases"])
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  cheney   : {len(payload['cheney']['from_u10'])} x2 conventions "
          f"+ {len(payload['cheney']['edge'])} edge")
    print(f"  byram    : {len(payload['byram']['cases'])} cases")
    print(f"  blend    : {len(payload['blend']['cases'])} cases")
    print(f"  levelset : {len(payload['levelset']['cases'])} cases, "
          f"{n_ls} field values")
    print(f"  size     : {OUT.stat().st_size / 1024:.1f} kB")

    fig8 = fig8_for_web()
    OUT_FIG8.write_text(json.dumps(fig8))
    print(f"wrote {OUT_FIG8.relative_to(ROOT)}")
    print(f"  scatter  : {len(fig8['natural'])} natural + {len(fig8['cut'])} cut "
          f"({OUT_FIG8.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
