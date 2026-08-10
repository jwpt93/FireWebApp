"""Golden vectors for the ported physics kernels.

The browser port reimplements the solver's kernels in JavaScript. Every one of
them has to be proved against the Python it came from, or the port is just a
plausible-looking rewrite.

METHOD
------
For each kernel: build deterministic inputs, call the REAL Python kernel from
model_outdoor/physics_3d/, and record inputs + outputs. docs/kerneltest.mjs
replays the same inputs through the JS and compares.

BIT-EXACT IS THE TARGET, not a tolerance. These kernels are explicit stencils:
each output cell is written by exactly one iteration, from reads of the input
buffer, using only +, -, *, / and comparisons. No reductions, no accumulation
order to disagree about. IEEE-754 gives the same answer in both languages, so
any difference at all is a porting bug rather than noise.

That claim is not hypothetical — the same approach already produced a
bit-exact 2D level set (docs/js/levelset.js) against a NumPy reference, and a
serial-vs-parallel numba comparison of step_species_transport that matched to
max|diff| = 0.

INPUTS are drawn from a seeded PCG64 generator and shaped to look like real
solver state (positive densities, a plausible wind profile, a hot patch) so
the tests exercise realistic branches — upwind both ways, limiter active and
inactive, boundary fallbacks — rather than a uniform field that would hide
half the code paths.

Run:
    .venv/bin/python scripts/gen_kernel_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = Path("/home/jw/projects/unitiedmodel2")
sys.path.insert(0, str(PARENT))

OUT = ROOT / "docs" / "data" / "kernel_vectors.json"

# Shapes: the 2D slab the applet actually runs, plus a 3D case so the port is
# not accidentally specialised to Ny = 1.
SHAPES = [
    ("slab2d", 12, 1, 16),
    ("small3d", 8, 4, 10),
]


def make_state(nz, ny, nx, seed=20260810):
    """Deterministic pseudo-solver state.

    Shaped rather than uniform: a sheared wind that reverses in places (so
    both upwind branches fire), a hot patch (so the limiter sees a real
    gradient), and non-uniform dz (so the z-direction exercises its
    per-cell spacing).
    """
    r = np.random.default_rng(seed)
    shape = (nz, ny, nx)

    z = np.arange(nz)[:, None, None] / max(nz - 1, 1)
    x = np.arange(nx)[None, None, :] / max(nx - 1, 1)

    # Wind: log-ish in z, reversing near the outlet so u changes sign.
    u = (0.5 + 4.0 * np.sqrt(z)) * np.ones(shape) - 2.5 * x
    v = 0.2 * (r.random(shape) - 0.5)
    w = 0.6 * np.sin(np.pi * x) * np.ones(shape) + 0.1 * (r.random(shape) - 0.5)

    # A hot patch with a sharp edge — the limiter's reason to exist.
    phi = 0.05 + 0.9 * np.exp(-30.0 * (x - 0.35) ** 2) * np.ones(shape)
    phi += 0.02 * r.random(shape)

    # Bed cells thin, atmosphere growing — mirrors build_z_axis_bed_atm.
    dz = 0.046 * np.ones(nz)
    dz[nz // 3:] = 0.046 * 1.3 ** np.arange(nz - nz // 3)
    d_above = np.empty(nz)
    d_below = np.empty(nz)
    for k in range(nz):
        d_above[k] = 0.5 * (dz[k] + dz[min(k + 1, nz - 1)])
        d_below[k] = 0.5 * (dz[k] + dz[max(k - 1, 0)])

    return dict(
        phi=np.ascontiguousarray(phi),
        u=np.ascontiguousarray(u),
        v=np.ascontiguousarray(v),
        w=np.ascontiguousarray(w),
        rho=np.ascontiguousarray(0.6 + 0.8 * r.random(shape)),
        S=np.ascontiguousarray(2.0 * (r.random(shape) - 0.5)),
        dz_arr=dz, d_face_above=d_above, d_face_below=d_below,
        dx=0.30, dy=0.10,
    )


def vec_muscl():
    """advect_3d_scalar_muscl + its two scalar helpers."""
    from model_outdoor.physics_3d import muscl_3d

    cases = []
    for name, nz, ny, nx in SHAPES:
        s = make_state(nz, ny, nx)
        rhs = np.zeros_like(s["phi"])
        # Pre-load the accumulator: the kernel SUBTRACTS into it, so a nonzero
        # start proves the port accumulates rather than assigns.
        rhs += 0.25 * np.arange(rhs.size, dtype=np.float64).reshape(rhs.shape) / rhs.size
        rhs_in = rhs.copy()
        muscl_3d.advect_3d_scalar_muscl(
            s["phi"], s["u"], s["v"], s["w"], s["dx"], s["dy"],
            s["d_face_above"], s["d_face_below"], rhs, 0.017,
            np.zeros((ny, nx)), False,
        )
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "dx": s["dx"], "dy": s["dy"], "phi_inlet": 0.017,
            "phi": s["phi"].ravel().tolist(),
            "u": s["u"].ravel().tolist(),
            "v": s["v"].ravel().tolist(),
            "w": s["w"].ravel().tolist(),
            "d_face_above": s["d_face_above"].tolist(),
            "d_face_below": s["d_face_below"].tolist(),
            "rhs_in": rhs_in.ravel().tolist(),
            "rhs_out": rhs.ravel().tolist(),
        })

    # Scalar helpers, including the sign-change and equal-magnitude edges.
    helper = []
    probes = [(-2.0, 1.0), (1.0, -2.0), (0.0, 3.0), (3.0, 0.0), (2.0, 5.0),
              (5.0, 2.0), (-4.0, -1.0), (-1.0, -4.0), (2.5, 2.5), (-2.5, -2.5)]
    for a, b in probes:
        helper.append({"fn": "minmod", "args": [a, b],
                       "want": muscl_3d.minmod(a, b)})
    face = [(0.1, 0.4, 0.9, 1.1, 1.0), (0.1, 0.4, 0.9, 1.1, -1.0),
            (1.0, 0.5, 0.2, 0.1, 2.0), (1.0, 0.5, 0.2, 0.1, -2.0),
            (0.3, 0.3, 0.3, 0.3, 0.0), (0.0, 1.0, 0.0, 1.0, 0.5)]
    for a, b, c, d, uf in face:
        helper.append({"fn": "muscl_face_value", "args": [a, b, c, d, uf],
                       "want": muscl_3d.muscl_face_value(a, b, c, d, uf)})

    return {"cases": cases, "helpers": helper}


def vec_species():
    """step_species_transport — the biggest kernel in the 2D profile."""
    from model_outdoor.physics_3d import species_3d

    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=777)
        # Y must be a mass fraction in [0, 1]; make it one, with a sharp
        # interface so the limiter engages and the [0,1] clip can fire.
        Y = np.clip(st["phi"], 0.0, 1.0).copy()
        Y_in = Y.copy()
        # Source strong enough that some cells would leave [0,1] without the
        # clip -- so the port's clip is actually exercised, not assumed.
        S = 40.0 * st["S"]
        species_3d.step_species_transport(
            Y, st["rho"], st["u"], st["v"], st["w"], S,
            2.0e-3, st["dx"], st["dy"], st["dz_arr"],
            st["d_face_above"], st["d_face_below"],
            1.0e-4, 0.021, np.zeros((ny, nx)), False,
        )
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "dt": 2.0e-3, "dx": st["dx"], "dy": st["dy"],
            "D": 1.0e-4, "Y_inlet": 0.021,
            "Y_in": Y_in.ravel().tolist(),
            "rho": st["rho"].ravel().tolist(),
            "u": st["u"].ravel().tolist(),
            "v": st["v"].ravel().tolist(),
            "w": st["w"].ravel().tolist(),
            "S": S.ravel().tolist(),
            "dz_arr": st["dz_arr"].tolist(),
            "d_face_above": st["d_face_above"].tolist(),
            "d_face_below": st["d_face_below"].tolist(),
            "Y_out": Y.ravel().tolist(),
            "n_clipped": int(((Y == 0.0) | (Y == 1.0)).sum()),
        })
    return {"cases": cases}


def vec_momentum():
    """step_tentative_velocity — advection + viscosity + buoyancy + body force."""
    from model_outdoor.physics_3d import momentum_3d

    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=31337)
        u, v, w = st["u"].copy(), st["v"].copy(), st["w"].copy()
        u_in, v_in, w_in = u.copy(), v.copy(), w.copy()
        r = np.random.default_rng(4242)
        # Temperature spanning ambient so buoyancy takes BOTH signs -- a
        # uniformly hot field would never exercise the sinking branch.
        T_g = 300.0 + 900.0 * np.exp(-20.0 * (np.arange(nx)[None, None, :]
                                              / max(nx - 1, 1) - 0.4) ** 2)
        T_g = np.ascontiguousarray(T_g * np.ones((nz, ny, nx)) - 40.0)
        F = [np.ascontiguousarray(6.0 * (r.random((nz, ny, nx)) - 0.5))
             for _ in range(3)]
        # Inlet profile: sheared, nonzero in all three components so the
        # i=0 Dirichlet ghost is genuinely tested.
        u_inl = np.ascontiguousarray(0.5 + 3.0 * np.sqrt(
            np.arange(nz)[:, None] / max(nz - 1, 1)) * np.ones((nz, ny)))
        v_inl = np.ascontiguousarray(0.05 * np.ones((nz, ny)))
        w_inl = np.ascontiguousarray(-0.03 * np.ones((nz, ny)))
        momentum_3d.step_tentative_velocity(
            u, v, w, st["rho"], T_g, F[0], F[1], F[2],
            1.5e-3, st["dx"], st["dy"], st["dz_arr"],
            st["d_face_above"], st["d_face_below"], 300.0,
            u_inl, v_inl, w_inl,
        )
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "dt": 1.5e-3, "dx": st["dx"], "dy": st["dy"], "T_amb": 300.0,
            "u_in": u_in.ravel().tolist(), "v_in": v_in.ravel().tolist(),
            "w_in": w_in.ravel().tolist(),
            "rho": st["rho"].ravel().tolist(), "T_g": T_g.ravel().tolist(),
            "Fx": F[0].ravel().tolist(), "Fy": F[1].ravel().tolist(),
            "Fz": F[2].ravel().tolist(),
            "dz_arr": st["dz_arr"].tolist(),
            "d_face_above": st["d_face_above"].tolist(),
            "d_face_below": st["d_face_below"].tolist(),
            "u_inlet": u_inl.ravel().tolist(),
            "v_inlet": v_inl.ravel().tolist(),
            "w_inlet": w_inl.ravel().tolist(),
            "u_out": u.ravel().tolist(), "v_out": v.ravel().tolist(),
            "w_out": w.ravel().tolist(),
            "n_buoy_neg": int((T_g < 300.0).sum()),
        })
    return {"cases": cases}


def _bed(nz, ny, nx, r):
    """Fuel bed in the lower third, zero above — so alpha_s = 0 branches fire."""
    a = np.zeros((nz, ny, nx))
    a[: max(nz // 3, 1)] = 0.05 + 0.25 * r.random((max(nz // 3, 1), ny, nx))
    return np.ascontiguousarray(a)


def vec_drag():
    from model_outdoor.physics_3d import drag_3d
    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=99)
        r = np.random.default_rng(5)
        alpha = _bed(nz, ny, nx, r)
        F = [np.ones((nz, ny, nx)) * -7.0 for _ in range(3)]  # pre-fill:
        # the kernel OVERWRITES, so a nonzero start proves it assigns.
        drag_3d.step_drag_force(st["u"], st["v"], st["w"], st["rho"], alpha,
                                2000.0, F[0], F[1], F[2], 0.30)
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "sigma_sav": 2000.0, "C_D": 0.30,
            "u": st["u"].ravel().tolist(), "v": st["v"].ravel().tolist(),
            "w": st["w"].ravel().tolist(), "rho": st["rho"].ravel().tolist(),
            "alpha_s": alpha.ravel().tolist(),
            "Fx": F[0].ravel().tolist(), "Fy": F[1].ravel().tolist(),
            "Fz": F[2].ravel().tolist(),
            "n_nofuel": int((alpha <= 0).sum()),
        })
    return {"cases": cases}


def vec_solid_conduction():
    from model_outdoor.physics_3d import solid_conduction_3d
    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=1234)
        r = np.random.default_rng(6)
        alpha = _bed(nz, ny, nx, r)
        Ts = np.ascontiguousarray(300.0 + 700.0 * r.random((nz, ny, nx)))
        Ts_in = Ts.copy()
        solid_conduction_3d.step_solid_conduction_vertical(
            Ts, alpha, st["dz_arr"], st["d_face_above"], st["d_face_below"],
            0.2, 500.0, 1300.0, 0.05)
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "k_solid": 0.2, "rho_solid": 500.0, "cp_solid": 1300.0, "dt": 0.05,
            "T_s_in": Ts_in.ravel().tolist(), "alpha_s": alpha.ravel().tolist(),
            "dz_arr": st["dz_arr"].tolist(),
            "d_face_above": st["d_face_above"].tolist(),
            "d_face_below": st["d_face_below"].tolist(),
            "T_s_out": Ts.ravel().tolist(),
        })
    return {"cases": cases}


def vec_coupling():
    from model_outdoor.physics_3d import coupling_3d
    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=2468)
        r = np.random.default_rng(7)
        alpha = _bed(nz, ny, nx, r)
        Tg = np.ascontiguousarray(300.0 + 1200.0 * r.random((nz, ny, nx)))
        Ts = np.ascontiguousarray(300.0 + 600.0 * r.random((nz, ny, nx)))
        # Some cells dry, some wet, so BOTH evaporation branches run and the
        # water cap is hit in at least some cells.
        # Water content and dt chosen so the evaporation CAP actually binds:
        # q_evap_max = mw*L_v/dt must fall below q_in_solid (~1e6 W/m^3) in
        # some cells, i.e. mw < q_in*dt/L_v ~ 0.1 kg/m^3 at dt = 0.5 s.
        # With the original mw ~ 2 kg/m^3 at dt = 1 ms the cap could never
        # bind and half the branch went untested.
        mw = np.ascontiguousarray(r.random((nz, ny, nx)) * 0.25)
        mw[mw < 0.05] = 0.0
        Tg_in, Ts_in, mw_in = Tg.copy(), Ts.copy(), mw.copy()
        qrad = np.ascontiguousarray(3.0e4 * r.random((nz, ny, nx)))
        Qp = np.ascontiguousarray(1.0e4 * r.random((nz, ny, nx)))
        Qc = np.ascontiguousarray(5.0e5 * (r.random((nz, ny, nx)) - 0.3))
        coupling_3d.step_gas_solid_coupling(
            Tg, Ts, st["rho"], st["u"], st["v"], st["w"], alpha, 2000.0,
            qrad, Qp, Qc, mw, 2.26e6, 0.5, st["dz_arr"], 300.0, True, 1.0)
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "sigma_sav": 2000.0, "L_v": 2.26e6, "dt": 0.5, "T_amb": 300.0,
            "T_g_in": Tg_in.ravel().tolist(), "T_s_in": Ts_in.ravel().tolist(),
            "m_water_in": mw_in.ravel().tolist(),
            "rho": st["rho"].ravel().tolist(), "u": st["u"].ravel().tolist(),
            "v": st["v"].ravel().tolist(), "w": st["w"].ravel().tolist(),
            "alpha_s": alpha.ravel().tolist(), "q_rad_in": qrad.ravel().tolist(),
            "Q_pyro": Qp.ravel().tolist(), "Q_comb": Qc.ravel().tolist(),
            "dz_arr": st["dz_arr"].tolist(),
            "T_g_out": Tg.ravel().tolist(), "T_s_out": Ts.ravel().tolist(),
            "m_water_out": mw.ravel().tolist(),
            "n_dried": int((mw_in > 0).sum() - (mw > 0).sum()),
        })
    return {"cases": cases}


def main() -> None:
    payload = {
        "_meta": {
            "purpose": "Pin the JS physics kernels in docs/js/physics/ to the "
                       "Python in model_outdoor/physics_3d/.",
            "target": "BIT-EXACT. These are explicit stencils with no "
                      "reductions, so IEEE-754 gives identical results in "
                      "both languages; any difference is a porting bug.",
            "source": str(PARENT / "model_outdoor/physics_3d"),
            "verify": "node docs/kerneltest.mjs",
        },
        "muscl": vec_muscl(),
        "species": vec_species(),
        "momentum": vec_momentum(),
        "drag": vec_drag(),
        "solid_conduction": vec_solid_conduction(),
        "coupling": vec_coupling(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    n = sum(len(c["rhs_out"]) for c in payload["muscl"]["cases"])
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} kB)")
    print(f"  muscl:   {len(payload['muscl']['cases'])} field cases "
          f"({n} values), {len(payload['muscl']['helpers'])} scalar probes")
    for c in payload["drag"]["cases"]:
        print(f"  drag:    {c['name']:8s} {c['n_nofuel']} no-fuel cells (early-return branch)")
    for c in payload["coupling"]["cases"]:
        print(f"  coupling:{c['name']:8s} {c['n_dried']} cells fully dried")
    for c in payload["momentum"]["cases"]:
        print(f"  momentum:{c['name']:8s} {c['nz']}x{c['ny']}x{c['nx']}, "
              f"{c['n_buoy_neg']} cells with negative buoyancy")
    for c in payload["species"]["cases"]:
        print(f"  species: {c['name']:8s} {c['nz']}x{c['ny']}x{c['nx']}, "
              f"{c['n_clipped']} cells hit the [0,1] clip")


if __name__ == "__main__":
    main()
