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


def vec_edc():
    """step_chemistry_ode_edc — Magnussen EDC with all three suppressions.

    Two variants per shape so BOTH extinction paths are covered: gates off
    (the production Cheney setting) and gates on.  Inputs are chosen so each
    suppression actually fires somewhere -- a quench that never triggers, or
    a cold-flame floor no cell falls below, would leave that branch untested.
    """
    from model_outdoor.physics_3d.chemistry_closures import edc

    cases = []
    for name, nz, ny, nx in SHAPES:
        for ext in (False, True):
            r = np.random.default_rng(8080 + int(ext))
            shape = (nz, ny, nx)
            rho = np.ascontiguousarray(0.3 + 0.9 * r.random(shape))
            # T spanning the 1200 K ignition floor AND the 2400 K cap, so the
            # cold-flame gate and the cap both fire.
            T_g = np.ascontiguousarray(400.0 + 2200.0 * r.random(shape))
            Y_f = np.ascontiguousarray(0.30 * r.random(shape))
            Y_f[Y_f < 0.02] = 0.0           # some cells fuel-starved
            Y_O2 = np.ascontiguousarray(0.05 + 0.18 * r.random(shape))
            # Y_H2O straddling the 0.18 quench limit: some cells unaffected,
            # some >50% suppressed (wet-bulb cascade), some fully quenched.
            Y_H2O = np.ascontiguousarray(0.28 * r.random(shape))
            k_t = np.ascontiguousarray(1e-5 + 2.0 * r.random(shape))
            eps = np.ascontiguousarray(1e-7 + 3.0 * r.random(shape))
            om = np.zeros(shape)
            Tg_in, Yf_in, YO2_in = T_g.copy(), Y_f.copy(), Y_O2.copy()
            edc.step_chemistry_ode_edc(
                rho, T_g, Y_f, Y_O2, k_t, eps, 0.34, 1100.0, 2.0e-3, 3,
                om, Y_H2O, ext, 1.3, 17_000_000.0)
            cases.append({
                "name": f"{name}_ext{int(ext)}", "nz": nz, "ny": ny, "nx": nx,
                "extinction_enable": ext, "chi_rad": 0.34, "cp_g": 1100.0,
                "dt": 2.0e-3, "n_substeps": 3,
                "s_stoich": 1.3, "hoc_J": 17_000_000.0,
                "rho": rho.ravel().tolist(),
                "T_g_in": Tg_in.ravel().tolist(),
                "Y_fuel_in": Yf_in.ravel().tolist(),
                "Y_O2_in": YO2_in.ravel().tolist(),
                "k_turb": k_t.ravel().tolist(), "eps_turb": eps.ravel().tolist(),
                "Y_H2O": Y_H2O.ravel().tolist(),
                "T_g_out": T_g.ravel().tolist(),
                "Y_fuel_out": Y_f.ravel().tolist(),
                "Y_O2_out": Y_O2.ravel().tolist(),
                "omega_out": om.ravel().tolist(),
                "n_quenched": int((Y_H2O >= 0.18).sum()),
                "n_below_Tign": int((Tg_in < 1200.0).sum()),
                "n_capped": int((T_g >= 2400.0).sum()),
            })
    return {"cases": cases}


def vec_poisson():
    """SeparableLaplacian3D.solve — the pressure Poisson solve.

    Ny = 1 only: the y-transform is an FFT over the periodic direction and is
    the identity for a single cell, which is the slab the applet runs.

    The RHS is a divergence-like field (mean removed, sharp local feature) so
    the solve is exercised the way the projection step exercises it, not on a
    smooth blob that any solver would get right.
    """
    from model_outdoor.physics_3d.fft_poisson_3d import SeparableLaplacian3D

    cases = []
    for name, nz, _ny, nx in SHAPES:
        st = make_state(nz, 1, nx, seed=606)
        sol = SeparableLaplacian3D(
            nz, 1, nx, st["dx"], st["dy"], st["dz_arr"],
            st["d_face_above"], st["d_face_below"], 1.0e-6)
        r = np.random.default_rng(11)
        rhs = (r.random((nz, 1, nx)) - 0.5) * 8.0
        # A sharp source, as a real divergence field has near a hot cell.
        rhs[nz // 3, 0, nx // 2] += 60.0
        rhs = np.ascontiguousarray(rhs - rhs.mean())
        pout = sol.solve(rhs)
        cases.append({
            "name": name, "nz": nz, "ny": 1, "nx": nx,
            "dx": st["dx"], "dy": st["dy"], "eps_reg": 1.0e-6,
            "dz_arr": st["dz_arr"].tolist(),
            "d_face_above": st["d_face_above"].tolist(),
            "d_face_below": st["d_face_below"].tolist(),
            "rhs": rhs.ravel().tolist(),
            "p_out": pout.ravel().tolist(),
            "lambda_x": sol.lambda_total_inv.shape[2],
        })
    return {"cases": cases}


def vec_turb_diff():
    """apply_turbulent_diffusion — species and gas-energy diffusion.

    nu_t is scaled so the kernel actually SUB-STEPS (n_sub > 1); with a small
    nu_t it would take one pass and the sub-cycling logic would go untested.
    """
    from model_outdoor.physics_3d import turbulence_3d

    cases = []
    for name, nz, ny, nx in SHAPES:
        for sc_t, tag in ((0.7, "sc"), (0.85, "pr")):
            st = make_state(nz, ny, nx, seed=515)
            r = np.random.default_rng(21)
            # Big enough that Fo forces several sub-steps on the thin bed cells.
            nu_t = np.ascontiguousarray(1e-4 + 0.05 * r.random((nz, ny, nx)))
            fld = np.ascontiguousarray(st["phi"].copy())
            fld_in = fld.copy()
            turbulence_3d.apply_turbulent_diffusion(
                fld, nu_t, sc_t, 2.0e-2, st["dx"], st["dy"],
                st["dz_arr"], st["d_face_above"], st["d_face_below"])
            # How many sub-steps did it take?  Recomputed here so the vector
            # records whether the sub-cycling path was exercised at all.
            import math as _m
            dz_min = float(np.min(st["dz_arr"]))
            h2 = min(st["dx"]**2, min(st["dy"]**2, dz_min**2))
            Dt = float(nu_t[1:nz-1, :, 1:nx-1].max())/sc_t
            n_sub = max(1, int(_m.ceil(2.0e-2 / (0.4*h2/Dt))))
            cases.append({
                "name": f"{name}_{tag}", "nz": nz, "ny": ny, "nx": nx,
                "sc_t": sc_t, "dt": 2.0e-2, "dx": st["dx"], "dy": st["dy"],
                "field_in": fld_in.ravel().tolist(),
                "nu_t": nu_t.ravel().tolist(),
                "dz_arr": st["dz_arr"].tolist(),
                "d_face_above": st["d_face_above"].tolist(),
                "d_face_below": st["d_face_below"].tolist(),
                "field_out": fld.ravel().tolist(),
                "n_sub": n_sub,
            })
    return {"cases": cases}


def vec_kepsilon():
    """step_k_epsilon — realizable k-eps with the FIXED dT/dz boundaries.

    Generated AFTER the 2026-08-10 upstream fix that replaced the
    out-of-bounds T_g[k+1]/T_g[k-1] reads with one-sided differences.  Vectors
    generated before that fix are not comparable.

    T_g is shaped so buoyancy takes both signs (unstable low down, stable
    aloft), and alpha_s is a bed so the Sanz canopy terms fire in some cells
    and not others.
    """
    from model_outdoor.physics_3d import turbulence_3d as T

    cases = []
    for name, nz, ny, nx in SHAPES:
        st = make_state(nz, ny, nx, seed=1717)
        r = np.random.default_rng(31)
        alpha = _bed(nz, ny, nx, r)
        # Hot near the ground, cooling aloft -> dT/dz negative low (unstable,
        # G_k > 0) and positive higher (stable, G_k clamped to 0).
        zc = np.arange(nz)[:, None, None] / max(nz - 1, 1)
        T_g = np.ascontiguousarray(1100.0 - 700.0 * zc + 60.0 * r.random((nz, ny, nx)))
        k_t = np.ascontiguousarray(1e-4 + 1.5 * r.random((nz, ny, nx)))
        eps = np.ascontiguousarray(1e-5 + 2.0 * r.random((nz, ny, nx)))
        nu_t = np.zeros((nz, ny, nx))
        S = np.zeros((nz, ny, nx)); O = np.zeros((nz, ny, nx))
        u_inl = np.ascontiguousarray(0.4 + 3.0 * np.sqrt(
            np.arange(nz)[:, None] / max(nz - 1, 1)) * np.ones((nz, ny)))
        kw = np.full((ny, nx), 1.0e-3)
        ew = np.full((ny, nx), 1.0e-2)
        k_in, e_in = k_t.copy(), eps.copy()
        T.step_k_epsilon(
            k_t, eps, nu_t, st["u"], st["v"], st["w"], T_g, st["rho"], alpha,
            2000.0, 1.5e-3, st["dx"], st["dy"], st["dz_arr"],
            st["d_face_above"], st["d_face_below"], 300.0, S, O, u_inl,
            kw, ew, 1.0, 4.0, 0.0, 0.0, 0.0,
            np.zeros(nz, dtype=np.int64), np.zeros(nz, dtype=np.int64))
        cases.append({
            "name": name, "nz": nz, "ny": ny, "nx": nx,
            "sigma_sav": 2000.0, "dt": 1.5e-3, "dx": st["dx"], "dy": st["dy"],
            "T_amb": 300.0, "beta_p": 1.0, "beta_d": 4.0,
            "k_in": k_in.ravel().tolist(), "eps_in": e_in.ravel().tolist(),
            "u": st["u"].ravel().tolist(), "v": st["v"].ravel().tolist(),
            "w": st["w"].ravel().tolist(), "T_g": T_g.ravel().tolist(),
            "rho": st["rho"].ravel().tolist(), "alpha_s": alpha.ravel().tolist(),
            "dz_arr": st["dz_arr"].tolist(),
            "d_face_above": st["d_face_above"].tolist(),
            "d_face_below": st["d_face_below"].tolist(),
            "u_inlet": u_inl.ravel().tolist(),
            "k_wall_ghost": kw.ravel().tolist(),
            "eps_wall_ghost": ew.ravel().tolist(),
            "k_out": k_t.ravel().tolist(), "eps_out": eps.ravel().tolist(),
            "nu_t_out": nu_t.ravel().tolist(),
            "n_canopy": int((alpha > 0).sum()),
        })
    return {"cases": cases}


def vec_dom():
    """DOMRadiationSolver.solve — the only non-local kernel in the port.

    Two variants: with and without the Y_H2O absorption channel, since that
    term is what closes the Cheney moisture gap and deserves its own coverage.
    """
    from model_outdoor.physics_3d.dom_3d import DOMRadiationSolver

    cases = []
    for name, nz, ny, nx in SHAPES:
        for with_h2o in (False, True):
            st = make_state(nz, ny, nx, seed=909)
            r = np.random.default_rng(43)
            alpha = _bed(nz, ny, nx, r)
            # Hot flame band inside the bed, cool aloft -> real optical
            # contrast rather than a uniform slab.
            T_s = np.ascontiguousarray(300.0 + 900.0 * alpha / max(alpha.max(), 1e-9))
            T_g = np.ascontiguousarray(300.0 + 1400.0 * r.random((nz, ny, nx)))
            om = np.ascontiguousarray(1e-4 + 5e-3 * r.random((nz, ny, nx)))
            YH = np.ascontiguousarray(0.15 * r.random((nz, ny, nx))) if with_h2o else None
            rho = st["rho"] if with_h2o else None
            bedM = np.ascontiguousarray(0.3 * r.random((nz, ny, nx)) * (alpha > 0))
            qs = np.zeros((nz, ny, nx)); qg = np.zeros((nz, ny, nx))
            qsoil = np.zeros((ny, nx))
            Tsoil = np.ascontiguousarray(300.0 + 40.0 * r.random((ny, nx)))
            sol = DOMRadiationSolver(nz, ny, nx, st["dy"], st["dx"],
                                     st["dz_arr"], st["d_face_above"],
                                     st["d_face_below"], "periodic", 4)
            sol.solve(T_s, T_g, alpha, om, 2000.0, 300.0, qs, qg,
                      Tsoil, qsoil, YH, rho, bedM)
            cases.append({
                "name": f"{name}_h2o{int(with_h2o)}", "nz": nz, "ny": ny, "nx": nx,
                "dx": st["dx"], "dy": st["dy"], "sigma_sav": 2000.0, "T_amb": 300.0,
                "with_h2o": with_h2o,
                "T_s": T_s.ravel().tolist(), "T_g": T_g.ravel().tolist(),
                "alpha_s": alpha.ravel().tolist(), "omega_comb": om.ravel().tolist(),
                "Y_H2O": (YH.ravel().tolist() if with_h2o else None),
                "rho": (rho.ravel().tolist() if with_h2o else None),
                "bed_moisture": bedM.ravel().tolist(),
                "T_soil": Tsoil.ravel().tolist(),
                "dz_arr": st["dz_arr"].tolist(),
                "q_rad_solid": qs.ravel().tolist(),
                "q_rad_gas": qg.ravel().tolist(),
                "q_in_soil": qsoil.ravel().tolist(),
            })
    return {"cases": cases}


def vec_lagrangian_bed():
    """The four Lagrangian bed-particle kernels.

    Particle-based, so the vectors carry flat per-slot arrays rather than the
    (Nz,Ny,Nx) fields every other kernel here uses.

    The initial particle state comes from the REAL initialiser, then is
    perturbed to spread the population across every branch that matters:

      - T_s spanning 290 K to 1100 K, so cells fall on both sides of
        T_SMOLD_ONSET (473 K) and T_CHAR_ONSET (600 K)
      - a fifth of the particles fully dry (m_water = 0), which switches off
        both the Arrhenius drying branch and the equilibrium override
      - pre-existing char on some particles, so char-ox has something to eat
        on step one rather than needing pyrolysis to run first
      - a handful pushed outside the domain, firing the retire-on-exit path
      - a handful driven under M_PARTICLE_BURNOUT, firing the burnout path
      - a low O2 patch that drops below Y_O2_MIN_OP, so oxidative pyrolysis
        switches off while thermal pyrolysis continues

    Two configurations, because the drying mode and the view-factor mode are
    genuinely different code paths and production uses the second of each:
      combined  = DRY_MODE_COMBINED + geometric view factor + ash penalty
      arrhenius = DRY_MODE_ARRHENIUS + scalar view factor + no ash penalty
    """
    from model_outdoor.physics_3d import lagrangian_bed_3d as lb

    cases = []
    configs = [
        ("combined", lb.DRY_MODE_COMBINED, True, 0.5),
        ("arrhenius", lb.DRY_MODE_ARRHENIUS, False, 0.0),
    ]
    for name, nz, ny, nx in SHAPES:
        for cname, dry_mode, geom, ash_exp in configs:
            st = make_state(nz, ny, nx, seed=4242)
            r = np.random.default_rng(8675309)
            dz = st["dz_arr"]
            dx, dy = st["dx"], st["dy"]
            z_face = np.concatenate(([0.0], np.cumsum(dz)))
            n_z_bed = max(nz // 3, 1)
            alpha = _bed(nz, ny, nx, r)

            n_per_cell = 4
            n_max = n_z_bed * ny * nx * n_per_cell
            buf = lb.allocate_bed_particle_buffers(n_max)
            n_alloc = lb.initialize_bed_particles_from_alpha_s(
                buf, alpha, rho_b_dry=0.9, moisture_frac=0.08, T_amb=300.0,
                dx=dx, dy=dy, dz_arr=dz, n_z_bed=n_z_bed,
                n_per_cell=n_per_cell, sav=lb.SAV_GRASS_DEFAULT,
            )
            # Snapshot the pristine initialiser output — that function is
            # verified on its own, before any perturbation.
            init_x = buf["x"].copy()
            init_y = buf["y"].copy()
            init_z = buf["z"].copy()
            init_ms = buf["m_solid"].copy()
            init_mw = buf["m_water"].copy()
            init_alive = buf["alive"].copy()

            # ── Perturb into a mid-fire population ──
            q = r.random(n_max)
            buf["T_s"][:] = 290.0 + 810.0 * q
            buf["m_char"][:] = buf["m_solid"] * 0.10 * r.random(n_max)
            buf["m_char_max"][:] = buf["m_char"] * (1.0 + 0.6 * r.random(n_max))
            dryers = q < 0.20
            buf["m_water"][dryers] = 0.0
            # A few sent out of the domain (retire path) and a few starved to
            # burnout (retire path's other half).
            if n_alloc > 12:
                buf["x"][3] = -1.0
                buf["y"][5] = (ny + 2) * dy
                buf["z"][7] = z_face[nz] + 1.0
                for p in (9, 11):
                    buf["m_solid"][p] = 1.0e-10
                    buf["m_water"][p] = 0.0
                    buf["m_char"][p] = 1.0e-10
                buf["alive"][13] = lb.ALIVE_FALSE   # already-dead slot skipped

            T_g = np.ascontiguousarray(300.0 + 1500.0 * r.random((nz, ny, nx)))
            Y_O2 = np.ascontiguousarray(0.23 * r.random((nz, ny, nx)))
            Y_O2[Y_O2 < 0.02] = 0.0005      # sub-Y_O2_MIN patch
            Q_ext = np.ascontiguousarray(1.0e4 * (r.random((nz, ny, nx)) - 0.4))

            inp = {k: buf[k].copy() for k in
                   ("x", "y", "z", "alive", "m_solid", "m_water", "m_char",
                    "T_s", "m_water_0", "sav", "m_char_max")}

            src = {k: np.zeros((nz, ny, nx)) for k in
                   ("S_pyro", "S_drying", "Q_pyro", "Q_drying", "Y_F_source",
                    "Q_char", "Q_smold", "Q_g_conv")}
            n_alive_out = np.zeros(1, dtype=np.int64)
            n_burned_out = np.zeros(1, dtype=np.int64)
            diag = np.zeros(16)

            h_bed = float(z_face[n_z_bed])
            lb.step_bed_particles(
                buf["x"], buf["y"], buf["z"], buf["alive"],
                buf["m_solid"], buf["m_water"], buf["m_char"], buf["T_s"],
                buf["m_water_0"], buf["sav"],
                T_g, Y_O2, Q_ext, n_per_cell,
                src["S_pyro"], src["S_drying"], src["Q_pyro"], src["Q_drying"],
                src["Y_F_source"], src["Q_char"], src["Q_smold"], src["Q_g_conv"],
                dx, dy, dz, z_face,
                lb.H_CONV_DEFAULT, lb.RHO_SOLID_TRUE_GRASS, lb.CP_SOLID_GRASS,
                lb.EPS_SOLID_DEFAULT, 300.0, 0.7, geom, h_bed, 4.5, 0.02,
                True, True, True, True,
                dry_mode, 1.0e5, ash_exp, buf["m_char_max"],
                n_alive_out, n_burned_out, diag,
            )

            # ── The two aggregators, run on the POST-step particle state ──
            T_s_grid = np.full((nz, ny, nx), 300.0)
            lb.aggregate_particles_to_T_s_grid(
                buf["x"], buf["y"], buf["z"], buf["alive"],
                buf["m_solid"], buf["m_water"], buf["m_char"], buf["T_s"],
                dx, dy, z_face, T_s_grid, 300.0,
            )
            M_grid = np.zeros((nz, ny, nx))
            lb.aggregate_particles_to_M_local_grid(
                buf["x"], buf["y"], buf["z"], buf["alive"],
                buf["m_solid"], buf["m_water"],
                dx, dy, z_face, M_grid,
            )

            # ── Horizontal conduction, on a copy so its inputs stay pinned ──
            cond_T_in = np.ascontiguousarray(T_s_grid.copy())
            cond_T = cond_T_in.copy()
            cond_part_T = buf["T_s"].copy()
            lb.step_horizontal_solid_conduction_scatter(
                buf["x"], buf["y"], buf["z"], buf["alive"],
                buf["m_solid"], buf["m_water"], buf["m_char"], cond_part_T,
                cond_T, alpha, dx, dy, z_face,
                0.09, lb.RHO_SOLID_TRUE_GRASS, lb.CP_SOLID_GRASS,
                n_z_bed, 0.02,
            )

            cases.append({
                "name": f"{name}_{cname}", "nz": nz, "ny": ny, "nx": nx,
                "dx": dx, "dy": dy, "dz_arr": dz.tolist(),
                "z_face": z_face.tolist(), "n_z_bed": n_z_bed,
                "n_per_cell": n_per_cell, "n_max": n_max, "n_alloc": n_alloc,
                "rho_b_dry": 0.9, "moisture_frac": 0.08, "T_amb": 300.0,
                "sav": lb.SAV_GRASS_DEFAULT, "h_bed": h_bed,
                "kappa_bed_eff": 4.5, "view_factor": 0.7,
                "view_factor_geometric": geom, "drying_mode": dry_mode,
                "char_ox_ash_exp": ash_exp, "char_ox_flux_cap": 1.0e5,
                "dt": 0.02, "h_conv": lb.H_CONV_DEFAULT,
                "rho_solid_true": lb.RHO_SOLID_TRUE_GRASS,
                "cp_solid": lb.CP_SOLID_GRASS, "eps_solid": lb.EPS_SOLID_DEFAULT,
                "k_solid": 0.09,
                "alpha_s": alpha.ravel().tolist(),
                "T_g": T_g.ravel().tolist(), "Y_O2": Y_O2.ravel().tolist(),
                "Q_solid_ext": Q_ext.ravel().tolist(),
                # initialiser output
                "init_x": init_x.tolist(), "init_y": init_y.tolist(),
                "init_z": init_z.tolist(), "init_m_solid": init_ms.tolist(),
                "init_m_water": init_mw.tolist(),
                "init_alive": init_alive.tolist(),
                # step inputs
                "in_x": inp["x"].tolist(), "in_y": inp["y"].tolist(),
                "in_z": inp["z"].tolist(),
                "in_alive": inp["alive"].tolist(),
                "in_m_solid": inp["m_solid"].tolist(),
                "in_m_water": inp["m_water"].tolist(),
                "in_m_char": inp["m_char"].tolist(),
                "in_T_s": inp["T_s"].tolist(),
                "in_m_water_0": inp["m_water_0"].tolist(),
                "in_sav": inp["sav"].tolist(),
                "in_m_char_max": inp["m_char_max"].tolist(),
                # step outputs
                "out_alive": buf["alive"].tolist(),
                "out_m_solid": buf["m_solid"].tolist(),
                "out_m_water": buf["m_water"].tolist(),
                "out_m_char": buf["m_char"].tolist(),
                "out_T_s": buf["T_s"].tolist(),
                "out_m_char_max": buf["m_char_max"].tolist(),
                "out_S_pyro": src["S_pyro"].ravel().tolist(),
                "out_S_drying": src["S_drying"].ravel().tolist(),
                "out_Q_pyro": src["Q_pyro"].ravel().tolist(),
                "out_Q_drying": src["Q_drying"].ravel().tolist(),
                "out_Y_F_source": src["Y_F_source"].ravel().tolist(),
                "out_Q_char": src["Q_char"].ravel().tolist(),
                "out_Q_smold": src["Q_smold"].ravel().tolist(),
                "out_Q_g_conv": src["Q_g_conv"].ravel().tolist(),
                "out_n_alive": int(n_alive_out[0]),
                "out_n_burned": int(n_burned_out[0]),
                "out_diag": diag.tolist(),
                # aggregators
                "out_T_s_grid": T_s_grid.ravel().tolist(),
                "out_M_local": M_grid.ravel().tolist(),
                # conduction
                "cond_T_in": cond_T_in.ravel().tolist(),
                "cond_T_out": cond_T.ravel().tolist(),
                "cond_part_T_out": cond_part_T.tolist(),
                # branch-coverage counters, printed by main()
                "n_dry": int((inp["m_water"] == 0.0).sum()),
                "n_above_char": int((inp["T_s"] >= 600.0).sum()),
                "n_lowO2": int((Y_O2 <= 0.001).sum()),
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
        "edc": vec_edc(),
        "poisson": vec_poisson(),
        "turb_diff": vec_turb_diff(),
        "kepsilon": vec_kepsilon(),
        "dom": vec_dom(),
        "lagrangian_bed": vec_lagrangian_bed(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    n = sum(len(c["rhs_out"]) for c in payload["muscl"]["cases"])
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} kB)")
    print(f"  muscl:   {len(payload['muscl']['cases'])} field cases "
          f"({n} values), {len(payload['muscl']['helpers'])} scalar probes")
    for c in payload["lagrangian_bed"]["cases"]:
        print(f"  bed:     {c['name']:18s} {c['n_alloc']:4d} particles, "
              f"{c['out_n_alive']} alive / {c['out_n_burned']} burned, "
              f"{c['n_dry']} dry, {c['n_above_char']} above char onset, "
              f"{c['n_lowO2']} low-O2 cells")
    for c in payload["dom"]["cases"]:
        pk = max(abs(v) for v in c["q_rad_solid"])
        print(f"  dom:     {c['name']:14s} |q_solid|max={pk:.4g} W/m2")
    for c in payload["kepsilon"]["cases"]:
        print(f"  k-eps:   {c['name']:8s} {c['nz']}x{c['ny']}x{c['nx']}, "
              f"{c['n_canopy']} canopy cells")
    for c in payload["turb_diff"]["cases"]:
        print(f"  turbdiff:{c['name']:12s} sc_t={c['sc_t']}, n_sub={c['n_sub']}")
    for c in payload["poisson"]["cases"]:
        import math as _m
        pk = max(abs(v) for v in c["p_out"])
        print(f"  poisson: {c['name']:8s} {c['nz']}x1x{c['nx']}, |p|max={pk:.4g}")
    for c in payload["edc"]["cases"]:
        print(f"  edc:     {c['name']:14s} {c['n_quenched']:3d} fully quenched, "
              f"{c['n_below_Tign']:3d} below T_ign, {c['n_capped']:3d} T-capped")
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
