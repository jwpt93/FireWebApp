"""Phase 17c — level-set passive test on Cheney Nat 4% U=4.

Same setup as _cheney_phase16_worker.py but with level_set_passive=True
so the level-set v_n forcing is zero.  The bed must self-ignite ahead
of the source patch via the gas-side CFD (advection + DOM + Frankman
+ h_conv) + bed coupling alone.

Tests the hypothesis that the kinematic v_n = q_in / E_ign surrogate
(Mell 2007 §3.4) was the dominant ROS driver, hiding what the CFD
itself could do.

CLI: _cheney_levelset_passive_test_worker.py LABEL FUEL MF_PCT U OUT_DIR
"""
import os
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_NUM_THREADS"]      = "8"
os.environ["NUMBA_NUM_THREADS"]    = "8"
os.environ["OMP_NUM_THREADS"]      = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json, math, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/jw/projects/cheney-web")
sys.path.insert(0, str(ROOT))
from model_outdoor.spread_3d import run_3d_spread
from model.io.text_input import load_text_input

DECK = ROOT / "inputs/validation_cases/Outdoor_Grass_GR1__free_burn.txt"

A_CH = {"Nat": 0.406, "Cut": 0.343}
BED = {
    "Nat": dict(rho_b=1.07, h_bed=0.37, sav=2000.0),
    "Cut": dict(rho_b=1.07, h_bed=0.10, sav=2000.0),
}


def cheney_eq6_m_min(fuel, mf_pct, U):
    u2 = U * 0.723
    if u2 <= 0.0:
        return 0.0
    return A_CH[fuel] * u2**0.987 * math.exp(-0.0707 * mf_pct) * 60.0


def main():
    label   = sys.argv[1]
    fuel    = sys.argv[2]
    mf_pct  = float(sys.argv[3])
    U       = float(sys.argv[4])
    out_dir = Path(sys.argv[5])
    out_dir.mkdir(parents=True, exist_ok=True)

    bed_cfg = BED[fuel]
    ri = load_text_input(DECK)
    ri.outdoor_overrides["bulk_density_kg_m3"]    = bed_cfg["rho_b"]
    ri.outdoor_overrides["fuel_depth_m"]          = bed_cfg["h_bed"]
    ri.outdoor_overrides["initial_moisture_frac"] = mf_pct / 100.0
    ri.outdoor_overrides["sav_ratio_1_m"]         = bed_cfg["sav"]
    ri.outdoor_overrides["canopy_C_d"]            = 0.30
    ri.outdoor_overrides["wall_bl_N"]             = 0
    ri.outdoor_overrides["wall_bl_first_dz"]      = 0.0
    ri.outdoor_overrides["wall_bl_growth"]        = 1.0
    ri.outdoor_overrides["atm_growth"]            = 1.20
    ri.outdoor_overrides["atm_max_dz"]            = 1.0
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)

    SIM_T = {0.5: 30.0, 1.0: 25.0, 2.0: 20.0, 4.0: 15.0, 8.0: 12.0}.get(U, 15.0)
    print(f"\n=== LEVEL-SET PASSIVE Cheney  {label}  {fuel} {mf_pct}% U={U}  "
          f"sim_t={SIM_T}s  ===", flush=True)

    t0 = time.time()
    r = run_3d_spread(
        ri, wind_speed_m_s=U,
        Lx=40.0, Ly=0.5, Lz=8.0, dx=0.10, dy=0.10,
        n_z_bed=8,
        cfl_factor=0.40, max_wall_time_s=SIM_T,
        y_bc="periodic", turbulence_model="k_epsilon",
        wall_function=False, combustion_closure="edc",
        wind_profile_type="log_law",
        bed_x_start=2.0, bed_x_end=37.0,
        projection_method="fft_pcg",
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        ignition_q_mult=3.0,
        ignition_width_mult=3.0,
        finney_tendril_enable=False,
        lagrangian_bed_enable=True,
        lagrangian_bed_N_per_cell=20,
        lagrangian_bed_h_conv=250.0,
        lagrangian_bed_view_factor=1.0,
        lagrangian_bed_view_factor_geometric=True,
        lagrangian_bed_drying_mode="combined",
        ignition_T_pin_enable=True,
        ignition_T_pin_K=1500.0,
        ignition_T_pin_height_m=0.30,
        ignition_T_pin_ramp_s=0.5,
        min_dt_s=1.0e-4,
        snapshot_dir=out_dir,
        snapshot_interval_s=1.0,
        level_set_passive=True,         # <-- the test
    )
    wall = time.time() - t0
    ros_m_min = float(r.ros_m_s) * 60.0
    eq6 = cheney_eq6_m_min(fuel, mf_pct, U)

    # ROS by T_s-based bed front (since lset is locked) — find max x with
    # any bed cell ≥ T_ign over time.
    snaps = sorted(out_dir.glob("snap_*.npz"))
    T_IGN = 600.0
    ts_front_t, ts_front_x = [], []
    # Also track omega-based and frac_consumed.
    m0 = None
    om_front_t, om_front_x = [], []
    for sp in snaps:
        s = np.load(sp)
        t = float(s["t"])
        T_s = s["T_s"]              # (Nz, Ny, Nx)
        x_mid = s["x_mid"]
        any_ignited_in_col = (T_s >= T_IGN).any(axis=(0, 1))
        if any_ignited_in_col.any():
            i_lead = int(np.where(any_ignited_in_col)[0].max())
            ts_front_t.append(t); ts_front_x.append(float(x_mid[i_lead]))
        om = s["omega"]
        active_col = (om > 1.0e-2).any(axis=(0, 1))
        if active_col.any():
            i_lead = int(np.where(active_col)[0].max())
            om_front_t.append(t); om_front_x.append(float(x_mid[i_lead]))
        if m0 is None and "bp_m_solid" in s.files:
            m0 = float(s["bp_m_solid"].sum())
        mF = float(s["bp_m_solid"].sum()) if "bp_m_solid" in s.files else None
    frac = (m0 - mF) / m0 if (m0 and mF is not None and m0 > 0) else 0.0
    # Linear-fit ROS post-ignition (skip first 1s pin)
    def fit_ros(ts, xs, t_skip=1.0):
        mask = [i for i, t in enumerate(ts) if t >= t_skip]
        if len(mask) >= 3:
            return float(np.polyfit([ts[i] for i in mask], [xs[i] for i in mask], 1)[0]) * 60.0
        return float("nan")
    ros_ts = fit_ros(ts_front_t, ts_front_x)
    ros_om = fit_ros(om_front_t, om_front_x)

    result = dict(
        label=label, fuel=fuel, mf_pct=mf_pct, U_m_s=U,
        sim_t_s=SIM_T,
        ROS_lset_m_min=ros_m_min,    # should be 0 or near 0 (level set locked)
        ROS_Ts_m_min=ros_ts,         # bed self-ignition front rate
        ROS_omega_m_min=ros_om,
        eq6_ref_m_min=eq6,
        frac_consumed=frac,
        wall_s=wall,
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {label} (level_set_passive=True)  "
          f"ROS_lset={ros_m_min:.2f}  ROS_Ts={ros_ts:.2f}  "
          f"ROS_omega={ros_om:.2f} m/min  "
          f"Eq.6={eq6:.2f}  ratio_Ts={ros_ts/eq6 if eq6>0 else float('nan'):.3f}  "
          f"frac={frac*100:.1f}%  wall={wall/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
