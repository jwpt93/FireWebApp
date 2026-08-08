"""Cheney Nat 4% U=1 ignition-method test.

Three modes:
  baseline   default: ign_dur=5s, T_pin_K=1500, width_mult=3, gas pin ON
  line       narrow brief gas pin: ign_dur=0.5s, width_mult=0.5 (one cell line)
  solid      solid-phase ignition: bed T_s pre-heated, gas pin OFF

CLI: _cheney_low_U_ignition_test_worker.py MODE OUT_DIR
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


def cheney_eq6_m_min(mf_pct, U):
    u2 = U * 0.723
    return 0.406 * u2**0.987 * math.exp(-0.0707 * mf_pct) * 60.0


MODES = {
    "baseline": dict(
        ign_dur=5.0, width_mult=3.0,
        T_pin_enable=True, T_pin_K=1500.0, T_pin_ramp_s=0.5,
        solid_ignite=False,
    ),
    "line": dict(
        ign_dur=0.5, width_mult=0.5,
        T_pin_enable=True, T_pin_K=1500.0, T_pin_ramp_s=0.1,
        solid_ignite=False,
    ),
    "solid": dict(
        ign_dur=0.0, width_mult=1.0,
        T_pin_enable=False, T_pin_K=300.0, T_pin_ramp_s=0.0,
        solid_ignite=True,
    ),
}


def main():
    mode    = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = MODES[mode]

    U = 1.0
    MF_PCT = 4
    SIM_T = 25.0

    ri = load_text_input(DECK)
    for k, v in {
        "bulk_density_kg_m3":    1.07,
        "fuel_depth_m":          0.37,
        "initial_moisture_frac": MF_PCT / 100.0,
        "sav_ratio_1_m":         2000.0,
        "canopy_C_d":            0.30,
        "ignition_duration_s":   cfg["ign_dur"],
        "wall_bl_N":             0,
        "wall_bl_first_dz":      0.0,
        "wall_bl_growth":        1.0,
        "atm_growth":            1.20,
        "atm_max_dz":            1.0,
    }.items():
        ri.outdoor_overrides[k] = v
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)

    print(f"\n=== Cheney Nat 4% U=1  ignition_mode={mode}  ===", flush=True)
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
        ignition_width_mult=cfg["width_mult"],
        finney_tendril_enable=False,
        lagrangian_bed_enable=True,
        lagrangian_bed_N_per_cell=20,
        lagrangian_bed_h_conv=250.0,
        lagrangian_bed_view_factor=1.0,
        lagrangian_bed_view_factor_geometric=True,
        lagrangian_bed_drying_mode="combined",
        ignition_T_pin_enable=cfg["T_pin_enable"],
        ignition_T_pin_K=cfg["T_pin_K"],
        ignition_T_pin_height_m=0.30,
        ignition_T_pin_ramp_s=cfg["T_pin_ramp_s"],
        min_dt_s=1.0e-4,
        snapshot_dir=out_dir,
        snapshot_interval_s=1.0,
        level_set_passive=True,
        solid_phase_ignition_enable=cfg["solid_ignite"],
        solid_phase_ignition_T_s_K=1000.0,   # well above T_ign=600K
    )
    wall = time.time() - t0

    snaps = sorted(out_dir.glob("snap_*.npz"))
    T_IGN = 600.0
    ts_t, ts_x, u_lead_t, u_ahead_t = [], [], [], []
    m0 = float(np.load(snaps[0])["bp_m_solid"].sum())
    for sp in snaps:
        s = np.load(sp)
        x_mid = s["x_mid"]
        col_burned = (s["T_s"] >= T_IGN).any(axis=(0, 1))
        if col_burned.any():
            ts_t.append(float(s["t"]))
            ts_x.append(float(x_mid[np.where(col_burned)[0].max()]))
        # u_bed at source-patch lead + ahead
        u = s["u"]
        y_mid = u.shape[1] // 2
        i_lead = np.argmin(np.abs(x_mid - 3.5))
        i_ahead = min(i_lead + 5, len(x_mid) - 1)
        u_lead_t.append((float(s["t"]),
                         float(u[:8, y_mid, i_lead].mean()),
                         float(u[:8, y_mid, i_ahead].mean())))
    if len(ts_t) >= 3:
        mask = [i for i, t in enumerate(ts_t) if t >= 1.0]
        if len(mask) >= 3:
            ros_Ts = float(np.polyfit([ts_t[i] for i in mask],
                                       [ts_x[i] for i in mask], 1)[0]) * 60.0
        else:
            ros_Ts = float("nan")
    else:
        ros_Ts = float("nan")
    mF = float(np.load(snaps[-1])["bp_m_solid"].sum())
    frac = (m0 - mF) / m0

    # End-of-sim u_bed
    s = np.load(snaps[-1])
    u = s["u"]; x_mid = s["x_mid"]
    y_mid = u.shape[1] // 2
    i_lead = np.argmin(np.abs(x_mid - 3.5))
    i_ahead = min(i_lead + 5, len(x_mid) - 1)
    u_bed_lead  = float(u[:8, y_mid, i_lead].mean())
    u_bed_ahead = float(u[:8, y_mid, i_ahead].mean())

    eq6 = cheney_eq6_m_min(MF_PCT, U)
    ratio = ros_Ts / eq6 if eq6 > 0 and ros_Ts == ros_Ts else float("nan")

    result = dict(
        mode=mode, **cfg,
        ROS_Ts_m_min=ros_Ts, eq6_ref_m_min=eq6, eq6_ratio=ratio,
        frac_consumed=frac,
        u_bed_lead_final=u_bed_lead,
        u_bed_ahead_final=u_bed_ahead,
        u_bed_trajectory=u_lead_t,
        wall_s=wall,
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[done] {mode}  ROS_Ts={ros_Ts:.2f} m/min  "
          f"u_bed lead={u_bed_lead:+.2f}  ahead={u_bed_ahead:+.2f}  "
          f"frac={frac*100:.1f}%  ratio={ratio:.3f}  wall={wall/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
