"""FIRETEC-class coarsened Cheney comparison — LES vs RANS k-ε.

Approximates the regime FIRETEC/WFDS use for Cheney AU grassland
validation (Linn 2002, Pimont & Linn 2009, Mell 2007): coarse dx,
wider/longer domain, longer sim_t, bed treated as a thin porous
source layer.

Argument order: <turb_model> <n_z_bed> <label> <out_dir>
  turb_model ∈ {smagorinsky, k_epsilon}
  n_z_bed    ∈ {2, 4}      — bed cells

Config:
  Lx = 40 m,  Ly = 20 m,  Lz = 10 m
  dx = dy = 0.50 m
  n_z_bed (CLI) — dz_bed = 0.37/n_z_bed
  wall_BL_first = 5 mm  (low-Re wall layer)
  sim_t = 90 s   (sized to give post-pulse-steady fit window beyond 60 s)
  ignition_duration = 5 s, q_mult = 3, width_mult = 3

Reference comparisons:
  mickey refined KE Finney OFF:   10.17 / 3.03 m/min (apples-to-apples)
  mickey coarse  LES Finney OFF:  10.99 / 3.75 m/min (Ly=0.5m caveat)
  mickey coarse  KE  Finney OFF:  10.84 / 3.85 m/min (Ly=0.5m caveat)
  Cheney 1993 Fig 8 envelope at U=4 m/s, h_bed=0.37m, ρ_b=1.07: ~40–60 m/min
"""
import os
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_NUM_THREADS"]   = "12"
os.environ["NUMBA_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"]   = "12"

import json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/jw/projects/cheney-web")
sys.path.insert(0, str(ROOT))
from model_outdoor.spread_3d import run_3d_spread
from model.io.text_input import load_text_input

DECK = ROOT / "inputs/validation_cases/Outdoor_Grass_GR1__free_burn.txt"
IGNITION_DURATION_S = 5.0
MAX_SIM_TIME_S      = 90.0
POST_PULSE_FIT_START_S = 10.0   # skip ignition pulse + a few seconds buffer


def _ros_diagnostics(out_dir):
    """Return per-window ROS and steady-state detection diagnostics.

    Fits ROS in 10-s sliding windows from POST_PULSE_FIT_START_S to t_end.
    Reports per-window slope (m/min) so steady-state can be eyeballed.
    Reports a steady-state time as the first t where the window-slope
    relative-change vs the previous window is <5%.
    """
    snaps = sorted(out_dir.glob("snap_*.npz"))
    ts, xs = [], []
    for sp in snaps:
        s = np.load(sp)
        ts.append(float(s["t"]))
        xs.append(float(s["front_x"]))
    ts = np.asarray(ts); xs = np.asarray(xs)
    if len(ts) < 6:
        return {
            "ros_overall_m_min": float("nan"),
            "ros_post_pulse_m_min": float("nan"),
            "ros_final_window_m_min": float("nan"),
            "steady_state_t_s": float("nan"),
            "windows": [],
            "n_snaps": int(len(ts)),
        }

    # Overall fit (t > POST_PULSE_FIT_START_S)
    post_mask = ts >= POST_PULSE_FIT_START_S
    if int(post_mask.sum()) >= 3:
        slope_post, _ = np.polyfit(ts[post_mask], xs[post_mask], 1)
        ros_post = slope_post * 60.0
    else:
        ros_post = float("nan")

    # 10-s sliding windows
    win_s = 10.0
    win_step_s = 5.0
    windows = []
    t_start = POST_PULSE_FIT_START_S
    while t_start + win_s <= ts[-1]:
        m = (ts >= t_start) & (ts < t_start + win_s)
        if int(m.sum()) >= 3:
            slope_w, _ = np.polyfit(ts[m], xs[m], 1)
            windows.append({"t0": t_start, "t1": t_start + win_s,
                            "ros_m_min": float(slope_w * 60.0),
                            "n": int(m.sum())})
        t_start += win_step_s

    # Steady-state: first window where change vs prev window < 5% (and ≥ 3 windows)
    steady_t = float("nan")
    if len(windows) >= 3:
        for k in range(1, len(windows)):
            prev = windows[k - 1]["ros_m_min"]
            cur  = windows[k]["ros_m_min"]
            if prev > 0 and abs(cur - prev) / prev < 0.05:
                steady_t = float(windows[k]["t0"])
                break

    return {
        "ros_post_pulse_m_min": float(ros_post),
        "ros_final_window_m_min": float(windows[-1]["ros_m_min"]) if windows else float("nan"),
        "steady_state_t_s": steady_t,
        "windows": windows,
        "n_snaps": int(len(ts)),
        "t_end_s": float(ts[-1]),
    }


def main():
    turb_model = sys.argv[1]
    n_z_bed    = int(sys.argv[2])
    label      = sys.argv[3]
    out_dir    = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    ri = load_text_input(DECK)
    ri.outdoor_overrides["bulk_density_kg_m3"]    = 1.07
    ri.outdoor_overrides["fuel_depth_m"]          = 0.37
    ri.outdoor_overrides["initial_moisture_frac"] = 0.04
    ri.outdoor_overrides["sav_ratio_1_m"]         = 2000.0
    ri.outdoor_overrides["ignition_duration_s"]   = IGNITION_DURATION_S
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)
    # FIRETEC/WFDS treat bed as sub-grid porous layer with NO resolved
    # wall BL.  Drop it here for both n_z_bed=2 and n_z_bed=4 — the bed
    # cells ARE the wall layer.  This relaxes the dt constraint that the
    # 5 mm wall_BL_first_dz was forcing (dt~2.5e-4 s → JIT+sim cost too
    # high to run 4 configs).  Aspect dx/dz_bed:
    #   n_z_bed=2: 0.5/0.185 = 2.7:1  (within target 5:1)
    #   n_z_bed=4: 0.5/0.0925 = 5.4:1 (at target 5:1)
    ri.outdoor_overrides["wall_bl_N"]        = 0
    ri.outdoor_overrides["wall_bl_first_dz"] = 0.0
    ri.outdoor_overrides["wall_bl_growth"]   = 1.0
    ri.outdoor_overrides["atm_growth"]       = 1.30
    ri.outdoor_overrides["atm_max_dz"]       = 2.0

    print(f"\n=== FIRETEC-class {label}  turb={turb_model}  n_z_bed={n_z_bed}  "
          f"Lx=40 Ly=20 Lz=10  dx=0.5  sim_t={MAX_SIM_TIME_S}s  "
          f"q=3 w=3 ===", flush=True)
    t0 = time.time()
    r = run_3d_spread(
        ri,
        wind_speed_m_s=4.0,
        Lx=40.0, Ly=20.0, Lz=10.0, dx=0.50, dy=0.50,
        n_z_bed=n_z_bed,
        cfl_factor=0.40, max_wall_time_s=MAX_SIM_TIME_S,
        y_bc="periodic",
        turbulence_model=turb_model,
        wall_function=False,
        combustion_closure="level_set_fsd",
        wind_profile_type="log_law",
        bed_x_start=2.0, bed_x_end=38.0,
        projection_method="fft_pcg",
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        ignition_q_mult=3.0,
        ignition_width_mult=3.0,
        finney_tendril_enable=False,
        snapshot_dir=out_dir,
        snapshot_interval_s=1.0,
    )
    wall = time.time() - t0
    ros_overall = float(r.ros_m_s) * 60.0
    diag = _ros_diagnostics(out_dir)
    result = {
        "label": label,
        "phase": "firetec_class_LES_vs_KE",
        "turbulence_model": turb_model,
        "Lx_m": 40.0, "Ly_m": 20.0, "Lz_m": 10.0,
        "dx_m": 0.50, "dy_m": 0.50,
        "n_z_bed": n_z_bed,
        "wall_bl_N": int(ri.outdoor_overrides.get("wall_bl_N", 0)),
        "wall_bl_first_dz": float(ri.outdoor_overrides.get("wall_bl_first_dz", 0.0)),
        "wall_bl_growth": float(ri.outdoor_overrides.get("wall_bl_growth", 1.0)),
        "finney_tendril_enable": False,
        "ignition_q_mult": 3.0, "ignition_width_mult": 3.0,
        "sim_t_target_s": MAX_SIM_TIME_S,
        "ros_overall_m_min": ros_overall,
        "wall_s": wall,
        **diag,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {label}  turb={turb_model} n_z_bed={n_z_bed}  "
          f"ROS_overall={ros_overall:.2f}  "
          f"ROS_post_pulse={diag['ros_post_pulse_m_min']:.2f}  "
          f"ROS_final_win={diag['ros_final_window_m_min']:.2f}  "
          f"steady_t={diag['steady_state_t_s']:.1f}s  "
          f"t_end={diag['t_end_s']:.1f}s  wall={wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
