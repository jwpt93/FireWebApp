"""Wide-Y Cheney case (Option A) — LES vs RANS-KE at mickey resolution.

Compared to mickey (Ly=0.5m), this widens cross-stream by 8× to give
Finney-finger structure (Finney 2015 measured ~20 cm spacing → ~20
fingers fit in 4 m).  Same dx, n_z_bed, wall-BL refinement as the
clean refined REF, so any LES vs RANS gap is attributable to the
turbulence model alone, not grid effects.

Config:
  Lx = 10 m,  Ly = 4 m,  Lz = 3 m
  dx = dy = 0.10 m
  n_z_bed = 18, wall_BL = (10 cells, 5 mm first, 1.2 growth)
  sim_t = 20 s (5 s ignition + 15 s burn)

Comparison reference (mickey Ly=0.5m, refined, Finney OFF):
  KE coarse n_z_bed=4:   10.84 / 3.85
  LES coarse n_z_bed=4:  10.99 / 3.75
  KE refined n_z_bed=18: 10.17 / 3.03   (clean REF)

Wide-Y comparison adds the wall-budget for actual finger-resolved LES
to potentially diverge from RANS.
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
MAX_WALL_TIME_S = 20.0
POST_PULSE_FIT_START_S = 6.0


def _post_pulse_ros(out_dir):
    snaps = sorted(out_dir.glob("snap_*.npz"))
    t_list, x_list = [], []
    for sp in snaps:
        s = np.load(sp)
        if float(s["t"]) >= POST_PULSE_FIT_START_S:
            t_list.append(float(s["t"]))
            x_list.append(float(s["front_x"]))
    if len(t_list) < 3:
        return float("nan"), len(t_list)
    slope, _ = np.polyfit(np.asarray(t_list), np.asarray(x_list), 1)
    return float(slope) * 60.0, len(t_list)


def main():
    turb_model = sys.argv[1]   # "smagorinsky" or "k_epsilon"
    label = sys.argv[2]
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    ri = load_text_input(DECK)
    ri.outdoor_overrides["bulk_density_kg_m3"]    = 1.07
    ri.outdoor_overrides["fuel_depth_m"]          = 0.37
    ri.outdoor_overrides["initial_moisture_frac"] = 0.04
    ri.outdoor_overrides["sav_ratio_1_m"]         = 2000.0
    ri.outdoor_overrides["ignition_duration_s"]   = IGNITION_DURATION_S
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)
    ri.outdoor_overrides["wall_bl_N"]        = 10
    ri.outdoor_overrides["wall_bl_first_dz"] = 0.005
    ri.outdoor_overrides["wall_bl_growth"]   = 1.20
    ri.outdoor_overrides["atm_growth"]       = 1.20
    ri.outdoor_overrides["atm_max_dz"]       = 1.0

    print(f"\n=== WIDE-Y {label}  turb={turb_model}  Lx=10 Ly=4 Lz=3  "
          f"dx=0.10  n_z_bed=18  FINNEY=OFF  q=3 w=3 ===", flush=True)
    t0 = time.time()
    r = run_3d_spread(
        ri,
        wind_speed_m_s=4.0,
        Lx=10.0, Ly=4.0, Lz=3.0, dx=0.10, dy=0.10,
        n_z_bed=18,
        cfl_factor=0.40, max_wall_time_s=MAX_WALL_TIME_S,
        y_bc="periodic",
        turbulence_model=turb_model,
        wall_function=False,
        combustion_closure="level_set_fsd",
        wind_profile_type="log_law",
        bed_x_start=1.0, bed_x_end=9.0,
        projection_method="fft_pcg",
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        ignition_q_mult=3.0,
        ignition_width_mult=3.0,
        finney_tendril_enable=False,
        snapshot_dir=out_dir,
        snapshot_interval_s=0.5,
    )
    wall = time.time() - t0
    ros_overall = float(r.ros_m_s) * 60.0
    ros_post, n_post = _post_pulse_ros(out_dir)
    result = {
        "label": label,
        "phase": "wideY_LES_vs_KE",
        "turbulence_model": turb_model,
        "Lx_m": 10.0, "Ly_m": 4.0, "Lz_m": 3.0,
        "dx_m": 0.10, "dy_m": 0.10,
        "n_z_bed": 18,
        "wall_bl_N": 10, "wall_bl_first_dz": 0.005, "wall_bl_growth": 1.20,
        "finney_tendril_enable": False,
        "ignition_q_mult": 3.0, "ignition_width_mult": 3.0,
        "sim_t_s": MAX_WALL_TIME_S,
        "ros_overall_m_min": ros_overall,
        "ros_post_pulse_m_min": ros_post,
        "n_post_pulse_snaps": n_post,
        "wall_s": wall,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {label}  turb={turb_model}  ROS_overall={ros_overall:.2f}  "
          f"ROS_post_pulse={ros_post:.2f} m/min  wall={wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
