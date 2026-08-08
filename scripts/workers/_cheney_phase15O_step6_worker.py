"""Phase 15O Step 6 — Cheney 4-case quick verify with tendril ON.

Single-case runner: takes Nat/Cut, moisture, U as args. Runs Lx=40m
production geometry with Phase 15O committed values
(Sr=0.20, duty=0.40, f_mass=0.05, fr_min=0.5) per Rule #2.

Expected outcome (from mickey Steps 2-4): ≈0% lift over Phase 14ax
18/20 PASS baseline. Will be in Cheney Eq.6 ratio band if pre-15O was.

CLI: _cheney_phase15O_step6_worker.py LABEL FUEL_TYPE MF_PCT U OUT_DIR
       FUEL_TYPE in {Nat, Cut}
       MF_PCT in {4, 8}
       U in m/s
"""
import os
os.environ.setdefault("MKL_CBWR", "AVX2")
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_NUM_THREADS"]   = "12"
os.environ["NUMBA_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"]   = "12"

import json, math, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/jw/projects/cheney-web")
sys.path.insert(0, str(ROOT))
from model_outdoor.spread_3d import run_3d_spread
from model.io.text_input import load_text_input

DECK = ROOT / "inputs/validation_cases/Outdoor_Grass_GR1__free_burn.txt"

# Cheney 1993 Eq.6 reference (the 18/20 PASS canonical):
#   ROS_ref = a_ch · u2^0.987 · exp(-0.0707·mf_pct)  [m/s], multiplied by 60 for m/min
#   u2 = 0.723 × U_10m  (midflame conversion)
#   a_ch_nat = 0.406, a_ch_cut = 0.343
A_CH = {"Nat": 0.406, "Cut": 0.343}

# Per-fuel-type bed parameters (canonical Phase 14ax):
BED = {
    "Nat": dict(rho_b=1.07, h_bed=0.37),
    "Cut": dict(rho_b=1.07, h_bed=0.10),   # Cut has shorter bed
}


def main():
    label   = sys.argv[1]
    fuel    = sys.argv[2]  # "Nat" or "Cut"
    mf_pct  = float(sys.argv[3])
    U       = float(sys.argv[4])
    out_dir = Path(sys.argv[5])
    out_dir.mkdir(parents=True, exist_ok=True)

    bed_cfg = BED[fuel]
    a_ch    = A_CH[fuel]
    mf_frac = mf_pct / 100.0

    ri = load_text_input(DECK)
    ri.outdoor_overrides["bulk_density_kg_m3"]    = bed_cfg["rho_b"]
    ri.outdoor_overrides["fuel_depth_m"]          = bed_cfg["h_bed"]
    ri.outdoor_overrides["initial_moisture_frac"] = mf_frac
    ri.outdoor_overrides["sav_ratio_1_m"]         = 2000.0
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)
    # Phase 14ax canonical: no wall_bl refinement, uniform bed cells
    ri.outdoor_overrides.pop("wall_bl_N", None)
    ri.outdoor_overrides.pop("wall_bl_first_dz", None)
    ri.outdoor_overrides.pop("wall_bl_growth", None)
    ri.outdoor_overrides["atm_growth"]       = 1.20
    ri.outdoor_overrides["atm_max_dz"]       = 1.0

    print(f"\n=== STEP6-CHENEY {label}  {fuel} {mf_pct}% U={U}  TENDRIL=ON  "
          f"committed Sr=0.20 ===", flush=True)
    t0 = time.time()
    r = run_3d_spread(
        ri,
        wind_speed_m_s=U,
        Lx=40.0, Ly=0.5, Lz=8.0, dx=0.10, dy=0.10,
        n_z_bed=4,    # production canonical
        cfl_factor=0.40, max_wall_time_s=15.0,
        y_bc="periodic", turbulence_model="k_epsilon",
        wall_function=False,
        combustion_closure="level_set_fsd",
        wind_profile_type="log_law",
        bed_x_start=2.0, bed_x_end=27.0,
        projection_method="fft_pcg",
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        finney_tendril_enable=True,
        snapshot_dir=out_dir,
        snapshot_interval_s=1.0,
    )
    wall = time.time() - t0
    ros_m_min = float(r.ros_m_s) * 60.0

    # Cheney Eq.6 reference + ratio
    u2 = U * 0.723
    ref = a_ch * u2**0.987 * math.exp(-0.0707 * mf_pct) * 60.0 if u2 > 0 else 0.0
    ratio = ros_m_min / ref if ref > 0 else float("nan")
    ok = (1.0 / 3.0) <= ratio <= 3.0  # Cheney Eq.6 [1/3, 3] band

    result = {
        "label": label, "fuel": fuel, "mf_pct": mf_pct, "U_m_s": U,
        "finney_tendril_enable": True,
        "sr": 0.20, "duty_cycle": 0.40, "f_mass": 0.05, "fr_min": 0.5,
        "ros_m_min": ros_m_min,
        "cheney_eq6_ref_m_min": ref,
        "ratio_to_ref": ratio,
        "PASS_eq6_band": ok,
        "wall_s": wall,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    pass_str = "PASS" if ok else "FAIL"
    print(f"\n[saved] {label}  ROS={ros_m_min:.3f}  ref={ref:.3f}  "
          f"ratio={ratio:.3f}  {pass_str}  wall={wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
