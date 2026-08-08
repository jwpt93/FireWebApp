"""Cheney Nat 4% U=0.5 with REFINED dx=0.025m test.

Tests the cell-residence-vs-EDC-timescale hypothesis: at production
dx=0.10m and U=0.5, gas spends ~670ms (~130 EDC timescales) in each
cell.  Y_F gets consumed locally; no advection chain to next cell →
production U=0.5 fails (ratio 0.046 FAIL vs Eq.6).

This worker uses dx=0.025m (4× finer in x), keeping dy=0.10 and
n_z_bed=8 unchanged.  Tractability: shrink Lx 40 → 15m + sim_t 30 → 12s
to keep wall time manageable (~2-3 hr at 8 cores).

Compares directly to the production Nat4_U0.5 case from the just-run
baseline.
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

# Cheney 1993 Eq.6
A_CH = {"Nat": 0.406, "Cut": 0.343}


def cheney_eq6_m_min(fuel, mf_pct, U):
    u2 = U * 0.723
    return A_CH[fuel] * u2**0.987 * math.exp(-0.0707 * mf_pct) * 60.0


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    fuel    = "Nat"
    mf_pct  = 4.0
    U       = 0.5
    DX      = 0.025          # ← the test: 4× finer than production
    LX      = 15.0           # ← reduce domain to keep wall time tractable
    SIM_T   = 12.0           # ← shorter than production 30s

    ri = load_text_input(DECK)
    ri.outdoor_overrides["bulk_density_kg_m3"]    = 1.07
    ri.outdoor_overrides["fuel_depth_m"]          = 0.37
    ri.outdoor_overrides["initial_moisture_frac"] = mf_pct / 100.0
    ri.outdoor_overrides["sav_ratio_1_m"]         = 2000.0
    ri.outdoor_overrides["canopy_C_d"]            = 0.30
    ri.outdoor_overrides["wall_bl_N"]             = 0
    ri.outdoor_overrides["wall_bl_first_dz"]      = 0.0
    ri.outdoor_overrides["wall_bl_growth"]        = 1.0
    ri.outdoor_overrides["atm_growth"]            = 1.20
    ri.outdoor_overrides["atm_max_dz"]            = 1.0
    ri.outdoor_overrides.pop("dz_first", None)
    ri.outdoor_overrides.pop("bl_growth", None)

    print(f"\n=== Cheney Nat 4% U=0.5  REFINED dx={DX}m  Lx={LX}m  "
          f"sim_t={SIM_T}s ===", flush=True)

    t0 = time.time()
    r = run_3d_spread(
        ri, wind_speed_m_s=U,
        Lx=LX, Ly=0.5, Lz=8.0, dx=DX, dy=0.10,
        n_z_bed=8,
        cfl_factor=0.40, max_wall_time_s=SIM_T,
        y_bc="periodic", turbulence_model="k_epsilon",
        wall_function=False, combustion_closure="edc",
        wind_profile_type="log_law",
        bed_x_start=1.0, bed_x_end=LX - 3.0,    # 1m upstream, 3m downstream
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
    )
    wall = time.time() - t0
    ros_m_min = float(r.ros_m_s) * 60.0

    # T_s-based bed-ignition front (the metric matched in the baseline sweep).
    snaps = sorted(out_dir.glob("snap_*.npz"))
    T_IGN = 600.0
    ts_t, ts_x = [], []
    m0 = float(np.load(snaps[0])["bp_m_solid"].sum()) if snaps else 0.0
    for sp in snaps:
        s = np.load(sp)
        x_mid = s["x_mid"]
        col_burned = (s["T_s"] >= T_IGN).any(axis=(0, 1))
        if col_burned.any():
            ts_t.append(float(s["t"]))
            ts_x.append(float(x_mid[np.where(col_burned)[0].max()]))
    if len(ts_t) >= 3:
        mask = [i for i, t in enumerate(ts_t) if t >= 1.0]
        if len(mask) >= 3:
            ros_Ts = float(np.polyfit([ts_t[i] for i in mask],
                                       [ts_x[i] for i in mask], 1)[0]) * 60.0
        else:
            ros_Ts = float("nan")
    else:
        ros_Ts = float("nan")
    mF = float(np.load(snaps[-1])["bp_m_solid"].sum()) if snaps else 0.0
    frac = (m0 - mF) / m0 if m0 > 0 else 0.0

    eq6 = cheney_eq6_m_min(fuel, mf_pct, U)
    result = dict(
        test="refined_dx_0p025_Nat4_U0p5",
        Lx=LX, dx=DX, sim_t_s=SIM_T,
        ROS_m_min=ros_m_min,
        ROS_Ts_m_min=ros_Ts,
        eq6_ref_m_min=eq6,
        eq6_ratio_Ts=ros_Ts / eq6 if eq6 > 0 else float("nan"),
        frac_consumed=frac,
        wall_s=wall,
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[done] refined_dx  ROS_Ts={ros_Ts:.2f} m/min  "
          f"frac={frac*100:.1f}%  Eq.6={eq6:.2f}  ratio={result['eq6_ratio_Ts']:.3f}  "
          f"wall={wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
