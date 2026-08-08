"""Generic Phase 16 Cheney worker — one case at a time, CLI-driven.

Use the Phase 16 Lagrangian-bed kernel + the A_p/Q-cap bug fixes that
landed 2026-06-15 (commit daf9023).  Sub-grid bed particles own the
solid; gas-side stays Eulerian with EDC combustion + FFT-PCG projection.

CLI: _cheney_phase16_worker.py LABEL FUEL MF_PCT U OUT_DIR [DRYING_MODE]

  FUEL          "Nat" | "Cut"
  MF_PCT        4 | 8
  U             m/s (typical 0.5 1 2 4 8)
  DRYING_MODE   "arrhenius" (default; Lautenberger 2009 bound water) |
                "equilibrium" (FIRETEC heat-rate-limited; Linn 2002)

Saves snapshots + result.json in OUT_DIR.
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

# Cheney 1993 Eq. 6:  ROS = a_ch · u2^0.987 · exp(-0.0707·mf_pct)  [m/s]
A_CH = {"Nat": 0.406, "Cut": 0.343}

# Per-fuel-type bed parameters (canonical Phase 14ax + Phase 15O Step 6
# convention: Cut uses h_bed=0.10, not 0.15 of the original 14x table).
BED = {
    "Nat": dict(rho_b=1.07, h_bed=0.37, sav=2000.0),
    "Cut": dict(rho_b=1.07, h_bed=0.10, sav=2000.0),
}

# Approximate EXP envelope (m/min) from Cheney Fig 8 bins.
# Source: validation_datasets/Papers/cheney1993_fig8_data_v2.json bin-mean
# ± stddev at the matching (mf, fuel, U).  These are the per-bin envelope
# rounded values used by the canonical Cheney plot script.
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
    drying_mode = sys.argv[6] if len(sys.argv) > 6 else "arrhenius"
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
    # Phase 20: drip-torch pulse duration reduced from base-deck default
    # 30s to physical 3s (Pyne 1993 §11.3).  Pin also gated below by U
    # threshold (see run_3d_spread call).  Solid-phase ignition alone is
    # enough at low U (empirical-ROS drives the level-set regardless),
    # but at high U (>=1.4 m/s) the resolved-physics regime needs the
    # pin as an ignition bootstrap — without it the source-patch flame
    # dies before self-sustaining combustion establishes.
    ri.outdoor_overrides["ignition_duration_s"]   = 3.0

    # Cheney sim-time schedule: low-U needs more time for the front to
    # cover comparable distance.  Match Phase 14x sweep convention.
    SIM_T = {0.5: 30.0, 1.0: 25.0, 2.0: 20.0, 4.0: 15.0, 8.0: 12.0}.get(U, 15.0)

    # Phase 17f (2026-06-29): dx-vs-U mesh rule.  Keeps cell residence
    # time within a constant factor of the EDC timescale across U,
    # so the source-patch chemistry resolves comparably at all winds.
    # Rule: dx = clip(0.025·U, 0.025, 0.10) — cap 0.10 (U=4/8 production),
    # floor 0.025 (U=0.5 source-patch resolution).
    #
    # HONEST LIMIT: this rule does NOT make U≤1 propagate.  A 30-s
    # Nat4 U=0.5 run under the rule (dx=0.025, Lx=15) bursts in the
    # source patch (frac→13.8% by t=12, ω→0 by t=15) and the T_s
    # front locks at x≈2.54 m.  The earlier 12-s standalone test
    # giving ratio=0.76 caught the source-patch BURST, not sustained
    # spread.  Root cause is not mesh — it is the closure:
    #   1. convective preheat ∝ √U makes downstream cell enthalpy
    #      budget ~8× lower at U=0.5 than at U=4
    #   2. radiation alone insufficient at h_bed=0.37 m grass scale
    #      (Anderson 1969 — radiation dominates only at U≲0.3 AND
    #      deeper beds)
    #   3. RANS k-ε + cell-averaged EDC structurally cannot represent
    #      intermittent flame-tongue contact (Finney 2015, 2020 —
    #      the real low-U bridging mechanism)
    # Production validity: U ≥ 4 m/s.  See Phase 18 plan for
    # FIRETEC-nonlocal / LES-with-sub-grid-bed paths.
    DX = max(0.025, min(0.10, 0.025 * U))
    # Lx also scales with U.  At low U the front doesn't advance far,
    # so we can shrink the domain to keep wall time tractable.
    LX = max(15.0, min(40.0, 4.0 + 6.0 * U))

    print(f"\n=== Phase 16 Cheney  {label}  {fuel} {mf_pct}% U={U}  "
          f"dx={DX}m  Lx={LX}m  sim_t={SIM_T}s  drying={drying_mode} ===",
          flush=True)

    # Bed extends nearly the full domain with small upstream/downstream buffers.
    BED_X_START = max(1.0, LX * 0.05)
    BED_X_END   = LX - 3.0

    t0 = time.time()
    r = run_3d_spread(
        ri, wind_speed_m_s=U,
        Lx=LX, Ly=0.5, Lz=8.0, dx=DX, dy=0.10,
        n_z_bed=8,                                # Cheney converged
        cfl_factor=0.40, max_wall_time_s=SIM_T,
        y_bc="periodic", turbulence_model="k_epsilon",
        wall_function=False, combustion_closure="edc",
        wind_profile_type="log_law",
        bed_x_start=BED_X_START, bed_x_end=BED_X_END,
        projection_method="fft_pcg",
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        # Env-var overrides for Phase 20 ignition-strength scouting.
        ignition_q_mult=float(os.environ.get("IGNITION_Q_MULT", 3.0)),
        ignition_width_mult=3.0,
        finney_tendril_enable=False,
        # Phase 16 + bug fixes:
        lagrangian_bed_enable=True,
        lagrangian_bed_N_per_cell=20,
        lagrangian_bed_h_conv=250.0,
        lagrangian_bed_view_factor=1.0,
        lagrangian_bed_view_factor_geometric=True,
        lagrangian_bed_drying_mode=drying_mode,
        # Phase 20: NO pin (Dirichlet BC on T_g isn't a physical heat
        # source per user 2026-07-16).  Instead use physical mechanisms
        # only: (1) solid_phase_ignition seeds bed particles at T_s_seed
        # (hot-ember analog, IC only), (2) drip-torch heat pulse for 3s
        # (Q source in gas + solid, ignition_q_mult × 240 kW/m²).
        ignition_T_pin_enable=False,
        solid_phase_ignition_enable=True,
        solid_phase_ignition_T_s_K=float(
            os.environ.get("SOLID_PHASE_IGNITION_T_S_K", 1000.0)),
        min_dt_s=1.0e-4,
        snapshot_dir=out_dir,
        snapshot_interval_s=float(os.environ.get("SNAPSHOT_INTERVAL_S", 1.0)),
        # Phase 17c: zero the kinematic v_n forcing; bed self-ignites via
        # CFD (advection + DOM + h_conv) + bed coupling.  See passive-test
        # memo for the hypothesis confirmation.
        level_set_passive=True,
        # Phase 19: empirical-ROS hybrid (Cheney Eq.6 imposed on level-set
        # at U <= 1.4 m/s, no-op at higher U).  a_ch per fuel:
        # Nat=0.406, Cut=0.343.  See phase19_empirical_ros_hybrid memory.
        empirical_ros_enable=True,
        empirical_ros_model="cheney_eq6",
        empirical_ros_a_ch=A_CH[fuel],
        empirical_ros_u_threshold_m_s=float(
            os.environ.get("EMPIRICAL_ROS_U_THRESHOLD", 1.4)),
        empirical_ros_blend_width_m_s=float(
            os.environ.get("EMPIRICAL_ROS_BLEND_WIDTH", 0.5)),
        # Phase 20 char-ox knobs (env-var overrides for A/B testing).
        char_ox_flux_cap_W_m2=float(os.environ.get("CHAR_OX_FLUX_CAP", 1.0e5)),
        char_ox_ash_exp=float(os.environ.get("CHAR_OX_ASH_EXP", 0.0)),
    )
    wall = time.time() - t0
    ros_m_min = float(r.ros_m_s) * 60.0

    eq6 = cheney_eq6_m_min(fuel, mf_pct, U)
    ratio = ros_m_min / eq6 if eq6 > 0 else float("nan")
    eq6_pass = (1.0/3.0) <= ratio <= 3.0

    # Phase 17b snap-ROS (robust to z-varying front).
    snaps = sorted(out_dir.glob("snap_*.npz"))
    if len(snaps) >= 2:
        snap_t  = np.array([float(np.load(sp)["t"])       for sp in snaps])
        snap_fx = np.array([float(np.load(sp)["front_x"]) for sp in snaps])
        if snap_t[-1] > snap_t[0]:
            ros_snap_m_min = (snap_fx[-1] - snap_fx[0]) / \
                              (snap_t[-1] - snap_t[0]) * 60.0
        else:
            ros_snap_m_min = float("nan")
    else:
        ros_snap_m_min = float("nan")
    # Phase 17c (passive level-set): bed-ignition ROS from T_s ≥ T_ign.
    # When v_n forcing is zero, the level-set is locked so ros_m_min ≈ 0;
    # the truthful "fire front" is where the bed has ignited.
    T_IGN = 600.0
    ts_t, ts_x = [], []
    for sp in snaps:
        s = np.load(sp)
        x_mid = s["x_mid"]
        col_burned = (s["T_s"] >= T_IGN).any(axis=(0, 1))
        if col_burned.any():
            ts_t.append(float(s["t"]))
            ts_x.append(float(x_mid[np.where(col_burned)[0].max()]))
    if len(ts_t) >= 3:
        # Linear fit, skip pre-ignition phase (t < 1s).
        mask = [i for i, t in enumerate(ts_t) if t >= 1.0]
        if len(mask) >= 3:
            ros_Ts_m_min = float(np.polyfit([ts_t[i] for i in mask],
                                              [ts_x[i] for i in mask], 1)[0]) * 60.0
        else:
            ros_Ts_m_min = float("nan")
    else:
        ros_Ts_m_min = float("nan")
    ratio_Ts = ros_Ts_m_min / eq6 if (eq6 > 0 and ros_Ts_m_min == ros_Ts_m_min) \
                                 else float("nan")
    eq6_pass_Ts = (1.0/3.0) <= ratio_Ts <= 3.0 if ratio_Ts == ratio_Ts else False
    ratio_snap = ros_snap_m_min / eq6 if eq6 > 0 else float("nan")
    eq6_pass_snap = (1.0/3.0) <= ratio_snap <= 3.0 if ratio_snap == ratio_snap \
                    else False

    result = {
        "phase": "16_lagrangian_bed_cheney",
        "label": label,
        "fuel": fuel,
        "mf_pct": mf_pct,
        "U_m_s": U,
        "sim_t_s": SIM_T,
        "drying_mode": drying_mode,
        "ROS_m_min":       ros_m_min,
        "ROS_snap_m_min":  ros_snap_m_min,
        "ROS_Ts_m_min":    ros_Ts_m_min,
        "ROS_m_s":         float(r.ros_m_s),
        "wall_s":          wall,
        "eq6_ref_m_min":   eq6,
        "eq6_ratio":       ratio,
        "eq6_ratio_snap":  ratio_snap,
        "eq6_ratio_Ts":    ratio_Ts,
        "eq6_pass_band_1over3_to_3":      bool(eq6_pass),
        "eq6_pass_band_1over3_to_3_snap": bool(eq6_pass_snap),
        "eq6_pass_band_1over3_to_3_Ts":   bool(eq6_pass_Ts),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"\n[saved] {label}  ROS_lset={ros_m_min:.2f}  ROS_Ts={ros_Ts_m_min:.2f} m/min  "
          f"Eq.6={eq6:.2f}  ratio_Ts={ratio_Ts:.3f}  "
          f"{'PASS' if eq6_pass_Ts else 'FAIL'}  wall={wall/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
