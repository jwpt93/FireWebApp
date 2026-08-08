"""Cheney Nat 4% U=4 with Phase 16 Lagrangian bed + the A_p/Q-cap fixes.

Confirmation that the two June-15 bug fixes (A_p includes m_char; Q_char/
Q_smold caps surface-area-scaled) don't break propagation in the canonical
validation case.

Baselines for comparison:
  Eulerian Phase 14ax with q_rad_gas WIRED: 13.25 m/min  (Eq.6 ratio 0.253 FAIL)
  Mickey Phase 16 (M_f=0, narrow, with pin): 24.31 m/min — confirms kernel runs

Cheney Eq.6 reference for Nat 4% U=4: ~52.4 m/min
EXP envelope band [Cheney 1993 Fig 8]: [0.638, 1.646] m/s = [38.3, 98.8] m/min

n_z_bed=8 per cheney_mesh_convergence_and_limits memory ("n_z_bed=8 is converged").
"""
import os, time, sys, json, math
os.environ.setdefault('MKL_CBWR', 'AVX2')
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_NUM_THREADS']   = '12'
os.environ['NUMBA_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS']   = '12'

import warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore', category=RuntimeWarning)
ROOT = Path('/home/jw/projects/cheney-web')
sys.path.insert(0, str(ROOT))
from model_outdoor.spread_3d import run_3d_spread
from model.io.text_input import load_text_input

DECK = ROOT / 'inputs/validation_cases/Outdoor_Grass_GR1__free_burn.txt'
OUT  = ROOT / 'local/diagnostics/cheney_phase16_Nat4U4'
OUT.mkdir(parents=True, exist_ok=True)

RHO_B, H_BED, MF, SAV = 1.07, 0.37, 0.04, 2000.0
U   = 4.0
SIM_T = 12.0
N_PER_CELL = 20


def main():
    ri = load_text_input(DECK)
    ri.outdoor_overrides['bulk_density_kg_m3']    = RHO_B
    ri.outdoor_overrides['fuel_depth_m']          = H_BED
    ri.outdoor_overrides['initial_moisture_frac'] = MF
    ri.outdoor_overrides['sav_ratio_1_m']         = SAV
    ri.outdoor_overrides['canopy_C_d']            = 0.30
    ri.outdoor_overrides['wall_bl_N']             = 0
    ri.outdoor_overrides['wall_bl_first_dz']      = 0.0
    ri.outdoor_overrides['wall_bl_growth']        = 1.0
    ri.outdoor_overrides['atm_growth']            = 1.20
    ri.outdoor_overrides['atm_max_dz']            = 1.0
    ri.outdoor_overrides.pop('dz_first', None)
    ri.outdoor_overrides.pop('bl_growth', None)

    print(f"\n=== Cheney Nat 4% U=4  Phase 16 Lagrangian  N_per_cell={N_PER_CELL} ===", flush=True)
    print(f"  M_f={MF} (drying path exercised)", flush=True)
    print(f"  n_z_bed=8 (Cheney converged)", flush=True)

    t0 = time.time()
    r = run_3d_spread(
        ri, wind_speed_m_s=U,
        Lx=40.0, Ly=0.5, Lz=8.0, dx=0.10, dy=0.10,
        n_z_bed=8,
        cfl_factor=0.40, max_wall_time_s=SIM_T,
        y_bc='periodic', turbulence_model='k_epsilon',
        wall_function=False, combustion_closure='edc',
        wind_profile_type='log_law',
        bed_x_start=2.0, bed_x_end=37.0,
        projection_method='fft_pcg',
        projection_cg_rtol=1e-6,
        dom_subcycle_every=5,
        ignition_q_mult=3.0,
        ignition_width_mult=3.0,
        finney_tendril_enable=False,
        lagrangian_bed_enable=True,
        lagrangian_bed_N_per_cell=N_PER_CELL,
        lagrangian_bed_h_conv=250.0,
        lagrangian_bed_view_factor=1.0,
        lagrangian_bed_view_factor_geometric=True,
        ignition_T_pin_enable=True,
        ignition_T_pin_K=1500.0,
        ignition_T_pin_height_m=0.30,
        ignition_T_pin_ramp_s=0.5,
        min_dt_s=1.0e-4,
        snapshot_dir=OUT,
        snapshot_interval_s=0.5,
    )
    wall = time.time() - t0

    u2 = U * 0.723
    eq6 = 0.406 * u2**0.987 * math.exp(-0.0707 * 4.0) * 60.0
    ros_m_min = float(r.ros_m_s) * 60.0

    result = {
        'phase': '16_lagrangian_bed_cheney_Nat4_U4',
        'ROS_m_min': ros_m_min,
        'ROS_m_s':   float(r.ros_m_s),
        'wall_s':    wall,
        'eq6_ref_m_min': eq6,
        'eq6_ratio':     ros_m_min / eq6,
        'envelope_low_m_min':  38.3,
        'envelope_high_m_min': 98.8,
        'eulerian_baseline_q_rad_gas_wired_m_min': 13.25,
    }
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))

    print(f"\n=== RESULT ===", flush=True)
    print(f"  ROS = {ros_m_min:.2f} m/min  ({float(r.ros_m_s):.4f} m/s)", flush=True)
    print(f"  wall = {wall:.0f}s = {wall/60:.1f}m", flush=True)
    print(f"  Eq.6 ref = {eq6:.2f} m/min  ratio = {ros_m_min/eq6:.3f}", flush=True)
    print(f"  EXP envelope = [38.3, 98.8] m/min  -> {'INSIDE' if 38.3 <= ros_m_min <= 98.8 else 'OUTSIDE'}", flush=True)
    print(f"  Eulerian q_rad_gas-wired baseline: 13.25 m/min", flush=True)


if __name__ == '__main__':
    main()
