"""Animate Phase 19 Cheney sweep — y-midplane multipanel gifs.

Adapted from scripts/animate_cheney_phase16_4case.py.  Iterates over all
completed cases in local/diagnostics/cheney_phase19_sweep/ and renders
a 4-panel gif per case:
  Panel 1: T_g [K]           — gas temperature (y-midplane, log color 300-2000)
  Panel 2: bp_T_s_avg [K]    — bed-particle mass-weighted T_s
  Panel 3: omega [kg/m³/s]   — combustion source (log color)
  Panel 4: bp_m_char [kg]    — accumulated char inventory

Skips cases without result.json (still running).  Skips re-rendering if
the .gif already exists.  Snapshots at 1.0-s interval per Phase 17f
canonical worker (feedback_animation_frame_density says 0.33s is ideal
but 1.0s is what the sweep wrote).

Usage:  python scripts/animate_cheney_phase19_sweep.py [--case Nat4_U0p5]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
SWEEP_BASE = ROOT / "local/diagnostics/cheney_phase19_sweep"
PLOT_BASE = ROOT / "plots/cheney_phase19_sweep_anim"
PLOT_BASE.mkdir(parents=True, exist_ok=True)


def render_frame(snap_path, out_png, case_label, eq6_ratio=None):
    d = np.load(snap_path, allow_pickle=True)
    t       = float(d["t"])
    front_x = float(d.get("front_x", -1.0))
    x_mid   = d["x_mid"]
    z_mid   = d["z_mid"]
    j = d["T_g"].shape[1] // 2
    Tg     = d["T_g"][:, j, :]
    bp_Ts  = d["bp_T_s_avg"][:, j, :]
    omega  = d["omega"][:, j, :]
    bp_mc  = d["bp_m_char"][:, j, :]

    k_max = int(np.searchsorted(z_mid, 3.0))
    if k_max < 5:
        k_max = z_mid.size
    Tg = Tg[:k_max]; bp_Ts = bp_Ts[:k_max]
    omega = omega[:k_max]; bp_mc = bp_mc[:k_max]
    z = z_mid[:k_max]
    X, Z = np.meshgrid(x_mid, z, indexing="xy")

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    im = ax.pcolormesh(X, Z, Tg, cmap="hot", vmin=300, vmax=2000, shading="auto")
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("T_g [K]   (cyan = level-set front)")
    ax.set_ylabel("z [m]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    bp_Ts_p = np.where(bp_Ts > 1.0, bp_Ts, np.nan)
    im = ax.pcolormesh(X, Z, bp_Ts_p, cmap="inferno", vmin=300, vmax=2200, shading="auto")
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("bp_T_s_avg [K]  (bed-particle T_s)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    om_p = np.where(omega > 1e-6, omega, np.nan)
    if np.any(np.isfinite(om_p)):
        im = ax.pcolormesh(X, Z, om_p, cmap="viridis",
                           norm=LogNorm(vmin=1e-3, vmax=10.0), shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("omega [kg/m³/s]  (combustion, log)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    ax = axes[1, 1]
    bp_mc_p = np.where(bp_mc > 1e-6, bp_mc, np.nan)
    if np.any(np.isfinite(bp_mc_p)):
        im = ax.pcolormesh(X, Z, bp_mc_p, cmap="copper", shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("bp_m_char [kg/cell]")
    ax.set_xlabel("x [m]")

    header = f"Cheney Phase 19  {case_label}   t={t:.2f}s   front={front_x:.2f}m"
    if eq6_ratio is not None:
        header += f"   ratio_Ts={eq6_ratio:.3f}"
    fig.suptitle(header, fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=110)   # NO bbox_inches='tight' — frames must be same shape for gif
    plt.close(fig)


def stitch_gif(frame_paths, out_gif):
    import imageio.v2 as imageio
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, frames, duration=0.4, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None,
                        help="Only animate this case label")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if .gif exists")
    args = parser.parse_args()

    case_dirs = sorted(d for d in SWEEP_BASE.iterdir()
                       if d.is_dir() and (d / "result.json").exists())
    if args.case:
        case_dirs = [d for d in case_dirs if d.name == args.case]

    print(f"Found {len(case_dirs)} completed cases:")
    for d in case_dirs:
        print(f"  - {d.name}")
    print()

    import json
    for case_dir in case_dirs:
        case = case_dir.name
        out_gif = PLOT_BASE / f"{case}.gif"
        if out_gif.exists() and not args.force:
            print(f"[skip] {out_gif.name} exists")
            continue
        snaps = sorted(case_dir.glob("snap_*.npz"))
        if not snaps:
            print(f"  ! no snaps in {case_dir}")
            continue
        try:
            rj = json.loads((case_dir / "result.json").read_text())
            eq6_ratio = rj.get("eq6_ratio_Ts")
        except Exception:
            eq6_ratio = None
        print(f"\n=== {case}  ({len(snaps)} frames) ===", flush=True)
        frame_dir = PLOT_BASE / case
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_pngs = []
        for sp in snaps:
            png = frame_dir / f"{sp.stem}.png"
            render_frame(sp, png, case, eq6_ratio=eq6_ratio)
            frame_pngs.append(png)
        stitch_gif(frame_pngs, out_gif)
        size_kb = out_gif.stat().st_size // 1024
        print(f"[saved] {out_gif}  ({size_kb} KB, {len(frame_pngs)} frames)", flush=True)


if __name__ == "__main__":
    main()
