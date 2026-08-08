"""Animate Phase 16 Cheney 4-case sweep — y-midplane multipanel gifs.

NOTE per memory feedback_animation_frame_density.md: standard interval
is 0.33 s but the 4-case sweep used 1.0 s for time-budget reasons.
Animations here use the 1-s frames as-is (~15 frames each); a smoother
re-run would need ~45 frames per case.

For each case directory:
  Renders one PNG per snapshot showing 4 panels:
    Panel 1: T_g [K]            — gas temperature (y-midplane)
    Panel 2: bp_T_s_avg [K]     — bed-particle mass-weighted T_s (NEW for Phase 16)
    Panel 3: omega [kg/m³/s]    — combustion source (log color)
    Panel 4: bp_m_char (kg/m³)  — accumulated char inventory
  Stitches frames into a .gif in plots/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
SWEEP_BASE = ROOT / "local/diagnostics/cheney_phase16_4case"
PLOT_BASE = ROOT / "plots/cheney_phase16_4case_anim"
PLOT_BASE.mkdir(parents=True, exist_ok=True)

CASES = ["Nat4_U4", "Nat8_U4", "Cut4_U4", "Cut8_U4"]


def render_frame(snap_path, out_png, case_label):
    d = np.load(snap_path, allow_pickle=True)
    t       = float(d["t"])
    front_x = float(d.get("front_x", -1.0))
    x_mid   = d["x_mid"]
    z_mid   = d["z_mid"]
    # y-midplane
    j = d["T_g"].shape[1] // 2
    Tg     = d["T_g"][:, j, :]
    bp_Ts  = d["bp_T_s_avg"][:, j, :]
    omega  = d["omega"][:, j, :]
    bp_mc  = d["bp_m_char"][:, j, :]

    # Truncate to z ≤ 3 m (focus on flame region)
    k_max = int(np.searchsorted(z_mid, 3.0))
    if k_max < 5:
        k_max = z_mid.size
    Tg = Tg[:k_max]; bp_Ts = bp_Ts[:k_max]
    omega = omega[:k_max]; bp_mc = bp_mc[:k_max]
    z = z_mid[:k_max]
    X, Z = np.meshgrid(x_mid, z, indexing="xy")

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)

    # Panel 1: T_g
    ax = axes[0, 0]
    im = ax.pcolormesh(X, Z, Tg, cmap="hot", vmin=300, vmax=2000, shading="auto")
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("T_g [K]   (cyan = front_x)")
    ax.set_ylabel("z [m]")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: bed-particle T_s (Phase 16 unique field)
    ax = axes[0, 1]
    bp_Ts_p = np.where(bp_Ts > 1.0, bp_Ts, np.nan)
    im = ax.pcolormesh(X, Z, bp_Ts_p, cmap="inferno", vmin=300, vmax=2200, shading="auto")
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("bp_T_s_avg [K]  (Phase 16 sub-grid char)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: omega (log)
    ax = axes[1, 0]
    om_p = np.where(omega > 1e-6, omega, np.nan)
    if np.any(np.isfinite(om_p)):
        im = ax.pcolormesh(X, Z, om_p, cmap="viridis",
                           norm=LogNorm(vmin=1e-3, vmax=10.0), shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("omega [kg/m³/s]  (combustion rate, log)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    # Panel 4: bp_m_char
    ax = axes[1, 1]
    bp_mc_p = np.where(bp_mc > 1e-6, bp_mc, np.nan)
    if np.any(np.isfinite(bp_mc_p)):
        im = ax.pcolormesh(X, Z, bp_mc_p, cmap="copper", shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if front_x > 0:
        ax.axvline(front_x, color="cyan", lw=1.0, alpha=0.7)
    ax.set_title("bp_m_char [kg/cell] (residual char)")
    ax.set_xlabel("x [m]")

    fig.suptitle(f"Cheney Phase 16 {case_label}   t={t:.2f}s   front={front_x:.2f}m",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def stitch_gif(frame_paths, out_gif):
    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(out_gif, frames, duration=0.4, loop=0)
        return True
    except Exception as e:
        print(f"  ! imageio stitch failed: {e}")
        return False


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for case in CASES:
        if only and case != only:
            continue
        case_dir = SWEEP_BASE / case
        snaps = sorted(case_dir.glob("snap_*.npz"))
        if not snaps:
            print(f"  ! no snaps in {case_dir}")
            continue
        print(f"\n=== {case}  ({len(snaps)} frames @ 1.0s interval) ===", flush=True)
        frame_dir = PLOT_BASE / case
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_pngs = []
        for sp in snaps:
            png = frame_dir / f"{sp.stem}.png"
            render_frame(sp, png, case)
            frame_pngs.append(png)
            print(f"  {sp.name} → {png.name}", flush=True)
        out_gif = PLOT_BASE / f"{case}.gif"
        ok = stitch_gif(frame_pngs, out_gif)
        if ok:
            print(f"  [saved] {out_gif}  ({out_gif.stat().st_size//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
