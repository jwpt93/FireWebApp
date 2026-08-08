"""Phase 19 sweep animator v2 — single main panel (T_g), bed strip inset.

Design (dataviz skill):
  Form:  sequential magnitude (T_g in a spatial slice) — 2D pcolormesh.
  Color: perceptually uniform sequential ('inferno' — fire-appropriate).
  Range: fixed vmin=300 K, vmax=1800 K across all frames (temporal
         comparability; peak flame T for grass is ~1500-1700 K).
  Fronts: two clear vertical markers with legend
     • cyan solid  = level-set front (empirical/resolved v_n)
     • lime dashed = T_s ignition front (max x where any bed cell T_s ≥ 600 K)
  Aspect: honor physical geometry — wide-and-short.
  Layout: main T_g panel (top), narrow bed-strip (bottom) showing bp_T_s_avg
          in the bed depth (same colormap so eye reads through).
  Header: case, sim time, ratio_Ts.  No trailing text clutter.

Usage:
  python scripts/animate_cheney_phase19_v2.py                    # all completed cases
  python scripts/animate_cheney_phase19_v2.py --case Nat4_U0p5   # one case
  python scripts/animate_cheney_phase19_v2.py --path <dir> --label <name>
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parent.parent

# Fixed color range: T_g ∈ [300, 1500] K, γ = 0.5 (power norm).
# Diagnostic showed 99% of the domain sits at ambient (~303 K), with the
# top 1% carrying all the fire signal (peak observed ~1200 K in no-pin
# runs, ~1500 K in buggy pin runs).  Linear norm on this distribution
# gives almost-all-black; γ<1 expands low-T resolution so the warming
# band (600–1000 K) actually shows colour.
# 'hot' colormap: black → red → orange → yellow → white (black-body
# radiation progression, immediately recognized as fire).
TG_VMIN, TG_VMAX, TG_GAMMA = 300.0, 1500.0, 0.5
CMAP = "hot"

# Front-marker colors (categorical: level-set vs T_s front).  Two hues, each
# validated for CVD separation vs 'inferno' background at their intended
# lightness.
LSET_COLOR = "#00e5ff"   # cyan   — level-set front
TS_COLOR   = "#00ff88"   # bright green — T_s ignition front
                          # (chosen distinct from hot's yellow/white peak
                          #  AND from level-set cyan; halo ensures legibility)


def _bed_top_z(dz_arr, n_z_bed):
    return float(np.cumsum(dz_arr)[n_z_bed - 1])


def _detect_n_z_bed(bp_T_s):
    """Bed layers = z levels where bp_T_s_avg is nonzero anywhere at t=0."""
    for k in range(bp_T_s.shape[0]):
        if bp_T_s[k].max() <= 0.0:
            return k
    return bp_T_s.shape[0]


def render_frame(sp, out_png, case_label, eq6_ratio, n_z_bed):
    s = np.load(sp)
    t       = float(s["t"])
    x_mid   = s["x_mid"]
    z_mid   = s["z_mid"]
    dz_arr  = s["dz_arr"]
    Tg      = s["T_g"]
    bp_Ts   = s["bp_T_s_avg"]
    j       = Tg.shape[1] // 2   # y-midplane

    # Domain focus: full x, z from 0 to max(2 m, 1.5 × bed_top).
    z_bed_top = _bed_top_z(dz_arr, n_z_bed)
    z_max_plot = max(2.5, 1.5 * z_bed_top + 1.0)
    kz_top = int(np.searchsorted(z_mid, z_max_plot))
    kz_top = min(kz_top, z_mid.size)

    Tg_slice = Tg[:kz_top, j, :]
    z_plot   = z_mid[:kz_top]
    X, Z     = np.meshgrid(x_mid, z_plot, indexing="xy")

    # T_s ignition front (max x with any bed cell T_s ≥ 600 K).
    ts_ign_mask = (s["T_s"][:n_z_bed] >= 600.0).any(axis=(0, 1))
    ts_front_x  = float(x_mid[np.where(ts_ign_mask)[0].max()]) if ts_ign_mask.any() else float("nan")
    lset_front  = float(s.get("front_x", -1.0))

    # Figure layout: 12 : 1 aspect for main T_g; tiny bed strip below.
    fig = plt.figure(figsize=(15, 4.2))
    gs = GridSpec(
        2, 1,
        height_ratios=[8, 1],
        hspace=0.05,
        left=0.06, right=0.95, top=0.86, bottom=0.10,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_bed  = fig.add_subplot(gs[1], sharex=ax_main)

    im = ax_main.pcolormesh(
        X, Z, Tg_slice, cmap=CMAP, norm=PowerNorm(gamma=TG_GAMMA, vmin=TG_VMIN, vmax=TG_VMAX),
        shading="auto", rasterized=True,
    )
    # Bed top reference line (dark grey, thin)
    ax_main.axhline(z_bed_top, color="#333", lw=0.8, alpha=0.6)

    # Front markers with a light halo for legibility on any background
    for x, color, label in [
        (lset_front, LSET_COLOR, "level-set front"),
        (ts_front_x, TS_COLOR,   "T_s ignition front"),
    ]:
        if np.isfinite(x) and x > 0:
            ax_main.axvline(x, color="black", lw=3.5, alpha=0.35)
            ax_main.axvline(x, color=color,   lw=1.6, label=label)

    ax_main.set_ylim(0, z_max_plot)
    ax_main.set_ylabel("z [m]")
    ax_main.tick_params(labelbottom=False)
    ax_main.set_aspect("auto")

    leg = ax_main.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc",
        fontsize=9, framealpha=0.95, borderpad=0.35,
    )
    for txt in leg.get_texts():
        txt.set_color("#111")

    # Colorbar for the main panel
    cbar = fig.colorbar(im, ax=[ax_main, ax_bed], pad=0.01, fraction=0.03,
                        aspect=30, extend="max")
    cbar.set_label("T [K]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ── bed strip: bp_T_s_avg in the bed (z=0 to z_bed_top), same colormap ──
    bp_Ts_bed = bp_Ts[:n_z_bed, j, :]
    z_bed = z_mid[:n_z_bed]
    Xb, Zb = np.meshgrid(x_mid, z_bed, indexing="xy")
    ax_bed.pcolormesh(
        Xb, Zb, np.where(bp_Ts_bed > 1.0, bp_Ts_bed, np.nan),
        cmap=CMAP, norm=PowerNorm(gamma=TG_GAMMA, vmin=TG_VMIN, vmax=TG_VMAX),
        shading="auto", rasterized=True,
    )
    ax_bed.set_ylim(0, z_bed_top)
    ax_bed.set_xlabel("x [m]")
    ax_bed.set_ylabel("bed T_s", fontsize=8)
    ax_bed.tick_params(labelsize=8)
    ax_bed.set_facecolor("#1a1418")   # dark warm-grey — reads as "cold soil"
                                       # against the black-at-ambient 'hot' cmap

    # Header: case, t, ratio
    ratio_str = f"   ratio_Ts = {eq6_ratio:.3f}" if eq6_ratio is not None else ""
    fig.suptitle(
        f"Cheney {case_label}    t = {t:5.2f} s{ratio_str}",
        fontsize=12, fontweight="bold", x=0.06, ha="left", y=0.965,
    )

    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


def stitch_gif(frame_paths, out_gif):
    import imageio.v2 as imageio
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, frames, duration=0.4, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--path", type=str, default=None,
                        help="Custom snapshot dir (with --label)")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out",  type=str,
                        default="plots/cheney_phase19_sweep_anim_v2",
                        help="Output directory relative to repo root")
    args = parser.parse_args()

    out_base = ROOT / args.out
    out_base.mkdir(parents=True, exist_ok=True)

    if args.path is not None:
        case_dirs = [(Path(args.path), args.label or Path(args.path).name)]
    else:
        sweep_base = ROOT / "local/diagnostics/cheney_phase19_sweep"
        dirs = sorted(d for d in sweep_base.iterdir()
                      if d.is_dir() and (d / "result.json").exists())
        if args.case:
            dirs = [d for d in dirs if d.name == args.case]
        case_dirs = [(d, d.name) for d in dirs]

    print(f"{len(case_dirs)} case(s) to animate")
    for case_dir, label in case_dirs:
        out_gif = out_base / f"{label}.gif"
        if out_gif.exists() and not args.force:
            print(f"[skip] {out_gif.name}")
            continue
        snaps = sorted(case_dir.glob("snap_*.npz"))
        if not snaps:
            print(f"[warn] no snaps in {case_dir}")
            continue
        try:
            rj = json.loads((case_dir / "result.json").read_text())
            eq6_ratio = rj.get("eq6_ratio_Ts")
        except Exception:
            eq6_ratio = None

        # Detect n_z_bed from first snapshot
        s0 = np.load(snaps[0])
        n_z_bed = _detect_n_z_bed(s0["bp_T_s_avg"])
        if n_z_bed == 0:  # Nothing seeded at t=0; try mid-run
            s_mid = np.load(snaps[len(snaps)//2])
            n_z_bed = _detect_n_z_bed(s_mid["bp_T_s_avg"])
        if n_z_bed == 0:
            n_z_bed = 4   # fallback

        print(f"[{label}] {len(snaps)} frames  n_z_bed={n_z_bed}")
        frame_dir = out_base / label
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_pngs = []
        for sp in snaps:
            png = frame_dir / f"{sp.stem}.png"
            render_frame(sp, png, label, eq6_ratio, n_z_bed)
            frame_pngs.append(png)
        stitch_gif(frame_pngs, out_gif)
        print(f"[saved] {out_gif}  ({out_gif.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
