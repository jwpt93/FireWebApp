"""Standard ROM comparison plot generators.

All functions return True on success, False if matplotlib is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def plot_rom_vs_exp(
    out_path: Path,
    case_label: str,
    rom_t: np.ndarray,
    rom_hrrpua: np.ndarray,
    rom_T_surf: np.ndarray,
    rom_T_mid: Optional[np.ndarray] = None,
    rom_T_inner: Optional[np.ndarray] = None,
    rom_mlr: Optional[np.ndarray] = None,
    exp_t: Optional[np.ndarray] = None,
    exp_hrrpua: Optional[np.ndarray] = None,
    T_nodes: Optional[List] = None,
    alpha_nodes: Optional[List] = None,
    dpi: int = 160,
) -> bool:
    """3-panel comparison plot: HRRPUA / Temperatures / Char fractions.

    Saves PNG to ``out_path``. Returns False (without raising) if matplotlib
    is not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  matplotlib not available: {exc}")
        return False

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), sharex=True)
    ax_hrr, ax_temp, ax_alpha = axes[0], axes[1], axes[2]

    # ── HRRPUA panel ─────────────────────────────────────────────────────────
    rom_peak = float(np.max(rom_hrrpua)) if rom_hrrpua.size > 0 else 1.0
    if exp_hrrpua is not None and exp_t is not None:
        mask = exp_t <= (rom_t[-1] if rom_t.size > 0 else np.inf)
        exp_in_range = exp_hrrpua[mask]
        exp_p90 = float(np.percentile(exp_in_range, 90)) if len(exp_in_range) > 0 else rom_peak
        y_max = max(rom_peak * 1.5, exp_p90 * 1.3, 10.0)
        exp_true_peak = float(np.max(exp_hrrpua))
    else:
        y_max = rom_peak * 1.5
        exp_true_peak = None

    ax_hrr.plot(rom_t, rom_hrrpua, label="ROM", color="tab:blue", lw=1.8, zorder=3)
    if exp_t is not None and exp_hrrpua is not None:
        ax_hrr.plot(exp_t, exp_hrrpua, label="EXP", color="tab:orange", lw=1.4, alpha=0.85, zorder=2)
    ax_hrr.axhline(rom_peak, color="tab:blue", lw=0.8, ls=":", alpha=0.6,
                   label=f"ROM peak: {rom_peak:.0f} kW/m²")
    ax_hrr.set_ylim(bottom=0, top=y_max)
    ax_hrr.set_ylabel("HRRPUA [kW/m²]")
    ax_hrr.legend(fontsize=8)
    ax_hrr.grid(True, alpha=0.3)
    if exp_true_peak is not None and exp_true_peak > y_max:
        ax_hrr.text(0.02, 0.96, f"EXP ignition peak: {exp_true_peak:.0f} kW/m² (above y-limit)",
                    transform=ax_hrr.transAxes, fontsize=7.5, va="top",
                    color="tab:orange", style="italic")

    # ── Temperature panel ─────────────────────────────────────────────────────
    _colors_T = ["tab:red", "tab:purple", "tab:green", "tab:brown", "tab:pink", "tab:cyan"]
    if T_nodes is not None and len(T_nodes) > 0:
        for _ni, _T_arr in enumerate(T_nodes):
            ax_temp.plot(rom_t, np.asarray(_T_arr) - 273.15,
                         label=f"T{_ni + 1}", color=_colors_T[_ni % len(_colors_T)],
                         lw=1.8 if _ni == 0 else 1.4, ls="-" if _ni == 0 else "--" if _ni == 1 else ":")
    else:
        ax_temp.plot(rom_t, rom_T_surf - 273.15, label="T_surf", color="tab:red", lw=1.8)
        if rom_T_mid is not None:
            ax_temp.plot(rom_t, rom_T_mid - 273.15, label="T_mid", color="tab:purple", lw=1.4, ls="--")
        if rom_T_inner is not None:
            ax_temp.plot(rom_t, rom_T_inner - 273.15, label="T_inner", color="tab:green", lw=1.4, ls=":")
    ax_temp.set_ylabel("Temperature [°C]")
    ax_temp.legend(fontsize=8)
    ax_temp.grid(True, alpha=0.3)

    # ── Char fraction panel ───────────────────────────────────────────────────
    if alpha_nodes is not None and len(alpha_nodes) > 0:
        _colors_a = ["tab:red", "tab:purple", "tab:green", "tab:brown", "tab:pink", "tab:cyan"]
        for _ni, _a_arr in enumerate(alpha_nodes):
            ax_alpha.plot(rom_t, np.asarray(_a_arr),
                          label=f"α{_ni + 1}", color=_colors_a[_ni % len(_colors_a)],
                          lw=1.8 if _ni == 0 else 1.4, ls="-" if _ni == 0 else "--" if _ni == 1 else ":")
        ax_alpha.set_ylim(0, 1.05)
    else:
        ax_alpha.text(0.5, 0.5, "char fractions not available",
                      ha="center", va="center", transform=ax_alpha.transAxes,
                      fontsize=9, color="gray")
    ax_alpha.set_ylabel("Char fraction [-]")
    ax_alpha.set_xlabel("Time [s]")
    ax_alpha.legend(fontsize=8)
    ax_alpha.grid(True, alpha=0.3)

    fig.suptitle(case_label, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return True
