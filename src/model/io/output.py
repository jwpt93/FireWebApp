"""ROM output dispatch: writes CSV, PNG, and JSON metrics from a RomSignals result.

Entry point: ``write_outputs(sig, out_cfg, case_id, deck_path, exp_csv_path)``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.runner import RomSignals
    from model.config.schemas import OutputConfig

# Metrics window for "sustained" average
_SUST_T0 = 30.0
_SUST_T1 = 600.0


def _out_dir(out_cfg: "OutputConfig", case_id: str) -> Path:
    base = Path(out_cfg.base_dir)
    if out_cfg.case_subdir:
        return base / case_id
    return base


def write_outputs(
    sig: "RomSignals",
    out_cfg: "OutputConfig",
    case_id: str,
    deck_path: Optional[Path] = None,
    exp_csv_path: Optional[str] = None,
) -> None:
    """Dispatch to CSV / PNG / JSON based on out_cfg flags.

    Parameters
    ----------
    sig:          RomSignals from run_rom()
    out_cfg:      OutputConfig controlling what to write
    case_id:      base filename stem (e.g. 'FSRI_Wood_Stud_3NODE__CONE_75')
    deck_path:    path to input deck (used for metadata in JSON)
    exp_csv_path: override path to EXP CSV; falls back to out_cfg.exp_csv_path
    """
    out_dir = _out_dir(out_cfg, case_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_path = exp_csv_path or (out_cfg.exp_csv_path if out_cfg.exp_csv_path else None)
    exp_t, exp_hrrpua = _load_exp_csv(exp_path) if exp_path else (None, None)

    if out_cfg.metrics_console:
        _print_metrics(sig, exp_t, exp_hrrpua, case_id)

    if out_cfg.png_enable:
        _write_png(sig, out_cfg, out_dir, case_id, exp_t, exp_hrrpua)

    if out_cfg.csv_enable:
        _write_csv(sig, out_cfg, out_dir, case_id)

    if out_cfg.json_metrics_enable:
        _write_json_metrics(sig, out_cfg, out_dir, case_id, deck_path, exp_t, exp_hrrpua)


# ── PNG ───────────────────────────────────────────────────────────────────────

def _write_png(sig, out_cfg, out_dir, case_id, exp_t, exp_hrrpua) -> None:
    from model.io.plots import plot_rom_vs_exp
    T_nodes = list(sig.T_nodes) if getattr(sig, "T_nodes", None) is not None else None
    alpha_nodes = list(sig.alpha_nodes) if getattr(sig, "alpha_nodes", None) is not None else None
    out_path = out_dir / f"{case_id}.png"
    plot_rom_vs_exp(
        out_path=out_path,
        case_label=case_id.replace("_", " "),
        rom_t=np.asarray(sig.t),
        rom_hrrpua=np.asarray(sig.hrrpua),
        rom_T_surf=np.asarray(sig.T_surf),
        rom_T_mid=np.asarray(sig.T_mid) if getattr(sig, "T_mid", None) is not None else None,
        rom_T_inner=np.asarray(sig.T_inner) if getattr(sig, "T_inner", None) is not None else None,
        exp_t=exp_t,
        exp_hrrpua=exp_hrrpua,
        T_nodes=T_nodes,
        alpha_nodes=alpha_nodes,
        dpi=out_cfg.png_dpi,
    )


# ── CSV ───────────────────────────────────────────────────────────────────────

def _write_csv(sig, out_cfg, out_dir, case_id) -> None:
    import csv as csv_mod
    cols = [c.strip() for c in out_cfg.csv_columns.split(",") if c.strip()]
    out_path = out_dir / f"{case_id}.csv"
    t = np.asarray(sig.t)
    col_map = {
        "t": t,
        "hrrpua": np.asarray(sig.hrrpua),
        "mlr": np.asarray(sig.mlr) if getattr(sig, "mlr", None) is not None else np.zeros_like(t),
        "T_surf": np.asarray(sig.T_surf),
        "T_mid": np.asarray(sig.T_mid) if getattr(sig, "T_mid", None) is not None else np.full_like(t, np.nan),
        "T_inner": np.asarray(sig.T_inner) if getattr(sig, "T_inner", None) is not None else np.full_like(t, np.nan),
        "alpha1": _node_col(sig, "alpha_nodes", 0, t),
        "alpha2": _node_col(sig, "alpha_nodes", 1, t),
        "alpha3": _node_col(sig, "alpha_nodes", 2, t),
    }
    rows = len(t)
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(cols)
        for i in range(rows):
            writer.writerow([float(col_map.get(c, np.full_like(t, np.nan))[i]) for c in cols])
    print(f"  Wrote CSV: {out_path}")


def _node_col(sig, attr, idx, t_ref):
    arr = getattr(sig, attr, None)
    if arr is not None and len(arr) > idx:
        return np.asarray(arr[idx])
    return np.full(len(t_ref), np.nan)


# ── JSON metrics ──────────────────────────────────────────────────────────────

def _write_json_metrics(sig, out_cfg, out_dir, case_id, deck_path, exp_t, exp_hrrpua) -> None:
    t = np.asarray(sig.t)
    h = np.asarray(sig.hrrpua)
    peak = float(np.max(h)) if h.size else float("nan")
    t_peak = float(t[int(np.argmax(h))]) if h.size else float("nan")
    auc = float(np.trapezoid(h, t) / 1000.0) if h.size > 1 else float("nan")
    mask = (t >= _SUST_T0) & (t <= _SUST_T1)
    sust = float(h[mask].mean()) if mask.sum() > 0 else float("nan")

    metrics: dict = {
        "case_id": case_id,
        "deck": str(deck_path) if deck_path else None,
        "peak_hrrpua_kW_m2": peak,
        "t_peak_s": t_peak,
        "auc_kJ_m2": auc,
        "sustained_30_600s_kW_m2": sust,
        "t_end_s": float(t[-1]) if t.size else float("nan"),
    }
    if exp_t is not None and exp_hrrpua is not None:
        exp_peak = float(np.max(exp_hrrpua))
        metrics["exp_peak_kW_m2"] = exp_peak
        metrics["peak_ratio"] = peak / exp_peak if exp_peak > 0 else float("nan")
        exp_mask = (exp_t >= _SUST_T0) & (exp_t <= _SUST_T1)
        exp_sust = float(exp_hrrpua[exp_mask].mean()) if exp_mask.sum() > 0 else float("nan")
        metrics["exp_sustained_kW_m2"] = exp_sust
        metrics["sustained_ratio"] = sust / exp_sust if exp_sust > 0 and not np.isnan(exp_sust) else float("nan")

    out_path = out_dir / f"{case_id}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics: {out_path}")


# ── Console metrics table ─────────────────────────────────────────────────────

def _print_metrics(sig, exp_t, exp_hrrpua, case_id: str) -> None:
    t = np.asarray(sig.t)
    h = np.asarray(sig.hrrpua)
    if h.size == 0:
        print(f"  {case_id}: no output data")
        return
    peak = float(np.max(h))
    t_peak = float(t[int(np.argmax(h))])
    auc = float(np.trapezoid(h, t) / 1000.0) if h.size > 1 else float("nan")
    mask = (t >= _SUST_T0) & (t <= _SUST_T1)
    sust = float(h[mask].mean()) if mask.sum() > 0 else float("nan")

    print(f"\n  {case_id}")
    print(f"  {'Metric':<22} {'ROM':>10}", end="")
    if exp_hrrpua is not None:
        print(f"  {'EXP':>10}  {'R/E':>6}", end="")
    print()
    print(f"  {'-'*50}")
    _row("Peak HRRPUA [kW/m²]", peak, exp_hrrpua, lambda e: float(np.max(e)))
    _row("  t_peak [s]", t_peak, None, None)
    _row("Sustained 30-600s [kW/m²]", sust, exp_hrrpua,
         lambda e: float(e[(exp_t >= _SUST_T0) & (exp_t <= _SUST_T1)].mean())
         if exp_t is not None and ((exp_t >= _SUST_T0) & (exp_t <= _SUST_T1)).any() else None)
    _row("AUC [kJ/m²]", auc, None, None)


def _row(label, rom_val, exp_arr, exp_fn) -> None:
    exp_val = None
    if exp_arr is not None and exp_fn is not None:
        try:
            exp_val = exp_fn(exp_arr)
        except Exception:
            pass
    ratio_str = ""
    if exp_val is not None and exp_val > 0 and not np.isnan(rom_val):
        ratio_str = f"  {rom_val / exp_val:>6.2f}"
    exp_str = f"  {exp_val:>10.1f}" if exp_val is not None else ""
    print(f"  {label:<22} {rom_val:>10.1f}{exp_str}{ratio_str}")


# ── EXP CSV loader ────────────────────────────────────────────────────────────

def _load_exp_csv(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load experimental HRRPUA CSV (Time, HRRPUA columns). Returns (t, hrrpua) or (None, None)."""
    import csv as csv_mod
    p = Path(path)
    if not p.exists():
        return None, None
    t_list, h_list = [], []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                t_list.append(float(row["Time"]))
                h_list.append(max(float(row["HRRPUA"]), 0.0))
            except (KeyError, ValueError):
                continue
    if not t_list:
        return None, None
    return np.array(t_list), np.array(h_list)
