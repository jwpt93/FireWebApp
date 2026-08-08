"""Command-line interface for the ROM pyrolysis solver.

Usage
-----
  rom -i inputs/validation_cases/FSRI_Wood_Stud_3NODE__CONE_75.txt
  rom -i deck.txt --out-dir results/ --csv --json
  python -m model -i deck.txt --no-plot

All output options (plot, CSV, JSON) can also be set in the input deck
via ``output.*`` keys — CLI flags override deck values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="rom",
        description="ROM pyrolysis solver — run a single input deck",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output options may also be set in the input deck with output.* keys:\n"
            "  output.png_enable      = true\n"
            "  output.csv_enable      = false\n"
            "  output.base_dir        = plots\n"
            "  output.exp_csv_path    = path/to/exp.csv\n"
        ),
    )
    ap.add_argument("-i", "--input", required=True, type=Path,
                    metavar="DECK", help="Input deck (.txt)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (overrides output.base_dir in deck)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Disable PNG plot even if deck says output.png_enable = true")
    ap.add_argument("--csv", action="store_true",
                    help="Force CSV time-series output")
    ap.add_argument("--json", action="store_true",
                    help="Force JSON metrics output")
    ap.add_argument("--exp", type=Path, default=None, metavar="EXP_CSV",
                    help="Path to experimental CSV for comparison overlay")
    args = ap.parse_args()

    deck = args.input.resolve()
    if not deck.exists():
        print(f"Error: deck not found: {deck}", file=sys.stderr)
        sys.exit(1)

    # ── Load deck ─────────────────────────────────────────────────────────────
    from model.io.text_input import load_text_input, output_config_from_inputs
    rom_inputs = load_text_input(deck)
    out_cfg = output_config_from_inputs(rom_inputs)

    # ── Apply CLI overrides ────────────────────────────────────────────────────
    if args.out_dir is not None:
        out_cfg.base_dir = str(args.out_dir)
    if args.no_plot:
        out_cfg.png_enable = False
    if args.csv:
        out_cfg.csv_enable = True
    if args.json:
        out_cfg.json_metrics_enable = True
    if args.exp is not None:
        out_cfg.exp_csv_path = str(args.exp)

    # ── Run ROM ───────────────────────────────────────────────────────────────
    from model.runner import run_rom
    case_id = deck.stem

    # Extract run parameters from parsed deck
    q_in = float(rom_inputs.q_in_constant or 0.0)
    # q_in in deck is W/m² after convert_q_in; convert to kW/m² for run_rom
    if rom_inputs.q_in_units and "kw" in rom_inputs.q_in_units.lower():
        q_in_kw = q_in  # already kW/m²
    else:
        q_in_kw = q_in / 1000.0  # W/m² → kW/m²

    sig = run_rom(
        q_in_kW_m2=q_in_kw,
        t_end_s=float(rom_inputs.t_end or 1800.0),
        area_m2=float(rom_inputs.area_m2 or 0.01),
        Tamb_K=float(rom_inputs.Tamb or 300.0),
        M1_init=float(rom_inputs.M1 if rom_inputs.M1 is not None else 1.0),
        hoc_eff=float(rom_inputs.hoc_eff or 15500.0),
        subcase_token="cli",
        rom_inputs=rom_inputs,
        case_id=case_id,
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    from model.io.output import write_outputs
    write_outputs(
        sig=sig,
        out_cfg=out_cfg,
        case_id=case_id,
        deck_path=deck,
        exp_csv_path=out_cfg.exp_csv_path or None,
    )


if __name__ == "__main__":
    main()
