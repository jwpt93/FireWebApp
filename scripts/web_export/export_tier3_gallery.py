#!/usr/bin/env python3
"""Collect Tier-3 animation outputs into the website gallery.

Scans a diagnostics directory (default: local/diagnostics/, produced by
Tier-3 worker + animation scripts) for rendered animations/stills
(*.gif, *.mp4, *.png), copies them into docs/assets/tier3/, and rewrites
gallery.json so the static site can embed them.

Typical workflow (from repo root):

    # 1. run a Tier-3 case (minutes to hours) — see scripts/workers/
    OMP_NUM_THREADS=8 .venv/bin/python scripts/workers/_cheney_phase16_Nat4U4_worker.py
    # 2. render animations (reads snap_*.npz from local/diagnostics/)
    .venv/bin/python scripts/animations/animate_cheney_phase16_4case.py
    # 3. publish to the site
    .venv/bin/python scripts/web_export/export_tier3_gallery.py

Captions default to the file's parent-directory + stem; edit the generated
docs/assets/tier3/gallery.json by hand afterwards if you want nicer prose —
re-running this script preserves hand-edited captions for files it already
knows.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "docs" / "assets" / "tier3"
MEDIA_EXT = {".gif", ".mp4", ".png"}
STILL_EXT = {".png"}


def humanize(stem: str) -> str:
    """'cheney_phase16_Nat4U4' -> 'cheney phase16 Nat4U4' (light touch)."""
    return stem.replace("_", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--src",
        type=Path,
        default=REPO / "local" / "diagnostics",
        help="directory to scan for animations (default: local/diagnostics/)",
    )
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    index_path = DEST / "gallery.json"
    existing = json.loads(index_path.read_text()) if index_path.exists() else {}
    # keep hand-edited captions for known files
    known = {
        (item.get("video") or item.get("still")): item
        for item in existing.get("items", [])
    }

    found = sorted(
        p for p in args.src.rglob("*") if p.suffix.lower() in MEDIA_EXT
    ) if args.src.is_dir() else []

    items = []
    for src in found:
        dest_name = f"{src.parent.name}__{src.name}" if src.parent != args.src else src.name
        dest = DEST / dest_name
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
        if dest_name in known:
            items.append(known[dest_name])
            continue
        title = humanize(src.parent.name if src.parent != args.src else src.stem)
        key = "still" if src.suffix.lower() in STILL_EXT else "video"
        items.append(
            {
                "title": title,
                "caption": f"Tier-3 3D reactive-flow result ({src.name}).",
                key: dest_name,
            }
        )

    index_path.write_text(
        json.dumps(
            {
                "_comment": existing.get(
                    "_comment",
                    "Tier-3 gallery index. Populated by "
                    "scripts/web_export/export_tier3_gallery.py.",
                ),
                "items": items,
            },
            indent=1,
        )
    )
    print(f"scanned {args.src}: {len(found)} media file(s) -> {DEST}")
    if not found:
        print("  (nothing found — gallery stays empty; run a Tier-3 case first)")


if __name__ == "__main__":
    main()
