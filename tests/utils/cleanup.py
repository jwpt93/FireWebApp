from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable


def prune_subdirs(parent: Path, keep: int = 3, exclude: Iterable[str] | None = None) -> None:
    """Keep the newest N subdirectories under parent; delete older ones."""
    if keep < 1 or not parent.exists():
        return
    exclude_set = set(exclude or [])
    subdirs = [p for p in parent.iterdir() if p.is_dir() and p.name not in exclude_set]
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in subdirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)
