#!/usr/bin/env python3
"""
reset_all.py — wipe ALL SignalLab state and begin fresh.

Removes: ChromaDB persist dir, DuckDB signal store, on-disk parse cache.
Does NOT touch source code or .env.

SAFE BY DEFAULT: prints what it *would* delete and exits. Pass --yes to delete.

Resolution order for each path:
  1. --chroma / --duckdb / --cache CLI overrides
  2. attributes on your config.py (tries several common names)
  3. conventional fallbacks under ./data

Usage:
  python reset_all.py                 # dry run, shows targets
  python reset_all.py --yes           # actually delete everything
  python reset_all.py --yes --keep-cache   # keep the parse cache
  python reset_all.py --duckdb data/signals.duckdb --chroma data/chroma --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CHROMA_ATTRS = ["CHROMA_DIR", "CHROMA_PERSIST_DIR", "CHROMADB_DIR",
                "VECTOR_STORE_DIR", "VECTOR_DIR", "CHROMA_PATH"]
DUCKDB_ATTRS = ["DUCKDB_PATH", "SIGNAL_DB_PATH", "SIGNAL_STORE_PATH",
                "DB_PATH", "DUCKDB_FILE"]
CACHE_ATTRS = ["PARSE_CACHE_DIR", "CACHE_DIR", "DOCLING_CACHE_DIR",
               "PARSER_CACHE_DIR"]
NARRATION_ATTRS = ["NARRATION_CACHE_DIR", "NARRATOR_CACHE_DIR"]

CHROMA_FALLBACK = "data/chroma"
DUCKDB_FALLBACK = "data/signals.duckdb"
CACHE_FALLBACK = "data/parse_cache"
NARRATION_FALLBACK = "data/narration_cache"


def _from_config(attrs: list[str]):
    try:
        import config  # your project's config.py
    except Exception:
        return None
    obj = getattr(config, "settings", config)
    for a in attrs:
        val = getattr(obj, a, None) or getattr(config, a, None)
        if val:
            return str(val)
    return None


def resolve(cli_val, attrs, fallback):
    return Path(cli_val or _from_config(attrs) or fallback).expanduser()


def show(label: str, path: Path):
    if not path.exists():
        print(f"  [absent ] {label}: {path}")
        return False
    kind = "dir " if path.is_dir() else "file"
    print(f"  [DELETE {kind}] {label}: {path.resolve()}")
    return True


def wipe(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"  removed: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Reset all SignalLab state.")
    p.add_argument("--yes", action="store_true", help="actually delete")
    p.add_argument("--chroma"); p.add_argument("--duckdb"); p.add_argument("--cache")
    p.add_argument("--narration")
    p.add_argument("--keep-chroma", action="store_true")
    p.add_argument("--keep-duckdb", action="store_true")
    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--keep-narration", action="store_true")
    a = p.parse_args()

    targets: list[tuple[str, Path]] = []
    if not a.keep_chroma:
        targets.append(("ChromaDB", resolve(a.chroma, CHROMA_ATTRS, CHROMA_FALLBACK)))
    if not a.keep_duckdb:
        targets.append(("DuckDB", resolve(a.duckdb, DUCKDB_ATTRS, DUCKDB_FALLBACK)))
    if not a.keep_cache:
        targets.append(("Parse cache", resolve(a.cache, CACHE_ATTRS, CACHE_FALLBACK)))
    if not a.keep_narration:
        targets.append(("Narration cache", resolve(a.narration, NARRATION_ATTRS, NARRATION_FALLBACK)))

    print("SignalLab reset — targets:")
    any_present = any(show(lbl, pth) for lbl, pth in targets)

    if not a.yes:
        print("\nDRY RUN. Nothing deleted. Re-run with --yes to delete.")
        if not any_present:
            print("(Nothing found at the resolved paths — pass --chroma/--duckdb/"
                  "--cache to point at the right locations.)")
        return 0

    print("\nDeleting...")
    for _, pth in targets:
        wipe(pth)
    print("\nDone. Fresh state.\n"
          "Next: bump SIGNAL_VERSION (and SCORER_VERSIONS), then Run Analysis "
          "to re-ingest via auto-pull. Verify with validate_run / diagnose_db.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
