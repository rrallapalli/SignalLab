"""
reset_ticker.py — wipe a ticker's stored signals, run-markers, ingested-doc log,
and vector chunks so it can be re-run cleanly under the new FISCAL-quarter labels.

Why this is needed: signals stored before the fiscal switch are labelled with the
OLD calendar convention ("Q1 2026" meaning Jan–Mar). After the switch the same
string means a fiscal quarter (Apr–Jun). Mixing the two in one table makes the
trend charts and completeness banner compare mislabelled periods. Clearing the
affected tickers and re-running is the clean path.

Usage (from the project root, inside your .venv):
    python reset_ticker.py --ticker TCS
    python reset_ticker.py --ticker TCS --ticker INFY
    python reset_ticker.py --all          # every ticker (nuclear option)
"""

from __future__ import annotations
import argparse

import duckdb
from config import settings
from store.vector_store import VectorStore

SIGNAL_TABLES = [
    "confidence_signals",
    "narrative_signals",
    "guidance_signals",
    "risk_signals",
    "ingested_documents",
    "signal_runs",
]


def reset_ticker(ticker: str) -> None:
    ticker = ticker.upper()

    # 1) DuckDB rows across every table that keys on ticker.
    conn = duckdb.connect(str(settings.DUCKDB_PATH))
    try:
        for table in SIGNAL_TABLES:
            try:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE ticker = ?", [ticker]
                ).fetchone()[0]
                conn.execute(f"DELETE FROM {table} WHERE ticker = ?", [ticker])
                print(f"  {table:<22} — deleted {n} row(s)")
            except Exception as e:
                print(f"  {table:<22} — skipped ({e})")
    finally:
        conn.close()

    # 2) Vector chunks for the ticker.
    try:
        VectorStore().clear_ticker(ticker)
        print(f"  chroma chunks          — cleared for {ticker}")
    except Exception as e:
        print(f"  chroma chunks          — skipped ({e})")

    print(f"✅ {ticker} reset. Re-run the pipeline to repopulate under fiscal quarters.\n")


def all_tickers() -> list[str]:
    conn = duckdb.connect(str(settings.DUCKDB_PATH))
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM confidence_signals"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", default=[], help="Ticker(s) to reset")
    ap.add_argument("--all", action="store_true", help="Reset every stored ticker")
    args = ap.parse_args()

    targets = all_tickers() if args.all else [t.upper() for t in args.ticker]
    if not targets:
        ap.error("pass --ticker SYMBOL (repeatable) or --all")

    print(f"Resetting: {', '.join(targets)}\n")
    for t in targets:
        reset_ticker(t)
