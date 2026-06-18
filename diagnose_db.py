"""
diagnose_db.py — inspect what signals are stored in signals.duckdb

Run from the project root:
    python diagnose_db.py
    python diagnose_db.py --ticker AAPL   # focus on one ticker
"""

import sys, json, argparse
from pathlib import Path

import duckdb
from rich.console import Console
from rich.table import Table
from rich import box

DB_PATH = Path("data/signals.duckdb")
console = Console()


def check_db():
    if not DB_PATH.exists():
        console.print(f"[red]DB not found at {DB_PATH}[/red]")
        sys.exit(1)
    conn = duckdb.connect(str(DB_PATH))
    return conn


def show_summary(conn, ticker_filter=None):
    where = f"WHERE ticker = '{ticker_filter}'" if ticker_filter else ""
    order = "ORDER BY ticker, fiscal_year DESC, quarter DESC"

    console.print("\n[bold cyan]── Stored Tickers ──────────────────────────────[/bold cyan]")
    rows = conn.execute(
        "SELECT ticker, COUNT(*) as runs FROM confidence_signals GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    for ticker, count in rows:
        mark = "◀" if ticker == ticker_filter else ""
        console.print(f"  {ticker}  ({count} confidence rows)  {mark}")

    for signal_type, table in [
        ("Confidence",  "confidence_signals"),
        ("Narrative",   "narrative_signals"),
        ("Guidance",    "guidance_signals"),
        ("Risk",        "risk_signals"),
    ]:
        console.print(f"\n[bold yellow]── {signal_type} signals {where or '(all tickers)'} ──[/bold yellow]")

        try:
            rows = conn.execute(f"""
                SELECT ticker, quarter, fiscal_year, generated_at
                FROM {table}
                {where}
                {order}
                LIMIT 40
            """).fetchall()
        except Exception as e:
            console.print(f"  [red]Table error: {e}[/red]")
            continue

        if not rows:
            console.print("  [red]No rows found[/red]")
            continue

        t = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        t.add_column("Ticker")
        t.add_column("Quarter")
        t.add_column("Year")
        t.add_column("Generated At")
        for ticker, quarter, year, gen_at in rows:
            t.add_row(ticker, quarter, str(year), str(gen_at)[:19])
        console.print(t)

    # Check sub-dimension completeness for confidence
    console.print("\n[bold cyan]── Sub-dimension completeness (confidence) ──────[/bold cyan]")
    try:
        rows = conn.execute(f"""
            SELECT ticker, quarter, fiscal_year,
                   score,
                   confidence_level, uncertainty_level,
                   defensiveness, specificity,
                   consistency, forward_strength
            FROM confidence_signals
            {where}
            {order}
            LIMIT 20
        """).fetchall()
        t2 = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        t2.add_column("Ticker")
        t2.add_column("Quarter")
        t2.add_column("Score")
        t2.add_column("ConfLvl")
        t2.add_column("Uncert")
        t2.add_column("Defens")
        t2.add_column("Specif")
        t2.add_column("Consist")
        t2.add_column("FwdStr")
        for row in rows:
            ticker, quarter, year, score, cl, ul, df, sp, co, fw = row
            def _fmt(v):
                if v is None: return "[red]NULL[/red]"
                if float(v) == 0.0: return "[yellow]0.0[/yellow]"
                return f"[green]{float(v):.1f}[/green]"
            t2.add_row(
                ticker, f"{quarter} {year}",
                _fmt(score), _fmt(cl), _fmt(ul),
                _fmt(df), _fmt(sp), _fmt(co), _fmt(fw),
            )
        console.print(t2)
    except Exception as e:
        console.print(f"  [red]{e}[/red]")

    # Cross-check: which quarters have ALL four signals
    console.print("\n[bold cyan]── Quarter coverage (all four signals present?) ─[/bold cyan]")
    try:
        tickers = conn.execute(
            f"SELECT DISTINCT ticker FROM confidence_signals {where} ORDER BY ticker"
        ).fetchall()

        for (ticker,) in tickers:
            conf_qtrs = {
                f"{r[0]} {r[1]}"
                for r in conn.execute(
                    "SELECT quarter, fiscal_year FROM confidence_signals WHERE ticker=?",
                    [ticker]
                ).fetchall()
            }
            narr_qtrs = {
                f"{r[0]} {r[1]}"
                for r in conn.execute(
                    "SELECT quarter, fiscal_year FROM narrative_signals WHERE ticker=?",
                    [ticker]
                ).fetchall()
            }
            guid_qtrs = {
                f"{r[0]} {r[1]}"
                for r in conn.execute(
                    "SELECT quarter, fiscal_year FROM guidance_signals WHERE ticker=?",
                    [ticker]
                ).fetchall()
            }
            risk_qtrs = {
                f"{r[0]} {r[1]}"
                for r in conn.execute(
                    "SELECT quarter, fiscal_year FROM risk_signals WHERE ticker=?",
                    [ticker]
                ).fetchall()
            }
            all_qtrs = sorted(conf_qtrs | narr_qtrs | guid_qtrs | risk_qtrs, reverse=True)

            console.print(f"\n  [bold]{ticker}[/bold]")
            for q in all_qtrs[:8]:
                c = "✅" if q in conf_qtrs else "❌"
                n = "✅" if q in narr_qtrs else "❌"
                g = "✅" if q in guid_qtrs else "❌"
                r = "✅" if q in risk_qtrs else "❌"
                console.print(f"    {q:<12}  Conf:{c}  Narr:{n}  Guid:{g}  Risk:{r}")

    except Exception as e:
        console.print(f"  [red]{e}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", "-t", default=None)
    args = parser.parse_args()
    conn = check_db()
    show_summary(conn, args.ticker)
    console.print()
