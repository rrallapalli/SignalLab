"""
main.py – CLI entry point.

Usage:
  python main.py run --ticker TCS --company "Tata Consultancy Services" --quarter Q2 --year 2024
  python main.py api        # start FastAPI server
  python main.py dashboard  # launch Streamlit

`--ticker` is the NSE trading symbol (e.g. TCS, INFY, RELIANCE, HDFCBANK).
Documents are pulled directly from NSE and BSE corporate-announcement
filings — no API key is required for ingestion.
"""

from __future__ import annotations
import asyncio, json, sys
import argparse
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from loguru import logger

from config import settings

console = Console()

logger.remove()
logger.add("data/signal_agent.log", level="DEBUG", rotation="10 MB", retention="30 days")
logger.add(lambda msg: console.print(f"[dim]{msg}[/dim]") if "DEBUG" not in msg else None, level="INFO")


def _check_config():
    missing = []
    if not settings.OPENAI_API_KEY:  missing.append("OPENAI_API_KEY")
    if missing:
        console.print(f"[red]❌ Missing API keys: {', '.join(missing)}[/red]")
        console.print("[yellow]Copy .env.example → .env and fill in your keys.[/yellow]")
        sys.exit(1)


def _print_bundle(bundle) -> None:
    console.print()
    console.print(Panel(
        f"[bold cyan]Signal Bundle: {bundle.ticker} {bundle.quarter} {bundle.fiscal_year}[/bold cyan]\n"
        f"Docs ingested: {bundle.docs_ingested}",
        border_style="cyan",
    ))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Signal",  style="bold white", min_width=24)
    table.add_column("Score / Status", min_width=18)
    table.add_column("Key Finding", min_width=60)

    if bundle.confidence:
        c = bundle.confidence
        score_color = "green" if c.score >= 7 else "yellow" if c.score >= 5 else "red"
        delta = f" ({c.change:+.1f} QoQ)" if c.change is not None else ""
        table.add_row(
            "Management Confidence",
            f"[{score_color}]{c.score:.1f}/10{delta}[/{score_color}]",
            c.summary[:100] + ("…" if len(c.summary) > 100 else ""),
        )

    if bundle.narrative:
        n = bundle.narrative
        shift_color = {"positive":"green","negative":"red","mixed":"yellow","neutral":"dim"}.get(n.overall_shift,"dim")
        table.add_row(
            "Narrative Shift",
            f"[{shift_color}]{n.overall_shift.upper()}[/{shift_color}]",
            n.shift_summary[:100] + ("…" if len(n.shift_summary)>100 else ""),
        )

    if bundle.guidance:
        g = bundle.guidance
        score_color = "green" if g.score >= 70 else "yellow" if g.score >= 50 else "red"
        table.add_row(
            "Guidance Credibility",
            f"[{score_color}]{g.score:.0f}/100[/{score_color}]",
            g.summary[:100] + ("…" if len(g.summary)>100 else ""),
        )

    if bundle.risk:
        r = bundle.risk
        risk_color = {"increasing":"red","stable":"yellow","decreasing":"green"}.get(r.overall_risk_direction,"dim")
        new_n = len(r.new_risks)
        esc_n = len(r.escalating)
        table.add_row(
            "Risk Emergence",
            f"[{risk_color}]{r.overall_risk_direction.upper()}[/{risk_color}]",
            f"{new_n} new · {esc_n} escalating — {r.summary[:70]}…",
        )

    console.print(table)

    if bundle.errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for e in bundle.errors:
            if e: console.print(f"  · {e}")


async def cmd_run(args):
    _check_config()
    from agents.orchestrator import run_pipeline

    console.print(Panel(
        f"[bold]🔬 Signal Intelligence Pipeline[/bold]\n"
        f"{args.ticker.upper()} · {args.quarter} {args.year} · {datetime.utcnow():%H:%M UTC}",
        border_style="magenta",
    ))

    with console.status("[bold magenta]Running pipeline…[/bold magenta]"):
        bundle = await run_pipeline(
            ticker=args.ticker.upper(), company=args.company,
            quarter=args.quarter, fiscal_year=int(args.year),
        )

    _print_bundle(bundle)

    out_path = Path(f"data/{args.ticker.lower()}_{args.quarter}_{args.year}_signals.json")
    out_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]✅ Signals saved → {out_path}[/green]")
    console.print("[dim]Open the dashboard: streamlit run ui/dashboard.py[/dim]")


def cmd_api(args):
    import uvicorn
    console.print(Panel("[bold]🌐 Starting FastAPI server on http://localhost:8000[/bold]", border_style="blue"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


def cmd_dashboard(args):
    import subprocess
    console.print(Panel("[bold]📊 Launching Streamlit dashboard…[/bold]", border_style="green"))
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/dashboard.py"])


def main():
    parser = argparse.ArgumentParser(
        description="Signal Intelligence – RAG-powered equity signal generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run         Run the signal pipeline for a ticker + quarter
  api         Start the FastAPI backend (port 8000)
  dashboard   Launch the Streamlit UI

Examples:
  python main.py run --ticker TCS --company "Tata Consultancy Services" --quarter Q2 --year 2024
  python main.py run --ticker INFY --company "Infosys" --quarter Q1 --year 2025
  python main.py dashboard
  python main.py api
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run signal pipeline")
    run_p.add_argument("--ticker",  "-t", required=True)
    run_p.add_argument("--company", "-c", required=True, help="Full company name")
    run_p.add_argument("--quarter", "-q", default="Q2", choices=["Q1","Q2","Q3","Q4"])
    run_p.add_argument("--year",    "-y", default=2024, type=int)

    sub.add_parser("api",       help="Start FastAPI server")
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "api":
        cmd_api(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)


if __name__ == "__main__":
    main()
