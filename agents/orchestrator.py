"""
agents/orchestrator.py

Runs the full signal pipeline across three quarters in one shot:
  - Latest  (current or user-specified)
  - QoQ     (prior quarter)
  - YoY     (same quarter, prior year)

Exposes:
  run_comparison_pipeline(ticker, company, quarter=None, year=None)
"""

from __future__ import annotations
import asyncio, operator
from datetime import datetime
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from loguru import logger

from config import settings
from ingestion.fetcher import fetch_documents
from ingestion.chunker import chunk_document
from store.vector_store import VectorStore
from store.signal_store import SignalStore
from agents.confidence_agent import ConfidenceAgent
from agents.narrative_agent import NarrativeAgent
from agents.guidance_agent import GuidanceAgent
from agents.risk_agent import RiskAgent
from models import SignalBundle


# ── Quarter arithmetic ────────────────────────────────────────────────────────

def _prior_quarter(q: str, yr: int) -> tuple[str, int]:
    n = int(q[1])
    return (f"Q{n-1}", yr) if n > 1 else ("Q4", yr - 1)


def _yoy_quarter(q: str, yr: int) -> tuple[str, int]:
    return q, yr - 1


def _latest_completed_quarter() -> tuple[str, int]:
    """
    Returns the most recently *completed* quarter based on today's date.
    Assumes companies report ~6 weeks after quarter end, so we use
    calendar quarter minus one as the safe default.
    """
    now = datetime.utcnow()
    current_q = (now.month - 1) // 3 + 1   # 1-4
    if current_q == 1:
        return "Q4", now.year - 1
    return f"Q{current_q - 1}", now.year


def resolve_quarters(
    quarter: str | None = None,
    year: int | None = None,
) -> tuple[tuple[str,int], tuple[str,int], tuple[str,int]]:
    """
    Returns (latest, qoq, yoy) quarter tuples.
    If quarter/year not provided, auto-detects latest completed quarter.
    """
    if quarter and year:
        latest = (quarter, year)
    else:
        latest = _latest_completed_quarter()

    qoq = _prior_quarter(*latest)
    yoy = _yoy_quarter(*latest)
    return latest, qoq, yoy


# ── LangGraph state ───────────────────────────────────────────────────────────

class ComparisonState(TypedDict):
    ticker:       str
    company:      str
    latest_q:     str
    latest_yr:    int
    qoq_q:        str
    qoq_yr:       int
    yoy_q:        str
    yoy_yr:       int
    docs_ingested: int
    chunks_stored: int
    latest_bundle: Optional[dict]
    qoq_bundle:    Optional[dict]
    yoy_bundle:    Optional[dict]
    errors:        Annotated[list[str], operator.add]
    current_node:  str


# ── Ingest node: fetch + embed docs for all three quarters ────────────────────

async def _ingest_quarter(
    ticker: str, company: str, quarter: str, year: int,
    vs: VectorStore, ss: SignalStore,
) -> tuple[int, int]:
    """Fetch, chunk, embed documents for one quarter. Returns (docs, chunks)."""
    try:
        docs = await fetch_documents(ticker, company, quarter, year, include_prior=False)
    except Exception as e:
        logger.error(f"Fetch failed {quarter} {year}: {e}")
        return 0, 0

    total_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc)
        n = vs.upsert_chunks(chunks)
        total_chunks += n
        ss.log_document(
            ticker=doc.ticker, company=doc.company, doc_type=doc.doc_type.value,
            quarter=doc.quarter, fiscal_year=doc.fiscal_year,
            source_url=doc.source_url, title=doc.title,
            chunk_count=n, doc_id=doc.doc_id,
        )
    return len(docs), total_chunks


async def ingest_all(state: ComparisonState, vs: VectorStore, ss: SignalStore) -> dict:
    """Fetch and embed documents for latest, QoQ, and YoY quarters in parallel."""
    ticker, company = state["ticker"], state["company"]
    logger.info(f"[ingest] Fetching 3 quarters for {ticker}: "
                f"{state['latest_q']} {state['latest_yr']} / "
                f"{state['qoq_q']} {state['qoq_yr']} / "
                f"{state['yoy_q']} {state['yoy_yr']}")

    results = await asyncio.gather(
        _ingest_quarter(ticker, company, state["latest_q"], state["latest_yr"], vs, ss),
        _ingest_quarter(ticker, company, state["qoq_q"],    state["qoq_yr"],    vs, ss),
        _ingest_quarter(ticker, company, state["yoy_q"],    state["yoy_yr"],    vs, ss),
        return_exceptions=True,
    )

    total_docs, total_chunks = 0, 0
    errors = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            total_docs   += r[0]
            total_chunks += r[1]

    logger.success(f"[ingest] {total_docs} docs → {total_chunks} chunks across 3 quarters")
    return {
        "docs_ingested": total_docs,
        "chunks_stored": total_chunks,
        "current_node": "ingested",
        "errors": errors,
    }


# ── Signal node: run all four agents for one quarter ─────────────────────────

async def _run_signals_for_quarter(
    ticker: str, company: str,
    quarter: str, year: int,
    prior_q: str,
    all_quarters: list[str],
    vs: VectorStore, ss: SignalStore,
) -> SignalBundle:
    conf_agent = ConfidenceAgent(vs)
    narr_agent = NarrativeAgent(vs)
    guid_agent = GuidanceAgent(vs)
    risk_agent = RiskAgent(vs)

    results = await asyncio.gather(
        conf_agent.run(ticker, company, quarter, year, prior_q),
        narr_agent.run(ticker, company, quarter, year, prior_q),
        guid_agent.run(ticker, company, quarter, year, all_quarters),
        risk_agent.run(ticker, company, quarter, year, prior_q),
        return_exceptions=True,
    )

    errors = []
    conf_sig = narr_sig = guid_sig = risk_sig = None
    labels = ["confidence", "narrative", "guidance", "risk"]
    for label, r in zip(labels, results):
        if isinstance(r, Exception):
            errors.append(f"{quarter} {year} {label}: {r}")
            logger.error(f"[signals] {label} {quarter} {year}: {r}")
        else:
            if label == "confidence": conf_sig = r; ss.save_confidence(r)
            elif label == "narrative": narr_sig = r; ss.save_narrative(r)
            elif label == "guidance":  guid_sig = r; ss.save_guidance(r)
            elif label == "risk":      risk_sig = r; ss.save_risk(r)

    return SignalBundle(
        ticker=ticker, company=company,
        quarter=quarter, fiscal_year=year,
        confidence=conf_sig, narrative=narr_sig,
        guidance=guid_sig, risk=risk_sig,
        errors=errors,
    )


async def run_all_signals(state: ComparisonState, vs: VectorStore, ss: SignalStore) -> dict:
    """Run signal agents for all three quarters concurrently."""
    ticker, company = state["ticker"], state["company"]
    all_quarters = [
        state["latest_q"], state["qoq_q"], state["yoy_q"],
    ]

    logger.info(f"[signals] Running agents across 3 quarters for {ticker}")

    latest_b, qoq_b, yoy_b = await asyncio.gather(
        _run_signals_for_quarter(
            ticker, company,
            state["latest_q"], state["latest_yr"],
            state["qoq_q"], all_quarters, vs, ss,
        ),
        _run_signals_for_quarter(
            ticker, company,
            state["qoq_q"], state["qoq_yr"],
            _prior_quarter(state["qoq_q"], state["qoq_yr"])[0],
            all_quarters, vs, ss,
        ),
        _run_signals_for_quarter(
            ticker, company,
            state["yoy_q"], state["yoy_yr"],
            _prior_quarter(state["yoy_q"], state["yoy_yr"])[0],
            all_quarters, vs, ss,
        ),
        return_exceptions=True,
    )

    errors = []
    def _safe_dump(b):
        if isinstance(b, Exception):
            errors.append(str(b))
            return None
        return b.model_dump(mode="json")

    logger.success(f"[signals] All three quarters complete for {ticker}")
    return {
        "latest_bundle": _safe_dump(latest_b),
        "qoq_bundle":    _safe_dump(qoq_b),
        "yoy_bundle":    _safe_dump(yoy_b),
        "current_node":  "signals_done",
        "errors":        errors,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def _has_chunks(state: ComparisonState) -> str:
    return "ok" if state.get("chunks_stored", 0) > 0 else "no_data"


def build_graph(vs: VectorStore, ss: SignalStore):
    async def _ingest(state):  return await ingest_all(state, vs, ss)
    async def _signals(state): return await run_all_signals(state, vs, ss)

    g = StateGraph(ComparisonState)
    g.add_node("ingest",  _ingest)
    g.add_node("signals", _signals)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _has_chunks, {"ok": "signals", "no_data": END})
    g.add_edge("signals", END)
    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

async def run_comparison_pipeline(
    ticker: str,
    company: str,
    quarter: str | None = None,
    year: int | None = None,
    vs: VectorStore | None = None,
    ss: SignalStore | None = None,
) -> dict:
    """
    Main entry point.
    Returns dict with keys: latest, qoq, yoy → each a SignalBundle dict.
    Also returns: latest_label, qoq_label, yoy_label for display.
    """
    if vs is None: vs = VectorStore()
    if ss is None: ss = SignalStore()

    (lq, ly), (qq, qy), (yq, yy) = resolve_quarters(quarter, year)

    logger.info(
        f"Comparison pipeline: {ticker} | "
        f"Latest={lq} {ly} | QoQ={qq} {qy} | YoY={yq} {yy}"
    )

    graph = build_graph(vs, ss)

    initial: ComparisonState = {
        "ticker":  ticker.upper(),
        "company": company,
        "latest_q": lq, "latest_yr": ly,
        "qoq_q":   qq, "qoq_yr":   qy,
        "yoy_q":   yq, "yoy_yr":   yy,
        "docs_ingested": 0,
        "chunks_stored": 0,
        "latest_bundle": None,
        "qoq_bundle":    None,
        "yoy_bundle":    None,
        "errors": [],
        "current_node": "start",
    }

    final: dict[str, Any] = {}
    async for chunk in graph.astream(initial, stream_mode="updates"):
        node = next(iter(chunk))
        final.update(chunk[node])

    return {
        "latest":       final.get("latest_bundle"),
        "qoq":          final.get("qoq_bundle"),
        "yoy":          final.get("yoy_bundle"),
        "latest_label": f"{lq} {ly}",
        "qoq_label":    f"{qq} {qy}",
        "yoy_label":    f"{yq} {yy}",
        "docs_ingested": final.get("docs_ingested", 0),
        "errors":       final.get("errors", []),
    }
