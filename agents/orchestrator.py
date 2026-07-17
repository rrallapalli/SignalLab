"""
agents/orchestrator.py

Runs the full signal pipeline across three quarters in one shot:
  - Latest  (current or user-specified)
  - QoQ     (prior quarter)
  - YoY     (same quarter, prior year)

Key reliability improvements over v1:
  - Agents run SEQUENTIALLY within each quarter (prevents rate limit thundering herd)
  - Quarters run with staggered starts (further reduces peak API concurrency)
  - Each save call is isolated in try/except so one DuckDB failure
    can't kill the remaining agents
  - Per-quarter chunk count tracked; quarters with 0 chunks are skipped
    (agents produce hallucinated signals with no RAG evidence)
  - Granular error reporting per quarter + agent
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

# ── Signal version ────────────────────────────────────────────────────────────
#
# BUMP THIS whenever a change would produce a different signal from identical
# documents: prompts, scoring rubrics, retrieval filters, parsing, the schema.
#
# It is part of the corpus fingerprint, so bumping it invalidates every stored
# signal and forces a re-score. Without it, incremental scoring would look at
# Q1 2025 — whose filings have not changed since 2025 — and skip it forever,
# preserving whatever the agents believed back then. A cache that cannot be
# invalidated by a bug fix is a bug that cannot be fixed.
#
# History:
#   2026-07-17.1  fiscal_year retrieval filter; priors/deltas computed not
#                 asked; failures raise instead of scoring 5.0/45; guidance
#                 periods as (quarter, year) pairs; key_quotes verified
#                 against source text.
SIGNAL_VERSION = "2026-07-17.1"


def _prior_quarter(q: str, yr: int) -> tuple[str, int]:
    n = int(q[1])
    return (f"Q{n-1}", yr) if n > 1 else ("Q4", yr - 1)


def _yoy_quarter(q: str, yr: int) -> tuple[str, int]:
    return q, yr - 1


def _latest_completed_quarter() -> tuple[str, int]:
    """Returns the most recently completed quarter (calendar Q minus one)."""
    now = datetime.utcnow()
    current_q = (now.month - 1) // 3 + 1
    if current_q == 1:
        return "Q4", now.year - 1
    return f"Q{current_q - 1}", now.year


def resolve_quarters(
    quarter: str | None = None,
    year: int | None = None,
) -> tuple[tuple[str, int], tuple[str, int], tuple[str, int]]:
    if quarter and year:
        latest = (quarter, year)
    else:
        latest = _latest_completed_quarter()
    return latest, _prior_quarter(*latest), _yoy_quarter(*latest)


# ── LangGraph state ───────────────────────────────────────────────────────────

class ComparisonState(TypedDict):
    ticker:          str
    company:         str
    latest_q:        str
    latest_yr:       int
    qoq_q:           str
    qoq_yr:          int
    yoy_q:           str
    yoy_yr:          int
    docs_ingested:   int
    chunks_by_quarter: dict   # quarter_label → chunk_count
    latest_bundle:   Optional[dict]
    qoq_bundle:      Optional[dict]
    yoy_bundle:      Optional[dict]
    errors:          Annotated[list[str], operator.add]
    current_node:    str
    model:           Optional[str]   # per-run LLM override; None → settings default


# ── Ingestion ─────────────────────────────────────────────────────────────────

# Minimum chunks required to run agents for a quarter.
# Below this threshold the RAG evidence is too thin to trust.
MIN_CHUNKS_TO_SCORE = 5


async def _ingest_quarter(
    ticker: str, company: str, quarter: str, year: int,
    vs: VectorStore, ss: SignalStore,
) -> tuple[int, int]:
    """
    Fetch + chunk + embed one quarter.
    Returns (doc_count, chunk_count).
    """
    try:
        docs = await fetch_documents(
            ticker, company, quarter, year, include_prior=False
        )
    except Exception as e:
        logger.error(f"[ingest] Fetch failed {quarter} {year}: {e}")
        return 0, 0

    total_chunks = 0
    for doc in docs:
        try:
            chunks = chunk_document(doc)
            n = vs.upsert_chunks(chunks)
            total_chunks += n
            ss.log_document(
                ticker=doc.ticker, company=doc.company,
                doc_type=doc.doc_type.value,
                quarter=doc.quarter, fiscal_year=doc.fiscal_year,
                source_url=doc.source_url, title=doc.title,
                chunk_count=n, doc_id=doc.doc_id,
                raw_text=doc.raw_text,
            )
        except Exception as e:
            logger.warning(f"[ingest] Chunk/embed failed for {doc.doc_id}: {e}")

    logger.info(f"[ingest] {ticker} {quarter} {year}: {len(docs)} docs, {total_chunks} chunks")
    return len(docs), total_chunks


async def ingest_all(state: ComparisonState, vs: VectorStore, ss: SignalStore) -> dict:
    """
    Ingest all three quarters in parallel (I/O bound — safe to parallelise).
    Tracks per-quarter chunk counts so agents can be skipped if empty.
    """
    ticker, company = state["ticker"], state["company"]
    quarters = [
        (state["latest_q"], state["latest_yr"]),
        (state["qoq_q"],    state["qoq_yr"]),
        (state["yoy_q"],    state["yoy_yr"]),
    ]
    labels = [f"{q} {y}" for q, y in quarters]

    logger.info(
        f"[ingest] {ticker} → {labels[0]} | {labels[1]} | {labels[2]}"
    )

    results = await asyncio.gather(
        *[_ingest_quarter(ticker, company, q, y, vs, ss) for q, y in quarters],
        return_exceptions=True,
    )

    total_docs = 0
    chunks_by_quarter: dict[str, int] = {}
    errors: list[str] = []

    for label, r in zip(labels, results):
        if isinstance(r, Exception):
            errors.append(f"ingest {label}: {r}")
            chunks_by_quarter[label] = 0
        else:
            total_docs += r[0]
            chunks_by_quarter[label] = r[1]

    total_chunks = sum(chunks_by_quarter.values())

    for label, count in chunks_by_quarter.items():
        status = "✅" if count >= MIN_CHUNKS_TO_SCORE else "⚠️ too few"
        logger.info(f"[ingest]   {label}: {count} chunks {status}")

    logger.success(f"[ingest] Total: {total_docs} docs, {total_chunks} chunks")
    return {
        "docs_ingested":    total_docs,
        "chunks_by_quarter": chunks_by_quarter,
        "current_node":     "ingested",
        "errors":           errors,
    }


# ── Signal execution ──────────────────────────────────────────────────────────

async def _save_safe(save_fn, signal, label: str, errors: list) -> None:
    """Wrap a save call so a DuckDB error doesn't kill sibling agents."""
    try:
        save_fn(signal)
    except Exception as e:
        msg = f"save {label}: {e}"
        errors.append(msg)
        logger.error(f"[signals] {msg}")


async def _run_signals_for_quarter(
    ticker: str, company: str,
    quarter: str, year: int,
    prior_q: str, prior_yr: int,
    all_periods: list[tuple[str, int]],
    chunk_count: int,
    vs: VectorStore,
    ss: SignalStore,
    model: str | None = None,
) -> SignalBundle:
    """
    Run all four signal agents for one quarter SEQUENTIALLY with small gaps.
    Sequential execution prevents the rate-limit thundering herd that occurs
    when 12 LLM calls fire simultaneously across 3 quarters.
    """
    errors: list[str] = []
    label = f"{quarter} {year}"

    # Skip if too few chunks — agents would produce hallucinated signals
    if chunk_count < MIN_CHUNKS_TO_SCORE:
        msg = (
            f"{label}: only {chunk_count} chunks found "
            f"(need ≥{MIN_CHUNKS_TO_SCORE}). "
            "Signals skipped — try running again, or check that the NSE symbol / BSE "
            "scrip code are correct and this machine can reach nseindia.com and bseindia.com."
        )
        logger.warning(f"[signals] {msg}")
        errors.append(msg)
        return SignalBundle(
            ticker=ticker, company=company,
            quarter=quarter, fiscal_year=year,
            errors=errors,
        )

    logger.info(f"[signals] {ticker} {label}: running agents ({chunk_count} chunks available)")

    conf_sig = narr_sig = guid_sig = risk_sig = None
    agent_defs = [
        ("confidence", lambda: ConfidenceAgent(vs, model).run(ticker, company, quarter, year, prior_q, prior_yr)),
        ("narrative",  lambda: NarrativeAgent(vs, model).run(ticker, company, quarter, year, prior_q, prior_yr)),
        ("guidance",   lambda: GuidanceAgent(vs, model).run(ticker, company, quarter, year, all_periods)),
        ("risk",       lambda: RiskAgent(vs, model).run(ticker, company, quarter, year, prior_q, prior_yr)),
    ]

    for agent_name, agent_fn in agent_defs:
        try:
            result = await agent_fn()

            if agent_name == "confidence":
                conf_sig = result
                await _save_safe(ss.save_confidence, result, f"{label}/confidence", errors)
            elif agent_name == "narrative":
                narr_sig = result
                await _save_safe(ss.save_narrative, result, f"{label}/narrative", errors)
            elif agent_name == "guidance":
                guid_sig = result
                await _save_safe(ss.save_guidance, result, f"{label}/guidance", errors)
            elif agent_name == "risk":
                risk_sig = result
                await _save_safe(ss.save_risk, result, f"{label}/risk", errors)

            logger.success(f"[signals] {ticker} {label}/{agent_name} ✅")

        except Exception as e:
            msg = f"{label}/{agent_name}: {e}"
            errors.append(msg)
            logger.error(f"[signals] ❌ {msg}")

        # Small gap between agents to stay within rate limits
        await asyncio.sleep(0.4)

    return SignalBundle(
        ticker=ticker, company=company,
        quarter=quarter, fiscal_year=year,
        confidence=conf_sig, narrative=narr_sig,
        guidance=guid_sig, risk=risk_sig,
        errors=errors,
    )


async def run_all_signals(state: ComparisonState, vs: VectorStore, ss: SignalStore) -> dict:
    """
    Run signals for all three quarters with staggered starts.
    Quarters are staggered (not fully parallel) to spread LLM load:
      Latest starts immediately
      QoQ starts after 1s
      YoY starts after 2s
    This keeps peak concurrent LLM calls at ~2–3 rather than 12.
    """
    ticker, company = state["ticker"], state["company"]
    chunks = state.get("chunks_by_quarter", {})
    # (quarter, year) pairs — a bare quarter label is not a period.
    all_periods = [
        (state["latest_q"], state["latest_yr"]),
        (state["qoq_q"],    state["qoq_yr"]),
        (state["yoy_q"],    state["yoy_yr"]),
    ]
    model = state.get("model")   # per-run, not global

    async def _run_latest():
        return await _run_signals_for_quarter(
            ticker, company,
            state["latest_q"], state["latest_yr"],
            state["qoq_q"], state["qoq_yr"],
            all_periods,
            chunks.get(f"{state['latest_q']} {state['latest_yr']}", 0),
            vs, ss, model,
        )

    async def _run_qoq():
        await asyncio.sleep(1.0)   # stagger start
        return await _run_signals_for_quarter(
            ticker, company,
            state["qoq_q"], state["qoq_yr"],
            *_prior_quarter(state["qoq_q"], state["qoq_yr"]),
            all_periods,
            chunks.get(f"{state['qoq_q']} {state['qoq_yr']}", 0),
            vs, ss, model,
        )

    async def _run_yoy():
        await asyncio.sleep(2.0)   # stagger start
        return await _run_signals_for_quarter(
            ticker, company,
            state["yoy_q"], state["yoy_yr"],
            *_prior_quarter(state["yoy_q"], state["yoy_yr"]),
            all_periods,
            chunks.get(f"{state['yoy_q']} {state['yoy_yr']}", 0),
            vs, ss, model,
        )

    logger.info(f"[signals] Running 3 quarters (staggered) for {ticker}")
    latest_b, qoq_b, yoy_b = await asyncio.gather(
        _run_latest(), _run_qoq(), _run_yoy(),
        return_exceptions=True,
    )

    errors: list[str] = []

    def _safe_dump(b, label: str) -> dict | None:
        if isinstance(b, Exception):
            errors.append(f"{label}: {b}")
            logger.error(f"[signals] Quarter {label} failed: {b}")
            return None
        if b.errors:
            errors.extend(b.errors)
        return b.model_dump(mode="json")

    result = {
        "latest_bundle": _safe_dump(latest_b, f"{state['latest_q']} {state['latest_yr']}"),
        "qoq_bundle":    _safe_dump(qoq_b,    f"{state['qoq_q']} {state['qoq_yr']}"),
        "yoy_bundle":    _safe_dump(yoy_b,     f"{state['yoy_q']} {state['yoy_yr']}"),
        "current_node":  "signals_done",
        "errors":        errors,
    }

    completed = sum(1 for k in ["latest_bundle","qoq_bundle","yoy_bundle"] if result[k])
    logger.success(f"[signals] {ticker}: {completed}/3 quarters completed")
    return result


# ── Graph ─────────────────────────────────────────────────────────────────────

def _has_any_chunks(state: ComparisonState) -> str:
    """
    Proceed if ANY quarter has enough chunks.
    (Previous version required TOTAL > 0, which skipped all signals
     if any one quarter failed ingestion.)
    """
    chunks = state.get("chunks_by_quarter", {})
    if any(c >= MIN_CHUNKS_TO_SCORE for c in chunks.values()):
        return "ok"
    logger.warning("[graph] No quarter reached minimum chunk threshold — skipping signals")
    return "no_data"


def build_graph(vs: VectorStore, ss: SignalStore):
    async def _ingest(state):  return await ingest_all(state, vs, ss)
    async def _signals(state): return await run_all_signals(state, vs, ss)

    g = StateGraph(ComparisonState)
    g.add_node("ingest",  _ingest)
    g.add_node("signals", _signals)
    g.add_edge(START, "ingest")
    g.add_conditional_edges(
        "ingest", _has_any_chunks, {"ok": "signals", "no_data": END}
    )
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
    model: str | None = None,
) -> dict:
    if vs is None: vs = VectorStore()
    if ss is None: ss = SignalStore()

    (lq, ly), (qq, qy), (yq, yy) = resolve_quarters(quarter, year)

    logger.info(
        f"Pipeline: {ticker} | "
        f"Latest={lq} {ly} | QoQ={qq} {qy} | YoY={yq} {yy}"
    )

    graph = build_graph(vs, ss)

    initial: ComparisonState = {
        "ticker":  ticker.upper(),
        "company": company,
        "latest_q": lq, "latest_yr": ly,
        "qoq_q":   qq, "qoq_yr":   qy,
        "yoy_q":   yq, "yoy_yr":   yy,
        "docs_ingested":    0,
        "chunks_by_quarter": {},
        "latest_bundle":    None,
        "qoq_bundle":       None,
        "yoy_bundle":       None,
        "errors":           [],
        "current_node":     "start",
        "model":            model,
    }

    final: dict[str, Any] = {}
    async for chunk in graph.astream(initial, stream_mode="updates"):
        node = next(iter(chunk))
        final.update(chunk[node])

    return {
        "latest":        final.get("latest_bundle"),
        "qoq":           final.get("qoq_bundle"),
        "yoy":           final.get("yoy_bundle"),
        "latest_label":  f"{lq} {ly}",
        "qoq_label":     f"{qq} {qy}",
        "yoy_label":     f"{yq} {yy}",
        "docs_ingested": final.get("docs_ingested", 0),
        "chunks_by_quarter": final.get("chunks_by_quarter", {}),
        "errors":        final.get("errors", []),
    }
