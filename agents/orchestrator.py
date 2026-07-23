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
from typing import Annotated, Any, Callable, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from loguru import logger

from config import settings
from ingestion.fetcher import fetch_all_periods
from ingestion.chunker import chunk_document
from store.vector_store import VectorStore
from store.signal_store import SignalStore
from agents.confidence_agent import ConfidenceAgent
from agents.narrative_agent import NarrativeAgent
from agents.guidance_agent import GuidanceAgent
from agents.risk_agent import RiskAgent
from models import SignalBundle, format_period


# ── Progress reporting ────────────────────────────────────────────────────────
#
# The pipeline reports coarse, ordered stages so a UI can show live progress.
# `progress` is an optional callable(event: str, detail: dict) invoked from
# whatever thread/loop the pipeline runs on — so a UI callback MUST be
# thread-safe (e.g. push onto a queue and update widgets from the main thread).
#
# Event vocabulary (in emission order):
#   pipeline_start   {ticker, latest, qoq, yoy}
#   ingest_start     {quarters: [label, ...]}
#   quarter_ingested {label, docs, chunks}         — one per quarter
#   ingest_done      {total_docs, total_chunks}
#   signals_skipped  {reason}                       — emitted iff no quarter had data
#   signals_start    {quarters: [label, ...]}
#   quarter_scored   {label, ok, errors}            — one per quarter
#   signals_done     {completed, total}
#   pipeline_done    {docs_ingested}

ProgressFn = Callable[[str, dict], None]


def _noop_progress(event: str, detail: dict) -> None:
    pass


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
#   2026-07-17.2  documents dated from their subject line, not the search
#                 window. Overlapping windows had filed one earnings call as
#                 both Q4 2025 and Q1 2026, so quarters scored duplicate
#                 evidence. Every stored signal predates this and must
#                 re-score — ingestion changed underneath them.
SIGNAL_VERSION = "2026-07-17.2"


def _retrieval_profile() -> str:
    """
    The retrieval settings that decide which chunks reach a prompt, as a stable
    string for corpus_fingerprint().

    Model and agent version were already fingerprinted; this closes the third
    input. Two runs over identical documents with identical prompts still
    produce different scores if one reranked 25 candidates per query down to 12
    and the other took the raw vector top-12 — the reranker changes what the
    agent reads, not just the order it reads it in.
    """
    if not settings.RERANK_ENABLED:
        return "vector_order"
    return (
        f"rerank:{settings.RERANK_MODEL}"
        f":cand{settings.RERANK_CANDIDATES_PER_QUERY}"
        f":top{settings.RERANK_TOP_N}"
    )


def _prior_quarter(q: str, yr: int) -> tuple[str, int]:
    n = int(q[1])
    return (f"Q{n-1}", yr) if n > 1 else ("Q4", yr - 1)


def _yoy_quarter(q: str, yr: int) -> tuple[str, int]:
    return q, yr - 1


def _latest_completed_quarter() -> tuple[str, int]:
    """Most recently completed Indian FISCAL quarter (year runs Apr–Mar)."""
    now = datetime.utcnow()
    m = now.month
    cq = (m - 4) % 12 // 3 + 1              # current fiscal quarter
    cfy = now.year + 1 if m >= 4 else now.year
    return _prior_quarter(f"Q{cq}", cfy)    # the one before it is the last completed


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


async def _embed_period_docs(
    docs: list, quarter: str, year: int,
    vs: VectorStore, ss: SignalStore,
    progress: ProgressFn = _noop_progress, label: str | None = None,
) -> tuple[int, int]:
    """
    Chunk + embed already-fetched documents for one period.
    Returns (doc_count, chunk_count). Fetching happens once, upstream, in
    fetch_all_periods — so a given PDF can only ever reach one period here.
    """
    label = label or format_period(quarter, year)

    # Drop chunks from documents that are no longer part of this period's
    # corpus BEFORE embedding the current set. Without this, a document that
    # stops being ingested (dropped as a duplicate, re-parsed under a different
    # id, cut by ranking) leaves its chunks behind, still retrievable — and
    # agents go on citing evidence that no ingested document contains.
    try:
        vs.prune_period(
            ticker=docs[0].ticker if docs else "",
            quarter=quarter, fiscal_year=year,
            keep_doc_ids={d.doc_id for d in docs},
        )
    except Exception as e:
        logger.warning(f"[ingest] Prune failed for {label}: {e}")

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

    logger.info(f"[ingest] {label}: {len(docs)} docs, {total_chunks} chunks")
    progress("quarter_ingested", {"label": label, "docs": len(docs), "chunks": total_chunks})
    return len(docs), total_chunks


async def ingest_all(
    state: ComparisonState, vs: VectorStore, ss: SignalStore,
    progress: ProgressFn = _noop_progress,
) -> dict:
    """
    Ingest all three periods from a SINGLE fetch pass.

    fetch_all_periods pulls one wide NSE/BSE window covering every requested
    period and buckets each unique document by the period it is actually about,
    so a filing can never land in two quarters (the old per-quarter fetch calls
    each had their own overlapping window + dedupe set, which let an undateable
    PDF be kept twice). Chunk/embed then runs per bucket.
    """
    ticker, company = state["ticker"], state["company"]
    quarters = [
        (state["latest_q"], state["latest_yr"]),
        (state["qoq_q"],    state["qoq_yr"]),
        (state["yoy_q"],    state["yoy_yr"]),
    ]
    labels = {p: format_period(*p) for p in quarters}

    logger.info(f"[ingest] {ticker} → " + " | ".join(labels[p] for p in quarters))
    progress("ingest_start", {"quarters": [labels[p] for p in quarters]})

    # One fetch for all periods.
    try:
        buckets = await fetch_all_periods(ticker, company, quarters)
    except Exception as e:
        logger.error(f"[ingest] Fetch failed: {e}")
        buckets = {p: [] for p in quarters}
        # Still emit per-quarter events so the UI resolves each stage.
        for p in quarters:
            progress("quarter_ingested", {"label": labels[p], "docs": 0, "chunks": 0})
        progress("ingest_done", {"total_docs": 0, "total_chunks": 0})
        return {
            "docs_ingested": 0, "chunks_by_quarter": {labels[p]: 0 for p in quarters},
            "current_node": "ingested", "errors": [f"fetch: {e}"],
        }

    # Chunk/embed per bucket. Sequential keeps DuckDB's single writer happy and
    # preserves the per-quarter progress cadence the dashboard renders.
    total_docs = 0
    chunks_by_quarter: dict[str, int] = {}
    errors: list[str] = []
    for p in quarters:
        label = labels[p]
        try:
            d, c = await _embed_period_docs(buckets.get(p, []), p[0], p[1], vs, ss, progress, label)
            total_docs += d
            chunks_by_quarter[label] = c
        except Exception as e:
            errors.append(f"ingest {label}: {e}")
            chunks_by_quarter[label] = 0
            progress("quarter_ingested", {"label": label, "docs": 0, "chunks": 0})

    total_chunks = sum(chunks_by_quarter.values())
    for label, count in chunks_by_quarter.items():
        status = "✅" if count >= MIN_CHUNKS_TO_SCORE else "⚠️ too few"
        logger.info(f"[ingest]   {label}: {count} chunks {status}")

    logger.success(f"[ingest] Total: {total_docs} docs, {total_chunks} chunks")
    progress("ingest_done", {"total_docs": total_docs, "total_chunks": total_chunks})
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
    label = format_period(quarter, year)

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

    # ── Reuse this period if nothing that produced its signal has changed ────
    #
    # A re-run of Q1 2026 previously re-scored Q4 2025 and Q1 2025 from scratch
    # every time — 3x the LLM cost for 1x of new information, overwriting
    # identical answers via INSERT OR REPLACE. The fingerprint covers the
    # documents, the model AND SIGNAL_VERSION, so a prompt or scoring fix still
    # invalidates every period even though old filings never change.
    effective_model = model or settings.OPENAI_MODEL
    fingerprint = ss.corpus_fingerprint(
        ticker, quarter, year, effective_model, SIGNAL_VERSION, _retrieval_profile()
    )

    if ss.is_signal_current(ticker, quarter, year, fingerprint):
        stored = ss.get_period_signals(ticker, quarter, year)
        if all(stored.get(k) is not None for k in ("confidence", "narrative", "guidance", "risk")):
            logger.info(
                f"[signals] {ticker} {label}: corpus, model and agent version unchanged "
                f"— reusing stored signals (no LLM calls)"
            )
            return SignalBundle(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=year,
                confidence=stored["confidence"], narrative=stored["narrative"],
                guidance=stored["guidance"], risk=stored["risk"],
                errors=[],
            )
        logger.info(
            f"[signals] {ticker} {label}: marked current but stored signals are "
            f"incomplete — re-scoring."
        )

    logger.info(f"[signals] {ticker} {label}: running agents ({chunk_count} chunks available)")

    conf_sig = narr_sig = guid_sig = risk_sig = None
    # Instances are held rather than created inside the lambda: each agent
    # records how retrieval actually behaved (reranked vs degraded to vector
    # order) on itself, and that has to be readable after run() returns.
    agents = {
        "confidence": ConfidenceAgent(vs, model),
        "narrative":  NarrativeAgent(vs, model),
        "guidance":   GuidanceAgent(vs, model),
        "risk":       RiskAgent(vs, model),
    }
    agent_defs = [
        ("confidence", lambda: agents["confidence"].run(ticker, company, quarter, year, prior_q, prior_yr)),
        ("narrative",  lambda: agents["narrative"].run(ticker, company, quarter, year, prior_q, prior_yr)),
        ("guidance",   lambda: agents["guidance"].run(ticker, company, quarter, year, all_periods)),
        ("risk",       lambda: agents["risk"].run(ticker, company, quarter, year, prior_q, prior_yr)),
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

    # A degraded run reads a different evidence set than the fingerprint claims.
    # The profile in the hash describes the CONFIGURED retrieval path; if the
    # reranker was meant to run and could not, the resulting signals are not
    # what that fingerprint stands for, and caching them would freeze a
    # fallback-quality answer behind a "current" marker. Leave the period
    # unmarked so a healthy run supersedes it. (RERANK_ENABLED=false is a
    # different case entirely — that is vector order on purpose, it profiles as
    # "vector_order", and it caches normally.)
    degraded = sorted(
        name for name, a in agents.items()
        if getattr(a, "retrieval_mode", "") == "vector_order_degraded"
    )
    if degraded:
        msg = (
            f"{label}: reranker unavailable during {', '.join(degraded)} — scored on "
            f"vector order instead. Signals kept but NOT cached; they will re-score "
            f"once the reranker loads."
        )
        logger.warning(f"[signals] {msg}")
        errors.append(msg)

    # Only record this period as scored if ALL FOUR agents succeeded. A partial
    # run must stay un-marked so the next run retries it — caching a failure is
    # how a transient rate-limit becomes a permanently missing signal.
    if not errors and all(s is not None for s in (conf_sig, narr_sig, guid_sig, risk_sig)):
        ss.mark_scored(ticker, quarter, year, fingerprint)
    else:
        logger.info(
            f"[signals] {ticker} {label}: not marked current "
            f"({len(errors)} error(s)) — will be retried on the next run."
        )

    return SignalBundle(
        ticker=ticker, company=company,
        quarter=quarter, fiscal_year=year,
        confidence=conf_sig, narrative=narr_sig,
        guidance=guid_sig, risk=risk_sig,
        errors=errors,
    )


async def run_all_signals(
    state: ComparisonState, vs: VectorStore, ss: SignalStore,
    progress: ProgressFn = _noop_progress,
) -> dict:
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

    labels = [
        format_period(state['latest_q'], state['latest_yr']),
        format_period(state['qoq_q'], state['qoq_yr']),
        format_period(state['yoy_q'], state['yoy_yr']),
    ]
    progress("signals_start", {"quarters": labels})

    def _emit_scored(label: str, bundle) -> None:
        # Called as each quarter's agents finish, regardless of gather ordering.
        ok = not isinstance(bundle, Exception) and not getattr(bundle, "errors", None)
        errs = (
            [str(bundle)] if isinstance(bundle, Exception)
            else list(getattr(bundle, "errors", []) or [])
        )
        progress("quarter_scored", {"label": label, "ok": ok, "errors": errs})

    async def _run_latest():
        b = await _run_signals_for_quarter(
            ticker, company,
            state["latest_q"], state["latest_yr"],
            state["qoq_q"], state["qoq_yr"],
            all_periods,
            chunks.get(labels[0], 0),
            vs, ss, model,
        )
        _emit_scored(labels[0], b)
        return b

    async def _run_qoq():
        await asyncio.sleep(1.0)   # stagger start
        b = await _run_signals_for_quarter(
            ticker, company,
            state["qoq_q"], state["qoq_yr"],
            *_prior_quarter(state["qoq_q"], state["qoq_yr"]),
            all_periods,
            chunks.get(labels[1], 0),
            vs, ss, model,
        )
        _emit_scored(labels[1], b)
        return b

    async def _run_yoy():
        await asyncio.sleep(2.0)   # stagger start
        b = await _run_signals_for_quarter(
            ticker, company,
            state["yoy_q"], state["yoy_yr"],
            *_prior_quarter(state["yoy_q"], state["yoy_yr"]),
            all_periods,
            chunks.get(labels[2], 0),
            vs, ss, model,
        )
        _emit_scored(labels[2], b)
        return b

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
        "latest_bundle": _safe_dump(latest_b, labels[0]),
        "qoq_bundle":    _safe_dump(qoq_b,    labels[1]),
        "yoy_bundle":    _safe_dump(yoy_b,    labels[2]),
        "current_node":  "signals_done",
        "errors":        errors,
    }

    completed = sum(1 for k in ["latest_bundle","qoq_bundle","yoy_bundle"] if result[k])
    logger.success(f"[signals] {ticker}: {completed}/3 quarters completed")
    progress("signals_done", {"completed": completed, "total": 3})
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


def build_graph(vs: VectorStore, ss: SignalStore, progress: ProgressFn = _noop_progress):
    async def _ingest(state):  return await ingest_all(state, vs, ss, progress)
    async def _signals(state): return await run_all_signals(state, vs, ss, progress)

    def _route(state: ComparisonState) -> str:
        decision = _has_any_chunks(state)
        if decision == "no_data":
            progress("signals_skipped", {"reason": "no quarter reached the minimum chunk threshold"})
        return decision

    g = StateGraph(ComparisonState)
    g.add_node("ingest",  _ingest)
    g.add_node("signals", _signals)
    g.add_edge(START, "ingest")
    g.add_conditional_edges(
        "ingest", _route, {"ok": "signals", "no_data": END}
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
    progress: ProgressFn | None = None,
) -> dict:
    progress = progress or _noop_progress
    if vs is None: vs = VectorStore()
    if ss is None: ss = SignalStore()

    (lq, ly), (qq, qy), (yq, yy) = resolve_quarters(quarter, year)

    logger.info(
        f"Pipeline: {ticker} | "
        f"Latest={lq} {ly} | QoQ={qq} {qy} | YoY={yq} {yy}"
    )
    progress("pipeline_start", {
        "ticker": ticker.upper(),
        "latest": format_period(lq, ly), "qoq": format_period(qq, qy), "yoy": format_period(yq, yy),
    })

    graph = build_graph(vs, ss, progress)

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

    progress("pipeline_done", {"docs_ingested": final.get("docs_ingested", 0)})
    return {
        "latest":        final.get("latest_bundle"),
        "qoq":           final.get("qoq_bundle"),
        "yoy":           final.get("yoy_bundle"),
        "latest_label":  format_period(lq, ly),
        "qoq_label":     format_period(qq, qy),
        "yoy_label":     format_period(yq, yy),
        "docs_ingested": final.get("docs_ingested", 0),
        "chunks_by_quarter": final.get("chunks_by_quarter", {}),
        "errors":        final.get("errors", []),
    }
