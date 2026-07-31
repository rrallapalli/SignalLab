"""agents/base.py – Shared agent infrastructure with retry and evidence formatting."""

from __future__ import annotations
import json, re, asyncio, os, uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import logging

from config import settings
from store.vector_store import VectorStore

# Exceptions worth retrying
try:
    import openai
    _RETRY_EXC = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
except ImportError:
    _RETRY_EXC = (Exception,)


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    """
    dict.get(key, default) only falls back when the key is ABSENT — if the
    LLM returns the key with an explicit `null`, .get() still returns None
    and float(None) raises. Use this instead of float(data.get(...)).

    Pass default=None for values where there is no honest fallback. A missing
    score is not a zero and not a midpoint — it is an absent measurement, and
    substituting a number for it publishes a figure no evidence supports.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Same null-safety as safe_float(), for integer fields."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Model capability detection
# ─────────────────────────────────────────────────────────────────────────────
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """
    Reasoning models — the GPT-5.x family (Sol / Terra / Luna) and the o-series —
    reject `temperature` and `seed` and use a reasoning-effort control instead.
    gpt-4o and gpt-4o-mini are NOT reasoning models and take temperature/seed.
    """
    return (model or "").lower().startswith(_REASONING_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Per-call token accounting + per-run cost log (logs/<TICKER>_<stamp>.log)
# ─────────────────────────────────────────────────────────────────────────────
# USD per 1,000,000 tokens: (input, output). Reasoning tokens are counted within
# `output` and billed at the output rate, so no special case is needed in the
# cost math; they are logged separately for visibility. Update as prices change.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15,  0.60),
    "gpt-5.6-luna":  (1.00,  6.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol":   (5.00, 30.00),
}


def _price_for(model: str) -> tuple[float, float] | None:
    m = (model or "").lower()
    if m in _MODEL_PRICES:
        return _MODEL_PRICES[m]
    for k, v in _MODEL_PRICES.items():      # tolerate dated snapshots
        if m.startswith(k):
            return v
    return None


def _slug(s: str) -> str:
    """Filesystem-safe token from a session id (or anything)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "", str(s))[:24] or "anon"


class RunCostLog:
    """Accumulates every LLM call in one ticker run; writes logs/<file> live."""

    def __init__(self, ticker: str, model: str | None, path: "Path | None" = None):
        self.ticker = (ticker or "UNKNOWN").upper()
        self.model = model or "?"
        self.started = datetime.now(timezone.utc)
        self.calls: list[dict] = []
        if path is None:
            stamp = self.started.strftime("%Y%m%d_%H%M%S")
            log_dir = Path(getattr(settings, "LOG_DIR", "logs")).expanduser()
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            path = log_dir / f"{self.ticker}_{stamp}.cost.log"
        self.path = path
        self._write(f"# SignalLab run cost log — {self.ticker} — {self.started.isoformat()}")
        self._write(f"# {'call':<22} {'model':<16} {'in':>8} {'out':>8} "
                    f"{'reason':>7} {'cost_usd':>10}")

    def _write(self, line: str) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning(f"[cost] log write failed: {e}")

    def record(self, model: str, in_tok: int, out_tok: int,
               reason_tok: int, label: str = "") -> None:
        price = _price_for(model)
        cost = (in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]) if price else None
        self.calls.append({"model": model, "in": in_tok, "out": out_tok,
                           "reasoning": reason_tok, "cost": cost, "label": label})
        self._write(f"  {label[:22]:<22} {model[:16]:<16} {in_tok:>8} {out_tok:>8} "
                    f"{reason_tok:>7} {('%.5f' % cost) if cost is not None else 'n/a':>10}")

    def finish(self) -> dict:
        tin = sum(c["in"] for c in self.calls)
        tout = sum(c["out"] for c in self.calls)
        tr = sum(c["reasoning"] for c in self.calls)
        tcost = sum(c["cost"] for c in self.calls if c["cost"] is not None)
        unpriced = any(c["cost"] is None for c in self.calls)
        secs = (datetime.now(timezone.utc) - self.started).total_seconds()
        self._write("# " + "-" * 72)
        self._write(f"# TOTAL  calls={len(self.calls)}  in={tin}  out={tout}  "
                    f"reasoning={tr}  cost_usd={tcost:.5f}"
                    + ("  (+unpriced call(s) — model not in price table)" if unpriced else "")
                    + f"  elapsed={secs:.1f}s")
        return {"calls": len(self.calls), "input_tokens": tin, "output_tokens": tout,
                "reasoning_tokens": tr, "cost_usd": round(tcost, 5),
                "unpriced": unpriced, "path": str(self.path)}


# contextvar (not a module global) so concurrent dashboard runs each get their
# own log — the same reason the model is passed per run, not via settings.
_run_cost_log: ContextVar["RunCostLog | None"] = ContextVar("_run_cost_log", default=None)


class run_cost_log:
    """
    Context manager: open a per-run cost log for `ticker` and finalise it on exit.
    Wrap a whole ticker run in this; every llm_reason() call inside records into
    it automatically (via a contextvar, so it survives asyncio.gather).
    """
    def __init__(self, ticker: str, model: str | None = None,
                 session_id: str | None = None):
        self.ticker, self.model, self.session_id = ticker, model, session_id
        # Unique per run. Tags this run's log records so its trace sink captures
        # ONLY them — the mechanism that keeps concurrent (multi-user) runs in
        # separate files instead of every open sink capturing every run.
        self.run_id = uuid.uuid4().hex[:8]
        self._token = None
        self._sink_id = None
        self._ctx = None
        self.log: "RunCostLog | None" = None
        self.trace_path: "Path | None" = None

    def __enter__(self) -> "RunCostLog":
        tk = (self.ticker or "UNKNOWN").upper()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = Path(getattr(settings, "LOG_DIR", "logs")).expanduser()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # All runs share one folder; the session id and a unique run_id live in
        # the filename, so concurrent users' files never collide and are easy to
        # filter (ls TICKER_session_*  /  ls *_<session>_*).
        sess = _slug(self.session_id) if self.session_id else "nosession"
        base = f"{tk}_{sess}_{stamp}_{self.run_id}"
        self.trace_path = log_dir / f"{base}.log"        # execution trace
        cost_path       = log_dir / f"{base}.cost.log"   # token/cost ledger

        # Bind run_id onto every log emitted within this run's context. loguru's
        # contextualize is contextvar-based, so the tag propagates across
        # asyncio.gather to the per-quarter tasks — no changes to any call site.
        self._ctx = logger.contextualize(run_id=self.run_id)
        self._ctx.__enter__()

        # Execution trace: a temporary loguru sink FILTERED to this run_id, so it
        # captures only this run's records even while other runs' sinks are open.
        try:
            _rid = self.run_id
            self._sink_id = logger.add(
                str(self.trace_path),
                level=getattr(settings, "RUN_LOG_LEVEL", "DEBUG"),
                filter=lambda rec, r=_rid: rec["extra"].get("run_id") == r,
                enqueue=True, backtrace=False, diagnose=False,
            )
        except Exception as e:
            self._sink_id = None
            logger.warning(f"[log] could not open run trace log: {e}")

        self.log = RunCostLog(self.ticker, self.model, path=cost_path)
        self._token = _run_cost_log.set(self.log)
        return self.log

    def __exit__(self, *exc) -> None:
        try:
            s = self.log.finish()
            logger.info(
                f"[cost] {self.ticker}: {s['calls']} calls, ${s['cost_usd']}"
                + (" (some unpriced)" if s['unpriced'] else "")
                + f" → {s['path']}  |  trace → {self.trace_path}"
            )
        finally:
            if self._token is not None:
                _run_cost_log.reset(self._token)
            if self._sink_id is not None:
                try:
                    logger.remove(self._sink_id)   # flushes the enqueued sink
                except Exception:
                    pass
            if self._ctx is not None:
                try:
                    self._ctx.__exit__(*exc)
                except Exception:
                    pass


def _extract_usage(resp) -> tuple[int, int, int]:
    """(input, output, reasoning) token counts from a LangChain response,
    tolerant of both usage_metadata and response_metadata['token_usage']."""
    um = getattr(resp, "usage_metadata", None) or {}
    if um:
        return (
            int(um.get("input_tokens", 0) or 0),
            int(um.get("output_tokens", 0) or 0),
            int((um.get("output_token_details") or {}).get("reasoning", 0) or 0),
        )
    tu = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {}) or {}
    return (
        int(tu.get("prompt_tokens", 0) or 0),
        int(tu.get("completion_tokens", 0) or 0),
        int((tu.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0),
    )


class BaseAgent:
    """
    Base for all signal agents.
    Provides:
      - rag_retrieve()  → retrieve evidence from vector store
      - llm_reason()    → structured JSON output with automatic retry
    """

    def __init__(self, vector_store: VectorStore, model: str | None = None):
        """
        `model` is passed per-run rather than read from global settings.
        Mutating settings.OPENAI_MODEL to choose a model is unsafe: the
        dashboard serves every browser session from one process, so a second
        user's choice would silently change the model mid-run for the first.
        """
        self.vs = vector_store
        self.model_name = model or settings.OPENAI_MODEL
        # What retrieval actually did on the last rag_retrieve() call. Read it
        # after scoring: "reranked" and "vector_order" are different evidence
        # sets, and a signal built on the degraded path should not be cached as
        # if it were built on the good one.
        self.retrieval_mode: str = "unknown"
        self.failed_queries: list[str] = []

        reasoning = _is_reasoning_model(self.model_name)

        model_kwargs: dict[str, Any] = {
            # Forces the API itself to guarantee syntactically valid JSON output,
            # instead of relying on the model voluntarily following "return only
            # JSON" prompt instructions (which occasionally breaks on embedded
            # quotes/apostrophes in quoted evidence text).
            "response_format": {"type": "json_object"},
        }
        llm_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            api_key=settings.OPENAI_API_KEY,
            model_kwargs=model_kwargs,
        )

        if reasoning:
            # Reasoning models (GPT-5.x incl. Luna, o-series) reject temperature
            # and seed. Omit both and use reasoning effort instead — default LOW,
            # because extraction is structured output, not a reasoning task, so
            # paying for deep reasoning tokens is mostly waste. Determinism from
            # seed/temperature is unavailable here, but the SCORES stay
            # reproducible regardless: they are computed in code from the
            # extracted features, not judged by the model.
            effort = getattr(settings, "OPENAI_REASONING_EFFORT", "low")
            if effort:
                model_kwargs["reasoning_effort"] = effort
        else:
            llm_kwargs["temperature"] = settings.OPENAI_TEMPERATURE
            if settings.OPENAI_SEED is not None:
                model_kwargs["seed"] = settings.OPENAI_SEED

        self.llm = ChatOpenAI(**llm_kwargs)

    def rag_retrieve(
        self,
        queries: list[str],
        ticker: str,
        periods: list[tuple[str, int | str]] | None = None,
        quarter: str | None = None,
        fiscal_year: int | str | None = None,
        doc_types: list[str] | None = None,
        sections: list[str] | None = None,
        management_only: bool = False,
        top_k_per_query: int | None = None,
        final_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[tuple[Any, float]]:
        """
        Multi-query RAG retrieval. Deduplicates by chunk_id keeping
        highest relevance score, then (optionally) reranks with a cross-encoder.

        Pass `fiscal_year` whenever you pass `quarter`. Quarter alone is not a
        period: "Q1" matches Q1 of every year in the store, so a Q1-2026 query
        silently pulls Q1-2025 evidence too. For several periods at once, pass
        `periods=[(quarter, year), ...]`.

        Two-stage retrieval: the vector store proposes a WIDE candidate set
        (cheap, intent-blind), then a cross-encoder picks the few that actually
        answer the query (expensive, intent-aware). Widening only pays off
        because chunk_id is now persisted — while every chunk of a document
        shared one id, dedup collapsed each document to a single chunk and a
        bigger top-k changed nothing.
        """
        rerank_on = use_rerank and settings.RERANK_ENABLED
        per_query = top_k_per_query or (
            settings.RERANK_CANDIDATES_PER_QUERY if rerank_on else 8
        )

        seen: dict[str, tuple[Any, float]] = {}
        self.failed_queries = []

        for q in queries:
            try:
                results = self.vs.retrieve(
                    query=q, ticker=ticker,
                    n_results=per_query,
                    quarter=quarter,
                    fiscal_year=fiscal_year,
                    periods=periods,
                    doc_types=doc_types,
                    sections=sections,
                    management_only=management_only,
                )
                for chunk, score in results:
                    cid = chunk.chunk_id
                    if cid not in seen or score > seen[cid][1]:
                        seen[cid] = (chunk, score)
            except Exception as e:
                self.failed_queries.append(q)
                logger.warning(f"RAG query failed ('{q[:40]}'): {e}")

        # "No chunk matched" and "every lookup errored" both used to arrive here
        # as an empty list, and the agent scored the empty string either way —
        # producing a number with no evidence under it, which is the one output
        # this system must never emit. They are different conditions: the first
        # is a fact about the corpus, the second is a broken vector store.
        if queries and len(self.failed_queries) == len(queries):
            raise RuntimeError(
                f"All {len(queries)} retrieval queries failed for {ticker} "
                f"({quarter or 'all quarters'}); refusing to score on no evidence. "
                f"Last failure is logged above."
            )
        if self.failed_queries:
            logger.warning(
                f"[retrieve] {len(self.failed_queries)}/{len(queries)} queries failed "
                f"for {ticker} — evidence is a partial set."
            )

        ordered = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        if not rerank_on or not ordered:
            self.retrieval_mode = "vector_order" if ordered else "empty"
            return ordered

        top_n = final_k or settings.RERANK_TOP_N
        try:
            from retrieval.reranker import rerank as _rerank
            reranked = _rerank(queries, ordered, top_n)
            self.retrieval_mode = "reranked"
            logger.debug(
                f"[rerank] {len(ordered)} candidates → top {len(reranked)} "
                f"for {len(queries)} quer{'y' if len(queries)==1 else 'ies'}"
            )
            return reranked
        except Exception as e:
            # Degrading to vector order is a real change of evidence, not a
            # cosmetic one, and corpus_fingerprint() cannot detect it — so a
            # signal scored on the fallback would be cached as though it came
            # from the reranked path and never re-scored. Loud by default,
            # fatal when the caller has asked for reproducibility.
            if settings.RERANK_REQUIRED:
                raise RuntimeError(
                    f"Reranker unavailable ({e}) and RERANK_REQUIRED is set. "
                    f"Refusing to score on a silently different evidence set."
                ) from e
            self.retrieval_mode = "vector_order_degraded"
            logger.warning(
                f"[rerank] Unavailable ({e}); using vector order. Evidence for this "
                f"signal differs from a reranked run — treat scores as not comparable."
            )
            return ordered[:top_n]

    def format_evidence(
        self,
        chunks_and_scores: list[tuple[Any, float]],
        max_chunks: int = 12,
        include_metadata: bool = True,
    ) -> str:
        """Format retrieved chunks into a prompt evidence block."""
        lines = []
        for i, (chunk, score) in enumerate(chunks_and_scores[:max_chunks], 1):
            meta = ""
            if include_metadata:
                meta = (
                    f"[{chunk.doc_type.value} | {chunk.quarter} {chunk.fiscal_year} | "
                    f"Section: {chunk.section.value} | Speaker: {chunk.speaker or 'Unknown'} | "
                    f"Relevance: {score:.2f}]"
                )
            lines.append(f"CHUNK {i} {meta}\n{chunk.text.strip()}")
        return "\n\n---\n\n".join(lines)

    async def llm_reason(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Call LLM with automatic retry on rate limits / timeouts.
        Up to 3 attempts with exponential backoff (2s → 8s).
        """
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=16),
            retry=retry_if_exception_type(_RETRY_EXC),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _call():
            resp = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            # seed only guarantees a repeatable draw while the backend build is
            # unchanged; OpenAI signals that with system_fingerprint. Logging it
            # is what lets you tell "the model wandered" from "OpenAI shipped a
            # new build" when a re-scored quarter comes back different.
            fp = (getattr(resp, "response_metadata", {}) or {}).get("system_fingerprint")
            if fp:
                logger.debug(f"[llm] {self.model_name} system_fingerprint={fp}")
            # Per-call token accounting into the active run's cost log, if one is
            # open (run_cost_log context manager). Best-effort; never fatal.
            _clog = _run_cost_log.get()
            if _clog is not None:
                try:
                    _in, _out, _rea = _extract_usage(resp)
                    _clog.record(self.model_name, _in, _out, _rea,
                                 label=type(self).__name__)
                except Exception as e:
                    logger.debug(f"[cost] usage capture failed: {e}")
            raw = resp.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}. Attempting repair…")
                candidate = raw
                m = re.search(r'\{.*\}', candidate, re.DOTALL)
                if m:
                    candidate = m.group()
                # Common LLM JSON mistakes that json.loads() rejects outright:
                candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
                candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)                # trailing commas
                candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", candidate)   # stray control chars
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    logger.error(f"JSON repair did not fix the payload; raw response head: {raw[:300]!r}")
                    raise

        return await _call()
