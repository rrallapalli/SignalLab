"""agents/base.py – Shared agent infrastructure with retry and evidence formatting."""

from __future__ import annotations
import json, re, asyncio
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

        model_kwargs: dict[str, Any] = {
            # Forces the API itself to guarantee syntactically valid JSON output,
            # instead of relying on the model voluntarily following "return only
            # JSON" prompt instructions (which occasionally breaks on embedded
            # quotes/apostrophes in quoted evidence text).
            "response_format": {"type": "json_object"},
        }
        if settings.OPENAI_SEED is not None:
            model_kwargs["seed"] = settings.OPENAI_SEED

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=settings.OPENAI_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            model_kwargs=model_kwargs,
        )

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
