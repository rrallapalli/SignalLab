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


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    dict.get(key, default) only falls back when the key is ABSENT — if the
    LLM returns the key with an explicit `null`, .get() still returns None
    and float(None) raises. Use this instead of float(data.get(...)).
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
        self.llm = ChatOpenAI(
            model=model or settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            # Forces the API itself to guarantee syntactically valid JSON output,
            # instead of relying on the model voluntarily following "return only
            # JSON" prompt instructions (which occasionally breaks on embedded
            # quotes/apostrophes in quoted evidence text).
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def rag_retrieve(
        self,
        queries: list[str],
        ticker: str,
        quarters: list[str] | None = None,
        quarter: str | None = None,
        doc_types: list[str] | None = None,
        sections: list[str] | None = None,
        management_only: bool = False,
        top_k_per_query: int = 8,
    ) -> list[tuple[Any, float]]:
        """
        Multi-query RAG retrieval. Deduplicates by chunk_id keeping
        highest relevance score.
        """
        seen: dict[str, tuple[Any, float]] = {}

        for q in queries:
            try:
                results = self.vs.retrieve(
                    query=q, ticker=ticker,
                    n_results=top_k_per_query,
                    quarter=quarter,
                    quarters=quarters,
                    doc_types=doc_types,
                    sections=sections,
                    management_only=management_only,
                )
                for chunk, score in results:
                    cid = chunk.chunk_id
                    if cid not in seen or score > seen[cid][1]:
                        seen[cid] = (chunk, score)
            except Exception as e:
                logger.warning(f"RAG query failed ('{q[:40]}'): {e}")

        return sorted(seen.values(), key=lambda x: x[1], reverse=True)

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
