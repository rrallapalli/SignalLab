"""
retrieval/reranker.py
Cross-encoder reranking over the vector store's top-k candidates.

WHY
---
The embedding step is a bi-encoder: query and chunk are embedded separately and
compared by cosine distance, so the model never sees them together. That is fast
enough to scan a whole corpus but blunt about intent — "revenue guidance range
for FY27" and "revenue was 12,345" look similar in vector space even though only
one answers the question.

A cross-encoder reads (query, chunk) as a pair and scores the actual match. It
is far too slow to run over a whole corpus, which is exactly why it goes second:
the vector store proposes a wide candidate set, the cross-encoder picks the few
that genuinely answer the query.

This is also the cheapest available correction for a general-purpose embedding
model working on financial text. Re-embedding the corpus with a finance-domain
model is a one-way migration; reranking is a flag you can turn off.

Degrades to a no-op if sentence-transformers isn't installed or the model can't
load — retrieval keeps working, just unranked.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from loguru import logger

from config import settings

_model: Optional[Any] = None
_load_failed = False


def _get_model():
    """Lazy-load the cross-encoder once. Never raises; returns None on failure."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"[rerank] Loading cross-encoder {settings.RERANK_MODEL} (first use only)…")
        _model = CrossEncoder(settings.RERANK_MODEL, max_length=512)
        logger.success("[rerank] Cross-encoder ready.")
    except ImportError:
        logger.warning(
            "[rerank] sentence-transformers not installed — reranking disabled. "
            "Install with: pip install sentence-transformers"
        )
        _load_failed = True
    except Exception as e:
        logger.warning(f"[rerank] Could not load {settings.RERANK_MODEL} ({e}) — reranking disabled.")
        _load_failed = True
    return _model


def available() -> bool:
    return settings.RERANK_ENABLED and _get_model() is not None


def _sigmoid(x: float) -> float:
    # Cross-encoders emit unbounded logits. Citations render `relevance` as a
    # 0–1 figure in the dashboard, so squash into the same range rather than
    # showing the user a raw logit where they used to see a similarity.
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank(
    queries: Sequence[str],
    candidates: list[tuple[Any, float]],
    top_n: int,
) -> list[tuple[Any, float]]:
    """
    Re-score (chunk, score) candidates against the agent's queries and return
    the best `top_n`.

    Scored against EVERY query, keeping each chunk's best score — mirroring how
    rag_retrieve already treats multi-query retrieval. Agents ask several
    differently-angled questions and a chunk that decisively answers one of them
    should rank on that, not be diluted by the ones it doesn't address.

    Returns candidates untouched (truncated to top_n) if reranking is
    unavailable, so callers never need to branch.
    """
    if not candidates:
        return []
    if not settings.RERANK_ENABLED:
        return candidates[:top_n]

    model = _get_model()
    if model is None:
        return candidates[:top_n]

    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return candidates[:top_n]

    # Guard the pair count: cost is len(queries) × len(candidates).
    max_pairs = max(1, settings.RERANK_MAX_PAIRS)
    pool = candidates
    if len(queries) * len(pool) > max_pairs:
        keep = max(1, max_pairs // len(queries))
        pool = pool[:keep]
        logger.debug(f"[rerank] Trimmed candidate pool to {keep} to stay under {max_pairs} pairs.")

    try:
        pairs = [(q, chunk.text) for q in queries for chunk, _ in pool]
        raw = model.predict(pairs)
    except Exception as e:
        logger.warning(f"[rerank] Scoring failed ({e}) — falling back to vector order.")
        return candidates[:top_n]

    n = len(pool)
    best: list[float] = [float("-inf")] * n
    for qi in range(len(queries)):
        offset = qi * n
        for ci in range(n):
            v = float(raw[offset + ci])
            if v > best[ci]:
                best[ci] = v

    reranked = [(pool[i][0], _sigmoid(best[i])) for i in range(n)]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_n]
