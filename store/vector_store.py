"""
store/vector_store.py
ChromaDB wrapper with metadata-aware retrieval.
Supports filtering by: ticker, quarter, doc_type, section, is_management.
"""

from __future__ import annotations
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from openai import OpenAI

from config import settings
from models import Citation, DocumentChunk


class VectorStore:
    """ChromaDB-backed vector store with rich metadata filtering."""

    COLLECTION = "signal_agent_chunks"

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._oai = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"VectorStore ready. Collection size: {self._col.count()} chunks")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._oai.embeddings.create(model=settings.EMBED_MODEL, input=texts)
        return [e.embedding for e in resp.data]

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert chunks. Returns count added."""
        if not chunks:
            return 0

        texts     = [c.text for c in chunks]
        ids       = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "ticker":       c.ticker,
                "company":      c.company,
                "doc_type":     c.doc_type.value,
                "section":      c.section.value,
                "quarter":      c.quarter,
                "fiscal_year":  str(c.fiscal_year),
                "speaker":      c.speaker,
                "is_management": str(c.is_management).lower(),
                "source_url":   c.source_url,
                "title":        c.title[:200],
                "doc_id":       c.doc_id,
            }
            for c in chunks
        ]

        # Batch embed (OpenAI max 2048 per call)
        batch_size = 256
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            all_embeddings.extend(self._embed(texts[i:i+batch_size]))

        self._col.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(chunks)} chunks")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        ticker: str,
        n_results: int | None = None,
        quarter: str | None = None,
        fiscal_year: int | str | None = None,
        doc_types: list[str] | None = None,
        sections: list[str] | None = None,
        management_only: bool = False,
        periods: list[tuple[str, int | str]] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Semantic retrieval with optional metadata filters.
        Returns list of (chunk, distance_score) sorted by relevance.

        NOTE ON fiscal_year — filtering on `quarter` ALONE is a correctness bug:
        quarter and fiscal_year are stored as separate metadata fields ("Q1" and
        "2026"), so {"quarter": "Q1"} matches Q1 of EVERY year in the corpus.
        A Q1-2026 query then returns Q1-2025 chunks, a YoY comparison ends up
        scoring the same pooled evidence twice, and the delta collapses to 0.0.
        Always pass fiscal_year alongside quarter for single-period retrieval.

        For MULTI-period retrieval use `periods` — a list of (quarter, year)
        pairs, e.g. [("Q1", 2026), ("Q4", 2025)]. The previous `quarters` list
        took bare quarter labels and had the same year-blind flaw at scale:
        {"quarter": {"$in": ["Q1","Q4"]}} matches Q1/Q4 of EVERY year. A period
        is a pair; there is deliberately no way to express it as one string.
        """
        n = n_results or settings.TOP_K_RETRIEVAL
        embedding = self._embed([query])[0]

        where: dict[str, Any] = {"ticker": ticker}
        filters: list[dict] = [{"ticker": ticker}]

        if quarter and not periods:
            filters.append({"quarter": quarter})
        if fiscal_year is not None and not periods:
            # Stored as a string at ingestion — compare like for like.
            filters.append({"fiscal_year": str(fiscal_year)})

        if periods:
            # Dedupe but keep order stable.
            uniq: list[tuple[str, str]] = []
            for q_, y_ in periods:
                pair = (q_, str(y_))
                if pair not in uniq:
                    uniq.append(pair)

            clauses = [
                {"$and": [{"quarter": q_}, {"fiscal_year": y_}]} for q_, y_ in uniq
            ]
            if len(clauses) == 1:
                # Chroma rejects $or/$and with fewer than two expressions, and a
                # rejected filter falls back to a ticker-only query — i.e. the
                # whole corpus, unfiltered. Flatten the single-period case.
                filters.extend([{"quarter": uniq[0][0]}, {"fiscal_year": uniq[0][1]}])
            elif clauses:
                filters.append({"$or": clauses})
        if doc_types:
            filters.append({"doc_type": {"$in": doc_types}})
        if sections:
            filters.append({"section": {"$in": sections}})
        if management_only:
            filters.append({"is_management": "true"})

        where_clause = {"$and": filters} if len(filters) > 1 else filters[0]

        try:
            results = self._col.query(
                query_embeddings=[embedding],
                n_results=min(n, self._col.count()),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # Falling back to a ticker-only filter silently widens the search
            # across every quarter and year, which is how period-mixing slips
            # through unnoticed. Log loudly enough to be greppable.
            logger.warning(
                f"Query failed with filters ({where_clause}), retrying with ticker only "
                f"— results will NOT be period-filtered: {e}"
            )
            results = self._col.query(
                query_embeddings=[embedding],
                n_results=min(n, self._col.count()),
                where={"ticker": ticker},
                include=["documents", "metadatas", "distances"],
            )

        chunks_and_scores: list[tuple[DocumentChunk, float]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            from models import DocumentSection, DocumentType
            chunk = DocumentChunk(
                chunk_id=meta.get("doc_id", "") + "_r",
                doc_id=meta.get("doc_id", ""),
                ticker=meta.get("ticker",""),
                company=meta.get("company",""),
                doc_type=DocumentType(meta.get("doc_type","news_article")),
                section=DocumentSection(meta.get("section","unknown")),
                quarter=meta.get("quarter",""),
                fiscal_year=int(meta.get("fiscal_year","0") or 0),
                speaker=meta.get("speaker",""),
                is_management=meta.get("is_management","false") == "true",
                text=text,
                source_url=meta.get("source_url",""),
                title=meta.get("title",""),
            )
            relevance = 1.0 - float(dist)   # cosine: distance → similarity
            chunks_and_scores.append((chunk, relevance))

        return chunks_and_scores

    def as_citations(
        self, chunks_and_scores: list[tuple[DocumentChunk, float]]
    ) -> list[Citation]:
        citations = []
        for chunk, score in chunks_and_scores:
            quote = chunk.text[:120].strip()
            if len(chunk.text) > 120:
                quote = quote.rsplit(" ", 1)[0] + "…"
            citations.append(Citation(
                chunk_id=chunk.chunk_id,
                doc_type=chunk.doc_type.value,
                # Full period, not a bare "Q1". A citation labelled only "Q1" is
                # ambiguous across years — which is precisely why evidence being
                # pooled from Q1-2025 into a Q1-2026 signal went unnoticed. The
                # displayed provenance has to be specific enough to falsify.
                quarter=f"{chunk.quarter} {chunk.fiscal_year}".strip(),
                source_url=chunk.source_url,
                speaker=chunk.speaker,
                quote=quote,
                relevance=round(score, 3),
            ))
        return citations

    def count(self, ticker: str) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0

    def clear_ticker(self, ticker: str) -> None:
        self._col.delete(where={"ticker": ticker})
        logger.info(f"Cleared all chunks for {ticker}")
