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
        """
        Embed and upsert chunks. Returns the TOTAL number of chunks this
        document contributes — not just the newly embedded ones.

        Chunks whose id is already in the collection are skipped: chunk_id is a
        hash of doc_id + offset + text, so an identical id means identical text
        and re-embedding it buys nothing but latency and OpenAI spend. On a
        re-run of an unchanged ticker this makes ingestion almost free.

        The return value deliberately counts skipped chunks too. The orchestrator
        compares it against MIN_CHUNKS_TO_SCORE to decide whether a quarter has
        enough evidence to score; returning only the new count would make every
        re-run look like a quarter with zero evidence and silently skip scoring.
        """
        if not chunks:
            return 0

        total = len(chunks)

        # Which of these are already embedded?
        existing: set[str] = set()
        try:
            found = self._col.get(ids=[c.chunk_id for c in chunks], include=[])
            existing = set(found.get("ids") or [])
        except Exception as e:
            logger.debug(f"Existence check failed ({e}); embedding all chunks.")

        chunks = [c for c in chunks if c.chunk_id not in existing]
        if not chunks:
            logger.debug(f"All {total} chunks already embedded — skipped.")
            return total
        if existing:
            logger.debug(f"Embedding {len(chunks)} new chunks ({len(existing)} already present).")

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
                # Persist the real chunk_id. It used to be absent, so retrieve()
                # rebuilt it as doc_id + "_r" — giving every chunk from one
                # document the SAME id. rag_retrieve() dedupes on that id, so all
                # but the top-scoring chunk per document were silently discarded:
                # ~12 documents meant ~12 chunks of evidence no matter how large
                # TOP_K_RETRIEVAL was set.
                "chunk_id":     c.chunk_id,
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
        return total

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
                chunk_id=meta.get("chunk_id") or (meta.get("doc_id", "") + "_r"),
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

    def prune_period(
        self, ticker: str, quarter: str, fiscal_year: int | str,
        keep_doc_ids: set[str],
    ) -> int:
        """
        Delete chunks for this ticker/period whose document is no longer part of
        the ingested corpus. Returns the number removed.

        Chroma and DuckDB drift apart without this. Ingestion adds chunks but
        nothing ever removes them, so when a document stops being ingested — it
        was dropped as a duplicate, the parser changed, the ranking cut it — its
        chunks stay behind and remain retrievable. Agents then cite evidence
        from documents the system no longer holds, which is exactly what
        validate_run's "quotes appear in the source documents" check catches:
        a real quote, from a real filing, that no ingested document contains.

        Refuses to run on an empty keep-set: a period whose fetch failed must
        not be interpreted as "this period has no documents, delete everything".
        """
        if not keep_doc_ids:
            logger.debug(f"prune_period({ticker} {quarter} {fiscal_year}): empty keep-set, skipping.")
            return 0

        try:
            found = self._col.get(
                where={"$and": [
                    {"ticker": ticker},
                    {"quarter": quarter},
                    {"fiscal_year": str(fiscal_year)},
                ]},
                include=["metadatas"],
            )
        except Exception as e:
            logger.warning(f"prune_period lookup failed for {ticker} {quarter} {fiscal_year}: {e}")
            return 0

        ids   = found.get("ids") or []
        metas = found.get("metadatas") or []
        stale = [
            cid for cid, meta in zip(ids, metas)
            if (meta or {}).get("doc_id") not in keep_doc_ids
        ]
        if not stale:
            return 0

        try:
            self._col.delete(ids=stale)
        except Exception as e:
            logger.warning(f"prune_period delete failed for {ticker} {quarter} {fiscal_year}: {e}")
            return 0

        orphan_docs = {
            (m or {}).get("doc_id") for c, m in zip(ids, metas) if c in set(stale)
        }
        logger.info(
            f"[prune] {ticker} {quarter} {fiscal_year}: removed {len(stale)} orphaned "
            f"chunk(s) from {len(orphan_docs)} document(s) no longer ingested."
        )
        return len(stale)

    def count(self, ticker: str) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0

    def clear_ticker(self, ticker: str) -> None:
        self._col.delete(where={"ticker": ticker})
        logger.info(f"Cleared all chunks for {ticker}")
