"""config.py – Central configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


class Settings:
    # ── APIs ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY:   str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL:     str = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBED_MODEL:      str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    # temperature=0 is not determinism. It removes deliberate sampling, but the
    # API still batches requests and reduces floats in a non-fixed order, so the
    # same prompt can return a slightly different score run to run — enough to
    # flip a `tone` or `severity` label sitting on a threshold. `seed` asks the
    # API for a repeatable draw; it is best-effort, and only holds while
    # system_fingerprint (logged per call) is unchanged. Set to empty to omit.
    OPENAI_SEED: int | None = (
        int(os.getenv("OPENAI_SEED")) if os.getenv("OPENAI_SEED", "").strip() else 7
    )

    # ── NSE / BSE direct fetch (no API key required — public endpoints) ───────
    # How many days after a quarter-end to keep searching for results /
    # investor decks / transcripts (Indian companies typically disclose
    # within 30-45 days, but transcripts can lag by another 1-2 weeks).
    # NSE's cookie handshake is blocked from many non-Indian IPs. When it fails,
    # the client init burns three attempts plus backoff on EVERY run before
    # falling through to BSE, which is the reliable source anyway. Turn NSE off
    # to skip that dead time entirely.
    USE_NSE:                    bool = os.getenv("USE_NSE", "true").lower() == "true"
    NSE_BSE_RESULT_LAG_DAYS:    int = int(os.getenv("NSE_BSE_RESULT_LAG_DAYS", "90"))
    # Annual reports are filed much later (around the AGM, several months
    # after fiscal year-end) so they get a wider window when requested.
    NSE_BSE_ANNUAL_LAG_DAYS:    int = int(os.getenv("NSE_BSE_ANNUAL_LAG_DAYS", "200"))
    NSE_BSE_MAX_DOCS_PER_QUARTER: int = int(os.getenv("NSE_BSE_MAX_DOCS_PER_QUARTER", "15"))
    NSE_BSE_MAX_PAGES:          int = int(os.getenv("NSE_BSE_MAX_PAGES", "5"))

    # ── Document parsing (Docling) ───────────────────────────────────────────
    # Docling does layout-aware extraction and emits markdown with tables
    # intact. pypdf flattens tables into a text stream, which severs row labels
    # from their figures — the exact structure the guidance agent depends on.
    USE_DOCLING:        bool = os.getenv("USE_DOCLING", "true").lower() == "true"
    # OCR only runs when the fast path returns almost no text (a scanned
    # filing). It is slow, so it is never the default path — but without it
    # those documents are silently dropped by the <200-char guard in the fetcher.
    DOCLING_OCR_FALLBACK: bool = os.getenv("DOCLING_OCR_FALLBACK", "true").lower() == "true"
    PDF_MAX_PAGES:      int  = int(os.getenv("PDF_MAX_PAGES", "60"))
    # Backpressure on how many documents may be in the parse stage at once.
    # NOTE: the actual conversion is serialized by a lock in parser.py because
    # Docling's native backend is not thread-safe — running two conversions
    # concurrently corrupts the heap and kills the process outright. Raising
    # this will not buy parallel parsing; it only queues work earlier. Genuine
    # parallelism requires separate processes.
    PARSE_CONCURRENCY:  int  = int(os.getenv("PARSE_CONCURRENCY", "1"))
    # "auto" uses the GPU (MPS on Apple Silicon) wherever the model supports it
    # and CPU where it does not — usually the single biggest parsing speedup on
    # a Mac. Force "cpu" only if a model misbehaves on MPS.
    DOCLING_DEVICE:     str  = os.getenv("DOCLING_DEVICE", "auto")
    # Threads PER parse. Total load is PARSE_CONCURRENCY x this, so keep the
    # product at or below your core count or the parses fight each other.
    # Threads used INSIDE a single conversion (conversions themselves are
    # serialized). Roughly your core count is a sensible ceiling.
    DOCLING_NUM_THREADS: int = int(os.getenv("DOCLING_NUM_THREADS", "4"))
    # Parse only the first N pages of each PDF (via page_range). Results
    # releases and decks put the tables that matter up front, so this is usually
    # free accuracy-wise and a large time saving. 0 = no limit.
    DOCLING_PAGE_LIMIT: int  = int(os.getenv("DOCLING_PAGE_LIMIT", "25"))
    # Which document types are worth layout-parsing. Docling's only advantage is
    # table structure, so transcripts (pure prose) gain nothing from it while
    # costing the most time — they are the longest documents in the corpus.
    # Empty string = every type. Widen this if the guidance agent starts missing
    # numbers that live in a type not listed here.
    DOCLING_DOC_TYPES:  str  = os.getenv(
        "DOCLING_DOC_TYPES", "investor_presentation,press_release"
    )
    # Documents longer than this skip Docling entirely and take the fast pypdf
    # path. With DOCLING_PAGE_LIMIT doing the real work (Docling only parses the
    # first N pages), this is now just a guard against pathologically large
    # files — a 591-page annual report is slow merely to open. Raised from 80 so
    # that 90-170 page investor decks, which are dense with the tables the
    # guidance agent needs, still go through Docling. 0 disables the cap.
    DOCLING_MAX_PAGES:  int  = int(os.getenv("DOCLING_MAX_PAGES", "250"))
    # Per-document text cap carried into the vector store. Raised from the old
    # hard-coded 30_000: better extraction yields more *usable* text per file,
    # so a cap tuned for pypdf's mangled output now truncates real evidence.
    DOC_MAX_CHARS:      int  = int(os.getenv("DOC_MAX_CHARS", "60000"))

    # ── Retrieval reranking ──────────────────────────────────────────────────
    # Two-stage retrieval: vector search proposes a wide candidate set, a
    # cross-encoder reranks it down. Cheapest correction for a general-purpose
    # embedding model reading financial text, and reversible — unlike
    # re-embedding the corpus with a domain model.
    RERANK_ENABLED:     bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    RERANK_MODEL:       str  = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_CANDIDATES_PER_QUERY: int = int(os.getenv("RERANK_CANDIDATES_PER_QUERY", "25"))
    RERANK_TOP_N:       int  = int(os.getenv("RERANK_TOP_N", "12"))
    # Cross-encoder cost is queries x candidates; cap it so a many-query agent
    # cannot stall the pipeline on a laptop CPU.
    RERANK_MAX_PAIRS:   int  = int(os.getenv("RERANK_MAX_PAIRS", "500"))
    # RERANK_MODEL is downloaded from HuggingFace on first use. When that fails
    # — no network, cold cache, memory pressure — rerank_retrieve() falls back
    # to raw vector order, which is a materially different evidence set for the
    # same query. corpus_fingerprint() cannot see that, so the cache will call
    # the resulting signal "current" and never re-score it. Set this true to
    # make the failure loud instead of silent: better no signal than a signal
    # whose provenance you cannot reconstruct.
    RERANK_REQUIRED:    bool = os.getenv("RERANK_REQUIRED", "false").lower() == "true"

    # ── Storage ───────────────────────────────────────────────────────────────
    DATA_DIR:         Path = BASE_DIR / "data"
    CHROMA_DIR:       Path = BASE_DIR / "data" / "chroma"
    DUCKDB_PATH:      Path = BASE_DIR / "data" / "signals.duckdb"
    NSE_CACHE_DIR:    Path = BASE_DIR / "data" / "nse_cache"
    BSE_CACHE_DIR:    Path = BASE_DIR / "data" / "bse_cache"
    # Parsed-document cache. Parsing is the most expensive per-document step and
    # its output never changes, so it is keyed by URL + parser version and
    # consulted BEFORE the network call.
    PARSE_CACHE_DIR:  Path = BASE_DIR / "data" / "parse_cache"
    # Raw downloaded PDFs, kept so that changing parse settings re-parses
    # locally instead of re-downloading the whole corpus.
    PDF_CACHE_DIR:    Path = BASE_DIR / "data" / "pdf_cache"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE:       int  = 512
    CHUNK_OVERLAP:    int  = 64
    TOP_K_RETRIEVAL:  int  = 12   # chunks per RAG query

    # ── Scoring ───────────────────────────────────────────────────────────────
    CONFIDENCE_SCALE: int  = 10   # 0–10

    # ── Document types ────────────────────────────────────────────────────────
    DOC_TYPES = [
        "earnings_call",
        "annual_report",
        "investor_presentation",
        "press_release",
        "broker_note",
        "news_article",
        "management_commentary",
    ]

    # ── Section tags the chunker assigns ─────────────────────────────────────
    SECTIONS = [
        "prepared_remarks", "qa_session", "financial_results",
        "guidance", "risk_factors", "strategy", "market_overview",
    ]


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
settings.NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.BSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
