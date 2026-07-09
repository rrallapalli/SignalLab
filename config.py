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

    # ── NSE / BSE direct fetch (no API key required — public endpoints) ───────
    # How many days after a quarter-end to keep searching for results /
    # investor decks / transcripts (Indian companies typically disclose
    # within 30-45 days, but transcripts can lag by another 1-2 weeks).
    NSE_BSE_RESULT_LAG_DAYS:    int = int(os.getenv("NSE_BSE_RESULT_LAG_DAYS", "90"))
    # Annual reports are filed much later (around the AGM, several months
    # after fiscal year-end) so they get a wider window when requested.
    NSE_BSE_ANNUAL_LAG_DAYS:    int = int(os.getenv("NSE_BSE_ANNUAL_LAG_DAYS", "200"))
    NSE_BSE_MAX_DOCS_PER_QUARTER: int = int(os.getenv("NSE_BSE_MAX_DOCS_PER_QUARTER", "15"))
    NSE_BSE_MAX_PAGES:          int = int(os.getenv("NSE_BSE_MAX_PAGES", "5"))

    # ── Storage ───────────────────────────────────────────────────────────────
    DATA_DIR:         Path = BASE_DIR / "data"
    CHROMA_DIR:       Path = BASE_DIR / "data" / "chroma"
    DUCKDB_PATH:      Path = BASE_DIR / "data" / "signals.duckdb"
    NSE_CACHE_DIR:    Path = BASE_DIR / "data" / "nse_cache"
    BSE_CACHE_DIR:    Path = BASE_DIR / "data" / "bse_cache"

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
