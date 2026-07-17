"""
store/signal_store.py
DuckDB-backed time-series store for structured signals and source documents.
"""

from __future__ import annotations
import hashlib
import re
import json
from datetime import datetime
from typing import Any

import duckdb
from loguru import logger

from config import settings
from models import ConfidenceSignal, GuidanceSignal, NarrativeSignal, RiskSignal


DDL = """
CREATE TABLE IF NOT EXISTS confidence_signals (
    id                  VARCHAR PRIMARY KEY,
    ticker              VARCHAR NOT NULL,
    company             VARCHAR,
    quarter             VARCHAR NOT NULL,
    fiscal_year         INTEGER,
    generated_at        TIMESTAMP,
    score               DOUBLE,
    previous_score      DOUBLE,
    change              DOUBLE,
    confidence_level    DOUBLE,
    uncertainty_level   DOUBLE,
    defensiveness       DOUBLE,
    specificity         DOUBLE,
    consistency         DOUBLE,
    forward_strength    DOUBLE,
    tone                VARCHAR,
    drivers             JSON,
    summary             VARCHAR,
    citations           JSON
);

CREATE TABLE IF NOT EXISTS narrative_signals (
    id              VARCHAR PRIMARY KEY,
    ticker          VARCHAR NOT NULL,
    company         VARCHAR,
    quarter         VARCHAR NOT NULL,
    fiscal_year     INTEGER,
    generated_at    TIMESTAMP,
    themes          JSON,
    accelerating    JSON,
    emerging        JSON,
    fading          JSON,
    newly_risky     JSON,
    overall_shift   VARCHAR,
    shift_summary   VARCHAR,
    citations       JSON
);

CREATE TABLE IF NOT EXISTS guidance_signals (
    id               VARCHAR PRIMARY KEY,
    ticker           VARCHAR NOT NULL,
    company          VARCHAR,
    quarter          VARCHAR NOT NULL,
    fiscal_year      INTEGER,
    generated_at     TIMESTAMP,
    score            DOUBLE,
    guidance_items   JSON,
    periods_tracked  INTEGER,
    beats            INTEGER,
    misses           INTEGER,
    in_line          INTEGER,
    beat_rate        DOUBLE,
    serial_miss_risk BOOLEAN,
    recent_pattern   JSON,
    summary          VARCHAR,
    citations        JSON
);

CREATE TABLE IF NOT EXISTS risk_signals (
    id                     VARCHAR PRIMARY KEY,
    ticker                 VARCHAR NOT NULL,
    company                VARCHAR,
    quarter                VARCHAR NOT NULL,
    fiscal_year            INTEGER,
    generated_at           TIMESTAMP,
    risks                  JSON,
    new_risks              JSON,
    escalating             JSON,
    diminishing            JSON,
    overall_risk_direction VARCHAR,
    summary                VARCHAR,
    citations              JSON
);

CREATE TABLE IF NOT EXISTS ingested_documents (
    doc_id       VARCHAR PRIMARY KEY,
    ticker       VARCHAR NOT NULL,
    company      VARCHAR,
    doc_type     VARCHAR,
    quarter      VARCHAR,
    fiscal_year  INTEGER,
    source_url   VARCHAR,
    title        VARCHAR,
    chunk_count  INTEGER,
    ingested_at  TIMESTAMP,
    raw_text     VARCHAR
);

CREATE TABLE IF NOT EXISTS signal_runs (
    id            VARCHAR PRIMARY KEY,   -- ticker::quarter::fiscal_year
    ticker        VARCHAR NOT NULL,
    quarter       VARCHAR NOT NULL,
    fiscal_year   INTEGER,
    fingerprint   VARCHAR,               -- corpus + model + agent version
    generated_at  TIMESTAMP
);
"""

# Migration: add raw_text to existing DBs that predate this column
MIGRATIONS = [
    "ALTER TABLE ingested_documents ADD COLUMN IF NOT EXISTS raw_text VARCHAR;",
]


def _j(val: Any) -> str:
    try:
        return json.dumps(val)
    except Exception:
        return "[]"


class SignalStore:

    def __init__(self):
        self.db_path = str(settings.DUCKDB_PATH)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute(DDL)
        for migration in MIGRATIONS:
            try:
                self._conn.execute(migration)
            except Exception:
                pass   # column already exists
        logger.info(f"SignalStore ready at {self.db_path}")

    def _sig_id(self, prefix: str, ticker: str, quarter: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{prefix}::{ticker}::{quarter}::{ts}"

    # ── Save methods ──────────────────────────────────────────────────────────

    def save_confidence(self, sig: ConfidenceSignal) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO confidence_signals
                (id, ticker, company, quarter, fiscal_year, generated_at,
                 score, previous_score, change,
                 confidence_level, uncertainty_level, defensiveness,
                 specificity, consistency, forward_strength,
                 tone, drivers, summary, citations)
            VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?)
        """, [
            self._sig_id("conf", sig.ticker, sig.quarter),
            sig.ticker, sig.company, sig.quarter, sig.fiscal_year, sig.generated_at,
            sig.score, sig.previous_score, sig.change,
            sig.confidence_level, sig.uncertainty_level, sig.defensiveness,
            sig.specificity, sig.consistency, sig.forward_strength,
            sig.tone,
            _j(sig.drivers),
            sig.summary,
            _j([c.model_dump() for c in sig.citations]),
        ])
        logger.debug(f"Saved confidence signal {sig.ticker} {sig.quarter}")

    def save_narrative(self, sig: NarrativeSignal) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO narrative_signals
                (id, ticker, company, quarter, fiscal_year, generated_at,
                 themes, accelerating, emerging, fading, newly_risky,
                 overall_shift, shift_summary, citations)
            VALUES (?,?,?,?,?,?, ?,?,?,?,?, ?,?,?)
        """, [
            self._sig_id("narr", sig.ticker, sig.quarter),
            sig.ticker, sig.company, sig.quarter, sig.fiscal_year, sig.generated_at,
            _j([t.model_dump() for t in sig.themes]),
            _j(sig.accelerating), _j(sig.emerging),
            _j(sig.fading), _j(sig.newly_risky),
            sig.overall_shift, sig.shift_summary,
            _j([c.model_dump() for c in sig.citations]),
        ])

    def save_guidance(self, sig: GuidanceSignal) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO guidance_signals
                (id, ticker, company, quarter, fiscal_year, generated_at,
                 score, guidance_items, periods_tracked,
                 beats, misses, in_line, beat_rate, serial_miss_risk,
                 recent_pattern, summary, citations)
            VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?,?,?, ?,?,?)
        """, [
            self._sig_id("guid", sig.ticker, sig.quarter),
            sig.ticker, sig.company, sig.quarter, sig.fiscal_year, sig.generated_at,
            sig.score,
            _j([g.model_dump() for g in sig.guidance_items]),
            sig.periods_tracked,
            sig.beats, sig.misses, sig.in_line, sig.beat_rate, sig.serial_miss_risk,
            _j(sig.recent_pattern),
            sig.summary,
            _j([c.model_dump() for c in sig.citations]),
        ])

    def save_risk(self, sig: RiskSignal) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO risk_signals
                (id, ticker, company, quarter, fiscal_year, generated_at,
                 risks, new_risks, escalating, diminishing,
                 overall_risk_direction, summary, citations)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?)
        """, [
            self._sig_id("risk", sig.ticker, sig.quarter),
            sig.ticker, sig.company, sig.quarter, sig.fiscal_year, sig.generated_at,
            _j([r.model_dump() for r in sig.risks]),
            _j(sig.new_risks), _j(sig.escalating), _j(sig.diminishing),
            sig.overall_risk_direction,
            sig.summary,
            _j([c.model_dump() for c in sig.citations]),
        ])

    def log_document(
        self, ticker: str, company: str, doc_type: str,
        quarter: str, fiscal_year: int, source_url: str,
        title: str, chunk_count: int, doc_id: str,
        raw_text: str = "",
    ) -> None:
        """Persist source document metadata and raw text for citation traceability."""
        self._conn.execute("""
            INSERT OR REPLACE INTO ingested_documents
                (doc_id, ticker, company, doc_type, quarter, fiscal_year,
                 source_url, title, chunk_count, ingested_at, raw_text)
            VALUES (?,?,?,?,?,?, ?,?,?,?,?)
        """, [
            doc_id, ticker, company, doc_type, quarter, fiscal_year,
            source_url, title, chunk_count, datetime.utcnow(),
            raw_text[:50000] if raw_text else "",   # cap at 50k chars
        ])

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_confidence_history(self, ticker: str, limit: int = 8) -> list[dict]:
        rows = self._conn.execute("""
            SELECT quarter, fiscal_year, score, previous_score, change,
                   confidence_level, uncertainty_level, defensiveness,
                   specificity, consistency, forward_strength,
                   tone, drivers, summary, citations, generated_at
            FROM confidence_signals
            WHERE ticker = ?
            ORDER BY fiscal_year DESC, quarter DESC
            LIMIT ?
        """, [ticker, limit]).fetchall()
        cols = [
            "quarter", "fiscal_year", "score", "previous_score", "change",
            "confidence_level", "uncertainty_level", "defensiveness",
            "specificity", "consistency", "forward_strength",
            "tone", "drivers", "summary", "citations", "generated_at",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["drivers"]   = json.loads(d["drivers"]   or "[]")
            d["citations"] = json.loads(d["citations"] or "[]")
            result.append(d)
        return result

    def get_narrative_history(self, ticker: str, limit: int = 8) -> list[dict]:
        rows = self._conn.execute("""
            SELECT quarter, fiscal_year, accelerating, emerging, fading,
                   newly_risky, overall_shift, shift_summary, themes,
                   citations, generated_at
            FROM narrative_signals WHERE ticker = ?
            ORDER BY fiscal_year DESC, quarter DESC LIMIT ?
        """, [ticker, limit]).fetchall()
        cols = [
            "quarter", "fiscal_year", "accelerating", "emerging", "fading",
            "newly_risky", "overall_shift", "shift_summary", "themes",
            "citations", "generated_at",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            for f in ["accelerating", "emerging", "fading", "newly_risky", "themes"]:
                d[f] = json.loads(d[f] or "[]")
            d["citations"] = json.loads(d["citations"] or "[]")
            result.append(d)
        return result

    def get_guidance_history(self, ticker: str, limit: int = 8) -> list[dict]:
        rows = self._conn.execute("""
            SELECT quarter, fiscal_year, score, beats, misses, in_line,
                   beat_rate, serial_miss_risk, recent_pattern, summary,
                   citations, generated_at
            FROM guidance_signals WHERE ticker = ?
            ORDER BY fiscal_year DESC, quarter DESC LIMIT ?
        """, [ticker, limit]).fetchall()
        cols = [
            "quarter", "fiscal_year", "score", "beats", "misses", "in_line",
            "beat_rate", "serial_miss_risk", "recent_pattern", "summary",
            "citations", "generated_at",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["recent_pattern"] = json.loads(d["recent_pattern"] or "[]")
            d["citations"]      = json.loads(d["citations"]      or "[]")
            result.append(d)
        return result

    def get_risk_history(self, ticker: str, limit: int = 8) -> list[dict]:
        rows = self._conn.execute("""
            SELECT quarter, fiscal_year, risks, new_risks, escalating,
                   diminishing, overall_risk_direction, summary,
                   citations, generated_at
            FROM risk_signals WHERE ticker = ?
            ORDER BY fiscal_year DESC, quarter DESC LIMIT ?
        """, [ticker, limit]).fetchall()
        cols = [
            "quarter", "fiscal_year", "risks", "new_risks", "escalating",
            "diminishing", "overall_risk_direction", "summary",
            "citations", "generated_at",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            for f in ["risks", "new_risks", "escalating", "diminishing"]:
                d[f] = json.loads(d[f] or "[]")
            d["citations"] = json.loads(d["citations"] or "[]")
            result.append(d)
        return result

    def get_source_documents(
        self, ticker: str, quarter: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Return ingested source documents for a ticker, optionally filtered by quarter."""
        where = "WHERE ticker = ?"
        params: list = [ticker]
        if quarter:
            where += " AND quarter = ?"
            params.append(quarter)
        rows = self._conn.execute(f"""
            SELECT doc_id, doc_type, quarter, fiscal_year, title,
                   source_url, chunk_count, ingested_at
            FROM ingested_documents
            {where}
            ORDER BY fiscal_year DESC, quarter DESC, ingested_at DESC
            LIMIT {limit}
        """, params).fetchall()
        cols = ["doc_id", "doc_type", "quarter", "fiscal_year",
                "title", "source_url", "chunk_count", "ingested_at"]
        return [dict(zip(cols, row)) for row in rows]

    def get_document_text(self, doc_id: str) -> str | None:
        """Retrieve stored raw text for a specific document."""
        rows = self._conn.execute(
            "SELECT raw_text FROM ingested_documents WHERE doc_id = ?", [doc_id]
        ).fetchall()
        return rows[0][0] if rows else None

    # ── Incremental scoring ───────────────────────────────────────────────────

    @staticmethod
    def _json_cols(table: str) -> list[str]:
        """
        JSON columns for a table, read from SCHEMA itself.

        Hand-listing these rots the moment a column is added: a missed column
        stays a raw string, the model rejects it, and the period silently never
        qualifies for reuse. (It did — 'risks' was missed on the first attempt.)
        Deriving from the schema means the list cannot drift from the table.
        """
        m = re.search(
            rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\);", DDL, re.S
        )
        if not m:
            return []
        return [
            line.split()[0]
            for line in m.group(1).strip().splitlines()
            if len(line.split()) >= 2 and line.split()[1].upper().startswith("JSON")
        ]

    def _row_for_period(self, table: str, ticker: str, quarter: str,
                        fiscal_year: int) -> dict | None:
        cur = self._conn.execute(
            f"SELECT * FROM {table} WHERE ticker = ? AND quarter = ? AND fiscal_year = ?",
            [ticker, quarter, fiscal_year],
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, rows[0]))
        for c in self._json_cols(table):
            if isinstance(d.get(c), str):
                try:
                    d[c] = json.loads(d[c] or "null")
                except json.JSONDecodeError:
                    d[c] = None
        return d

    def get_period_signals(self, ticker: str, quarter: str, fiscal_year: int) -> dict:
        """
        Rehydrate the four stored signals for one period as model objects.

        Used to reuse a period whose corpus hasn't changed rather than paying to
        re-score it. Returns {"confidence": ConfidenceSignal|None, ...} — a kind
        is None if its row is missing or no longer parses into the current model
        (which is itself a signal that SIGNAL_VERSION should have been bumped).
        """
        from models import ConfidenceSignal, GuidanceSignal, NarrativeSignal, RiskSignal

        spec = [
            ("confidence", "confidence_signals", ConfidenceSignal),
            ("narrative",  "narrative_signals",  NarrativeSignal),
            ("guidance",   "guidance_signals",   GuidanceSignal),
            ("risk",       "risk_signals",       RiskSignal),
        ]
        out: dict = {}
        for key, table, cls in spec:
            row = self._row_for_period(table, ticker, quarter, fiscal_year)
            if not row:
                out[key] = None
                continue
            try:
                out[key] = cls(**row)
            except Exception as e:   # noqa: BLE001 — a shape change, not a crash
                logger.warning(
                    f"[store] Stored {key} for {ticker} {quarter} {fiscal_year} no longer "
                    f"fits the current model ({e}). Treating as absent so it re-scores."
                )
                out[key] = None
        return out


    def corpus_fingerprint(
        self, ticker: str, quarter: str, fiscal_year: int,
        model: str, signal_version: str,
    ) -> str:
        """
        A stable hash of everything that determines a period's signal.

        Deliberately NOT a timestamp: log_document() stamps ingested_at with
        utcnow() on every run, so "any document newer than the signal" is always
        true after a re-run and would never skip anything.

        Includes more than the documents, because a stored signal is stale if
        ANY input to it changed:

          · doc_ids + chunk_counts — the evidence itself
          · model                  — gpt-4o and gpt-4o-mini do not agree
          · signal_version         — the agent logic

        The last one is not optional. Q1 2025's filings have not changed since
        2025, but the signal stored against them was built with pooled
        cross-year evidence and an invented prior score. A fingerprint of the
        documents alone would call that "unchanged" and skip it forever, quietly
        preserving a wrong answer — which is the failure mode this whole codebase
        keeps producing. Bump SIGNAL_VERSION and every stored signal re-scores.
        """
        rows = self._conn.execute("""
            SELECT doc_id, chunk_count FROM ingested_documents
            WHERE ticker = ? AND quarter = ? AND fiscal_year = ?
            ORDER BY doc_id
        """, [ticker, quarter, fiscal_year]).fetchall()

        payload = "|".join(f"{d}:{c}" for d, c in rows)
        payload += f"||model={model}||agents={signal_version}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def is_signal_current(
        self, ticker: str, quarter: str, fiscal_year: int, fingerprint: str
    ) -> bool:
        """True if this period was already scored under identical conditions."""
        rows = self._conn.execute(
            "SELECT fingerprint FROM signal_runs WHERE id = ?",
            [f"{ticker}::{quarter}::{fiscal_year}"],
        ).fetchall()
        return bool(rows) and rows[0][0] == fingerprint

    def mark_scored(
        self, ticker: str, quarter: str, fiscal_year: int, fingerprint: str
    ) -> None:
        """Record that this period was scored under these exact conditions."""
        self._conn.execute("""
            INSERT OR REPLACE INTO signal_runs
                (id, ticker, quarter, fiscal_year, fingerprint, generated_at)
            VALUES (?,?,?,?,?,?)
        """, [f"{ticker}::{quarter}::{fiscal_year}", ticker, quarter,
              fiscal_year, fingerprint, datetime.utcnow()])

    # ── YTD helpers ───────────────────────────────────────────────────────────

    def get_ytd_guidance(self, ticker: str, year: int) -> dict:
        rows = self._conn.execute("""
            SELECT beats, misses, in_line, quarter, guidance_items
            FROM guidance_signals
            WHERE ticker = ? AND fiscal_year = ?
            ORDER BY quarter
        """, [ticker, year]).fetchall()
        if not rows: return {}

        total_beats = total_misses = total_in_line = 0
        quarters_covered, all_items = [], []
        for beats, misses, in_line, quarter, items_json in rows:
            total_beats   += int(beats  or 0)
            total_misses  += int(misses or 0)
            total_in_line += int(in_line or 0)
            quarters_covered.append(quarter)
            try: all_items.extend(json.loads(items_json or "[]"))
            except Exception: pass

        total_tracked = total_beats + total_misses + total_in_line
        # None, not 0.0. Nothing tracked is an ABSENCE of measurement; 0.0 reads
        # as "beat none of them", which the dashboard then paints red — a
        # damning verdict on a company whose guidance was simply never found.
        ytd_rate = (total_beats / total_tracked) if total_tracked > 0 else None

        from collections import Counter
        miss_counts = Counter(
            item.get("metric","") for item in all_items if item.get("outcome","") == "miss"
        )
        # Same rule as agents.guidance_agent._serial_miss_metrics — a metric
        # missed 2+ times. Keep these two in step; they are both rendered.
        serial_misses = [m for m, c in miss_counts.items() if c >= 2]
        return {
            "year": year, "quarters_covered": quarters_covered,
            "total_beats": total_beats, "total_misses": total_misses,
            "total_in_line": total_in_line, "total_tracked": total_tracked,
            "ytd_beat_rate": (round(ytd_rate, 3) if ytd_rate is not None else None),
            "serial_misses": serial_misses,
        }

    def get_ytd_risks(self, ticker: str, year: int) -> dict:
        rows = self._conn.execute("""
            SELECT quarter, new_risks, escalating, diminishing, risks
            FROM risk_signals
            WHERE ticker = ? AND fiscal_year = ?
            ORDER BY quarter
        """, [ticker, year]).fetchall()
        if not rows: return {}

        all_new: list[str] = []
        all_escalating: list[str] = []
        all_diminishing: list[str] = []
        quarters_covered, severity_map = [], {}

        for quarter, new_j, esc_j, dim_j, risks_j in rows:
            quarters_covered.append(quarter)
            try:
                for r in json.loads(new_j or "[]"):
                    if r not in all_new: all_new.append(r)
                for r in json.loads(esc_j or "[]"):
                    if r not in all_escalating: all_escalating.append(r)
                for r in json.loads(dim_j or "[]"):
                    if r not in all_diminishing: all_diminishing.append(r)
                for item in json.loads(risks_j or "[]"):
                    name, sev = item.get("risk",""), item.get("severity","low")
                    sev_order = {"critical":4,"high":3,"medium":2,"low":1}
                    if name and sev_order.get(sev,0) > sev_order.get(severity_map.get(name,"low"),0):
                        severity_map[name] = sev
            except Exception: pass

        resolved = set(all_diminishing)
        return {
            "year": year, "quarters_covered": quarters_covered,
            "new_risks_ytd": all_new,
            "new_risks_active": [r for r in all_new if r not in resolved],
            "escalating_ytd": list(set(all_escalating)),
            "diminishing_ytd": all_diminishing,
            "total_new": len(all_new),
            "total_active": len([r for r in all_new if r not in resolved]),
            "severity_map": severity_map,
        }

    def get_all_tickers(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT ticker FROM confidence_signals ORDER BY ticker"
        ).fetchall()
        return [r[0] for r in rows]
