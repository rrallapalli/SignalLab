"""
models.py – All Pydantic models.

Two families:
  1. Document models  – represent ingested source material
  2. Signal models    – structured output from signal agents
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
# Document models
# ═══════════════════════════════════════════════════════

class DocumentType(str, Enum):
    EARNINGS_CALL          = "earnings_call"
    ANNUAL_REPORT          = "annual_report"
    INVESTOR_PRESENTATION  = "investor_presentation"
    PRESS_RELEASE          = "press_release"
    BROKER_NOTE            = "broker_note"
    NEWS_ARTICLE           = "news_article"
    MANAGEMENT_COMMENTARY  = "management_commentary"


class DocumentSection(str, Enum):
    PREPARED_REMARKS  = "prepared_remarks"
    QA_SESSION        = "qa_session"
    FINANCIAL_RESULTS = "financial_results"
    GUIDANCE          = "guidance"
    RISK_FACTORS      = "risk_factors"
    STRATEGY          = "strategy"
    MARKET_OVERVIEW   = "market_overview"
    UNKNOWN           = "unknown"


class SourceDocument(BaseModel):
    """A raw ingested document before chunking."""
    doc_id:      str
    ticker:      str
    company:     str
    doc_type:    DocumentType
    quarter:     str           # e.g. "Q2 2024"
    fiscal_year: int
    event_date:  Optional[datetime] = None
    source_url:  str = ""
    title:       str = ""
    raw_text:    str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """A single retrievable chunk with rich metadata."""
    chunk_id:    str
    doc_id:      str
    ticker:      str
    company:     str
    doc_type:    DocumentType
    section:     DocumentSection = DocumentSection.UNKNOWN
    quarter:     str
    fiscal_year: int
    event_date:  Optional[datetime] = None
    speaker:     str = ""          # CEO, CFO, Analyst, etc.
    is_management: bool = False
    text:        str = ""
    char_start:  int = 0
    source_url:  str = ""
    title:       str = ""


# ═══════════════════════════════════════════════════════
# Evidence / Citation
# ═══════════════════════════════════════════════════════

class Citation(BaseModel):
    """Quote-level evidence backing a signal."""
    chunk_id:   str
    doc_type:   str
    quarter:    str
    source_url: str = ""
    speaker:    str = ""
    quote:      str        # verbatim excerpt (≤120 chars)
    relevance:  float = 1.0   # 0–1 retrieval score


# ═══════════════════════════════════════════════════════
# Signal models  (structured outputs)
# ═══════════════════════════════════════════════════════

class SignalBase(BaseModel):
    ticker:      str
    company:     str
    quarter:     str
    fiscal_year: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    citations:   list[Citation] = Field(default_factory=list)


# ── 1. Management Confidence ─────────────────────────────────────────────────

class ConfidenceSignal(SignalBase):
    """
    Scores management confidence on a 0–10 scale across six dimensions.
    Compares to prior quarter to surface QoQ change and drivers.
    """
    score:           float          # 0–10
    previous_score:  Optional[float] = None
    change:          Optional[float] = None   # current − prior

    # Sub-dimension scores (0–10 each)
    confidence_level:    float = 0.0    # certainty vs. hedging
    uncertainty_level:   float = 0.0    # explicit uncertainty signals (inverted)
    defensiveness:       float = 0.0    # reactive / justifying tone (inverted)
    specificity:         float = 0.0    # concrete numbers vs. vague language
    consistency:         float = 0.0    # alignment with prior statements
    forward_strength:    float = 0.0    # positive forward-looking signals

    drivers:     list[str] = Field(default_factory=list)   # plain-English reasons for change
    tone:        str = ""    # overall tone label: "bullish" | "cautious" | "defensive" | "mixed"
    summary:     str = ""


# ── 2. Narrative Shift ───────────────────────────────────────────────────────

class ThemeStatus(str, Enum):
    ACCELERATING = "accelerating"
    EMERGING     = "emerging"
    STABLE       = "stable"
    FADING       = "fading"
    NEWLY_RISKY  = "newly_risky"
    RESOLVED     = "resolved"


class ThemeSignal(BaseModel):
    theme:                     str
    status:                    ThemeStatus
    evidence_count_current:    int = 0
    evidence_count_previous:   int = 0
    count_change:              int = 0
    sentiment_current:         float = 0.0    # –1 to +1
    sentiment_previous:        float = 0.0
    sentiment_change:          float = 0.0
    interpretation:            str = ""
    key_quotes:                list[str] = Field(default_factory=list)
    citations:                 list[Citation] = Field(default_factory=list)


class NarrativeSignal(SignalBase):
    """
    Compares themes across quarters to detect narrative shifts.
    """
    themes:           list[ThemeSignal] = Field(default_factory=list)
    accelerating:     list[str] = Field(default_factory=list)
    emerging:         list[str] = Field(default_factory=list)
    fading:           list[str] = Field(default_factory=list)
    newly_risky:      list[str] = Field(default_factory=list)
    overall_shift:    str = ""    # "positive" | "negative" | "neutral" | "mixed"
    shift_summary:    str = ""


# ── 3. Guidance Credibility ──────────────────────────────────────────────────

class GuidanceItem(BaseModel):
    metric:        str      # e.g. "Revenue", "EPS", "NIM", "Operating Margin"
    period:        str      # quarter the guidance was FOR
    guided_in:     str      # quarter the guidance was GIVEN
    guidance:      str      # stated value/range
    actual:        Optional[str] = None
    outcome:       str = ""  # "beat" | "miss" | "in_line" | "withdrew" | "pending"
    miss_reason:   str = ""
    citation:      Optional[Citation] = None


class GuidanceSignal(SignalBase):
    """
    Scores how accurately management's guidance matched actual results.
    Tracks history of guidance items across periods.
    """
    score:                    float   # 0–100
    guidance_items:           list[GuidanceItem] = Field(default_factory=list)
    periods_tracked:          int = 0
    beats:                    int = 0
    misses:                   int = 0
    in_line:                  int = 0
    withdrawals:              int = 0
    beat_rate:                float = 0.0    # beats / (beats + misses + in_line)
    serial_miss_risk:         bool = False
    summary:                  str = ""
    recent_pattern:           list[str] = Field(default_factory=list)   # e.g. ["beat","miss","beat"]


# ── 4. Risk Emergence ────────────────────────────────────────────────────────

class RiskSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class RiskStatus(str, Enum):
    NEWLY_MATERIAL = "newly_material"
    ESCALATING     = "escalating"
    STABLE         = "stable"
    DIMINISHING    = "diminishing"
    RESOLVED       = "resolved"


class RiskItem(BaseModel):
    risk:              str
    status:            RiskStatus
    severity:          RiskSeverity
    mention_count_current:  int = 0
    mention_count_previous: int = 0
    count_change:      int = 0
    evidence:          str = ""    # one-line evidence statement
    key_quotes:        list[str] = Field(default_factory=list)
    citations:         list[Citation] = Field(default_factory=list)


class RiskSignal(SignalBase):
    """
    Detects newly material or escalating risks from document evidence.
    """
    risks:          list[RiskItem] = Field(default_factory=list)
    new_risks:      list[str] = Field(default_factory=list)
    escalating:     list[str] = Field(default_factory=list)
    diminishing:    list[str] = Field(default_factory=list)
    overall_risk_direction: str = ""   # "increasing" | "stable" | "decreasing"
    summary:        str = ""


# ── Composite output ─────────────────────────────────────────────────────────

class SignalBundle(BaseModel):
    """All four signals for one company + quarter."""
    ticker:      str
    company:     str
    quarter:     str
    fiscal_year: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    docs_ingested: int = 0

    confidence:  Optional[ConfidenceSignal]  = None
    narrative:   Optional[NarrativeSignal]   = None
    guidance:    Optional[GuidanceSignal]    = None
    risk:        Optional[RiskSignal]        = None

    errors:      list[str] = Field(default_factory=list)
