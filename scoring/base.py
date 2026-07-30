"""
scoring/base.py — deterministic scoring framework.

Design contract (this is the whole point):
  1. The LLM (or a classifier) produces ONLY atomic, source-anchored features.
     Each feature carries an EvidenceRef so it can be verified and audited.
  2. Scores are PURE FUNCTIONS over frozen features. No LLM call happens at
     scoring time. Given identical features, the score is bitwise reproducible.
  3. Every score returns a ScoreLedger: baseline + itemised deltas, each delta
     tied to the evidence that produced it. The ledger IS the audit trail.

Determinism rules enforced here:
  - No reliance on dict ordering; we sort before iterating anywhere it matters.
  - Rounding happens exactly once, at presentation time, never mid-computation.
  - Classifier providers are pinned (fixed revision, eval mode, no_grad).
  - Framework/version constants below feed cache invalidation upstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Protocol, Sequence

# Bump when the framework's math or data shapes change. Roll this into your
# corpus_fingerprint() so any change invalidates cached signals.
SCORING_FRAMEWORK_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Evidence + features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceRef:
    """Points at the exact source span a feature/delta came from."""
    chunk_id: str
    quote: str                      # verbatim, verified against source text
    doc_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page: Optional[int] = None
    speaker: Optional[str] = None   # e.g. "CEO", "CFO", "Analyst"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Feature:
    """
    One atomic, extracted observation. `value` is whatever the feature needs
    (bool flag, count, polarity float, phrase string). Kept generic so the
    same container serves every agent.
    """
    name: str
    value: object
    evidence: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "evidence": [e.to_dict() for e in self.evidence],
        }


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
@dataclass
class LedgerItem:
    label: str
    delta: float
    detail: str = ""
    evidence: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "delta": round(self.delta, 4),
            "detail": self.detail,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class ScoreLedger:
    baseline: float
    items: list[LedgerItem] = field(default_factory=list)
    lo: float = 0.0
    hi: float = 10.0

    def add(self, label: str, delta: float, detail: str = "",
            evidence: Sequence[EvidenceRef] = ()) -> None:
        # A zero delta is still recorded — "we looked and it didn't move" is
        # itself auditable information.
        self.items.append(LedgerItem(label, float(delta), detail, tuple(evidence)))

    @property
    def raw_total(self) -> float:
        return self.baseline + sum(i.delta for i in self.items)

    @property
    def total(self) -> float:
        return clamp(self.raw_total, self.lo, self.hi)

    @property
    def clamped(self) -> bool:
        return self.raw_total != self.total

    def to_dict(self) -> dict:
        return {
            "baseline": round(self.baseline, 4),
            "raw_total": round(self.raw_total, 4),
            "total": self.total,
            "range": [self.lo, self.hi],
            "clamped": self.clamped,
            "items": [i.to_dict() for i in self.items],
        }


# --------------------------------------------------------------------------- #
# Result envelope (this is your per-score audit record / manifest payload)
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    signal: str                     # "confidence" | "narrative" | ...
    scorer_version: str
    value: object                   # float | str | dict, per signal
    ledger: Optional[ScoreLedger] = None
    status: Optional[str] = None    # e.g. narrative/risk status enums
    confidence: str = "unknown"     # "high" | "medium" | "low"
    confidence_reason: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "scorer_version": self.scorer_version,
            "framework_version": SCORING_FRAMEWORK_VERSION,
            "value": self.value,
            "status": self.status,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "ledger": self.ledger.to_dict() if self.ledger else None,
            "extras": self.extras,
        }


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def present(x: float, ndigits: int = 1) -> float:
    """Round exactly once, at presentation time."""
    return round(x, ndigits)


def content_hash(*parts: object) -> str:
    """Stable hash for cache keys / manifest fingerprints."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def confidence_label(
    n_evidence: int,
    conflict: bool = False,
    spread: Optional[float] = None,
    spread_hi: float = 0.5,
) -> tuple[str, str]:
    """
    How much to trust THIS score. Prefer an ensemble `spread` (std dev across
    seeds/models) when you have it; otherwise fall back to evidence sufficiency.
    Returning low confidence honestly beats sounding certain on thin evidence.
    """
    if spread is not None:
        if spread <= spread_hi / 2:
            return "high", f"tight ensemble agreement (spread {spread:.2f})"
        if spread <= spread_hi:
            return "medium", f"moderate ensemble spread ({spread:.2f})"
        return "low", f"wide ensemble spread ({spread:.2f})"
    if n_evidence == 0:
        return "low", "no supporting evidence extracted"
    if conflict:
        return "low", "conflicting evidence across sources"
    if n_evidence < 3:
        return "medium", f"sparse evidence ({n_evidence} items)"
    return "high", f"sufficient evidence ({n_evidence} items)"


# --------------------------------------------------------------------------- #
# Tone provider — pluggable so you can start with zero new deps
# --------------------------------------------------------------------------- #
class ToneProvider(Protocol):
    """Returns polarity in [-1, 1] for a piece of text."""
    def polarity(self, text: str) -> float: ...
    @property
    def name(self) -> str: ...


class LLMToneProvider:
    """
    Default: tone already extracted by your LLM step (no new dependency).
    You pass the polarity the extractor emitted; this just passes it through
    so the scorer's interface is uniform.
    """
    name = "llm-extracted-tone"

    def polarity(self, text: str) -> float:  # not used in passthrough mode
        raise NotImplementedError(
            "LLMToneProvider is passthrough; supply polarity in the feature."
        )


class FinBERTToneProvider:
    """
    Optional, more deterministic tone. A pinned classifier's argmax rarely
    flips on float wobble, unlike generative decoding. Requires transformers
    + torch; imported lazily so the base package stays light.

    Pinning for reproducibility:
      - fixed model revision
      - eval() + torch.no_grad()
      - deterministic flags
    """
    def __init__(self, model: str = "yiyanghkust/finbert-tone",
                 revision: str = "main"):
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision).eval()
        self._model_name = f"{model}@{revision}"
        # finbert-tone label order: [neutral, positive, negative]
        self._pos, self._neg = 1, 2

    @property
    def name(self) -> str:
        return self._model_name

    def polarity(self, text: str) -> float:
        t = self._torch
        with t.no_grad():
            enc = self._tok(text, return_tensors="pt", truncation=True,
                            max_length=256)
            probs = t.softmax(self._model(**enc).logits, dim=-1)[0]
        return float(probs[self._pos] - probs[self._neg])
