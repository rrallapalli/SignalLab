"""
scoring/confidence.py — Management Confidence (0-10), LEVEL score.

Confidence is a property of THIS quarter's language — how confident, specific
and unhedged management sounds — so it is scored from this quarter's features
alone. It is deliberately NOT quarter-over-quarter:

  * Only the anchor quarter of a run ever had its prior ingested, so a QoQ term
    made the Latest column score on a different basis than the QoQ/YoY columns
    beside it — the three were not comparable.
  * A level score is reproducible: it depends only on the quarter's own evidence,
    not on whether a neighbour happened to be fetched.

QoQ movement is still shown — as the delta between stored level scores — but it
is computed at the display layer, not baked into the number.

LLM extraction contract: the model reads MANAGEMENT statements only and emits
atomic features, each with a verbatim quote. It does NOT emit a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import (EvidenceRef, ScoreLedger, ScoreResult, confidence_label,
                   present)

CONFIDENCE_SCORING_VERSION = "2.0.0"   # 2.x = level score (was QoQ-relative)

# --- weights (named, defensible, tied to the published rubric) -------------- #
BASELINE = 7.0                 # managements skew positive; 7 = steady/neutral
W_TONE = 2.0                   # mean management polarity in [-1,1] -> +/-2.0
W_HEDGE_DENSITY = -1.5         # hedging per 1k words, scaled
W_DEFENSIVE = -0.6             # each defensive term above a small floor (capped)
W_WIDE_GUIDANCE = -0.8         # guidance range notably wide / uncertain
W_SUPERLATIVE = 0.4            # strong positive assertions (capped)
W_TOPIC_AVOIDANCE = -1.0       # declined to give a number it normally would
DEFENSIVE_FLOOR = 3            # a few caveats are normal; penalise the excess
CAP_DEFENSIVE = 1.8            # max magnitude from the defensive term
CAP_SUPERLATIVE = 0.8


@dataclass
class ConfidenceFeatures:
    """Frozen features for one company-quarter (management statements only)."""
    mean_tone: float = 0.0                       # [-1, 1]
    tone_evidence: list[EvidenceRef] = field(default_factory=list)
    hedge_per_1k: float = 0.0                     # hedging markers / 1k words
    hedge_evidence: list[EvidenceRef] = field(default_factory=list)
    defensive_terms: int = 0                      # count this quarter
    defensive_evidence: list[EvidenceRef] = field(default_factory=list)
    wide_guidance: bool = False                   # range notably wide THIS quarter
    guidance_evidence: list[EvidenceRef] = field(default_factory=list)
    superlative_terms: int = 0
    superlative_evidence: list[EvidenceRef] = field(default_factory=list)
    topic_avoidance: bool = False
    avoidance_evidence: list[EvidenceRef] = field(default_factory=list)
    total_words: int = 0                          # for conflict/sufficiency checks


def _hedge_penalty(per_1k: float) -> float:
    # ~5 hedges/1k words is unremarkable; penalise the excess, saturating.
    excess = max(0.0, per_1k - 5.0)
    return W_HEDGE_DENSITY * min(1.0, excess / 10.0)


def score_confidence(curr: ConfidenceFeatures,
                     prev: Optional[ConfidenceFeatures] = None,
                     ensemble_spread: Optional[float] = None) -> ScoreResult:
    """
    Level score in [0, 10]. `prev` is accepted for call-signature compatibility
    but IGNORED — confidence is no longer quarter-over-quarter.
    """
    L = ScoreLedger(baseline=BASELINE, lo=0.0, hi=10.0)

    # Tone
    L.add("tone", W_TONE * curr.mean_tone,
          detail=f"mean management polarity {curr.mean_tone:+.2f}",
          evidence=curr.tone_evidence)

    # Hedging density
    L.add("hedging", _hedge_penalty(curr.hedge_per_1k),
          detail=f"{curr.hedge_per_1k:.1f} hedges / 1k words",
          evidence=curr.hedge_evidence)

    # Defensive language (level: excess above a small floor of normal caveats)
    if curr.defensive_terms > DEFENSIVE_FLOOR:
        delta = max(-CAP_DEFENSIVE,
                    W_DEFENSIVE * (curr.defensive_terms - DEFENSIVE_FLOOR))
        L.add("defensive_language", delta,
              detail=f"{curr.defensive_terms} defensive terms",
              evidence=curr.defensive_evidence)

    # Wide / uncertain guidance range
    if curr.wide_guidance:
        L.add("wide_guidance", W_WIDE_GUIDANCE,
              detail="guidance range notably wide / uncertain",
              evidence=curr.guidance_evidence)

    # Superlatives (capped positive)
    if curr.superlative_terms:
        delta = min(CAP_SUPERLATIVE, W_SUPERLATIVE * curr.superlative_terms)
        L.add("superlatives", delta,
              detail=f"{curr.superlative_terms} strong positive assertions",
              evidence=curr.superlative_evidence)

    # Topic avoidance
    if curr.topic_avoidance:
        L.add("topic_avoidance", W_TOPIC_AVOIDANCE,
              detail="declined to give a number it normally would",
              evidence=curr.avoidance_evidence)

    n_ev = sum(len(i.evidence) for i in L.items)
    conflict = curr.mean_tone > 0.3 and curr.hedge_per_1k > 12  # bullish+hedgy
    conf, reason = confidence_label(n_evidence=n_ev, conflict=conflict,
                                    spread=ensemble_spread)

    return ScoreResult(
        signal="confidence",
        scorer_version=CONFIDENCE_SCORING_VERSION,
        value=present(L.total, 1),
        ledger=L,
        confidence=conf,
        confidence_reason=reason,
        extras={},
    )


CONFIDENCE_EXTRACTION_PROMPT = """\
You extract evidence, NOT scores. Read ONLY management statements (CEO/CFO/IR);
ignore analyst questions. Return strict JSON with these fields. For every item,
include the verbatim quote and its chunk_id. Invent nothing.

{
  "management_statements": [ {"chunk_id": "...", "quote": "...", "speaker": "CEO", "tone": "positive|neutral|negative"} ],
  "hedging_markers":       [ {"chunk_id": "...", "quote": "...phrase..."} ],
  "defensive_terms":       [ {"chunk_id": "...", "quote": "...", "term": "headwind"} ],
  "superlative_terms":     [ {"chunk_id": "...", "quote": "...", "term": "record"} ],
  "wide_guidance":         false,
  "topic_avoidance":       false
}
Counts and densities are computed in code, not by you. Everything describes THIS
quarter only — no comparison to a prior quarter.
"""
