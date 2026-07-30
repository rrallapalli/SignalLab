"""
scoring/confidence.py — Management Confidence (0-10).

Confidence is inherently QoQ-relative, so the scorer takes current AND prior
features. If no prior exists, QoQ terms are skipped and confidence is lowered
rather than fabricated.

LLM extraction contract (CONFIDENCE_EXTRACTION_PROMPT below): the model reads
MANAGEMENT statements only (exclude analyst turns) and emits atomic features,
each with a verbatim quote + chunk_id. It does NOT emit a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import (EvidenceRef, ScoreLedger, ScoreResult, confidence_label,
                   present)

CONFIDENCE_SCORING_VERSION = "1.0.0"

# --- weights (named, defensible, tied to the published rubric) -------------- #
BASELINE = 7.0                 # managements skew positive; 7 = steady/neutral
W_TONE = 2.0                   # mean management polarity in [-1,1] -> +/-2.0
W_HEDGE_DENSITY = -1.5         # hedging per 1k words, scaled
W_DEFENSIVE_QOQ = -0.6         # each net new defensive term QoQ (capped)
W_GUIDANCE_WIDENED = -0.8      # guidance band widened materially
W_SUPERLATIVE = 0.4            # strong positive assertions (capped)
W_TOPIC_AVOIDANCE = -1.0       # declined to give a previously-given number
CAP_DEFENSIVE = 1.8            # max magnitude from the defensive-QoQ term
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
    guidance_band_widened: bool = False
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
    L = ScoreLedger(baseline=BASELINE, lo=0.0, hi=10.0)

    # Tone (level term, always present)
    L.add("tone", W_TONE * curr.mean_tone,
          detail=f"mean management polarity {curr.mean_tone:+.2f}",
          evidence=curr.tone_evidence)

    # Hedging density (level term)
    L.add("hedging", _hedge_penalty(curr.hedge_per_1k),
          detail=f"{curr.hedge_per_1k:.1f} hedges / 1k words",
          evidence=curr.hedge_evidence)

    # Defensive language — QoQ delta if we have a prior, else level-vs-baseline
    if prev is not None:
        net = curr.defensive_terms - prev.defensive_terms
        delta = max(-CAP_DEFENSIVE, min(CAP_DEFENSIVE, W_DEFENSIVE_QOQ * net))
        L.add("defensive_language_qoq", delta,
              detail=f"{curr.defensive_terms} vs {prev.defensive_terms} prior "
                     f"(net {net:+d})",
              evidence=curr.defensive_evidence)
    else:
        delta = max(-CAP_DEFENSIVE, W_DEFENSIVE_QOQ * max(0, curr.defensive_terms - 3))
        L.add("defensive_language_level", delta,
              detail=f"{curr.defensive_terms} defensive terms (no prior quarter)",
              evidence=curr.defensive_evidence)

    # Guidance band
    if curr.guidance_band_widened:
        L.add("guidance_band_widened", W_GUIDANCE_WIDENED,
              detail="guidance range widened materially QoQ",
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
              detail="declined to give a previously-provided number",
              evidence=curr.avoidance_evidence)

    n_ev = sum(len(i.evidence) for i in L.items)
    conflict = curr.mean_tone > 0.3 and curr.hedge_per_1k > 12  # bullish+hedgy
    conf, reason = confidence_label(n_evidence=n_ev, conflict=conflict,
                                    spread=ensemble_spread)
    if prev is None and conf == "high":
        conf, reason = "medium", "no prior quarter for QoQ comparison"

    return ScoreResult(
        signal="confidence",
        scorer_version=CONFIDENCE_SCORING_VERSION,
        value=present(L.total, 1),
        ledger=L,
        confidence=conf,
        confidence_reason=reason,
        extras={"previous": present(BASELINE, 1) if prev is None else None},
    )


CONFIDENCE_EXTRACTION_PROMPT = """\
You extract evidence, NOT scores. Read ONLY management statements (CEO/CFO/IR);
ignore analyst questions. Return strict JSON with these fields. For every item,
include the verbatim quote and its chunk_id. Invent nothing.

{
  "management_statements": [ {"chunk_id": "...", "quote": "...", "speaker": "CEO"} ],
  "hedging_markers":       [ {"chunk_id": "...", "quote": "...phrase..."} ],
  "defensive_terms":       [ {"chunk_id": "...", "quote": "...", "term": "headwind"} ],
  "superlative_terms":     [ {"chunk_id": "...", "quote": "...", "term": "record"} ],
  "guidance_statements":   [ {"chunk_id": "...", "quote": "...", "low": null, "high": null} ],
  "topic_avoidance":       [ {"chunk_id": "...", "quote": "...", "what": "declined FY margin"} ]
}
Counts, densities and QoQ deltas are computed in code, not by you.
"""
