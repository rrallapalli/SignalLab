"""
scoring/guidance.py — Guidance Credibility (0-100).

Two parts:
  - TRACK RECORD (historical, code-computed): beat/miss/in_line pattern from
    reconciling prior guidance against reported actuals. beat_rate and
    serial-miss are computed here, never requested from the LLM.
  - LANGUAGE (this quarter): specificity vs vagueness, hedging, conditionality.

If there is NO track record yet, the score is language-only, capped, and
flagged low-confidence — it must NOT silently return a midpoint (that was the
old 45/100-for-zero-items bug).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import EvidenceRef, ScoreLedger, ScoreResult, confidence_label, present

GUIDANCE_SCORING_VERSION = "1.0.0"

OUTCOMES = {"beat", "in_line", "miss"}


@dataclass
class GuidanceFeatures:
    # historical outcomes, oldest -> newest, each derived by code from actuals
    outcomes: list[str] = field(default_factory=list)          # e.g. ["beat","miss",...]
    outcome_evidence: list[EvidenceRef] = field(default_factory=list)
    # current-quarter language
    is_specific: bool = False        # gave a concrete number/range
    is_hedged: bool = False          # heavy conditionality around the guide
    withdrew_guidance: bool = False
    language_evidence: list[EvidenceRef] = field(default_factory=list)


def _hit_rate(outcomes: list[str]) -> Optional[float]:
    """Share of tracked periods where management HIT its guidance (met or beat).
    In-line counts as a hit: delivering what you guided is the point of guidance;
    beating is a bonus, not the bar. This is the credibility basis — distinct
    from a beats-only "beat rate", which measures how often they OVERSHOT."""
    tracked = [o for o in outcomes if o in OUTCOMES]
    if not tracked:
        return None
    met = sum(1 for o in tracked if o in ("beat", "in_line"))
    return met / len(tracked)


def _serial_miss(outcomes: list[str], n: int = 2) -> bool:
    streak = 0
    for o in outcomes:                # oldest -> newest; trailing streak matters
        streak = streak + 1 if o == "miss" else 0
        if streak >= n:
            return True
    return outcomes[-n:].count("miss") >= n if len(outcomes) >= n else False


def score_guidance(f: GuidanceFeatures,
                   ensemble_spread: Optional[float] = None) -> ScoreResult:
    L = ScoreLedger(baseline=50.0, lo=0.0, hi=100.0)
    br = _hit_rate(f.outcomes)

    if br is None:
        # No history: language-only, explicitly low-confidence, capped band.
        L.baseline = 50.0
        if f.is_specific:
            L.add("specific_guidance", +8, "gave a concrete number/range",
                  evidence=f.language_evidence)
        if f.is_hedged:
            L.add("hedged_guidance", -8, "heavy conditionality",
                  evidence=f.language_evidence)
        if f.withdrew_guidance:
            L.add("withdrew_guidance", -15, "withdrew or declined to guide",
                  evidence=f.language_evidence)
        L.lo, L.hi = 35.0, 65.0       # cannot claim strong credibility w/o record
        return ScoreResult(
            signal="guidance", scorer_version=GUIDANCE_SCORING_VERSION,
            value=int(round(L.total)), ledger=L,
            confidence="low",
            confidence_reason="no reconciled guidance-vs-actuals history yet",
            extras={"hit_rate": None, "tracked_periods": 0,
                    "serial_miss_risk": False},
        )

    # Track record drives the baseline: beat_rate 0->10, .5->50, 1->90
    L.baseline = 50.0 + (br - 0.5) * 80.0
    tracked = [o for o in f.outcomes if o in OUTCOMES]
    L.add("track_record", 0.0,
          detail=f"hit {sum(o in ('beat','in_line') for o in tracked)}/"
                 f"{len(tracked)} guidance targets (hit rate {br:.2f})",
          evidence=f.outcome_evidence)  # baseline carries it; delta 0 documents it

    serial = _serial_miss(f.outcomes)
    if serial:
        L.add("serial_miss_risk", -15, "two or more consecutive recent misses",
              evidence=f.outcome_evidence)

    # Current-quarter language adjustments (smaller than the record)
    if f.is_specific:
        L.add("specific_guidance", +5, "gave a concrete number/range",
              evidence=f.language_evidence)
    if f.is_hedged:
        L.add("hedged_guidance", -5, "heavy conditionality around the guide",
              evidence=f.language_evidence)
    if f.withdrew_guidance:
        L.add("withdrew_guidance", -12, "withdrew or declined to guide",
              evidence=f.language_evidence)

    n_ev = len(tracked)
    conf, reason = confidence_label(n_evidence=n_ev, spread=ensemble_spread)
    if n_ev < 4 and conf == "high":
        conf, reason = "medium", f"short track record ({n_ev} periods)"

    return ScoreResult(
        signal="guidance", scorer_version=GUIDANCE_SCORING_VERSION,
        value=int(round(L.total)), ledger=L,
        confidence=conf, confidence_reason=reason,
        extras={"hit_rate": round(br, 2), "tracked_periods": n_ev,
                "serial_miss_risk": serial,
                "recent_pattern": tracked[-6:]},
    )
