"""
scoring/narrative.py — Narrative Shift (per theme).

Not a 0-10. Output is a status (accelerating / building / stable / fading /
new / dropped) plus a magnitude in [0,1] and a direction. Counts are computed
in code from extracted theme-tagged mentions — never asserted by the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import EvidenceRef, ScoreLedger, ScoreResult, confidence_label, present

NARRATIVE_SCORING_VERSION = "1.0.0"

# thresholds
GROWTH_STRONG = 1.5     # current/prior mention ratio for "accelerating"
GROWTH_MILD = 1.15
SENT_DELTA_STRONG = 0.25


@dataclass
class NarrativeFeatures:
    theme: str
    mentions_current: int = 0
    mentions_previous: int = 0
    sentiment_current: float = 0.0        # [-1, 1]
    sentiment_previous: float = 0.0
    in_opening_remarks: bool = False      # prominence: prepared vs Q&A-only
    evidence: list[EvidenceRef] = field(default_factory=list)


def _ratio(cur: int, prev: int) -> float:
    if prev == 0:
        return float("inf") if cur > 0 else 1.0
    return cur / prev


def score_narrative(f: NarrativeFeatures,
                    ensemble_spread: Optional[float] = None) -> ScoreResult:
    L = ScoreLedger(baseline=0.0, lo=0.0, hi=1.0)  # baseline=magnitude accumulator

    ratio = _ratio(f.mentions_current, f.mentions_previous)
    sent_delta = f.sentiment_current - f.sentiment_previous

    # frequency component
    if ratio == float("inf"):
        freq_mag, freq_txt = 0.6, "newly introduced theme"
    else:
        freq_mag = min(1.0, abs(ratio - 1.0))
        freq_txt = f"mentions {f.mentions_current} vs {f.mentions_previous} (x{ratio:.2f})"
    L.add("frequency_shift", freq_mag * 0.6, detail=freq_txt, evidence=f.evidence)

    # sentiment component
    L.add("sentiment_shift", min(1.0, abs(sent_delta) / 0.5) * 0.3,
          detail=f"sentiment {f.sentiment_current:+.2f} vs {f.sentiment_previous:+.2f}")

    # prominence component
    if f.in_opening_remarks:
        L.add("prominence", 0.1, detail="raised in prepared remarks, not only Q&A")

    magnitude = L.total

    # status = deterministic rule over direction + magnitude
    rising = ratio > 1.0 or sent_delta > 0
    if f.mentions_previous == 0 and f.mentions_current > 0:
        status = "new"
    elif f.mentions_current == 0 and f.mentions_previous > 0:
        status = "dropped"
    elif ratio >= GROWTH_STRONG or sent_delta >= SENT_DELTA_STRONG:
        status = "accelerating"
    elif ratio >= GROWTH_MILD:
        status = "building"
    elif ratio <= 1 / GROWTH_MILD or sent_delta <= -SENT_DELTA_STRONG:
        status = "fading"
    else:
        status = "stable"

    n_ev = len(f.evidence)
    conf, reason = confidence_label(n_evidence=n_ev, spread=ensemble_spread)

    return ScoreResult(
        signal="narrative",
        scorer_version=NARRATIVE_SCORING_VERSION,
        value=present(magnitude, 2),
        status=status,
        ledger=L,
        confidence=conf,
        confidence_reason=reason,
        extras={"theme": f.theme, "direction": "up" if rising else "down",
                "count_change": f.mentions_current - f.mentions_previous},
    )


NARRATIVE_EXTRACTION_PROMPT = """\
For each theme you are tracking, tag every distinct mention with its chunk_id
and verbatim quote, and mark whether it appeared in prepared remarks. Do NOT
count — return the individual mentions; code aggregates them.

{
  "themes": [
    {"theme": "AI demand",
     "mentions": [ {"chunk_id":"...","quote":"...","in_opening_remarks": true} ]}
  ]
}
"""
