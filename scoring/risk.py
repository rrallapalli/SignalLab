"""
scoring/risk.py — Risk Emergence (per risk).

Output: status (newly_material / escalating / stable / diminishing) + severity
(low / medium / high). Mention counts are code-computed from extracted risk
mentions. Severity is a rule over extracted qualifier language + who raised it.

Governance red-flags (promoter pledge changes, related-party growth, auditor
changes, contingent-liability creep) slot in as additional risk `category`
values feeding this same scorer once you add their extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import EvidenceRef, ScoreLedger, ScoreResult, confidence_label

RISK_SCORING_VERSION = "1.0.0"

ESCALATION_RATIO = 1.5


@dataclass
class RiskFeatures:
    risk: str
    category: str = "operational"          # operational | demand | governance | ...
    mentions_current: int = 0
    mentions_previous: int = 0
    raised_by_management: bool = False     # proactive flag vs analyst-only
    severe_qualifiers: int = 0             # "primary", "material", "significant"
    evidence: list[EvidenceRef] = field(default_factory=list)


def _status(cur: int, prev: int) -> str:
    if prev == 0 and cur > 0:
        return "newly_material"
    if cur == 0 and prev > 0:
        return "diminishing"
    if prev > 0 and cur / prev >= ESCALATION_RATIO:
        return "escalating"
    if prev > 0 and cur / prev <= 1 / ESCALATION_RATIO:
        return "diminishing"
    return "stable"


def _severity(f: RiskFeatures) -> tuple[str, ScoreLedger]:
    # transparent points -> band, so severity is auditable too
    L = ScoreLedger(baseline=0.0, lo=0.0, hi=10.0)
    L.add("mention_volume", min(4.0, f.mentions_current * 0.8),
          detail=f"{f.mentions_current} mentions this period", evidence=f.evidence)
    if f.raised_by_management:
        L.add("management_raised", 3.0, "flagged proactively by management")
    if f.severe_qualifiers:
        L.add("severe_language", min(3.0, f.severe_qualifiers * 1.5),
              detail=f"{f.severe_qualifiers} strong qualifiers")
    pts = L.total
    band = "high" if pts >= 7 else "medium" if pts >= 3.5 else "low"
    return band, L


def score_risk(f: RiskFeatures,
               ensemble_spread: Optional[float] = None) -> ScoreResult:
    status = _status(f.mentions_current, f.mentions_previous)
    severity, L = _severity(f)

    conf, reason = confidence_label(n_evidence=len(f.evidence),
                                    spread=ensemble_spread)
    return ScoreResult(
        signal="risk", scorer_version=RISK_SCORING_VERSION,
        value=severity, status=status, ledger=L,
        confidence=conf, confidence_reason=reason,
        extras={"risk": f.risk, "category": f.category,
                "mention_count_current": f.mentions_current,
                "mention_count_previous": f.mentions_previous},
    )


RISK_EXTRACTION_PROMPT = """\
Tag every distinct risk mention with chunk_id, verbatim quote, whether
management raised it proactively (vs. only answering an analyst), and any
strong qualifier used. Do not count — return individual mentions.

{
  "risks": [
    {"risk": "deposit competition", "category": "demand",
     "mentions": [ {"chunk_id":"...","quote":"...",
                    "raised_by_management": true, "qualifier": "primary"} ]}
  ]
}
"""
