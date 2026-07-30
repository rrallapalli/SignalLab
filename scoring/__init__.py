"""Deterministic, evidence-anchored scoring package.

The LLM extracts source-anchored features; scores are pure functions over
those features and always return an auditable ledger.
"""
from .base import (
    SCORING_FRAMEWORK_VERSION, EvidenceRef, Feature, LedgerItem, ScoreLedger,
    ScoreResult, clamp, present, content_hash, confidence_label,
    ToneProvider, LLMToneProvider, FinBERTToneProvider,
)
from .confidence import (ConfidenceFeatures, score_confidence,
                         CONFIDENCE_SCORING_VERSION, CONFIDENCE_EXTRACTION_PROMPT)
from .narrative import (NarrativeFeatures, score_narrative,
                        NARRATIVE_SCORING_VERSION, NARRATIVE_EXTRACTION_PROMPT)
from .guidance import (GuidanceFeatures, score_guidance,
                       GUIDANCE_SCORING_VERSION)
from .risk import (RiskFeatures, score_risk,
                   RISK_SCORING_VERSION, RISK_EXTRACTION_PROMPT)

# Roll these into corpus_fingerprint() so any scoring change invalidates cache.
SCORER_VERSIONS = {
    "framework": SCORING_FRAMEWORK_VERSION,
    "confidence": CONFIDENCE_SCORING_VERSION,
    "narrative": NARRATIVE_SCORING_VERSION,
    "guidance": GUIDANCE_SCORING_VERSION,
    "risk": RISK_SCORING_VERSION,
}
