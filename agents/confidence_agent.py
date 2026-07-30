"""
agents/confidence_agent.py
RAG → LLM (EXTRACTION ONLY) → deterministic ManagementConfidenceScore (0–10).

LEVEL score: the LLM extracts atomic, quoted features for THIS quarter only
(no prior-quarter retrieval), scoring.confidence.score_confidence computes the
0–10 from them, and the six sub-dimensions are DERIVED from the same features.
QoQ movement is shown at the display layer as the delta between stored scores,
not baked into the number — so all three columns are scored on one basis and
the score is reproducible from the quarter's own evidence.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, ConfidenceSignal
from agents.base import BaseAgent
from store.vector_store import VectorStore

# Deterministic scorer + feature container (repo-root `scoring/` package).
from scoring import ConfidenceFeatures, score_confidence, EvidenceRef

TONE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def _tone_mean(statements: list) -> float:
    vals = [TONE_MAP.get((s.get("tone") or "neutral").strip().lower(), 0.0)
            for s in (statements or [])]
    return sum(vals) / len(vals) if vals else 0.0


def _refs(items: list) -> list[EvidenceRef]:
    """Quote-only evidence refs (confidence evidence is not chunk-id verified,
    matching the prior agent, which never verified confidence quotes)."""
    return [EvidenceRef(chunk_id="", quote=(it.get("quote") or ""))
            for it in (items or []) if it.get("quote")]


def _tone_label(f: ConfidenceFeatures) -> str:
    if f.mean_tone >= 0.3 and f.hedge_per_1k < 8:
        return "bullish"
    if f.defensive_terms >= 4 or f.mean_tone <= -0.2:
        return "defensive"
    if f.hedge_per_1k >= 8:
        return "cautious"
    return "mixed"


def _sub_dimensions(f: ConfidenceFeatures) -> dict:
    """
    First-pass decomposition of the SAME features that drive the headline score,
    on the existing 0–10 dashboard dimensions (uncertainty_level and
    defensiveness are inverted: 10 = low uncertainty / not defensive). Heuristic
    weights, meant to be tuned; the headline score is the principled number.
    """
    hedge_pen = _clamp((f.hedge_per_1k - 5) / 10.0, 0.0, 1.0)      # 0..1
    def_pen = _clamp(f.defensive_terms / 8.0, 0.0, 1.0)           # 0..1
    sup = min(2.0, f.superlative_terms * 0.7)
    band = 2.0 if f.wide_guidance else 0.0
    return {
        "confidence_level":  round(_clamp(5 + 3 * f.mean_tone - 4 * hedge_pen), 1),
        "uncertainty_level": round(_clamp(10 - 7 * hedge_pen), 1),
        "defensiveness":     round(_clamp(10 - 7 * def_pen), 1),
        "specificity":       round(_clamp(6 - band + sup), 1),
        "consistency":       round(_clamp(7 - band), 1),
        "forward_strength":  round(_clamp(5 + 3 * f.mean_tone + sup), 1),
    }


SYSTEM_PROMPT = """You EXTRACT evidence about management confidence — you do NOT
output a score. The 0–10 confidence score is computed downstream from the
features you return, so any score you write here would be ignored.

Read MANAGEMENT statements only (CEO / CFO / IR); ignore analyst questions.
Describe THIS quarter only — do NOT compare to a prior quarter. Extract atomic
observations, each with a verbatim quote. Invent nothing; empty categories -> [].

Return ONLY valid JSON:
{
  "management_statements": [{"quote": "...", "tone": "positive|neutral|negative"}],
  "hedging_markers":       [{"quote": "...we are monitoring closely..."}],
  "defensive_terms":       [{"quote": "...", "term": "headwind"}],
  "superlative_terms":     [{"quote": "...", "term": "record"}],
  "wide_guidance":         false,
  "topic_avoidance":       false
}

tone: your read of each management statement's stance.
wide_guidance: true if the guidance range given this quarter is notably wide or
  uncertain (a lower-confidence signal on its own, no comparison needed).
topic_avoidance: true if management declined to give a number it normally would.
Counts and densities are computed in code — do not compute them.
"""


class ConfidenceAgent(BaseAgent):

    EVIDENCE_QUERIES = [
        "management confidence outlook forward guidance expectations",
        "CEO CFO tone language certainty uncertainty",
        "challenging headwinds cautious confident strong",
        "we expect we are confident we believe we are monitoring",
        "guidance raised lowered maintained reiterated",
        "margin revenue growth target committed",
    ]

    def __init__(self, vs: VectorStore, model: str | None = None):
        super().__init__(vs, model)

    async def run(
        self,
        ticker: str,
        company: str,
        quarter: str,
        fiscal_year: int,
        prior_quarter: str = "",   # accepted for orchestrator call-compat; unused
        prior_year: int | str = "",
    ) -> ConfidenceSignal:
        logger.info(f"[ConfidenceAgent] Running for {ticker} {quarter} {fiscal_year}")

        # Current quarter only — a level score needs no prior retrieval.
        current_chunks = self.rag_retrieve(
            queries=self.EVIDENCE_QUERIES, ticker=ticker,
            quarter=quarter, fiscal_year=fiscal_year,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            management_only=True, top_k_per_query=6,
        )
        current_evidence = self.format_evidence(current_chunks[:10])
        citations = self.vs.as_citations(current_chunks[:8])
        cur_words = sum(len((c.text or "").split()) for c, _ in current_chunks) or 1

        user_prompt = f"""Company: {company} ({ticker})
Quarter: {quarter} {fiscal_year}

=== EVIDENCE (management only) ===
{current_evidence or "No evidence retrieved."}

Extract the confidence features for THIS quarter, per the schema. Quote verbatim.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            curr_feats = ConfidenceFeatures(
                mean_tone=_tone_mean(data.get("management_statements")),
                tone_evidence=_refs(data.get("management_statements")),
                hedge_per_1k=len(data.get("hedging_markers") or []) / cur_words * 1000.0,
                hedge_evidence=_refs(data.get("hedging_markers")),
                defensive_terms=len(data.get("defensive_terms") or []),
                defensive_evidence=_refs(data.get("defensive_terms")),
                wide_guidance=bool(data.get("wide_guidance")),
                superlative_terms=len(data.get("superlative_terms") or []),
                superlative_evidence=_refs(data.get("superlative_terms")),
                topic_avoidance=bool(data.get("topic_avoidance")),
                avoidance_evidence=[],
                total_words=cur_words,
            )

            result = score_confidence(curr_feats)

            subdims = _sub_dimensions(curr_feats)
            tone = _tone_label(curr_feats)
            drivers = [f"{i.label}: {i.detail}"
                       for i in result.ledger.items if abs(i.delta) > 1e-9][:4]
            # "reliability" = how much to trust THIS reading (evidence
            # sufficiency), deliberately NOT called "confidence": that word is
            # the signal itself.
            summary = (f"Management confidence {result.value}/10. "
                       f"Read reliability: {result.confidence} "
                       f"({result.confidence_reason})."
                       + (f" {drivers[0]}." if drivers else ""))
            _md = result.to_dict()
            _md["reliability"] = _md.pop("confidence")
            _md["reliability_reason"] = _md.pop("confidence_reason")
            manifest = {**_md, "sub_dimensions": subdims, "tone": tone}

            return ConfidenceSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                score=result.value,              # computed, not judged
                previous_score=None,             # dashboard computes the delta
                change=None,
                confidence_level=subdims["confidence_level"],
                uncertainty_level=subdims["uncertainty_level"],
                defensiveness=subdims["defensiveness"],
                specificity=subdims["specificity"],
                consistency=subdims["consistency"],
                forward_strength=subdims["forward_strength"],
                tone=tone,
                drivers=drivers,
                summary=summary,
                citations=citations,
                manifest=manifest,
            )
        except Exception as e:
            logger.error(f"[ConfidenceAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
