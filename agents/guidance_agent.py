"""
agents/guidance_agent.py
RAG → LLM (EXTRACTION ONLY) → deterministic score → GuidanceSignal.

The LLM's job is now to extract guidance-vs-actuals OUTCOMES, not to judge a
credibility number. The 0–100 score is computed in code by scoring.guidance
from those outcomes, and the full ledger is attached as sig.manifest for audit.

Unchanged: retrieval, citations, and the None-vs-0 "not assessed" distinction.
"""

from __future__ import annotations

from collections import Counter

from loguru import logger
from models import Citation, GuidanceItem, GuidanceSignal
from agents.base import BaseAgent, safe_int
from store.vector_store import VectorStore

# Deterministic scorer + feature container (repo-root `scoring/` package).
from scoring import GuidanceFeatures, score_guidance, EvidenceRef


# The prompt no longer asks for a score, a rubric, beat_rate, serial_miss_risk,
# or counts — all of those are derived in code from the items below.
SYSTEM_PROMPT = """You are a quantitative equity analyst EXTRACTING guidance evidence.

You do NOT output a credibility score. That number is computed downstream from
the outcomes you return, so a score here would be ignored — spend your effort on
getting each item's outcome right instead.

You receive evidence from multiple quarters: prior-quarter GUIDANCE STATEMENTS
and subsequent ACTUAL RESULTS. For each trackable guidance item, compare what
management said against what actually happened and classify the outcome.

Return ONLY valid JSON:
{
  "guidance_items": [
    {
      "metric": "Revenue",
      "period": "Q3 2024",
      "guided_in": "Q2 2024",
      "guidance": "$20.0B-$21.0B",
      "actual": "$20.5B",
      "outcome": "in_line",
      "miss_reason": ""
    },
    {
      "metric": "Operating Margin",
      "period": "Q3 2024",
      "guided_in": "Q2 2024",
      "guidance": "~28%",
      "actual": "25.1%",
      "outcome": "miss",
      "miss_reason": "Higher R&D spend and FX headwinds not anticipated in guidance"
    }
  ]
}

outcome must be one of: beat | miss | in_line | withdrew | pending.
Only beat / miss / in_line count toward the track record; withdrew and pending
are recorded but do not score. Do NOT return counts, rates, or a score — those
are computed in code from the items you return.
"""


def _serial_miss_metrics(items: list) -> list[str]:
    """
    Metrics this company has missed 2+ times (for the displayed serial_miss_risk
    flag). One definition, shared with store.get_ytd_guidance(). Unchanged.
    """
    counts = Counter(
        (i.metric or "").strip().lower()
        for i in items
        if (i.outcome or "").lower() == "miss" and (i.metric or "").strip()
    )
    return sorted(m for m, c in counts.items() if c >= 2)


def _guidance_summary(value: int, beats: int, in_line: int, misses: int,
                      tracked: int, serial: bool) -> str:
    """
    Summary generated from the computed result — never from the LLM.

    Reports the HIT rate (met or beat), which is what the score is based on, and
    breaks out beats vs in-line separately. The old wording said "met or beat N
    of N (beat rate 0.00)" — mixing the hit count with a beats-only rate, so a
    company that hit every target read as "beat rate 0.00" beside a 95 score.
    """
    met = beats + in_line
    hit_rate = (met / tracked) if tracked else 0.0
    parts = [
        f"Guidance credibility {value}/100.",
        f"Hit {met} of {tracked} guidance targets ({hit_rate:.0%}): "
        f"{beats} beat, {in_line} in-line, {misses} miss.",
    ]
    if serial:
        parts.append("Serial-miss risk flagged: a metric missed 2+ times.")
    return " ".join(parts)


class GuidanceAgent(BaseAgent):

    GUIDANCE_QUERIES = [
        "we expect guidance outlook forecast next quarter",
        "revenue guidance target range full year",
        "margin operating income EPS guidance",
        "we reiterate we raised we lowered guidance",
        "actual results reported revenue earnings",
        "beat miss exceeded fell short of expectations",
        "NIM cost-income ratio return on equity targets",
    ]

    def __init__(self, vs: VectorStore, model: str | None = None):
        super().__init__(vs, model)

    async def run(
        self,
        ticker: str, company: str,
        quarter: str, fiscal_year: int,
        periods_to_compare: list[tuple[str, int]],   # (quarter, year) pairs
    ) -> GuidanceSignal:
        logger.info(f"[GuidanceAgent] Running for {ticker} {quarter} {fiscal_year}")

        # Retrieval + citations: UNCHANGED. Pairs, not bare quarter labels.
        chunks = self.rag_retrieve(
            queries=self.GUIDANCE_QUERIES, ticker=ticker,
            periods=periods_to_compare,
            sections=["guidance", "financial_results", "prepared_remarks"],
            top_k_per_query=6,
        )
        citations = self.vs.as_citations(chunks[:8])
        periods_label = ", ".join(f"{q} {y}" for q, y in periods_to_compare)

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter Being Scored: {quarter} {fiscal_year}
Periods Being Compared: {periods_label}

=== EVIDENCE (guidance statements + actual results across quarters) ===
{self.format_evidence(chunks[:14]) or "No guidance evidence retrieved."}

Compare guidance given in PRIOR quarters vs ACTUAL results reported in
SUBSEQUENT quarters. Classify each trackable item's outcome.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            items = []
            for g in data.get("guidance_items", []):
                items.append(GuidanceItem(
                    metric=g.get("metric", "") or "",
                    period=g.get("period", "") or "",
                    guided_in=g.get("guided_in", "") or "",
                    guidance=g.get("guidance", "") or "",
                    actual=g.get("actual"),
                    outcome=(g.get("outcome", "") or "").strip().lower(),
                    miss_reason=g.get("miss_reason", "") or "",
                ))

            # Counts derived from the items IN CODE (no longer trusting the
            # model's separate count fields, which could disagree with its items).
            oc = Counter(i.outcome for i in items)
            _beats, _misses, _in_line = oc["beat"], oc["miss"], oc["in_line"]
            _withdrawals = oc["withdrew"]
            _tracked = _beats + _misses + _in_line

            # Nothing tracked → guidance was NOT assessed. score=None (not 0),
            # valid signal so the period still counts as scored. UNCHANGED.
            if _tracked == 0:
                logger.info(
                    f"[GuidanceAgent] {ticker} {quarter} {fiscal_year}: no trackable "
                    f"guidance — recording as not assessed."
                )
                return GuidanceSignal(
                    ticker=ticker, company=company,
                    quarter=quarter, fiscal_year=fiscal_year,
                    score=None,
                    guidance_items=[],
                    periods_tracked=0,
                    beats=0, misses=0, in_line=0, withdrawals=_withdrawals,
                    beat_rate=0.0,
                    serial_miss_risk=False,
                    recent_pattern=[],
                    summary="No formal guidance issued this period — nothing to "
                            "assess for credibility.",
                    citations=citations,
                    manifest=None,
                )

            # ── Deterministic score over the extracted outcomes ──────────────
            outcomes = [i.outcome for i in items
                        if i.outcome in ("beat", "in_line", "miss")]
            _ev = [EvidenceRef(chunk_id=c.chunk_id, quote=c.quote,
                               speaker=getattr(c, "speaker", "") or "")
                   for c in citations[:4]]
            result = score_guidance(GuidanceFeatures(
                outcomes=outcomes,
                is_specific=True,                 # trackable guidance was issued
                is_hedged=False,
                withdrew_guidance=_withdrawals > 0,
                outcome_evidence=_ev,
                language_evidence=_ev,
            ))

            _serial_metrics = _serial_miss_metrics(items)
            _beat_rate = round(_beats / _tracked, 3)   # beats-only (dashboard "Beat Rate")
            _recent_pattern = [i.outcome for i in items if i.outcome][:12]

            manifest = {
                **result.to_dict(),
                "counts": {"beats": _beats, "misses": _misses,
                           "in_line": _in_line, "withdrawals": _withdrawals},
                "serial_miss_metrics": _serial_metrics,
            }

            return GuidanceSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                score=result.value,               # computed, not judged
                guidance_items=items,
                periods_tracked=_tracked,
                beats=_beats, misses=_misses, in_line=_in_line,
                withdrawals=_withdrawals,
                beat_rate=_beat_rate,
                serial_miss_risk=bool(_serial_metrics),
                recent_pattern=_recent_pattern,
                summary=_guidance_summary(result.value, _beats, _in_line,
                                          _misses, _tracked, bool(_serial_metrics)),
                citations=citations,
                manifest=manifest,
            )

        except Exception as e:
            logger.error(f"[GuidanceAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
