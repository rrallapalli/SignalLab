"""
agents/guidance_agent.py
RAG → LLM → GuidanceSignal.
Retrieves PAST guidance statements and ACTUAL results, then scores credibility.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, GuidanceItem, GuidanceSignal
from agents.base import BaseAgent, safe_float, safe_int
from store.vector_store import VectorStore


SYSTEM_PROMPT = """You are a quantitative equity analyst scoring guidance credibility.

You receive evidence from multiple quarters: prior-quarter GUIDANCE STATEMENTS
and subsequent ACTUAL RESULTS. Your job is to compare what management said
vs what actually happened — not to summarise.

For each trackable guidance item, determine:
- Was it met, beaten, or missed?
- Was guidance withdrawn or substantially revised?

Return ONLY valid JSON:
{
  "score": 72,
  "guidance_items": [
    {
      "metric": "Revenue",
      "period": "Q3 2024",
      "guided_in": "Q2 2024",
      "guidance": "$20.0B–$21.0B",
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
      "miss_reason": "Higher-than-expected R&D spend and FX headwinds not anticipated in guidance"
    }
  ],
  "periods_tracked": 6,
  "beats": 3,
  "misses": 2,
  "in_line": 1,
  "withdrawals": 0,
  "serial_miss_risk": false,
  "summary": "Guidance credibility score of 72/100. Management met or exceeded revenue guidance in 5 of 6 periods but margin guidance has been consistently optimistic, missing in 4 of 6 periods. Serial miss risk on margin guidance is elevated."
}

Scoring (0–100):
- 85–100: >85% guidance met, reliable track record
- 65–84:  60–85% accuracy, generally credible
- 45–64:  Mixed, ~50% accuracy, some serial misses
- 25–44:  Frequent misses, credibility concerns
- 0–24:   Systematic over-promising

Do NOT return serial_miss_risk, beat_rate or recent_pattern — they are counted
from the guidance_items you return, not judged. Focus on getting each item's
metric, period, guidance, actual and outcome right.
"""


def _serial_miss_metrics(items: list) -> list[str]:
    """
    Metrics this company has missed repeatedly.

    ONE definition, used by both the agent flag and the YTD banner, which
    previously disagreed: the prompt said "3+ consecutive periods" while
    store.get_ytd_guidance() counted "2+ misses in total".

    "Consecutive" is not honestly computable here — GuidanceItem.period is a
    free-text label ("Q1 2026", "FY26 Q1", "next quarter"), so we cannot order
    periods reliably enough to claim consecutiveness. Counting total misses per
    metric is a claim the data actually supports; asserting consecutiveness from
    unordered labels would be the same fabrication in a different costume.
    """
    from collections import Counter
    counts = Counter(
        (i.metric or "").strip().lower()
        for i in items
        if (i.outcome or "").lower() == "miss" and (i.metric or "").strip()
    )
    return sorted(m for m, c in counts.items() if c >= 2)


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

        # Retrieve guidance statements across the compared periods. Pairs, not
        # bare quarter labels — otherwise this pulls guidance from every year on
        # record and scores the company's credibility against the wrong promises.
        chunks = self.rag_retrieve(
            queries=self.GUIDANCE_QUERIES, ticker=ticker,
            periods=periods_to_compare,
            sections=["guidance", "financial_results", "prepared_remarks"],
            top_k_per_query=6,
        )
        citations = self.vs.as_citations(chunks[:8])

        # Full periods in the prompt too — "Q1, Q4, Q1" tells the model nothing
        # about which year's guidance it is auditing against which year's results.
        periods_label = ", ".join(f"{q} {y}" for q, y in periods_to_compare)

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter Being Scored: {quarter} {fiscal_year}
Periods Being Compared: {periods_label}

=== EVIDENCE (guidance statements + actual results across quarters) ===
{self.format_evidence(chunks[:14]) or "No guidance evidence retrieved."}

Compare guidance given in PRIOR quarters vs ACTUAL results reported in SUBSEQUENT quarters.
Score guidance credibility based on the full history available.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            items = []
            for g in data.get("guidance_items", []):
                items.append(GuidanceItem(
                    metric=g.get("metric","") or "",
                    period=g.get("period","") or "",
                    guided_in=g.get("guided_in","") or "",
                    guidance=g.get("guidance","") or "",
                    actual=g.get("actual"),
                    outcome=g.get("outcome","") or "",
                    miss_reason=g.get("miss_reason","") or "",
                ))

            _beats   = safe_int(data.get("beats"))
            _misses  = safe_int(data.get("misses"))
            _in_line = safe_int(data.get("in_line"))
            _tracked = _beats + _misses + _in_line

            # Nothing tracked means guidance credibility was NOT assessed — the
            # company issued no trackable guidance this period (common for Indian
            # names giving qualitative commentary, not US-style point/range
            # guidance). This is a LEGITIMATE result, not a failure: return a
            # valid signal with score=None so the quarter still counts as scored
            # (the orchestrator only marks a period done when all four agents
            # succeed — raising here left the quarter permanently "incomplete"
            # and forced a full re-score of the other three agents every run).
            #
            # score=None (not 0) is deliberate: 0 is a verdict ("not credible"),
            # absence is not. The dashboard renders None as "No guidance issued".
            if _tracked == 0:
                logger.info(f"[GuidanceAgent] {ticker} {quarter} {fiscal_year}: no trackable guidance — recording as not assessed.")
                return GuidanceSignal(
                    ticker=ticker, company=company,
                    quarter=quarter, fiscal_year=fiscal_year,
                    score=None,
                    guidance_items=[],
                    periods_tracked=0,
                    beats=0, misses=0, in_line=0, withdrawals=0,
                    beat_rate=0.0,
                    serial_miss_risk=False,
                    recent_pattern=[],
                    summary=(data.get("summary") or
                             "No formal guidance issued this period — nothing to assess for credibility."),
                    citations=citations,
                )

            # Items exist but the model gave no usable score — that IS a failure
            # (a glitch, not an honest absence), so keep raising to retry.
            _score = safe_float(data.get("score"), None)
            if _score is None:
                raise ValueError("model returned no usable 'score'")

            _serial_metrics = _serial_miss_metrics(items)
            _recent_pattern = [i.outcome for i in items if i.outcome][:12]

            return GuidanceSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                score=_score,
                guidance_items=items,
                periods_tracked=safe_int(data.get("periods_tracked")),
                beats=_beats,
                misses=_misses,
                in_line=_in_line,
                withdrawals=safe_int(data.get("withdrawals")),
                # Division, not judgement — we hold both operands. Asking the
                # model for it invites a rate that contradicts its own counts.
                beat_rate=round(_beats / _tracked, 3),
                # Counted from the items, not asked of the model. The prompt
                # previously defined this as "same metric missed 3+ CONSECUTIVE
                # periods" while store.get_ytd_guidance() used "missed 2+ times
                # in total" — two different meanings for one concept, both shown
                # on screen. One definition now, in code: see _serial_miss_metrics.
                serial_miss_risk=bool(_serial_metrics),
                recent_pattern=_recent_pattern,
                summary=data.get("summary") or "",
                citations=citations,
            )
        except Exception as e:
            # Re-raised, not swallowed. score=50 published a fabricated
            # "middling credibility" verdict on a company whose guidance was
            # never actually assessed. The orchestrator catches this, records
            # the error, and leaves the signal None.
            logger.error(f"[GuidanceAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
