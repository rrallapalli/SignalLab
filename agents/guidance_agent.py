"""
agents/guidance_agent.py
RAG → LLM → GuidanceSignal.
Retrieves PAST guidance statements and ACTUAL results, then scores credibility.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, GuidanceItem, GuidanceSignal
from agents.base import BaseAgent
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
  "beat_rate": 0.50,
  "serial_miss_risk": false,
  "recent_pattern": ["beat","miss","beat","in_line","miss","beat"],
  "summary": "Guidance credibility score of 72/100. Management met or exceeded revenue guidance in 5 of 6 periods but margin guidance has been consistently optimistic, missing in 4 of 6 periods. Serial miss risk on margin guidance is elevated."
}

Scoring (0–100):
- 85–100: >85% guidance met, reliable track record
- 65–84:  60–85% accuracy, generally credible
- 45–64:  Mixed, ~50% accuracy, some serial misses
- 25–44:  Frequent misses, credibility concerns
- 0–24:   Systematic over-promising

serial_miss_risk = true if the same metric has been missed 3+ consecutive periods.
"""


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

    def __init__(self, vs: VectorStore):
        super().__init__(vs)

    async def run(
        self,
        ticker: str, company: str,
        quarter: str, fiscal_year: int,
        quarters_to_compare: list[str],   # all quarters in store
    ) -> GuidanceSignal:
        logger.info(f"[GuidanceAgent] Running for {ticker} {quarter} {fiscal_year}")

        # Retrieve guidance statements from ALL available quarters
        chunks = self.rag_retrieve(
            queries=self.GUIDANCE_QUERIES, ticker=ticker,
            quarters=quarters_to_compare,
            sections=["guidance", "financial_results", "prepared_remarks"],
            top_k_per_query=6,
        )
        citations = self.vs.as_citations(chunks[:8])

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter Being Scored: {quarter} {fiscal_year}
All Available Quarters: {', '.join(quarters_to_compare)}

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
                    metric=g.get("metric",""),
                    period=g.get("period",""),
                    guided_in=g.get("guided_in",""),
                    guidance=g.get("guidance",""),
                    actual=g.get("actual"),
                    outcome=g.get("outcome",""),
                    miss_reason=g.get("miss_reason",""),
                ))

            return GuidanceSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                score=float(data.get("score", 50)),
                guidance_items=items,
                periods_tracked=int(data.get("periods_tracked", 0)),
                beats=int(data.get("beats", 0)),
                misses=int(data.get("misses", 0)),
                in_line=int(data.get("in_line", 0)),
                withdrawals=int(data.get("withdrawals", 0)),
                beat_rate=float(data.get("beat_rate", 0.5)),
                serial_miss_risk=bool(data.get("serial_miss_risk", False)),
                recent_pattern=data.get("recent_pattern", []),
                summary=data.get("summary", ""),
                citations=citations,
            )
        except Exception as e:
            logger.error(f"[GuidanceAgent] Failed: {e}")
            return GuidanceSignal(
                ticker=ticker, company=company, quarter=quarter,
                fiscal_year=fiscal_year, score=50,
                summary=f"Scoring failed: {str(e)}", citations=citations,
            )
