"""
agents/narrative_agent.py
RAG → LLM → NarrativeSignal: theme-level QoQ shift detection.

Evidence retrieved across BOTH quarters to compare theme emphasis.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, NarrativeSignal, ThemeSignal, ThemeStatus
from agents.base import BaseAgent
from store.vector_store import VectorStore


SYSTEM_PROMPT = """You are an equity research analyst detecting narrative shifts in management commentary.

You will receive evidence chunks from the CURRENT quarter and the PRIOR quarter.
Your task is to identify how management's narrative around key themes has CHANGED.

Track these themes (and any others you observe):
AI/ML, Cloud, Pricing Power, Margins, China, Competition, Hiring/Headcount,
Regulatory Risk, Supply Chain, Consumer Demand, Innovation/R&D,
Capital Allocation, Macro Environment, Cost Cutting, ESG/Sustainability,
Interest Rates, Geopolitics, M&A, Digital Transformation.

For each theme:
- Count evidence mentions in current vs prior quarter
- Score sentiment (–1.0 to +1.0)
- Classify status: accelerating | emerging | stable | fading | newly_risky | resolved
- Write one-line interpretation

Return ONLY valid JSON:
{
  "themes": [
    {
      "theme": "AI demand",
      "status": "accelerating",
      "evidence_count_current": 18,
      "evidence_count_previous": 9,
      "count_change": 9,
      "sentiment_current": 0.85,
      "sentiment_previous": 0.70,
      "sentiment_change": 0.15,
      "interpretation": "AI-related demand discussion doubled QoQ; management now citing AI as primary growth driver vs secondary tailwind last quarter.",
      "key_quotes": ["We are seeing extraordinary demand for our AI products", "AI is becoming the primary lens through which customers evaluate our platform"]
    }
  ],
  "accelerating": ["AI demand", "Cloud"],
  "emerging": ["Sovereign AI", "Edge computing"],
  "fading": ["Supply chain disruptions", "Cost optimization"],
  "newly_risky": ["China export controls"],
  "overall_shift": "positive",
  "shift_summary": "Narrative has shifted decisively toward AI-led growth while legacy cost pressures are fading. China remains a wildcard with newly material export control risk."
}

Status rules:
- accelerating: >50% more mentions AND positive sentiment
- emerging: present this quarter but minimal/absent last quarter
- fading: >50% fewer mentions or dropped off
- newly_risky: newly negative or newly prominent in risk context
- resolved: was a concern, now explicitly addressed/removed
- stable: no material change
"""


class NarrativeAgent(BaseAgent):

    EVIDENCE_QUERIES = [
        "AI cloud demand growth strategy",
        "China macro geopolitics pricing competition",
        "margins costs headcount hiring",
        "risk supply chain regulatory",
        "innovation product pipeline R&D",
        "customer demand consumer spend",
        "interest rates inflation macro environment",
        "capital allocation buyback dividend M&A",
    ]

    def __init__(self, vs: VectorStore):
        super().__init__(vs)

    async def run(
        self,
        ticker: str, company: str,
        quarter: str, fiscal_year: int,
        prior_quarter: str,
    ) -> NarrativeSignal:
        logger.info(f"[NarrativeAgent] Running for {ticker} {quarter} {fiscal_year}")

        current_chunks = self.rag_retrieve(
            queries=self.EVIDENCE_QUERIES, ticker=ticker, quarter=quarter,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            top_k_per_query=5,
        )
        prior_chunks = self.rag_retrieve(
            queries=self.EVIDENCE_QUERIES, ticker=ticker, quarter=prior_quarter,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            top_k_per_query=5,
        )

        citations = self.vs.as_citations(current_chunks[:8])

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter: {quarter} {fiscal_year}  |  Prior Quarter: {prior_quarter}

=== CURRENT QUARTER EVIDENCE ===
{self.format_evidence(current_chunks[:12]) or "No current evidence."}

=== PRIOR QUARTER EVIDENCE ===
{self.format_evidence(prior_chunks[:10]) or "No prior evidence."}

Identify theme-level narrative shifts between these two quarters.
Count evidence mentions, assess sentiment, and classify each theme's trajectory.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            themes = []
            for t in data.get("themes", []):
                status_str = t.get("status","stable")
                try: status = ThemeStatus(status_str)
                except: status = ThemeStatus.STABLE
                themes.append(ThemeSignal(
                    theme=t.get("theme",""),
                    status=status,
                    evidence_count_current=int(t.get("evidence_count_current",0)),
                    evidence_count_previous=int(t.get("evidence_count_previous",0)),
                    count_change=int(t.get("count_change",0)),
                    sentiment_current=float(t.get("sentiment_current",0.0)),
                    sentiment_previous=float(t.get("sentiment_previous",0.0)),
                    sentiment_change=float(t.get("sentiment_change",0.0)),
                    interpretation=t.get("interpretation",""),
                    key_quotes=t.get("key_quotes",[])[:2],
                ))

            return NarrativeSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                themes=themes,
                accelerating=data.get("accelerating",[]),
                emerging=data.get("emerging",[]),
                fading=data.get("fading",[]),
                newly_risky=data.get("newly_risky",[]),
                overall_shift=data.get("overall_shift","neutral"),
                shift_summary=data.get("shift_summary",""),
                citations=citations,
            )
        except Exception as e:
            logger.error(f"[NarrativeAgent] Failed: {e}")
            return NarrativeSignal(
                ticker=ticker, company=company, quarter=quarter,
                fiscal_year=fiscal_year, shift_summary=f"Signal generation failed: {str(e)}",
                citations=citations,
            )
