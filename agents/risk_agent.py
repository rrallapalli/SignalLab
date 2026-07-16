"""
agents/risk_agent.py
RAG → LLM → RiskSignal: detects newly material or escalating risks.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, RiskItem, RiskSeverity, RiskSignal, RiskStatus
from agents.base import BaseAgent, safe_int
from store.vector_store import VectorStore


SYSTEM_PROMPT = """You are a risk analyst at a hedge fund. Detect and score emerging risks.

You receive evidence from the CURRENT and PRIOR quarter.
Your job is NOT to list all risks — it is to identify risks that are:
  - NEW this quarter (not mentioned or minor last quarter)
  - ESCALATING (mentioned significantly more or with more severity)
  - DIMINISHING (explicitly resolved or mentioned less)

For each risk item, count verbatim mentions and assess severity.

Return ONLY valid JSON:
{
  "risks": [
    {
      "risk": "deposit competition",
      "status": "newly_material",
      "severity": "high",
      "mention_count_current": 11,
      "mention_count_previous": 3,
      "count_change": 8,
      "evidence": "Mentioned 11 times this quarter vs 3 times last quarter; CEO flagged it as a key NIM headwind for H2.",
      "key_quotes": [
        "Deposit competition has intensified materially and is now our primary margin headwind",
        "We're seeing peers offering 50-75bps higher rates on savings products"
      ]
    },
    {
      "risk": "China export controls",
      "status": "escalating",
      "severity": "high",
      "mention_count_current": 8,
      "mention_count_previous": 2,
      "count_change": 6,
      "evidence": "Export control risk escalated from operational note to strategic concern; management added specific revenue exposure ($2.1B at risk).",
      "key_quotes": ["We now estimate $2.1 billion of revenue at risk from additional export restrictions"]
    }
  ],
  "new_risks": ["deposit competition"],
  "escalating": ["China export controls", "regulatory capital requirements"],
  "diminishing": ["supply chain disruptions", "raw material costs"],
  "overall_risk_direction": "increasing",
  "summary": "Risk profile has deteriorated this quarter. Deposit competition has emerged as a newly material NIM headwind while China export control exposure has been quantified for the first time at $2.1B."
}

Severity: critical | high | medium | low
Status: newly_material | escalating | stable | diminishing | resolved
overall_risk_direction: increasing | stable | decreasing
"""


class RiskAgent(BaseAgent):

    RISK_QUERIES = [
        "risk concern headwind challenge",
        "regulatory risk compliance legal",
        "competition competitive pricing pressure",
        "China geopolitical macro risk",
        "interest rate deposit funding cost",
        "supply chain input cost inflation",
        "customer churn attrition demand weakness",
        "litigation lawsuit enforcement action",
        "FX currency exposure hedge",
        "credit risk default delinquency loan loss",
    ]

    def __init__(self, vs: VectorStore, model: str | None = None):
        super().__init__(vs, model)

    async def run(
        self,
        ticker: str, company: str,
        quarter: str, fiscal_year: int,
        prior_quarter: str,
        prior_year: int,
    ) -> RiskSignal:
        logger.info(f"[RiskAgent] Running for {ticker} {quarter} {fiscal_year}")

        current_chunks = self.rag_retrieve(
            queries=self.RISK_QUERIES, ticker=ticker, quarter=quarter, fiscal_year=fiscal_year,
            sections=["risk_factors","prepared_remarks","qa_session"],
            top_k_per_query=5,
        )
        prior_chunks = self.rag_retrieve(
            queries=self.RISK_QUERIES, ticker=ticker, quarter=prior_quarter, fiscal_year=prior_year,
            sections=["risk_factors","prepared_remarks","qa_session"],
            top_k_per_query=5,
        )

        citations = self.vs.as_citations(current_chunks[:8])

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter: {quarter} {fiscal_year}  |  Prior Quarter: {prior_quarter} {prior_year}

=== CURRENT QUARTER RISK EVIDENCE ===
{self.format_evidence(current_chunks[:12]) or "No current risk evidence."}

=== PRIOR QUARTER RISK EVIDENCE ===
{self.format_evidence(prior_chunks[:10]) or "No prior risk evidence."}

Identify risks that are NEW, ESCALATING, or DIMINISHING between these two quarters.
Count mentions and assess severity. Focus on material changes.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            risks = []
            for r in data.get("risks", []):
                try: status = RiskStatus(r.get("status") or "stable")
                except: status = RiskStatus.STABLE
                try: severity = RiskSeverity(r.get("severity") or "medium")
                except: severity = RiskSeverity.MEDIUM
                risks.append(RiskItem(
                    risk=r.get("risk","") or "",
                    status=status, severity=severity,
                    mention_count_current=safe_int(r.get("mention_count_current")),
                    mention_count_previous=safe_int(r.get("mention_count_previous")),
                    count_change=safe_int(r.get("count_change")),
                    evidence=r.get("evidence","") or "",
                    key_quotes=(r.get("key_quotes") or [])[:2],
                ))

            return RiskSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                risks=risks,
                new_risks=data.get("new_risks") or [],
                escalating=data.get("escalating") or [],
                diminishing=data.get("diminishing") or [],
                overall_risk_direction=data.get("overall_risk_direction") or "stable",
                summary=data.get("summary") or "",
                citations=citations,
            )
        except Exception as e:
            # Re-raised, not swallowed. A stored RiskSignal with an empty
            # overall_risk_direction is a row asserting a quarter was assessed
            # for risk when it wasn't. The orchestrator handles agent failure.
            logger.error(f"[RiskAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
