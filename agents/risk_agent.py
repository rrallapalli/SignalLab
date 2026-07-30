"""
agents/risk_agent.py
RAG → LLM → RiskSignal: detects newly material or escalating risks.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, RiskItem, RiskSeverity, RiskSignal, RiskStatus
from agents.base import BaseAgent, safe_int
from store.vector_store import VectorStore


SYSTEM_PROMPT = """You are a risk analyst at a hedge fund EXTRACTING risk evidence.

You receive evidence from the CURRENT and PRIOR quarter. Identify material risks
and, for each, count verbatim mentions in each quarter and assess severity. You do
NOT classify a risk's status (newly_material / escalating / diminishing) — that is
computed in code from the mention counts you return.

Return ONLY valid JSON:
{
  "risks": [
    {
      "risk": "deposit competition",
      "severity": "high",
      "mention_count_current": 11,
      "mention_count_previous": 3,
      "evidence": "Mentioned far more this quarter; CEO flagged it as a key NIM headwind for H2.",
      "key_quotes": [
        "Deposit competition has intensified materially and is now our primary margin headwind"
      ]
    }
  ],
  "summary": "Risk profile deteriorated: deposit competition emerged as a newly material NIM headwind and China export exposure was quantified for the first time at $2.1B."
}

severity is YOUR assessment of how damaging the risk is (content-based — a single
mention of "$2.1B at risk" can be high). One of: critical | high | medium | low.
Do NOT return status, count_change, new_risks, escalating, diminishing, or
overall_risk_direction — those are computed in code from the mention counts.
"""


# Bump when the classification rules below change (part of SIGNAL_VERSION disc.).
RISK_SCORING_VERSION = "1.0.0"
ESCALATION_RATIO = 1.5


def _classify_risk_status(cur: int, prev: int) -> tuple[RiskStatus, str]:
    """
    Deterministic risk status from mention counts. Vocabulary already matches
    RiskStatus. Replaces the model's own status label so it cannot disagree with
    the counts shown beside it. RESOLVED is not emitted from counts (a risk that
    vanished reads as DIMINISHING); severity remains the model's assessment.
    """
    if prev == 0 and cur > 0:
        return RiskStatus.NEWLY_MATERIAL, "absent prior quarter, present now"
    if cur == 0 and prev > 0:
        return RiskStatus.DIMINISHING, "no longer mentioned"
    if prev > 0 and cur / prev >= ESCALATION_RATIO:
        return RiskStatus.ESCALATING, ">=50% more mentions than prior"
    if prev > 0 and cur / prev <= 1 / ESCALATION_RATIO:
        return RiskStatus.DIMINISHING, "fewer mentions than prior"
    return RiskStatus.STABLE, "no material change"


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
            _manifest_items = []
            for r in data.get("risks", []):
                _name = r.get("risk","") or ""
                if not _name:
                    continue
                _cur  = safe_int(r.get("mention_count_current"))
                _prev = safe_int(r.get("mention_count_previous"))
                # Status computed in code from the mention deltas.
                status, _rule = _classify_risk_status(_cur, _prev)
                # Severity kept as the model's content assessment.
                try: severity = RiskSeverity(r.get("severity") or "medium")
                except ValueError: severity = RiskSeverity.MEDIUM
                risks.append(RiskItem(
                    risk=_name,
                    status=status, severity=severity,
                    mention_count_current=_cur,
                    mention_count_previous=_prev,
                    count_change=_cur - _prev,
                    evidence=r.get("evidence","") or "",
                    key_quotes=(r.get("key_quotes") or [])[:2],
                ))
                _manifest_items.append({
                    "risk": _name, "status": status.value, "rule": _rule,
                    "severity": severity.value,
                    "current": _cur, "previous": _prev, "count_change": _cur - _prev,
                })

            # Rollups derived from the computed statuses, not from the model.
            _new = [x.risk for x in risks if x.status == RiskStatus.NEWLY_MATERIAL]
            _esc = [x.risk for x in risks if x.status == RiskStatus.ESCALATING]
            _dim = [x.risk for x in risks if x.status in
                    (RiskStatus.DIMINISHING, RiskStatus.RESOLVED)]
            _up, _down = len(_new) + len(_esc), len(_dim)
            _direction = ("increasing" if _up > _down else
                          "decreasing" if _down > _up else "stable")

            return RiskSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                risks=risks,
                new_risks=_new, escalating=_esc, diminishing=_dim,
                overall_risk_direction=_direction,
                summary=data.get("summary") or "",
                citations=citations,
                manifest={
                    "signal": "risk",
                    "scorer_version": RISK_SCORING_VERSION,
                    "risks": _manifest_items,
                    "overall_risk_direction": _direction,
                },
            )
        except Exception as e:
            # Re-raised, not swallowed. A stored RiskSignal with an empty
            # overall_risk_direction is a row asserting a quarter was assessed
            # for risk when it wasn't. The orchestrator handles agent failure.
            logger.error(f"[RiskAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
