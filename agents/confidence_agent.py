"""
agents/confidence_agent.py
RAG → LLM → ManagementConfidenceScore (0–10).

Evidence retrieved from: prepared remarks, Q&A, management commentary
across current and prior quarter for delta calculation.
"""

from __future__ import annotations

from loguru import logger
from models import Citation, ConfidenceSignal
from agents.base import BaseAgent, safe_float
from store.vector_store import VectorStore


SYSTEM_PROMPT = """You are a senior equity analyst scoring management confidence.

You will receive EVIDENCE chunks retrieved from earnings calls, press releases,
and management commentary. Your job is NOT to summarise — your job is to score
and explain WHY the score changed.

Score management confidence on a 0–10 scale (10 = highest confidence).
Also score six sub-dimensions (0–10 each):
- confidence_level:    certainty vs hedging language
- uncertainty_level:   explicit uncertainty signals (lower = more uncertain, but report as 0–10 inverted: 10=low uncertainty)
- defensiveness:       reactive/justifying tone (10 = not defensive at all)
- specificity:         concrete numbers vs vague language
- consistency:         alignment with prior statements
- forward_strength:    positive forward-looking signals

Return ONLY valid JSON. No markdown. No explanation outside the JSON.

Required schema:
{
  "score": 7.2,
  "confidence_level": 7.0,
  "uncertainty_level": 6.0,
  "defensiveness": 8.0,
  "specificity": 6.5,
  "consistency": 7.5,
  "forward_strength": 7.2,
  "tone": "cautious",
  "drivers": [
    "More cautious language around China demand with 'monitoring closely' replacing prior 'strong growth'",
    "Guidance range widened from $500M band to $800M band indicating higher uncertainty",
    "CEO used 'challenging' 4 times vs 1 time last quarter"
  ],
  "summary": "Management Confidence Score declined from 7.8 to 7.2 QoQ driven by increased hedging on China demand, wider guidance ranges, and more defensive language around pricing power."
}

CRITICAL: The summary must follow this format:
'[Score dimension] [direction] from [prev] to [curr] [QoQ/YoY] because [specific evidence-backed drivers].'
Not a transcript summary. A signal.
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
        prior_quarter: str,
        prior_year: int,
    ) -> ConfidenceSignal:
        logger.info(f"[ConfidenceAgent] Running for {ticker} {quarter} {fiscal_year}")

        # Retrieve evidence from current AND prior quarter
        current_chunks = self.rag_retrieve(
            queries=self.EVIDENCE_QUERIES,
            ticker=ticker,
            quarter=quarter,
            fiscal_year=fiscal_year,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            management_only=True,
            top_k_per_query=6,
        )
        prior_chunks = self.rag_retrieve(
            queries=self.EVIDENCE_QUERIES,
            ticker=ticker,
            quarter=prior_quarter,
            fiscal_year=prior_year,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            management_only=True,
            top_k_per_query=5,
        )

        current_evidence = self.format_evidence(current_chunks[:10])
        prior_evidence = self.format_evidence(prior_chunks[:8])

        citations = self.vs.as_citations(current_chunks[:8])

        user_prompt = f"""Company: {company} ({ticker})
Current Quarter: {quarter} {fiscal_year}
Prior Quarter: {prior_quarter} {prior_year}

=== CURRENT QUARTER EVIDENCE ===
{current_evidence or "No current quarter evidence retrieved."}

=== PRIOR QUARTER EVIDENCE (for QoQ comparison) ===
{prior_evidence or "No prior quarter evidence retrieved."}

Score management confidence for the CURRENT quarter versus the prior quarter.
Extract specific language changes that drove any score movement.

IMPORTANT — do NOT state, guess, or narrate a numeric score for the prior
quarter. You have not been given it. Describe the change QUALITATIVELY, from
the language itself ("hedging replaced firm commitments on pricing"), never as
"increased from 7.2 to 7.5". The prior score is looked up from the database and
the delta is computed arithmetically; any number you invent will contradict the
figure shown next to your summary.
"""

        try:
            data = await self.llm_reason(SYSTEM_PROMPT, user_prompt)

            # A signal with no score is not a signal. Defaulting to 5.0 here
            # would publish a plausible mid-range number that no evidence
            # supports; raising lets the orchestrator record an error and skip.
            score = safe_float(data.get("score"), None)
            if score is None:
                raise ValueError("model returned no usable 'score'")

            return ConfidenceSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                score=score,
                # NOT from the LLM. It has no access to the stored prior score,
                # so anything it returns here is invented — and the dashboard
                # computes the real delta from actual scores anyway, which is how
                # "increased from 7.2 to 7.5" ended up printed next to "— 0.0".
                previous_score=None,
                change=None,
                confidence_level=safe_float(data.get("confidence_level"), 5.0),
                uncertainty_level=safe_float(data.get("uncertainty_level"), 5.0),
                defensiveness=safe_float(data.get("defensiveness"), 5.0),
                specificity=safe_float(data.get("specificity"), 5.0),
                consistency=safe_float(data.get("consistency"), 5.0),
                forward_strength=safe_float(data.get("forward_strength"), 5.0),
                tone=data.get("tone") or "neutral",
                drivers=data.get("drivers") or [],
                summary=data.get("summary") or "",
                citations=citations,
            )
        except Exception as e:
            # Deliberately re-raised, not swallowed into a placeholder signal.
            # Returning score=5.0 wrote a fabricated mid-range score into DuckDB
            # that then plotted on the trend chart and fed delta maths — the
            # "Scoring failed" summary underneath it fooled nobody looking at a
            # big green gauge. The orchestrator already catches this per agent,
            # records the error, and leaves the signal None. Let it.
            logger.error(f"[ConfidenceAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
