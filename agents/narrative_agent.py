"""
agents/narrative_agent.py
RAG → LLM → NarrativeSignal: theme-level QoQ shift detection.

Evidence retrieved across BOTH quarters to compare theme emphasis.

Theme tracking is market-aware (India macro themes apply to every company)
and sector-aware (a bank talks about NIMs and asset quality; an IT services
firm talks about deal TCV and attrition; a generic US-tech theme list like
"AI/ML, Cloud, China export controls" fits neither well). The sector is
auto-detected from the company name / ticker unless explicitly passed in.
"""

from __future__ import annotations

import unicodedata

from loguru import logger
from models import (Citation, NarrativeSignal, ThemeSignal, ThemeStatus,
                    normalize_quote_text, format_period)
from agents.base import BaseAgent, safe_float, safe_int
from agents.prior_lookup import match_prior
from store.vector_store import VectorStore


# -- India-wide macro themes -- tracked for every company regardless of sector
MARKET_THEMES: list[str] = [
    "RBI Monetary Policy / Repo Rate",
    "Inflation (CPI/WPI)",
    "INR Depreciation / Currency Impact",
    "Monsoon & Rural Demand",
    "Government Capex / Union Budget Policy",
    "GST / Tax Policy Changes",
    "FII/DII Flows",
    "Election / Policy Uncertainty",
    "Global Macro Spillover (US Fed, China slowdown)",
    "Capital Allocation (buyback / dividend / M&A)",
]

BASE_QUERIES: list[str] = [
    "demand growth outlook strategy",
    "margins costs pricing power",
    "risk regulatory competition",
    "capital allocation buyback dividend M&A",
    "interest rates inflation currency macro environment",
    "monsoon rural urban demand government policy budget",
]

# -- Sector taxonomy: label, tracked themes, and extra targeted RAG queries --
SECTOR_TAXONOMY: dict[str, dict] = {
    "banking_nbfc": {
        "label": "Banking / NBFC / Financial Services",
        "match": [
            "bank", "banking", "finance", "financial services", "nbfc", "housing finance",
            "hdfc", "icici", "kotak", "axis", "sbi", "bajaj finance", "bajaj finserv",
            "chola", "cholamandalam", "shriram", "muthoot", "mannapuram", "pnb", "canara",
            "indusind", "yes bank", "idfc", "au small finance", "bandhan",
        ],
        "themes": [
            "Net Interest Margin (NIM) / Spread Compression",
            "Asset Quality / NPAs (GNPA, NNPA)",
            "Credit Growth / Loan Book Growth",
            "CASA Ratio / Deposit Mobilization & Cost",
            "Provisioning / Credit Costs",
            "RBI Regulatory Action / Compliance",
            "Digital Lending / Fintech Competition",
            "Capital Adequacy (CAR)",
            "Unsecured Lending Risk",
        ],
        "queries": [
            "net interest margin NIM spread deposit cost",
            "asset quality NPA gross net slippages provisioning",
            "loan book credit growth disbursement CASA",
            "RBI regulatory guideline compliance capital adequacy",
        ],
    },
    "it_services": {
        "label": "IT Services / Technology",
        "match": [
            "infosys", "tcs", "tata consultancy", "wipro", "hcl", "tech mahindra",
            "ltimindtree", "mindtree", "mphasis", "persistent", "coforge", "cyient",
            "technologies", "information technology", "it services", "software",
        ],
        "themes": [
            "Deal Wins / TCV (Total Contract Value)",
            "Discretionary IT Spend",
            "Client Budget Cuts / Furloughs",
            "Attrition / Talent Costs",
            "GenAI Adoption & Deal Cannibalization",
            "Cloud & Digital Transformation",
            "USD-INR Currency Impact",
            "Vertical Demand (BFSI, Retail, Healthcare clients)",
            "Utilization / Margin Levers",
        ],
        "queries": [
            "deal wins TCV total contract value pipeline",
            "discretionary spend client budget furlough",
            "attrition talent cost utilization margin",
            "GenAI artificial intelligence cannibalization cloud digital",
        ],
    },
    "pharma_healthcare": {
        "label": "Pharmaceuticals / Healthcare",
        "match": [
            "pharma", "pharmaceutical", "labs", "laboratories", "healthcare", "sun pharma",
            "cipla", "dr reddy", "divi's", "divis", "lupin", "aurobindo", "biocon", "hospital",
            "torrent pharma", "zydus", "apollo hospitals", "fortis",
        ],
        "themes": [
            "US Generics Pricing Pressure",
            "USFDA Inspections / Compliance",
            "R&D Pipeline / New Launches",
            "Domestic Formulations Growth",
            "API / Raw Material Costs",
            "Patent Cliffs / Para IV Litigation",
            "Biosimilars",
        ],
        "queries": [
            "US generics pricing pressure launches",
            "USFDA inspection warning letter compliance plant",
            "R&D pipeline ANDA approval patent litigation",
            "domestic formulations India business growth",
        ],
    },
    "fmcg_consumer": {
        "label": "FMCG / Consumer",
        "match": [
            "hindustan unilever", "hul", "itc", "nestle", "britannia", "dabur", "marico",
            "godrej consumer", "colgate", "tata consumer", "varun beverages", "united spirits",
            "consumer goods", "fmcg",
        ],
        "themes": [
            "Rural Demand Recovery",
            "Urban Consumption Trends",
            "Input Cost Inflation (palm oil, crude derivatives)",
            "Volume Growth vs Price-led Growth",
            "Distribution Expansion (GT / MT / E-commerce / Quick Commerce)",
            "Premiumization",
        ],
        "queries": [
            "rural urban demand recovery consumption",
            "input cost inflation raw material palm oil",
            "volume growth price led growth premiumization",
            "distribution general trade modern trade e-commerce quick commerce",
        ],
    },
    "auto_ancillary": {
        "label": "Auto / Auto Ancillaries",
        "match": [
            "maruti", "tata motors", "mahindra", "bajaj auto", "hero motocorp", "eicher",
            "tvs motor", "ashok leyland", "motherson", "bosch", "exide", "amara raja",
            "motors", "automobile", "auto ancillary",
        ],
        "themes": [
            "EV Transition / EV Mix",
            "Semiconductor Availability",
            "Rural vs Urban Demand",
            "Commodity Costs (steel, aluminum)",
            "Export Demand",
            "Two-Wheeler vs PV vs CV Cycle",
        ],
        "queries": [
            "EV electric vehicle mix transition launch",
            "semiconductor chip shortage availability",
            "rural urban demand two wheeler passenger vehicle",
            "commodity cost steel aluminum export demand",
        ],
    },
    "metals_cement": {
        "label": "Metals, Mining & Cement",
        "match": [
            "tata steel", "jsw steel", "hindalco", "vedanta", "sail", "nmdc", "coal india",
            "ultratech", "ambuja", "acc", "shree cement", "jk cement", "steel", "cement", "mining",
        ],
        "themes": [
            "Commodity Price Cycle",
            "China Demand / Overcapacity",
            "Input Cost (coking coal, iron ore, pet coke)",
            "Capacity Utilization",
            "Infrastructure / Construction Demand",
            "Realizations vs Volumes",
        ],
        "queries": [
            "commodity price realization volume cycle",
            "China demand overcapacity export",
            "input cost coking coal iron ore capacity utilization",
            "infrastructure construction demand capex",
        ],
    },
    "energy_power": {
        "label": "Energy / Oil & Gas / Power",
        "match": [
            "reliance industries", "ongc", "oil india", "bpcl", "hpcl", "iocl", "gail",
            "ntpc", "power grid", "tata power", "adani power", "adani green", "adani energy",
            "jsw energy", "power", "energy", "oil", "gas", "refinery",
        ],
        "themes": [
            "Crude Oil Price Volatility",
            "Refining Margins (GRM)",
            "Renewable Energy Transition",
            "Power Demand Growth",
            "Regulatory / Tariff Changes",
            "Green Energy Capex",
        ],
        "queries": [
            "crude oil price refining margin GRM",
            "renewable solar wind energy transition capex",
            "power demand growth tariff regulatory",
            "green energy capex investment",
        ],
    },
    "telecom": {
        "label": "Telecom",
        "match": ["bharti airtel", "reliance jio", "vodafone idea", "vi ", "telecom", "airtel"],
        "themes": [
            "ARPU Growth",
            "Subscriber Additions / Churn",
            "5G Rollout & Capex",
            "Tariff Hikes",
            "AGR Dues / Regulatory",
        ],
        "queries": [
            "ARPU subscriber growth churn 5G",
            "tariff hike capex rollout",
            "AGR dues regulatory spectrum",
        ],
    },
    "infra_capital_goods": {
        "label": "Infrastructure / Capital Goods",
        "match": [
            "larsen", "l&t", "adani ports", "adani enterprises", "irb infra", "gmr",
            "siemens", "abb india", "cummins india", "bhel", "capital goods", "infrastructure",
            "engineering construction",
        ],
        "themes": [
            "Order Book / Order Inflow",
            "Execution Pace / Project Delays",
            "Government Capex (roads, railways, defense)",
            "Working Capital Cycle",
            "Raw Material Cost Pass-through",
        ],
        "queries": [
            "order book order inflow execution pace",
            "government capex roads railways defense infrastructure",
            "working capital raw material cost pass through",
        ],
    },
    "real_estate": {
        "label": "Real Estate",
        "match": ["dlf", "godrej properties", "oberoi realty", "prestige estates", "sobha", "real estate", "realty"],
        "themes": [
            "Pre-sales / Bookings",
            "Inventory Levels",
            "Launch Pipeline",
            "Housing Demand / Affordability",
            "Land Acquisition",
        ],
        "queries": [
            "pre-sales bookings launch pipeline",
            "inventory levels housing demand affordability",
            "land acquisition project execution",
        ],
    },
}

DEFAULT_SECTOR = "general"
GENERAL_THEMES = [
    "Demand Environment", "Pricing Power", "Margins", "Competition",
    "Capacity / Capex Plans", "Cost Inflation", "Regulatory Risk", "Innovation / New Products",
]


def _infer_sector(company: str, ticker: str) -> str:
    """Best-effort keyword match against company name / ticker. Falls back to 'general'."""
    text = f"{company} {ticker}".lower()
    for sector_key, meta in SECTOR_TAXONOMY.items():
        if any(kw in text for kw in meta["match"]):
            return sector_key
    return DEFAULT_SECTOR


def _theme_list_for(sector_key: str) -> list[str]:
    sector_themes = SECTOR_TAXONOMY.get(sector_key, {}).get("themes", GENERAL_THEMES)
    return sector_themes + MARKET_THEMES


def _queries_for(sector_key: str) -> list[str]:
    sector_queries = SECTOR_TAXONOMY.get(sector_key, {}).get("queries", [])
    return BASE_QUERIES + sector_queries


def _build_system_prompt(sector_key: str) -> str:
    sector_label = SECTOR_TAXONOMY.get(sector_key, {}).get("label", "General / Diversified")
    themes = _theme_list_for(sector_key)
    theme_block = "\n".join(f"- {t}" for t in themes)

    return f"""You are an equity research analyst detecting narrative shifts in management commentary
for an India-listed company in the **{sector_label}** sector.

You will receive evidence chunks from the CURRENT quarter only. For each theme,
count how prominently it features THIS quarter and score its sentiment. The
quarter-over-quarter shift is computed in code by comparing against the prior
quarter's stored counts — you neither see nor estimate the prior quarter.

Track these themes (sector-specific themes first, India-wide macro themes always apply),
and any others you observe that are clearly material to this company:

{theme_block}

For each theme:
- Count evidence mentions in THIS quarter
- Score sentiment (-1.0 to +1.0)
- Write one-line interpretation
(The theme's status and the accelerating/emerging/fading/newly_risky rollups are
computed in code from these counts and sentiment — do not classify them here.)

key_quotes MUST be copied VERBATIM, character for character, from the evidence
above. Do not paraphrase, tidy, shorten, or compose them. If you cannot find an
exact sentence supporting the theme, return an empty key_quotes list. Every quote
is checked against the source text and silently dropped if it does not appear
there — an invented quote will not reach the user, it will just cost the theme
its evidence.

Return ONLY valid JSON:
{{
  "themes": [
    {{
      "theme": "Net Interest Margin (NIM) / Spread Compression",
      "evidence_count_current": 12,
      "sentiment_current": -0.4,
      "interpretation": "Management flagged NIM compression from deposit repricing for the first time this quarter, versus a confident tone last quarter.",
      "key_quotes": ["We expect some NIM pressure as deposit costs catch up", "Margins should stabilize in the back half of the year"]
    }}
  ],
  "shift_summary": "Growth narrative remains intact but margin commentary turned cautious for the first time in several quarters, with deposit cost pressure now explicitly flagged as a near-term headwind."
}}

Do NOT return status, count_change, sentiment_change, or the
accelerating/emerging/fading/newly_risky rollups — those are computed in code
from the counts and sentiment you provide.
"""


# Bump when the classification rules below change (part of SIGNAL_VERSION disc.).
NARRATIVE_SCORING_VERSION = "1.0.0"


def _classify_theme(cur_n: int, prev_n: int,
                    cur_s: float, prev_s: float) -> tuple[ThemeStatus, str]:
    """
    Deterministic theme status from the counts and sentiment the LLM extracted.

    Replaces the model's own `status` label so the classification is a
    reproducible function of the evidence, not a judgement that can disagree
    with the counts shown beside it. Rules mirror the ones previously stated in
    the prompt. RESOLVED is not emitted — "a concern now explicitly removed"
    cannot be told from counts alone, so a vanished theme reads as FADING.
    """
    ratio = (cur_n / prev_n) if prev_n > 0 else float("inf")
    # newly negative this quarter -> newly_risky (checked first: a new negative
    # theme is a risk, not merely "emerging").
    if cur_s <= -0.2 and cur_s < prev_s:
        return ThemeStatus.NEWLY_RISKY, "sentiment turned negative vs prior"
    # present now, minimal/absent last quarter
    if prev_n <= 1 and cur_n >= 2:
        return ThemeStatus.EMERGING, "minimal/absent prior, present now"
    # dropped off, or >50% fewer mentions
    if cur_n == 0 or ratio <= 0.5:
        return ThemeStatus.FADING, "dropped off or >50% fewer mentions"
    # >50% more mentions AND positive sentiment
    if ratio >= 1.5 and cur_s > 0:
        return ThemeStatus.ACCELERATING, ">50% more mentions, positive sentiment"
    return ThemeStatus.STABLE, "no material change"


class NarrativeAgent(BaseAgent):

    def __init__(self, vs: VectorStore, model: str | None = None):
        super().__init__(vs, model)

    async def run(
        self,
        ticker: str, company: str,
        quarter: str, fiscal_year: int,
        prior_quarter: str,
        prior_year: int,
        sector: str | None = None,
        store=None,               # SignalStore, for the QoQ baseline lookup
    ) -> NarrativeSignal:
        sector_key = sector if sector in SECTOR_TAXONOMY else _infer_sector(company, ticker)
        sector_label = SECTOR_TAXONOMY.get(sector_key, {}).get("label", "General / Diversified")
        logger.info(f"[NarrativeAgent] Running for {ticker} {quarter} {fiscal_year} (sector={sector_label})")

        queries = _queries_for(sector_key)

        # CURRENT quarter only — the QoQ baseline comes from the prior quarter's
        # STORED counts (below), not from re-reading its chunks here.
        current_chunks = self.rag_retrieve(
            queries=queries, ticker=ticker, quarter=quarter, fiscal_year=fiscal_year,
            doc_types=["earnings_call", "press_release", "investor_presentation"],
            top_k_per_query=5,
        )

        citations = self.vs.as_citations(current_chunks[:8])

        user_prompt = f"""Company: {company} ({ticker})  |  Sector: {sector_label}
Quarter: {quarter} {fiscal_year}

=== EVIDENCE (this quarter) ===
{self.format_evidence(current_chunks[:12]) or "No evidence."}

For each theme, count how prominently it features THIS quarter and score its
sentiment. Do not estimate any prior quarter.
"""

        try:
            data = await self.llm_reason(_build_system_prompt(sector_key), user_prompt)

            # Belt and braces: an instruction is not a guarantee. Anything the
            # model calls a quote is checked against the actual retrieved text
            # and dropped if it isn't there. Citation.quote is extracted by code
            # from chunk.text and is verbatim by construction; key_quotes come
            # out of the model's JSON and are not.
            #
            # The check must reject PARAPHRASE without rejecting faithful
            # quotes that merely differ in TYPOGRAPHY. Two things were causing
            # real evidence to be thrown away:
            #
            #  1. Chunks were concatenated in RELEVANCE order, so a quote
            #     spanning two adjacent passages was split by an unrelated chunk
            #     and became unfindable. They are joined in document order
            #     (doc_id, char_start), which restores contiguity.
            #  2. PDFs carry curly quotes, en/em dashes and ligatures ("ﬁnal");
            #     the model emits ASCII equivalents, so a byte-exact match
            #     failed on an otherwise perfect quote.
            #
            # normalize_quote_text handles (2) and is THE shared definition of
            # verbatim — validate_run audits stored quotes with the same
            # function. An earlier version of this check also ignored
            # punctuation, which let through quotes where the model had dropped
            # a comma; the validator then flagged them as appearing in no
            # source document. A quote that needs punctuation ignored in order
            # to match is not verbatim, so it is dropped here instead.
            _ordered_chunks = sorted(
                (c for c, _ in current_chunks),
                key=lambda c: (getattr(c, "doc_id", "") or "",
                               getattr(c, "char_start", 0) or 0),
            )
            _evidence_norm = normalize_quote_text(" ".join(
                (getattr(c, "text", "") or "") for c in _ordered_chunks
            ))

            def _verified_quotes(raw: list) -> list[str]:
                out = []
                for q in (raw or [])[:2]:
                    if not isinstance(q, str):
                        continue
                    needle = normalize_quote_text(q)
                    if len(needle) < 15:
                        continue
                    if needle in _evidence_norm:
                        out.append(q)
                        continue
                    logger.warning(
                        f"[NarrativeAgent] Dropped unverifiable key_quote "
                        f"(not found in source text): {q[:70]!r}"
                    )
                return out

            # ── QoQ baseline from the prior quarter's STORED counts ──────────
            # No re-ingestion: read the prior period's stored narrative signal
            # and match themes by name. If it was never scored, there is no
            # baseline — flagged in the manifest, not silently treated as "all
            # new" (which would paint every theme EMERGING).
            _prior_themes: dict[str, dict] = {}
            _prior_available = False
            if store is not None:
                _prow = store.get_signal_row("narrative", ticker, prior_quarter, prior_year)
                if _prow and _prow.get("themes"):
                    _prior_available = True
                    for pt in _prow["themes"]:
                        nm = pt.get("theme", "")
                        if nm:
                            _prior_themes[nm] = {
                                "count": safe_int(pt.get("evidence_count_current")),
                                "sentiment": safe_float(pt.get("sentiment_current")),
                            }
            _prior_names = list(_prior_themes.keys())

            themes = []
            _manifest_items = []
            for t in data.get("themes", []):
                _theme = t.get("theme","") or ""
                if not _theme:
                    continue
                _cur_n = safe_int(t.get("evidence_count_current"))
                _cur_s = safe_float(t.get("sentiment_current"))

                # Prior counts from the matched stored theme (0 if unmatched or
                # no prior signal). Status is a code function of both.
                _match  = match_prior(_theme, _prior_names) if _prior_names else None
                _pinfo  = _prior_themes.get(_match, {}) if _match else {}
                _prev_n = _pinfo.get("count", 0)
                _prev_s = _pinfo.get("sentiment", 0.0)

                status, _rule = _classify_theme(_cur_n, _prev_n, _cur_s, _prev_s)

                themes.append(ThemeSignal(
                    theme=_theme,
                    status=status,
                    evidence_count_current=_cur_n,
                    evidence_count_previous=_prev_n,
                    count_change=_cur_n - _prev_n,
                    sentiment_current=_cur_s,
                    sentiment_previous=_prev_s,
                    sentiment_change=round(_cur_s - _prev_s, 3),
                    interpretation=t.get("interpretation","") or "",
                    key_quotes=_verified_quotes(t.get("key_quotes")),
                ))
                _manifest_items.append({
                    "theme": _theme, "status": status.value, "rule": _rule,
                    "current": _cur_n, "previous": _prev_n,
                    "count_change": _cur_n - _prev_n,
                    "sentiment_current": _cur_s, "sentiment_previous": _prev_s,
                    "matched_prior": _match,
                })

            # Rollups derived from the computed statuses, not from the model.
            _accel = [t.theme for t in themes if t.status == ThemeStatus.ACCELERATING]
            _emerg = [t.theme for t in themes if t.status == ThemeStatus.EMERGING]
            _fade  = [t.theme for t in themes if t.status == ThemeStatus.FADING]
            _risky = [t.theme for t in themes if t.status == ThemeStatus.NEWLY_RISKY]
            _pos, _neg = len(_accel) + len(_emerg), len(_fade) + len(_risky)
            _overall = ("positive" if _pos > _neg else
                        "negative" if _neg > _pos else
                        "mixed" if (_pos and _neg) else "neutral")

            # shift_summary is narrated from the COMPUTED status rollups (the
            # model no longer sees the prior quarter, so its own prose could not
            # describe a shift). Grounded, hash-cached, deterministic fallback.
            _sdrivers = []
            if _accel: _sdrivers.append(f"accelerating: {', '.join(_accel)}")
            if _emerg: _sdrivers.append(f"newly emerging: {', '.join(_emerg)}")
            if _risky: _sdrivers.append(f"turning risky: {', '.join(_risky)}")
            if _fade:  _sdrivers.append(f"fading: {', '.join(_fade)}")
            if not _sdrivers:
                _sdrivers = ["no material theme shifts versus the prior quarter"]
            _sfallback = f"Narrative shift is {_overall}. " + "; ".join(_sdrivers) + "."
            _nfacts = {
                "signal": "narrative shift (how management's key themes moved vs the prior quarter)",
                "headline": f"Overall narrative shift: {_overall}",
                "drivers": _sdrivers,
                "fallback": _sfallback,
            }

            return NarrativeSignal(
                ticker=ticker, company=company,
                quarter=quarter, fiscal_year=fiscal_year,
                themes=themes,
                accelerating=_accel, emerging=_emerg,
                fading=_fade, newly_risky=_risky,
                overall_shift=_overall,
                shift_summary=_sfallback,
                citations=citations,
                manifest={
                    "signal": "narrative",
                    "scorer_version": NARRATIVE_SCORING_VERSION,
                    "prior_period": format_period(prior_quarter, prior_year),
                    "prior_available": _prior_available,
                    "themes": _manifest_items,
                    "overall_shift": _overall,
                    "narration": _nfacts,
                },
            )
        except Exception as e:
            # Re-raised, not swallowed into a placeholder signal. Returning a
            # NarrativeSignal here wrote a row into DuckDB for a quarter that was
            # never analysed; it then appeared in the trend table and All Periods
            # Summary as though it were a reading. The orchestrator already
            # catches this per agent, records the error, and leaves it None.
            logger.error(f"[NarrativeAgent] Failed for {ticker} {quarter} {fiscal_year}: {e}")
            raise
