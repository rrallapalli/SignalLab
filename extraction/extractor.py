"""
extraction/extractor.py — ONE LLM call per ticker-quarter.

Replaces four separate retrieve-and-reason passes with a single extraction over
the quarter's chunks. The chunk text (which dominates the token bill) is sent
through the model once. Output is atomic, source-anchored features; every quote
is verified against its chunk before it is allowed to influence a score.

Pipeline:
    retrieved chunks ─► extract() ─► FeatureBundle ─► score_*()  (pure, cached)
                         (1 call)

Seams you wire to your repo:
  * Completer  — adapt to your OpenAI client (OpenAICompleter provided).
  * chunks     — pass dicts: {"chunk_id","text","doc_id","page","speaker"}.
  * prior QoQ  — attach previous-quarter counts via attach_prior() from storage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

from scoring import (
    EvidenceRef,
    ConfidenceFeatures, NarrativeFeatures, GuidanceFeatures, RiskFeatures,
    score_confidence, score_narrative, score_guidance, score_risk,
)

TONE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_ANALYST_MARKERS = ("analyst", "operator", "moderator")


# --------------------------------------------------------------------------- #
# LLM seam
# --------------------------------------------------------------------------- #
class Completer(Protocol):
    """Your pinned LLM client. Must be deterministic (seed + temperature=0)."""
    def __call__(self, system: str, user: str) -> str: ...


class OpenAICompleter:
    """
    Ready-to-use adapter for your existing OpenAI setup. Lazily imports openai
    so this module loads without it. Captures system_fingerprint for the
    audit manifest.
    """
    def __init__(self, model: str, seed: int = 7, temperature: float = 0.0):
        from openai import OpenAI          # lazy
        self._client = OpenAI()
        self.model, self.seed, self.temperature = model, seed, temperature
        self.last_system_fingerprint: Optional[str] = None

    def __call__(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, seed=self.seed, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self.last_system_fingerprint = getattr(resp, "system_fingerprint", None)
        return resp.choices[0].message.content or "{}"


# --------------------------------------------------------------------------- #
# Prompt — the single unified extraction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You extract evidence, never scores. Read the provided earnings/filing chunks and
return STRICT JSON only (no prose, no markdown). Every item MUST include a
chunk_id that exists in the input and a verbatim quote copied from that chunk.
Invent nothing. If a category has no evidence, return an empty list.
Counts, densities and quarter-over-quarter deltas are computed downstream in
code — do not compute them.
"""

USER_TEMPLATE = """\
{theme_instruction}

Return JSON with EXACTLY this shape:
{{
  "management_statements": [{{"chunk_id":"","quote":"","speaker":"CEO",
                             "tone":"positive|neutral|negative"}}],
  "hedging_markers":       [{{"chunk_id":"","quote":""}}],
  "defensive_terms":       [{{"chunk_id":"","quote":"","term":""}}],
  "superlative_terms":     [{{"chunk_id":"","quote":"","term":""}}],
  "guidance": {{
    "statements": [{{"chunk_id":"","quote":"","low":null,"high":null}}],
    "is_specific": false, "is_hedged": false, "withdrew": false
  }},
  "topic_avoidance": [{{"chunk_id":"","quote":"","what":""}}],
  "themes": [{{"theme":"","mentions":[{{"chunk_id":"","quote":"",
              "in_opening_remarks":false,"sentiment":"positive|neutral|negative"}}]}}],
  "risks":  [{{"risk":"","category":"operational","mentions":[{{"chunk_id":"",
              "quote":"","raised_by_management":false,"qualifier":""}}]}}]
}}

CHUNKS:
{chunks}
"""


def _theme_instruction(themes: Optional[Sequence[str]]) -> str:
    if themes:
        return ("Track ONLY these themes: " + ", ".join(themes) +
                ". Ignore other themes.")
    return "Identify the salient recurring themes yourself."


def _render_chunks(chunks: Sequence[dict]) -> str:
    out = []
    for c in chunks:
        spk = c.get("speaker", "")
        head = f"[chunk_id={c['chunk_id']}" + (f" speaker={spk}" if spk else "") + "]"
        out.append(f"{head}\n{c['text']}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Robust JSON parse + quote verification
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # strip fences
    return json.loads(text)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _verify_quote(quote: str, chunk_text: str) -> bool:
    return _norm(quote) in _norm(chunk_text)


# --------------------------------------------------------------------------- #
# Feature bundle
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    confidence: ConfidenceFeatures
    guidance: GuidanceFeatures
    narrative: list[NarrativeFeatures] = field(default_factory=list)
    risk: list[RiskFeatures] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)   # unverified items, for diagnostics
    system_fingerprint: Optional[str] = None


def _ref(item: dict, by_id: dict[str, dict]) -> Optional[EvidenceRef]:
    """Build a verified EvidenceRef, or None if the quote can't be confirmed."""
    cid = item.get("chunk_id")
    chunk = by_id.get(cid)
    if not chunk or not _verify_quote(item.get("quote", ""), chunk["text"]):
        return None
    return EvidenceRef(
        chunk_id=cid, quote=item.get("quote", ""),
        doc_id=chunk.get("doc_id"), page=chunk.get("page"),
        speaker=chunk.get("speaker") or item.get("speaker"),
    )


def build_features(raw: dict, chunks: Sequence[dict]) -> FeatureBundle:
    by_id = {c["chunk_id"]: c for c in chunks}
    dropped: list[dict] = []

    def verified(items, label):
        refs = []
        for it in items or []:
            r = _ref(it, by_id)
            (refs.append(r) if r else dropped.append({"category": label, "item": it}))
        return refs

    # words attributable to management (for hedge density)
    mgmt_words = sum(
        len(c["text"].split()) for c in chunks
        if not any(m in _norm(c.get("speaker", "")) for m in _ANALYST_MARKERS)
    ) or sum(len(c["text"].split()) for c in chunks) or 1

    # ---- confidence ----
    mgmt = raw.get("management_statements", [])
    tone_vals, tone_refs = [], []
    for st in mgmt:
        r = _ref(st, by_id)
        if r:
            tone_vals.append(TONE_MAP.get(st.get("tone", "neutral"), 0.0))
            tone_refs.append(r)
        else:
            dropped.append({"category": "management_statements", "item": st})
    hedge_refs = verified(raw.get("hedging_markers"), "hedging_markers")
    def_refs = verified(raw.get("defensive_terms"), "defensive_terms")
    sup_refs = verified(raw.get("superlative_terms"), "superlative_terms")
    avoid_refs = verified(raw.get("topic_avoidance"), "topic_avoidance")
    g = raw.get("guidance", {}) or {}
    g_refs = verified(g.get("statements"), "guidance_statements")

    confidence = ConfidenceFeatures(
        mean_tone=(sum(tone_vals) / len(tone_vals)) if tone_vals else 0.0,
        tone_evidence=tone_refs,
        hedge_per_1k=len(hedge_refs) / mgmt_words * 1000.0,
        hedge_evidence=hedge_refs,
        defensive_terms=len(def_refs), defensive_evidence=def_refs,
        guidance_band_widened=False,           # QoQ — attach_prior() sets this
        guidance_evidence=g_refs,
        superlative_terms=len(sup_refs), superlative_evidence=sup_refs,
        topic_avoidance=len(avoid_refs) > 0, avoidance_evidence=avoid_refs,
        total_words=mgmt_words,
    )

    # ---- guidance (language only; outcomes attached from reconciliation) ----
    guidance = GuidanceFeatures(
        outcomes=[],                            # attach_prior() / reconciler fills
        is_specific=bool(g.get("is_specific")),
        is_hedged=bool(g.get("is_hedged")),
        withdrew_guidance=bool(g.get("withdrew")),
        language_evidence=g_refs,
    )

    # ---- narrative (per theme) ----
    narrative = []
    for th in raw.get("themes", []) or []:
        refs, sents = [], []
        for m in th.get("mentions", []) or []:
            r = _ref(m, by_id)
            if r:
                refs.append(r)
                sents.append(TONE_MAP.get(m.get("sentiment", "neutral"), 0.0))
            else:
                dropped.append({"category": "theme_mention", "item": m})
        if not refs:
            continue
        narrative.append(NarrativeFeatures(
            theme=th.get("theme", ""),
            mentions_current=len(refs),
            mentions_previous=0,                # QoQ attached later
            sentiment_current=sum(sents) / len(sents),
            sentiment_previous=0.0,
            in_opening_remarks=any(m.get("in_opening_remarks") for m in th.get("mentions", [])),
            evidence=refs,
        ))

    # ---- risk (per risk) ----
    risk = []
    for rk in raw.get("risks", []) or []:
        refs, sev, mgmt_raised = [], 0, False
        for m in rk.get("mentions", []) or []:
            r = _ref(m, by_id)
            if r:
                refs.append(r)
                if m.get("qualifier"):
                    sev += 1
                mgmt_raised = mgmt_raised or bool(m.get("raised_by_management"))
            else:
                dropped.append({"category": "risk_mention", "item": m})
        if not refs:
            continue
        risk.append(RiskFeatures(
            risk=rk.get("risk", ""), category=rk.get("category", "operational"),
            mentions_current=len(refs), mentions_previous=0,
            raised_by_management=mgmt_raised, severe_qualifiers=sev, evidence=refs,
        ))

    return FeatureBundle(confidence=confidence, guidance=guidance,
                         narrative=narrative, risk=risk, dropped=dropped)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract(chunks: Sequence[dict], completer: Completer,
            themes: Optional[Sequence[str]] = None) -> FeatureBundle:
    """ONE LLM call. Returns verified, source-anchored features for all signals."""
    user = USER_TEMPLATE.format(theme_instruction=_theme_instruction(themes),
                                chunks=_render_chunks(chunks))
    raw = _parse_json(completer(SYSTEM_PROMPT, user))
    bundle = build_features(raw, chunks)
    bundle.system_fingerprint = getattr(completer, "last_system_fingerprint", None)
    return bundle


def attach_prior(bundle: FeatureBundle,
                 prev_defensive_terms: Optional[int] = None,
                 prev_guidance_band_widened: bool = False,
                 prev_theme_counts: Optional[dict[str, int]] = None,
                 prev_theme_sentiment: Optional[dict[str, float]] = None,
                 prev_risk_counts: Optional[dict[str, int]] = None,
                 guidance_outcomes: Optional[list[str]] = None) -> None:
    """
    Fill QoQ fields from stored prior-quarter features (from DuckDB) and the
    guidance-vs-actuals reconciliation. Everything here is data you already
    persist — no extra LLM call.
    """
    if prev_defensive_terms is not None:
        # rebuild confidence with prior context (dataclass is not frozen)
        bundle.confidence.guidance_band_widened = prev_guidance_band_widened
    if prev_theme_counts:
        for n in bundle.narrative:
            n.mentions_previous = prev_theme_counts.get(n.theme, 0)
            n.sentiment_previous = (prev_theme_sentiment or {}).get(n.theme, 0.0)
    if prev_risk_counts:
        for r in bundle.risk:
            r.mentions_previous = prev_risk_counts.get(r.risk, 0)
    if guidance_outcomes is not None:
        bundle.guidance.outcomes = guidance_outcomes
    # confidence QoQ prior counts handled by passing prev ConfidenceFeatures
    # into score_confidence(); see extract_and_score below.


def extract_and_score(chunks: Sequence[dict], completer: Completer,
                      themes: Optional[Sequence[str]] = None,
                      prev_confidence: Optional[ConfidenceFeatures] = None) -> dict:
    """Full one-call pipeline: extract once, then run all four pure scorers."""
    b = extract(chunks, completer, themes)
    return {
        "confidence": score_confidence(b.confidence, prev_confidence),
        "guidance": score_guidance(b.guidance),
        "narrative": [score_narrative(n) for n in b.narrative],
        "risk": [score_risk(r) for r in b.risk],
        "_bundle": b,
    }
