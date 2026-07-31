"""
narration/narrator.py — fluent prose over a DETERMINISTIC result.

The score and its drivers are already computed. The LLM's ONLY job is to phrase
them; it never sees raw evidence, never re-judges, never invents a number or a
driver. So the deterministic layer stays the single source of truth and the
prose is a gloss on it.

Reproducibility comes from hash-caching: the narrative is keyed by a fingerprint
of the exact facts it describes. Identical facts -> identical stored prose,
generated once and reused. If a factor or the score changes, the fingerprint
changes and it regenerates. The audit trail remains the deterministic manifest;
this is just a cached, human-readable view of it.

No base.py dependency: narrate() takes an agent's bound `llm_reason` coroutine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Awaitable, Callable, Optional

from loguru import logger

# Bump to invalidate every cached narrative when the prompt/contract changes.
NARRATOR_VERSION = "1.0.0"

try:                                    # settings are optional at import time
    from config import settings as _settings
except Exception:                       # pragma: no cover
    _settings = None


def _cfg(name: str, default):
    return getattr(_settings, name, default) if _settings is not None else default


def _cache_dir() -> Path:
    d = Path(_cfg("NARRATION_CACHE_DIR", "data/narration_cache")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Fingerprint + cache
# --------------------------------------------------------------------------- #
def fingerprint(facts: dict) -> str:
    """Stable key over the facts the prose describes (order of drivers matters —
    it is the deterministic 'heaviest first' order the agent produced)."""
    payload = {
        "v": NARRATOR_VERSION,
        "signal": facts.get("signal", ""),
        "headline": facts.get("headline", ""),
        "drivers": list(facts.get("drivers", [])),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cache_get(fp: str) -> Optional[str]:
    try:
        return (_cache_dir() / f"{fp}.txt").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def cache_set(fp: str, text: str) -> None:
    try:
        (_cache_dir() / f"{fp}.txt").write_text(text, encoding="utf-8")
    except OSError as e:                 # cache is best-effort, never fatal
        logger.warning(f"[narrate] cache write failed: {e}")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
NARRATOR_SYSTEM = """You are an equity analyst writing a one- to two-sentence,
plain-English gloss on an ALREADY-COMPUTED signal, for a portfolio manager. You
are given the headline result and the exact factors that produced it.

STRICT RULES:
- Use ONLY the factors listed. Introduce no driver, number, company fact, or
  claim that is not among them.
- Do NOT re-judge, re-rank, or alter the headline result or any number in it.
- Lead with the result, then explain the main factors in natural prose, most
  important first. Do not just list the factors back — read them as an analyst.
- No bullet points, no quotation marks, no preamble, no hedging about being an AI.
  One or two sentences.

Return ONLY JSON: {"narrative": "..."}"""


def build_user_prompt(facts: dict) -> str:
    drivers = facts.get("drivers", []) or ["(no notable factors)"]
    lines = "\n".join(f"{i+1}. {d}" for i, d in enumerate(drivers))
    return (
        f"Signal: {facts.get('signal','')}\n"
        f"Headline (fixed — restate, do not change): {facts.get('headline','')}\n"
        f"Factors that produced it (most important first):\n{lines}\n\n"
        f"Write the gloss."
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
LLMReason = Callable[[str, str], Awaitable[dict]]


async def narrate(facts: dict, llm_reason: LLMReason) -> str:
    """
    Return cached or freshly-generated prose for `facts`. On any failure or if
    narration is disabled, returns facts['fallback'] (the deterministic summary),
    so the pipeline never depends on the LLM to produce a usable summary.

    facts = {
        "signal":   short label, e.g. "management confidence",
        "headline": fixed result string, e.g. "Management confidence 8.3/10",
        "drivers":  list[str] plain-English factors, heaviest first,
        "fallback": deterministic summary to use if narration is unavailable,
    }
    """
    fallback = (facts.get("fallback") or "").strip()

    if not _cfg("NARRATION_ENABLED", True):
        return fallback

    fp = fingerprint(facts)
    cached = cache_get(fp)
    if cached is not None:
        return cached

    text = ""
    try:
        data = await llm_reason(NARRATOR_SYSTEM, build_user_prompt(facts))
        text = re.sub(r"\s+", " ", (data.get("narrative") or "")).strip()
    except Exception as e:
        logger.warning(f"[narrate] narration failed ({e}); using deterministic summary.")

    if not text:
        return fallback

    cache_set(fp, text)
    return text


BATCH_SYSTEM = """You are an equity analyst. For EACH signal listed below, write a
one- to two-sentence, plain-English gloss for a portfolio manager, using ONLY that
signal's listed factors. Per signal: restate its headline result, do not change any
number, introduce no driver/number/fact not listed, no bullet points, no quotation
marks, no preamble.

Return ONLY a JSON object mapping each signal id to its gloss:
{"<id>": "...", "<id>": "...", ...}"""


def build_batch_prompt(misses: list) -> str:
    blocks = []
    for key, facts, _fp in misses:
        drivers = facts.get("drivers") or ["(no notable factors)"]
        lines = "\n".join(f"    - {d}" for d in drivers)
        blocks.append(
            f'id "{key}" — signal: {facts.get("signal","")}\n'
            f'  headline (restate, do not change): {facts.get("headline","")}\n'
            f'  factors (most important first):\n{lines}'
        )
    return "Signals:\n\n" + "\n\n".join(blocks) + '\n\nReturn JSON mapping id -> gloss.'


async def narrate_batch(items: list, llm_reason: LLMReason) -> dict:
    """
    One LLM call per quarter instead of one per signal.

    `items` = list of (key, facts). Each is looked up in the per-signal hash
    cache first; only the cache MISSES go into a single batched call, and each
    result is cached individually. So batching cuts token overhead (one shared
    system prompt) while keeping per-signal cache granularity — a change to one
    signal does not invalidate the others.

    Returns {key: narrative}; any missing/failed item resolves to its fallback.
    """
    out: dict[str, str] = {}
    enabled = _cfg("NARRATION_ENABLED", True)
    misses: list = []

    for key, facts in items:
        fallback = (facts.get("fallback") or "").strip()
        if not enabled:
            out[key] = fallback
            continue
        fp = fingerprint(facts)
        cached = cache_get(fp)
        if cached is not None:
            out[key] = cached
        else:
            out[key] = fallback              # provisional; replaced if narrated
            misses.append((key, facts, fp))

    if misses and enabled:
        data = {}
        try:
            data = await llm_reason(BATCH_SYSTEM, build_batch_prompt(misses))
        except Exception as e:
            logger.warning(f"[narrate] batch narration failed ({e}); using fallbacks.")
        if isinstance(data, dict):
            for key, facts, fp in misses:
                text = re.sub(r"\s+", " ", (data.get(key) or "")).strip()
                if text:
                    cache_set(fp, text)
                    out[key] = text
    return out
