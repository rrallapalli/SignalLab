"""
agents/prior_lookup.py — QoQ baseline from stored data.

Narrative and risk are comparative signals, but re-reading the prior quarter's
chunks on every run is costly and non-deterministic. Instead each quarter
extracts its OWN counts (self-contained), and the shift is computed by matching
this quarter's themes/risks to the prior quarter's STORED counts.

Cross-quarter names rarely match byte-for-byte (the model rephrases), so we
normalise and fall back to a conservative fuzzy match. No external deps.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def match_prior(name: str, prior_names, threshold: float = 0.85):
    """
    Best-matching prior name for `name` — exact (normalised) first, then a
    fuzzy match above `threshold`. Returns the raw prior name, or None if
    nothing matches well enough (treated by callers as "no prior baseline").
    """
    n = _norm(name)
    norm_map = {_norm(p): p for p in prior_names if p}
    if n in norm_map:
        return norm_map[n]
    best, best_r = None, 0.0
    for pn, raw in norm_map.items():
        r = SequenceMatcher(None, n, pn).ratio()
        if r > best_r:
            best, best_r = raw, r
    return best if best_r >= threshold else None
