#!/usr/bin/env python3
"""
diagnose_quote.py — locate quotes that validate_run reports as unverifiable, and
show WHY they don't match.

validate_run tells you a quote isn't in any ingested document. That's the right
answer to the audit question, but it's the wrong level of detail for fixing
anything: you still need the full quote, which document it should have come
from, and the exact point where it stops agreeing with the source.

This walks each stored quote, and for the failures does a binary search for the
longest PREFIX of the quote that does appear in the corpus. That prefix is the
divergence point — everything before it is genuinely in the document, and the
first character after it is where quote and source part company. Printing the
source text around that point usually makes the cause obvious at a glance
(dropped comma, stitched-together sentences, a table header repeated mid-chunk).

Usage:
    python diagnose_quote.py --ticker INFY
    python diagnose_quote.py --ticker INFY --kind confidence
    python diagnose_quote.py --ticker INFY --context 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import normalize_quote_text          # noqa: E402
from store.signal_store import SignalStore       # noqa: E402

DIM, RED, GRN, YEL, CYA, BOLD, OFF = (
    "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[1m", "\033[0m"
)


def longest_matching_prefix(needle: str, corpus: str) -> int:
    """Length of the longest prefix of `needle` that occurs in `corpus`."""
    if not needle or needle[:1] not in corpus:
        return 0
    lo, hi = 1, len(needle)          # lo always matches, hi may not
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if needle[:mid] in corpus:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--kind", help="only check one of: confidence narrative guidance risk")
    ap.add_argument("--context", type=int, default=260,
                    help="characters of source text to show around the divergence point")
    args = ap.parse_args()
    ticker = args.ticker.upper()

    store = SignalStore()

    # Per-document corpora, so a failure can be attributed to a document rather
    # than to "the corpus" as a whole.
    docs = store.get_source_documents(ticker, limit=500)
    if not docs:
        print(f"{RED}No ingested documents for {ticker}.{OFF}")
        return 1

    per_doc: list[tuple[dict, str]] = []
    for d in docs:
        text = store.get_document_text(d["doc_id"]) or ""
        if text:
            per_doc.append((d, normalize_quote_text(text)))
    corpus = " ".join(t for _, t in per_doc)

    rows_by_kind = {
        "confidence": store.get_confidence_history(ticker, 20),
        "narrative":  store.get_narrative_history(ticker, 20),
        "guidance":   store.get_guidance_history(ticker, 20),
        "risk":       store.get_risk_history(ticker, 20),
    }
    if args.kind:
        rows_by_kind = {args.kind: rows_by_kind.get(args.kind, [])}

    print(f"\n{BOLD}Quote diagnosis: {ticker}{OFF}")
    print(f"{DIM}{len(per_doc)} document(s), {len(corpus):,} normalised characters{OFF}\n")

    checked = failed = 0

    def report(kind: str, period: str, quote: str, origin: str) -> None:
        nonlocal failed
        failed += 1
        n = normalize_quote_text(quote)
        print(f"{RED}✗ {kind} {period}{OFF}  {DIM}({origin}){OFF}")
        print(f"  {BOLD}full quote:{OFF} {quote}")

        k = longest_matching_prefix(n, corpus)
        if k == 0:
            print(f"  {YEL}Nothing in this quote appears in any document — likely paraphrase "
                  f"or invented.{OFF}\n")
            return

        matched, rest = n[:k], n[k:]
        print(f"  {GRN}matches for {k}/{len(n)} chars:{OFF} …{matched[-90:]}")
        print(f"  {RED}diverges at:{OFF} {rest[:90]}…")

        # Which document held the matching part, and what does it actually say there?
        for d, text in per_doc:
            pos = text.find(matched[-120:] if len(matched) > 120 else matched)
            if pos == -1:
                continue
            start = max(0, pos - 40)
            end = min(len(text), pos + args.context)
            print(f"  {CYA}document:{OFF} {d.get('title', '')[:70]}")
            print(f"  {DIM}{d.get('doc_type','')} · {d.get('quarter','')} {d.get('fiscal_year','')}{OFF}")
            print(f"  {DIM}{d.get('source_url','')}{OFF}")
            print(f"  {BOLD}source says:{OFF} …{text[start:end]}…")
            break
        else:
            print(f"  {YEL}Matching prefix spans a document boundary — the quote is stitched "
                  f"from more than one document.{OFF}")
        print()

    for kind, rows in rows_by_kind.items():
        for row in rows or []:
            period = f"{row.get('quarter','?')} {row.get('fiscal_year','?')}"
            for c in row.get("citations", []) or []:
                q = (c.get("quote") or "")
                n = normalize_quote_text(q).rstrip("…").rstrip(".").strip()
                if len(n) < 20:
                    continue
                checked += 1
                if n not in corpus:
                    report(kind, period, q, f"citation · chunk {c.get('chunk_id','?')}")
            for th in row.get("themes", []) or []:
                for kq in th.get("key_quotes", []) or []:
                    n = normalize_quote_text(kq)
                    if len(n) < 20:
                        continue
                    checked += 1
                    if n not in corpus:
                        report(kind, period, kq, f"theme key_quote · {th.get('theme','?')}")

    if failed:
        print(f"{RED}{failed} of {checked} quote(s) could not be verified.{OFF}")
    else:
        print(f"{GRN}All {checked} quotes verified against the stored source text.{OFF}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
