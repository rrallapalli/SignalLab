#!/usr/bin/env python3
"""
validate_run.py — check that a stored ticker run is internally honest.

This does NOT check whether a score is *right*. Whether 7.5 is the correct
reading of management's tone is a judgement, and no script settles it.

It checks the three things that ARE objectively checkable, and that every bug
found in this codebase so far has violated:

  PROVENANCE  Is the evidence real, and from the period it claims?
  ARITHMETIC  Do the numbers agree with each other?
  COVERAGE    Was there enough evidence to justify scoring at all?

Each check below exists because something silently broke it. A green run does
not mean the signals are good — it means they are *defensible*: the quotes are
real, the periods are separate, and the maths adds up. You still have to read
the quotes and decide whether you'd have said 7.5.

Usage:
    python -m validation.validate_run --ticker HDFCBANK
    python -m validation.validate_run --ticker HDFCBANK --verbose
    python -m validation.validate_run --all          # every stored ticker

Exit code 0 = all checks passed (CI-friendly), 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from store.signal_store import SignalStore  # noqa: E402

# ── Reporting ────────────────────────────────────────────────────────────────

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_ICON = {PASS: "✅", FAIL: "❌", WARN: "⚠️ ", SKIP: "⏭️ "}


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    items: list[str] = field(default_factory=list)
    why: str = ""          # which real bug this check exists to catch


class Report:
    def __init__(self, ticker: str, verbose: bool = False):
        self.ticker = ticker
        self.verbose = verbose
        self.results: list[Result] = []

    def add(self, r: Result) -> None:
        self.results.append(r)

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    @property
    def inconclusive(self) -> bool:
        """
        Too little was actually checked to say anything.

        Without this, a run where every check SKIPs reported "DEFENSIBLE, 0
        passed, 0 failed" and exited 0 — a green light derived from no evidence,
        which is the exact failure this harness exists to catch. A validator is
        not exempt from its own standard.
        """
        real = [r for r in self.results if r.status in (PASS, FAIL)]
        return len(real) < 3

    def render(self) -> None:
        print(f"\n{'═' * 74}")
        print(f"  Validating stored run: {self.ticker}")
        print(f"{'═' * 74}")
        for r in self.results:
            print(f"\n{_ICON[r.status]} {r.status}  {r.name}")
            if r.detail:
                print(f"        {r.detail}")
            if r.status == FAIL and r.why:
                print(f"        └─ catches: {r.why}")
            shown = r.items if self.verbose else r.items[:4]
            for it in shown:
                print(f"        · {it}")
            if len(r.items) > len(shown):
                print(f"        · … {len(r.items) - len(shown)} more (--verbose)")

        n_fail = sum(r.status == FAIL for r in self.results)
        n_warn = sum(r.status == WARN for r in self.results)
        n_pass = sum(r.status == PASS for r in self.results)
        n_skip = sum(r.status == SKIP for r in self.results)
        print(f"\n{'─' * 74}")
        if n_fail:
            verdict = "❌ NOT TRUSTWORTHY"
        elif self.inconclusive:
            verdict = "🚫 INCONCLUSIVE"
        elif n_warn:
            verdict = "⚠️  PASSED WITH WARNINGS"
        else:
            verdict = "✅ DEFENSIBLE"
        print(f"  {verdict}   {n_pass} passed · {n_warn} warnings · {n_fail} failed · {n_skip} skipped")
        if self.inconclusive and not n_fail:
            print(f"  Only {n_pass + n_fail} checks could actually run — that is not a pass.")
            print("  Usually means the stored data is too old or too thin to verify.")
        elif not n_fail:
            print("  Note: 'defensible' ≠ 'correct'. Read the quotes and judge the score yourself.")
        print(f"{'─' * 74}\n")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    """Whitespace/quote-insensitive form for substring comparison."""
    s = (s or "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip('"\'').lower()


def _period(row: dict) -> str:
    return f"{row.get('quarter','?')} {row.get('fiscal_year','?')}".strip()


def _cite_period(c: dict) -> str:
    """Citation.quarter now carries the year ('Q1 2026'); older rows may not."""
    return (c.get("quarter") or "").strip()


# ── Checks ───────────────────────────────────────────────────────────────────


# Confidence, narrative and risk build citations from current_chunks only, so
# every citation must name the signal's own period. GUIDANCE IS DIFFERENT BY
# DESIGN: it retrieves across periods_to_compare because auditing "did they
# deliver?" requires the guidance given in an EARLIER quarter alongside the
# actuals reported in a LATER one. An earlier version of this check applied the
# single-period rule to all four kinds and reported 16 false failures on real
# data — the check was wrong, not the app.
SINGLE_PERIOD_KINDS = {"confidence", "narrative", "risk"}


def check_citation_periods(rows_by_kind: dict[str, list[dict]]) -> Result:
    """Citations must come from a period the signal is entitled to cite."""
    known_periods = {
        _period(row) for rows in rows_by_kind.values() for row in rows
    }
    bad, undated, checked = [], 0, 0

    for kind, rows in rows_by_kind.items():
        for row in rows:
            want = _period(row)
            for c in row.get("citations", []):
                got = _cite_period(c)
                if not got:
                    continue
                if not re.search(r"\b\d{4}\b", got):
                    undated += 1
                    continue
                checked += 1
                if kind in SINGLE_PERIOD_KINDS:
                    if got != want:
                        bad.append(f"{kind} {want} cites evidence from {got}")
                else:
                    # Multi-period agent: any period it compared is legitimate,
                    # but evidence from outside this ticker's known periods is not.
                    if known_periods and got not in known_periods:
                        bad.append(
                            f"{kind} {want} cites {got}, which is not among the "
                            f"periods analysed for this ticker"
                        )

    if bad:
        return Result(
            "Citation periods match their signal", FAIL,
            f"{len(bad)} citation(s) drawn from a period the signal cannot justify.",
            bad,
            why="retrieval filtering on quarter without fiscal_year — 'Q1' matched "
                "Q1 of every year, so Latest and YoY scored identical evidence.",
        )
    if not checked:
        return Result("Citation periods match their signal", SKIP,
                      "No dated citations to check.")
    if undated:
        return Result(
            "Citation periods match their signal", WARN,
            f"{undated} citation(s) carry a bare quarter with no year — cannot verify. "
            "These pre-date the fix; re-run the ticker.",
        )
    return Result("Citation periods match their signal", PASS,
                  f"All {checked} citations come from a period their signal is entitled "
                  f"to cite (guidance may span the periods it compares).")


def check_periods_disjoint(conf_rows: list[dict]) -> Result:
    """Different periods must not be built from the same chunks."""
    by_period: dict[str, set] = {}
    for row in conf_rows:
        ids = {c.get("chunk_id") for c in row.get("citations", []) if c.get("chunk_id")}
        if ids:
            by_period[_period(row)] = ids
    if len(by_period) < 2:
        return Result("Periods use disjoint evidence", SKIP,
                      "Fewer than two scored periods stored — nothing to compare.")

    overlaps = []
    periods = sorted(by_period)
    for i, a in enumerate(periods):
        for b in periods[i + 1:]:
            shared = by_period[a] & by_period[b]
            if shared:
                overlaps.append(f"{a} and {b} share {len(shared)} identical chunk(s)")
    if overlaps:
        return Result(
            "Periods use disjoint evidence", FAIL,
            "Two periods were scored from the same chunks — their comparison is meaningless.",
            overlaps,
            why="the fiscal_year bug: Latest and YoY issued identical filters, so "
                "the YoY delta was a quarter compared with itself (— 0.0).",
        )
    return Result("Periods use disjoint evidence", PASS,
                  f"{len(by_period)} periods, no shared chunks — comparisons are real.")


def check_quotes_verbatim(store: SignalStore, ticker: str,
                          rows_by_kind: dict[str, list[dict]]) -> Result:
    """Every quote must appear in the actual ingested document text."""
    docs = store.get_source_documents(ticker, limit=200)
    corpus = " ".join(_norm(store.get_document_text(d["doc_id"]) or "") for d in docs)
    if not corpus.strip():
        return Result("Quotes appear in the source documents", SKIP,
                      "No raw document text stored — cannot verify. (Pre-dates raw_text.)")

    missing, checked = [], 0
    for kind, rows in rows_by_kind.items():
        for row in rows:
            for c in row.get("citations", []):
                q = _norm(c.get("quote", ""))
                q = q.rstrip("…").rstrip(".").strip()
                if len(q) < 20:
                    continue
                checked += 1
                if q not in corpus:
                    missing.append(f"{kind} {_period(row)}: “{c.get('quote','')[:65]}…”")

            for th in row.get("themes", []) or []:
                for kq in th.get("key_quotes", []) or []:
                    kq_n = _norm(kq)
                    if len(kq_n) < 20:
                        continue
                    checked += 1
                    if kq_n not in corpus:
                        missing.append(f"{kind} {_period(row)} theme key_quote: “{kq[:65]}…”")

    if not checked:
        return Result("Quotes appear in the source documents", SKIP, "No quotes long enough to check.")
    if missing:
        return Result(
            "Quotes appear in the source documents", FAIL,
            f"{len(missing)} of {checked} quote(s) do NOT appear in any ingested document.",
            missing,
            why="theme key_quotes were written by the model with no verbatim "
                "instruction and no check — a quote nobody said.",
        )
    return Result("Quotes appear in the source documents", PASS,
                  f"All {checked} quotes found verbatim in the stored source text.")


def check_guidance_arithmetic(guid_rows: list[dict]) -> Result:
    """beat_rate and serial_miss_risk must follow from the counts."""
    problems = []
    for row in guid_rows:
        p = _period(row)
        beats, misses = row.get("beats") or 0, row.get("misses") or 0
        in_line = row.get("in_line") or 0
        tracked = beats + misses + in_line

        if tracked == 0 and row.get("score") is not None:
            problems.append(f"{p}: score {row['score']} published with 0 items tracked")
            continue

        rate = row.get("beat_rate")
        if tracked and rate is not None:
            expected = round(beats / tracked, 3)
            if abs(float(rate) - expected) > 0.01:
                problems.append(
                    f"{p}: beat_rate {rate} but {beats}/{tracked} = {expected}"
                )

        items = row.get("guidance_items") or []
        if items:
            counts = Counter(
                (i.get("metric") or "").strip().lower() for i in items
                if (i.get("outcome") or "").lower() == "miss" and (i.get("metric") or "").strip()
            )
            expected_serial = bool([m for m, c in counts.items() if c >= 2])
            actual = bool(row.get("serial_miss_risk"))
            if expected_serial != actual:
                problems.append(
                    f"{p}: serial_miss_risk={actual} but items show {expected_serial}"
                )

    if not guid_rows:
        return Result("Guidance arithmetic is consistent", SKIP, "No guidance signals stored.")
    if problems:
        return Result(
            "Guidance arithmetic is consistent", FAIL,
            "Stored figures disagree with the counts they derive from.",
            problems,
            why="beat_rate and serial_miss_risk were asked of the LLM instead of "
                "counted, and a 45/100 score was published on 0 tracked items.",
        )
    return Result("Guidance arithmetic is consistent", PASS,
                  f"beat_rate and serial_miss_risk follow from the counts in {len(guid_rows)} period(s).")


def check_theme_arithmetic(narr_rows: list[dict]) -> Result:
    """Theme deltas must equal current − previous."""
    problems, checked = [], 0
    for row in narr_rows:
        for th in row.get("themes", []) or []:
            name = (th.get("theme") or "?")[:40]
            cur_n, prev_n = th.get("evidence_count_current"), th.get("evidence_count_previous")
            if cur_n is not None and prev_n is not None and th.get("count_change") is not None:
                checked += 1
                if int(th["count_change"]) != int(cur_n) - int(prev_n):
                    problems.append(
                        f"{_period(row)} '{name}': count_change={th['count_change']} "
                        f"but {cur_n}−{prev_n}={int(cur_n)-int(prev_n)}"
                    )
            cur_s, prev_s = th.get("sentiment_current"), th.get("sentiment_previous")
            if cur_s is not None and prev_s is not None and th.get("sentiment_change") is not None:
                checked += 1
                if abs(float(th["sentiment_change"]) - (float(cur_s) - float(prev_s))) > 0.02:
                    problems.append(
                        f"{_period(row)} '{name}': sentiment_change={th['sentiment_change']} "
                        f"but {cur_s}−{prev_s}={round(float(cur_s)-float(prev_s),2)}"
                    )
    if not checked:
        return Result("Theme deltas equal current − previous", SKIP, "No themes with both operands stored.")
    if problems:
        return Result(
            "Theme deltas equal current − previous", FAIL,
            "A stored delta contradicts its own operands.",
            problems,
            why="count_change / sentiment_change were asked of the model rather "
                "than subtracted from fields it had already returned.",
        )
    return Result("Theme deltas equal current − previous", PASS,
                  f"All {checked} theme deltas match their operands.")


def check_no_invented_priors(conf_rows: list[dict]) -> Result:
    """
    HEURISTIC. Looks for prose narrating a numeric before/after score.

    This is a regex over free text, so it is a smoke detector, not a proof:

      · It cannot enumerate every phrasing. "Confidence rose to 7.5 from last
        quarter's 7.2" is an invented prior this will not catch unless the
        pattern below happens to fit.
      · Business facts legitimately contain before/after numbers. "Net income
        grew from 9% to 11%" is a real quote from a real transcript, and an
        earlier version of this check failed it.

    So it only fires on numbers that look like SCORES (0–10, not followed by a
    unit) in a sentence that is talking about confidence/score/tone — and it
    reports WARN, never FAIL. The deterministic version of this question is
    check_stored_priors_are_computed(), which reads a column rather than prose.
    Trust that one; treat this as a prompt to go and read the summary.
    """
    hits = []
    # A score-like number: 0–10, optionally one decimal, NOT followed by a unit
    # (%, bps, x, crore, bn…) — that's what separates a score from a metric.
    num = r"(?:10(?:\.0)?|[0-9](?:\.[0-9])?)"
    unit = r"(?!\s*(?:%|percent|bps|bp|x\b|cr\b|crore|lakh|bn\b|mn\b|billion|million))"
    patterns = [
        rf"\b(?:increased|decreased|improved|declined|fell|rose|moved|went)\s+from\s+({num}){unit}\s*(?:to|→)\s*({num}){unit}",
        rf"\bfrom\s+({num}){unit}\s*(?:to|→)\s*({num}){unit}\s+(?:QoQ|YoY|quarter)",
        rf"\b(?:rose|fell|moved|improved|declined)\s+to\s+({num}){unit}\s+from\s+(?:[^.]{{0,25}}?)({num}){unit}",
    ]
    subject = re.compile(r"\b(confidence|score|tone|rating)\b", re.I)

    for row in conf_rows:
        for sentence in re.split(r"(?<=[.!?])\s+", row.get("summary", "") or ""):
            if not subject.search(sentence):
                continue          # a number in a sentence about revenue is a fact
            for pat in patterns:
                m = re.search(pat, sentence, re.I)
                if m:
                    hits.append(f"{_period(row)}: “…{m.group(0)}…”")
                    break

    if hits:
        return Result(
            "Summaries don't narrate invented prior scores", WARN,
            "A summary appears to state a before/after score the model was never "
            "given. HEURISTIC — read these and judge; a real business figure can "
            "look like this.",
            hits,
            why="previous_score/change were requested from the LLM, which had no "
                "access to them — it printed 'increased from 0 to 7.5' beside a "
                "computed '— 0.0'.",
        )
    return Result("Summaries don't narrate invented prior scores", PASS,
                  "No score-like transitions found in summaries (heuristic).")


def check_stored_priors_are_computed(conf_rows: list[dict]) -> Result:
    """previous_score/change must not be persisted from the model."""
    stale = [
        f"{_period(r)}: previous_score={r.get('previous_score')} change={r.get('change')}"
        for r in conf_rows
        if r.get("previous_score") is not None or r.get("change") is not None
    ]
    if stale:
        return Result(
            "Prior scores are not model-supplied", WARN,
            "Rows still carry LLM-supplied previous_score/change. These pre-date the "
            "fix — re-run the ticker to clear them.",
            stale,
        )
    return Result("Prior scores are not model-supplied", PASS,
                  "No model-supplied priors stored; deltas are computed from real scores.")


def check_coverage(store: SignalStore, ticker: str, conf_rows: list[dict]) -> Result:
    """Every scored period should have documents behind it."""
    docs = store.get_source_documents(ticker, limit=200)
    doc_periods = Counter(f"{d['quarter']} {d['fiscal_year']}" for d in docs)
    thin = []
    for row in conf_rows:
        p = _period(row)
        n = doc_periods.get(p, 0)
        if n == 0:
            thin.append(f"{p}: scored, but NO ingested documents recorded")
        elif n < 2:
            thin.append(f"{p}: scored from only {n} document")
    if not conf_rows:
        return Result("Scored periods have documents behind them", SKIP, "No confidence signals stored.")
    if any("NO ingested" in t for t in thin):
        return Result(
            "Scored periods have documents behind them", FAIL,
            "A period was scored with no source documents recorded.",
            thin,
            why="agents returned placeholder signals (score=5.0 / 45) on failure, "
                "writing rows for quarters that were never analysed.",
        )
    if thin:
        return Result("Scored periods have documents behind them", WARN,
                      "Thin coverage — the score rests on very little.", thin)
    return Result("Scored periods have documents behind them", PASS,
                  f"All {len(conf_rows)} scored periods have ≥2 source documents.")


# ── Driver ───────────────────────────────────────────────────────────────────


def validate(ticker: str, verbose: bool = False) -> Report:
    store = SignalStore()
    rep = Report(ticker, verbose)

    conf = store.get_confidence_history(ticker, 20)
    narr = store.get_narrative_history(ticker, 20)
    guid = store.get_guidance_history(ticker, 20)
    risk = store.get_risk_history(ticker, 20)

    if not any([conf, narr, guid, risk]):
        rep.add(Result(f"Stored signals for {ticker}", FAIL,
                       "No signals found. Run the ticker in the dashboard first."))
        return rep

    rows_by_kind = {"confidence": conf, "narrative": narr, "guidance": guid, "risk": risk}

    # PROVENANCE
    rep.add(check_citation_periods(rows_by_kind))
    rep.add(check_periods_disjoint(conf))
    rep.add(check_quotes_verbatim(store, ticker, rows_by_kind))
    # ARITHMETIC
    rep.add(check_guidance_arithmetic(guid))
    rep.add(check_theme_arithmetic(narr))
    rep.add(check_no_invented_priors(conf))
    rep.add(check_stored_priors_are_computed(conf))
    # COVERAGE
    rep.add(check_coverage(store, ticker, conf))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a stored SignalLab run.")
    ap.add_argument("--ticker", help="NSE symbol, e.g. HDFCBANK")
    ap.add_argument("--all", action="store_true", help="Validate every stored ticker")
    ap.add_argument("--verbose", action="store_true", help="List every offending item")
    args = ap.parse_args()

    if not args.ticker and not args.all:
        ap.error("give --ticker SYMBOL or --all")

    tickers = SignalStore().get_all_tickers() if args.all else [args.ticker.upper()]
    if not tickers:
        print("No tickers stored yet.")
        return 0

    any_failed = False
    for t in tickers:
        rep = validate(t, args.verbose)
        rep.render()
        # Inconclusive is not a pass. CI should not ship on "couldn't check".
        any_failed |= (rep.failed or rep.inconclusive)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
