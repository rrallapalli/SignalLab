"""
ingestion/fetcher.py – Discover and fetch source documents directly from
NSE (National Stock Exchange of India) and BSE (Bombay Stock Exchange).

Replaces the earlier Tavily-based web search. Indian-listed companies file
earnings-related disclosures — financial results, investor/analyst
presentations, transcripts of concalls, press releases, annual reports —
as PDF attachments to exchange corporate-announcement filings. Both
exchanges expose these publicly with no API key required:

  NSE  -> nseindia.com corporate-announcements endpoint (via the `nse` pkg)
  BSE  -> bseindia.com corporate-announcements endpoint (via the `bse` pkg)

`ticker` is expected to be the NSE trading symbol (e.g. "TCS", "INFY",
"RELIANCE", "HDFCBANK"). The matching BSE scrip code is resolved
automatically from the company name, but can be overridden with
`bse_scripcode=` if auto-lookup picks the wrong listing.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import io
from datetime import datetime, timedelta
from typing import Optional

import httpx
from dateutil import parser as dtparser
from loguru import logger
from pypdf import PdfReader

from bse import BSE
from nse import NSE

from config import settings
from models import DocumentType, SourceDocument

# ── Which announcement subjects map to which document type ──────────────────
# Order matters: first match wins.
SUBJECT_KEYWORDS: list[tuple[DocumentType, tuple[str, ...]]] = [
    (DocumentType.EARNINGS_CALL, (
        "transcript", "con call", "concall", "conference call", "earnings call",
    )),
    (DocumentType.INVESTOR_PRESENTATION, (
        "investor presentation", "analyst presentation", "investor/analyst presentation",
        "investor call presentation", "earnings presentation",
    )),
    (DocumentType.ANNUAL_REPORT, (
        "annual report",
    )),
    (DocumentType.PRESS_RELEASE, (
        "financial results", "un-audited financial", "unaudited financial",
        "audited financial", "results for the quarter", "outcome of board meeting",
        "press release", "financial result",
    )),
    (DocumentType.MANAGEMENT_COMMENTARY, (
        "credit rating", "clarification", "newspaper publication",
        "intimation", "outcome of the meeting",
    )),
]

# Only these doc types have a real chance of being found on the exchanges.
DEFAULT_DOC_TYPES = ["earnings_call", "press_release", "investor_presentation", "annual_report"]

BSE_ATTACHMENT_BASES = (
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/",
)

# NSE trading symbol → BSE scrip code.
#
# bse.getScripCode() does a name-based fuzzy lookup that fails on common name
# variants ("Infosys Ltd" vs the registered "Infosys Limited"), which silently
# drops all BSE filings for the ticker. These codes are stable identifiers, so
# resolving them from the symbol we already have is both correct and faster.
# Extend as new tickers are tested; unknown symbols fall back to the fuzzy lookup.
BSE_SCRIP_OVERRIDES: dict[str, str] = {
    "INFY":       "500209",
    "TCS":        "532540",
    "RELIANCE":   "500325",
    "HDFCBANK":   "500180",
    "ICICIBANK":  "532174",
    "SBIN":       "500112",
    "HINDUNILVR": "500696",
    "BHARTIARTL": "532454",
    "ITC":        "500875",
    "WIPRO":      "507685",
}


def _doc_id(ticker: str, doc_type: str, quarter: str, url: str) -> str:
    return hashlib.md5(f"{ticker}::{doc_type}::{quarter}::{url}".encode()).hexdigest()[:16]



# ── Which period is a document actually ABOUT? ────────────────────────────────
#
# Not "which search window found it". _search_window extends each quarter by
# NSE_BSE_RESULT_LAG_DAYS (results are announced after the quarter ends), so
# consecutive windows overlap by ~89 days. Labelling a document with the
# REQUESTED quarter therefore filed the same earnings call as both Q4 2025 and
# Q1 2026 — and since _doc_id hashes the quarter, it became two documents, two
# chunk sets, and two quarters scoring identical evidence.
#
# Derived, in order of confidence:
#   1. an explicit "quarter ended <date>" in the subject   — unambiguous
#   2. "Q<n> FY<yy>" in the subject                        — needs the fiscal map
#   3. the most recently ENDED quarter before the event    — heuristic fallback
#
# Everything maps onto _quarter_bounds' FISCAL-quarter convention (year runs
# Apr–Mar), so the labels stay consistent with the rest of the app.

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _fiscal_quarter(d: datetime) -> tuple[str, int]:
    """
    Calendar date → Indian FISCAL quarter (year runs Apr–Mar).

        Apr–Jun -> Q1 (FY = calendar year + 1)
        Jul–Sep -> Q2 (FY = calendar year + 1)
        Oct–Dec -> Q3 (FY = calendar year + 1)
        Jan–Mar -> Q4 (FY = calendar year)

    e.g. Jun 2026 -> ("Q1", 2027);  Dec 2025 -> ("Q3", 2026);  Feb 2026 -> ("Q4", 2026).
    """
    m = d.month
    q = (m - 4) % 12 // 3 + 1          # Apr=Q1 … Jan/Feb/Mar=Q4
    fy = d.year + 1 if m >= 4 else d.year
    return f"Q{q}", fy


def _period_from_subject(subject: str) -> tuple[str, int] | None:
    """
    Parse the reporting period out of an NSE/BSE announcement subject.

    Only the MONTH and YEAR matter for a quarter, so the day is ignored — which
    also sidesteps the ordering trap: Indian filings write both "quarter ended
    June 30, 2025" and "quarter ended 30th June, 2025", and a day-first regex
    reads the 30 in the first form as the year (→ 2030).
    """
    s = (subject or "").lower()

    # Text following "…ended …" — the date lives in there in some order.
    m = re.search(r"(?:quarter|period|year|qtr)[^.]{0,40}?end(?:ed|ing)?\s+(?:on\s+)?(.{0,28})", s)
    if m:
        tail = m.group(1)

        # Month name anywhere in the tail + a 4-digit (or 2-digit) year
        month = None
        for name, idx in _MONTHS.items():
            if re.search(rf"\b{name}[a-z]*\b", tail):
                month = idx
                break
        if month:
            ym = re.search(r"\b(\d{4})\b", tail) or re.search(r"[\s,'](\d{2})\b", tail)
            if ym:
                year = int(ym.group(1))
                if year < 100:
                    year += 2000
                if 2000 <= year <= 2100:
                    return _fiscal_quarter(datetime(year, month, 1))

        # Numeric: 31.12.2025 / 30-06-2025 / 31/03/26  (day first, Indian style)
        dm = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", tail)
        if dm:
            mon, yr = int(dm.group(2)), int(dm.group(3))
            if yr < 100:
                yr += 2000
            if 1 <= mon <= 12 and 2000 <= yr <= 2100:
                return _fiscal_quarter(datetime(yr, mon, 1))

    # "Q1 FY26" · "Q1 FY2026" — already fiscal, so return it directly.
    m = re.search(r"\bq([1-4])\s*(?:fy|f\.y\.?|fiscal)\s*[\s\-]?(\d{2,4})\b", s)
    if m:
        fq, fy_raw = int(m.group(1)), int(m.group(2))
        fy = fy_raw + 2000 if fy_raw < 100 else fy_raw
        return f"Q{fq}", fy

    return None


def _period_from_event_date(event_date: datetime | None) -> tuple[str, int] | None:
    """
    Fallback: results are published shortly AFTER the fiscal quarter they report
    on, so the subject period is the most recently ENDED fiscal quarter.
    """
    if not event_date:
        return None
    start, _ = _quarter_bounds(*_fiscal_quarter(event_date))
    prev_end = start - timedelta(days=1)
    return _fiscal_quarter(prev_end)


def _document_period(subject: str, event_date: datetime | None) -> tuple[tuple[str, int] | None, str]:
    p = _period_from_subject(subject)
    if p:
        return p, "subject"
    p = _period_from_event_date(event_date)
    if p:
        return p, "event_date"
    return None, "unknown"


def _classify(subject: str, allowed: set[str]) -> Optional[DocumentType]:
    s = subject.lower()
    for dtype, patterns in SUBJECT_KEYWORDS:
        if dtype.value not in allowed:
            continue
        if any(p in s for p in patterns):
            return dtype
    return None


def _quarter_bounds(quarter: str, year: int) -> tuple[datetime, datetime]:
    """
    FISCAL-quarter bounds (year runs Apr–Mar), `year` = 4-digit fiscal year.

        Q1 FY2027 -> (Apr 1 2026, Jun 30 2026)
        Q2 FY2027 -> (Jul 1 2026, Sep 30 2026)
        Q3 FY2027 -> (Oct 1 2026, Dec 31 2026)
        Q4 FY2027 -> (Jan 1 2027, Mar 31 2027)
    """
    q = int(quarter[1])
    start_month = 4 + (q - 1) * 3       # Q1→Apr, Q2→Jul, Q3→Oct, Q4→13(→Jan)
    start_year = year - 1
    if start_month > 12:                # Q4 rolls into the next calendar year
        start_month -= 12
        start_year += 1
    start = datetime(start_year, start_month, 1)
    end_month, end_year = start_month + 2, start_year
    if end_month == 12:
        end = datetime(end_year, 12, 31, 23, 59, 59)
    else:
        end = datetime(end_year, end_month + 1, 1) - timedelta(seconds=1)
    return start, end


def _search_window(quarter: str, year: int, doc_types: set[str]) -> tuple[datetime, datetime]:
    start, q_end = _quarter_bounds(quarter, year)
    lag = settings.NSE_BSE_ANNUAL_LAG_DAYS if "annual_report" in doc_types else settings.NSE_BSE_RESULT_LAG_DAYS
    end = min(q_end + timedelta(days=lag), datetime.utcnow())
    if end < start:
        end = start
    return start, end


async def _fetch_nse(nse: NSE, ticker: str, start: datetime, end: datetime) -> list[dict]:
    try:
        anns = await asyncio.to_thread(nse.announcements, symbol=ticker, from_date=start, to_date=end)
        return anns or []
    except Exception as e:
        logger.warning(f"NSE announcements failed for {ticker}: {e}")
        return []


async def _fetch_bse(bse: BSE, scripcode: str, start: datetime, end: datetime) -> list[dict]:
    rows: list[dict] = []
    page = 1
    total_pages = 1
    try:
        while page <= total_pages and page <= settings.NSE_BSE_MAX_PAGES:
            data = await asyncio.to_thread(
                bse.announcements, page_no=page, from_date=start, to_date=end, scripcode=str(scripcode),
            )
            batch = (data or {}).get("Table") or []
            if not batch:
                break
            total_pages = batch[0].get("TotalPageCnt") or 1
            rows.extend(batch)
            page += 1
    except Exception as e:
        logger.warning(f"BSE announcements failed for scripcode {scripcode}: {e}")
    return rows


def _nse_pdf_url(item: dict) -> str:
    return item.get("attchmntFile", "") or ""


def _nse_subject(item: dict) -> str:
    return f"{item.get('desc','')} {item.get('attchmntText','')}".strip()


def _nse_event_date(item: dict) -> Optional[datetime]:
    try:
        return dtparser.parse(item.get("an_dt", ""))
    except Exception:
        return None


def _bse_pdf_urls(item: dict) -> list[str]:
    name = item.get("ATTACHMENTNAME", "")
    if not name:
        return []
    return [base + name for base in BSE_ATTACHMENT_BASES]


def _bse_subject(item: dict) -> str:
    return f"{item.get('HEADLINE','')} {item.get('NEWSSUB','')}".strip()


def _bse_event_date(item: dict) -> Optional[datetime]:
    for key in ("NEWS_DT", "DT_TM"):
        try:
            if item.get(key):
                return dtparser.parse(item[key])
        except Exception:
            continue
    return None


async def _download_pdf_text(client: httpx.AsyncClient, urls: list[str]) -> str:
    """Try each candidate URL (BSE has both a 'live' and 'historical' path) until one works."""
    for url in urls:
        try:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    # BSE's AttachLive/AttachHis endpoints 403 without a same-site
                    # referer. Harmless for NSE URLs, which ignore it.
                    "Referer": "https://www.bseindia.com/",
                },
                timeout=25,
                follow_redirects=True,
            )
            if resp.status_code != 200 or not resp.content:
                continue
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:60])
            text = text.strip()
            if text:
                return text
        except Exception as e:
            logger.debug(f"PDF extract failed {url}: {e}")
            continue
    return ""


async def fetch_documents(
    ticker: str, company: str, quarter: str, year: int,
    doc_types: list[str] | None = None, include_prior: bool = True,
    bse_scripcode: str | None = None,
) -> list[SourceDocument]:
    if doc_types is None:
        doc_types = list(DEFAULT_DOC_TYPES)
    doc_types_set = set(doc_types)
    if "news_article" in doc_types_set or "broker_note" in doc_types_set:
        logger.debug("news_article/broker_note are not available from NSE/BSE filings; skipping those types.")
        doc_types_set -= {"news_article", "broker_note"}

    q_num = int(quarter[1])
    windows = [(quarter, q_num, year)]
    if include_prior:
        pq, py = (q_num - 1, year) if q_num > 1 else (4, year - 1)
        windows.append((f"Q{pq}", pq, py))

    nse: Optional[NSE] = None
    for attempt in range(3):
        try:
            nse = await asyncio.to_thread(NSE, download_folder=str(settings.NSE_CACHE_DIR))
            break
        except Exception as e:
            if attempt == 2:
                logger.warning(
                    f"NSE client init failed after 3 attempts (cookie handshake blocked?) "
                    f"— NSE filings will be skipped: {e}"
                )
            else:
                # Back off before retrying; concurrent handshakes from one IP
                # are what NSE's bot protection throttles most aggressively.
                await asyncio.sleep(2 * (attempt + 1))

    bse = BSE(download_folder=str(settings.BSE_CACHE_DIR))

    scripcode = bse_scripcode or BSE_SCRIP_OVERRIDES.get(ticker.upper())
    if scripcode is None:
        try:
            scripcode = await asyncio.to_thread(bse.getScripCode, company)
        except Exception as e:
            logger.warning(f"BSE scrip code lookup failed for '{company}' — BSE filings will be skipped: {e}")
            scripcode = None

    raw_items: list[tuple[str, dict, str, int]] = []   # (source, item, quarter_label, fiscal_year)
    for q_label, q_n, yr in windows:
        start, end = _search_window(q_label, yr, doc_types_set)

        if nse is not None:
            nse_items = await _fetch_nse(nse, ticker, start, end)
            for item in nse_items:
                raw_items.append(("NSE", item, q_label, yr))

        if scripcode:
            bse_items = await _fetch_bse(bse, scripcode, start, end)
            for item in bse_items:
                raw_items.append(("BSE", item, q_label, yr))

    if nse is not None:
        try:
            nse.exit()
        except Exception:
            pass
    try:
        bse.exit()
    except Exception:
        pass

    # Classify + dedupe by PDF url before downloading anything
    candidates: list[dict] = []
    seen_urls: set[str] = set()
    for source, item, q_label, yr in raw_items:
        if source == "NSE":
            subject = _nse_subject(item)
            urls = [_nse_pdf_url(item)] if _nse_pdf_url(item) else []
            event_date = _nse_event_date(item)
        else:
            subject = _bse_subject(item)
            urls = _bse_pdf_urls(item)
            event_date = _bse_event_date(item)

        if not urls or not urls[0]:
            continue
        dtype = _classify(subject, doc_types_set)
        if dtype is None:
            continue
        key = urls[0]
        if key in seen_urls:
            continue

        # Which period is this document actually ABOUT? Consecutive search
        # windows overlap by ~NSE_BSE_RESULT_LAG_DAYS, so the same filing turns
        # up in two windows. Labelling it with the requested quarter filed one
        # earnings call as BOTH Q4 2025 and Q1 2026 — two doc_ids, two chunk
        # sets, two quarters silently scoring identical evidence.
        #
        # Annual reports cover a year, not a quarter, so this doesn't apply.
        if dtype is not DocumentType.ANNUAL_REPORT:
            period, how = _document_period(subject, event_date)
            if period is None:
                logger.debug(
                    f"[fetch] Cannot date {subject[:60]!r} — keeping under the "
                    f"requested {q_label} {yr} (unverified)."
                )
            elif period != (q_label, yr):
                logger.debug(
                    f"[fetch] Skipping {subject[:60]!r}: it reports on "
                    f"{period[0]} {period[1]} (from {how}), not {q_label} {yr}. "
                    f"The {period[0]} {period[1]} window will pick it up."
                )
                continue

        seen_urls.add(key)
        candidates.append({
            "urls": urls, "subject": subject, "doc_type": dtype,
            "quarter": q_label, "fiscal_year": yr, "event_date": event_date,
            "source": source,
        })

    # Rank: prefer earnings call / investor presentation / press release, then recency
    _priority = {
        DocumentType.EARNINGS_CALL: 0, DocumentType.INVESTOR_PRESENTATION: 1,
        DocumentType.PRESS_RELEASE: 2, DocumentType.ANNUAL_REPORT: 3,
        DocumentType.MANAGEMENT_COMMENTARY: 4,
    }
    candidates.sort(key=lambda c: (_priority.get(c["doc_type"], 9), c["event_date"] or datetime.min), reverse=False)
    candidates = candidates[: settings.NSE_BSE_MAX_DOCS_PER_QUARTER * len(windows)]

    docs: list[SourceDocument] = []
    async with httpx.AsyncClient() as client:
        texts = await asyncio.gather(*[_download_pdf_text(client, c["urls"]) for c in candidates])

    for c, text in zip(candidates, texts):
        if len(text) < 200:
            continue
        url = c["urls"][0]
        docs.append(SourceDocument(
            doc_id=_doc_id(ticker, c["doc_type"].value, c["quarter"], url),
            ticker=ticker, company=company, doc_type=c["doc_type"],
            quarter=c["quarter"], fiscal_year=c["fiscal_year"],
            event_date=c["event_date"], source_url=url,
            title=c["subject"][:200], raw_text=text[:30000],
        ))

    logger.success(
        f"Fetched {len(docs)} documents for {ticker} {quarter} {year} "
        f"(NSE symbol={ticker}, BSE scripcode={scripcode or 'n/a'})"
    )
    return docs
