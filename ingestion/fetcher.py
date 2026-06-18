"""ingestion/fetcher.py – Discover and fetch source documents via Tavily."""

from __future__ import annotations
import asyncio, hashlib, re
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient
from loguru import logger

from config import settings
from models import DocumentType, SourceDocument

QUERIES: dict[str, list[str]] = {
    "earnings_call": [
        "{company} {quarter} {year} earnings call transcript",
        "{ticker} Q{q} {year} earnings call transcript CEO CFO full",
        "{company} {quarter} {year} quarterly results transcript seekingalpha",
    ],
    "annual_report": [
        "{company} {year} annual report 10-K highlights CEO letter",
    ],
    "investor_presentation": [
        "{company} {quarter} {year} investor day presentation slides",
        "{ticker} {year} capital markets day analyst day",
    ],
    "press_release": [
        "{company} {quarter} {year} earnings press release financial results",
        "{ticker} {quarter} {year} quarterly earnings announcement",
    ],
    "news_article": [
        "{company} {quarter} {year} earnings analysis commentary reaction",
        "{ticker} Q{q} {year} management guidance analyst notes",
    ],
}

TRANSCRIPT_SOURCES = [
    "seekingalpha.com", "motleyfool.com", "fool.com", "rev.com",
    "stockanalysis.com", "finance.yahoo.com", "businesswire.com", "prnewswire.com",
]


def _doc_id(ticker: str, doc_type: str, quarter: str, url: str) -> str:
    return hashlib.md5(f"{ticker}::{doc_type}::{quarter}::{url}".encode()).hexdigest()[:16]


def _guess_quarter(text: str, fallback_q: str = "Q1", fallback_yr: int = 2024) -> tuple[str, int]:
    m = re.search(r'(Q[1-4])\s*(20\d{2})', text, re.I)
    if m: return m.group(1).upper(), int(m.group(2))
    m = re.search(r'(20\d{2})\s*(Q[1-4])', text, re.I)
    if m: return m.group(2).upper(), int(m.group(1))
    return fallback_q, fallback_yr


async def _fetch_html(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200: return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script","style","nav","header","footer","aside"]): tag.decompose()
            return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.debug(f"Fetch failed {url}: {e}")
        return ""


async def _search(tavily: TavilyClient, query: str) -> list[dict]:
    try:
        r = await asyncio.to_thread(
            tavily.search, query=query, max_results=settings.TOP_K_RETRIEVAL,
            search_depth="advanced", include_raw_content=True,
        )
        return r.get("results", [])
    except Exception as e:
        logger.warning(f"Search failed '{query[:50]}': {e}")
        return []


def _infer_doc_type(text: str, url: str) -> DocumentType:
    body = text.lower()
    if "transcript" in body or "earnings call" in body: return DocumentType.EARNINGS_CALL
    if "prnewswire" in url or "businesswire" in url or "press release" in body: return DocumentType.PRESS_RELEASE
    if "investor day" in body or "analyst day" in body or "capital markets day" in body: return DocumentType.INVESTOR_PRESENTATION
    if "annual report" in body or "10-k" in body: return DocumentType.ANNUAL_REPORT
    return DocumentType.NEWS_ARTICLE


async def fetch_documents(
    ticker: str, company: str, quarter: str, year: int,
    doc_types: list[str] | None = None, include_prior: bool = True,
) -> list[SourceDocument]:
    if doc_types is None:
        doc_types = ["earnings_call", "press_release", "investor_presentation", "news_article"]

    q_num = int(quarter[1])
    tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)

    quarters_to_search = [(quarter, q_num, year)]
    if include_prior:
        pq, py = (q_num - 1, year) if q_num > 1 else (4, year - 1)
        quarters_to_search.append((f"Q{pq}", pq, py))

    queries: list[str] = []
    for ql, qn, yr in quarters_to_search:
        for dt in doc_types:
            for tmpl in QUERIES.get(dt, []):
                queries.append(tmpl.format(company=company, ticker=ticker, quarter=ql, year=yr, q=qn))

    all_results: list[dict] = []
    for i in range(0, len(queries), 6):
        batch = await asyncio.gather(*[_search(tavily, q) for q in queries[i:i+6]])
        for res in batch: all_results.extend(res)
        await asyncio.sleep(0.3)

    seen: set[str] = set()
    unique = []
    for r in all_results:
        url = r.get("url","")
        if url and url not in seen:
            seen.add(url); unique.append(r)

    def _rank(r: dict) -> int:
        body = (r.get("content","") + r.get("title","")).lower()
        return sum([
            3 if "transcript" in body else 0,
            2 if "earnings call" in body else 0,
            2 if ticker.lower() in body else 0,
            1 if "q&a" in body else 0,
            2 if any(s in r.get("url","").lower() for s in TRANSCRIPT_SOURCES) else 0,
        ])

    unique.sort(key=_rank, reverse=True)

    async def _enrich(r: dict) -> dict:
        if len(r.get("raw_content","") or "") < 600:
            full = await _fetch_html(r.get("url",""))
            if full: r["raw_content"] = full
        return r

    enriched = await asyncio.gather(*[_enrich(r) for r in unique[:15]])
    docs: list[SourceDocument] = []

    for r in enriched:
        text = r.get("raw_content") or r.get("content","")
        if len(text) < 200: continue
        title = r.get("title",""); url = r.get("url","")
        dtype = _infer_doc_type(text + title, url)
        q_label, yr = _guess_quarter(title + text[:500], quarter, year)
        docs.append(SourceDocument(
            doc_id=_doc_id(ticker, dtype.value, q_label, url),
            ticker=ticker, company=company, doc_type=dtype,
            quarter=q_label, fiscal_year=yr, source_url=url,
            title=title, raw_text=text[:30000],
        ))

    logger.success(f"Fetched {len(docs)} documents for {ticker} {quarter} {year}")
    return docs
