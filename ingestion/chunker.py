"""
ingestion/chunker.py
Smart chunker that tags chunks with rich metadata:
  - speaker, section, doc_type, quarter, is_management
Preserves sentence boundaries and speaker attribution.
"""

from __future__ import annotations
import hashlib, re
from models import DocumentChunk, DocumentSection, SourceDocument


# ── Section detection patterns ────────────────────────────────────────────────

SECTION_PATTERNS: list[tuple[re.Pattern, DocumentSection]] = [
    (re.compile(r'\b(prepared remarks?|opening remarks?|management discussion)\b', re.I), DocumentSection.PREPARED_REMARKS),
    (re.compile(r'\b(question[s]? and answer|q&a|question[s]? from|operator:.*question)\b', re.I), DocumentSection.QA_SESSION),
    (re.compile(r'\b(financial results?|financial highlights?|revenue|earnings per share|net income)\b', re.I), DocumentSection.FINANCIAL_RESULTS),
    (re.compile(r'\b(guidance|outlook|forecast|expect.*quarter|full[- ]year.*expect)\b', re.I), DocumentSection.GUIDANCE),
    (re.compile(r'\b(risk factors?|risk[s]? include|material risk)\b', re.I), DocumentSection.RISK_FACTORS),
    (re.compile(r'\b(strategy|strategic|long[- ]term|roadmap|priorities)\b', re.I), DocumentSection.STRATEGY),
    (re.compile(r'\b(market|industry|macro|economic environment|sector)\b', re.I), DocumentSection.MARKET_OVERVIEW),
]

# ── Speaker detection ─────────────────────────────────────────────────────────

MGMT_ROLES = re.compile(
    r'\b(CEO|Chief Executive|CFO|Chief Financial|COO|Chief Operating|'
    r'CTO|Chief Technology|Chairman|President|EVP|SVP|'
    r'Executive Vice President|Senior Vice President)\b', re.I
)
SPEAKER_LINE = re.compile(
    r'^([A-Z][a-zA-Z\s\-]{2,40})\s*[:\-–]\s*(.*)$', re.M
)
# Common operator/intro phrases that aren't speakers
NON_SPEAKERS = re.compile(
    r'^(operator|moderator|presentation|conference|unidentified|unknown)', re.I
)


def _detect_section(text: str) -> DocumentSection:
    for pattern, section in SECTION_PATTERNS:
        if pattern.search(text):
            return section
    return DocumentSection.UNKNOWN


def _extract_speaker_blocks(text: str) -> list[tuple[str, str, bool]]:
    """
    Returns list of (speaker, block_text, is_management).
    Falls back to treating the whole text as one block if no speaker markers.
    """
    blocks = []
    matches = list(SPEAKER_LINE.finditer(text))

    if not matches:
        return [("", text, False)]

    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        if NON_SPEAKERS.match(speaker):
            is_mgmt = False
        else:
            is_mgmt = bool(MGMT_ROLES.search(text[max(0, m.start()-200):m.start()+200]))

        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start():end].strip()
        blocks.append((speaker, block, is_mgmt))

    return blocks


def _split_sentences(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping sentence-boundary-respecting chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_sents, overlap_len = [], 0
            for s in reversed(current):
                overlap_len += len(s)
                if overlap_len >= overlap: break
                overlap_sents.insert(0, s)
            current = overlap_sents
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 50]


def chunk_document(
    doc: SourceDocument,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """
    Main entry point. Converts a SourceDocument into tagged DocumentChunks.
    Each chunk inherits doc metadata + speaker + section + is_management.
    """
    chunks: list[DocumentChunk] = []
    char_offset = 0

    speaker_blocks = _extract_speaker_blocks(doc.raw_text)

    for speaker, block_text, is_mgmt in speaker_blocks:
        section = _detect_section(block_text)

        for chunk_text in _split_sentences(block_text, chunk_size, overlap):
            chunk_id = hashlib.md5(
                f"{doc.doc_id}::{char_offset}::{chunk_text[:40]}".encode()
            ).hexdigest()[:16]

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                ticker=doc.ticker,
                company=doc.company,
                doc_type=doc.doc_type,
                section=section,
                quarter=doc.quarter,
                fiscal_year=doc.fiscal_year,
                event_date=doc.event_date,
                speaker=speaker,
                is_management=is_mgmt,
                text=chunk_text,
                char_start=char_offset,
                source_url=doc.source_url,
                title=doc.title,
            ))
            char_offset += len(chunk_text)

    return chunks
