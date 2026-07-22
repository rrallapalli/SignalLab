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


# ── Markdown structure (Docling output) ───────────────────────────────────────
#
# Docling emits markdown with tables preserved as pipe tables. That structure is
# the entire reason for using it: a financial table only means something while
# its header row is still attached to its numbers. So tables are extracted as
# WHOLE blocks before any sentence splitting runs — the sentence splitter would
# otherwise shred them, and it splits on ". " which fires inside "Rs. 500", i.e.
# in the middle of exactly the rows we care about.

_TABLE_LINE = re.compile(r'^\s*\|.*\|\s*$')
_HEADING    = re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$')
# A markdown table's separator row: |---|:---:|
_TABLE_SEP  = re.compile(r'^\s*\|[\s:\-|]+\|\s*$')

# Tables bigger than this get split by rows rather than becoming one huge chunk,
# with the header rows repeated so every part stays self-describing.
#
# Raised from 3000 because splitting is the one case where a chunk stops being a
# verbatim substring of its document (the repeated header is not adjacent to the
# rows it precedes in part 2 onwards), which makes citations from that chunk
# unverifiable. Most financial tables now fit whole and keep the invariant;
# only genuinely huge ones trade it away for readable fragments.
MAX_TABLE_CHARS = 6000


def _split_blocks(text: str) -> list[tuple[str, str, str]]:
    """
    Split markdown into ("table"|"prose", block_text, heading) blocks.

    `heading` is the nearest preceding markdown heading, carried along so a bare
    table of numbers still knows it sits under "Consolidated Financial Results"
    — without it, a table chunk is unattributable and the section classifier has
    nothing but digits to work with.

    Text with no markdown structure (e.g. the pypdf fallback) comes back as a
    single prose block, so behaviour is unchanged for non-Docling input.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str, str]] = []
    buf: list[str] = []
    buf_kind = "prose"
    heading = ""

    def flush():
        nonlocal buf, buf_kind
        body = "\n".join(buf).strip()
        if body:
            blocks.append((buf_kind, body, heading))
        buf = []

    for line in lines:
        is_table = bool(_TABLE_LINE.match(line) or _TABLE_SEP.match(line))
        kind = "table" if is_table else "prose"

        if kind != buf_kind:
            flush()
            buf_kind = kind

        if kind == "prose":
            m = _HEADING.match(line)
            if m:
                # A new heading ends the current prose block and becomes the
                # context for whatever follows.
                flush()
                heading = m.group(1).strip()

        buf.append(line)

    flush()
    return blocks


def _split_table(table_md: str) -> list[str]:
    """
    Break an oversized table into row groups, repeating the header rows in each
    part. A table fragment without its header is just a grid of numbers.
    """
    rows = [r for r in table_md.splitlines() if r.strip()]
    if not rows:
        return []

    # Header = leading rows up to and including the separator, when present.
    header: list[str] = []
    body_start = 0
    for i, r in enumerate(rows[:3]):
        header.append(r)
        if _TABLE_SEP.match(r):
            body_start = i + 1
            break
    else:
        header, body_start = rows[:1], 1

    body = rows[body_start:]
    if not body:
        return ["\n".join(rows)]

    parts: list[str] = []
    current: list[str] = []
    header_len = sum(len(h) for h in header)
    current_len = header_len

    for row in body:
        if current and current_len + len(row) > MAX_TABLE_CHARS:
            parts.append("\n".join(header + current))
            current, current_len = [], header_len
        current.append(row)
        current_len += len(row)

    if current:
        parts.append("\n".join(header + current))

    return parts


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

    def _emit(text_: str, section_: DocumentSection, speaker_: str, is_mgmt_: bool,
              heading: str = "") -> None:
        nonlocal char_offset
        chunk_id = hashlib.md5(
            f"{doc.doc_id}::{char_offset}::{text_[:40]}".encode()
        ).hexdigest()[:16]
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            ticker=doc.ticker,
            company=doc.company,
            doc_type=doc.doc_type,
            section=section_,
            quarter=doc.quarter,
            fiscal_year=doc.fiscal_year,
            event_date=doc.event_date,
            speaker=speaker_,
            is_management=is_mgmt_,
            text=text_,
            char_start=char_offset,
            source_url=doc.source_url,
            # Heading rides in the title so a bare table of figures is still
            # attributable, without contaminating the verbatim text.
            title=(f"{doc.title} — {heading}"[:200] if heading else doc.title),
        ))
        char_offset += len(text_)

    for kind, block, heading in _split_blocks(doc.raw_text):
        if kind == "table":
            # Tables bypass speaker extraction and sentence splitting entirely.
            # Section comes from the heading above the table plus the table's own
            # text, since a results table's giveaway words ("Revenue from
            # operations", "Total income") live in the row labels.
            # The heading informs the SECTION tag and the chunk title, but is
            # deliberately NOT prepended to the text. A document rarely places a
            # heading immediately before its table — here Infosys puts "in US $
            # million, except per equity share data" between them — so a chunk
            # built as heading + table exists nowhere in the source, and every
            # citation drawn from it is unverifiable. Chunk text stays a verbatim
            # span; context travels in metadata, where it costs nothing.
            section = _detect_section(f"{heading}\n{block}")
            for part in (
                [block] if len(block) <= MAX_TABLE_CHARS else _split_table(block)
            ):
                if len(part.strip()) > 50:
                    # Tables are management-authored disclosure, not analyst
                    # commentary — mark them so `management_only` retrieval
                    # (which the guidance agent uses) can actually see them.
                    _emit(part.strip(), section, "", True, heading=heading)
            continue

        for speaker, block_text, is_mgmt in _extract_speaker_blocks(block):
            section = _detect_section(block_text if not heading else f"{heading}\n{block_text}")
            for chunk_text in _split_sentences(block_text, chunk_size, overlap):
                _emit(chunk_text, section, speaker, is_mgmt)

    return chunks
