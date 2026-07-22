"""
ingestion/parser.py
PDF → text extraction via Docling, fronted by an on-disk parse cache.

WHY THIS EXISTS
---------------
pypdf returns a flat text stream: a financial table comes back with its cells
in reading order, so "Revenue 12,345" loses the tie between label and figure
and the guidance agent has nothing reliable to read. Docling does layout-aware
extraction and emits markdown with tables intact, which is the whole point —
a table chunk keeps its header row attached to its numbers.

pypdf also returns ~nothing for scanned/image-only filings (common for older
BSE attachments and signed letters). The fetcher drops any doc whose text is
under 200 chars, so those documents were being silently discarded — fetched
successfully, never read, never counted as missing. OCR recovers them, but
OCR is slow, so it only runs when the fast path comes back empty.

Parsing is the most expensive step per document and its result never changes,
so it is cached on disk keyed by URL *and parser version*. Bumping
PARSER_VERSION invalidates every cached parse — without that, upgrading Docling
would silently keep serving text produced by the old version.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings

# Bump this whenever the parsing behaviour changes (new Docling version, new
# pipeline options, different post-processing). Cached entries whose parser tag
# doesn't match are treated as misses and re-parsed.
PARSER_VERSION = "docling-v1"

# Text shorter than this means the fast path found no usable text layer —
# i.e. the PDF is probably scanned images. Worth paying for OCR.
_EMPTY_TEXT_THRESHOLD = 200


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _cache_path(url: str) -> Path:
    return settings.PARSE_CACHE_DIR / f"{_cache_key(url)}.json"


def cache_get(url: str) -> Optional[str]:
    """Return cached text for this URL, or None on miss / stale parser version."""
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"[parse-cache] Unreadable entry {path.name}: {e}")
        return None

    # Page limit is part of what produced this text: raising it should yield
    # more content, so a cached parse made under a different limit is stale.
    if rec.get("page_limit") != settings.DOCLING_PAGE_LIMIT:
        logger.debug(f"[parse-cache] Page-limit change for {url[:50]} — re-parsing.")
        return None

    if rec.get("parser_version") != PARSER_VERSION:
        logger.debug(
            f"[parse-cache] Stale entry for {url[:60]} "
            f"({rec.get('parser_version')} != {PARSER_VERSION}) — re-parsing."
        )
        return None

    text = rec.get("text") or ""
    return text or None


def cache_put(url: str, text: str, how: str) -> None:
    """Persist parsed text. Failures here are non-fatal — the caller has the text."""
    try:
        settings.PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(url).write_text(
            json.dumps({
                "url": url,
                "parser_version": PARSER_VERSION,
                "page_limit": settings.DOCLING_PAGE_LIMIT,
                "how": how,
                "chars": len(text),
                "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "text": text,
            }),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[parse-cache] Could not write entry for {url[:60]}: {e}")


def _pdf_cache_path(url: str) -> Path:
    return settings.PDF_CACHE_DIR / f"{_cache_key(url)}.pdf"


def pdf_cache_get(url: str) -> Optional[bytes]:
    """
    Raw downloaded bytes, cached separately from the parsed text.

    Parsed text is invalidated whenever parsing behaviour changes (parser
    version, page limit). Without this the invalidation would also force a
    re-download of every PDF, making it expensive to tune parse settings. With
    it, re-parsing is a purely local operation.
    """
    path = _pdf_cache_path(url)
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        return data or None
    except Exception:
        return None


def pdf_cache_put(url: str, data: bytes) -> None:
    if not data:
        return
    try:
        settings.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _pdf_cache_path(url).write_bytes(data)
    except Exception as e:
        logger.debug(f"[pdf-cache] Could not store {url[:60]}: {e}")


def cache_stats() -> dict:
    """Small helper for diagnostics: how many parses are cached, and how big."""
    try:
        files = list(settings.PARSE_CACHE_DIR.glob("*.json"))
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}
    except Exception:
        return {"entries": 0, "bytes": 0}


# ── Docling ───────────────────────────────────────────────────────────────────

_converters: dict[bool, object] = {}
# Building a converter loads the full layout + TableFormer weights. Documents
# are parsed from many worker threads at once, so without this lock every
# thread that arrives before the first one finishes builds its OWN converter —
# a dozen copies of the model loaded and held in memory simultaneously, which
# is what turned a laptop run into a swap storm.
_converter_lock = threading.Lock()

# Parsing is CPU-bound. asyncio.to_thread will happily run ~12 of them in
# parallel, which on a laptop is slower than running two, because they contend
# for the same cores and multiply peak memory. Bound it explicitly.
_parse_slots = threading.BoundedSemaphore(max(1, settings.PARSE_CONCURRENCY))

# The converter is a shared object wrapping NATIVE state: the PDF backend
# (pypdfium2), the torch layout/table models, and — when the accelerator picks
# MPS — a Metal context. None of those are thread-safe. Calling convert() from
# two threads at once corrupts the allocator's free list and takes the whole
# process down with "malloc: Heap corruption detected", which is a hard crash,
# not a catchable exception.
#
# So conversion is serialized here rather than left to PARSE_CONCURRENCY. The
# semaphore above provides backpressure; this lock provides safety, and safety
# must not depend on a value someone can raise in .env. Real parsing
# parallelism needs separate PROCESSES, not threads.
_convert_lock = threading.Lock()


def _get_converter(ocr: bool):
    """
    Build (once per OCR mode) a Docling converter.

    Deliberately the STANDARD layout+TableFormer pipeline, not the VLM one:
    the VLM pipeline reads each page as an image and generates table structure
    token by token, which is minutes-per-document on a laptop CPU and has been
    reported to hallucinate values on dense numeric tables. Standard pipeline
    is far faster and does not invent cells.
    """
    if ocr in _converters:              # fast path, no lock
        return _converters[ocr]

    with _converter_lock:
        if ocr in _converters:          # another thread built it while we waited
            return _converters[ocr]
        return _build_converter(ocr)


def _accelerator_options():
    """
    Hardware acceleration for the layout/table models.

    Import path moved between Docling versions, and AUTO is safer than forcing
    MPS: some models declare CPU/CUDA-only support and Docling drops MPS for
    those rather than failing. AUTO picks the GPU where it is actually
    supported and quietly uses CPU where it is not.
    """
    AcceleratorDevice = AcceleratorOptions = None
    for mod in ("docling.datamodel.accelerator_options",
                "docling.datamodel.pipeline_options"):
        try:
            m = __import__(mod, fromlist=["AcceleratorDevice", "AcceleratorOptions"])
            AcceleratorDevice = getattr(m, "AcceleratorDevice")
            AcceleratorOptions = getattr(m, "AcceleratorOptions")
            break
        except Exception:
            continue
    if AcceleratorOptions is None:
        return None

    try:
        device = AcceleratorDevice((settings.DOCLING_DEVICE or "auto").lower())
    except Exception:
        device = getattr(AcceleratorDevice, "AUTO", None)

    try:
        return AcceleratorOptions(
            num_threads=max(1, settings.DOCLING_NUM_THREADS), device=device
        )
    except Exception:
        return None


def _build_converter(ocr: bool):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = ocr
    opts.do_table_structure = True
    try:
        # Match recovered cell text back to the PDF's own text layer where it
        # exists — keeps numbers exact instead of re-recognised.
        opts.table_structure_options.do_cell_matching = True
    except Exception:
        pass

    # Rendered page images are only needed for VLM/picture work. We export
    # markdown, so generating them is pure cost — time and a lot of memory.
    for flag in ("generate_page_images", "generate_picture_images"):
        try:
            setattr(opts, flag, False)
        except Exception:
            pass

    accel = _accelerator_options()
    if accel is not None:
        try:
            opts.accelerator_options = accel
            logger.info(f"[parse] Accelerator: device={settings.DOCLING_DEVICE} "
                        f"threads={settings.DOCLING_NUM_THREADS}")
        except Exception as e:
            logger.warning(f"[parse] Could not apply accelerator options ({e}) — running on CPU.")
    else:
        # Silence here previously meant "no GPU and no way to tell" — which is
        # the difference between a fast run and a very slow one.
        logger.warning(
            "[parse] Accelerator options unavailable for this Docling version — "
            "running on CPU. Try setting DOCLING_DEVICE / OMP_NUM_THREADS in the "
            "environment instead."
        )

    logger.info(f"[parse] Loading Docling models (ocr={ocr}) — first document only…")
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    _converters[ocr] = conv
    logger.success("[parse] Docling ready.")
    return conv


def warm_up() -> None:
    """
    Build the converter once, up front, on a single thread.

    Without this the first N concurrent documents all race into _get_converter
    together; the lock now makes them queue rather than duplicate, but they
    still all sit blocked waiting for one slow model load. Warming first means
    the load happens once, visibly, before any parallel work starts.
    """
    if not settings.USE_DOCLING:
        return
    try:
        _get_converter(False)
    except Exception as e:
        logger.warning(f"[parse] Warm-up failed ({e}); will fall back per-document.")


def _page_count(pdf_bytes: bytes) -> int:
    """Cheap page count via pypdf, used to skip Docling on very long documents."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception:
        return 0


def _docling_markdown(pdf_bytes: bytes, name: str, ocr: bool) -> str:
    from docling.datamodel.base_models import DocumentStream

    conv = _get_converter(ocr)
    source = DocumentStream(name=name, stream=io.BytesIO(pdf_bytes))

    # Cap pages rather than the whole document: in a results release or investor
    # deck the tables that matter are near the front, so parsing the first N
    # pages gets the evidence at a fraction of the cost of the full file.
    #
    # This must be page_range, NOT max_num_pages. max_num_pages is a validation
    # ceiling — a document longer than it is REJECTED ("Document has 46 pages,
    # exceeding the max_num_pages limit of 25", status: failure), which quietly
    # pushed every long investor presentation onto the pypdf fallback: the
    # table-heavy files Docling was added for were the ones not getting it.
    # page_range is 1-indexed and inclusive.
    limit = settings.DOCLING_PAGE_LIMIT
    result = None
    with _convert_lock:
        if limit and limit > 0:
            try:
                result = conv.convert(source, page_range=(1, limit))
            except TypeError:
                # Older Docling has no page_range kwarg — the stream has been
                # consumed by the attempt, so rebuild it before retrying.
                source = DocumentStream(name=name, stream=io.BytesIO(pdf_bytes))
        if result is None:
            result = conv.convert(source)
    return (result.document.export_to_markdown() or "").strip()


def _pypdf_text(pdf_bytes: bytes, max_pages: int) -> str:
    """Last-resort fallback so a Docling failure never costs us the document."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(
        (page.extract_text() or "") for page in reader.pages[:max_pages]
    ).strip()


# ── Public entry point ────────────────────────────────────────────────────────

def _docling_applies(doc_type: str | None) -> bool:
    """
    Docling buys exactly one thing: table structure. An earnings call transcript
    is prose from end to end, so layout-parsing it costs minutes and yields text
    pypdf already produced. Restricting Docling to the table-bearing document
    types is the single largest saving available without losing anything the
    signal agents actually read.

    Empty DOCLING_DOC_TYPES means "every type".
    """
    allowed = [t.strip() for t in settings.DOCLING_DOC_TYPES.split(",") if t.strip()]
    if not allowed:
        return True
    return (doc_type or "") in allowed


def extract_text(
    pdf_bytes: bytes, name: str = "document.pdf", doc_type: str | None = None
) -> tuple[str, str]:
    """
    Extract text from PDF bytes. Returns (text, how) where `how` records which
    path produced it — useful when a quarter's evidence looks wrong and you need
    to know whether it came from a clean text layer, OCR, or the fallback.

    Order: Docling (no OCR) → Docling (OCR, only if the first came back empty
    and OCR is enabled) → pypdf. Never raises; returns ("", "failed") instead,
    because one unreadable filing should not abort a whole quarter's ingest.
    """
    if not pdf_bytes:
        return "", "empty"

    pages = _page_count(pdf_bytes)
    if settings.USE_DOCLING and not _docling_applies(doc_type):
        logger.debug(
            f"[parse] {name}: doc_type={doc_type} is not table-bearing — "
            f"using fast pypdf extraction."
        )
    elif settings.USE_DOCLING and 0 < settings.DOCLING_MAX_PAGES < pages:
        # Annual reports and similar run to hundreds of pages. Layout-parsing
        # those on a laptop costs minutes each for evidence the agents rarely
        # quote; take the fast flat text instead of stalling the whole run.
        logger.info(
            f"[parse] {name}: {pages} pages exceeds DOCLING_MAX_PAGES "
            f"({settings.DOCLING_MAX_PAGES}) — using fast pypdf extraction."
        )
    elif settings.USE_DOCLING:
        try:
            with _parse_slots:
                text = _docling_markdown(pdf_bytes, name, ocr=False)
            if len(text) >= _EMPTY_TEXT_THRESHOLD:
                return text, "docling"
            logger.debug(
                f"[parse] {name}: text layer yielded {len(text)} chars — "
                f"likely a scanned document."
            )
            if settings.DOCLING_OCR_FALLBACK:
                logger.info(f"[parse] {name}: retrying with OCR (scanned document).")
                with _parse_slots:
                    ocr_text = _docling_markdown(pdf_bytes, name, ocr=True)
                if len(ocr_text) >= _EMPTY_TEXT_THRESHOLD:
                    return ocr_text, "docling+ocr"
                return ocr_text, "docling+ocr-thin"
            return text, "docling-thin"
        except ImportError:
            logger.warning(
                "[parse] Docling not installed — falling back to pypdf. "
                "Install it with: pip install docling"
            )
        except Exception as e:
            logger.warning(f"[parse] Docling failed on {name} ({e}) — falling back to pypdf.")

    try:
        return _pypdf_text(pdf_bytes, settings.PDF_MAX_PAGES), "pypdf"
    except Exception as e:
        logger.debug(f"[parse] pypdf also failed on {name}: {e}")
        return "", "failed"


def extract_text_cached(
    pdf_bytes: bytes, url: str, name: str = "document.pdf", doc_type: str | None = None
) -> tuple[str, str]:
    """extract_text() with the on-disk cache in front of it."""
    cached = cache_get(url)
    if cached is not None:
        return cached, "cache"

    text, how = extract_text(pdf_bytes, name=name, doc_type=doc_type)
    if text:
        cache_put(url, text, how)
    return text, how
