"""
ui/dashboard.py – Latest / QoQ / YoY comparison dashboard.
Run: streamlit run ui/dashboard.py
"""

from __future__ import annotations
import sys, json, asyncio, threading, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from store.signal_store import SignalStore
from store.vector_store import VectorStore


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Signal Intelligence",
    page_icon="📡",
    layout="wide",
    # "auto" = hidden on small devices, shown otherwise. "expanded" made the
    # sidebar cover the whole screen on a phone before you could see anything.
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
  :root {
    --bg-app:      #0b0e17;
    --bg-panel:    #141a2e;
    --bg-card:     #171d33;
    --bg-card-alt: #10162a;
    --border:      #2a3252;
    --text-hi:     #f1f5f9;
    --text-mid:    #cbd5e1;
    --text-low:    #8b95b3;
    --text-faint:  #4b5573;
    --accent:      #6366f1;
    --accent-soft: #4338ca;
    --good:        #34d399;
    --warn:        #fbbf24;
    --bad-soft:    #fb923c;
    --bad:         #f87171;
  }

  /* Streamlit fades elements to ~60-70% opacity while a script rerun is in
     flight (its built-in "stale content" indicator). Combined with a dark
     custom theme this reads as permanently washed-out, low-contrast text —
     so we force full opacity everywhere and rely on our own spinners/status
     messages to communicate "in progress" instead. */
  [data-stale="true"], .element-container.stale-element {
    opacity: 1 !important;
  }

  [data-testid="stSidebar"] {
    background: var(--bg-panel);
    overflow-y: auto;
    max-height: 100vh;
  }
  [data-testid="stSidebarContent"] { overflow: visible; padding-bottom: 2rem; }
  [data-testid="stAppViewContainer"] > .main { background: var(--bg-app); }

  /* Explicit high-contrast text colors for the sidebar — don't rely on
     Streamlit's default theme, which is tuned for a light background and
     looks washed out against ours. Deliberately scoped to labels/headers/
     captions that sit directly on the dark sidebar background — NOT to
     input/select internals, which already render dark-text-on-white and
     would become unreadable under a blanket white-text rule. */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
    color: var(--text-hi) !important; opacity: 1 !important;
  }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--text-low) !important; opacity: 1 !important;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-hi) !important;
  }
  [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { opacity: 0.85 !important; }
  [data-testid="stSidebar"] hr { border-color: var(--border) !important; opacity: 1 !important; }

  .col-header {
    text-align:center; font-size:0.78rem; font-weight:700;
    letter-spacing:0.06em; text-transform:uppercase;
    padding:0.55rem 0; border-radius:8px; margin-bottom:0.5rem;
    border:1px solid var(--border);
  }
  .col-latest  { background:#241a4d; color:#c7b8ff; border-color:#3d2d7a; }
  .col-qoq     { background:#0d2530; color:#7dd3e8; border-color:#1c4a5a; }
  .col-yoy     { background:#0f2818; color:#8fe0ac; border-color:#20502e; }

  .score-big   { font-size:2.3rem; font-weight:800; text-align:center; letter-spacing:-0.02em; }
  .delta-up    { color:var(--good); font-size:0.95rem; font-weight:700; text-align:center; }
  .delta-down  { color:var(--bad);  font-size:0.95rem; font-weight:700; text-align:center; }
  .delta-flat  { color:var(--text-low); font-size:0.95rem; font-weight:700; text-align:center; }

  .signal-card {
    background: var(--bg-card); border-radius:12px;
    padding:1rem 1.2rem; margin-bottom:0.8rem;
    border:1px solid var(--border); border-left:4px solid var(--accent);
  }
  .driver-item {
    padding:0.55rem 0.8rem; background: var(--bg-card-alt); border-radius:8px;
    border:1px solid var(--border); border-left:3px solid var(--accent);
    margin:0.35rem 0; color: var(--text-mid); font-size:0.87rem;
  }
  .risk-card {
    background: var(--bg-card); border-radius:10px; padding:0.8rem 1rem;
    margin:0.4rem 0; border:1px solid var(--border); border-left:3px solid var(--bad);
  }
  .status-pill {
    display:inline-block; padding:0.18rem 0.65rem; border-radius:999px;
    font-size:0.74rem; font-weight:700; letter-spacing:0.01em;
  }
  .pill-green  { background:#0f3d2c; color:#5eead4; }
  .pill-red    { background:#431515; color:#fca5a5; }
  .pill-yellow { background:#3a2a05; color:#fcd34d; }
  .pill-purple { background:#2a1f5c; color:#c4b5fd; }
  .pill-orange { background:#3d1f0a; color:#fdba74; }

  /* Explanatory captions under section headers */
  .section-help {
    color: var(--text-low); font-size:0.83rem; line-height:1.5;
    margin: -0.3rem 0 0.9rem 0; padding: 0.5rem 0.8rem;
    background: var(--bg-card-alt); border-radius:8px; border:1px solid var(--border);
  }

  /* Inline "ⓘ" tooltip.
     Driven by a data-* attribute, NOT title=. The mobile cells already prove
     data-* attributes survive Streamlit's html sanitiser (the quarter labels
     render from attr(data-q)), whereas title= produced no tooltip at all.
     This also gives a styled, instant tooltip instead of the browser's slow
     default, and :focus makes it work on tap as well as hover. */
  .info-tip {
    position:relative; cursor:help; color:var(--text-low); font-size:0.8rem;
    border-bottom:1px dotted var(--text-faint); outline:none;
  }
  .info-tip:hover, .info-tip:focus { color:var(--accent); }
  .info-tip::after {
    content: attr(data-tip);
    position:absolute; left:0; top:150%; z-index:9999;
    width:max-content; max-width:min(300px, 70vw);
    background:#080c18; border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:6px;
    padding:0.55rem 0.7rem;
    font-size:0.75rem; line-height:1.5; font-weight:400;
    color:var(--text-mid); text-align:left; white-space:normal;
    box-shadow:0 8px 24px rgba(0,0,0,0.6);
    opacity:0; visibility:hidden; transition:opacity 0.12s ease;
    pointer-events:none;
  }
  .info-tip:hover::after, .info-tip:focus::after { opacity:1; visibility:visible; }

  /* Small colour-coded legend chips explaining the gauge thresholds */
  .legend-row { display:flex; gap:1.1rem; flex-wrap:wrap; font-size:0.78rem; color: var(--text-low); margin: 0.2rem 0 0.8rem 0; }
  .legend-dot { display:inline-block; width:0.55rem; height:0.55rem; border-radius:50%; margin-right:0.35rem; vertical-align:middle; }

  /* ══════════════════════════════════════════════════════════════════════
     RESPONSIVE COMPARISON GRID
     ══════════════════════════════════════════════════════════════════════
     st.columns() is a desktop layout instruction with no responsive escape
     hatch — Streamlit owns that markup, so a 4-across comparison just gets
     squeezed to ~90px a column on a phone. The comparison sections are
     emitted as one HTML block instead, so media queries can restack them:

       desktop  →  label | Latest | QoQ | YoY   (4 across)
       mobile   →  one card per metric, each quarter on its own line
  */
  .cmp { display:flex; flex-direction:column; gap:0.35rem; margin:0.4rem 0 0.2rem; }
  .cmp-row {
    display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr;
    gap:0.5rem; align-items:center; padding:0.35rem 0;
  }
  .cmp-row + .cmp-row { border-top:1px solid var(--border); }
  .cmp-label { color:var(--text-mid); font-size:0.85rem; font-weight:600; }
  .cmp-cell { text-align:center; }
  .cmp-cell.empty { color:var(--text-faint); font-size:0.85rem; }
  .cmp-val   { }              /* value block — the delta sits under it */
  /* The delta slot reserves its line height even when empty. Without this the
     Latest cell (no delta) is shorter than the QoQ/YoY cells, and align-items
     :center then drops its score lower than theirs. */
  .cmp-delta { min-height:1.25rem; }
  .cmp-qhead { font-size:0.78rem; font-weight:700; text-align:center; margin-bottom:0.15rem; }
  .q-latest { color:#a78bfa; }
  .q-qoq    { color:#67e8f9; }
  .q-yoy    { color:#86efac; }

  /* "Why?" evidence disclosure under each score. Native <details>, so it
     works on click and tap with no JS, on every screen size. */
  .cmp-why { margin-top:0.35rem; text-align:left; }
  .cmp-why summary {
    cursor:pointer; list-style:none; font-size:0.72rem;
    color:var(--text-faint); text-align:center; padding:0.15rem 0;
  }
  .cmp-why summary::-webkit-details-marker { display:none; }
  .cmp-why summary::after { content:'ⓘ why'; }
  .cmp-why[open] summary::after { content:'▾ hide'; color:var(--text-low); }
  .cmp-why summary:hover { color:var(--accent); }
  .why-body {
    background:var(--bg-card-alt); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:6px;
    padding:0.55rem 0.65rem; margin-top:0.3rem;
    font-size:0.78rem; line-height:1.45; color:var(--text-mid);
  }
  .why-summary { margin-bottom:0.35rem; }
  .why-driver  { color:var(--text-low); font-size:0.75rem; }
  .why-cite {
    margin-top:0.4rem; padding-top:0.35rem; border-top:1px solid var(--border);
    font-style:italic; color:var(--text-mid); font-size:0.76rem;
  }
  .why-meta { font-style:normal; color:var(--text-faint); font-size:0.68rem; margin-top:0.15rem; }
  .why-link { color:#67e8f9; text-decoration:none; }

  /* Mini bars used by the sub-dimension rows */
  .cmp-bar-track { background:#0f172a; border-radius:3px; height:6px; margin-top:0.2rem; }
  .cmp-bar-fill  { height:6px; border-radius:3px; }

  /* ── Phones ─────────────────────────────────────────────────────────── */
  @media (max-width: 640px) {

    /* Comparison grid → stacked cards, one per metric */
    .cmp-row {
      display:block; border:1px solid var(--border); border-radius:10px;
      background:var(--bg-card-alt); padding:0.7rem 0.8rem; margin-bottom:0.55rem;
    }
    .cmp-label {
      display:block; font-size:0.95rem; color:var(--text-hi);
      margin-bottom:0.5rem; padding-bottom:0.35rem; border-bottom:1px solid var(--border);
    }
    /* Each quarter on its own line: label · value · delta.
       Fixed-width slots rather than space-between — otherwise the value's
       x-position drifts with the length of its neighbours. */
    .cmp-cell {
      display:flex; align-items:center; gap:0.6rem;
      padding:0.35rem 0; min-height:2.2rem;
    }
    .cmp-cell::before {
      content: attr(data-q);
      flex:0 0 7.5rem;                /* label slot */
      font-size:0.78rem; font-weight:700;
      color:var(--text-low); text-align:left;
    }
    .cmp-cell { flex-wrap:wrap; }         /* lets .cmp-why drop to its own line */
    .cmp-val   { flex:1 1 auto; text-align:right; }
    /* Mobile cells are flex ROWS, so the desktop min-height reservation isn't
       needed — the fixed slot width does the aligning here. */
    .cmp-delta { flex:0 0 3.6rem; text-align:right; font-size:0.8rem; min-height:0; }
    .cmp-why   { flex:1 1 100%; margin-top:0.1rem; }
    .cmp-why summary { text-align:right; padding:0.35rem 0; }   /* thumb target */
    .cmp-cell[data-tone="latest"]::before { color:#a78bfa; }
    .cmp-cell[data-tone="qoq"]::before    { color:#67e8f9; }
    .cmp-cell[data-tone="yoy"]::before    { color:#86efac; }
    .cmp-bar-track { flex:1 1 auto; min-width:60px; }
    .cmp-qhead { display:none; }      /* headers move inline into each cell */

    /* Per-quarter st.columns(3) blocks already carry their own labels —
       they only need to stop being squeezed side by side. */
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:0.5rem; }
    [data-testid="stColumn"] { min-width:100% !important; flex:1 1 100% !important; }

    /* Reclaim horizontal space Streamlit reserves for desktop */
    .block-container { padding:1rem 0.75rem 3rem !important; }
    h1 { font-size:1.45rem !important; }
    h2 { font-size:1.2rem !important; }
    h3 { font-size:1.05rem !important; }
    .section-help { font-size:0.78rem; line-height:1.45; }
    .col-header  { font-size:0.8rem; }
    .signal-card { padding:0.7rem; }
    .legend-row  { gap:0.6rem; font-size:0.72rem; }

    /* Tap targets */
    .stButton button { min-height:2.75rem; font-size:0.95rem; }
    [data-testid="stSidebar"] .stButton button { width:100%; }

    /* Three redundant gauges stacked is noise — the score is already text
       right above each one. Hide ONLY the gauge rows (tagged via
       st.container(key="gauge-row-*")), never all Plotly charts: the Trend
       tab's line charts are its entire content. */
    .hide-mobile { display:none !important; }
    [class*="st-key-gauge-row"] { display:none !important; }

    /* Wide tables scroll instead of overflowing the viewport */
    [data-testid="stDataFrame"] { overflow-x:auto; }
  }

  /* ── Small tablets ──────────────────────────────────────────────────── */
  @media (min-width: 641px) and (max-width: 900px) {
    .cmp-row { grid-template-columns:1.2fr 1fr 1fr 1fr; gap:0.35rem; }
    .cmp-label { font-size:0.8rem; }
    .block-container { padding-left:1.5rem !important; padding-right:1.5rem !important; }
  }
</style>
""", unsafe_allow_html=True)


# ── Async helper ──────────────────────────────────────────────────────────────

def _run_async(coro):
    result, exc = [None], [None]
    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:    result[0] = loop.run_until_complete(coro)
        except Exception as e: exc[0] = e
        finally: loop.close()
    t = threading.Thread(target=_worker, daemon=True)
    t.start(); t.join()
    if exc[0]: raise exc[0]
    return result[0]


# ── Shared resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _running_tickers() -> tuple[set, threading.Lock]:
    """
    Tickers with a pipeline in flight, shared across all browser sessions.
    Streamlit runs each session's script in its own thread inside one process,
    so two people (or two tabs) can start the same ticker at once and
    double-write the same rows. This is the guard against that.
    """
    return set(), threading.Lock()


@st.cache_resource
def _stores():
    return VectorStore(), SignalStore()

vs, ss = _stores()


# ── Score helpers ─────────────────────────────────────────────────────────────

def _color(score: float, max_val: float = 10) -> str:
    p = score / max_val
    return "#34d399" if p >= 0.75 else "#fbbf24" if p >= 0.55 else "#fb923c" if p >= 0.35 else "#f87171"


def _delta_html(current: float, reference: float, invert: bool = False) -> str:
    """Return coloured delta HTML between two scores."""
    if reference == 0: return ""
    delta = current - reference
    if invert: delta = -delta   # for risk: lower = better
    sym  = "▲" if delta > 0 else "▼" if delta < 0 else "—"
    cls  = "delta-up" if delta > 0 else "delta-down" if delta < 0 else "delta-flat"
    return f"<div class='{cls}'>{sym} {abs(delta):.1f}</div>"


def _score_cell(score: float | None, max_val: float = 10) -> str:
    if score is None: return "<div class='score-big' style='color:#475569'>—</div>"
    c = _color(score, max_val)
    return f"<div class='score-big' style='color:{c}'>{score:.1f}</div>"


def _pill(text: str, cls: str) -> str:
    return f"<span class='status-pill {cls}'>{text}</span>"


# ── Responsive comparison grid ────────────────────────────────────────────────
#
# Renders label | Latest | QoQ | YoY as ONE html block rather than st.columns(),
# so CSS can restack it into per-metric cards on a phone. See the .cmp rules in
# the stylesheet. Each cell carries data-q (the quarter label) which the mobile
# media query pulls out via ::before — that's what keeps a stacked cell readable
# without duplicating the label in every row.

_TONES = ("latest", "qoq", "yoy")
_MARKS = {"latest": "📍", "qoq": "↔", "yoy": "📅"}


def _cmp_cell(inner: str | None, tone: str, qlabel: str,
              delta: str = "", why: str = "", empty: str = "No data") -> str:
    """
    One cell of a comparison row. `inner` None → muted placeholder.

    The .cmp-delta slot is ALWAYS emitted, even empty. On mobile the cell is a
    flex row (label · value · delta); if some cells had a delta and others
    didn't, the child count would differ and space-between would park the
    values at different x-positions. A constant slot keeps them aligned.
    """
    mark = _MARKS.get(tone, "")
    q = f"{mark} {qlabel}".strip()
    if inner is None:
        return (f"<div class='cmp-cell empty' data-q='{q}' data-tone='{tone}'>"
                f"<div class='cmp-val'>{empty}</div><div class='cmp-delta'></div></div>")
    return (f"<div class='cmp-cell' data-q='{q}' data-tone='{tone}'>"
            f"<div class='cmp-val'>{inner}</div><div class='cmp-delta'>{delta}</div>{why}</div>")


def _cmp_row(label: str, cells: list[str], help_text: str = "") -> str:
    """
    One comparison row. `help_text` adds an ⓘ explaining WHAT the metric is
    (a static definition). Rendered as a CSS tooltip off a data-tip attribute
    and shown on :hover or :focus, so it works on desktop and on tap.
    For WHY a given number is what it is, see `why=` on _cmp_cell.
    """
    tip = ""
    if help_text:
        # data-tip, not title: title= gets stripped by Streamlit's sanitiser.
        # tabindex makes it focusable, so a tap opens it on touch devices.
        tip = (f" <span class='info-tip' tabindex='0' "
               f"data-tip='{html.escape(help_text, quote=True)}'>ⓘ</span>")
    return f"<div class='cmp-row'><div class='cmp-label'>{label}{tip}</div>{''.join(cells)}</div>"


def _why_html(summary: str = "", drivers: list | None = None,
              citations: list | None = None, max_cites: int = 2) -> str:
    """
    The evidence behind a single number, as a <details> disclosure.

    <details> rather than a title= tooltip or st.popover: it opens on click AND
    tap so it works identically on desktop and phone, needs no JS, and — unlike
    st.popover, which is a Streamlit element — it can live inside the html grid
    right next to the number it explains. That adjacency is the whole point:
    a score with its reasoning one tap away is the difference between a signal
    and a number someone has to trust blindly.
    """
    parts = []
    if summary:
        parts.append(f"<div class='why-summary'>{html.escape(summary)}</div>")

    for d in (drivers or [])[:3]:
        text = d if isinstance(d, str) else (d.get("text") or d.get("driver") or str(d))
        parts.append(f"<div class='why-driver'>• {html.escape(str(text))}</div>")

    for c in (citations or [])[:max_cites]:
        if not isinstance(c, dict):
            continue
        quote = (c.get("quote") or "").strip()
        if not quote:
            continue
        meta = " · ".join(x for x in [
            (c.get("doc_type") or "").replace("_", " ").title(),
            c.get("quarter") or "",
            c.get("speaker") or "",
        ] if x)
        url = c.get("source_url") or ""
        link = (f" <a href='{html.escape(url, quote=True)}' target='_blank' "
                f"class='why-link'>↗</a>") if url else ""
        parts.append(
            f"<div class='why-cite'>“{html.escape(quote)}”"
            f"<div class='why-meta'>{html.escape(meta)}{link}</div></div>"
        )

    if not parts:
        return ""
    return ("<details class='cmp-why'><summary></summary>"
            f"<div class='why-body'>{''.join(parts)}</div></details>")


def _cmp_grid(rows: list[str]) -> str:
    return f"<div class='cmp'>{''.join(rows)}</div>"


def _cmp_render(rows: list[str]) -> None:
    st.markdown(_cmp_grid(rows), unsafe_allow_html=True)


def _bar_html(value: float | None) -> str | None:
    """Sub-dimension mini bar: number + proportional fill. None → placeholder."""
    if value is None:
        return None
    c = _color(value)
    return (
        f"<span style='color:{c};font-weight:700'>{value:.1f}</span>"
        f"<div class='cmp-bar-track'><div class='cmp-bar-fill' "
        f"style='background:{c};width:{value*10:.0f}%'></div></div>"
    )


def _big_text(text: str, colour: str) -> str:
    return f"<div style='font-size:1.5rem;font-weight:800;color:{colour}'>{text}</div>"


def _score_legend(max_val: float = 10) -> None:
    """Small colour-coded legend explaining what the gauge colours mean."""
    st.markdown(f"""
    <div class='legend-row'>
      <span><span class='legend-dot' style='background:#34d399'></span>Strong (≥{max_val*0.75:.0f})</span>
      <span><span class='legend-dot' style='background:#fbbf24'></span>Moderate (≥{max_val*0.55:.0f})</span>
      <span><span class='legend-dot' style='background:#fb923c'></span>Weak (≥{max_val*0.35:.0f})</span>
      <span><span class='legend-dot' style='background:#f87171'></span>Poor (&lt;{max_val*0.35:.0f})</span>
    </div>
    """, unsafe_allow_html=True)


# ── Citation renderer ─────────────────────────────────────────────────────────

_DOC_ICONS = {
    "earnings_call":          "📞",
    "press_release":          "📰",
    "annual_report":          "📋",
    "investor_presentation":  "📊",
    "news_article":           "🗞️",
    "broker_note":            "🔬",
    "management_commentary":  "💬",
}


def _render_citations(citations: list, label: str = "Evidence") -> None:
    """Render a collapsible evidence block with source links."""
    if not citations:
        return
    with st.expander(f"📎 {label} — {len(citations)} citation(s)", expanded=False):
        for c in citations[:8]:
            doc_type   = c.get("doc_type", "") if isinstance(c, dict) else ""
            quarter    = c.get("quarter", "")  if isinstance(c, dict) else ""
            speaker    = c.get("speaker", "")  if isinstance(c, dict) else ""
            url        = c.get("source_url","") if isinstance(c, dict) else ""
            quote      = c.get("quote","")      if isinstance(c, dict) else ""
            relevance  = float(c.get("relevance", 1.0) if isinstance(c, dict) else 1.0)

            icon       = _DOC_ICONS.get(doc_type, "📄")
            type_label = doc_type.replace("_", " ").title()
            speaker_html = f" &nbsp;·&nbsp; <span style='color:#a78bfa'>{speaker}</span>" if speaker else ""
            url_html = (
                f" &nbsp;·&nbsp; <a href='{url}' target='_blank' "
                f"style='color:#67e8f9;text-decoration:none'>↗ source</a>"
            ) if url else ""

            st.markdown(f"""
            <div style='background:#0a0a1a;border-radius:8px;padding:0.65rem 1rem;
                        margin:0.3rem 0;border-left:3px solid #4f46e5'>
                <div style='color:#475569;font-size:0.73rem;margin-bottom:0.3rem'>
                    {icon} {type_label} &nbsp;·&nbsp; {quarter}{speaker_html}{url_html}
                    <span style='float:right;color:#334155'>relevance {relevance:.2f}</span>
                </div>
                <div style='color:#cbd5e1;font-size:0.87rem;font-style:italic;line-height:1.5'>
                    "{quote}"
                </div>
            </div>
            """, unsafe_allow_html=True)


def _get(bundle: dict | None, *path, default=None):
    """Safe nested dict getter."""
    if bundle is None: return default
    d = bundle
    for k in path:
        if not isinstance(d, dict): return default
        d = d.get(k, default)
        if d is None: return default
    return d


def _score(bundle: dict | None, signal: str, window: str = "latest") -> float | None:
    """Extract a score from a signal bundle for a given time window."""
    w_map = {"latest": "latest", "3m": "three_month", "6m": "six_month", "12m": "twelve_month"}
    w = w_map.get(window, window)
    return _get(bundle, signal, "temporal", w, "score")


# ── Gauge ─────────────────────────────────────────────────────────────────────

def _gauge(value: float | None, max_val: float, title: str = "", height: int = 160) -> go.Figure:
    v = value or 0
    c = _color(v, max_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        title={"text": title, "font": {"size": 11, "color": "#94a3b8"}},
        number={"font": {"size": 28, "color": c}, "suffix": f"/{int(max_val)}"},
        gauge={
            "axis": {"range": [0, max_val], "tickfont": {"color": "#475569"}},
            "bar":  {"color": c if value is not None else "#1e293b", "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "steps": [
                {"range": [0, max_val*0.35], "color": "#18080a"},
                {"range": [max_val*0.35, max_val*0.55], "color": "#1a1500"},
                {"range": [max_val*0.55, max_val*0.75], "color": "#0c1a10"},
                {"range": [max_val*0.75, max_val], "color": "#091509"},
            ],
        },
    ))
    fig.update_layout(
        height=height, margin=dict(l=10,r=10,t=30,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e2e8f0"},
    )
    return fig


def _trend_chart(history: list[dict], col: str, color: str = "#7c3aed",
                 ymax: float | None = None) -> go.Figure:
    if not history: return go.Figure()
    df = pd.DataFrame(history).sort_values(["fiscal_year","quarter"])
    df["lbl"] = df["quarter"] + " " + df["fiscal_year"].astype(str)
    fig = go.Figure(go.Scatter(
        x=df["lbl"], y=df[col], mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
        fill="tozeroy", fillcolor=f"rgba(124,58,237,0.07)",
    ))
    fig.update_layout(
        height=130, margin=dict(l=5,r=5,t=8,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#475569", tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#475569",
                   range=[0, ymax] if ymax else None),
        showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📡 Signal Intelligence")
    st.caption("AI-powered equity signal engine for India-listed companies")
    st.divider()

    stored = ss.get_all_tickers()
    if stored:
        st.markdown("### Stored Tickers")
        view_ticker = st.selectbox(
            "Load", ["— choose —"] + stored,
            help="Reload a previously-run ticker from the local database without re-running the pipeline.",
        )
        st.divider()
    else:
        view_ticker = "— choose —"

    st.markdown("### Run Analysis")
    ticker_in  = st.text_input(
        "Ticker", placeholder="INFY, TCS, MSFT…", max_chars=10,
        help="The NSE trading symbol for the company (e.g. TCS, INFY, RELIANCE). "
             "Documents are fetched directly from NSE and BSE using this symbol.",
    ).upper().strip()
    company_in = st.text_input(
        "Company name", placeholder="Infosys Limited",
        help="Full legal/listed name — used to resolve the matching BSE scrip code. "
             "Closer to the official listed name = more reliable match.",
    )

    col_q, col_y = st.columns(2)
    with col_q:
        quarter_in = st.selectbox(
            "Quarter", ["Auto", "Q1","Q2","Q3","Q4"], index=0,
            help="Leave on Auto to analyze the most recently completed quarter. "
                 "Pick a specific quarter to anchor the analysis to an earlier period instead.",
        )
    with col_y:
        from datetime import datetime as _dt_year
        year_in = st.number_input(
            "Year", min_value=2019, max_value=2030, value=_dt_year.now().year, step=1,
            help="Only used when Quarter is set to a specific value (ignored on Auto).",
        )

    quarter_val = None if quarter_in == "Auto" else quarter_in
    year_val    = None if quarter_in == "Auto" else int(year_in)

    if quarter_in == "Auto":
        from agents.orchestrator import _latest_completed_quarter
        auto_q, auto_yr = _latest_completed_quarter()
        st.caption(f"Auto-detect: **{auto_q} {auto_yr}** as latest")

    model_choice = st.selectbox(
        "Model", ["gpt-4o-mini","gpt-4o"], index=0,
        help="gpt-4o-mini is faster and cheaper — good for most runs. "
             "gpt-4o gives higher-quality reasoning but costs more per run.",
    )
    run_disabled = not (ticker_in and company_in)
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                        disabled=run_disabled,
                        help="Fetches NSE/BSE documents and runs all four signal agents "
                             "for Latest, QoQ, and YoY in one click." if not run_disabled
                             else "Enter both a Ticker and Company name to enable this.")


# ════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ════════════════════════════════════════════════════════════

if run_btn:
    from agents.orchestrator import run_comparison_pipeline

    active, active_lock = _running_tickers()
    ticker_key = ticker_in.strip().upper()

    with active_lock:
        already_running = ticker_key in active
        if not already_running:
            active.add(ticker_key)

    if already_running:
        st.warning(
            f"**{ticker_key} is already being analysed** — in another tab, or by "
            "someone else on this machine. Wait for that run to finish rather than "
            "starting a second one: both would write the same signals and bill the "
            "OpenAI key twice."
        )
        st.stop()

    status_box = st.status("🔍 Fetching documents for 3 quarters…", expanded=True, state="running")
    progress   = st.progress(0, text="Starting…")

    try:
        with status_box:
            st.write("📡 Fetching Latest, QoQ, and YoY documents in parallel…")
            progress.progress(15, text="Fetching documents…")

            result = _run_async(run_comparison_pipeline(
                ticker=ticker_in, company=company_in,
                quarter=quarter_val, year=year_val,
                vs=vs, ss=ss,
                # Passed per-run. Never assign to settings.OPENAI_MODEL: it is a
                # process-wide singleton shared by every browser session, so one
                # user's choice would change the model under another user's run.
                model=model_choice,
            ))
            progress.progress(85, text="Running signal agents…")
            st.write(f"✅ {result['docs_ingested']} docs ingested across 3 quarters")
            st.write(f"🧠 Signals: **{result['latest_label']}** (Latest) · "
                     f"**{result['qoq_label']}** (QoQ) · **{result['yoy_label']}** (YoY)")
            progress.progress(100, text="Done!")

        status_box.update(
            label=f"✅ Complete — {result['latest_label']} vs {result['qoq_label']} vs {result['yoy_label']}",
            state="complete", expanded=False,
        )
        for e in result.get("errors",[]):
            if e: st.warning(e)

        st.session_state["active_ticker"]  = ticker_in
        st.session_state["pipeline_result"] = result

    except Exception as e:
        progress.empty()
        status_box.update(label="❌ Pipeline failed", state="error", expanded=True)
        st.error(f"**Error:** {e}")
        st.info(
            "Check your `.env` has a valid `OPENAI_API_KEY`, that the ticker is a valid "
            "NSE symbol (e.g. TCS, INFY, RELIANCE), and that this machine can reach "
            "nseindia.com / bseindia.com."
        )
        st.stop()

    finally:
        # Must release even on failure/rerun, or the ticker stays locked out
        # until the server restarts.
        with active_lock:
            active.discard(ticker_key)


# ════════════════════════════════════════════════════════════
# RESOLVE ACTIVE TICKER + LOAD DATA
# ════════════════════════════════════════════════════════════

if run_btn and ticker_in:
    active_ticker = ticker_in
elif view_ticker != "— choose —":
    active_ticker = view_ticker
    if st.session_state.get("active_ticker") != view_ticker:
        st.session_state.pop("pipeline_result", None)
    st.session_state["active_ticker"] = view_ticker
else:
    active_ticker = st.session_state.get("active_ticker")

if not active_ticker:
    st.markdown("""
    ## Welcome to Signal Intelligence

    **An AI-powered signal intelligence platform for India-listed equities.**
    Signal Intelligence turns raw corporate disclosures — quarterly results, investor
    presentations, and concall transcripts, pulled directly from NSE and BSE — into
    structured, evidence-backed investment signals. A multi-agent RAG pipeline reads
    the filings so you don't have to: it scores management's tone and confidence,
    tracks how the narrative around key sector themes is shifting quarter over quarter,
    audits whether guidance is actually being delivered on, and surfaces emerging risks
    before they show up in the price. Every signal is cited back to the source document —
    this is decision-grade evidence synthesis, not another LLM summary of a transcript.

    Enter a ticker in the sidebar and click **Run Analysis**.

    The pipeline automatically fetches and scores three time periods in one run:

    | Period | What it is |
    |--------|-----------|
    | **Latest** | Most recently completed quarter (auto-detected) |
    | **QoQ** | Prior quarter — shows momentum |
    | **YoY** | Same quarter last year — shows structural change |

    Set a specific quarter to anchor the analysis to a different point in time.
    """)
    st.stop()

# Load stored history for trend charts
conf_history = ss.get_confidence_history(active_ticker, limit=20)
narr_history = ss.get_narrative_history(active_ticker, limit=20)
guid_history = ss.get_guidance_history(active_ticker, limit=20)

if not conf_history:
    st.warning(f"No signals stored for **{active_ticker}**. Run the pipeline from the sidebar.")
    st.stop()

# Use pipeline_result if from current session, otherwise reconstruct from DB
result = st.session_state.get("pipeline_result")
if result and result.get("latest"):
    latest_b = result["latest"]
    qoq_b    = result["qoq"]
    yoy_b    = result["yoy"]
    label_l  = result["latest_label"]
    label_q  = result["qoq_label"]
    label_y  = result["yoy_label"]
else:
    # Reconstruct from DB: latest = first row, qoq = second, yoy = matching quarter -1yr
    def _find_row(history, quarter_label):
        for r in history:
            if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == quarter_label:
                return r
        return history[0] if history else {}

    label_l = f"{conf_history[0].get('quarter','')} {conf_history[0].get('fiscal_year','')}" if conf_history else "—"
    l_q, l_yr = label_l.split(" ") if " " in label_l else ("Q1","2024")
    from agents.orchestrator import resolve_quarters
    (lq,ly),(qq,qy),(yq,yy) = resolve_quarters(l_q, int(l_yr))
    label_q = f"{qq} {qy}"
    label_y = f"{yq} {yy}"

    # Build minimal bundle dicts from DB rows
    def _row_to_bundle(conf_row, narr_row, guid_row, risk_row):
        if not conf_row: return None
        return {
            "confidence": {
                "score":             conf_row.get("score"),
                "change":            conf_row.get("change"),
                "previous_score":    conf_row.get("previous_score"),
                "tone":              conf_row.get("tone"),
                "summary":           conf_row.get("summary",""),
                "drivers":           conf_row.get("drivers",[]),
                "citations":         conf_row.get("citations",[]),
                "confidence_level":  conf_row.get("confidence_level"),
                "uncertainty_level": conf_row.get("uncertainty_level"),
                "defensiveness":     conf_row.get("defensiveness"),
                "specificity":       conf_row.get("specificity"),
                "consistency":       conf_row.get("consistency"),
                "forward_strength":  conf_row.get("forward_strength"),
            },
            "narrative": {
                "overall_shift":  (narr_row or {}).get("overall_shift"),
                "shift_summary":  (narr_row or {}).get("shift_summary",""),
                "themes":         (narr_row or {}).get("themes",[]),
                "accelerating":   (narr_row or {}).get("accelerating",[]),
                "emerging":       (narr_row or {}).get("emerging",[]),
                "fading":         (narr_row or {}).get("fading",[]),
                "newly_risky":    (narr_row or {}).get("newly_risky",[]),
                "citations":      (narr_row or {}).get("citations",[]),
            },
            "guidance": {
                "score":            (guid_row or {}).get("score"),
                "beats":            (guid_row or {}).get("beats", 0),
                "misses":           (guid_row or {}).get("misses", 0),
                "in_line":          (guid_row or {}).get("in_line", 0),
                "beat_rate":        (guid_row or {}).get("beat_rate", 0),
                "serial_miss_risk": (guid_row or {}).get("serial_miss_risk", False),
                "recent_pattern":   (guid_row or {}).get("recent_pattern", []),
                "summary":          (guid_row or {}).get("summary", ""),
                "citations":        (guid_row or {}).get("citations", []),
            },
            "risk": {
                "overall_risk_direction": (risk_row or {}).get("overall_risk_direction"),
                "risks":       (risk_row or {}).get("risks", []),
                "new_risks":   (risk_row or {}).get("new_risks", []),
                "escalating":  (risk_row or {}).get("escalating", []),
                "diminishing": (risk_row or {}).get("diminishing", []),
                "summary":     (risk_row or {}).get("summary", ""),
                "citations":   (risk_row or {}).get("citations", []),
            },
        }

    def _get_row(history, label):
        for r in history:
            if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == label:
                return r
        return {}

    latest_b = _row_to_bundle(
        _get_row(conf_history, label_l),
        _get_row(narr_history, label_l),
        _get_row(guid_history, label_l),
        _get_row(ss.get_risk_history(active_ticker, 20), label_l),
    )
    qoq_b = _row_to_bundle(
        _get_row(conf_history, label_q),
        _get_row(narr_history, label_q),
        _get_row(guid_history, label_q),
        _get_row(ss.get_risk_history(active_ticker, 20), label_q),
    )
    yoy_b = _row_to_bundle(
        _get_row(conf_history, label_y),
        _get_row(narr_history, label_y),
        _get_row(guid_history, label_y),
        _get_row(ss.get_risk_history(active_ticker, 20), label_y),
    )


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════

st.markdown(f"# {active_ticker} — Signal Intelligence")
st.divider()

# Column headers
# Column headers. Hidden on mobile — each cell labels its own quarter there.
st.markdown(
    "<div class='cmp'><div class='cmp-row'><div class='cmp-label'></div>"
    f"<div class='cmp-qhead q-latest hide-mobile'>📍 Latest<br>{label_l}</div>"
    f"<div class='cmp-qhead q-qoq hide-mobile'>↔ QoQ vs<br>{label_q}</div>"
    f"<div class='cmp-qhead q-yoy hide-mobile'>📅 YoY vs<br>{label_y}</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-help'>"
    "<b>Latest</b> = most recently completed quarter &nbsp;·&nbsp; "
    "<b>QoQ</b> = prior quarter, shows short-term momentum &nbsp;·&nbsp; "
    "<b>YoY</b> = same quarter last year, shows structural change beneath the seasonal noise."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Data coverage banner ──────────────────────────────────────────────────────

def _has_signal(b, key):
    if not b or not isinstance(b, dict): return False
    sig = b.get(key)
    if not sig or not isinstance(sig, dict): return False
    return sig.get("score") is not None or sig.get("overall_shift") is not None or sig.get("overall_risk_direction") is not None

_coverage = {
    label_l: {"Confidence": _has_signal(latest_b,"confidence"), "Narrative": _has_signal(latest_b,"narrative"), "Guidance": _has_signal(latest_b,"guidance"), "Risk": _has_signal(latest_b,"risk")},
    label_q: {"Confidence": _has_signal(qoq_b,"confidence"),    "Narrative": _has_signal(qoq_b,"narrative"),    "Guidance": _has_signal(qoq_b,"guidance"),    "Risk": _has_signal(qoq_b,"risk")},
    label_y: {"Confidence": _has_signal(yoy_b,"confidence"),    "Narrative": _has_signal(yoy_b,"narrative"),    "Guidance": _has_signal(yoy_b,"guidance"),    "Risk": _has_signal(yoy_b,"risk")},
}
_missing = [
    f"**{period}**: {', '.join(s for s, ok in sigs.items() if not ok)}"
    for period, sigs in _coverage.items()
    if not all(sigs.values())
]
if _missing:
    with st.expander("⚠️ Some periods have incomplete data — click for details", expanded=False):
        st.markdown(
            "These periods are missing one or more signal types. "
            "Re-run the pipeline to fill gaps.\n\n" +
            "\n\n".join(f"- {m}" for m in _missing)
        )
        st.caption("This usually means the ticker was last run with an older version of the code, or the pipeline hit a rate limit mid-run.")

# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Confidence", "📈 Narrative", "✅ Guidance", "⚠️ Risks", "🕒 Trend", "🔗 Sources"
])


# ── Tab 1: Management Confidence ──────────────────────────────────────────────

with tab1:

    st.markdown(
        "<div class='section-help'>"
        "Scores management's tone in prepared remarks and Q&A on a 0–10 scale — "
        "higher means more confident, specific, and consistent language, and less hedging or defensiveness. "
        "Compare the three columns to see whether tone is improving or deteriorating."
        "</div>",
        unsafe_allow_html=True,
    )
    _score_legend(10)

    # Helper to extract confidence data from either bundle dict or DB row
    def _conf(b):
        if b is None: return {}
        if "confidence" in b and b["confidence"]: return b["confidence"]
        return b  # DB row

    lc = _conf(latest_b)
    qc = _conf(qoq_b)
    yc = _conf(yoy_b)

    l_score = float(lc.get("score") or 0)
    q_score = float(qc.get("score") or 0)
    y_score = float(yc.get("score") or 0)

    # Score row
    _cmp_render([
        _cmp_row("Score", [
            _cmp_cell(_score_cell(l_score), "latest", label_l,
                      why=_why_html(lc.get("summary",""), lc.get("drivers"), lc.get("citations"))),
            _cmp_cell(_score_cell(q_score) if q_score else None, "qoq", label_q,
                      delta=_delta_html(l_score, q_score) if q_score else "",
                      why=_why_html(qc.get("summary",""), qc.get("drivers"), qc.get("citations"))),
            _cmp_cell(_score_cell(y_score) if y_score else None, "yoy", label_y,
                      delta=_delta_html(l_score, y_score) if y_score else "",
                      why=_why_html(yc.get("summary",""), yc.get("drivers"), yc.get("citations"))),
        ], help_text=(
            "0–10 score for how confident management sounds in prepared remarks and Q&A. "
            "Built from six sub-dimensions below — higher means more certainty, more concrete "
            "numbers, less hedging and defensiveness. Tap 'why' under a score for the model's "
            "reasoning and the quotes it drew on."
        )),
    ])

    # Gauges stay native Plotly (can't live inside the html grid) and are
    # hidden on phones by the stylesheet — the scores above already say it.
    with st.container(key="gauge-row-confidence"):
        # Leading spacer matches the grid's 1.4fr label column so each gauge
        # lands under its own score.
        _, g1, g2, g3 = st.columns([1.4, 1, 1, 1])
        with g1: st.plotly_chart(_gauge(l_score, 10), use_container_width=True, key="chart_1")
        with g2:
            if q_score: st.plotly_chart(_gauge(q_score, 10), use_container_width=True, key="chart_2")
        with g3:
            if y_score: st.plotly_chart(_gauge(y_score, 10), use_container_width=True, key="chart_3")

    st.divider()

    # Sub-dimensions
    st.markdown("#### Sub-dimension Breakdown")
    dims = [
        ("Confidence Level",  "confidence_level",
         "Certainty of language vs. hedging. High = definite claims; low = 'we hope', 'should', 'aim to'."),
        ("Low Uncertainty",   "uncertainty_level",
         "Inverted: how FEW explicit uncertainty signals appear. High = little stated uncertainty."),
        ("Not Defensive",     "defensiveness",
         "Inverted: how little reactive or justifying tone there is, especially under Q&A pressure. High = not defensive."),
        ("Specificity",       "specificity",
         "Concrete numbers, dates and named drivers vs. vague language. High = management commits to specifics."),
        ("Consistency",       "consistency",
         "Alignment with what management said in prior quarters. Low = the story has changed."),
        ("Forward Strength",  "forward_strength",
         "Strength of positive forward-looking statements. High = confident, specific guidance about what's ahead."),
    ]
    # Only render sub-dimension bars if at least one has a real value
    # Show the sub-dimension section whenever there is an overall confidence score.
    # Individual dimensions show '—' when their value is None or 0 (not stored).
    _has_subdim_data = bool(lc.get('score'))

    if not _has_subdim_data:
        st.info(
            "Sub-dimension scores are not yet available for this period. "
            "Re-run the pipeline to populate them.",
            icon="ℹ️",
        )
    else:
        _rows = []
        for dim_name, dim_key, dim_help in dims:
            _lv_raw = lc.get(dim_key)
            _qv_raw = qc.get(dim_key)
            _yv_raw = yc.get(dim_key)
            # Treat None AND 0.0 as 'not scored' — show — instead of a misleading zero bar
            lv = float(_lv_raw) if (_lv_raw is not None and float(_lv_raw) > 0) else None
            qv = float(_qv_raw) if (_qv_raw is not None and float(_qv_raw) > 0) else None
            yv = float(_yv_raw) if (_yv_raw is not None and float(_yv_raw) > 0) else None

            _rows.append(_cmp_row(dim_name, [
                _cmp_cell(_bar_html(lv), "latest", label_l, empty="—"),
                _cmp_cell(_bar_html(qv), "qoq",    label_q, empty="—"),
                _cmp_cell(_bar_html(yv), "yoy",    label_y, empty="—"),
            ], help_text=dim_help))
        _cmp_render(_rows)

    st.divider()

    # Citations
    _render_citations(lc.get("citations",[]) or [], "Confidence Evidence")

    st.divider()

    # Summaries + drivers
    st.markdown("#### Signal Narratives")
    r1, r2, r3 = st.columns(3)
    for col, bundle_data, lbl in [(r1, lc, label_l), (r2, qc, label_q), (r3, yc, label_y)]:
        with col:
            st.markdown(f"**{lbl}**")
            summary = bundle_data.get("summary","")
            if summary:
                st.markdown(f"<div class='signal-card'><p style='color:#e2e8f0;font-size:0.88rem;margin:0;line-height:1.5'>{summary}</p></div>", unsafe_allow_html=True)
                drivers = bundle_data.get("drivers",[])
                for d in drivers[:3]:
                    st.markdown(f"<div class='driver-item'>• {d}</div>", unsafe_allow_html=True)
            else:
                st.caption("No data for this period.")


# ── Tab 2: Narrative Shift ────────────────────────────────────────────────────

with tab2:

    st.markdown(
        "<div class='section-help'>"
        "Tracks how much management emphasizes each theme (AI, margins, China, competition, etc.) "
        "and whether that emphasis is growing, fading, or newly risky, compared to the prior quarter. "
        "<b>Overall Shift</b> summarizes the net direction: positive, negative, mixed, or neutral."
        "</div>",
        unsafe_allow_html=True,
    )

    def _narr(b):
        if b is None: return {}
        n = b.get("narrative") if isinstance(b, dict) and "narrative" in b else b
        return n or {}

    ln = _narr(latest_b)
    qn = _narr(qoq_b)
    yn = _narr(yoy_b)

    shift_map  = {"positive":"#22c55e","negative":"#ef4444","mixed":"#f97316","neutral":"#94a3b8"}

    # Shift row
    def _shift_cell(nd):
        shift = nd.get("overall_shift","—") or "—"
        return _big_text(shift.upper(), shift_map.get(shift, "#94a3b8"))

    _cmp_render([
        _cmp_row("Narrative Shift", [
            _cmp_cell(_shift_cell(ln), "latest", label_l,
                      why=_why_html(ln.get("summary",""), None, ln.get("citations"))),
            _cmp_cell(_shift_cell(qn), "qoq",    label_q,
                      why=_why_html(qn.get("summary",""), None, qn.get("citations"))),
            _cmp_cell(_shift_cell(yn), "yoy",    label_y,
                      why=_why_html(yn.get("summary",""), None, yn.get("citations"))),
        ], help_text=(
            "Direction the story is moving in, across the sector and macro themes management "
            "discusses. Positive/negative/mixed/neutral is the net of themes accelerating, "
            "emerging, fading, or turning risky versus the comparison quarter."
        )),
    ])

    st.divider()

    # Theme movement columns
    st.markdown("#### Theme Movement")
    r1, r2, r3 = st.columns(3)
    sections = [
        ("🚀 Accelerating", "accelerating", "pill-green"),
        ("✦ Emerging",      "emerging",     "pill-purple"),
        ("📉 Fading",       "fading",       "pill-red"),
        ("⚠️ Newly Risky",  "newly_risky",  "pill-orange"),
    ]
    for col, nd, period_label in [(r1,ln,label_l),(r2,qn,label_q),(r3,yn,label_y)]:
        with col:
            st.markdown(f"**{period_label}**")
            if not nd:
                st.caption("No data")
                continue
            for label, key, pill_cls in sections:
                items = nd.get(key,[]) or []
                if items:
                    st.markdown(f"**{label}**")
                    st.markdown(" ".join(_pill(t, pill_cls) for t in items), unsafe_allow_html=True)
                    st.markdown("")

    st.divider()

    # Theme sentiment table — latest only with QoQ/YoY deltas
    st.markdown("#### Theme Sentiment — Latest vs Prior Periods")
    latest_themes = ln.get("themes",[]) or []
    if latest_themes:
        def _theme_sentiment(nd, theme_name):
            for t in (nd.get("themes",[]) or []):
                if t.get("theme","") == theme_name:
                    return float(t.get("sentiment_current",0))
            return None

        rows = []
        for t in latest_themes:
            name    = t.get("theme","")
            s_now   = float(t.get("sentiment_current",0))
            s_qoq   = _theme_sentiment(qn, name)
            s_yoy   = _theme_sentiment(yn, name)
            d_qoq   = f"{(s_now - s_qoq):+.2f}" if s_qoq is not None else "—"
            d_yoy   = f"{(s_now - s_yoy):+.2f}" if s_yoy is not None else "—"
            rows.append({
                "Theme":            name,
                "Status":           t.get("status",""),
                f"Sentiment ({label_l})": f"{s_now:+.2f}",
                f"vs {label_q}":    d_qoq,
                f"vs {label_y}":    d_yoy,
                "Evidence (now)":   t.get("evidence_count_current",0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No theme data available.")

    _render_citations(ln.get("citations",[]) or [], "Narrative Evidence")


# ── Tab 3: Guidance Credibility ───────────────────────────────────────────────

with tab3:

    st.markdown(
        "<div class='section-help'>"
        "Scores how reliably management delivers on its own guidance, on a 0–100 scale. "
        "Compares guidance given in past quarters against the actual results reported later, "
        "and flags a company as <b>serial-miss risk</b> if it has repeatedly missed its own targets."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── YTD Guidance callout ──────────────────────────────────────────────────
    from datetime import datetime as _dt
    _ytd_yr   = int(label_l.split()[-1]) if label_l != "—" else _dt.utcnow().year
    _ytd_guid = ss.get_ytd_guidance(active_ticker, _ytd_yr)

    if _ytd_guid:
        _tracked  = _ytd_guid["total_tracked"]
        _beats    = _ytd_guid["total_beats"]
        _misses   = _ytd_guid["total_misses"]
        _in_line  = _ytd_guid["total_in_line"]
        _rate     = _ytd_guid["ytd_beat_rate"]
        _qtrs     = ", ".join(_ytd_guid["quarters_covered"])
        _serial   = _ytd_guid["serial_misses"]
        # _rate is None when no guidance items were tracked at all. Showing a
        # red 0% there says "missed everything" when it means "measured nothing".
        if _rate is None:
            _rate_color, _rate_text = "#64748b", "—"
        else:
            _rate_color = "#22c55e" if _rate >= 0.7 else "#eab308" if _rate >= 0.5 else "#ef4444"
            _rate_text  = f"{_rate:.0%}"

        _serial_html = (
            "<div style='color:#ef4444; font-size:0.82rem'>⚠️ Serial misses: "
            + ", ".join(_serial) + "</div>"
        ) if _serial else ""
        _ytd_guid_html = "".join([
            "<div style='background:#12121f;border-radius:10px;padding:1rem 1.4rem;",
            "border:1px solid #2e1065;margin-bottom:1rem;",
            "display:flex;align-items:center;gap:2rem;flex-wrap:wrap'>",
            "<div>",
            "<div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;",
            f"letter-spacing:0.08em;margin-bottom:0.2rem'>📅 YTD Guidance — {_ytd_yr} ({_qtrs})</div>",
            f"<div style='font-size:1.6rem;font-weight:800;color:{_rate_color}'>{_rate_text}</div>",
            "</div>",
            "<div style='color:#94a3b8;font-size:0.88rem;line-height:1.8'>",
            f"<span style='color:#22c55e;font-weight:700'>✓ {_beats} beat</span> &nbsp;·&nbsp; ",
            f"<span style='color:#ef4444;font-weight:700'>✗ {_misses} miss</span> &nbsp;·&nbsp; ",
            f"<span style='color:#eab308;font-weight:700'>~ {_in_line} in-line</span> &nbsp;·&nbsp; ",
            f"{_tracked} items tracked",
            "</div>",
            _serial_html,
            "</div>",
        ])
        st.markdown(_ytd_guid_html, unsafe_allow_html=True)

    st.divider()
    # ─────────────────────────────────────────────────────────────────────────

    def _guid(b):
        if b is None: return {}
        g = b.get("guidance") if isinstance(b, dict) and "guidance" in b else b
        return g or {}

    lg = _guid(latest_b)
    qg = _guid(qoq_b)
    yg = _guid(yoy_b)

    l_gs = float(lg.get("score") or 0)
    q_gs = float(qg.get("score") or 0)
    y_gs = float(yg.get("score") or 0)

    _cmp_render([
        _cmp_row("Guidance Score", [
            _cmp_cell(_score_cell(l_gs, 100), "latest", label_l,
                      why=_why_html(lg.get("summary",""), None, lg.get("citations"))),
            _cmp_cell(_score_cell(q_gs, 100) if q_gs else None, "qoq", label_q,
                      delta=_delta_html(l_gs, q_gs) if q_gs else "",
                      why=_why_html(qg.get("summary",""), None, qg.get("citations"))),
            _cmp_cell(_score_cell(y_gs, 100) if y_gs else None, "yoy", label_y,
                      delta=_delta_html(l_gs, y_gs) if y_gs else "",
                      why=_why_html(yg.get("summary",""), None, yg.get("citations"))),
        ], help_text=(
            "0–100 score for how reliably management delivers on its OWN guidance. Compares "
            "guidance given in past quarters against results actually reported later. A low "
            "score with repeated misses is flagged as serial-miss risk."
        )),
    ])

    with st.container(key="gauge-row-guidance"):
        _, g4, g5, g6 = st.columns([1.4, 1, 1, 1])
        with g4: st.plotly_chart(_gauge(l_gs, 100), use_container_width=True, key="chart_4")
        with g5:
            if q_gs: st.plotly_chart(_gauge(q_gs, 100), use_container_width=True, key="chart_5")
        with g6:
            if y_gs: st.plotly_chart(_gauge(y_gs, 100), use_container_width=True, key="chart_6")

    st.divider()

    m1,m2,m3 = st.columns(3)
    for col, gd, lbl in [(m1,lg,label_l),(m2,qg,label_q),(m3,yg,label_y)]:
        with col:
            st.markdown(f"**{lbl}**")
            br = float(gd.get("beat_rate") or 0)
            beats  = gd.get("beats",  0) or 0
            misses = gd.get("misses", 0) or 0
            serial = gd.get("serial_miss_risk", False)
            pattern= gd.get("recent_pattern", []) or []
            st.metric("Beat Rate", f"{br:.0%}")
            st.metric("Beats",  beats)
            st.metric("Misses", misses)
            if serial: st.error("⚠️ Serial miss risk")
            if pattern:
                cells = [
                    _pill("✓","pill-green") if p=="beat"
                    else _pill("✗","pill-red") if p=="miss"
                    else _pill("~","pill-yellow")
                    for p in pattern[-6:]
                ]
                st.markdown("&nbsp;→&nbsp;".join(cells), unsafe_allow_html=True)
            summary = gd.get("summary","")
            if summary:
                st.markdown(f"<div class='signal-card'><p style='color:#e2e8f0;font-size:0.85rem;margin:0;line-height:1.55'>{summary}</p></div>", unsafe_allow_html=True)

    _render_citations(lg.get("citations",[]) or [], "Guidance Evidence")


# ── Tab 4: Risk Emergence ─────────────────────────────────────────────────────

with tab4:

    st.markdown(
        "<div class='section-help'>"
        "Detects risks that became newly material or escalated between quarters, by comparing how often "
        "and how seriously they're discussed. Severity runs low → medium → high → critical; "
        "status shows whether a risk is new, escalating, stable, diminishing, or resolved."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── YTD Risk callout ──────────────────────────────────────────────────────
    from datetime import datetime as _dt2
    _ytd_yr2  = int(label_l.split()[-1]) if label_l != "—" else _dt2.utcnow().year
    _ytd_risk = ss.get_ytd_risks(active_ticker, _ytd_yr2)

    if _ytd_risk:
        _total_new    = _ytd_risk["total_new"]
        _total_active = _ytd_risk["total_active"]
        _esc          = _ytd_risk["escalating_ytd"]
        _dim          = _ytd_risk["diminishing_ytd"]
        _qtrs2        = ", ".join(_ytd_risk["quarters_covered"])
        _sev_map      = _ytd_risk["severity_map"]
        _active_risks = _ytd_risk["new_risks_active"]
        _sev_colors   = {"critical":"#dc2626","high":"#ef4444","medium":"#f97316","low":"#eab308"}

        top_html = ""
        for rname in _active_risks[:5]:
            sev = _sev_map.get(rname, "medium")
            sc  = _sev_colors.get(sev, "#94a3b8")
            top_html += f"<span style='color:{sc}; margin-right:0.6rem'>● {rname}</span>"

        _top_html_div = (
            "<div style='font-size:0.82rem; margin-top:0.3rem'>" + top_html + "</div>"
        ) if top_html else ""
        _ytd_risk_html = "".join([
            "<div style='background:#12121f;border-radius:10px;padding:1rem 1.4rem;",
            "border:1px solid #450a0a;margin-bottom:1rem'>",
            "<div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;",
            f"letter-spacing:0.08em;margin-bottom:0.6rem'>📅 YTD Risk Emergence — {_ytd_yr2} ({_qtrs2})</div>",
            "<div style='display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:0.6rem'>",
            f"<div><span style='color:#ef4444;font-size:1.6rem;font-weight:800'>{_total_new}</span>",
            "<span style='color:#94a3b8;font-size:0.82rem;margin-left:0.3rem'>new risks</span></div>",
            f"<div><span style='color:#fb923c;font-size:1.6rem;font-weight:800'>{len(_esc)}</span>",
            "<span style='color:#94a3b8;font-size:0.82rem;margin-left:0.3rem'>escalated</span></div>",
            f"<div><span style='color:#22c55e;font-size:1.6rem;font-weight:800'>{len(_dim)}</span>",
            "<span style='color:#94a3b8;font-size:0.82rem;margin-left:0.3rem'>diminished</span></div>",
            f"<div><span style='color:#f87171;font-size:1.6rem;font-weight:800'>{_total_active}</span>",
            "<span style='color:#94a3b8;font-size:0.82rem;margin-left:0.3rem'>still active</span></div>",
            "</div>",
            _top_html_div,
            "</div>",
        ])
        st.markdown(_ytd_risk_html, unsafe_allow_html=True)

    st.divider()
    # ─────────────────────────────────────────────────────────────────────────

    def _risk(b):
        if b is None: return {}
        r = b.get("risk") if isinstance(b, dict) and "risk" in b else b
        return r or {}

    lr = _risk(latest_b)
    qr = _risk(qoq_b)
    yr_ = _risk(yoy_b)

    risk_dir_map = {"increasing":"#ef4444","stable":"#eab308","decreasing":"#22c55e"}

    def _risk_cell(rd):
        rdir = rd.get("overall_risk_direction","—") or "—"
        new_n = len(rd.get("new_risks",[]) or [])
        esc_n = len(rd.get("escalating",[]) or [])
        return (
            _big_text(rdir.upper(), risk_dir_map.get(rdir,"#94a3b8"))
            + f"<div style='color:#94a3b8;font-size:0.8rem'>🆕{new_n} new · 📈{esc_n} esc</div>"
        )

    _cmp_render([
        _cmp_row("Risk Direction", [
            _cmp_cell(_risk_cell(lr),  "latest", label_l,
                      why=_why_html(lr.get("summary",""), None, lr.get("citations"))),
            _cmp_cell(_risk_cell(qr),  "qoq",    label_q,
                      why=_why_html(qr.get("summary",""), None, qr.get("citations"))),
            _cmp_cell(_risk_cell(yr_), "yoy",    label_y,
                      why=_why_html(yr_.get("summary",""), None, yr_.get("citations"))),
        ], help_text=(
            "Whether the risk profile is getting worse, holding, or improving versus the "
            "comparison quarter — based on risks newly raised, escalating in emphasis, or "
            "fading from the disclosure."
        )),
    ])

    st.divider()

    # Risk items side by side
    st.markdown("#### Risk Items")
    r1, r2, r3 = st.columns(3)
    sev_colors = {"critical":"#dc2626","high":"#ef4444","medium":"#f97316","low":"#eab308"}
    for col, rd, lbl in [(r1,lr,label_l),(r2,qr,label_q),(r3,yr_,label_y)]:
        with col:
            st.markdown(f"**{lbl}**")
            risks = rd.get("risks",[]) or []
            if not risks:
                st.caption("No data")
            else:
                for r in sorted(risks, key=lambda x: {"critical":0,"high":1,"medium":2,"low":3}.get(x.get("severity","low"),3))[:5]:
                    sev = r.get("severity","medium")
                    sc = sev_colors.get(sev,"#94a3b8")
                    status = r.get("status","stable")
                    icons = {"newly_material":"🆕","escalating":"📈","stable":"→","diminishing":"📉","resolved":"✅"}
                    st.markdown(f"""
                    <div class='risk-card' style='border-left-color:{sc}'>
                      <div style='color:{sc};font-weight:700;font-size:0.82rem'>
                        {icons.get(status,"")} {r.get("risk","")} [{sev}]
                      </div>
                      <div style='color:#94a3b8;font-size:0.78rem;margin-top:0.2rem'>
                        {r.get("mention_count_current",0)} mentions (vs {r.get("mention_count_previous",0)})
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Summary
            summary = rd.get("summary","")
            if summary:
                st.markdown(f"<div class='signal-card' style='border-left-color:#ef4444;margin-top:0.6rem'><p style='color:#e2e8f0;font-size:0.85rem;margin:0;line-height:1.55'>{summary}</p></div>", unsafe_allow_html=True)


    _render_citations(lr.get("citations",[]) or [], "Risk Evidence")


# ── Tab 5: Trend History ──────────────────────────────────────────────────────

with tab5:
    if len(conf_history) < 2:
        st.info("Run the pipeline for additional quarters to see trend data.")
    else:
        st.markdown("### Signal Trends — Full History")
        st.markdown(
            "<div class='section-help'>"
            "Every stored quarter for this ticker, plotted in order — useful for spotting a multi-quarter "
            "trend that a single QoQ/YoY snapshot could miss."
            "</div>",
            unsafe_allow_html=True,
        )

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Management Confidence Score**")
            st.plotly_chart(_trend_chart(conf_history,"score","#7c3aed",10), use_container_width=True, key="chart_7")
        with t2:
            st.markdown("**Guidance Credibility Score**")
            st.plotly_chart(_trend_chart(guid_history,"score","#22c55e",100), use_container_width=True, key="chart_8")

        st.divider()
        st.markdown("### All Periods Summary")
        all_rows = []
        risk_all = ss.get_risk_history(active_ticker, 20)
        def _get_risk_row(lbl):
            for r in risk_all:
                if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == lbl: return r
            return {}
        # conf_history is ordered newest-first; the delta for a row is its score
        # minus the NEXT row's (the chronologically previous quarter). Computed
        # here from real stored scores rather than read from the 'change' column,
        # which the LLM used to invent — it produced deltas that contradicted the
        # ones shown in the comparison view above.
        for _i, cr in enumerate(conf_history):
            _prev = conf_history[_i + 1] if _i + 1 < len(conf_history) else None
            _cur_s  = cr.get("score")
            _prev_s = _prev.get("score") if _prev else None
            if _cur_s is None or _prev_s is None:
                _dconf = "—"          # no prior period stored → no delta to show
            else:
                _dconf = f"{float(_cur_s) - float(_prev_s):+.1f}"
            lbl = f"{cr.get('quarter','')} {cr.get('fiscal_year','')}"
            gr  = next((r for r in guid_history if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == lbl), {})
            nr  = next((r for r in narr_history if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == lbl), {})
            rr  = _get_risk_row(lbl)
            current_lbl = lbl == label_l
            all_rows.append({
                "Quarter":    ("📍 " if current_lbl else "") + lbl,
                "Confidence": f"{float(cr.get('score',0)):.1f}/10",
                "Δ Conf":     _dconf,
                "Tone":       (cr.get("tone","") or "").title(),
                "Guidance":   f"{float(gr.get('score',0)):.0f}/100" if gr else "—",
                "Narrative":  (nr.get("overall_shift","") or "").title() if nr else "—",
                "Risk":       (rr.get("overall_risk_direction","") or "").title() if rr else "—",
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)


# ── Tab 6: Sources ────────────────────────────────────────────────────────────

with tab6:
    st.markdown("### Source Documents")
    st.caption(
        "All documents ingested for this ticker. Each row is a source that provided "
        "evidence for the signal agents. Click ↗ to open the original document."
    )

    source_docs = ss.get_source_documents(active_ticker)

    if not source_docs:
        st.info("No source documents found. Run the pipeline to ingest documents.")
    else:
        # Summary counts by doc type
        from collections import Counter
        type_counts = Counter(d.get("doc_type","unknown") for d in source_docs)
        type_cols = st.columns(min(len(type_counts), 5))
        for i, (dtype, count) in enumerate(sorted(type_counts.items())):
            icon = _DOC_ICONS.get(dtype, "📄")
            with type_cols[i % len(type_cols)]:
                st.metric(f"{icon} {dtype.replace('_',' ').title()}", count)

        st.divider()

        # Filter by quarter
        all_quarters_in_docs = sorted(
            set(f"{d.get('quarter','')} {d.get('fiscal_year','')}" for d in source_docs),
            reverse=True,
        )
        selected_source_q = st.selectbox(
            "Filter by quarter", ["All quarters"] + all_quarters_in_docs,
            key="sources_quarter_filter",
        )

        # Display documents
        filtered = source_docs if selected_source_q == "All quarters" else [
            d for d in source_docs
            if f"{d.get('quarter','')} {d.get('fiscal_year','')}" == selected_source_q
        ]

        for doc in filtered:
            dtype     = doc.get("doc_type", "")
            quarter   = f"{doc.get('quarter','')} {doc.get('fiscal_year','')}"
            title     = doc.get("title","") or "Untitled"
            url       = doc.get("source_url","")
            chunks    = doc.get("chunk_count", 0)
            icon      = _DOC_ICONS.get(dtype, "📄")
            type_label = dtype.replace("_"," ").title()

            url_html = (
                f"<a href='{url}' target='_blank' "
                f"style='color:#67e8f9;font-size:0.8rem;text-decoration:none'>↗ open source</a>"
            ) if url else "<span style='color:#334155;font-size:0.8rem'>no URL</span>"

            st.markdown(f"""
            <div style='background:#1e1e2e;border-radius:8px;padding:0.8rem 1.1rem;
                        margin:0.35rem 0;border-left:3px solid #4f46e5;
                        display:flex;align-items:center;justify-content:space-between;
                        flex-wrap:wrap;gap:0.5rem'>
                <div>
                    <span style='color:#a78bfa;font-size:0.75rem;font-weight:700;
                                 text-transform:uppercase;letter-spacing:0.06em'>
                        {icon} {type_label} &nbsp;·&nbsp; {quarter}
                    </span>
                    <div style='color:#e2e8f0;font-size:0.9rem;margin-top:0.15rem'>
                        {title[:100]}{"…" if len(title) > 100 else ""}
                    </div>
                </div>
                <div style='text-align:right;min-width:120px'>
                    <div style='color:#475569;font-size:0.75rem'>{chunks} chunks extracted</div>
                    <div style='margin-top:0.2rem'>{url_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
