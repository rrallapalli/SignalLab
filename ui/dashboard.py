"""
ui/dashboard.py – Latest / QoQ / YoY comparison dashboard.
Run: streamlit run ui/dashboard.py
"""

from __future__ import annotations
import sys, json, asyncio, threading
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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background:#0f0f1a; }
  .col-header {
    text-align:center; font-size:0.78rem; font-weight:700;
    letter-spacing:0.08em; text-transform:uppercase;
    padding:0.4rem 0; border-radius:6px; margin-bottom:0.6rem;
  }
  .col-latest  { background:#1a1040; color:#c4b5fd; }
  .col-qoq     { background:#0c1a1a; color:#67e8f9; }
  .col-yoy     { background:#0c1a0c; color:#86efac; }
  .score-big   { font-size:2.4rem; font-weight:800; text-align:center; }
  .delta-up    { color:#22c55e; font-size:1rem; font-weight:700; text-align:center; }
  .delta-down  { color:#ef4444; font-size:1rem; font-weight:700; text-align:center; }
  .delta-flat  { color:#94a3b8; font-size:1rem; font-weight:700; text-align:center; }
  .signal-card {
    background:#1e1e2e; border-radius:10px;
    padding:1rem 1.2rem; margin-bottom:0.8rem;
    border-left:4px solid #7c3aed;
  }
  .driver-item {
    padding:0.5rem 0.7rem; background:#12121f; border-radius:7px;
    border-left:3px solid #7c3aed; margin:0.35rem 0; color:#e2e8f0; font-size:0.87rem;
  }
  .risk-card {
    background:#1e1e2e; border-radius:8px; padding:0.8rem 1rem;
    margin:0.4rem 0; border-left:3px solid #ef4444;
  }
  .status-pill {
    display:inline-block; padding:0.15rem 0.6rem; border-radius:999px;
    font-size:0.75rem; font-weight:700;
  }
  .pill-green  { background:#14532d; color:#4ade80; }
  .pill-red    { background:#450a0a; color:#f87171; }
  .pill-yellow { background:#422006; color:#fbbf24; }
  .pill-purple { background:#2e1065; color:#c4b5fd; }
  .pill-orange { background:#431407; color:#fb923c; }
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
def _stores():
    return VectorStore(), SignalStore()

vs, ss = _stores()


# ── Score helpers ─────────────────────────────────────────────────────────────

def _color(score: float, max_val: float = 10) -> str:
    p = score / max_val
    return "#22c55e" if p >= 0.75 else "#eab308" if p >= 0.55 else "#f97316" if p >= 0.35 else "#ef4444"


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
    st.caption("Latest · QoQ · YoY comparison")
    st.divider()

    st.markdown("### Run Analysis")
    ticker_in  = st.text_input("Ticker", placeholder="INFY, TCS, MSFT…", max_chars=10).upper().strip()
    company_in = st.text_input("Company name", placeholder="Infosys Limited")

    st.markdown("**Quarter** *(optional — leave blank for auto-detect)*")
    col_q, col_y = st.columns(2)
    with col_q:
        quarter_in = st.selectbox("Q", ["Auto", "Q1","Q2","Q3","Q4"], index=0, label_visibility="collapsed")
    with col_y:
        year_in = st.number_input("Year", min_value=2019, max_value=2030, value=2024, step=1)

    quarter_val = None if quarter_in == "Auto" else quarter_in
    year_val    = None if quarter_in == "Auto" else int(year_in)

    if quarter_in == "Auto":
        from agents.orchestrator import _latest_completed_quarter
        auto_q, auto_yr = _latest_completed_quarter()
        st.caption(f"Auto-detect: **{auto_q} {auto_yr}** as latest")

    model_choice = st.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0)
    run_disabled = not (ticker_in and company_in)
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                        disabled=run_disabled)

    st.divider()
    st.markdown("### Stored Tickers")
    stored = ss.get_all_tickers()
    view_ticker = st.selectbox("Load", ["— choose —"] + stored) if stored else "— choose —"
    if not stored: st.caption("No signals stored yet.")


# ════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ════════════════════════════════════════════════════════════

if run_btn:
    from config import settings as cfg
    cfg.OPENAI_MODEL = model_choice
    from agents.orchestrator import run_comparison_pipeline

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
        st.info("Check your `.env` has valid `OPENAI_API_KEY` and `TAVILY_API_KEY`.")
        st.stop()


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
                "score": conf_row.get("score"), "change": conf_row.get("change"),
                "tone": conf_row.get("tone"), "summary": conf_row.get("summary",""),
                "drivers": conf_row.get("drivers",[]),
                "confidence_level": conf_row.get("confidence_level"),
                "uncertainty_level": conf_row.get("uncertainty_level"),
                "defensiveness": conf_row.get("defensiveness"),
                "specificity": conf_row.get("specificity"),
                "consistency": conf_row.get("consistency"),
                "forward_strength": conf_row.get("forward_strength"),
            },
            "narrative": narr_row,
            "guidance":  guid_row,
            "risk":      risk_row,
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
_, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
with c1: st.markdown(f"<div class='col-header col-latest'>📍 Latest<br>{label_l}</div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='col-header col-qoq'>↔ QoQ vs<br>{label_q}</div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='col-header col-yoy'>📅 YoY vs<br>{label_y}</div>", unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Confidence", "📈 Narrative", "✅ Guidance", "⚠️ Risks", "🕒 Trend"
])


# ── Tab 1: Management Confidence ──────────────────────────────────────────────

with tab1:

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
    label_col, l_col, q_col, y_col = st.columns([1.4,1,1,1])
    with label_col: st.markdown("**Score**")
    with l_col:
        st.markdown(_score_cell(l_score), unsafe_allow_html=True)
        st.plotly_chart(_gauge(l_score, 10), use_container_width=True, key="chart_1")
    with q_col:
        if q_score:
            st.markdown(_score_cell(q_score), unsafe_allow_html=True)
            st.markdown(_delta_html(l_score, q_score), unsafe_allow_html=True)
            st.plotly_chart(_gauge(q_score, 10), use_container_width=True, key="chart_2")
        else:
            st.caption("No data")
    with y_col:
        if y_score:
            st.markdown(_score_cell(y_score), unsafe_allow_html=True)
            st.markdown(_delta_html(l_score, y_score), unsafe_allow_html=True)
            st.plotly_chart(_gauge(y_score, 10), use_container_width=True, key="chart_3")
        else:
            st.caption("No data")

    st.divider()

    # Sub-dimensions
    st.markdown("#### Sub-dimension Breakdown")
    dims = [
        ("Confidence Level",  "confidence_level"),
        ("Low Uncertainty",   "uncertainty_level"),
        ("Not Defensive",     "defensiveness"),
        ("Specificity",       "specificity"),
        ("Consistency",       "consistency"),
        ("Forward Strength",  "forward_strength"),
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
        for dim_name, dim_key in dims:
            _lv_raw = lc.get(dim_key)
            _qv_raw = qc.get(dim_key)
            _yv_raw = yc.get(dim_key)
            # Treat None AND 0.0 as 'not scored' — show — instead of a misleading zero bar
            lv = float(_lv_raw) if (_lv_raw is not None and float(_lv_raw) > 0) else None
            qv = float(_qv_raw) if (_qv_raw is not None and float(_qv_raw) > 0) else None
            yv = float(_yv_raw) if (_yv_raw is not None and float(_yv_raw) > 0) else None

            lbl, l_d, q_d, y_d = st.columns([1.4,1,1,1])
            with lbl:
                st.markdown(
                    f"<span style='color:#94a3b8;font-size:0.85rem'>{dim_name}</span>",
                    unsafe_allow_html=True,
                )
            with l_d:
                if lv is not None:
                    c = _color(lv)
                    bar = (
                        f"<div style='background:#0f172a;border-radius:3px;height:6px'>"
                        f"<div style='background:{c};width:{lv*10:.0f}%;height:6px;border-radius:3px'></div></div>"
                    )
                    st.markdown(
                        f"<span style='color:{c};font-weight:700'>{lv:.1f}</span> {bar}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#475569' title='Re-run pipeline to score this dimension'>— </span>",
                        unsafe_allow_html=True,
                    )
            with q_d:
                if qv is not None and lv is not None:
                    delta = lv - qv
                    sym = "▲" if delta > 0 else "▼" if delta < 0 else "—"
                    dc = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#94a3b8"
                    st.markdown(
                        f"<span style='color:#94a3b8'>{qv:.1f}</span> "
                        f"<span style='color:{dc};font-size:0.8rem'>{sym}{abs(delta):.1f}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<span style='color:#475569'>—</span>", unsafe_allow_html=True)
            with y_d:
                if yv is not None and lv is not None:
                    delta = lv - yv
                    sym = "▲" if delta > 0 else "▼" if delta < 0 else "—"
                    dc = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#94a3b8"
                    st.markdown(
                        f"<span style='color:#94a3b8'>{yv:.1f}</span> "
                        f"<span style='color:{dc};font-size:0.8rem'>{sym}{abs(delta):.1f}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<span style='color:#475569'>—</span>", unsafe_allow_html=True)

    # Show note if any dimension is missing
    _missing_dims = [name for name, key in dims if not (lc.get(key) and float(lc.get(key, 0)) > 0)]
    if _missing_dims and _has_subdim_data:
        st.caption(
            f"ℹ️ {len(_missing_dims)} dimension(s) not scored for this period "
            "(shown as —). Re-run the pipeline to populate them."
        )

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

    def _narr(b):
        if b is None: return {}
        n = b.get("narrative") if isinstance(b, dict) and "narrative" in b else b
        return n or {}

    ln = _narr(latest_b)
    qn = _narr(qoq_b)
    yn = _narr(yoy_b)

    shift_map  = {"positive":"#22c55e","negative":"#ef4444","mixed":"#f97316","neutral":"#94a3b8"}

    # Shift row
    lbl_col, l_col, q_col, y_col = st.columns([1.4,1,1,1])
    with lbl_col: st.markdown("**Narrative Shift**")
    for col, nd, lbl in [(l_col, ln, label_l), (q_col, qn, label_q), (y_col, yn, label_y)]:
        with col:
            shift = nd.get("overall_shift","—") or "—"
            c = shift_map.get(shift, "#94a3b8")
            st.markdown(f"<div style='text-align:center;font-size:1.5rem;font-weight:800;color:{c}'>{shift.upper()}</div>", unsafe_allow_html=True)
            st.caption(lbl)

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


# ── Tab 3: Guidance Credibility ───────────────────────────────────────────────

with tab3:

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
        _rate_color = "#22c55e" if _rate >= 0.7 else "#eab308" if _rate >= 0.5 else "#ef4444"

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
            f"<div style='font-size:1.6rem;font-weight:800;color:{_rate_color}'>{_rate:.0%}</div>",
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

    lbl_col, l_col, q_col, y_col = st.columns([1.4,1,1,1])
    with lbl_col: st.markdown("**Guidance Score**")
    with l_col:
        st.markdown(_score_cell(l_gs, 100), unsafe_allow_html=True)
        st.plotly_chart(_gauge(l_gs, 100), use_container_width=True, key="chart_4")
    with q_col:
        if q_gs:
            st.markdown(_score_cell(q_gs, 100), unsafe_allow_html=True)
            st.markdown(_delta_html(l_gs, q_gs), unsafe_allow_html=True)
            st.plotly_chart(_gauge(q_gs, 100), use_container_width=True, key="chart_5")
        else: st.caption("No data")
    with y_col:
        if y_gs:
            st.markdown(_score_cell(y_gs, 100), unsafe_allow_html=True)
            st.markdown(_delta_html(l_gs, y_gs), unsafe_allow_html=True)
            st.plotly_chart(_gauge(y_gs, 100), use_container_width=True, key="chart_6")
        else: st.caption("No data")

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
                st.markdown(f"<div class='signal-card'><p style='color:#e2e8f0;font-size:0.82rem;margin:0'>{summary[:200]}…</p></div>", unsafe_allow_html=True)


# ── Tab 4: Risk Emergence ─────────────────────────────────────────────────────

with tab4:

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

    lbl_col, l_col, q_col, y_col = st.columns([1.4,1,1,1])
    with lbl_col: st.markdown("**Risk Direction**")
    for col, rd, lbl in [(l_col,lr,label_l),(q_col,qr,label_q),(y_col,yr_,label_y)]:
        with col:
            rdir = rd.get("overall_risk_direction","—") or "—"
            rc = risk_dir_map.get(rdir,"#94a3b8")
            st.markdown(f"<div style='text-align:center;font-size:1.5rem;font-weight:800;color:{rc}'>{rdir.upper()}</div>", unsafe_allow_html=True)
            new_n = len(rd.get("new_risks",[]) or [])
            esc_n = len(rd.get("escalating",[]) or [])
            st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:0.8rem'>🆕{new_n} new · 📈{esc_n} esc</div>", unsafe_allow_html=True)

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
                st.markdown(f"<div style='color:#64748b;font-size:0.8rem;margin-top:0.5rem'>{summary[:180]}…</div>", unsafe_allow_html=True)


# ── Tab 5: Trend History ──────────────────────────────────────────────────────

with tab5:
    if len(conf_history) < 2:
        st.info("Run the pipeline for additional quarters to see trend data.")
    else:
        st.markdown("### Signal Trends — Full History")

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
        for cr in conf_history:
            lbl = f"{cr.get('quarter','')} {cr.get('fiscal_year','')}"
            gr  = next((r for r in guid_history if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == lbl), {})
            nr  = next((r for r in narr_history if f"{r.get('quarter','')} {r.get('fiscal_year','')}" == lbl), {})
            rr  = _get_risk_row(lbl)
            current_lbl = lbl == label_l
            all_rows.append({
                "Quarter":    ("📍 " if current_lbl else "") + lbl,
                "Confidence": f"{float(cr.get('score',0)):.1f}/10",
                "Δ Conf":     f"{float(cr.get('change',0) or 0):+.1f}",
                "Tone":       (cr.get("tone","") or "").title(),
                "Guidance":   f"{float(gr.get('score',0)):.0f}/100" if gr else "—",
                "Narrative":  (nr.get("overall_shift","") or "").title() if nr else "—",
                "Risk":       (rr.get("overall_risk_direction","") or "").title() if rr else "—",
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)
