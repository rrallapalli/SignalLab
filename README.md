# Signal Intelligence

A RAG-powered equity signal generation system for publicly listed stocks.

RAG is the **evidence layer**. The product is **structured signals** — not transcript summaries.

> *"Management Confidence declined from 7.8 → 6.4 QoQ because margin uncertainty, China weakness, and pricing pressure increased materially."*

---

## What it does

Signal Intelligence ingests earnings call transcripts, press releases, investor presentations, and news articles for a given stock ticker, then runs four specialised LLM agents over the retrieved evidence to produce structured, time-series-tracked signals across three time horizons automatically:

| Horizon | What it represents |
|---------|-------------------|
| **Latest** | Most recently completed quarter |
| **QoQ** | Prior quarter — shows momentum |
| **YoY** | Same quarter last year — shows structural change |

---

## Signals generated

### 🎯 Management Confidence Score (0–10)
Scores management tone across six sub-dimensions: confidence level, uncertainty, defensiveness, specificity, consistency, and forward-looking strength. Tracks QoQ and YoY delta with evidence-backed drivers.

```json
{
  "company": "ASML",
  "quarter": "Q2 2024",
  "score": 6.4,
  "previous_score": 7.8,
  "change": -1.4,
  "tone": "cautious",
  "drivers": [
    "Cautious language on China demand — 'monitoring closely' replaced prior 'strong growth'",
    "Guidance range widened from $500M to $800M band",
    "CEO used 'challenging' 4× this quarter vs 1× last quarter"
  ]
}
```

### 📈 Narrative Shift
Tracks 15+ themes (AI, Cloud, Margins, China, Competition, Hiring, Regulation, etc.) quarter-over-quarter. Classifies each as accelerating, emerging, stable, fading, or newly risky.

```json
{
  "theme": "AI demand",
  "status": "accelerating",
  "evidence_count_current": 18,
  "evidence_count_previous": 9,
  "interpretation": "AI-related demand discussion doubled QoQ; now cited as primary growth driver."
}
```

### ✅ Guidance Credibility Score (0–100)
Retrieves past guidance statements and actual results, then scores how reliably management delivers on what it says. Tracks beat/miss patterns and flags serial miss risk.

```json
{
  "score": 72,
  "beat_rate": 0.71,
  "serial_miss_risk": false,
  "recent_pattern": ["beat", "beat", "miss", "beat", "beat", "in_line"]
}
```

### ⚠️ Risk Emergence
Detects newly material or escalating risks by comparing mention counts and context across quarters.

```json
{
  "risk": "deposit competition",
  "status": "newly_material",
  "severity": "high",
  "mention_count_current": 11,
  "mention_count_previous": 3,
  "evidence": "Mentioned 11× vs 3× prior quarter; CEO flagged as primary NIM headwind for H2."
}
```

---

## How it works

```
Tavily search (earnings calls · press releases · investor days · news)
        │
        ▼
Chunk + tag  →  ChromaDB
(speaker, section, quarter, doc_type, is_management)
        │
        ▼
Multi-query RAG  →  evidence blocks per agent
        │
        ▼
LLM reasoning  →  structured JSON signal + citations
        │
        ▼
DuckDB time-series  →  Streamlit dashboard
```

Each agent uses targeted retrieval queries specific to its signal type — confidence agents filter for management-only chunks from prepared remarks and Q&A, risk agents target risk_factors and QA sections, guidance agents pull across all stored quarters to compare stated vs actual.

Agents run **sequentially with staggered starts** across quarters to avoid OpenAI rate limits. LLM calls include **automatic retry** with exponential backoff. Each signal save is isolated so one failure cannot prevent others from completing.

---

## Project structure

```
SignalLab/
├── main.py                     CLI entry point
├── config.py                   Settings loaded from .env
├── models.py                   Pydantic models for documents and signals
├── diagnose_db.py              Database inspection utility
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── fetcher.py              Tavily parallel search + HTML fetching
│   └── chunker.py              Sentence-boundary chunker with metadata tagging
│
├── store/
│   ├── vector_store.py         ChromaDB wrapper with metadata-filtered retrieval
│   └── signal_store.py         DuckDB signal time-series + YTD aggregations
│
├── agents/
│   ├── base.py                 RAG retrieval + LLM reasoning with retry
│   ├── confidence_agent.py     Management Confidence Score (0–10)
│   ├── narrative_agent.py      Narrative Shift (theme QoQ detection)
│   ├── guidance_agent.py       Guidance Credibility Score (0–100)
│   ├── risk_agent.py           Risk Emergence (new / escalating / diminishing)
│   └── orchestrator.py         LangGraph graph: ingest → chunk → signal
│
├── api/
│   └── main.py                 FastAPI backend
│
└── ui/
    └── dashboard.py            Streamlit dashboard (Latest / QoQ / YoY)
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph |
| Document search | Tavily API |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local persistent) |
| Signal storage | DuckDB (local) |
| LLM reasoning | OpenAI GPT-4o / GPT-4o-mini |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |

---

## API keys required

| Key | Where to get it | Free tier |
|-----|----------------|-----------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | No — pay per use |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Yes — 1,000 searches/month |

**Estimated cost per ticker run:** ~$0.05–0.15 with `gpt-4o-mini` · ~$0.40–1.00 with `gpt-4o`

---

## API endpoints

```
POST /run                           Run pipeline for ticker + quarter
GET  /signals/{ticker}/confidence   Confidence signal history
GET  /signals/{ticker}/narrative    Narrative signal history
GET  /signals/{ticker}/guidance     Guidance signal history
GET  /signals/{ticker}/risk         Risk signal history
GET  /tickers                       All tickers with stored signals
GET  /health                        Health check
```

---

## See also

- **[HOWTORUN.md](HOWTORUN.md)** — step-by-step setup and usage guide
