# Signal Intelligence

**RAG-powered equity signal generation system.**

RAG is the *evidence layer*. The *product* is structured signals.

```
Documents (transcripts · filings · presentations · news)
     │
     ▼
Chunking + Embeddings  →  ChromaDB
     │
     ▼
Multi-query Retrieval (filtered by quarter, section, speaker, doc_type)
     │
     ▼
LLM Reasoning over evidence
     │
     ▼
Structured Signals  →  DuckDB time-series
     │
     ▼
Streamlit Dashboard  +  FastAPI
```

The output is not "Here is a summary of the transcript."
It is: **"Management Confidence declined from 7.8 → 6.4 QoQ because margin uncertainty, China weakness, and pricing pressure increased materially."**

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env → add OPENAI_API_KEY and TAVILY_API_KEY

# 4. Run signal pipeline
python main.py run --ticker ASML --company "ASML Holding" --quarter Q2 --year 2024

# 5. Launch dashboard
python main.py dashboard

# 6. Or start the API
python main.py api
```

---

## Signals Generated

### 🎯 Management Confidence Score (0–10)
```json
{
  "company": "ASML",
  "quarter": "Q2 2024",
  "score": 6.4,
  "previous_score": 7.8,
  "change": -1.4,
  "tone": "cautious",
  "drivers": [
    "More cautious language on China demand — 'monitoring closely' replaced prior 'strong growth'",
    "Guidance range widened from $500M to $800M band — higher uncertainty signal",
    "CEO used 'challenging' 4× this quarter vs 1× last quarter"
  ],
  "summary": "Management Confidence declined 7.8→6.4 QoQ driven by China demand hedging, wider guidance bands, and more defensive language around pricing."
}
```

### 📈 Narrative Shift Score
```json
{
  "theme": "AI demand",
  "status": "accelerating",
  "evidence_count_current": 18,
  "evidence_count_previous": 9,
  "count_change": 9,
  "sentiment_current": 0.85,
  "interpretation": "AI-related demand discussion doubled QoQ; now cited as primary growth driver."
}
```

### ✅ Guidance Credibility Score (0–100)
```json
{
  "score": 72,
  "beat_rate": 0.71,
  "serial_miss_risk": false,
  "recent_pattern": ["beat","beat","miss","beat","beat","in_line"],
  "summary": "Guidance credibility 72/100. Met or exceeded revenue guidance in 5/7 periods. Margin guidance consistently optimistic."
}
```

### ⚠️ Risk Emergence Signal
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

## Architecture

```
signal_intelligence/
├── main.py                     CLI: run · api · dashboard
├── config.py                   Settings from .env
├── models.py                   Pydantic models (documents + signals)
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── fetcher.py              Tavily multi-query document discovery
│   └── chunker.py              Smart chunker: speaker + section + metadata tags
│
├── store/
│   ├── vector_store.py         ChromaDB: embed + retrieve with metadata filters
│   └── signal_store.py         DuckDB: structured signal time-series
│
├── agents/
│   ├── base.py                 BaseAgent: RAG retrieve + LLM reason
│   ├── confidence_agent.py     Management Confidence Score (0–10)
│   ├── narrative_agent.py      Narrative Shift (theme QoQ detection)
│   ├── guidance_agent.py       Guidance Credibility (0–100)
│   ├── risk_agent.py           Risk Emergence (new / escalating / diminishing)
│   └── orchestrator.py         LangGraph graph: ingest → embed → signal (parallel)
│
├── api/
│   └── main.py                 FastAPI: /run, /signals/{ticker}/*, /tickers
│
└── ui/
    └── dashboard.py            Streamlit: score gauges, trend charts, citations
```

---

## API Keys

| Key | Source | Notes |
|-----|--------|-------|
| `OPENAI_API_KEY` | platform.openai.com | Used for embeddings + reasoning |
| `TAVILY_API_KEY` | tavily.com | Free tier available |

Cost per run: ~$0.10–0.30 with `gpt-4o-mini`, ~$0.40–1.00 with `gpt-4o`.

## API Endpoints

```
POST /run                           Run pipeline for ticker + quarter
GET  /signals/{ticker}/confidence   Confidence signal history
GET  /signals/{ticker}/narrative    Narrative signal history
GET  /signals/{ticker}/guidance     Guidance signal history
GET  /signals/{ticker}/risk         Risk signal history
GET  /tickers                       All tickers with stored signals
GET  /health                        Health check
```
