# 📡 Signal Intelligence — AI-Powered Equity Signal Engine for India-Listed Companies

### Multi-Agent RAG · Direct NSE/BSE Ingestion · Sector-Aware Signal Synthesis

---

## 🚀 What This Project Demonstrates

Signal Intelligence turns raw corporate disclosures — quarterly results, investor
presentations, and concall transcripts, fetched directly from NSE and BSE — into
structured, evidence-backed investment signals for India-listed equities.

RAG is the **evidence layer**, not the product. The output is never "here is a
summary of the transcript." It is:

> **"Management Confidence declined 7.8 → 6.4 QoQ because margin uncertainty,
> deposit-cost pressure, and cautious language on loan growth increased materially."**

The project showcases:

- Direct-from-exchange document ingestion — no third-party search API in the loop
- A market-aware + sector-aware theme taxonomy (a bank's NIM commentary and an
  IT services firm's deal pipeline are tracked as what they actually are, not
  squeezed into a generic "AI/Cloud/China" template)
- A four-agent signal synthesis layer (Confidence, Narrative, Guidance, Risk),
  each independently retrieving evidence and reasoning over it
- Automatic three-way time comparison (Latest / QoQ / YoY) in a single run
- Evidence-cited output — every signal traces back to the source filing
- A Streamlit dashboard and a FastAPI service over the same signal store

---

## 🎯 Business Problem

Sell-side and buy-side associate analysts spend a disproportionate amount of
earnings season doing the same manual work for every name they cover:

- Reading (or skimming) the results PDF, investor deck, and concall transcript
- Comparing management's tone and language to the prior quarter, by memory
- Checking whether guidance given last quarter was actually met this quarter
- Noticing which risks are newly showing up in Q&A vs. being repeated boilerplate

None of this is hard analytically — it's just slow, repetitive, and easy to miss
a subtle shift in language across a 60-minute call. This project asks:

1. Can tone, guidance accountability, and risk emergence be scored consistently,
   quarter after quarter, directly from the primary disclosures?
2. Can that scoring be sector-aware, instead of applying the same generic theme
   list to a bank, a pharma company, and an IT services firm?
3. Can every signal stay traceable to its source, so it's usable as analyst
   evidence rather than a black-box LLM opinion?

---

## 🧠 Key Differentiators

### RAG is the evidence layer, not the product
Retrieval feeds evidence to four specialized agents that reason over it and
emit structured, cited signals — not a document summary.

### Direct-from-exchange ingestion
No web-search API sits between the company and the pipeline. NSE and BSE
corporate-announcement filings (results, investor presentations, concall
transcripts, annual reports) are fetched and parsed directly, with the PDF
attachment text extracted and chunked with metadata (speaker, section,
quarter, document type).

### Market-aware + sector-aware theme tracking
Every company is tracked against a shared **India macro layer** (RBI policy,
inflation, INR, monsoon/rural demand, government capex, GST, FII/DII flows)
*plus* a **sector-specific theme set**, auto-detected from the company name
and ticker across 11 sectors — Banking/NBFC, IT Services, Pharma, FMCG, Auto,
Metals/Cement, Energy/Power, Telecom, Infrastructure, and Real Estate. A bank's
narrative agent looks for NIM compression and asset quality; an IT services
firm's looks for deal TCV and attrition — not a one-size-fits-all US-tech
theme list.

### Guidance accountability, not just sentiment
The Guidance agent tracks whether specific numeric commitments made in past
quarters were actually met, missed, or withdrawn — and flags serial-miss risk.

---

## 🧱 Architecture

```
NSE + BSE corporate announcements
(results · investor decks · concall transcripts · annual reports)
        │
        ▼
Download PDF attachments  →  extract text (pypdf)
        │
        ▼
Chunk + tag  →  ChromaDB
(speaker, section, quarter, doc_type, is_management)
        │
        ▼
Multi-query RAG  →  sector-aware evidence retrieval per agent
        │
        ▼
Four signal agents (parallel, LangGraph-orchestrated):
  🎯 Confidence   📈 Narrative   ✅ Guidance   ⚠️ Risk
        │
        ▼
LLM reasoning (OpenAI JSON mode)  →  structured signal + citations
        │
        ▼
DuckDB time-series  →  Streamlit dashboard  +  FastAPI
```

---

## 📊 Signals Generated

### 🎯 Management Confidence Score (0–10)

```json
{
  "company": "Tata Consultancy Services",
  "quarter": "Q2 2026",
  "score": 6.5,
  "previous_score": 7.5,
  "change": -1.0,
  "tone": "cautious",
  "drivers": [
    "Cautious language on discretionary IT spend — 'monitoring closely' replaced prior 'strong growth'",
    "Deal-pipeline commentary softened versus last quarter's investor call",
    "CEO used 'challenging' 4x this quarter vs 1x last quarter"
  ],
  "summary": "Management Confidence declined 7.5 -> 6.5 QoQ driven by cautious language on discretionary spend and a softer deal-pipeline tone."
}
```

### 📈 Narrative Shift Score (sector-aware)

```json
{
  "theme": "Net Interest Margin (NIM) / Spread Compression",
  "status": "newly_risky",
  "evidence_count_current": 12,
  "evidence_count_previous": 4,
  "count_change": 8,
  "sentiment_current": -0.4,
  "interpretation": "Management flagged NIM compression from deposit repricing for the first time this quarter, versus a confident tone last quarter."
}
```

### ✅ Guidance Credibility Score (0–100)

```json
{
  "score": 72,
  "beat_rate": 0.71,
  "serial_miss_risk": false,
  "recent_pattern": ["beat", "beat", "miss", "beat", "beat", "in_line"],
  "summary": "Guidance credibility 72/100. Met or exceeded loan-growth guidance in 5/7 periods. Margin guidance consistently optimistic relative to outcomes."
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
  "evidence": "Mentioned 11x vs 3x prior quarter; CEO flagged as primary NIM headwind for H2."
}
```

---

## 🧠 Analytical Flow

### 1. Document Discovery & Ingestion
NSE (`nse` package) and BSE (`bse` package) are queried directly for
corporate-announcement filings in a date window around the target quarter
(quarter-end + up to 90 days for results/decks/transcripts, +200 days if
annual reports are requested). PDF attachments are downloaded and text is
extracted with `pypdf`.

### 2. Chunking & Indexing
Documents are chunked with metadata — speaker, section, quarter, fiscal year,
document type, whether the speaker is management — and embedded into ChromaDB.

### 3. Sector Detection
The company name and ticker are matched against an 11-sector taxonomy to
select the right theme list and targeted RAG queries for the Narrative agent.
Falls back to a general theme set if no sector match is found.

### 4. Multi-Agent Signal Synthesis
Four agents run per quarter, each retrieving its own targeted evidence and
reasoning over it independently via an LLM call constrained to OpenAI's JSON
mode (guaranteeing syntactically valid structured output):

| Agent | Signal | Range |
|---|---|---|
| `confidence_agent.py` | Management tone/confidence | 0–10 |
| `narrative_agent.py` | Theme-level QoQ narrative shift | qualitative + sentiment |
| `guidance_agent.py` | Guidance-to-actual accountability | 0–100 |
| `risk_agent.py` | New/escalating/diminishing risk | severity-scored |

### 5. Three-Way Time Comparison
Every run automatically fetches and scores **Latest**, **QoQ**, and **YoY**
in one pass, so momentum and structural change are both visible without a
second run.

### 6. Storage & Presentation
Signals are persisted to DuckDB as a time series and surfaced through a
Streamlit dashboard (gauges, trend charts, evidence citations) or a FastAPI
service for programmatic access.

---

## 📂 Project Structure

```
SignalLab/
├── main.py                     CLI: run · api · dashboard
├── config.py                   Settings from .env (incl. NSE/BSE fetch tuning)
├── models.py                   Pydantic models (documents + signals)
├── diagnose_fetch.py           Standalone NSE/BSE connectivity + PDF download check
├── diagnose_db.py              Standalone DuckDB signal-coverage check
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── fetcher.py               NSE + BSE corporate-announcement fetch, PDF text extraction
│   └── chunker.py                Speaker + section + metadata-aware chunking
│
├── store/
│   ├── vector_store.py          ChromaDB: embed + retrieve with metadata filters
│   └── signal_store.py           DuckDB: structured signal time-series
│
├── agents/
│   ├── base.py                   BaseAgent: RAG retrieve + LLM reason (OpenAI JSON mode)
│   ├── confidence_agent.py       Management Confidence Score (0–10)
│   ├── narrative_agent.py        Narrative Shift — market + sector-aware theme taxonomy
│   ├── guidance_agent.py         Guidance Credibility (0–100)
│   ├── risk_agent.py             Risk Emergence (new / escalating / diminishing)
│   └── orchestrator.py           LangGraph graph: ingest → embed → signal (parallel, 3-quarter)
│
├── api/
│   └── main.py                   FastAPI: /run, /signals/{ticker}/*, /tickers, /health
│
├── ui/
│   └── dashboard.py               Streamlit: score gauges, trend charts, citations
│
├── retrieval/                     Reserved for future standalone retrieval utilities
└── validation/                    Reserved for future signal-quality validation tooling
```

---

## 🛠️ Tech Stack

- **LangGraph** — agent orchestration
- **NSE / BSE direct fetch** (`nse`, `bse` packages) — document ingestion, no third-party search API
- **ChromaDB** — vector store for chunked evidence
- **DuckDB** — structured signal time-series storage
- **OpenAI** — embeddings + reasoning (JSON mode for guaranteed-valid structured output)
- **FastAPI** — programmatic access to the signal store
- **Streamlit** — interactive dashboard
- **pypdf** — PDF attachment text extraction

---

## ▶️ How to Run

Refer to [HOWTORUN.md](HOWTORUN.md) for full setup instructions.

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key (only OpenAI is needed — NSE/BSE ingestion is keyless)
cp .env.example .env
# Edit .env → add OPENAI_API_KEY

# 4. Sanity-check NSE/BSE connectivity before a real run
python diagnose_fetch.py --ticker TCS --company "Tata Consultancy Services"

# 5. Launch the dashboard (recommended entry point)
streamlit run ui/dashboard.py

# Or start the API
python main.py api
```

---

## 🔑 API Keys & Cost

| Key | Source | Notes |
|---|---|---|
| `OPENAI_API_KEY` | platform.openai.com | Used for embeddings + reasoning |

Document ingestion needs **no API key** — NSE and BSE corporate-announcement
filings are public. You do need outbound network access to `nseindia.com`
and `bseindia.com` from wherever the pipeline runs.

Cost per run: ~$0.10–0.30 with `gpt-4o-mini`, ~$0.40–1.00 with `gpt-4o`.

---

## 🌐 API Endpoints

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

## 💡 Key Insight

> Sector context changes what "signal" even means. A confident tone and a
> falling risk count mean something different for a bank managing NIM
> compression than for an IT services firm managing deal pipeline softness.
> Treating RAG as the evidence layer — and routing that evidence through a
> theme taxonomy that actually matches the company's sector — is what makes
> the output usable as analyst-grade evidence rather than a generic LLM
> opinion of a transcript.

---

## ⚠️ Notes & Known Limitations

- **This is an analytical tool, not investment advice.** Signals are LLM-
  generated interpretations of public disclosures and should be treated as
  analyst input, not a trading recommendation.
- **Sector detection is keyword-based**, matched against the company name and
  ticker against an 11-sector taxonomy. Unusual or ambiguous company names can
  fall through to a general theme set; sector can also be passed explicitly
  to `NarrativeAgent.run(..., sector=...)` to override auto-detection.
- **BSE scrip-code resolution is a fuzzy company-name lookup** and can
  occasionally resolve to the wrong listing. `fetch_documents(...,
  bse_scripcode=...)` accepts an explicit override if needed.
- **NSE's session handshake and both exchanges' endpoints are unofficial /
  publicly-scraped**, not a documented partner API — they can change format
  or rate-limit without notice. `diagnose_fetch.py` isolates each step
  (NSE handshake → NSE announcements → BSE scrip lookup → BSE announcements →
  PDF download) for quick troubleshooting.
- **The CLI `run` command and the FastAPI `/run` endpoint currently import a
  `run_pipeline` function that doesn't exist** in `agents/orchestrator.py`
  (only `run_comparison_pipeline` does) — a pre-existing gap unrelated to the
  NSE/BSE ingestion work. The Streamlit dashboard is unaffected and is the
  supported entry point today.
- **Guidance and risk tracking quality depends on disclosure completeness**
  for a given ticker/quarter — thinly-covered smaller companies may not file
  a formal concall transcript, in which case results filings and investor
  presentations still populate most signals.

---

## 📬 Contact

If you're interested in discussing equity research tooling, RAG system design,
or Data/AI solutions more broadly, feel free to connect.

Rakesh Rallapalli
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/rakesh-rallapalli/)
