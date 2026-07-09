# How to Run Signal Intelligence

---

## Prerequisites

- Python **3.11** or **3.12**
- One API key (OpenAI) — document ingestion from NSE/BSE needs no key
- Network access to `nseindia.com` and `bseindia.com` (both are public but geo/rate sensitive; a VPN inside India can help if requests are being blocked)
- ~500MB disk space for the virtual environment

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/rrallapalli/SignalLab.git
cd SignalLab
```

---

## Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes 2–3 minutes the first time.

---

## Step 4 — Get your API key

**OpenAI** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create an account, add a payment method, generate a key
- Recommended model: `gpt-4o-mini` (fast and cheap for development)

No key is needed for document ingestion — NSE and BSE corporate-announcement
filings (results, investor decks, concall transcripts, annual reports) are
fetched directly and are public.

---

## Step 5 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
```

Optional settings you can also add to `.env`:

```
OPENAI_MODEL=gpt-4o-mini    # or gpt-4o for higher quality
OPENAI_TEMPERATURE=0.0
TOP_K_RETRIEVAL=12
```

---

## Step 6 — Launch the dashboard

```bash
streamlit run ui/dashboard.py
```

This opens `http://localhost:8501` in your browser automatically.

---

## Step 7 — Run your first analysis

In the dashboard sidebar:

1. Enter a **ticker** — the NSE trading symbol, e.g. `TCS`, `INFY`, `RELIANCE`, `HDFCBANK`
2. Enter the **company name** — e.g. `Tata Consultancy Services` (used to resolve the matching BSE scrip code)
3. Leave **Quarter** set to `Auto` — the system detects the latest completed quarter and automatically runs **Latest + QoQ + YoY** in one shot
4. Select a model — `gpt-4o-mini` is recommended for first runs
5. Click **🚀 Run Analysis**

The pipeline will:
- Pull corporate-announcement filings (results, investor presentations, concall transcripts, annual reports) directly from NSE and BSE for the relevant date windows
- Download and extract text from each PDF attachment
- Chunk and embed all documents into ChromaDB
- Run four signal agents (Confidence, Narrative, Guidance, Risk) across three quarters
- Save all results to a local DuckDB database
- Display the full dashboard automatically

**Good tickers to start with:** `TCS`, `INFY`, `RELIANCE`, `HDFCBANK`, `ICICIBANK`

---

## Running a specific quarter

If you want to anchor the analysis to a specific quarter rather than the auto-detected latest:

1. Change the **Quarter** dropdown from `Auto` to e.g. `Q2`
2. Set the **Year** to e.g. `2024`
3. Click **Run Analysis**

The system will then run signals for Q2 2024 (Latest), Q1 2024 (QoQ), and Q2 2023 (YoY).

---

## Loading previously analysed tickers

Once you have run a ticker, it appears in the **Stored Tickers** dropdown in the sidebar. Select it to reload the full dashboard from the local database without re-running the pipeline.

---

## Diagnosing missing data

If sections appear blank or show "No data" for some periods, run the diagnostic tool:

```bash
python diagnose_db.py

# Focus on one ticker
python diagnose_db.py --ticker TCS
```

This shows a grid of which signals (Confidence / Narrative / Guidance / Risk) are stored for each quarter:

```
TCS
  Q1 2026   Conf:✅  Narr:✅  Guid:✅  Risk:✅
  Q4 2025   Conf:✅  Narr:❌  Guid:✅  Risk:✅
  Q1 2025   Conf:❌  Narr:❌  Guid:❌  Risk:❌
```

Any ❌ means that signal was not stored — re-run the pipeline for that ticker to fill the gap.

---

## Resetting the database

If you want a completely clean start:

```bash
rm data/signals.duckdb      # removes all stored signals
rm -rf data/chroma/         # removes all vector embeddings
```

Then re-run your tickers. Each ticker takes one click to fully populate all three quarters.

---

## Optional — Start the FastAPI backend

```bash
python main.py api
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

Useful endpoints:

```
POST /run                           Trigger pipeline programmatically
GET  /signals/{ticker}/confidence   Get confidence signal history
GET  /signals/{ticker}/narrative    Get narrative signal history
GET  /signals/{ticker}/guidance     Get guidance signal history
GET  /signals/{ticker}/risk         Get risk signal history
GET  /tickers                       List all stored tickers
```

---

## Troubleshooting

**`ModuleNotFoundError`**
Make sure your virtual environment is active: `source .venv/bin/activate`

**`chromadb` install fails on Windows**
```bash
pip install chromadb --only-binary :all:
```

**`duckdb` version conflict**
```bash
pip install duckdb==1.0.0 --force-reinstall
```

**Pipeline fails with "Pipeline failed" in the dashboard**
- Check that `OPENAI_API_KEY` in `.env` is valid
- Confirm the ticker is a real NSE symbol and that this machine can reach `nseindia.com` and `bseindia.com` (some cloud/CI networks block these)
- Try switching to `gpt-4o-mini` in the model selector — it has higher rate limits
- Check `data/signal_agent.log` for detailed error messages

**No documents found for a ticker**
- Double check the NSE symbol spelling and that the company name closely matches its official listed name (used to resolve the BSE scrip code)
- Smaller/thinly-covered companies may not publish concall transcripts as a formal filing — investor presentations and financial-results filings are usually still available
- Widen `NSE_BSE_RESULT_LAG_DAYS` in `.env` if results were declared later than usual that quarter

**Sub-dimension scores showing `—`**
The data exists but sub-dimensions were not scored in that run. Re-run the pipeline for that ticker — the fixed pipeline will populate all six sub-dimensions.

**Rate limit errors**
The pipeline staggers LLM calls and retries automatically. If you consistently hit limits, add `OPENAI_MODEL=gpt-4o-mini` to your `.env` — it has a much higher rate limit than gpt-4o.
