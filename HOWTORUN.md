# ▶️ How to Run — Signal Intelligence

There are two ways to run SignalLab:

| | Who it's for | Time |
|---|---|---|
| **🚀 Quick Start** — double-click a start file | Anyone. No terminal, no Python knowledge. | ~5 min first time, ~10 sec after |
| **🧰 Manual Setup** — venv + pip by hand | Developers who want control over the environment | ~10 min |

Both end up in exactly the same place. Start with Quick Start.

---

# 🚀 Quick Start

## 1 — Install Python (one time)

Download from **[python.org/downloads](https://www.python.org/downloads/)** — version **3.10 to 3.13**.

> **Windows users:** on the first installer screen, tick the box that says
> **"Add python.exe to PATH"** before clicking Install. This is the single
> most common thing people miss.

macOS and most Linux distributions already ship with Python — the start file
will tell you if yours is too old.

## 2 — Get SignalLab onto your computer

**🍎 macOS — use this method.** Open **Terminal** (press `Cmd+Space`, type
"Terminal", Enter) and paste:

```bash
git clone https://github.com/rrallapalli/SignalLab.git ~/SignalLab
cd ~/SignalLab
```

This is not just a shortcut for developers. Files that arrive via a **browser
download get tagged by macOS as untrusted**, and recent versions refuse to run
them with a warning whose only buttons are *Done* and *Move to Bin* — no way to
proceed. `git clone` doesn't apply that tag, so the app just runs. If you don't
have `git`, macOS offers to install it the first time you type the command.

**🪟 Windows / 🐧 Linux.** On the
[GitHub page](https://github.com/rrallapalli/SignalLab): green **Code** button
→ **Download ZIP** → unzip it.

### Where to put the folder

Somewhere **not synced to the cloud**. SignalLab writes ~500MB of libraries
plus a local database into its own folder, and cloud sync tools fight with
that — especially when the drive is full.

| | Good | Avoid |
|---|---|---|
| 🍎 macOS | `~/SignalLab` (your home folder) | **iCloud-synced `Documents` or `Desktop`** |
| 🪟 Windows | `C:\SignalLab` or `C:\Users\you\SignalLab` | OneDrive-synced folders |
| 🐧 Linux | anywhere in your home folder | — |

> **macOS:** `Documents` and `Desktop` are synced to iCloud by default. Check
> System Settings → your name → iCloud → iCloud Drive. If files in the folder
> show a ☁️ cloud icon in Finder, move it: `mv ~/Documents/SignalLab ~/SignalLab`

## 3 — Start it

**🍎 macOS** — in the Terminal window you already have open:

```bash
cd ~/SignalLab
./start.command
```

**🪟 Windows** — double-click **`start.bat`**.

**🐧 Linux** — double-click **`start.sh`**, or run `./start.sh` in a terminal.

A black terminal window opens (or the one you're in gets busy). **That's
normal — leave it open.** It is SignalLab's engine; closing it stops the app.

> **🍎 macOS — "Apple could not verify start.command is free of malware"?**
> You downloaded the ZIP instead of cloning. Don't click *Move to Bin*, click
> **Done**, then in Terminal:
>
> ```bash
> cd ~/SignalLab      # or wherever you put it
> xattr -cr .
> chmod +x start.command start.sh
> ./start.command
> ```
>
> `xattr -cr .` removes the untrusted-download tag from the folder. After that
> double-clicking works too. (The old right-click → **Open** trick no longer
> works on current macOS. The GUI route is System Settings → Privacy &
> Security → scroll to the bottom → **Open Anyway** — the Terminal command is
> quicker.)
>
> **🍎 macOS — "permission denied"?** The executable flag was lost in the ZIP:
> `chmod +x start.command start.sh`
>
> **🐧 Linux:** if double-clicking opens the file in a text editor instead of
> running it, right-click → Properties → tick *"Allow executing file as
> program"*, or run `chmod +x start.sh` once.

## 4 — Paste your OpenAI key into the pop-up

On the very first run a small setup window appears:

```
┌───────────────────────────────────────────┐
│  SignalLab Setup                          │
│  One-time setup. Your key is stored       │
│  locally in a file called .env.           │
│  ─────────────────────────────────────    │
│  OpenAI API key               (required)  │
│  [ ••••••••••••••••••••••••••••••••••• ]  │
│  ☐ Show key      [ Get a key from OpenAI ]│
│                                           │
│  Reasoning model                          │
│  [ gpt-4o-mini              ▾ ]           │
│                                           │
│  Embedding model                          │
│  [ text-embedding-3-small   ▾ ]           │
│                                           │
│              [ Save & Start ]  [ Cancel ] │
└───────────────────────────────────────────┘
```

1. Click **Get a key from OpenAI** — it opens
   [platform.openai.com/api-keys](https://platform.openai.com/api-keys) in your
   browser. Create a key and copy it. You'll need a payment method on the
   OpenAI account; a typical run costs **$0.10–$0.30** with `gpt-4o-mini`.
2. Paste it into the box. Tick **Show key** if you want to check it pasted.
3. Leave both model dropdowns alone — the defaults are the recommended ones.
4. Click **Save & Start**.

The window checks your key against OpenAI live and tells you straight away
whether it works, rather than letting you find out four minutes later. It then
saves everything to a local file called `.env` and setup continues on its own.
**You are never asked for the key again.**

> **No key is needed for the data itself.** NSE and BSE corporate filings are
> public and fetched directly.

> **Don't see the window?** It may be behind the terminal window, or on your
> other monitor.
>
> **On Linux** the pop-up needs Tk installed — `sudo apt install python3-tk`
> (Ubuntu/Debian) or `sudo dnf install python3-tkinter` (Fedora). Without it,
> SignalLab automatically falls back to asking for the key in the terminal
> instead. Nothing breaks either way.

## 5 — Wait for the first install

The start file downloads about 500MB of libraries. This takes **2–5 minutes
and happens only once** — every later start skips straight to launching.

## 6 — The dashboard opens

Your browser opens at **`http://localhost:8501`**. Jump to
[Run your first analysis](#️-run-your-first-analysis) below.

To stop: close the browser tab and press `Ctrl+C` in the terminal window (or
just close it).

**To start again:** double-click the same start file (Windows/Linux), or on
macOS run `cd ~/SignalLab && ./start.command`. That's it.

---

## What the start file actually does

No magic — it runs the manual steps for you, in order, and skips any that are
already done:

1. Finds Python and checks the version is supported
2. Opens the secrets pop-up if `.env` doesn't exist yet (terminal prompt if no display)
3. Creates the `.venv` virtual environment if it isn't there
4. Installs `requirements.txt` — and re-installs automatically if
   `requirements.txt` has changed since last time
5. Launches Streamlit and opens your browser

If any step fails it stops and prints a plain-English reason and a fix,
rather than a stack trace.

## The files involved

| File | What it is |
|---|---|
| `start.bat` | Windows double-click launcher — finds Python, calls `start.py` |
| `start.command` | macOS double-click launcher — same job |
| `start.sh` | Linux launcher — same job |
| `start.py` | The real logic: venv, dependencies, secrets check, launch. Cross-platform. |
| `setup_secrets.py` | The API-key pop-up (and its terminal fallback). Writes `.env`. Runs standalone too. |
| `.env` | Your saved keys. **Git-ignored — never commit or share it.** |
| `.env.example` | Template showing every setting, if you'd rather edit by hand |

## Changing your API key later

```bash
python setup_secrets.py           # re-open the pop-up, pre-filled
python setup_secrets.py --console # same, but in the terminal
python setup_secrets.py --show    # see current config (key masked)
```

Or double-click the start file with the `--configure` flag — see below.

Or edit `.env` directly — it's a plain text file with comments explaining
each setting.

## Start file options

For anything beyond a plain launch, pass a flag to the start file (or run
`start.py` directly):

```bash
python start.py                # dashboard (what double-clicking does)
python start.py --api          # FastAPI backend on :8000 instead
python start.py --configure    # re-open the key pop-up, then start
python start.py --console      # ask for the key in the terminal, no pop-up
python start.py --reinstall    # nuke .venv and rebuild from scratch
python start.py --setup-only   # install everything, don't launch
```

Windows: `start.bat --api` · macOS/Linux: `./start.sh --api`

---

# 🧰 Manual Setup

Everything below is what the start file automates. Use this if you want to
manage the environment yourself.

## Prerequisites

- Python **3.11** or **3.12** (3.10 and 3.13 also work)
- One API key (**OpenAI**) — document ingestion from NSE/BSE needs no key
- Outbound network access to `nseindia.com` and `bseindia.com` (both are public
  but geo/rate sensitive; a VPN inside India can help if requests are blocked
  from your network)
- ~500MB disk space for the virtual environment

## Step 1 — Clone the repo

```bash
git clone https://github.com/rrallapalli/SignalLab.git
cd SignalLab
```

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

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes 2–3 minutes the first time. This installs the `nse` and `bse`
packages used for direct exchange ingestion, alongside LangGraph, ChromaDB,
DuckDB, and the rest of the signal-synthesis stack.

## Step 4 — Get your API key

**OpenAI** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create an account, add a payment method, generate a key
- Recommended model: `gpt-4o-mini` (fast, cheap, and has a higher rate limit
  than `gpt-4o` — good for development and most production runs)

No key is needed for document ingestion — NSE and BSE corporate-announcement
filings (results, investor decks, concall transcripts, annual reports) are
fetched directly and are public.

## Step 5 — Configure environment

Easiest way, even here:

```bash
python setup_secrets.py
```

Or by hand:

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
```

Optional settings you can also set in `.env`:

```
OPENAI_MODEL=gpt-4o-mini    # or gpt-4o for higher quality
EMBED_MODEL=text-embedding-3-small
OPENAI_TEMPERATURE=0.0

# NSE/BSE fetch tuning
NSE_BSE_RESULT_LAG_DAYS=90      # how far past quarter-end to keep searching
NSE_BSE_ANNUAL_LAG_DAYS=200     # wider window when annual reports are requested
NSE_BSE_MAX_DOCS_PER_QUARTER=15
```

> **Note:** the signal agents call OpenAI with `response_format: json_object`
> (JSON mode) to guarantee syntactically valid structured output. Both
> `gpt-4o-mini` and `gpt-4o` support this — if you swap in a different model,
> confirm it supports JSON mode first.

## Step 6 — Sanity-check NSE/BSE connectivity

Before running the full pipeline (which costs OpenAI tokens), confirm your
machine can actually reach the exchanges and download a filing:

```bash
python diagnose_fetch.py --ticker TCS --company "Tata Consultancy Services"
```

This walks through five steps independently and tells you exactly where
things break, if they do:

1. NSE cookie handshake
2. NSE `announcements()` call
3. BSE scrip-code lookup (from the company name)
4. BSE `announcements()` call
5. PDF download + validity check

If steps 1–2 fail but 3–5 succeed, the pipeline will still run on BSE-only
data (and vice versa) — the pipeline degrades gracefully rather than failing
outright when one exchange is unreachable.

## Step 7 — Launch the dashboard

```bash
streamlit run ui/dashboard.py
```

This opens `http://localhost:8501` in your browser automatically.

---

## ▶️ Run your first analysis

In the dashboard sidebar:

1. Enter a **ticker** — the NSE trading symbol, e.g. `TCS`, `INFY`, `RELIANCE`, `HDFCBANK`
2. Enter the **company name** — e.g. `Tata Consultancy Services` (used to resolve the matching BSE scrip code)
3. Leave **Quarter** set to `Auto` — the system detects the latest completed quarter and automatically runs **Latest + QoQ + YoY** in one shot
4. Select a model — `gpt-4o-mini` is recommended for first runs
5. Click **🚀 Run Analysis**

The pipeline will:
- Pull corporate-announcement filings (results, investor presentations, concall transcripts, annual reports) directly from NSE and BSE for the relevant date windows
- Download and extract text from each PDF attachment
- Auto-detect the company's sector (Banking/NBFC, IT Services, Pharma, FMCG, Auto, Metals/Cement, Energy/Power, Telecom, Infrastructure, Real Estate, or General) to select the right theme taxonomy
- Chunk and embed all documents into ChromaDB
- Run four signal agents (Confidence, Narrative, Guidance, Risk) across three quarters
- Save all results to a local DuckDB database
- Display the full dashboard automatically

**Good tickers to start with:** `TCS`, `INFY`, `RELIANCE`, `HDFCBANK`, `ICICIBANK`

---

## Running a specific quarter

If you want to anchor the analysis to a specific quarter rather than the auto-detected latest:

1. Change the **Quarter** dropdown from `Auto` to e.g. `Q2`
2. Set the **Year** to e.g. `2025`
3. Click **Run Analysis**

The system will then run signals for Q2 2025 (Latest), Q1 2025 (QoQ), and Q2 2024 (YoY).

---

## Loading previously analysed tickers

Once you have run a ticker, it appears in the **Stored Tickers → Load**
dropdown at the top of the sidebar. Select it to reload the full dashboard
from the local database without re-running the pipeline.

---

## Diagnosing missing signals

If sections appear blank or show "No data" for some periods, run the
diagnostic tool:

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

Any ❌ means that signal was not stored — re-run the pipeline for that ticker
to fill the gap. Check `data/signal_agent.log` for the specific error (a
common one: the LLM occasionally returns a stray `null` for a numeric field
or malformed JSON — both are handled with null-safe coercion and a JSON
repair fallback, but check the log if a signal still comes back empty).

---

## Resetting the database

If you want a completely clean start:

```bash
rm data/signals.duckdb      # removes all stored signals
rm -rf data/chroma/         # removes all vector embeddings
rm -rf data/nse_cache/ data/bse_cache/   # removes NSE/BSE session cache
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

> **Known gap:** `main.py run` (CLI) and the `/run` API endpoint both import
> a `run_pipeline` function that doesn't currently exist in
> `agents/orchestrator.py` (only `run_comparison_pipeline` does), so both
> will fail on import today. This predates the NSE/BSE ingestion work and is
> unrelated to it. The Streamlit dashboard calls the correct function and is
> the supported entry point until this is fixed.

---

## Troubleshooting

**macOS: "Apple could not verify ‘start.command’ is free of malware"**
Click **Done** (never *Move to Bin*), then `cd` to the folder and run
`xattr -cr .` — this clears the untrusted-download tag macOS applies to files
from a browser-downloaded ZIP. Cloning with `git clone` avoids this entirely.

**macOS: `permission denied: ./start.command`**
The executable bit didn't survive the download: `chmod +x start.command start.sh`

**macOS: install fails, or files show a ☁️ cloud icon in Finder**
The folder is inside iCloud-synced `Documents`/`Desktop`, and SignalLab needs to
write ~500MB locally. Move it out: `mv ~/Documents/SignalLab ~/SignalLab`

**macOS: `(base)` in your prompt / Anaconda Python**
The launcher will build `.venv` from conda's Python, which normally works. If
the install misbehaves, run `conda deactivate` and start again to use system
Python instead.


**`ModuleNotFoundError`**
Make sure your virtual environment is active: `source .venv/bin/activate`,
and that you've re-run `pip install -r requirements.txt` after pulling any
update to `requirements.txt` (e.g. the `nse`/`bse` packages).

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
- Confirm the ticker is a real NSE symbol and that this machine can reach `nseindia.com` and `bseindia.com` (some cloud/CI networks block these) — run `python diagnose_fetch.py --ticker <TICKER> --company "<Company Name>"` to isolate the failure
- Try switching to `gpt-4o-mini` in the model selector — it has higher rate limits
- Check `data/signal_agent.log` for detailed error messages

**No documents found for a ticker**
- Double check the NSE symbol spelling and that the company name closely matches its official listed name (used to resolve the BSE scrip code)
- Smaller/thinly-covered companies may not publish concall transcripts as a formal filing — investor presentations and financial-results filings are usually still available
- Widen `NSE_BSE_RESULT_LAG_DAYS` in `.env` if results were declared later than usual that quarter

**BSE scrip code resolved to the wrong company**
Auto-lookup is a fuzzy match on company name. Call `fetch_documents(...,
bse_scripcode="500209")` with the correct code (found on bseindia.com)
if you need to override it — this isn't yet wired into the dashboard UI as
a manual field.

**Sub-dimension scores showing `—`**
The data exists but sub-dimensions were not scored in that run. Re-run the
pipeline for that ticker to populate all six confidence sub-dimensions.

**Rate limit errors**
The pipeline staggers LLM calls and retries automatically with exponential
backoff. If you consistently hit limits, set `OPENAI_MODEL=gpt-4o-mini` in
your `.env` — it has a much higher rate limit than `gpt-4o`.

**A signal fails with a JSON-related error**
This should be rare now that agents use OpenAI's JSON mode, but if you see
it: check `data/signal_agent.log` for the "JSON parse failed... Attempting
repair" line and what follows it. The repair step handles trailing commas,
smart quotes, and stray control characters; if it still fails, the raw
response head is logged for debugging.
