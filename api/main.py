"""api/main.py – FastAPI backend exposing signal pipeline and history."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import run_pipeline
from store.signal_store import SignalStore
from store.vector_store import VectorStore

app = FastAPI(title="Signal Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_vs: VectorStore | None = None
_ss: SignalStore | None = None

def _get_stores():
    global _vs, _ss
    if _vs is None: _vs = VectorStore()
    if _ss is None: _ss = SignalStore()
    return _vs, _ss


class RunRequest(BaseModel):
    ticker:      str
    company:     str
    quarter:     str       # "Q2"
    fiscal_year: int       # 2024


@app.post("/run")
async def run_signals(req: RunRequest) -> dict[str, Any]:
    """Run the full RAG signal pipeline for a ticker + quarter."""
    vs, ss = _get_stores()
    try:
        bundle = await run_pipeline(
            ticker=req.ticker.upper(), company=req.company,
            quarter=req.quarter, fiscal_year=req.fiscal_year,
            vs=vs, ss=ss,
        )
        return bundle.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{ticker}/confidence")
def confidence_history(ticker: str, limit: int = 8) -> list[dict]:
    _, ss = _get_stores()
    return ss.get_confidence_history(ticker.upper(), limit)


@app.get("/signals/{ticker}/narrative")
def narrative_history(ticker: str, limit: int = 8) -> list[dict]:
    _, ss = _get_stores()
    return ss.get_narrative_history(ticker.upper(), limit)


@app.get("/signals/{ticker}/guidance")
def guidance_history(ticker: str, limit: int = 8) -> list[dict]:
    _, ss = _get_stores()
    return ss.get_guidance_history(ticker.upper(), limit)


@app.get("/signals/{ticker}/risk")
def risk_history(ticker: str, limit: int = 8) -> list[dict]:
    _, ss = _get_stores()
    return ss.get_risk_history(ticker.upper(), limit)


@app.get("/tickers")
def list_tickers() -> list[str]:
    _, ss = _get_stores()
    return ss.get_all_tickers()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
