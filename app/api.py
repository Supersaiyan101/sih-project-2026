"""SIH26017 FastAPI endpoint (bonus demo of deliverable #11).

Wraps src/predict.py. Run:
  .venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000

The Streamlit dashboard imports predict.py directly (offline, one process) — this API
is a standalone demonstration of the integration surface for NIC/government systems.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# allow `from predict import ...` (src) and `from landing import ...` (app)
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR.parent / "src"))

from landing import LANDING_HTML

from predict import (
    DEFAULTS,
    load_artifacts,
    score_batch,
    score_parcel,
)

app = FastAPI(title="BhoomiSetu — Land Acquisition Delay Predictor", version="2.0")


@app.get("/", response_class=HTMLResponse)
def landing():
    """BhoomiSetu product landing page (front door -> Streamlit dashboard on :8501)."""
    return LANDING_HTML

# load once at startup
load_artifacts()


class ParcelFeatures(BaseModel):
    parcel_id: str | None = None
    project_type: str = "road"
    compensation_status: str = "paid"
    land_class: str = "agri"
    affected_families: float = 0
    rehab_progress_pct: float = 100.0
    stakeholder_responsiveness: float = 1.0
    historical_performance_score: float = 1.0
    owner_count: float = 1
    area_sqm: float = 1000.0
    pending_mutations: float = 0
    court_stay: float = 0
    encumbrances: float = 0


class BatchRequest(BaseModel):
    parcels: list[ParcelFeatures]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": load_artifacts() is not None}


@app.post("/predict")
def predict(p: ParcelFeatures):
    features = p.model_dump()
    pid = features.pop("parcel_id", None)
    return score_parcel(features, parcel_id=pid)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    import pandas as pd
    rows = []
    for p in req.parcels:
        d = p.model_dump()
        d.pop("parcel_id", None)
        rows.append({k: d.get(k, DEFAULTS[k]) for k in DEFAULTS})
    df = pd.DataFrame(rows)
    df["parcel_id"] = [p.parcel_id or f"REQ_{i}" for i, p in enumerate(req.parcels)]
    results = score_batch(df)
    return {"results": results.to_dict(orient="records")}
