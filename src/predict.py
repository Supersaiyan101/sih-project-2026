"""SIH26017 prediction engine.

Loads the trained models + encoders and produces the prediction contract for new
parcels. Used by the FastAPI (bonus demo) and Streamlit (main path) via direct import.

Contract fields (see INTERFACES.md):
  risk_score, risk_level, expected_overrun_days, max_delay_prob, stages[5],
  top_factors, recommended_actions, overrun_while_ongoing_days.

Important distinction (do NOT conflate):
  - expected_overrun_days      = PREDICTED total future overrun (sum of regressor stage
                                 overruns, clamped >= 0). Comes from the models.
  - overrun_while_ongoing_days = MEASURED current overrun (elapsed_days - statutory_days)
                                 for the parcel's ongoing stage. Comes from the live
                                 timeline, never the regressor.

Usage:
  .venv/bin/python src/predict.py --refresh-portfolio
  .venv/bin/python src/predict.py --json parcel.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from actions import recommend_actions
from features import DATA, MODELS, STAGES, compute_risk

STATUTORY = {"SIA": 180, "NOTIFICATION": 60, "DECLARATION": 365, "AWARD": 365, "POSSESSION": 90}
TAU = 250.0  # days scale for risk_score (1 - exp(-overrun/TAU))

PROJECT_FEATURE_COLS = [
    "project_id", "project_type", "affected_families", "compensation_status",
    "rehab_progress_pct", "stakeholder_responsiveness", "historical_performance_score",
]

# safe defaults so a partial feature dict still scores
DEFAULTS = {
    "project_type": "road", "compensation_status": "paid", "land_class": "agri",
    "affected_families": 0, "rehab_progress_pct": 100.0, "stakeholder_responsiveness": 1.0,
    "historical_performance_score": 1.0, "owner_count": 1, "area_sqm": 1000.0,
    "pending_mutations": 0, "court_stay": 0, "encumbrances": 0,
}

_ARTIFACTS: dict | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_artifacts(force: bool = False) -> dict:
    global _ARTIFACTS
    if _ARTIFACTS is None or force:
        encoders = joblib.load(MODELS / "encoders.joblib")
        spec = json.loads((MODELS / "feature_columns.json").read_text())["feature_columns"]
        feature_columns = [s["name"] for s in spec]
        categoricals = [s["name"] for s in spec if s["type"] == "categorical"]
        models = {}
        for stage in STAGES:
            models[stage] = {
                "classifier": joblib.load(MODELS / f"{stage}_classifier.joblib"),
                "regressor": joblib.load(MODELS / f"{stage}_regressor.joblib"),
            }
        _ARTIFACTS = {
            "encoders": encoders,
            "feature_columns": feature_columns,
            "categoricals": categoricals,
            "models": models,
            "explainers": {},
        }
    return _ARTIFACTS


def _encode(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X = df[artifacts["feature_columns"]].copy()
    for col in artifacts["categoricals"]:
        enc = artifacts["encoders"][col]
        X[col] = enc.transform(X[[col]])
    numeric = [c for c in artifacts["feature_columns"] if c not in artifacts["categoricals"]]
    X[numeric] = X[numeric].astype(float)
    return X


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def risk_level(score: float) -> str:
    if score > 0.70:
        return "RED"
    if score >= 0.40:
        return "YELLOW"
    return "GREEN"


def score_batch(df: pd.DataFrame, artifacts: dict | None = None,
                include_stages: bool = False) -> pd.DataFrame:
    """Score many parcels (fast path: no SHAP). df must carry the raw feature columns.

    include_stages=True adds per-stage {STAGE}_prob and {STAGE}_overrun columns.
    """
    if artifacts is None:
        artifacts = load_artifacts()
    X = _encode(df, artifacts)
    probs = {s: artifacts["models"][s]["classifier"].predict_proba(X)[:, 1] for s in STAGES}
    overruns = {s: artifacts["models"][s]["regressor"].predict(X) for s in STAGES}

    sev = sum(np.clip(overruns[s], 0, None) for s in STAGES)
    risk = 1.0 - np.exp(-sev / TAU)
    max_prob = np.maximum.reduce([probs[s] for s in STAGES])

    out = pd.DataFrame({
        "parcel_id": df["parcel_id"].values,
        "risk_score": np.round(risk, 4),
        "risk_level": np.select([risk > 0.70, risk >= 0.40], ["RED", "YELLOW"], "GREEN"),
        "expected_overrun_days": np.round(sev, 1),
        "max_delay_prob": np.round(max_prob, 4),
    })
    if include_stages:
        for s in STAGES:
            out[f"{s}_prob"] = np.round(probs[s], 4)
            out[f"{s}_overrun"] = np.round(overruns[s], 1)
    return out


def _global_shap(X_row: pd.DataFrame, artifacts: dict) -> dict:
    """SHAP contributions for a single parcel, aggregated across the 5 stage models."""
    import shap
    contribs = {c: 0.0 for c in X_row.columns}
    per_stage = {}
    for stage in STAGES:
        if stage not in artifacts["explainers"]:
            artifacts["explainers"][stage] = shap.TreeExplainer(
                artifacts["models"][stage]["classifier"])
        sv = artifacts["explainers"][stage].shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        vals = np.asarray(sv)[0]
        per_stage[stage] = [[c, float(vals[i])] for i, c in enumerate(X_row.columns)]
        for i, c in enumerate(X_row.columns):
            contribs[c] += abs(float(vals[i]))
    top = sorted(contribs.items(), key=lambda kv: -kv[1])[:8]
    return {
        "top_factors": [[k, round(v, 4)] for k, v in top],
        "per_stage": {s: sorted(v, key=lambda x: -abs(x[1]))[:3] for s, v in per_stage.items()},
    }


def score_parcel(features: dict, artifacts: dict | None = None,
                 timeline: dict | None = None, parcel_id: str | None = None) -> dict:
    """Full contract for a single parcel (with SHAP + actions)."""
    if artifacts is None:
        artifacts = load_artifacts()
    row = {c: features.get(c, DEFAULTS[c]) for c in artifacts["feature_columns"]}
    X = _encode(pd.DataFrame([row]), artifacts)

    probs = {s: float(artifacts["models"][s]["classifier"].predict_proba(X)[0, 1]) for s in STAGES}
    overruns = {s: float(artifacts["models"][s]["regressor"].predict(X)[0]) for s in STAGES}
    risk = compute_risk(probs, overruns)
    shap = _global_shap(X, artifacts)
    actions = recommend_actions(row)

    stages = {}
    overrun_ongoing = None
    for s in STAGES:
        st = {
            "delay_prob": round(probs[s], 4),
            "expected_overrun": round(overruns[s], 1),
            "statutory_days": STATUTORY[s],
            "status": None, "elapsed_days": None, "actual_days": None,
            "top_factors": shap["per_stage"][s],
        }
        if timeline and s in timeline:
            st.update(timeline[s])
            if timeline[s].get("status") == "ongoing" and timeline[s].get("elapsed_days") is not None:
                overrun_ongoing = float(timeline[s]["elapsed_days"] - STATUTORY[s])
        stages[s] = st

    return {
        "parcel_id": parcel_id,
        "risk_score": round(risk["risk_score"], 4),
        "risk_level": risk_level(risk["risk_score"]),
        "expected_overrun_days": round(risk["expected_overrun_days"], 1),
        "max_delay_prob": round(risk["max_delay_prob"], 4),
        "stages": stages,
        "top_factors": shap["top_factors"],
        "recommended_actions": actions,
        "overrun_while_ongoing_days": overrun_ongoing,
    }


# --------------------------------------------------------------------------- #
# Portfolio cache
# --------------------------------------------------------------------------- #

def refresh_portfolio(output_path: Path | None = None) -> pd.DataFrame:
    """Score all live parcels and persist the portfolio cache (geo + features + stages)."""
    artifacts = load_artifacts()
    parcels = pd.read_parquet(DATA / "parcels.parquet")
    live = parcels[parcels["is_live"] == 1].copy()
    projects = pd.read_parquet(DATA / "projects.parquet")[PROJECT_FEATURE_COLS + ["spatial_type"]]
    villages = pd.read_parquet(DATA / "villages.parquet")[["village", "lat", "lon"]]
    timelines = pd.read_parquet(DATA / "stage_timelines_live.parquet")

    live = live.merge(projects, on="project_id", how="left")
    scores = score_batch(live, artifacts, include_stages=True)

    # keep raw features + geo (incl. codes for cascading filter) alongside scores
    feat_cols = artifacts["feature_columns"]
    geo_cols = ["parcel_id", "project_id", "village", "village_code", "tehsil",
                "district", "district_code", "state", "state_code", "spatial_type"]
    geo = live[geo_cols + feat_cols].copy()
    out = scores.merge(geo, on="parcel_id", how="left")
    out = out.merge(villages, on="village", how="left")

    # measured overrun-while-ongoing from the live timeline (NOT the regressor)
    ongoing = timelines[timelines["status"] == "ongoing"][
        ["parcel_id", "stage", "elapsed_days", "statutory_days"]].copy()
    ongoing["overrun_while_ongoing_days"] = ongoing["elapsed_days"] - ongoing["statutory_days"]
    ongoing = ongoing.rename(columns={"stage": "current_stage"})
    out = out.merge(ongoing[["parcel_id", "current_stage", "overrun_while_ongoing_days"]],
                    on="parcel_id", how="left")

    output = output_path or DATA / "portfolio_scores.parquet"
    out.to_parquet(output, index=False)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parcel_to_features(parcel_id: str) -> dict:
    parcels = pd.read_parquet(DATA / "parcels.parquet")
    projects = pd.read_parquet(DATA / "projects.parquet")[PROJECT_FEATURE_COLS]
    p = parcels[parcels["parcel_id"] == parcel_id]
    if p.empty:
        raise SystemExit(f"parcel_id {parcel_id} not found")
    rec = p.merge(projects, on="project_id", how="left").iloc[0]
    features = {c: rec[c] for c in DEFAULTS}
    return features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-portfolio", action="store_true", help="score all live parcels -> cache")
    ap.add_argument("--json", type=str, help="path to JSON file with raw features")
    ap.add_argument("--parcel-id", type=str, help="score a live parcel by id")
    args = ap.parse_args()

    if args.refresh_portfolio:
        import time
        t0 = time.time()
        out = refresh_portfolio()
        print(f"portfolio refresh: {len(out)} live parcels in {time.time() - t0:.2f}s")
        print(out[["parcel_id", "risk_score", "risk_level", "expected_overrun_days"]].head().to_string(index=False))
        return

    if args.json:
        features = json.loads(Path(args.json).read_text())
        contract = score_parcel(features, parcel_id=features.get("parcel_id"))
        print(json.dumps(contract, indent=2))
        return

    if args.parcel_id:
        features = _parcel_to_features(args.parcel_id)
        contract = score_parcel(features, parcel_id=args.parcel_id)
        print(json.dumps(contract, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
