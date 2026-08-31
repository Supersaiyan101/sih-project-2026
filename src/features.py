"""SIH26017 feature engineering.

Builds a geo-free feature matrix (parcel-level) and per-stage targets from the
generated data. Models NEVER see district/state/tehsil/village names or ids.

Locked decisions (PROJECT_CONTEXT.md):
  - parcel is the atomic unit; X is parcel-level (one row per parcel).
  - 12 features: 3 categorical (ordinal-encoded) + 9 numeric.
  - per-stage targets: delay_flag (binary) + delay_days (regression).
  - risk score = max per-stage delay probability; severity = sum of expected overruns.
  - rollups (project/village/district) are area-weighted.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "generated"
MODELS = ROOT / "models"

STAGES = ["SIA", "NOTIFICATION", "DECLARATION", "AWARD", "POSSESSION"]

CATEGORICAL_FEATURES = ["project_type", "compensation_status", "land_class"]
NUMERIC_FEATURES = [
    "affected_families",
    "rehab_progress_pct",
    "stakeholder_responsiveness",
    "historical_performance_score",
    "owner_count",
    "area_sqm",
    "pending_mutations",
    "court_stay",
    "encumbrances",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# project-level feature columns (for merging); project geo cols are dropped (parcel geo is used)
PROJECT_FEATURE_COLS = [
    "project_id",
    "project_type",
    "affected_families",
    "compensation_status",
    "rehab_progress_pct",
    "stakeholder_responsiveness",
    "historical_performance_score",
]

# parcel geo/id columns used ONLY for rollup/validation (never in X)
META_COLUMNS = ["parcel_id", "project_id", "village", "tehsil", "district", "state", "area_sqm"]


def load_training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    parcels = pd.read_parquet(DATA / "parcels.parquet")
    projects = pd.read_parquet(DATA / "projects.parquet")
    return parcels, projects


def build_features(parcels: pd.DataFrame, projects: pd.DataFrame,
                   fit_encoder: bool = True, encoder_path: Path | None = None
                   ) -> dict:
    """Return {X, y, meta, encoders, feature_columns}.

    X:      parcel-level encoded feature matrix (index = parcel_id)
    y:      parcel-level targets, columns {stage}_delay_flag / {stage}_delay_days
    meta:   parcel-level geo/identifier columns for rollup + validation
    """
    # historical (completed) parcels only -> training set
    parcels = parcels[parcels["is_live"] == 0].copy()

    # merge project-level features down to each parcel (no geo cols from projects)
    projects_feat = projects[PROJECT_FEATURE_COLS].copy()
    merged = parcels.merge(projects_feat, on="project_id", how="left")

    X_raw = merged[FEATURE_COLUMNS].copy()
    meta = merged[META_COLUMNS].copy()
    meta = meta.set_index("parcel_id")

    # ordinal-encode categoricals (robust to unknown categories at predict time)
    if fit_encoder:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(X_raw[[col]])
            encoders[col] = enc
            X_raw[col] = enc.transform(X_raw[[col]])
    else:
        encoders = joblib.load(encoder_path)
        for col in CATEGORICAL_FEATURES:
            enc = encoders[col]
            X_raw[col] = enc.transform(X_raw[[col]])

    X = X_raw.astype({c: float for c in NUMERIC_FEATURES})
    X = X.set_index(meta.index)

    # per-stage targets from long-format timelines
    timelines = pd.read_parquet(DATA / "stage_timelines_historical.parquet")
    y = _build_targets(timelines)

    # align X/y/meta to the same parcel set
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    meta = meta.loc[common]

    feature_columns = list(X.columns)
    return {
        "X": X,
        "y": y,
        "meta": meta,
        "encoders": encoders,
        "feature_columns": feature_columns,
    }


def _build_targets(timelines: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for stage in STAGES:
        sub = timelines[timelines["stage"] == stage].set_index("parcel_id")
        cols[f"{stage}_delay_flag"] = sub["delay_flag"]
        cols[f"{stage}_delay_days"] = sub["delay_days"]
    return pd.DataFrame(cols)


def save_encoders(encoders: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, path)


def save_feature_columns(feature_columns: list[str], encoders: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = []
    for col in feature_columns:
        if col in encoders:
            spec.append({
                "name": col,
                "type": "categorical",
                "categories": [str(c) for c in encoders[col].categories_[0]],
            })
        else:
            spec.append({"name": col, "type": "numeric"})
    path.write_text(json.dumps({"feature_columns": spec}, indent=2))


def compute_risk(probs: dict[str, float], overruns: dict[str, float]) -> dict:
    """Compose parcel risk score + severity.

    Locked intent: "max per-stage delay probability + overrun severity". The max-prob
    alone saturates (delay is near-universal), so the headline risk_score is
    SEVERITY-based (exponential saturation of total expected overrun days), which
    discriminates parcels well. max_delay_prob is retained for the per-stage bars.
    """
    probs = {s: float(probs.get(s, 0.0)) for s in STAGES}
    overruns = {s: float(overruns.get(s, 0.0)) for s in STAGES}
    max_delay_prob = max(probs.values()) if probs else 0.0
    total_overrun = sum(max(0.0, v) for v in overruns.values())
    TAU = 250.0  # days at which risk_score ~ 0.63 (smooth 0..1 scale)
    risk_score = float(1.0 - np.exp(-total_overrun / TAU))
    return {
        "risk_score": risk_score,
        "expected_overrun_days": total_overrun,
        "max_delay_prob": max_delay_prob,
    }


def rollup_risk(parcel_scores: pd.DataFrame, meta: pd.DataFrame,
                level: str) -> pd.DataFrame:
    """Area-weighted rollup of per-parcel risk to project/village/district/state.

    parcel_scores: DataFrame indexed by parcel_id with a 'risk_score' column.
    meta:          DataFrame indexed by parcel_id with geo cols + 'area_sqm'.
    level:         one of {project_id, village, tehsil, district, state}.
    """
    df = parcel_scores.join(meta[[level, "area_sqm"]], how="left")
    if df["area_sqm"].sum() == 0:
        df["area_sqm"] = 1.0
    df["_w"] = df["area_sqm"].fillna(0.0)
    num = df.groupby(level).apply(lambda g: (g["risk_score"] * g["_w"]).sum(),
                                  include_groups=False)
    den = df.groupby(level)["_w"].sum()
    out = (num / den.replace(0, 1)).rename("risk_score").reset_index()
    out["n_parcels"] = df.groupby(level).size().values
    return out
