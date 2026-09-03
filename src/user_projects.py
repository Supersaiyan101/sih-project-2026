"""SIH26017 user-created project persistence (parquet-only, no DB).

Officials create new projects and tag land parcels (via CSV of semantic parcel IDs).
This module persists those projects + parcels so they survive dashboard refresh and
appear in the Portfolio tagged "user-created".

Stores files under data/generated/user/:
  - user_projects.parquet        (project-level record)
  - user_parcels.parquet         (scored parcels, schema-matched to portfolio_scores.parquet)
  - project_state_overrides.parquet  (lifecycle state updates: compensation/rehab)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from states import PROJECT_TYPE_CODES, STATES

DATA = Path(__file__).resolve().parent.parent / "data" / "generated"
USER_DIR = DATA / "user"
USER_PROJECTS_PATH = USER_DIR / "user_projects.parquet"
USER_PARCELS_PATH = USER_DIR / "user_parcels.parquet"
OVERRIDES_PATH = USER_DIR / "project_state_overrides.parquet"

# column order must match portfolio_scores.parquet so concat works
PORTFOLIO_COLS = [
    "parcel_id", "risk_score", "risk_level", "expected_overrun_days", "max_delay_prob",
    "SIA_prob", "SIA_overrun", "NOTIFICATION_prob", "NOTIFICATION_overrun",
    "DECLARATION_prob", "DECLARATION_overrun", "AWARD_prob", "AWARD_overrun",
    "POSSESSION_prob", "POSSESSION_overrun", "project_id", "village", "village_code",
    "tehsil", "district", "district_code", "state", "state_code", "spatial_type",
    "project_type", "compensation_status", "land_class", "affected_families",
    "rehab_progress_pct", "stakeholder_responsiveness", "historical_performance_score",
    "owner_count", "area_sqm", "pending_mutations", "court_stay", "encumbrances",
    "lat", "lon", "current_stage", "overrun_while_ongoing_days",
]

PROJECT_COLS = [
    "project_id", "project_type", "spatial_type", "coord_path", "affected_families",
    "compensation_status", "rehab_progress_pct", "stakeholder_responsiveness",
    "historical_performance_score", "state", "state_code", "district", "tehsil",
]


def load_user_projects() -> pd.DataFrame:
    if USER_PROJECTS_PATH.exists():
        return pd.read_parquet(USER_PROJECTS_PATH)
    return pd.DataFrame(columns=PROJECT_COLS)


def load_user_parcels() -> pd.DataFrame:
    if USER_PARCELS_PATH.exists():
        return pd.read_parquet(USER_PARCELS_PATH)
    return pd.DataFrame(columns=PORTFOLIO_COLS)


def reset_user_data() -> None:
    for p in (USER_PROJECTS_PATH, USER_PARCELS_PATH, OVERRIDES_PATH):
        if p.exists():
            p.unlink()


def load_overrides() -> pd.DataFrame:
    """Lifecycle state overrides per project (compensation/rehab), applied on portfolio load."""
    if OVERRIDES_PATH.exists():
        return pd.read_parquet(OVERRIDES_PATH)
    return pd.DataFrame(columns=["project_id", "compensation_status", "rehab_progress_pct"])


def save_override(project_id: str, compensation_status: str, rehab_progress_pct: float) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    ov = load_overrides()
    ov = ov[ov["project_id"] != project_id]
    ov = pd.concat([ov, pd.DataFrame([{"project_id": project_id,
                                       "compensation_status": compensation_status,
                                       "rehab_progress_pct": float(rehab_progress_pct)}])],
                   ignore_index=True)
    ov.to_parquet(OVERRIDES_PATH, index=False)


def derive_institutional_profile(district: str) -> dict:
    """Institutional prior for a NEW project in `district`: median responsiveness + track
    record of existing projects there (falls back to state, then all). Not user-editable —
    these are derived from the district's observed performance."""
    projects = pd.read_parquet(DATA / "projects.parquet")
    sub = projects[projects["district"] == district]
    if len(sub) < 3:
        if len(sub):
            state = sub["state"].mode()[0]
            sub = projects[projects["state"] == state]
        else:
            sub = projects
    return {
        "stakeholder_responsiveness": round(float(sub["stakeholder_responsiveness"].median()), 3),
        "historical_performance_score": round(float(sub["historical_performance_score"].median()), 3),
    }


def pull_records(ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Pull existing parcel records for the given semantic IDs. Returns (found, missing)."""
    parcels = pd.read_parquet(DATA / "parcels.parquet")
    found = parcels[parcels["parcel_id"].isin(ids)]
    found_ids = set(found["parcel_id"])
    missing = [i for i in ids if i not in found_ids]
    return found, missing


def generate_user_project_id(state_code: str, project_type: str) -> str:
    existing = load_user_projects()
    seq = len(existing) + 1
    type_code = PROJECT_TYPE_CODES.get(project_type, "RDH")
    return f"{state_code}-{type_code}-{seq:04d}"


def state_code_for(state: str) -> str:
    return STATES[state]["code"]


def persist_user(project_row: dict, parcel_rows: pd.DataFrame) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    proj = load_user_projects()
    proj = pd.concat([proj, pd.DataFrame([project_row])], ignore_index=True)
    proj.to_parquet(USER_PROJECTS_PATH, index=False)

    parcels = load_user_parcels()
    parcels = pd.concat([parcels, parcel_rows[PORTFOLIO_COLS]], ignore_index=True)
    parcels.to_parquet(USER_PARCELS_PATH, index=False)
