"""Day-1 / Stage-1 sanity checks for generated data (v2 multi-state).

Checks:
  1. Volumes + null pattern.
  2. Semantic ID format correctness (parcel + project).
  3. Spatial validity: linear spans >1 district (some >1 state); point stays in 1 district.
  4. Per-state volumes are reasonable.
  5. Delay rules recoverable (correlation factor -> delay).
  6. Hidden confound: no visible column leaks district/state identity; institutional
     features are a MODERATE proxy (partial, not perfect).

Run:  .venv/bin/python src/sanity_check.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

GEN = Path(__file__).resolve().parent.parent / "data" / "generated"

PARCEL_ID_RE = re.compile(r"^(HP|PB|UK)-[A-Z]{3}-\d{4}-\d{4}$")
PROJECT_ID_RE = re.compile(r"^(HP|PB|UK)-(RDH|RLY|IRR|DAM|IND)-\d{4}$")


def load():
    return (pd.read_parquet(GEN / "projects.parquet"),
            pd.read_parquet(GEN / "parcels.parquet"),
            pd.read_parquet(GEN / "villages.parquet"),
            pd.read_parquet(GEN / "districts.parquet"),
            pd.read_parquet(GEN / "states.parquet"),
            pd.read_parquet(GEN / "stage_timelines_historical.parquet"),
            pd.read_parquet(GEN / "stage_timelines_live.parquet"))


def check_volumes(projects, parcels, villages, states, hist, live):
    print("=" * 60); print("[1] VOLUMES + NULLS"); print("=" * 60)
    print(f"states: {len(states)}  (expect 3)")
    print(f"districts: {villages['district'].nunique()}")
    print(f"projects: {len(projects)}")
    print(f"parcels : {len(parcels)}  (live={int(parcels['is_live'].sum())})")
    print(f"villages: {len(villages)}")
    print(f"historical rows: {len(hist)}  (expect 5 x parcels)")
    print(f"live rows      : {len(live)}")
    assert len(states) == 3
    assert len(hist) == 5 * len(parcels)
    assert (hist["status"] == "completed").all()
    assert live.loc[live["status"] == "pending", "actual_days"].isna().all()
    assert live.loc[live["status"] == "ongoing", "actual_days"].isna().all()
    assert live.loc[live["status"] == "ongoing", "elapsed_days"].notna().all()
    print("  -> null/status pattern OK")


def check_ids(projects, parcels):
    print("=" * 60); print("[2] SEMANTIC ID FORMAT"); print("=" * 60)
    ok_p = bool(parcels["parcel_id"].str.match(PARCEL_ID_RE).all())
    ok_r = bool(projects["project_id"].str.match(PROJECT_ID_RE).all())
    print(f"parcel IDs valid: {ok_p}   project IDs valid: {ok_r}")
    # consistency: ID components match the parcel's assigned geography
    st = parcels["state_code"] == parcels["parcel_id"].str[:2]
    dc = parcels["district_code"] == parcels["parcel_id"].str.split("-").str[1]
    vc = parcels["village_code"] == parcels["parcel_id"].str.split("-").str[2]
    print(f"ID<->state consistent: {bool(st.all())}  ID<->district: {bool(dc.all())}  ID<->village: {bool(vc.all())}")
    assert ok_p and ok_r and st.all() and dc.all() and vc.all()


def check_spatial(projects, parcels):
    print("=" * 60); print("[3] SPATIAL VALIDITY"); print("=" * 60)
    lin = projects[projects["spatial_type"] == "linear"]
    pt = projects[projects["spatial_type"] == "point"]
    lin_multi_district = lin_multi_state = 0
    for _, p in lin.iterrows():
        sub = parcels[parcels["project_id"] == p["project_id"]]
        if sub["district_code"].nunique() > 1:
            lin_multi_district += 1
        if sub["state_code"].nunique() > 1:
            lin_multi_state += 1
    bad_point = 0
    for _, p in pt.iterrows():
        if parcels[parcels["project_id"] == p["project_id"]]["district_code"].nunique() > 1:
            bad_point += 1
    print(f"linear total: {len(lin)}  | span>1 district: {lin_multi_district}  | span>1 state: {lin_multi_state}")
    print(f"point total: {len(pt)}  | point crossing districts: {bad_point}")
    assert len(lin) > 0
    assert lin_multi_district == len(lin), "every linear project must span >1 district"
    assert lin_multi_state > 0, "need at least one cross-state linear project"
    assert bad_point == 0, "point projects must stay in 1 district"


def check_state_volumes(parcels):
    print("=" * 60); print("[4] PER-STATE VOLUMES"); print("=" * 60)
    vc = parcels["state_code"].value_counts()
    print(vc.to_string())
    frac = vc / len(parcels)
    assert (frac > 0.10).all(), "each state needs a reasonable parcel share"


def check_rules(parcels, hist):
    print("=" * 60); print("[5] RULE RECOVERY"); print("=" * 60)
    aw = hist[hist["stage"] == "AWARD"].set_index("parcel_id")["delay_days"].rename("award_delay")
    po = hist[hist["stage"] == "POSSESSION"].set_index("parcel_id")["delay_days"].rename("poss_delay")
    de = hist[hist["stage"] == "DECLARATION"].set_index("parcel_id")["delay_days"].rename("decl_delay")
    df = parcels.set_index("parcel_id").join([aw, po, de])
    for cond, label, col in [
        (df["court_stay"] == 1, "court_stay=1", "award_delay"),
        (df["land_class"] == "orchard", "orchard", "award_delay"),
        (df["owner_count"] > 4, "owner_count>4", "decl_delay"),
        (df["pending_mutations"] >= 2, "pending_mutations>=2", "decl_delay"),
    ]:
        print(f"  {label:24s} {col:12s} {df.loc[cond, col].mean():8.1f} vs base {df.loc[~cond, col].mean():8.1f}")


def check_confound(projects, parcels):
    print("=" * 60); print("[6] HIDDEN CONFOUND (leak + proxy band)"); print("=" * 60)
    feats = ["owner_count", "pending_mutations", "court_stay", "encumbrances"]
    d_dummies = pd.get_dummies(parcels["district_code"]).astype(float)
    worst = 0.0
    for f in feats:
        c = d_dummies.corrwith(parcels[f]).abs().max()
        worst = max(worst, c)
    print(f"  worst parcel-feature <-> district corr: {worst:.3f}")
    assert worst < 0.5, "parcel feature leaks district identity"

    pdum = pd.get_dummies(projects["district"]).astype(float)
    sdum = pd.get_dummies(projects["state_code"]).astype(float)
    for f in ["stakeholder_responsiveness", "historical_performance_score"]:
        cd = pdum.corrwith(projects[f]).abs().max()
        cs = sdum.corrwith(projects[f]).abs().max()
        print(f"  {f}: |corr w/ district|={cd:.3f}  |corr w/ state|={cs:.3f}")
        assert 0.10 <= cd <= 0.70, f"{f} district proxy out of band: {cd:.3f}"
        assert 0.10 <= cs <= 0.70, f"{f} state proxy out of band: {cs:.3f}"
    print("  -> partial proxy present, no direct leak")


def check_ongoing(live):
    print("=" * 60); print("[7] ONGOING OVERRUN EYEBALL"); print("=" * 60)
    ong = live[live["status"] == "ongoing"]
    ong = ong.copy()
    ong["overrun"] = ong["elapsed_days"] > ong["statutory_days"]
    print(f"  ongoing stages past statutory: {ong['overrun'].mean():.1%}")
    assert ong["overrun"].mean() > 0.05


def main():
    projects, parcels, villages, _districts, states, hist, live = load()
    check_volumes(projects, parcels, villages, states, hist, live)
    check_ids(projects, parcels)
    check_spatial(projects, parcels)
    check_state_volumes(parcels)
    check_rules(parcels, hist)
    check_confound(projects, parcels)
    check_ongoing(live)
    print("\nALL SANITY CHECKS PASSED.")


if __name__ == "__main__":
    main()
