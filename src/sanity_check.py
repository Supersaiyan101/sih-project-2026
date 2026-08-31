"""Day 1 sanity checks for generated data.

Checks:
  1. Volumes + null counts match expectations.
  2. Delay rules are recoverable (correlation checks on court_stay, orchard, etc.).
  3. Hidden confound: delay varies ACROSS districts AND is NOT directly leaked into
     any visible column (no near-perfect collinearity with a feature).
  4. Ongoing-row eyeball: live stages with elapsed_days sometimes exceeding statutory.

Run:  .venv/bin/python src/sanity_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GEN = Path(__file__).resolve().parent.parent / "data" / "generated"


def load():
    projects = pd.read_parquet(GEN / "projects.parquet")
    parcels = pd.read_parquet(GEN / "parcels.parquet")
    villages = pd.read_parquet(GEN / "villages.parquet")
    districts = pd.read_parquet(GEN / "districts.parquet")
    hist = pd.read_parquet(GEN / "stage_timelines_historical.parquet")
    live = pd.read_parquet(GEN / "stage_timelines_live.parquet")
    return projects, parcels, villages, districts, hist, live


def check_volumes(projects, parcels, villages, hist, live):
    print("=" * 60)
    print("[1] VOLUMES + NULLS")
    print("=" * 60)
    print(f"projects: {len(projects)}  (expect 5000)")
    print(f"parcels : {len(parcels)}  (expect 100000)")
    print(f"villages: {len(villages)}")
    print(f"historical rows: {len(hist)}  (expect ~500000)")
    print(f"live rows      : {len(live)}  (expect ~5 x live parcels)")
    n_live = parcels["is_live"].sum()
    print(f"live parcels   : {n_live}")
    print(f"  live stage breakdown:\n{live['status'].value_counts().to_string()}")

    assert len(projects) == 5000
    assert len(parcels) == 100000
    assert len(hist) == 500000
    assert live["status"].notna().all()
    assert (hist["status"] == "completed").all()
    # null pattern in live: pending stages must have no actual/elapsed/delay
    assert live.loc[live["status"] == "pending", "actual_days"].isna().all()
    assert live.loc[live["status"] == "ongoing", "actual_days"].isna().all()
    assert live.loc[live["status"] == "ongoing", "elapsed_days"].notna().all()
    assert live.loc[live["status"] == "completed", "actual_days"].notna().all()
    print("  -> null/status pattern OK")


def check_rules(parcels, hist):
    print("=" * 60)
    print("[2] RULE RECOVERY (correlation of factor -> delay)")
    print("=" * 60)
    # wide delay per parcel-stage for AWARD / POSSESSION / DECLARATION
    aw = hist[hist["stage"] == "AWARD"].set_index("parcel_id")["delay_days"].rename("award_delay")
    po = hist[hist["stage"] == "POSSESSION"].set_index("parcel_id")["delay_days"].rename("poss_delay")
    de = hist[hist["stage"] == "DECLARATION"].set_index("parcel_id")["delay_days"].rename("decl_delay")
    df = parcels.set_index("parcel_id").join([aw, po, de])

    def show(cond, label, col):
        mean_delay = df.loc[cond, col].mean()
        base = df.loc[~cond, col].mean()
        print(f"  {label:38s} {col:15s} mean_delay {mean_delay:8.1f} vs base {base:8.1f}")

    show(df["court_stay"] == 1, "court_stay=1", "award_delay")
    show(df["land_class"] == "orchard", "orchard land", "award_delay")
    show(df["encumbrances"] >= 2, "encumbrances>=2", "award_delay")
    show(df["court_stay"] == 1, "court_stay=1", "poss_delay")
    show(df["owner_count"] > 4, "owner_count>4", "decl_delay")
    show(df["pending_mutations"] >= 2, "pending_mutations>=2", "decl_delay")
    print("  (delays should be meaningfully higher for the flagged group)")


def check_confound_leak(projects, parcels, villages, districts, hist):
    print("=" * 60)
    print("[3] HIDDEN CONFOUND (district admin_capacity)")
    print("=" * 60)
    # admin_capacity is dropped from outputs; re-derive is impossible -> leak check on
    # visible columns: verify no visible column is a near-perfect proxy of district
    # (i.e., delay should vary across districts, but no single feature should fully
    # determine district identity).
    per_district = (
        hist.merge(parcels[["parcel_id", "district"]], on="parcel_id")
        .groupby("district")["delay_days"].mean().sort_values()
    )
    print("  mean delay_days by district (hidden confound drives spread):")
    print(per_district.round(1).to_string())

    # Leak check: correlation of each numeric feature with district dummies should not
    # be near-perfect (i.e., feature alone must not equal district). Report max abs corr.
    feats = ["owner_count", "pending_mutations", "court_stay", "encumbrances"]
    dummies = pd.get_dummies(parcels["district"]).astype(float)
    worst = 0.0
    worst_name = ""
    for f in feats:
        corr = dummies.corrwith(parcels[f]).abs().max()
        if corr > worst:
            worst, worst_name = corr, f
        print(f"  max |corr(district, {f})| = {corr:.3f}")
    print(f"  -> worst feature->district correlation: {worst_name} ({worst:.3f})")
    assert worst < 0.5, "feature leaks district identity too strongly"
    print("  -> no direct leak (no visible column encodes the hidden confound)")


def check_ongoing_eyeball(live):
    print("=" * 60)
    print("[4] ONGOING-ROW EYEBALL (overrun-while-ongoing money shot)")
    print("=" * 60)
    ongoing = live[live["status"] == "ongoing"].copy()
    ongoing["overrun"] = ongoing["elapsed_days"] > ongoing["statutory_days"]
    frac = ongoing["overrun"].mean()
    print(f"  ongoing stages already past statutory: {frac:.1%}")
    print("  sample ongoing rows (incl. some overrun):")
    cols = ["parcel_id", "stage", "statutory_days", "elapsed_days"]
    sample = ongoing[ongoing["overrun"]].head(6)
    if sample.empty:
        sample = ongoing.head(6)
    print(sample[cols].to_string(index=False))
    assert frac > 0.05, "not enough overrun-while-ongoing cases"


def main():
    projects, parcels, villages, districts, hist, live = load()
    check_volumes(projects, parcels, villages, hist, live)
    check_rules(parcels, hist)
    check_confound_leak(projects, parcels, villages, districts, hist)
    check_ongoing_eyeball(live)
    print("\nALL SANITY CHECKS PASSED.")


if __name__ == "__main__":
    main()
