"""Print a 'Demo facts' block sourced LIVE from metrics_report.json + portfolio cache.

Purpose: DEMO_SCRIPT.md and the pitch should never hardcode numbers that can go stale.
If you regenerate data or retrain, re-run this script to refresh the facts:

    .venv/bin/python src/demo_numbers.py

Output is copy-paste ready for the demo narrative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_artifacts, score_parcel, DEFAULTS  # noqa: E402

REPORT = ROOT / "models" / "metrics_report.json"
PORTFOLIO = ROOT / "data" / "generated" / "portfolio_scores.parquet"

# curated demo parcels (chosen for a clean story; re-check after regeneration)
DEMO_PARCEL = "PRCL_0000006"   # active court stay, already 124d past award limit, RED
WHATIF_PARCEL = "PRCL_0000006"  # same parcel -> clearing court stay flips RED -> GREEN


def main() -> None:
    report = json.loads(REPORT.read_text())
    pf = pd.read_parquet(PORTFOLIO)
    artifacts = load_artifacts()

    print("=" * 66)
    print("SIH26017 DEMO FACTS (auto-sourced, do not hand-edit)")
    print("=" * 66)

    # --- portfolio ---
    n = len(pf)
    red = (pf["risk_level"] == "RED").sum()
    yellow = (pf["risk_level"] == "YELLOW").sum()
    green = (pf["risk_level"] == "GREEN").sum()
    overrun = (pf["overrun_while_ongoing_days"] > 0).sum()
    print(f"\nLive portfolio: {n:,} parcels")
    print(f"  RED {red:,} | YELLOW {yellow:,} | GREEN {green:,}")
    print(f"  ongoing stages already past statutory: {overrun:,} ({overrun/n:.1%})")

    # --- model metrics ---
    stages = report["stages"]
    print("\nPer-stage classifier AUROC (in-sample):")
    for s, m in stages.items():
        print(f"  {s:14s} AUROC {m['classifier']['auroc']:.3f}   delay rate {m['delay_rate']:.0%}")

    drop = report["cold_start_lodo"]["drop_pct"]
    print("\nCold-start (leave-one-district-out) AUROC drop:")
    for s, d in drop.items():
        print(f"  {s:14s} -{d:.1f}%")

    # --- district ranking ---
    dr = report["rollup_district_risk"]
    top = dr[0]
    bot = dr[-1]
    print(f"\nDistrict risk ranking (area-weighted):")
    print(f"  worst: {top['district']} {top['risk_score']:.2f}   best: {bot['district']} {bot['risk_score']:.2f}")
    for r in dr[:3]:
        print(f"    {r['district']:20s} {r['risk_score']:.3f}")

    # --- curated parcels ---
    for pid in (DEMO_PARCEL, WHATIF_PARCEL):
        row = pf[pf["parcel_id"] == pid]
        if row.empty:
            print(f"\n(parcel {pid} not found in portfolio)")
            continue
        r = row.iloc[0]
        print(f"\nDemo parcel {pid} ({r['district']}):")
        print(f"  land_class={r['land_class']}, owners={int(r['owner_count'])}, "
              f"court_stay={int(r['court_stay'])}, compensation={r['compensation_status']}")
        print(f"  risk={r['risk_score']:.3f} ({r['risk_level']}), expected overrun {r['expected_overrun_days']:.0f}d")
        if r["overrun_while_ongoing_days"] and r["overrun_while_ongoing_days"] > 0:
            print(f"  ALREADY {r['overrun_while_ongoing_days']:.0f}d past statutory at stage {r['current_stage']}")

    # --- what-if ---
    w = pf[pf["parcel_id"] == WHATIF_PARCEL].iloc[0]
    feat = {c: w[c] for c in DEFAULTS}
    before = score_parcel(feat, artifacts=artifacts, parcel_id=WHATIF_PARCEL)
    after = score_parcel({**feat, "court_stay": 0},
                         artifacts=artifacts, parcel_id=WHATIF_PARCEL)
    print(f"\nWhat-if on {WHATIF_PARCEL} (clear the court stay):")
    print(f"  risk {before['risk_score']:.3f} ({before['risk_level']}) -> "
          f"{after['risk_score']:.3f} ({after['risk_level']})")
    print(f"  expected overrun {before['expected_overrun_days']:.0f}d -> {after['expected_overrun_days']:.0f}d")

    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
