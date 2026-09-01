"""Print a 'Demo facts' block sourced LIVE from metrics_report.json + portfolio cache.

Numbers are auto-discovered (no hardcoded IDs), so they never go stale. Re-run after
any regeneration/retrain:

    .venv/bin/python src/demo_numbers.py
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
PROJECTS = ROOT / "data" / "generated" / "projects.parquet"


def main() -> None:
    report = json.loads(REPORT.read_text())
    pf = pd.read_parquet(PORTFOLIO)
    projects = pd.read_parquet(PROJECTS)
    artifacts = load_artifacts()

    print("=" * 68)
    print("SIH26017 DEMO FACTS (auto-sourced, do not hand-edit)")
    print("=" * 68)

    # --- portfolio ---
    n = len(pf)
    red = (pf["risk_level"] == "RED").sum()
    yellow = (pf["risk_level"] == "YELLOW").sum()
    overrun = (pf["overrun_while_ongoing_days"] > 0).sum()
    print(f"\nLive portfolio: {n:,} parcels across {pf['state'].nunique()} states")
    print(f"  RED {red:,} | YELLOW {yellow:,} | GREEN {n - red - yellow:,}")
    print(f"  ongoing stages already past statutory: {overrun:,} ({overrun/n:.1%})")

    # --- model metrics ---
    print("\nPer-stage classifier AUROC (in-sample):")
    for s, m in report["stages"].items():
        print(f"  {s:14s} AUROC {m['classifier']['auroc']:.3f}   delay rate {m['delay_rate']:.0%}")

    lodo = report["cold_start_lodo"]["drop_pct"]
    loso = report["cold_start_loso"]["drop_pct"]
    print("\nCold-start drop (leave-one-district-out vs leave-one-state-out):")
    for s in report["stages"]:
        print(f"  {s:14s} LODO -{lodo[s]:.1f}%   LOSO -{loso[s]:.1f}%")
    loso_avg = float(np.mean(list(loso.values())))
    print(f"  -> LOSO avg drop {loso_avg:.1f}% (gate 2-15%)")

    # --- auto-pick demo projects ---
    def project_risk(pid):
        return float(pf[pf["project_id"] == pid]["risk_score"].mean())

    # point dam (highest risk with >=2 parcels)
    dams = projects[(projects["project_type"] == "dam") & (projects["spatial_type"] == "point")]
    dams = dams.assign(_n=dams["project_id"].map(lambda pid: (pf["project_id"] == pid).sum()),
                       _r=dams["project_id"].map(project_risk))
    dam = dams[dams["_n"] >= 2].sort_values("_r", ascending=False).iloc[0]

    # cross-state linear ROAD (highway) preferred over rail
    linear = projects[(projects["spatial_type"] == "linear")]
    linear = linear.sort_values("project_type", key=lambda s: (s != "road"))
    cross = None
    for _, p in linear.iterrows():
        st = pf[pf["project_id"] == p["project_id"]]["state_code"].unique()
        if len(st) > 1:
            cross = (p, list(st))
            break

    print("\nDemo project — point (dam):")
    print(f"  {dam['project_id']}  {dam['state']} / {dam['district']}  "
          f"risk {dam['_r']:.3f}  ({dam['_n']} parcels)")
    if cross is not None:
        p, states = cross
        print(f"Demo project — cross-state linear ({p['project_type']} highway):")
        print(f"  {p['project_id']}  spans {states}  risk {project_risk(p['project_id']):.3f}")

    # --- what-if: find a court-stay parcel that flips RED -> GREEN on clearing ---
    stay = pf[(pf["court_stay"] == 1) & (pf["compensation_status"] == "paid")]
    best = None
    for _, r in stay.head(1500).iterrows():
        feat = {c: r[c] for c in DEFAULTS}
        b = score_parcel(feat, artifacts=artifacts, parcel_id=r["parcel_id"])
        if b["risk_level"] != "RED":
            continue
        a = score_parcel({**feat, "court_stay": 0}, artifacts=artifacts, parcel_id=r["parcel_id"])
        if a["risk_level"] == "GREEN":
            best = (r, b, a)
            break
    if best is None:
        r = stay.sort_values("risk_score", ascending=False).iloc[0]
        feat = {c: r[c] for c in DEFAULTS}
        best = (r, score_parcel(feat, artifacts=artifacts, parcel_id=r["parcel_id"]),
                score_parcel({**feat, "court_stay": 0}, artifacts=artifacts, parcel_id=r["parcel_id"]))
    r, before, after = best
    print(f"\nWhat-if parcel {r['parcel_id']} ({r['district']}, court stay + {r['land_class']}):")
    print(f"  clear court stay: {before['risk_score']:.3f} ({before['risk_level']}) -> "
          f"{after['risk_score']:.3f} ({after['risk_level']})")
    print(f"  expected overrun {before['expected_overrun_days']:.0f}d -> {after['expected_overrun_days']:.0f}d")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()
