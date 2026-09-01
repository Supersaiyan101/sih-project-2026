"""SIH26017 end-to-end test — the final gate.

Proves two things:
  1. In-place: existing data + models + portfolio are valid, the prediction contract
     is well-formed, the FastAPI answers correctly, and the dashboard renders all views.
  2. Fresh-clone proof: copying ONLY the source (no data/, no models/) and running
     `scripts/bootstrap.sh --no-launch` regenerates a working demo. This demonstrates
     the "fresh clone -> working demo" claim rather than merely documenting it.

Usage:
  .venv/bin/python src/e2e_test.py                 # full (incl. fresh-clone, ~2 min)
  .venv/bin/python src/e2e_test.py --skip-fresh    # in-place checks only (fast)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASS = []
FAIL = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail and not cond else ""))


# --------------------------------------------------------------------------- #
# 1. In-place artifact + contract checks
# --------------------------------------------------------------------------- #

def check_artifacts() -> None:
    print("== 1. Artifacts ==")
    ok("data generated", (ROOT / "data/generated/parcels.parquet").exists())
    ok("states generated", (ROOT / "data/generated/states.parquet").exists())
    ok("10 models", all((ROOT / "models" / f"{s}_{m}.joblib").exists()
                        for s in ["SIA", "NOTIFICATION", "DECLARATION", "AWARD", "POSSESSION"]
                        for m in ["classifier", "regressor"]))
    ok("encoders", (ROOT / "models/encoders.joblib").exists())
    ok("metrics report", (ROOT / "models/metrics_report.json").exists())
    ok("portfolio cache", (ROOT / "data/generated/portfolio_scores.parquet").exists())


def check_data_invariants() -> None:
    print("== 2. Data invariants (multi-state, IDs, spatial) ==")
    import re
    import pandas as pd
    pr = pd.read_parquet(ROOT / "data/generated/projects.parquet")
    pa = pd.read_parquet(ROOT / "data/generated/parcels.parquet")
    st = pd.read_parquet(ROOT / "data/generated/states.parquet")

    ok("3 states", len(st) == 3)
    parcel_re = re.compile(r"^(HP|PB|UK)-[A-Z]{3}-\d{4}-\d{4}$")
    project_re = re.compile(r"^(HP|PB|UK)-(RDH|RLY|IRR|DAM|IND)-\d{4}-\d{4}$")
    ok("parcel ID format", bool(pa["parcel_id"].str.match(parcel_re).all()))
    ok("project ID format", bool(pr["project_id"].str.match(project_re).all()))
    ok("ID<->geo consistent",
       bool((pa["state_code"] == pa["parcel_id"].str[:2]).all()))

    lin = pr[pr["spatial_type"] == "linear"]
    pt = pr[pr["spatial_type"] == "point"]
    lin_multi = sum((pa[pa["project_id"] == p["project_id"]]["district_code"].nunique() > 1)
                    for _, p in lin.iterrows())
    lin_state = sum((pa[pa["project_id"] == p["project_id"]]["state_code"].nunique() > 1)
                    for _, p in lin.iterrows())
    pt_bad = sum((pa[pa["project_id"] == p["project_id"]]["district_code"].nunique() > 1)
                 for _, p in pt.iterrows())
    ok("linear spans >1 district", lin_multi == len(lin))
    ok("some linear cross state", lin_state > 0)
    ok("point stays in 1 district", pt_bad == 0)

    import json
    report = json.loads((ROOT / "models/metrics_report.json").read_text())
    ok("LOSO present", "cold_start_loso" in report)
    ok("gates passed", report.get("gates_ok") is True)


def check_contract() -> None:
    print("== 3. Prediction contract ==")
    from predict import load_artifacts, score_parcel, DEFAULTS
    artifacts = load_artifacts()
    c = score_parcel({**DEFAULTS, "court_stay": 1}, artifacts=artifacts, parcel_id="E2E")
    ok("contract keys", all(k in c for k in
                            ["risk_score", "risk_level", "expected_overrun_days",
                             "max_delay_prob", "stages", "top_factors",
                             "recommended_actions", "overrun_while_ongoing_days"]))
    ok("5 stages", len(c["stages"]) == 5)
    ok("risk in [0,1]", 0.0 <= c["risk_score"] <= 1.0)
    ok("actions non-empty", len(c["recommended_actions"]) > 0)
    ok("top_factors present", len(c["top_factors"]) > 0)


def check_api() -> None:
    print("== 4. FastAPI ==")
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    ok("GET /health", client.get("/health").json().get("status") == "ok")
    r = client.post("/predict", json={"parcel_id": "E2E", "court_stay": 1,
                                      "compensation_status": "pending"})
    ok("POST /predict 200", r.status_code == 200)
    ok("POST /predict shape", "risk_score" in r.json())
    rb = client.post("/predict/batch", json={"parcels": [
        {"parcel_id": "A"}, {"parcel_id": "B", "court_stay": 1}]})
    ok("POST /predict/batch", rb.status_code == 200 and len(rb.json()["results"]) == 2)


def check_dashboard() -> None:
    print("== 5. Dashboard (all views) ==")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=240)
    at.run()
    ok("no exceptions", len(at.exception) == 0,
       "; ".join(repr(e.value) for e in at.exception))
    for page in ["Project", "New Project", "Detail", "What-if", "Alerts",
                 "Area of Interest", "Map"]:
        at.radio[0].set_value(page)
        at.run()
        ok(f"view {page}", len(at.exception) == 0)
    # viewer gating
    at.sidebar.selectbox[0].set_value("Viewer")
    at.run()
    ok("viewer hides active views", [o for o in at.radio[0].options]
       == ["Portfolio", "Project", "Detail", "Map"])


# --------------------------------------------------------------------------- #
# 5. Fresh-clone proof
# --------------------------------------------------------------------------- #

IGNORE = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "models",
          ".gitignore"}


def check_fresh_clone() -> None:
    print("== 5. Fresh-clone bootstrap (from code only) ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "clone"
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns(*IGNORE))
        # simulate a fresh clone: no generated data, no trained models
        assert not (tmp / "data/generated/parcels.parquet").exists()
        assert not (tmp / "models").exists()

        proc = subprocess.run(
            ["bash", "scripts/bootstrap.sh", "--no-launch"],
            cwd=tmp, capture_output=True, text=True, timeout=900,
        )
        ok("bootstrap exit 0", proc.returncode == 0,
           (proc.stdout + proc.stderr)[-2000:])
        ok("data regenerated", (tmp / "data/generated/parcels.parquet").exists())
        ok("models regenerated", (tmp / "models/POSSESSION_classifier.joblib").exists())
        ok("portfolio regenerated", (tmp / "data/generated/portfolio_scores.parquet").exists())

        # models actually load + score in the fresh clone
        p = subprocess.run(
            [str(tmp / ".venv/bin/python"), "-c",
             "import sys; sys.path.insert(0, 'src');"
             "from predict import load_artifacts, score_parcel, DEFAULTS;"
             "a = load_artifacts();"
             "c = score_parcel(dict(DEFAULTS, court_stay=1), artifacts=a, parcel_id='X');"
             "assert 0 <= c['risk_score'] <= 1; print('fresh score OK')"],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        ok("fresh models score", p.returncode == 0, (p.stdout + p.stderr)[-1000:])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fresh", action="store_true")
    args = ap.parse_args()

    print("SIH26017 end-to-end test")
    check_artifacts()
    check_data_invariants()
    check_contract()
    check_api()
    check_dashboard()
    if not args.skip_fresh:
        check_fresh_clone()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
        sys.exit(1)
    print("ALL E2E CHECKS PASSED.")


if __name__ == "__main__":
    main()
