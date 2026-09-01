"""SIH26017 model training.

Trains 10 models (5 stages x {classifier, regressor}), runs leave-one-district-out
cold-start validation, exports SHAP explanations, and writes a metrics report.

Notes:
  - X is parcel-level (one row per parcel), so any row split is already parcel-safe
    (no long-format leakage across stages).
  - Final saved models are refit on the FULL historical set (best for deployment);
    the 80/20 split is used only for honest in-sample metric reporting.
  - --incremental is a REFIT-on-appended-data path (HistGB has no partial_fit), NOT
    online learning. It folds in any data/generated/append_*.parquet files if present.

Usage:
  .venv/bin/python src/train.py                # full train + validation + SHAP
  .venv/bin/python src/train.py --incremental  # refit on historical (+ appended) data
  .venv/bin/python src/train.py --skip-lodo     # skip cold-start (faster dev loop)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import train_test_split

from features import (DATA, MODELS, STAGES, build_features, rollup_risk,
                      save_encoders, save_feature_columns)

CLF_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=42)
REG_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=42)
SHAP_SAMPLES = 1000


def _clf():
    return HistGradientBoostingClassifier(**CLF_PARAMS)


def _reg():
    return HistGradientBoostingRegressor(**REG_PARAMS)


# --------------------------------------------------------------------------- #
# Training + metrics
# --------------------------------------------------------------------------- #

def fit_stage_models(X, y) -> dict:
    models = {}
    for stage in STAGES:
        models[stage] = {
            "classifier": _clf().fit(X, y[f"{stage}_delay_flag"]),
            "regressor": _reg().fit(X, y[f"{stage}_delay_days"]),
        }
    return models


def evaluate(X_test, y_test, models) -> dict:
    out = {}
    for stage in STAGES:
        yc = y_test[f"{stage}_delay_flag"]
        yr = y_test[f"{stage}_delay_days"]
        p = models[stage]["classifier"].predict_proba(X_test)[:, 1]
        d = models[stage]["regressor"].predict(X_test)
        out[stage] = {
            "delay_rate": float(yc.mean()),
            "classifier": {
                "auroc": float(roc_auc_score(yc, p)),
                "average_precision": float(average_precision_score(yc, p)),
            },
            "regressor": {
                "mae": float(mean_absolute_error(yr, d)),
                "rmse": float(mean_squared_error(yr, d) ** 0.5),
                "r2": float(1 - np.sum((yr - d) ** 2) / np.sum((yr - yr.mean()) ** 2)),
            },
        }
    return out


# --------------------------------------------------------------------------- #
# Cold-start: leave-one-group-out (district OR state)
# --------------------------------------------------------------------------- #

def _leave_one_group_out(X, y, meta, group_col: str) -> dict:
    groups = sorted(meta[group_col].unique())
    per_group = {}
    per_stage = {s: [] for s in STAGES}

    for g in groups:
        test_mask = (meta[group_col] == g).to_numpy()
        X_tr, X_te = X[~test_mask], X[test_mask]
        y_tr, y_te = y[~test_mask], y[test_mask]
        row = {}
        for stage in STAGES:
            clf = _clf().fit(X_tr, y_tr[f"{stage}_delay_flag"])
            p = clf.predict_proba(X_te)[:, 1]
            auroc = float(roc_auc_score(y_te[f"{stage}_delay_flag"], p))
            row[stage] = auroc
            per_stage[stage].append(auroc)
        per_group[g] = row

    heldout_avg = {s: float(np.mean(per_stage[s])) for s in STAGES}
    return {"per_group": per_group, "heldout_auroc_avg": heldout_avg}


def lodo_cold_start(X, y, meta) -> dict:
    r = _leave_one_group_out(X, y, meta, "district")
    r["per_district"] = r.pop("per_group")
    return r


def loso_cold_start(X, y, meta) -> dict:
    r = _leave_one_group_out(X, y, meta, "state_code")
    r["per_state"] = r.pop("per_group")
    return r


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #

def export_shap(X, y, models, shap_dir: Path) -> dict:
    """TreeExplainer on classifiers for a sampled subset; fallback to permutation."""
    shap_dir.mkdir(parents=True, exist_ok=True)
    global_importance = {}
    idx = np.random.RandomState(42).choice(len(X), size=min(SHAP_SAMPLES, len(X)), replace=False)
    Xs = X.iloc[idx]

    for stage in STAGES:
        clf = models[stage]["classifier"]
        try:
            import shap
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):  # binary classifier returns list
                sv = sv[1]
            np.save(shap_dir / f"{stage}_shap_values.npy", np.asarray(sv))
            np.save(shap_dir / f"{stage}_shap_base.npy", np.asarray(explainer.expected_value))
            np.save(shap_dir / f"{stage}_sample_parcel_ids.npy", np.asarray(Xs.index))
            np.save(shap_dir / f"{stage}_sample_X.npy", np.asarray(Xs))
            global_importance[stage] = [
                [c, float(np.abs(sv).mean(axis=0)[i])]
                for i, c in enumerate(X.columns)
            ]
        except Exception as e:
            print(f"  SHAP failed for {stage} ({e}); using permutation_importance")
            from sklearn.inspection import permutation_importance
            r = permutation_importance(clf, Xs, y.loc[Xs.index, f"{stage}_delay_flag"],
                                       n_repeats=5, random_state=42)
            global_importance[stage] = [
                [c, float(r.importances_mean[i])] for i, c in enumerate(X.columns)
            ]

    # sort by importance desc
    for stage in STAGES:
        global_importance[stage] = sorted(global_importance[stage], key=lambda x: -x[1])
    return global_importance


# --------------------------------------------------------------------------- #
# Incremental data
# --------------------------------------------------------------------------- #

def load_incremental_extra() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold in any data/generated/append_*.parquet files (refit-on-new-data path)."""
    extra = {"parcels": [], "projects": [], "stage_timelines_historical": []}
    for p in DATA.glob("append_*.parquet"):
        name = p.stem.replace("append_", "")
        if name in extra:
            extra[name].append(pd.read_parquet(p))
    for k in extra:
        if extra[k]:
            print(f"  appended {len(extra[k])} file(s) for '{k}'")
    return (pd.concat(extra["parcels"]) if extra["parcels"] else None,
            pd.concat(extra["projects"]) if extra["projects"] else None,
            pd.concat(extra["stage_timelines_historical"]) if extra["stage_timelines_historical"] else None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", help="refit on historical (+ appended) data")
    ap.add_argument("--skip-lodo", action="store_true", help="skip cold-start validation")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading + building features ...")
    parcels = pd.read_parquet(DATA / "parcels.parquet")
    projects = pd.read_parquet(DATA / "projects.parquet")

    if args.incremental:
        ep, epr, et = load_incremental_extra()
        if ep is not None:
            parcels = pd.concat([parcels, ep], ignore_index=True)
        if epr is not None:
            projects = pd.concat([projects, epr], ignore_index=True)
        if et is not None:
            # appended timelines need matching parcels to be meaningful
            print("  (appended stage_timelines files are read but treated as source-of-truth refit)")

    feat = build_features(parcels, projects, fit_encoder=True)
    X, y, meta = feat["X"], feat["y"], feat["meta"]

    save_encoders(feat["encoders"], MODELS / "encoders.joblib")
    save_feature_columns(feat["feature_columns"], feat["encoders"], MODELS / "feature_columns.json")

    print(f"  parcels: {len(X)}, features: {len(X.columns)}")

    # --- 80/20 parcel-safe split (X is parcel-level, one row per parcel) ---
    tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    print("Training split models (for in-sample metrics) ...")
    split_models = fit_stage_models(X_tr, y_tr)
    metrics = evaluate(X_te, y_te, split_models)

    # --- cold-start LODO + LOSO ---
    cold_start = None
    cold_start_state = None
    if not args.skip_lodo:
        print("Running leave-one-district-out (LODO) cold-start validation ...")
        cold_start = lodo_cold_start(X, y, meta)
        print("Running leave-one-state-out (LOSO) cold-start validation ...")
        cold_start_state = loso_cold_start(X, y, meta)

    # --- SHAP ---
    print("Exporting SHAP explanations ...")
    shap_importance = export_shap(X, y, split_models, MODELS / "shap")

    # --- final models refit on FULL data (deployment) ---
    print("Refitting final models on full data ...")
    final_models = fit_stage_models(X, y)
    for stage in STAGES:
        joblib.dump(final_models[stage]["classifier"], MODELS / f"{stage}_classifier.joblib")
        joblib.dump(final_models[stage]["regressor"], MODELS / f"{stage}_regressor.joblib")

    # --- rollup smoke test (project + district level) ---
    probs = {s: final_models[s]["classifier"].predict_proba(X)[:, 1] for s in STAGES}
    overruns = {s: final_models[s]["regressor"].predict(X) for s in STAGES}
    from features import compute_risk
    risk = pd.DataFrame({
        "risk_score": [compute_risk({s: probs[s][i] for s in STAGES},
                                    {s: overruns[s][i] for s in STAGES})["risk_score"]
                       for i in range(len(X))]
    }, index=X.index)
    district_risk = rollup_risk(risk, meta, "district").sort_values("risk_score", ascending=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_parcels": int(len(X)),
        "n_features": len(X.columns),
        "feature_columns": X.columns.tolist(),
        "stages": metrics,
        "shap_importance": shap_importance,
        "cold_start_lodo": cold_start,
        "cold_start_loso": cold_start_state,
        "rollup_district_risk": district_risk.to_dict(orient="records"),
    }
    if cold_start:
        report["cold_start_lodo"]["drop_pct"] = {
            s: round((metrics[s]["classifier"]["auroc"] - cold_start["heldout_auroc_avg"][s])
                     / metrics[s]["classifier"]["auroc"] * 100, 2)
            for s in STAGES
        }
    if cold_start_state:
        report["cold_start_loso"]["drop_pct"] = {
            s: round((metrics[s]["classifier"]["auroc"] - cold_start_state["heldout_auroc_avg"][s])
                     / metrics[s]["classifier"]["auroc"] * 100, 2)
            for s in STAGES
        }

    (MODELS / "metrics_report.json").write_text(json.dumps(report, indent=2))

    print("\n================= RESULTS =================")
    for s in STAGES:
        m = metrics[s]
        top = shap_importance[s][:3]
        print(f"[{s}] AUROC={m['classifier']['auroc']:.3f}  AP={m['classifier']['average_precision']:.3f}  "
              f"MAE={m['regressor']['mae']:.1f}  RMSE={m['regressor']['rmse']:.1f}  "
              f"delay_rate={m['delay_rate']:.2f}  top3={[t[0] for t in top]}")
    if cold_start:
        print("\nCold-start LODO (district) held-out AUROC vs in-sample:")
        for s in STAGES:
            print(f"  {s:14s} in-sample={metrics[s]['classifier']['auroc']:.3f}  "
                  f"held-out={cold_start['heldout_auroc_avg'][s]:.3f}  "
                  f"drop={report['cold_start_lodo']['drop_pct'][s]:.1f}%")
    if cold_start_state:
        print("\nCold-start LOSO (state) held-out AUROC vs in-sample:")
        for s in STAGES:
            print(f"  {s:14s} in-sample={metrics[s]['classifier']['auroc']:.3f}  "
                  f"held-out={cold_start_state['heldout_auroc_avg'][s]:.3f}  "
                  f"drop={report['cold_start_loso']['drop_pct'][s]:.1f}%")

    # --- Stage 2 hard gates ---
    gates_ok = True
    if cold_start and cold_start_state:
        lodo_drop = max(report["cold_start_lodo"]["drop_pct"].values())
        loso_drop_avg = float(np.mean(list(report["cold_start_loso"]["drop_pct"].values())))
        loso_drop_max = max(report["cold_start_loso"]["drop_pct"].values())
        min_auroc = min(metrics[s]["classifier"]["auroc"] for s in STAGES)
        print("\n================= GATES =================")
        print(f"LODO max drop: {lodo_drop:.2f}%  (gate: <=10%)")
        print(f"LOSO avg drop: {loso_drop_avg:.2f}%  (gate: 2-15%)")
        print(f"min in-sample AUROC: {min_auroc:.3f}  (gate: >=0.70)")
        if not (lodo_drop <= 10.0):
            print("GATE FAIL: LODO drop > 10%"); gates_ok = False
        if not (2.0 <= loso_drop_avg <= 15.0):
            print("GATE FAIL: LOSO drop outside 2-15%"); gates_ok = False
        if not (min_auroc >= 0.70):
            print("GATE FAIL: AUROC floor < 0.70"); gates_ok = False
        print("ALL GATES PASSED." if gates_ok else "GATES FAILED -> return to Stage 0.")
    report["gates_ok"] = gates_ok
    (MODELS / "metrics_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nDistrict risk ranking (area-weighted):")
    print(district_risk[["district", "risk_score", "n_parcels"]].to_string(index=False))
    print(f"\nDone in {time.time() - t0:.1f}s. Models + report in {MODELS}/")


if __name__ == "__main__":
    main()
