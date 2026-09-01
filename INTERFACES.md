# SIH26017 — Prediction Contract & API Spec

The `predict.py` engine emits a fixed JSON contract that any frontend (Streamlit, a
custom dashboard, or an external consumer) can render. The FastAPI endpoint
(`app/api.py`) wraps this same contract.

## 1. The prediction contract (per parcel)

```json
{
  "parcel_id": "PRCL_0000006",
  "risk_score": 0.8517,
  "risk_level": "RED",
  "expected_overrun_days": 477.1,
  "max_delay_prob": 0.9997,
  "stages": {
    "SIA": {
      "delay_prob": 0.4982,
      "expected_overrun": -1.8,
      "statutory_days": 180,
      "status": null,
      "elapsed_days": null,
      "actual_days": null,
      "top_factors": [["stakeholder_responsiveness", -0.984], ["project_type", 0.517]]
    }
  },
  "top_factors": [["court_stay", 11.67], ["affected_families", 3.57]],
  "recommended_actions": [
    {"factor": "court_stay", "action": "...", "priority": 1, "priority_label": "high"}
  ],
  "overrun_while_ongoing_days": 32.0
}
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `risk_score` | float 0–1 | Severity-based composite: `1 - exp(-expected_overrun_days / 250)`. Drives RED/YELLOW/GREEN. |
| `risk_level` | enum | `RED` (>0.70), `YELLOW` (0.40–0.70), `GREEN` (<0.40). |
| `expected_overrun_days` | float | **PREDICTED** total future overrun = sum of the 5 stage regressor outputs (clamped ≥0). Comes from the models. |
| `max_delay_prob` | float 0–1 | Max per-stage delay probability across the 5 classifiers. |
| `stages[].delay_prob` | float 0–1 | Probability that the stage overruns its statutory limit. |
| `stages[].expected_overrun` | float | Predicted days-overrun for that stage (regressor). |
| `stages[].statutory_days` | int | RFCTLARR statutory clock for the stage. |
| `stages[].status` | enum/null | `completed`/`ongoing`/`pending` from the live timeline (null if none supplied). |
| `stages[].elapsed_days` | int/null | Days elapsed so far (for `ongoing` stages). |
| `stages[].actual_days` | int/null | Actual days taken (for `completed` stages). |
| `stages[].top_factors` | list | Top-3 signed SHAP contributions for this parcel at that stage (+, increases delay; −, decreases). |
| `top_factors` | list | Global driver ranking = sum of |SHAP| across all 5 stages. |
| `recommended_actions` | list | Rule-based corrective actions (factor → action, priority 1=high…3=low). |
| `overrun_while_ongoing_days` | float/null | **MEASURED** current overrun = `elapsed_days − statutory_days` for the parcel's ongoing stage; `null` if none. |

### ⚠️ Two different "overrun" fields — do not conflate

- **`expected_overrun_days`** is a *prediction* (future, from the regressors, summed over
  all 5 stages, clamped ≥0). It answers: *"how many days over limit will this parcel run
  in total?"*
- **`overrun_while_ongoing_days`** is a *measurement* (present, from the live timeline,
  `elapsed − statutory` for the stage currently in progress). It answers: *"is this parcel
  already past its legal limit and still not finished?"*

They are deliberately both in the contract, but they are computed from completely
different sources (models vs live timeline).

## 2. The 12 features (model input)

`project_type`, `compensation_status`, `land_class` (categorical, ordinal-encoded) +
`affected_families`, `rehab_progress_pct`, `stakeholder_responsiveness`,
`historical_performance_score`, `owner_count`, `area_sqm`, `pending_mutations`,
`court_stay`, `encumbrances` (numeric).

Geo/identifier fields (state, state_code, district, district_code, village, village_code,
khasra, parcel_id, project_id) are **never** model features — they exist only for
rollup/validation/map display.

## 2a. Semantic IDs (multi-state)

- **Parcel ID:** `<STATE>-<DISTRICT_CODE>-<VILLAGE_CODE>-<KHASRA_NO>` — e.g. `HP-KNG-0001-0123`.
- **Project ID:** `<STATE>-<TYPE>-<YEAR>-<SEQ>` — e.g. `HP-DAM-2024-0001`.
- **Home-state rule:** a cross-state linear project's `<STATE>` is the state of its first
  path parcel.
- IDs are **derived from** the parcel's assigned geography (state/district/village codes
  always match the geo columns).
- State codes: `HP` (Himachal Pradesh), `PB` (Punjab), `UK` (Uttarakhand).
- Project type codes: `RDH` road, `RLY` rail, `IRR` irrigation, `DAM` dam, `IND` industrial.

## 2b. Spatial types

- `point` — localized project (1 district, 1–2 villages). `coord_path` is `[]`.
- `linear` — corridor project (road/rail) routed through contiguous villages, may cross
  district/state borders. `coord_path` is a JSON list of ordered `[lat, lon]` village
  coordinates used to render the polyline on the offline map.

## 2c. Cold-start validation (Stage 2)

Two leave-one-group-out levels, split by each **parcel's own** district/state:
- **LODO** (leave-one-district-out, 48 folds) — drop ≤10% relative.
- **LOSO** (leave-one-state-out, 3 folds) — drop 2–15% relative (avg), the strong
  pan-India cold-start proof.
Both recorded in `models/metrics_report.json` with a `gates_ok` flag.

## 3. FastAPI endpoints (`app/api.py`)

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok", "models_loaded": true}` |
| POST | `/predict` | `{...12 raw features..., "parcel_id": "..."}` | contract (section 1) |
| POST | `/predict/batch` | `{"parcels": [{...features...}, ...]}` | `{"results": [contract, ...]}` |

`/predict` accepts the raw (unencoded) feature values — encoding is handled server-side
using the saved `encoders.joblib` + `feature_columns.json`, guaranteeing the transform is
identical to training.

## 4. Portfolio cache

`src/predict.py --refresh-portfolio` scores every live parcel and writes
`data/generated/portfolio_scores.parquet` with `risk_score`, `risk_level`,
`expected_overrun_days`, `max_delay_prob`, per-stage `{STAGE}_prob` / `{STAGE}_overrun`,
the 12 raw features, geo (village/village_code/tehsil/district/district_code/state/
state_code + village lat/lon), `spatial_type`, `current_stage`, and
`overrun_while_ongoing_days`. The dashboard reads this cache (~0.4s to regenerate live).

User-created projects (onboarding) persist to `data/generated/user/user_projects.parquet`
+ `user_parcels.parquet` (schema-matched to the portfolio cache) and are merged at load,
tagged `is_user=1`.
