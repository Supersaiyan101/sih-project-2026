# SIH26017 — Predictive Analytics for Early Detection of Land Acquisition Delays
**Project Context & Decision Log — updated every session. Any AI/human: read this first.**

## 1. Problem (one paragraph)
Ministry of Rural Development PS SIH26017: build an AI-powered early-warning system that
predicts which land acquisition projects will get delayed, using (a) static land-record
features and (b) historical stage-wise timelines under the RFCTLARR Act, 2013. Output per
parcel: risk score, per-stage delay probability, explainable contributing factors, and
recommended actions.

## 2. Core Architecture (LOCKED)
- Two data types merged:
  - **A. Static land features** (HimBhoomi-style): khasra number, owner count, land class
    (orchard/barren/agri/residential), area, pending mutations, court stays, encumbrances.
  - **B. Historical lifecycle timelines**: actual days taken per legal stage vs statutory limits.
- **Target variable:** Delay = Actual Days − Statutory Days (per stage).
- Models NEVER see district/city names → cold-start for new regions works via feature
  similarity ("parcels with these features behaved like X historically").

## 3. The 5 RFCTLARR Stages + Statutory Clocks (the legal backbone)
| Stage | Plain meaning | Statutory limit |
|---|---|---|
| SIA | Social Impact Assessment study + public hearings | ~6 months (~180 days) |
| Notification (Sec 11) | Public intent to acquire | After SIA approval |
| Objections → Declaration (Sec 19) | 60-day objection window, then official declaration | ≤12 months from Notification (else LAPSES → restart) |
| Award (Sec 25) | Compensation fixed by Collector | ≤12 months from Declaration |
| Possession | Payment + handover | After award |
Delay per stage = actual − statutory. Always expose days-overrun.

## 4. Locked Decisions
- **Regional strategy:** Pan-India architecture, Himachal Pradesh as demo pilot. Model is
  region-agnostic (features only); pitch: "plug in Bhoomi/Bhulekh data to extend states."
- **Granularity:** Per-PARCEL risk scores, rolled up to village/tehsil/district views.
  Village-level rollup is the primary dashboard "portfolio" view; drill-down to parcels.
- **Synthetic data strategy:** scraping is blocked/slow → hand-sample real schemas,
  generate 100k+ parcels + ~5k historical projects with realistic embedded delay rules
  (e.g., many joint owners → consent delays; court stay → award freeze; orchard land →
  higher compensation disputes).
- **Synthetic-data defense (pitch):** schema mirrors real HimBhoomi records; delay rules
  encode documented ground realities; validation includes district hold-out (model trained
  without a district still predicts it accurately → proof of cold start); framing: "NIC
  data access requires govt approval — beyond student scope; architecture is
  production-ready once access is granted."
- **Development order:** Backend 100% solid FIRST (schema → generator → models →
  prediction contract), UI only after. predict.py emits a fixed CSV/JSON contract that
  any frontend consumes.
- **Dashboard (later):** Streamlit + Plotly, offline-capable on a laptop. Screens:
  risk table (red/yellow/green), parcel/village detail with per-stage probability bars,
  SHAP "why" chart, recommended actions panel, what-if widget ("clear the court stay →
  score drops live").
- **Explicitly NOT building:** maps/GIS, login/accounts, live scraping, databases, Docker.

## 5. Tech Stack (machine: Python 3.14, 8GB RAM, 12 cores, no Node)
- venv + pandas, numpy, scikit-learn (HistGradientBoostingClassifier/Regressor)
- SHAP (fallback: sklearn permutation_importance if Py3.14 wheels break)
- Streamlit + Plotly (UI, later)
- joblib (models), parquet (data)

## 6. Repo Layout
```
~/sih-land-delay/
├── PROJECT_CONTEXT.md      ← THIS FILE, updated every session
├── INTERFACES.md           ← predict.py output contract for frontend builders
├── schema.json             ← land record + lifecycle column definitions
├── requirements.txt
├── data/{raw_sample,generated}/
├── src/{data_generator.py, features.py, train.py, predict.py}
├── models/                 ← saved .joblib per-stage models
└── app/streamlit_app.py    ← Phase 2 only
```

## 7. Execution Plan (4 days)
- **Day 1:** schema.json (RFCTLARR-aligned) + data_generator.py with delay-rule logic →
  generate & eyeball sanity-check data.
- **Day 2:** features.py + train.py → 5 per-stage models + cold-start validation
  (district hold-out) → SHAP explanations → metrics report.
- **Day 3:** Streamlit dashboard (risk table, detail page, SHAP, actions, what-if).
- **Day 4:** End-to-end test, README + architecture diagram, rehearse demo script.

## 8. Current Status
- [x] Project folder + this file created (Day 0)
- [ ] schema.json
- [ ] data_generator.py + generated data
- [ ] features.py + train.py + models
- [ ] SHAP + cold-start validation
- [ ] INTERFACES.md (prediction contract)
- [ ] Streamlit UI
- [ ] README + demo rehearsal

## 9. Demo Script (90 seconds, for judges)
1. Show village/district risk table (red/yellow/green).
2. Click a red corridor/village → per-stage delay probability bars vs statutory clocks.
3. SHAP chart: "Delay driven by: 14 joint owners, court stay, orchard land."
4. Recommended actions panel (rule-based factor→action mapping).
5. What-if: toggle "court stay cleared" → risk drops live.
6. Cold start: brand-new district scored instantly ("region-agnostic by design").

## 10. Session Resume Protocol
Open a new chat and say:
> "Read ~/sih-land-delay/PROJECT_CONTEXT.md and continue from the Status checklist."
