# SIH26017 — Predictive Analytics for Early Detection of Land Acquisition Delays
**Project Context & Decision Log — updated every session. Any AI/human: read this first.**

## 1. Problem (one paragraph)
Ministry of Rural Development PS SIH26017: build an AI-powered early-warning system that
predicts which land acquisition projects will get delayed, using (a) static land-record
features and (b) historical stage-wise timelines under the RFCTLARR Act, 2013. Output per
parcel (rolled to project/village/district/state): risk score, per-stage delay
probability, explainable contributing factors, and recommended actions.

## 2. Core Architecture (LOCKED)
- Two data types merged:
  - **A. Static land features** (HimBhoomi-style): khasra number, owner count, land class
    (orchard/barren/agri/residential), area, pending mutations, court stays, encumbrances.
  - **B. Historical lifecycle timelines**: actual days taken per legal stage vs statutory limits.
- **Project/R&R features (NEW, from statement):** project_type (road/rail/irrigation/dam…),
  affected_families, compensation_status, rehab_progress_pct, stakeholder_responsiveness,
  historical_performance_score. These sit at the project level and join down to parcels.
- **Target variable:** Delay = Actual Days − Statutory Days (per stage).
- Models NEVER see district/city names → cold-start for new regions works via feature
  similarity ("parcels with these features behaved like X historically").
- **Granularity hierarchy (LOCKED):** parcel (atomic) → **project** → village → tehsil →
  district → state. Parcel is the atomic prediction unit; project-level risk is a rollup of
  its parcels (satisfies "project-wise scoring" in the statement). Rollups use weighted
  aggregation (e.g., parcel area × risk).

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
- **Granularity:** Per-PARCEL risk scores, rolled up parcel → project → village → tehsil →
  district → state. Village-level rollup is the primary dashboard "portfolio" view;
  drill-down to project → parcels.
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
- **GIS (NEW — supersedes "no maps"):** lightweight Folium map (pure-Python, offline)
  showing district/state risk hotspots. Covers statement deliverable #7 without
  Node/Docker/GIS servers.
- **Alerts (NEW):** rule-based alert feed, e.g. red project > threshold days overrun,
  active court stay, award pending past statutory clock. No notification transport —
  surface in-dashboard as a feed.
- **API (NEW):** FastAPI endpoint wrapping predict.py (POST → risk JSON) alongside the
  CSV/JSON contract. Covers statement deliverable #11.
- **Auth/RBAC/audit (DEFERRED):** documented out-of-scope; UI gets a mock role-switcher
  (Admin/Officer/Viewer) for demo flavor. Real RBAC + audit trails noted as production
  extension. Covers statement deliverable #12 nominally.
- **Continuous learning (NEW):** explicit incremental retraining path — `train.py
  --incremental` refits on new records; pitch: "updates as new project data arrives".
- **Explicitly NOT building:** live scraping, databases, Docker, real auth/audit.

## 5. Tech Stack (machine: Python 3.14, 8GB RAM, 12 cores, no Node)
- venv + pandas, numpy, scikit-learn (HistGradientBoostingClassifier/Regressor)
- SHAP (fallback: sklearn permutation_importance if Py3.14 wheels break)
- Streamlit + Plotly (UI, later)
- Folium (GIS hotspot map)
- FastAPI + uvicorn (prediction API)
- joblib (models), parquet (data)

## 6. Repo Layout
```
~/sih-land-delay/
├── PROJECT_CONTEXT.md      ← THIS FILE, updated every session
├── INTERFACES.md           ← predict.py output contract + API spec
├── schema.json             ← land record + lifecycle column definitions
├── requirements.txt
├── data/{raw_sample,generated}/
├── src/{data_generator.py, features.py, train.py, predict.py}
├── models/                 ← saved .joblib per-stage models
└── app/{streamlit_app.py, api.py}   ← Phase 2 only (FastAPI beside Streamlit)
```

## 7. Execution Plan (4 days)
- **Day 1:** schema.json (RFCTLARR-aligned + project/R&R/compensation/rehab params) +
  data_generator.py with delay-rule logic (incl. project-level aggregation) →
  generate & eyeball sanity-check data.
- **Day 2:** features.py + train.py → 5 per-stage models + project rollup + cold-start
  validation (district hold-out) → SHAP explanations → metrics report. Add
  `--incremental` retrain path.
- **Day 3:** predict.py + FastAPI endpoint + Streamlit dashboard (risk table, detail
  page, SHAP, actions, what-if, alerts feed, Folium map, mock role-switcher).
- **Day 4:** End-to-end test, README + architecture diagram, INTERFACES.md (contract +
  API spec), rehearse demo script.

## 8. Current Status
- [x] Project folder + this file created (Day 0)
- [x] Scope analysis vs official statement + decisions locked (project rollup, Folium, alerts, FastAPI, mock RBAC, incremental training)
- [x] schema.json (3 tables + villages + hidden admin_capacity confound, all fields tagged)
- [x] data_generator.py + generated data (100k parcels / 5k projects / 227 villages + live subset)
- [x] sanity_check.py (volumes, rule recovery, confound leak check, overrun-while-ongoing eyeball)
- [ ] features.py + train.py + models
- [ ] SHAP + cold-start validation
- [ ] INTERFACES.md (prediction contract + API spec)
- [ ] Streamlit UI (incl. Folium map, alerts feed, role-switcher)
- [ ] FastAPI endpoint
- [ ] README + demo rehearsal

### Day 1 results (verified by sanity_check.py)
- Generated: 12 HP districts → 227 villages (40 tehsils) → 5,000 projects → 100,000
  parcels (20/project avg) → 500,000 historical timeline rows + 12,005 live parcels.
- Delay rules recoverable: court_stay → award delay 283d vs 41d base; orchard → 156d
  vs 35d; owner_count>4 → declaration 61d vs 27d.
- Hidden confound works without leaking: district mean delays span 27→63 days, max
  feature↔district correlation 0.14 (no visible column encodes admin_capacity).
- Balanced targets: delay-flag rate 58–88% per stage (fixed from ~98% over-positive).
- Money-shot ready: ~25% of ongoing live stages already past statutory deadline.
- Fixed realism: parcels spread across all 227 villages (was hardcoded to 1 village/district).

## 9. Demo Script (90 seconds, for judges)
1. Show village/district risk table (red/yellow/green).
2. Click a red corridor/village → per-stage delay probability bars vs statutory clocks.
3. SHAP chart: "Delay driven by: 14 joint owners, court stay, orchard land."
4. Recommended actions panel (rule-based factor→action mapping).
5. What-if: toggle "court stay cleared" → risk drops live.
6. Cold start: brand-new district scored instantly ("region-agnostic by design").
7. Folium hotspot map + alert feed ("high-risk projects auto-flagged").
8. FastAPI endpoint demo + mock role-switch (Admin/Officer/Viewer).

## 10. Session Resume Protocol
Open a new chat and say:
> "Read ~/sih-land-delay/PROJECT_CONTEXT.md and continue from the Status checklist."

## 11. Environment & Run Credentials (setup is done — reuse, don't redo)
- **Python:** 3.14.4 (system). No `pip`/`venv`/`ensurepip` on the machine → use **uv**.
- **uv:** installed at `~/.local/bin/uv`; add to PATH via `export PATH="$HOME/.local/bin:$PATH"`.
- **venv:** `~/sih-land-delay/.venv` (created with `uv venv .venv`). Activate:
  `source .venv/bin/activate` (or call `.venv/bin/python` directly).
- **Installed deps (locked versions):** pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
  shap 0.52.0, pyarrow 25.0.1, joblib 1.6.0. (Streamlit/Plotly/Folium/FastAPI not yet
  installed — add on Day 3.)
- **Seed:** `SEED = 42` in `data_generator.py` (fully reproducible).
- **Run commands (from repo root):**
  - regen data: `.venv/bin/python src/data_generator.py`
  - sanity check: `.venv/bin/python src/sanity_check.py`
- **Data outputs:** `data/generated/*.parquet` (historical = training, live = demo).
- **Git ignore:** add `.gitignore` for `.venv/`, `__pycache__/`, `models/`, `*.pyc`
  (generated data is committed — only ~7MB, makes demo work out-of-the-box).
