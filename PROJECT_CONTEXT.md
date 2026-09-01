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
- **GIS (LOCKED — offline plotly):** offline Plotly lat/lon scatter of district risk
  hotspots (chosen over Folium to stay fully offline — no tile server). Covers statement
  deliverable #7.
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
- Streamlit + Plotly (UI + offline risk map)
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
  page, SHAP, actions, what-if, alerts feed, offline plotly map, mock role-switcher).
- **Day 4:** End-to-end test, README + architecture diagram, INTERFACES.md (contract +
  API spec), rehearse demo script.

## 8. Current Status
**Phase 1 (Day 1–4, HP single-state) — DONE and committed.**
**Phase 2 (Architecture Expansion — multi-state) — Stages 0–5 DONE, Stage 6 final gate.**

- [x] Stage 0 — schema.json v2 (semantic IDs, spatial_type, coord_path, state+district hidden confound) + states.py
- [x] Stage 1 — generator rebuild (3 states / 48 districts, semantic IDs, point+linear adjacency routing) + calibrated single regeneration
- [x] Stage 1.5 — LOSO calibration: K_STATE=35, ADMIN_EFFECT=30, W=(0.50,0.28,0.22)
- [x] Stage 2 — retrain + LODO/LOSO gates (LOSO avg 2.5%, all gates pass)
- [x] Stage 3 — dashboard: cascading filter, clickable tables, project detail, point/linear map
- [x] Stage 4 — new-project onboarding (CSV + record pull + parquet persistence)
- [x] Stage 5 — Area of Interest + incremental hook
- [x] Stage 6 — docs (README/INTERFACES/DEMO_SCRIPT) + e2e refresh
- [x] Stage 6 — fresh-state e2e gate (38/38) + final commit (6a4be02)

### ⚠️ Historical Phase-1 results (HP single-state — SUPERSEDED by Phase 2)
The Day 1–4 blocks below record the ORIGINAL single-state build. After Phase 2 the data
was regenerated (3 states, semantic IDs); these numbers/IDs are no longer current.
Current results live in `models/metrics_report.json` + `src/demo_numbers.py`.

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
- Refinement (pre-Day-2): stakeholder_responsiveness + historical_performance_score are now
  MODERATE partial proxies of the hidden confound (corr ~0.27 with district, band 0.15–0.70)
  so cold-start is learnable but not leaked. Sanity check [4] enforces the band.

### Day 2 results (verified in metrics_report.json)
- 10 models trained (5 stages x classifier+regressor) on 88,041 historical parcels / 12 features.
- In-sample AUROC by stage: SIA 0.76, NOTIFICATION 0.86, DECLARATION 0.91, AWARD 0.91,
  POSSESSION 0.91. Regressor MAE ~17 days, RMSE ~22 days.
- Cold-start (leave-one-district-out) drop is small & believable: SIA -3.9%, NOTIFICATION
  -3.0%, DECLARATION -1.3%, AWARD -0.6%, POSSESSION -0.7% → model generalizes to unseen
  districts via feature similarity (the pitch proof).
- SHAP top features align with embedded rules (AWARD: encumbrances/land_class/court_stay;
  DECLARATION: owner_count/affected_families; POSSESSION: compensation_status/court_stay).
- Risk-score fix: max-prob saturated (delay near-universal) → headline risk_score is now
  severity-based (1-exp(-overrun/250)) with clean spread (median 0.55); max_delay_prob kept
  for per-stage bars. District ranking: Mandi 0.64 (worst) … Sirmaur 0.40 (best).

### Day 3 results (verified via AppTest + curl)
- predict.py contract works end-to-end: score_parcel emits risk_score/level,
  expected_overrun_days, max_delay_prob, per-stage delay_prob + overrun + statutory, signed
  SHAP top_factors (global + per-stage), rule-based actions, overrun_while_ongoing_days.
- portfolio cache: 11,959 live parcels scored in 0.37s (live refresh is demo-able).
- FastAPI: /health, /predict, /predict/batch all verified via curl (court-stay+orchard+dam
  parcel → risk 0.97 RED, 909 expected overrun days).
- Streamlit dashboard: all 6 views render with 0 exceptions (AppTest). Viewer role correctly
  hides What-if/Alerts (nav → Portfolio/Detail/Map only). Map is offline Plotly lat/lon scatter.
- Folium dropped from requirements; GIS decision updated to offline plotly.

### Day 4 results (verified by e2e_test.py)
- bootstrap.sh: fresh-clone one-command setup (venv → install → data → train → portfolio).
- e2e_test.py: 25/25 checks pass — artifacts, contract, FastAPI (TestClient), dashboard
  (all views + Viewer gating), AND a fresh-clone proof (copy code-only → bootstrap →
  regenerated data+models+portfolio → models score).
- demo_numbers.py: demo facts auto-sourced from metrics_report.json + portfolio (no stale
  hardcoded numbers). Curated money-shot parcel PRCL_0000006: court stay + 124d overrun
  ongoing at Award → clear stay flips RED 0.85 → GREEN 0.23.
- README.md (mermaid + ASCII diagram, results, defenses) + DEMO_SCRIPT.md (90s timed).
- models/ committed (removed from .gitignore) so the demo works out-of-the-box after clone.

## 9. Demo Script (90 seconds, for judges)
**Authoritative walkthrough: see `DEMO_SCRIPT.md`** (timed, auto-sourced numbers, curated
point-dam + cross-state-highway parcels, fallbacks). Short arc:
1. Portfolio — cascading filter (State → District → Type) + risk table (R/Y/G).
2. Project detail — per-stage bottleneck bars + segment profile.
3. Parcel detail + What-if — "clear the court stay" → risk drops live.
4. New Project — CSV onboarding + instant scoring.
5. Area of Interest — village center + radius catchment.
6. Map — point markers + linear corridors; then cold-start close (LOSO proof).

## 10. Session Resume Protocol
Open a new chat and say:
> "Read ~/sih-land-delay/PROJECT_CONTEXT.md and continue from the Status checklist."

## 11. Environment & Run Credentials (setup is done — reuse, don't redo)
- **Python:** 3.14.4 (system). No `pip`/`venv`/`ensurepip` on the machine → use **uv**.
- **uv:** installed at `~/.local/bin/uv`; add to PATH via `export PATH="$HOME/.local/bin:$PATH"`.
- **venv:** `~/sih-land-delay/.venv` (created with `uv venv .venv`). Activate:
  `source .venv/bin/activate` (or call `.venv/bin/python` directly).
- **Installed deps (locked versions):** pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
  shap 0.52.0, pyarrow 25.0.1, joblib 1.6.0, streamlit 1.62.0, plotly 7.0.0,
  fastapi 0.141.1, uvicorn 0.52.4, httpx 0.28.1. (Folium dropped — map is offline plotly.)
- **Seed:** `SEED = 42` in `data_generator.py` (fully reproducible).
- **Run commands (from repo root):**
  - regen data: `.venv/bin/python src/data_generator.py`
  - sanity check: `.venv/bin/python src/sanity_check.py`
- **Data outputs:** `data/generated/*.parquet` (historical = training, live = demo).
- **Git ignore:** `.venv/`, `__pycache__/`, `*.pyc`, `data/raw_sample/` (models/ and
  data/generated are committed — ~12MB, makes demo work out-of-the-box).

## 12. Post-Demo Enhancement Plan (AUTHORITATIVE — active)
Supersedes the Day 1–4 work. Overhauls the data architecture: pan-India (3 states),
semantic IDs, point/linear spatial realism, and leave-one-state-out (LOSO) validation.
Execute in order (dependency-sequenced). See README/DEMO_SCRIPT after Stage 6.

### Locked decisions
1. **Confound:** hidden state + district admin capacity; stakeholder_responsiveness &
   historical_performance_score proxy BOTH; calibrated on a small sample before the one
   full regeneration.
2. **IDs:** parcel `<STATE>-<DISTRICT_CODE>-<VILLAGE_CODE>-<KHASRA_NO>`; project
   `<STATE>-<TYPE>-<YEAR>-<SEQ>`; cross-state project's STATE = home state (first path
   parcel). IDs derived from assigned geography (never independent).
3. **Spatial:** point projects (1 district, 1–2 villages); linear projects via
   centroid-adjacency routing (nearest-village walk across district/state borders).
4. **Gates:** LODO drop ≤10% rel + district AUROC ≥0.70; LOSO drop 2–15% rel. Fail ⇒
   return to Stage 0.
5. **UI:** all dashboard UI in one pass (Stage 3) after regeneration.
6. **Scale:** ~100k parcels / ~5k projects across 3 states (~33k each), modest linear
   cross-state fraction.
7. **Onboarding:** CSV + record pull + reject-unknown; dashboard-only (no new API
   routes); parquet persistence (no DB).
8. **Dropped:** corridor/route tracing, site comparator, GeoJSON, simulate-unknown IDs,
   real GIS.

### Stages
- **Stage 0 — Lock design:** schema.json (IDs, spatial_type, coord_path, hidden
  state_admin_capacity); states dict (HP 12 / Punjab 23 / Uttarakhand 13 districts, 48
  total); state-confound formula (state_effect = (1-state_cap)*K_state; proxy blend
  w_ind*base + w_state*state_cap + w_dist*district_cap + noise); cold-start split by
  parcel's own district/state. No data touched.
- **Stage 1 — Generator + calibration + single regen:** states hierarchy, semantic IDs,
  point/linear geometry (adjacency routing); ~12k calibration sample → tune K_state +
  blend weights until LOSO 2–15%; then ONE clean ~100k regeneration; extend sanity_check.
- **Stage 2 — Retrain + gates:** 10 models; LODO (48 districts) ≤10% + AUROC≥0.70;
  LOSO (3-fold) 2–15%; metrics_report with both. Fail ⇒ stop.
- **Stage 3 — Dashboard:** score_batch(include_stages=True); cascading filter
  State→District→Type→Project (+risk/village); clickable tables; project detail dashboard;
  map upgrade (point markers + linear polylines, click→project).
- **Stage 4 — Onboarding:** user_projects/user_parcels parquet; New Project form; CSV
  semantic-ID upload + record pull + reject-unknown; instant score; "user-created" tag.
- **Stage 5 — Time-boxed:** Area of Interest (map-click center + radius); --incremental
  lifecycle doc.
- **Stage 6 — Docs/demo/e2e:** update PROJECT_CONTEXT/README/INTERFACES; repick curated
  parcels (point dam + cross-state highway); demo_numbers += LOSO; rewrite DEMO_SCRIPT;
  extend e2e; fresh-state gate.

### Accepted consequences
- All current baked numbers/IDs replaced (Mandi 0.64, PRCL_0000006, LODO %s).
- Runtime: LODO ~3–5 min; fresh e2e ~8–12 min (--skip-lodo for iteration).
- No new dependencies (Area of Interest = haversine).
