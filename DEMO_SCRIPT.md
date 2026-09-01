# SIH26017 — 90-Second Demo Script (v2 multi-state)

> Numbers are **auto-sourced** — run `python src/demo_numbers.py` before the demo. Nothing
> here is hardcoded; if you regenerate data, re-run that command and the IDs/numbers update.

**Setup**

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py       # terminal 1
python src/demo_numbers.py               # print + memorize the facts below
```

## The 90-second arc

### 0–15s — Hook + Portfolio (cascading filter)
> "Land acquisition is the #1 cause of infrastructure delay. We built an early-warning
> system that flags projects **before** they slip — across 3 states."

**Click:** Portfolio → use the cascading filter (State → District → Project type).
**Say:** "Every parcel is scored RED/YELLOW/GREEN. Here are **12,013 live parcels** —
4,201 RED, 5,354 YELLOW. ~29% of in-progress stages are *already* past their legal limit."

### 15–35s — Drill into a project (project dashboard)
**Click** a project row → Project detail.
**Point at:** summary strip, per-stage bottleneck bars, segment (village) profile.
**Say:** "This is the project's weak point — which stage, which village cluster."

### 35–55s — The money-shot (parcel + what-if)
**Click** a parcel → Detail. Then **What-if** → toggle court stay → off.
**Say:** "Clearing the court stay drops it from **RED → GREEN** live (e.g. 0.88 → 0.36)."

### 55–75s — Area of Interest + onboarding
**Click:** Area of Interest → pick a village + radius → catchment risk.
**Then** New Project → upload a CSV of parcel IDs → instant scoring.
**Say:** "Officials define a new site two ways: a radius around a village, or a precise
CSV of khasra IDs — scored instantly, persisted, tagged 'user-created'."

### 75–90s — Map + cold-start close
**Click:** Map → point markers + linear corridors colored by risk.
**Say:** "The model never sees geography. **Leave-one-state-out: held-out states predicted
within ~2.5% of in-sample accuracy** — cold-start by design. Plug in Bhoomi/Bhulekh data
and this scales to any state."

**Close:** "Reactive monitoring becomes predictive governance."

## Demo facts (auto-sourced, paste from `demo_numbers.py`)
- Live: 12,013 parcels / 3 states · RED 4,201 · YELLOW 5,354 · ~28.6% overrun-while-ongoing.
- AUROC: SIA 0.772 · NOTIF 0.869 · DECL 0.915 · AWARD 0.902 · POSS 0.907.
- LOSO drop: SIA −6.0% … POSS −1.2% (avg 2.5%).
- Point dam + cross-state road/highway project IDs (auto-picked).

## Fallbacks
| Risk | Fallback |
|---|---|
| API won't start | Dashboard uses predict.py directly (no API needed). |
| No internet | Everything offline (map is lat/lon scatter, no tiles). |
| Numbers look stale | Re-run `python src/demo_numbers.py` — never quote hand-typed figures. |
| Click doesn't navigate | Use the parcel/project dropdown pickers, or the sidebar nav. |
