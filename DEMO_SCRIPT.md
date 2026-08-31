# SIH26017 — 90-Second Demo Script

> Numbers below are **auto-sourced** from `metrics_report.json` + `portfolio_scores.parquet`.
> Re-run `python src/demo_numbers.py` before the demo to refresh them (they will never be
> silently stale — if you regenerate data, re-run that command).

**Setup before the demo**

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py       # terminal 1 (dashboard)
# optional: uvicorn app.api:app --port 8000  # terminal 2 (API, bonus)
python src/demo_numbers.py               # print + memorize the facts below
```

---

## The 90-second arc

### 0–15s — Hook + portfolio (risk table)
> "Land acquisition is the #1 cause of infrastructure delay. We built an early-warning
> system that flags projects **before** they slip."

**Click:** sidebar **Portfolio** (default). Group by **district**.

**Say:** "Every live parcel is scored. RED = act now, YELLOW = watch, GREEN = on track."
- 11,959 live parcels · RED 3,084 / YELLOW 5,388 / GREEN 3,487.
- Worst district **Mandi 0.64**, best **Sirmaur 0.40**.

### 15–35s — Drill into a parcel (per-stage bars + SHAP "why")
**Click:** a RED district → a RED parcel, or use **Detail** and pick `PRCL_0000006` (Shimla).

**Say:** "This parcel has an active **court stay**, and it's **already 124 days past** its
Award statutory limit — while still marked 'ongoing'." (the overrun-while-ongoing flag)

**Point at:** the per-stage delay-probability bars (blue) vs the statutory clock (dotted),
and the SHAP bars on the right.

**Say:** "The model explains *why*: the court stay is the dominant driver."

### 35–50s — Recommended actions + what-if (the money shot)
**Say:** "It also tells us what to do." (point at the actions panel).

**Click:** sidebar **What-if** (parcel `PRCL_0000006` already selected).

**Toggle:** court stay → off.

**Say:** "Clear the court stay and the risk **drops from 0.85 RED to 0.23 GREEN** live —
overrun forecast falls from ~477 to ~64 days." *(numbers auto-sourced)*

### 50–70s — Offline map + alerts
**Click:** sidebar **Map**.

**Say:** "Hotspots roll up to a district map — fully offline, no tile server." (point at
Mandi / Solan / Bilaspur as the warm spots.)

**Click:** sidebar **Alerts**.

**Say:** "Every high-risk parcel auto-raises alerts — ~25% of in-progress stages are
already past their legal limit." (3,013 of 11,959)

### 70–90s — Cold-start + API (the pitch close)
**Say:** "The model never sees district names. Trained without a district, it still
predicts it within ~1–4% of in-sample accuracy — cold-start by design." (AUROC drop
SIA −3.9% … POSSESSION −0.7%.)

**Optional (bonus):** show `localhost:8000/docs` → POST `/predict` with a JSON parcel →
same contract. Then flip the role-switcher to **Viewer** to show read-only gating.

**Close:** "Reactive monitoring becomes predictive governance. Architecture is
production-ready — plug in Bhoomi/Bhulekh data to extend to any state."

---

## Fallbacks

| Risk | Fallback |
|---|---|
| API won't start | The dashboard uses `predict.py` directly (no API needed). Demo the API only if time. |
| No internet | Everything is offline: dashboard, map (lat/lon scatter), models, data. |
| What-if looks stuck | Ensure the parcel has `court_stay=1` (toggle is only meaningful then). Use `PRCL_0000006`. |
| Numbers look off vs live dashboard | Re-run `python src/demo_numbers.py`; never quote hardcoded figures. |

## Curated parcels (re-check after regeneration)

| parcel_id | story |
|---|---|
| `PRCL_0000006` (Shimla) | court stay + overrun-while-ongoing (124d past Award limit), RED 0.85 → clear stay → GREEN 0.23 |
| `PRCL_0094902` (Mandi) | court stay + pending compensation + 8 owners, RED 0.98 (severity example) |
