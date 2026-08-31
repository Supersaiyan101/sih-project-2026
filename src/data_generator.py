"""SIH26017 synthetic data generator.

Generates a HimBhoomi-style land-record dataset + RFCTLARR lifecycle timelines
with realistic embedded delay rules. Emits two datasets:

  1. historical  - all parcels completed (training data, full delay labels)
  2. live        - in-progress parcels (early-warning demo: mixed status,
                   NULL future delays, some ongoing stages already overrun)

Design notes (see PROJECT_CONTEXT.md):
  - Models must never see geo names; the generator keeps them for rollup/validation.
  - district `admin_capacity` is a HIDDEN confound: it drives between-district
    delay variance and is partially proxied by parcel/project features (so cold-start
    via feature similarity is learnable but not trivially ~100%).
  - Seeded for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 42
N_PROJECTS = 5000
N_PARCELS = 100_000
LIVE_FRACTION = 0.12

STAGES = ["SIA", "NOTIFICATION", "DECLARATION", "AWARD", "POSSESSION"]
STATUTORY = {"SIA": 180, "NOTIFICATION": 60, "DECLARATION": 365, "AWARD": 365, "POSSESSION": 90}

PROJECT_TYPES = ["road", "rail", "irrigation", "dam", "industrial"]
PROJECT_TYPE_W = [0.30, 0.15, 0.20, 0.15, 0.20]  # sampling weights
# Base slowness (extra days per stage) by project type: dam > irrigation > industrial > rail > road
PROJECT_PACE = {"road": 0.0, "rail": 8.0, "industrial": 14.0, "irrigation": 20.0, "dam": 35.0}

LAND_CLASSES = ["agri", "orchard", "barren", "residential"]
LAND_CLASS_W = [0.45, 0.20, 0.20, 0.15]

COMPENSATION_STATUSES = ["paid", "partial", "pending"]
COMPENSATION_W = [0.60, 0.25, 0.15]

# Real Himachal Pradesh districts + approximate centroid lat/lon for the map.
HP_DISTRICTS = {
    "Bilaspur": (31.33, 76.75),
    "Chamba": (32.55, 76.13),
    "Hamirpur": (31.68, 76.52),
    "Kangra": (32.10, 76.27),
    "Kinnaur": (31.60, 78.40),
    "Kullu": (31.96, 77.11),
    "Lahaul and Spiti": (32.50, 77.60),
    "Mandi": (31.71, 76.93),
    "Shimla": (31.10, 77.17),
    "Sirmaur": (30.75, 77.55),
    "Solan": (30.90, 77.10),
    "Una": (31.47, 76.27),
}
STATE = "Himachal Pradesh"

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clip_int(x, lo, hi):
    return int(round(float(np.clip(x, lo, hi))))


# --------------------------------------------------------------------------- #
# Stage 1: districts + villages (with hidden admin_capacity)
# --------------------------------------------------------------------------- #

def generate_districts(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for name, (lat, lon) in HP_DISTRICTS.items():
        # Hidden confound: administrative capacity. Low capacity -> slower pipeline.
        # Sampled independently of visible features; its EFFECT is partially proxied
        # via shifted feature distributions in generate_parcels().
        capacity = float(rng.beta(3.0, 2.0))  # ~0.4-1.0, mean ~0.6
        rows.append({"state": STATE, "district": name, "lat": lat, "lon": lon,
                     "admin_capacity": capacity})
    return pd.DataFrame(rows)


def generate_villages(districts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n_tehsil = 0
    for _, d in districts.iterrows():
        n_tehsils = int(rng.integers(2, 5))
        for t in range(n_tehsils):
            tehsil = f"{d['district']}_Tehsil_{t + 1}"
            n_villages = int(rng.integers(3, 9))
            for v in range(n_villages):
                village = f"{d['district']}_Vill_{n_tehsil + 1}"
                n_tehsil += 1
                # jitter around district centroid
                lat = d["lat"] + rng.normal(0, 0.15)
                lon = d["lon"] + rng.normal(0, 0.15)
                rows.append({"village": village, "tehsil": tehsil,
                             "district": d["district"], "state": STATE,
                             "lat": float(lat), "lon": float(lon)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage 2: projects
# --------------------------------------------------------------------------- #

def generate_projects(districts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = N_PROJECTS
    district_names = districts["district"].tolist()
    capacity_by_district = dict(zip(districts["district"], districts["admin_capacity"]))

    project_type = rng.choice(PROJECT_TYPES, size=n, p=PROJECT_TYPE_W)

    # affected_families: lognormal, scaled by project type (dams/irrigation displace more)
    fam_mult = {"road": 1.0, "rail": 1.3, "industrial": 0.8, "irrigation": 1.8, "dam": 2.5}
    base_fam = rng.lognormal(mean=4.5, sigma=0.9, size=n)
    affected = (base_fam * np.array([fam_mult[t] for t in project_type])).astype(int)
    affected = np.clip(affected, 0, 5000)

    compensation = rng.choice(COMPENSATION_STATUSES, size=n, p=COMPENSATION_W)

    # rehab progress: higher when fewer families, lower when more families / worse compensation
    rehab = 100.0 - (affected / 5000.0) * 40.0 - rng.normal(10, 8, size=n)
    rehab = np.clip(rehab, 0, 100)

    stakeholder = np.clip(rng.beta(4, 2, size=n), 0, 1)  # mostly responsive
    hist_perf = np.clip(rng.beta(4, 2, size=n), 0, 1)

    # geo: assign project to a district + tehsil (tehsil derived in villages; here reuse district)
    district = rng.choice(district_names, size=n)

    rows = {
        "project_id": [f"PRJ_{i:05d}" for i in range(n)],
        "project_type": project_type,
        "affected_families": affected,
        "compensation_status": compensation,
        "rehab_progress_pct": np.round(rehab, 2),
        "stakeholder_responsiveness": np.round(stakeholder, 3),
        "historical_performance_score": np.round(hist_perf, 3),
        "state": STATE,
        "district": district,
        "tehsil": [f"{d}_Tehsil_1" for d in district],
    }
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage 3: parcels (feature distributions shifted by hidden admin_capacity)
# --------------------------------------------------------------------------- #

def generate_parcels(projects: pd.DataFrame, districts: pd.DataFrame,
                     villages: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = N_PARCELS
    capacity_by_district = dict(zip(districts["district"], districts["admin_capacity"]))

    # sample a project per parcel (weighted so more parcels in bigger projects is implicit)
    proj_idx = rng.integers(0, len(projects), size=n)
    proj = projects.iloc[proj_idx].reset_index(drop=True)

    # hidden confound proxy: low-capacity districts have more litigation-heavy parcels
    cap = np.array([capacity_by_district[d] for d in proj["district"]])
    low_cap = 1.0 - cap  # 0 = great admin, 1 = poor admin

    # court_stay probability scales with low admin capacity (+ base rate)
    p_stay = np.clip(0.03 + 0.14 * low_cap + rng.normal(0, 0.02, size=n), 0.01, 0.6)
    court_stay = (rng.random(n) < p_stay).astype(int)

    # owner_count: more joint owners in low-capacity districts (documentation lag)
    owner_count = np.clip(
        rng.poisson(3.0 + 4.0 * low_cap, size=n) + 1, 1, 30)

    # pending mutations: higher in low-capacity districts
    pending_mutations = np.clip(
        rng.poisson(0.5 + 2.5 * low_cap, size=n), 0, 10)

    # encumbrances
    encumbrances = np.clip(rng.poisson(0.4 + 1.2 * low_cap, size=n), 0, 5)

    land_class = rng.choice(LAND_CLASSES, size=n, p=LAND_CLASS_W)

    # area: lognormal, scaled by land class (agri/orchard large, residential small)
    area_mult = {"agri": 1.0, "orchard": 0.7, "barren": 1.5, "residential": 0.15}
    area = rng.lognormal(mean=7.5, sigma=1.0, size=n) * np.array([area_mult[c] for c in land_class])
    area = np.clip(area, 20, 1_000_000)

    # live flag: ~LIVE_FRACTION of parcels are in-progress
    is_live = (rng.random(n) < LIVE_FRACTION).astype(int)

    # assign each parcel to a real village (and its tehsil) within its district
    vill_lookup = {d: g[["village", "tehsil"]].to_numpy() for d, g in villages.groupby("district")}
    district = proj["district"].tolist()
    vidx = [int(rng.integers(0, len(vill_lookup[d]))) for d in district]
    village = [vill_lookup[d][i, 0] for d, i in zip(district, vidx)]
    tehsil = [vill_lookup[d][i, 1] for d, i in zip(district, vidx)]

    rows = {
        "parcel_id": [f"PRCL_{i:07d}" for i in range(n)],
        "project_id": proj["project_id"].tolist(),
        "khasra_number": [f"KHS_{i:07d}" for i in range(n)],
        "owner_count": owner_count,
        "land_class": land_class,
        "area_sqm": np.round(area, 1),
        "pending_mutations": pending_mutations,
        "court_stay": court_stay,
        "encumbrances": encumbrances,
        "village": village,
        "tehsil": tehsil,
        "district": district,
        "state": STATE,
        "is_live": is_live,
    }
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage 4: delay rules engine -> full completed timelines
# --------------------------------------------------------------------------- #

def _compute_delays(projects: pd.DataFrame, parcels: pd.DataFrame,
                    districts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Compute delay_days (actual - statutory) per parcel per stage.

    Returns long-format DataFrame with per-stage delay for every parcel (all completed).
    """
    n = len(parcels)
    proj_lookup = projects.set_index("project_id")
    capacity_by_district = dict(zip(districts["district"], districts["admin_capacity"]))

    proj_id = parcels["project_id"].tolist()
    p_type = proj_lookup.loc[proj_id, "project_type"].to_numpy()
    affected = proj_lookup.loc[proj_id, "affected_families"].to_numpy()
    compensation = proj_lookup.loc[proj_id, "compensation_status"].to_numpy()
    rehab = proj_lookup.loc[proj_id, "rehab_progress_pct"].to_numpy()
    stakeholder = proj_lookup.loc[proj_id, "stakeholder_responsiveness"].to_numpy()
    hist_perf = proj_lookup.loc[proj_id, "historical_performance_score"].to_numpy()

    owner_count = parcels["owner_count"].to_numpy()
    land_class = parcels["land_class"].to_numpy()
    pending_mut = parcels["pending_mutations"].to_numpy()
    court_stay = parcels["court_stay"].to_numpy()
    encumbrances = parcels["encumbrances"].to_numpy()
    district = parcels["district"].to_numpy()
    admin_cap = np.array([capacity_by_district[d] for d in district])

    pace = np.array([PROJECT_PACE[t] for t in p_type])

    # per-parcel latent "luck" (unexplained heterogeneity); slight negative mean so
    # clean parcels can finish on time or early
    parcel_luck = rng.normal(-12, 16, size=n)

    # global slowness from project-level responsiveness/perf
    global_slow = (1.0 - stakeholder) * 20.0 + (1.0 - hist_perf) * 15.0

    # hidden confound effect: low admin capacity adds delay (district-level, not in features)
    admin_effect = (1.0 - admin_cap) * 40.0

    # baseline efficiency offset: centers a clean parcel near zero delay (on-time)
    BASELINE = 30.0

    orchard = (land_class == "orchard")
    compensation_unpaid = (compensation != "paid")
    many_owners = (owner_count > 4).astype(float)
    fam_load = affected / 200.0
    rehab_shortfall = (100.0 - rehab) / 100.0

    def _add_noise():
        return rng.normal(0, 14, size=n)

    delays = {}
    # SIA
    delays["SIA"] = (pace + global_slow + admin_effect + parcel_luck
                     + (1.0 - stakeholder) * 30.0 + _add_noise() - BASELINE)
    # NOTIFICATION
    delays["NOTIFICATION"] = (pace + global_slow + admin_effect + parcel_luck
                              + pending_mut * 15.0 + (1.0 - stakeholder) * 25.0 + _add_noise() - BASELINE)
    # DECLARATION
    delays["DECLARATION"] = (pace + global_slow + admin_effect + parcel_luck
                             + many_owners * (owner_count - 4.0) * 12.0
                             + pending_mut * 10.0 + fam_load * 20.0 + _add_noise() - BASELINE)
    # AWARD
    delays["AWARD"] = (pace + global_slow + admin_effect + parcel_luck
                       + court_stay * 240.0 + orchard * 120.0
                       + encumbrances * 25.0 + _add_noise() - BASELINE)
    # POSSESSION
    delays["POSSESSION"] = (pace + global_slow + admin_effect + parcel_luck
                            + court_stay * 180.0 + compensation_unpaid * 150.0
                            + rehab_shortfall * fam_load * 60.0 + _add_noise() - BASELINE)

    records = []
    for si, stage in enumerate(STAGES):
        d = np.rint(delays[stage]).astype(int)
        actual = np.clip(STATUTORY[stage] + d, 10, 2000)
        delay = actual - STATUTORY[stage]
        for i in range(n):
            records.append({
                "parcel_id": parcels["parcel_id"].iloc[i],
                "stage": stage,
                "statutory_days": STATUTORY[stage],
                "actual_days": int(actual[i]),
                "elapsed_days": int(actual[i]),
                "status": "completed",
                "delay_days": int(delay[i]),
                "delay_flag": int(delay[i] > 0),
            })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Stage 5: build live (in-progress) view from completed timelines
# --------------------------------------------------------------------------- #

def build_live_timelines(parcels: pd.DataFrame, completed: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Truncate completed timelines into in-progress views for live parcels."""
    live_parcels = parcels[parcels["is_live"] == 1]
    live_ids = set(live_parcels["parcel_id"])

    live_df = completed[completed["parcel_id"].isin(live_ids)].copy()

    # current stage per live parcel (weighted toward mid-pipeline)
    stage_weights = [0.10, 0.15, 0.25, 0.30, 0.20]
    cur_stage = {pid: int(rng.choice(len(STAGES), p=stage_weights)) for pid in live_ids}

    rows = []
    for pid, grp in live_df.groupby("parcel_id", sort=False):
        grp = grp.sort_values("stage", key=lambda s: s.map({x: i for i, x in enumerate(STAGES)}))
        cur = cur_stage[pid]
        for i, (_, r) in enumerate(grp.iterrows()):
            rec = {
                "parcel_id": pid,
                "stage": r["stage"],
                "statutory_days": r["statutory_days"],
                "actual_days": None,
                "elapsed_days": None,
                "status": None,
                "delay_days": None,
                "delay_flag": None,
            }
            if i < cur:
                # already completed
                rec.update(actual_days=r["actual_days"], elapsed_days=r["actual_days"],
                           status="completed", delay_days=r["delay_days"],
                           delay_flag=r["delay_flag"])
            elif i == cur:
                # ongoing: sample elapsed as a fraction of the stage's eventual duration
                # (statutory + delay). A delayed stage still in progress can therefore be
                # already past statutory (overrun-while-ongoing), with probability growing
                # with the parcel's true delay. Early-finishing stages never overrun.
                stage_delay = r["delay_days"]
                duration = max(r["statutory_days"] + stage_delay, 1)
                fraction = rng.uniform(0.05, 1.1)  # just-started -> slightly overrun
                elapsed = int(round(fraction * duration))
                elapsed = max(1, min(elapsed, 2000))
                rec.update(status="ongoing", elapsed_days=elapsed)
            else:
                rec.update(status="pending")
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Generating districts + villages ...")
    districts = generate_districts(rng)
    villages = generate_villages(districts, rng)

    print("Generating projects ...")
    projects = generate_projects(districts, rng)

    print("Generating parcels ...")
    parcels = generate_parcels(projects, districts, villages, rng)

    print("Computing full completed timelines ...")
    completed = _compute_delays(projects, parcels, districts, rng)

    print("Building live (in-progress) view ...")
    live = build_live_timelines(parcels, completed, rng)

    # Drop the hidden confound from districts before writing (keep internal only)
    districts_out = districts.drop(columns=["admin_capacity"])

    print("Writing outputs ...")
    projects.to_parquet(OUT_DIR / "projects.parquet", index=False)
    parcels.to_parquet(OUT_DIR / "parcels.parquet", index=False)
    villages.to_parquet(OUT_DIR / "villages.parquet", index=False)
    districts_out.to_parquet(OUT_DIR / "districts.parquet", index=False)
    completed.to_parquet(OUT_DIR / "stage_timelines_historical.parquet", index=False)
    live.to_parquet(OUT_DIR / "stage_timelines_live.parquet", index=False)

    # CSV samples for eyeballing
    projects.head(200).to_csv(OUT_DIR / "sample_projects.csv", index=False)
    parcels.head(200).to_csv(OUT_DIR / "sample_parcels.csv", index=False)
    live.head(200).to_csv(OUT_DIR / "sample_live.csv", index=False)
    completed.head(200).to_csv(OUT_DIR / "sample_historical.csv", index=False)

    print("Done.")
    print(f"  projects: {len(projects)}")
    print(f"  parcels : {len(parcels)} (live={parcels['is_live'].sum()})")
    print(f"  villages: {len(villages)}")
    print(f"  historical timeline rows: {len(completed)}")
    print(f"  live timeline rows      : {len(live)}")


if __name__ == "__main__":
    main()
