"""SIH26017 synthetic data generator (v2 — pan-India multi-state).

Generates a HimBhoomi-style land-record dataset + RFCTLARR lifecycle timelines across
three states (HP, Punjab, Uttarakhand) with:

  - semantic IDs: parcel `<STATE>-<DISTRICT_CODE>-<VILLAGE_CODE>-<KHASRA_NO>`,
                  project `<STATE>-<TYPE>-<SEQ>` (home-state rule for cross-state)
  - spatial types: point projects (1 district) and linear projects (ordered village path
    via centroid-adjacency routing, may cross district/state borders)
  - hidden confounds: state_admin_capacity + district_admin_capacity; both partially
    proxied by stakeholder_responsiveness + historical_performance_score

Emits historical (training) + live (in-progress) datasets. Seeded, reproducible.

Calibration flags let you tune the state-effect scale + proxy blend on a small sample
BEFORE the single full regeneration (see Stage 1 of PROJECT_CONTEXT.md).

Usage:
  .venv/bin/python src/data_generator.py                       # full run
  .venv/bin/python src/data_generator.py --parcels 12000 --projects 600   # calibration
  .venv/bin/python src/data_generator.py --k-state 25 --w-state 0.30 --w-dist 0.30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from states import PROJECT_TYPE_CODES, STATES

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 42
N_PROJECTS = 5000
N_PARCELS = 100_000
LIVE_FRACTION = 0.12
LINEAR_FRACTION = 0.33       # ~share of road/rail projects treated as linear (~15% overall)
CROSS_STATE_PROB = 0.35      # prob a linear project crosses a state border

# State confound effect + proxy blend (CALIBRATED — see Stage 1.5; keep locked)
K_STATE = 35.0               # (1 - state_capacity) * K_STATE days added per stage
ADMIN_EFFECT = 30.0          # (1 - district_capacity) * ADMIN_EFFECT days per stage
W_IND = 0.50                 # individual variation weight in feature proxy
W_STATE = 0.28               # state-capacity weight in feature proxy
W_DIST = 0.22                # district-capacity weight in feature proxy

STAGES = ["SIA", "NOTIFICATION", "DECLARATION", "AWARD", "POSSESSION"]
STATUTORY = {"SIA": 180, "NOTIFICATION": 60, "DECLARATION": 365, "AWARD": 365, "POSSESSION": 90}

PROJECT_TYPES = ["road", "rail", "irrigation", "dam", "industrial"]
PROJECT_TYPE_W = [0.30, 0.15, 0.20, 0.15, 0.20]
PROJECT_PACE = {"road": 0.0, "rail": 8.0, "industrial": 14.0, "irrigation": 20.0, "dam": 35.0}
LINEAR_TYPES = {"road", "rail"}  # these can be linear (highways/rail corridors)

LAND_CLASSES = ["agri", "orchard", "barren", "residential"]
LAND_CLASS_W = [0.45, 0.20, 0.20, 0.15]

COMPENSATION_STATUSES = ["paid", "partial", "pending"]
COMPENSATION_W = [0.60, 0.25, 0.15]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Stage 1: states + districts (hidden state & district confounds)
# --------------------------------------------------------------------------- #

def generate_states_and_districts(rng: np.random.Generator):
    state_rows, district_rows = [], []
    for state_name, info in STATES.items():
        state_code = info["code"]
        state_cap = float(rng.beta(3.0, 2.0))  # hidden state-level pace
        state_rows.append({"state": state_name, "state_code": state_code,
                           "admin_capacity": state_cap})
        for district_name, (dcode, lat, lon) in info["districts"].items():
            # district capacity correlates with its state (so state has a real aggregate)
            dcap = float(np.clip(0.5 * state_cap + 0.5 * rng.beta(3.0, 2.0), 0.05, 0.95))
            district_rows.append({"state": state_name, "state_code": state_code,
                                  "district": district_name, "district_code": dcode,
                                  "lat": lat, "lon": lon, "admin_capacity": dcap})
    states_df = pd.DataFrame(state_rows)
    districts_df = pd.DataFrame(district_rows)
    return states_df, districts_df


# --------------------------------------------------------------------------- #
# Stage 2: villages
# --------------------------------------------------------------------------- #

def generate_villages(districts_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, d in districts_df.iterrows():
        n_tehsils = int(rng.integers(2, 5))
        seq = 0
        for t in range(n_tehsils):
            tehsil = f"{d['district']}_Tehsil_{t + 1}"
            n_villages = int(rng.integers(2, 5))
            for _ in range(n_villages):
                seq += 1
                village_code = f"{seq:04d}"
                village = f"{d['district']}_V{seq:03d}"
                lat = d["lat"] + rng.normal(0, 0.12)
                lon = d["lon"] + rng.normal(0, 0.12)
                rows.append({"village": village, "village_code": village_code,
                             "tehsil": tehsil, "district": d["district"],
                             "district_code": d["district_code"],
                             "state": d["state"], "state_code": d["state_code"],
                             "lat": float(lat), "lon": float(lon)})
    return pd.DataFrame(rows)


def build_adjacency(districts_df: pd.DataFrame, k: int = 4, cross_max_km: float = 130.0) -> dict:
    """Per-district nearest neighbors (4 overall + 1 nearest cross-state within threshold)."""
    codes = districts_df["district_code"].tolist()
    lat = districts_df["lat"].to_numpy()
    lon = districts_df["lon"].to_numpy()
    state = districts_df["state_code"].tolist()
    adj = {}
    for i, c in enumerate(codes):
        dists = _haversine(lat[i], lon[i], lat, lon)
        order = np.argsort(dists)[1:]  # exclude self
        nbrs = [codes[j] for j in order[:k]]
        # nearest cross-state neighbor within threshold (guarantees cross-state edges exist)
        cross = [codes[j] for j in order if state[j] != state[i] and dists[j] < cross_max_km]
        if cross:
            nbrs.append(cross[0])
        adj[c] = list(dict.fromkeys(nbrs))  # dedupe, keep order
    return adj


# --------------------------------------------------------------------------- #
# Stage 3: projects (point / linear geometry + semantic IDs)
# --------------------------------------------------------------------------- #

def _route_linear(start_code: str, adjacency: dict, state_of: dict, rng,
                  cross_state: bool, n_steps: int) -> list:
    path = [start_code]
    visited = {start_code}
    start_state = state_of[start_code]
    current = start_code
    for _ in range(n_steps - 1):
        nbrs = [n for n in adjacency.get(current, []) if n not in visited]
        if not nbrs:
            break
        if cross_state:
            candidates = nbrs
        else:
            candidates = [n for n in nbrs if state_of[n] == start_state] or nbrs
        current = candidates[int(rng.integers(0, len(candidates)))]
        path.append(current)
        visited.add(current)
    return path


def generate_projects(states_df, districts_df, villages_df, adjacency, rng):
    n = N_PROJECTS
    dcode_to_state = dict(zip(districts_df["district_code"], districts_df["state_code"]))
    state_cap = dict(zip(states_df["state_code"], states_df["admin_capacity"]))
    dist_cap = dict(zip(districts_df["district_code"], districts_df["admin_capacity"]))
    villages_by_district = {c: g["village"].tolist()
                            for c, g in villages_df.groupby("district_code")}
    village_info = {r["village"]: r for r in villages_df.to_dict("records")}
    district_codes = districts_df["district_code"].tolist()

    project_type = rng.choice(PROJECT_TYPES, size=n, p=PROJECT_TYPE_W)
    is_linear = np.array([rng.random() < LINEAR_FRACTION if t in LINEAR_TYPES else False
                          for t in project_type])
    cross_state = (rng.random(n) < CROSS_STATE_PROB) & is_linear

    fam_mult = {"road": 1.0, "rail": 1.3, "industrial": 0.8, "irrigation": 1.8, "dam": 2.5}
    base_fam = rng.lognormal(mean=4.5, sigma=0.9, size=n)
    affected = (base_fam * np.array([fam_mult[t] for t in project_type])).astype(int)
    affected = np.clip(affected, 0, 5000)
    # linear projects displace more families (cross more settlements)
    affected = np.where(is_linear, np.clip((affected * 1.8).astype(int), 0, 5000), affected)

    compensation = rng.choice(COMPENSATION_STATUSES, size=n, p=COMPENSATION_W)
    rehab = np.clip(100.0 - (affected / 5000.0) * 40.0 - rng.normal(10, 8, size=n), 0, 100)

    # seq counter for project_id per (state, type)
    seq_counter = {}

    rows = []
    project_geo = {}
    for i in range(n):
        ptype = project_type[i]
        # --- geometry ---
        if is_linear[i]:
            start_code = district_codes[int(rng.integers(0, len(district_codes)))]
            n_steps = int(rng.integers(3, 7))  # 3-7 districts along the path
            path = _route_linear(start_code, adjacency, dcode_to_state, rng,
                                 bool(cross_state[i]), n_steps)
            vill_names = []
            for dc in path:
                vl = villages_by_district.get(dc, [])
                if vl:
                    vill_names.append(vl[int(rng.integers(0, len(vl)))])
            if not vill_names:
                vill_names = [village_info[list(village_info)[int(rng.integers(0, len(village_info)))]]]
                villages = [village_info[vill_names[0]]]
            else:
                villages = [village_info[v] for v in vill_names]
            spatial_type = "linear"
        else:
            start_code = district_codes[int(rng.integers(0, len(district_codes)))]
            vl = villages_by_district.get(start_code, [])
            if not vl:
                start_code = district_codes[int(rng.integers(0, len(district_codes)))]
                vl = villages_by_district.get(start_code, [])
            n_vill = int(rng.integers(1, 3))
            idx = rng.choice(len(vl), size=min(n_vill, len(vl)), replace=False)
            villages = [village_info[vl[j]] for j in idx]
            spatial_type = "point"

        home = villages[0]
        home_state = home["state"]
        home_state_code = home["state_code"]

        # --- semantic project id ---
        type_code = PROJECT_TYPE_CODES[ptype]
        key = (home_state_code, type_code)
        seq_counter[key] = seq_counter.get(key, 0) + 1
        pid = f"{home_state_code}-{type_code}-{seq_counter[key]:04d}"

        # --- institutional features proxy (state x district x individual) ---
        base_resp = rng.beta(4, 2)
        base_perf = rng.beta(4, 2)
        sc = state_cap[home_state_code]
        dc = dist_cap[home["district_code"]]
        stakeholder = float(np.clip(W_IND * base_resp + W_STATE * sc + W_DIST * dc
                                   + rng.normal(0, 0.02), 0, 1))
        hist_perf = float(np.clip(W_IND * base_perf + W_STATE * sc + W_DIST * dc
                                  + rng.normal(0, 0.02), 0, 1))

        coord_path = json.dumps([[round(float(v["lat"]), 4), round(float(v["lon"]), 4)]
                                 for v in villages])

        rows.append({
            "project_id": pid,
            "project_type": ptype,
            "spatial_type": spatial_type,
            "coord_path": coord_path,
            "affected_families": int(affected[i]),
            "compensation_status": compensation[i],
            "rehab_progress_pct": round(float(rehab[i]), 2),
            "stakeholder_responsiveness": round(stakeholder, 3),
            "historical_performance_score": round(hist_perf, 3),
            "state": home_state,
            "state_code": home_state_code,
            "district": home["district"],
            "tehsil": home["tehsil"],
        })
        project_geo[pid] = [v["village"] for v in villages]

    projects_df = pd.DataFrame(rows)
    return projects_df, project_geo


# --------------------------------------------------------------------------- #
# Stage 4: parcels (semantic IDs derived from assigned geography)
# --------------------------------------------------------------------------- #

def generate_parcels(projects_df, project_geo, districts_df, villages_df, rng):
    dist_cap = dict(zip(districts_df["district_code"], districts_df["admin_capacity"]))
    vill_info = villages_df.set_index("village")

    # per-project parcel counts (linear projects get more), scaled to ~N_PARCELS
    spatial = projects_df["spatial_type"].to_numpy()
    base_count = np.array([rng.poisson(15) if s == "point" else rng.poisson(40)
                           for s in spatial])
    base_count = np.where(spatial == "point", np.clip(base_count, 6, 30),
                          np.clip(base_count, 15, 70))
    total = int(base_count.sum())
    scale = N_PARCELS / total if total else 1.0
    counts = np.maximum(1, np.rint(base_count * scale).astype(int))

    rows = []
    for idx, pid in enumerate(projects_df["project_id"]):
        villages = project_geo[pid]
        for _ in range(counts[idx]):
            village = villages[int(rng.integers(0, len(villages)))]
            rows.append({"project_id": pid, "village": village})

    parcels = pd.DataFrame(rows)
    n = len(parcels)

    # attach village geography
    parcels["village_code"] = parcels["village"].map(vill_info["village_code"])
    parcels["tehsil"] = parcels["village"].map(vill_info["tehsil"])
    parcels["district"] = parcels["village"].map(vill_info["district"])
    parcels["district_code"] = parcels["village"].map(vill_info["district_code"])
    parcels["state"] = parcels["village"].map(vill_info["state"])
    parcels["state_code"] = parcels["village"].map(vill_info["state_code"])

    # hidden confound proxy: low-capacity districts -> more litigation-heavy parcels
    cap = parcels["district_code"].map(dist_cap).to_numpy()
    low_cap = 1.0 - cap
    p_stay = np.clip(0.03 + 0.14 * low_cap + rng.normal(0, 0.02, size=n), 0.01, 0.6)
    parcels["court_stay"] = (rng.random(n) < p_stay).astype(int)
    parcels["owner_count"] = np.clip(rng.poisson(3.0 + 4.0 * low_cap, size=n) + 1, 1, 30)
    parcels["pending_mutations"] = np.clip(rng.poisson(0.5 + 2.5 * low_cap, size=n), 0, 10)
    parcels["encumbrances"] = np.clip(rng.poisson(0.4 + 1.2 * low_cap, size=n), 0, 5)
    parcels["land_class"] = rng.choice(LAND_CLASSES, size=n, p=LAND_CLASS_W)

    area_mult = {"agri": 1.0, "orchard": 0.7, "barren": 1.5, "residential": 0.15}
    area = rng.lognormal(mean=7.5, sigma=1.0, size=n) * \
        np.array([area_mult[c] for c in parcels["land_class"]])
    parcels["area_sqm"] = np.round(np.clip(area, 20, 1_000_000), 1)
    parcels["is_live"] = (rng.random(n) < LIVE_FRACTION).astype(int)

    # khasra number: sequential within village, then semantic parcel id
    parcels["khasra_seq"] = parcels.groupby("village").cumcount() + 1
    parcels["khasra_number"] = parcels["khasra_seq"].map(lambda x: f"{x:04d}")
    parcels["parcel_id"] = (parcels["state_code"] + "-" + parcels["district_code"] + "-"
                            + parcels["village_code"] + "-" + parcels["khasra_number"])

    parcels = parcels.drop(columns=["khasra_seq"])
    return parcels.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 5: delay rules engine
# --------------------------------------------------------------------------- #

def _compute_delays(projects_df, parcels, districts_df, states_df, rng):
    n = len(parcels)
    proj = projects_df.set_index("project_id")
    dist_cap = dict(zip(districts_df["district_code"], districts_df["admin_capacity"]))
    state_cap = dict(zip(states_df["state_code"], states_df["admin_capacity"]))

    pid = parcels["project_id"].tolist()
    p_type = proj.loc[pid, "project_type"].to_numpy()
    affected = proj.loc[pid, "affected_families"].to_numpy()
    compensation = proj.loc[pid, "compensation_status"].to_numpy()
    rehab = proj.loc[pid, "rehab_progress_pct"].to_numpy()
    stakeholder = proj.loc[pid, "stakeholder_responsiveness"].to_numpy()
    hist_perf = proj.loc[pid, "historical_performance_score"].to_numpy()

    owner_count = parcels["owner_count"].to_numpy()
    land_class = parcels["land_class"].to_numpy()
    pending_mut = parcels["pending_mutations"].to_numpy()
    court_stay = parcels["court_stay"].to_numpy()
    encumbrances = parcels["encumbrances"].to_numpy()

    dcode = parcels["district_code"].to_numpy()
    scode = parcels["state_code"].to_numpy()
    admin_cap = np.array([dist_cap[c] for c in dcode])
    state_capacity = np.array([state_cap[c] for c in scode])

    pace = np.array([PROJECT_PACE[t] for t in p_type])
    parcel_luck = rng.normal(-12, 16, size=n)
    global_slow = (1.0 - stakeholder) * 20.0 + (1.0 - hist_perf) * 15.0

    # hidden confound effects (state + district, not in features)
    state_effect = (1.0 - state_capacity) * K_STATE
    admin_effect = (1.0 - admin_cap) * ADMIN_EFFECT
    BASELINE = 30.0

    orchard = (land_class == "orchard")
    compensation_unpaid = (compensation != "paid")
    many_owners = (owner_count > 4).astype(float)
    fam_load = affected / 200.0
    rehab_shortfall = (100.0 - rehab) / 100.0

    def noise():
        return rng.normal(0, 14, size=n)

    base = pace + global_slow + admin_effect + state_effect + parcel_luck - BASELINE
    delays = {
        "SIA": base + (1.0 - stakeholder) * 30.0 + noise(),
        "NOTIFICATION": base + pending_mut * 15.0 + (1.0 - stakeholder) * 25.0 + noise(),
        "DECLARATION": base + many_owners * (owner_count - 4.0) * 12.0
        + pending_mut * 10.0 + fam_load * 20.0 + noise(),
        "AWARD": base + court_stay * 240.0 + orchard * 120.0
        + encumbrances * 25.0 + noise(),
        "POSSESSION": base + court_stay * 180.0 + compensation_unpaid * 150.0
        + rehab_shortfall * fam_load * 60.0 + noise(),
    }

    records = []
    for stage in STAGES:
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


def build_live_timelines(parcels, completed, rng):
    live_parcels = parcels[parcels["is_live"] == 1]
    live_ids = set(live_parcels["parcel_id"])
    live_df = completed[completed["parcel_id"].isin(live_ids)].copy()
    stage_weights = [0.10, 0.15, 0.25, 0.30, 0.20]
    cur_stage = {pid: int(rng.choice(len(STAGES), p=stage_weights)) for pid in live_ids}

    rows = []
    for pid, grp in live_df.groupby("parcel_id", sort=False):
        grp = grp.sort_values("stage", key=lambda s: s.map({x: i for i, x in enumerate(STAGES)}))
        cur = cur_stage[pid]
        for i, (_, r) in enumerate(grp.iterrows()):
            rec = {"parcel_id": pid, "stage": r["stage"],
                   "statutory_days": r["statutory_days"], "actual_days": None,
                   "elapsed_days": None, "status": None, "delay_days": None, "delay_flag": None}
            if i < cur:
                rec.update(actual_days=r["actual_days"], elapsed_days=r["actual_days"],
                           status="completed", delay_days=r["delay_days"],
                           delay_flag=r["delay_flag"])
            elif i == cur:
                stage_delay = r["delay_days"]
                duration = max(r["statutory_days"] + stage_delay, 1)
                fraction = rng.uniform(0.05, 1.1)
                rec.update(status="ongoing", elapsed_days=max(1, min(round(fraction * duration), 2000)))
            else:
                rec.update(status="pending")
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    global N_PARCELS, N_PROJECTS, K_STATE, W_STATE, W_DIST, W_IND, ADMIN_EFFECT
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcels", type=int, default=N_PARCELS)
    ap.add_argument("--projects", type=int, default=N_PROJECTS)
    ap.add_argument("--k-state", type=float, default=K_STATE)
    ap.add_argument("--admin-effect", type=float, default=ADMIN_EFFECT)
    ap.add_argument("--w-state", type=float, default=W_STATE)
    ap.add_argument("--w-dist", type=float, default=W_DIST)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    N_PARCELS, N_PROJECTS = args.parcels, args.projects
    K_STATE = args.k_state
    ADMIN_EFFECT = args.admin_effect
    W_STATE, W_DIST = args.w_state, args.w_dist
    W_IND = max(0.0, 1.0 - W_STATE - W_DIST)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Generating states + districts ...")
    states_df, districts_df = generate_states_and_districts(rng)
    print("Generating villages ...")
    villages = generate_villages(districts_df, rng)
    print("Building district adjacency ...")
    adjacency = build_adjacency(districts_df)
    print("Generating projects ...")
    projects, project_geo = generate_projects(states_df, districts_df, villages, adjacency, rng)
    print("Generating parcels ...")
    parcels = generate_parcels(projects, project_geo, districts_df, villages, rng)
    print("Computing full completed timelines ...")
    completed = _compute_delays(projects, parcels, districts_df, states_df, rng)
    print("Building live view ...")
    live = build_live_timelines(parcels, completed, rng)

    states_out = states_df.drop(columns=["admin_capacity"])
    districts_out = districts_df.drop(columns=["admin_capacity"])

    print("Writing outputs ...")
    states_out.to_parquet(OUT_DIR / "states.parquet", index=False)
    districts_out.to_parquet(OUT_DIR / "districts.parquet", index=False)
    villages.to_parquet(OUT_DIR / "villages.parquet", index=False)
    projects.to_parquet(OUT_DIR / "projects.parquet", index=False)
    parcels.to_parquet(OUT_DIR / "parcels.parquet", index=False)
    completed.to_parquet(OUT_DIR / "stage_timelines_historical.parquet", index=False)
    live.to_parquet(OUT_DIR / "stage_timelines_live.parquet", index=False)

    projects.head(200).to_csv(OUT_DIR / "sample_projects.csv", index=False)
    parcels.head(200).to_csv(OUT_DIR / "sample_parcels.csv", index=False)

    print("Done.")
    print(f"  states  : {len(states_out)}")
    print(f"  districts: {len(districts_out)}")
    print(f"  projects: {len(projects)} (linear={int((projects['spatial_type']=='linear').sum())})")
    print(f"  parcels : {len(parcels)} (live={int(parcels['is_live'].sum())})")
    print(f"  villages: {len(villages)}")
    print(f"  historical timeline rows: {len(completed)}")
    print(f"  live timeline rows      : {len(live)}")


if __name__ == "__main__":
    main()
