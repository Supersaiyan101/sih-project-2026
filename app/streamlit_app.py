"""SIH26017 Streamlit dashboard (v2 — multi-state, interactive).

Offline-capable. Reads the portfolio cache + projects geometry, calls predict.py
directly. Views: Portfolio (cascading filter + clickable project/parcel tables),
Project detail (summary + per-stage bottleneck + segment profile + parcels),
Parcel detail (per-stage bars + SHAP + actions), What-if, Alerts, Map
(point markers + linear polylines).

Run:
  .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402
import ui  # noqa: E402
import user_projects  # noqa: E402
from features import STAGES  # noqa: E402
from predict import load_artifacts, score_parcel, risk_level, STATUTORY  # noqa: E402
from ui import (COLORS, EMOJI, inject_css, setup_plotly, brand_sidebar,  # noqa: E402
                hero, kpi_row, section, footer)

PORTFOLIO_PATH = ROOT / "data" / "generated" / "portfolio_scores.parquet"
PROJECTS_PATH = ROOT / "data" / "generated" / "projects.parquet"

st.set_page_config(page_title="BhoomiSetu — Land Delay Early Warning", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #

@st.cache_data
def load_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_PATH.exists():
        predict.refresh_portfolio()
    main = pd.read_parquet(PORTFOLIO_PATH)
    main["is_user"] = 0
    user = user_projects.load_user_parcels()
    if len(user):
        user = user.copy()
        user["is_user"] = 1
        main = pd.concat([main, user], ignore_index=True)
    ovr = user_projects.load_overrides()
    if len(ovr):
        main = _apply_overrides(main, ovr)
    return main


def _apply_overrides(df: pd.DataFrame, ovr: pd.DataFrame) -> pd.DataFrame:
    """Apply lifecycle state overrides (compensation/rehab) and re-score those projects."""
    artifacts = predict.load_artifacts()
    for _, o in ovr.iterrows():
        mask = df["project_id"] == o["project_id"]
        if not mask.any():
            continue
        df.loc[mask, "compensation_status"] = o["compensation_status"]
        df.loc[mask, "rehab_progress_pct"] = float(o["rehab_progress_pct"])
        scores = predict.score_batch(df[mask], artifacts, include_stages=True)
        cols = (["risk_score", "risk_level", "expected_overrun_days", "max_delay_prob"]
                + [f"{s}_prob" for s in STAGES] + [f"{s}_overrun" for s in STAGES])
        for col in cols:
            df.loc[mask, col] = scores[col].values
    return df


@st.cache_data
def load_projects() -> pd.DataFrame:
    return pd.read_parquet(PROJECTS_PATH)


@st.cache_data
def load_districts() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "generated" / "districts.parquet")


@st.cache_resource
def load_artifacts_cached() -> dict:
    return load_artifacts()


def load_live_timeline(parcel_id: str) -> dict:
    tl = pd.read_parquet(ROOT / "data" / "generated" / "stage_timelines_live.parquet")
    sub = tl[tl["parcel_id"] == parcel_id]
    out = {}
    for _, r in sub.iterrows():
        out[r["stage"]] = {
            "status": r["status"],
            "elapsed_days": None if pd.isna(r["elapsed_days"]) else int(r["elapsed_days"]),
            "actual_days": None if pd.isna(r["actual_days"]) else int(r["actual_days"]),
        }
    return out


def refresh() -> None:
    predict.refresh_portfolio()
    load_portfolio.clear()
    st.rerun()


def nav_to(page: str) -> None:
    st.session_state["nav"] = page
    st.rerun()


def risk_color(score: float) -> str:
    return COLORS[risk_level(score)]


@st.cache_data
def metrics_summary_footer() -> str:
    """Auto-sourced cold-start footnote (never stale — reads metrics_report.json)."""
    import json as _json
    try:
        r = _json.loads((ROOT / "models" / "metrics_report.json").read_text())
        loso = r["cold_start_loso"]["drop_pct"]
        loso_avg = float(np.mean(list(loso.values())))
        auroc = r["stages"]["POSSESSION"]["classifier"]["auroc"]
        return (f"BhoomiSetu · Cold-start validated: leave-one-state-out avg drop "
                f"<b>{loso_avg:.1f}%</b> (AUROC ≈ {auroc:.2f}). "
                f"Figures auto-sourced from <code>metrics_report.json</code>.")
    except Exception:
        return "BhoomiSetu · Early warning for land acquisition · RFCTLARR 2013."


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Portfolio view
# --------------------------------------------------------------------------- #

def filter_bar(df: pd.DataFrame) -> pd.DataFrame:
    c = st.columns(4)
    states = ["All"] + sorted(df["state"].unique())
    state = c[0].selectbox("State", states)
    d = df if state == "All" else df[df["state"] == state]

    districts = ["All"] + sorted(d["district"].unique())
    district = c[1].selectbox("District", districts)
    d = d if district == "All" else d[d["district"] == district]

    ptypes = ["All"] + sorted(d["project_type"].unique())
    ptype = c[2].selectbox("Project type", ptypes)
    d = d if ptype == "All" else d[d["project_type"] == ptype]

    levels = c[3].multiselect("Risk level", ["RED", "YELLOW", "GREEN"],
                              default=["RED", "YELLOW", "GREEN"])
    d = d[d["risk_level"].isin(levels)] if levels else d
    return d


def project_table(df: pd.DataFrame) -> None:
    agg = (df.groupby("project_id")
             .agg(project_type=("project_type", "first"),
                  spatial=("spatial_type", "first"),
                  state=("state_code", "first"),
                  district=("district", "first"),
                  n_parcels=("parcel_id", "size"),
                  avg_risk=("risk_score", "mean"),
                  red=("risk_level", lambda s: (s == "RED").sum()),
                  avg_overrun=("expected_overrun_days", "mean"))
             .reset_index()
             .sort_values("avg_risk", ascending=False))
    agg["level"] = agg["avg_risk"].map(risk_level)
    disp = agg[["project_id", "project_type", "spatial", "state", "district",
                "n_parcels", "avg_risk", "red", "avg_overrun", "level"]]
    disp.columns = ["project_id", "type", "spatial", "state", "district",
                    "parcels", "avg risk", "RED", "avg overrun", "level"]
    disp = disp.reset_index(drop=True)
    event = st.dataframe(disp, width="stretch", hide_index=True, key="proj_table",
                         on_select="rerun", selection_mode="single-row")
    if event.selection and event.selection.rows:
        sel = disp.iloc[event.selection.rows[0]]["project_id"]
        st.session_state["selected_project"] = sel
        nav_to("Project")


def parcel_table(df: pd.DataFrame, key: str, limit: int = 100) -> None:
    sub = df.sort_values("risk_score", ascending=False).head(limit).reset_index(drop=True)
    cols = ["parcel_id", "risk_level", "risk_score", "expected_overrun_days",
            "current_stage", "overrun_while_ongoing_days", "court_stay", "compensation_status"]
    event = st.dataframe(sub[cols], width="stretch", hide_index=True, key=key,
                         on_select="rerun", selection_mode="single-row")
    if event.selection and event.selection.rows:
        sel = sub.iloc[event.selection.rows[0]]["parcel_id"]
        st.session_state["selected_parcel"] = sel
        nav_to("Detail")


def view_portfolio(df: pd.DataFrame) -> None:
    st.subheader("Portfolio risk table")
    d = filter_bar(df)

    kpi_row([
        ("Projects", f"{d['project_id'].nunique():,}", None, ui.NAVY),
        ("Live parcels", f"{len(d):,}", None, ui.STEEL),
        ("RED", f"{(d['risk_level'] == 'RED').sum():,}", None, ui.RED),
        ("YELLOW", f"{(d['risk_level'] == 'YELLOW').sum():,}", None, ui.YEL),
        ("Avg risk", f"{d['risk_score'].mean():.2f}", None, ui.NAVY),
    ])

    st.markdown("**Projects** (click a row to open)")
    project_table(d)


# --------------------------------------------------------------------------- #
# Project detail view
# --------------------------------------------------------------------------- #

def view_project(df: pd.DataFrame) -> None:
    st.subheader("Project detail")
    pid = st.session_state.get("selected_project")
    if not pid:
        st.info("Select a project from the Portfolio view first.")
        return
    sub = df[df["project_id"] == pid]
    if sub.empty:
        st.warning("Project not found in the live portfolio.")
        return
    p0 = sub.iloc[0]
    ptype = p0["project_type"]
    spatial = p0["spatial_type"]
    state = p0["state"]
    district = p0["district"]

    kpi_row([
        ("Project", pid, None, ui.NAVY),
        ("Type", f"{ptype} · {spatial}", None, ui.STEEL),
        ("State / District", f"{state} / {district}", None, ui.NAVY),
        ("Parcels", f"{len(sub):,}", None, ui.STEEL),
        ("Aggregate risk", f"{sub['risk_score'].mean():.3f}", None, ui.NAVY),
    ])

    st.markdown(f"Affected families **{int(p0['affected_families']):,}** · "
                f"compensation **{p0['compensation_status']}** · "
                f"rehab **{p0['rehab_progress_pct']:.0f}%** · "
                f"responsiveness **{p0['stakeholder_responsiveness']:.2f}**")

    red = (sub["risk_level"] == "RED").sum()
    yel = (sub["risk_level"] == "YELLOW").sum()
    st.markdown(f"Risk mix: 🔴 {red} · 🟡 {yel} · 🟢 {len(sub) - red - yel}   |   "
                f"avg expected overrun **{sub['expected_overrun_days'].mean():.0f} days**")

    if st.session_state.get("role", "Officer") != "Viewer":
        with st.expander("Update project state (compensation / rehab) → re-score"):
            comp_opts = ["pending", "partial", "paid"]
            cur_comp = p0["compensation_status"]
            comp = st.selectbox("Compensation status", comp_opts,
                                index=comp_opts.index(cur_comp) if cur_comp in comp_opts else 0)
            rehab = st.slider("Rehab progress %", 0, 100,
                              int(round(min(max(float(p0["rehab_progress_pct"]), 0), 100))))
            if st.button("Apply update"):
                user_projects.save_override(pid, comp, rehab)
                load_portfolio.clear()
                st.rerun()

    left, right = st.columns(2)
    with left:
        st.markdown("**Per-stage bottleneck** (mean delay probability)")
        sp = {s: sub[f"{s}_prob"].mean() for s in STAGES}
        fig = go.Figure(go.Bar(x=list(sp.keys()), y=list(sp.values()), marker_color="#3498db",
                               text=[f"{v:.0%}" for v in sp.values()], textposition="outside"))
        fig.update_layout(template="sih", height=300, yaxis=dict(range=[0, 1], title="P(delay)"))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Segment profile** (risk by village)")
        seg = sub.groupby("village")["risk_score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        fig = go.Figure(go.Bar(x=seg["mean"], y=seg.index, orientation="h", marker_color="#e67e22"))
        fig.update_layout(template="sih", height=300, xaxis_title="avg risk")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Timeline analysis — statutory vs expected duration per stage**")
    stat = {s: STATUTORY[s] for s in STAGES}
    expected = {s: max(0.0, float(sub[f"{s}_overrun"].mean())) for s in STAGES}
    act = {s: stat[s] + expected[s] for s in STAGES}
    fig = go.Figure()
    fig.add_trace(go.Bar(name="statutory", x=[stat[s] for s in STAGES], y=STAGES,
                         orientation="h", marker_color="#95a5a6"))
    fig.add_trace(go.Bar(name="expected (statutory + overrun)", x=[act[s] for s in STAGES],
                         y=STAGES, orientation="h", marker_color="#2E75B6"))
    fig.update_layout(barmode="group", template="sih", height=260,
                      xaxis_title="days", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Parcels** (click a row to open)")
    parcel_table(sub, key="proj_parcels")


# --------------------------------------------------------------------------- #
# Parcel detail view
# --------------------------------------------------------------------------- #

def _pick_parcel(df: pd.DataFrame) -> str:
    if "selected_parcel" in st.session_state and st.session_state["selected_parcel"] in df["parcel_id"].values:
        default = st.session_state["selected_parcel"]
    else:
        default = df["parcel_id"].iloc[df["risk_score"].argmax()]
    idx = df["parcel_id"].tolist().index(default)
    return st.selectbox("Parcel", df["parcel_id"].tolist(), index=idx, key="parcel_selector")


def _stage_bars(contract: dict) -> go.Figure:
    names = list(contract["stages"].keys())
    probs = [contract["stages"][s]["delay_prob"] for s in names]
    overruns = [contract["stages"][s]["expected_overrun"] for s in names]
    statutory = [contract["stages"][s]["statutory_days"] for s in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="delay probability", x=names, y=probs, marker_color="#3498db",
                         yaxis="y", text=[f"{p:.0%}" for p in probs], textposition="outside"))
    fig.add_trace(go.Bar(name="expected overrun (days)", x=names, y=overruns, marker_color="#e67e22",
                         yaxis="y2", opacity=0.85))
    fig.add_trace(go.Scatter(name="statutory days", x=names, y=statutory, yaxis="y2",
                             mode="lines+markers", line=dict(dash="dot", color="#555"), marker=dict(size=6)))
    fig.update_layout(barmode="group", template="sih", height=380,
                      yaxis=dict(title="P(delay)", range=[0, 1]),
                      yaxis2=dict(title="days", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    return fig


def _shap_bars(contract: dict) -> go.Figure:
    tf = contract["top_factors"]
    names = [t[0] for t in tf][::-1]
    vals = [t[1] for t in tf][::-1]
    colors = [COLORS["RED"] if v >= 0 else "#3498db" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(template="sih", height=340, xaxis_title="|SHAP| impact")
    return fig


def view_detail(df: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("Parcel detail")
    pid = _pick_parcel(df)
    row = df[df["parcel_id"] == pid].iloc[0]
    features = {c: row[c] for c in predict.DEFAULTS}
    timeline = load_live_timeline(pid)
    contract = score_parcel(features, artifacts=artifacts, timeline=timeline, parcel_id=pid)

    kpi_row([
        ("Risk score", f"{contract['risk_score']:.3f}", None, ui.NAVY),
        ("Level", contract["risk_level"],
         None, COLORS.get(contract["risk_level"], ui.NAVY)),
        ("Expected overrun", f"{contract['expected_overrun_days']:.0f} days", None, ui.STEEL),
        ("Already overrun (ongoing)",
         f"{contract['overrun_while_ongoing_days']:.0f} d"
         if contract["overrun_while_ongoing_days"] else "—", None, ui.RED),
    ])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Per-stage delay probability vs statutory clock**")
        st.plotly_chart(_stage_bars(contract), width="stretch")
    with right:
        st.markdown("**Why is this parcel at risk? (SHAP)**")
        st.plotly_chart(_shap_bars(contract), width="stretch")
        st.markdown("**Recommended actions**")
        for a in contract["recommended_actions"]:
            badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}[a["priority_label"]]
            st.markdown(f"- {badge} **{a['factor']}** — {a['action']}")


# --------------------------------------------------------------------------- #
# What-if view
# --------------------------------------------------------------------------- #

def view_whatif(df: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("What-if simulator")
    pid = _pick_parcel(df)
    row = df[df["parcel_id"] == pid].iloc[0]
    features = {c: row[c] for c in predict.DEFAULTS}

    c1, c2 = st.columns(2)
    new_stay = c1.selectbox("Court stay", [0, 1], index=int(features["court_stay"]))
    new_comp = c2.selectbox("Compensation status", ["paid", "partial", "pending"],
                            index=["paid", "partial", "pending"].index(features["compensation_status"]))
    changed = (new_stay != int(features["court_stay"])) or (new_comp != features["compensation_status"])

    before = score_parcel(features, artifacts=artifacts, parcel_id=pid)
    after = score_parcel({**features, "court_stay": new_stay, "compensation_status": new_comp},
                         artifacts=artifacts, parcel_id=pid)

    kpi_row([
        ("Risk (before)", f"{before['risk_score']:.3f}", None, ui.NAVY),
        ("Risk (after)", f"{after['risk_score']:.3f}",
         f"{after['risk_score'] - before['risk_score']:+.3f}", ui.STEEL),
        ("Expected overrun (after)", f"{after['expected_overrun_days']:.0f} d",
         f"{after['expected_overrun_days'] - before['expected_overrun_days']:+.0f} d", ui.NAVY),
    ])
    if not changed:
        st.info("Change the toggles to see the risk move live.")
    else:
        st.success(f"Level {before['risk_level']} → {after['risk_level']}")


# --------------------------------------------------------------------------- #
# Alerts view
# --------------------------------------------------------------------------- #

def view_alerts(df: pd.DataFrame) -> None:
    st.subheader("Automated alert feed")
    alerts = []
    for _, r in df.iterrows():
        if r["overrun_while_ongoing_days"] and r["overrun_while_ongoing_days"] > 0:
            alerts.append((r["parcel_id"], "overrun-while-ongoing",
                           f"{r['overrun_while_ongoing_days']:.0f} days past legal limit"))
        if r["risk_level"] == "RED":
            alerts.append((r["parcel_id"], "high-risk", f"risk {r['risk_score']:.2f}"))
        if int(r["court_stay"]) == 1:
            alerts.append((r["parcel_id"], "court-stay", "active court stay"))
        if r["compensation_status"] != "paid":
            alerts.append((r["parcel_id"], "compensation", f"compensation {r['compensation_status']}"))
        if r["expected_overrun_days"] > 365:
            alerts.append((r["parcel_id"], "severe-overrun",
                           f"{r['expected_overrun_days']:.0f} days expected overrun"))
    a = pd.DataFrame(alerts, columns=["parcel_id", "alert_type", "detail"])
    st.markdown(f"**{len(a)} active alerts** across {a['parcel_id'].nunique()} parcels")
    if not a.empty:
        st.dataframe(a.head(200), width="stretch", hide_index=True)

        st.markdown("**Notification log (simulated SMS / Email / Push)**")
        channels = ["SMS", "Email", "Push"]
        recipients = ["District Collector", "Project Manager", "Acquiring Officer", "Admin"]
        rows = []
        for i, (_, r) in enumerate(a.head(60).iterrows()):
            rows.append({
                "channel": channels[i % 3],
                "recipient": recipients[i % 4],
                "parcel_id": r["parcel_id"],
                "alert": r["alert_type"],
                "message": f"{r['parcel_id']} — {r['detail']}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Map view
# --------------------------------------------------------------------------- #

def view_map(df: pd.DataFrame) -> None:
    st.subheader("Project risk map (offline)")
    projects = load_projects()
    districts = load_districts()
    dcentroid = dict(zip(districts["district"], zip(districts["lat"], districts["lon"])))
    proj_risk = df.groupby("project_id")["risk_score"].mean()

    fig = go.Figure()

    # point projects -> markers at district centroid
    pt = projects[projects["spatial_type"] == "point"]
    pt_lat, pt_lon, pt_risk, pt_ids, pt_names = [], [], [], [], []
    for _, p in pt.iterrows():
        lat, lon = dcentroid.get(p["district"], (None, None))
        if lat is None:
            continue
        pt_lat.append(lat); pt_lon.append(lon)
        pt_risk.append(float(proj_risk.get(p["project_id"], 0.0)))
        pt_ids.append(p["project_id"]); pt_names.append(f"{p['project_id']} ({p['project_type']})")
    fig.add_trace(go.Scatter(
        x=pt_lon, y=pt_lat, mode="markers", name="point",
        marker=dict(size=10, color=[risk_color(r) for r in pt_risk],
                    line=dict(width=1, color="#333")),
        customdata=pt_ids, text=pt_names, hovertemplate="%{text}<br>risk: %{marker.color}",
        visible="legendonly" if False else True))

    # linear projects -> polylines through coord_path
    lin = projects[projects["spatial_type"] == "linear"]
    for _, p in lin.iterrows():
        try:
            path = json.loads(p["coord_path"])
        except (ValueError, TypeError):
            continue
        if not path:
            continue
        lats = [pt_[0] for pt_ in path]
        lons = [pt_[1] for pt_ in path]
        r = float(proj_risk.get(p["project_id"], 0.0))
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode="lines", name=p["project_id"],
            line=dict(color=risk_color(r), width=3),
            customdata=[p["project_id"]] * len(lats), text=[p["project_id"]] * len(lats),
            hovertemplate=f"{p['project_id']} ({p['project_type']})<extra></extra>"))

    fig.update_layout(template="sih", height=560, showlegend=False,
                      xaxis_title="longitude", yaxis_title="latitude",
                      title="Point projects (markers) + linear corridors (lines), colored by risk")

    sel = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                          key="map_select")
    if sel.selection and sel.selection.points:
        cd = sel.selection.points[0].get("customdata")
        if cd:
            st.session_state["selected_project"] = cd
            nav_to("Project")

    # fallback picker
    options = sorted(projects["project_id"].tolist())
    pick = st.selectbox("Or pick a project", ["—"] + options)
    if pick != "—":
        st.session_state["selected_project"] = pick
        nav_to("Project")

    st.caption("Offline lat/lon visualization (no tile server). Linear corridors are straight "
               "village-path approximations, not real road geometry.")


# --------------------------------------------------------------------------- #
# New project view
# --------------------------------------------------------------------------- #

def view_new_project(artifacts: dict) -> None:
    st.subheader("New project onboarding")
    st.caption("Enter only what is known at project sanctioning. Compensation & rehab progress "
               "are lifecycle states — tracked later, not typed here. Responsiveness & track "
               "record are derived from the district's observed performance.")

    c = st.columns(3)
    project_type = c[0].selectbox("Project type", ["road", "rail", "irrigation", "dam", "industrial"])
    spatial_type = c[1].selectbox("Spatial type", ["point", "linear"])
    affected = c[2].number_input("Affected families (from SIA/DPR)", 1, 50_000, 100)

    uploaded = st.file_uploader("Upload CSV of parcel IDs (one per line, or comma-separated)",
                                type=["csv", "txt"])
    ids: list[str] = []
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8")
        ids = list(dict.fromkeys(t for t in re.split(r"[,\s\n;]+", raw.strip()) if t))
        found, missing = user_projects.pull_records(ids)
        st.info(f"{len(ids)} IDs parsed → {len(found)} found, {len(missing)} unknown")
        if missing:
            st.warning("Unknown parcel IDs (ignored): " + ", ".join(missing[:20])
                       + ("…" if len(missing) > 20 else ""))

    if st.button("Create & score project", disabled=(not ids)):
        found, _missing = user_projects.pull_records(ids)
        if found.empty:
            st.error("No valid parcels found — nothing to create.")
            return

        home_state = found["state"].mode()[0]
        home_state_code = found["state_code"].mode()[0]
        home_district = found["district"].mode()[0]

        # validation: affected_families should be roughly consistent with parcel count
        ratio = affected / max(len(found), 1)
        if ratio > 20 or ratio < 0.1:
            st.warning(f"affected_families ({affected:,}) vs {len(found)} parcels (ratio "
                       f"{ratio:.1f}) looks inconsistent — creating anyway.")

        # lifecycle fields start at their honest defaults; soft scores are DERIVED
        compensation = "pending"
        rehab = 0.0
        profile = user_projects.derive_institutional_profile(home_district)
        responsiveness = profile["stakeholder_responsiveness"]
        hist_perf = profile["historical_performance_score"]

        feat = found[["parcel_id", "owner_count", "land_class", "area_sqm",
                      "pending_mutations", "court_stay", "encumbrances"]].copy()
        feat["project_type"] = project_type
        feat["affected_families"] = affected
        feat["compensation_status"] = compensation
        feat["rehab_progress_pct"] = rehab
        feat["stakeholder_responsiveness"] = responsiveness
        feat["historical_performance_score"] = hist_perf

        scores = predict.score_batch(feat, artifacts, include_stages=True)
        geo = found[["parcel_id", "village", "village_code", "tehsil", "district",
                     "district_code", "state", "state_code"]]
        villages = pd.read_parquet(ROOT / "data" / "generated" / "villages.parquet")[["village", "lat", "lon"]]
        rows = scores.merge(feat, on="parcel_id", how="left").merge(geo, on="parcel_id", how="left")
        rows = rows.merge(villages, on="village", how="left")

        pid = user_projects.generate_user_project_id(home_state_code, project_type)

        rows["spatial_type"] = spatial_type
        rows["project_id"] = pid
        rows["current_stage"] = None
        rows["overrun_while_ongoing_days"] = None

        if spatial_type == "linear":
            coords = [[float(v["lat"]), float(v["lon"])]
                      for _, v in rows[["village", "lat", "lon"]].drop_duplicates("village").iterrows()]
        else:
            coords = []

        project_row = {
            "project_id": pid, "project_type": project_type, "spatial_type": spatial_type,
            "coord_path": json.dumps(coords), "affected_families": affected,
            "compensation_status": compensation, "rehab_progress_pct": rehab,
            "stakeholder_responsiveness": responsiveness,
            "historical_performance_score": hist_perf, "state": home_state,
            "state_code": home_state_code, "district": home_district,
            "tehsil": found["tehsil"].mode()[0],
        }
        user_projects.persist_user(project_row, rows)
        load_portfolio.clear()
        st.session_state["selected_project"] = pid
        st.success(f"Project {pid} created with {len(rows)} parcels.")
        nav_to("Project")


# --------------------------------------------------------------------------- #
# Area of Interest view
# --------------------------------------------------------------------------- #

def view_area(df: pd.DataFrame) -> None:
    st.subheader("Area of Interest (site catchment analysis)")
    villages = pd.read_parquet(ROOT / "data" / "generated" / "villages.parquet")

    c = st.columns(3)
    state = c[0].selectbox("State", sorted(villages["state"].unique()))
    vd = villages[villages["state"] == state]
    district = c[1].selectbox("District", sorted(vd["district"].unique()))
    vdd = vd[vd["district"] == district]
    village = c[2].selectbox("Center village", sorted(vdd["village"].unique()))
    radius = st.slider("Radius (km)", 1, 50, 15)

    center = vdd[vdd["village"] == village].iloc[0]
    d = haversine(center["lat"], center["lon"], df["lat"], df["lon"])
    subset = df[d <= radius]

    kpi_row([
        ("Parcels in area", f"{len(subset):,}", None, ui.NAVY),
        ("RED", f"{(subset['risk_level'] == 'RED').sum():,}", None, ui.RED),
        ("Avg risk", f"{subset['risk_score'].mean():.2f}" if len(subset) else "—", None, ui.STEEL),
        ("Avg expected overrun",
         f"{subset['expected_overrun_days'].mean():.0f} d" if len(subset) else "—", None, ui.NAVY),
    ])

    if len(subset):
        factors = {
            "court stay": int(subset["court_stay"].sum()),
            "compensation pending": int((subset["compensation_status"] != "paid").sum()),
            "orchard land": int((subset["land_class"] == "orchard").sum()),
            "owners > 4": int((subset["owner_count"] > 4).sum()),
            "mutations >= 2": int((subset["pending_mutations"] >= 2).sum()),
        }
        fig = go.Figure(go.Bar(x=list(factors.values()), y=list(factors.keys()),
                               orientation="h", marker_color="#8e44ad"))
        fig.update_layout(template="sih", height=260, xaxis_title="parcels affected")
        st.markdown("**Risk-factor prevalence in the area**")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Riskiest parcels in the area** (click to open)")
        parcel_table(subset, key="area_parcels")
    else:
        st.info("No parcels within this radius — widen the radius or pick a different center.")


# --------------------------------------------------------------------------- #
# Trends view
# --------------------------------------------------------------------------- #

def view_trends(df: pd.DataFrame) -> None:
    st.subheader("Delay trends")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average risk by state**")
        s = df.groupby("state")["risk_score"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#2E75B6",
                               text=[f"{v:.2f}" for v in s.values], textposition="outside"))
        fig.update_layout(template="sih", height=220, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Mean delay probability per stage**")
        sp = {stg: float(df[f"{stg}_prob"].mean()) for stg in STAGES}
        fig = go.Figure(go.Bar(x=list(sp.keys()), y=list(sp.values()), marker_color="#3498db",
                               text=[f"{v:.0%}" for v in sp.values()], textposition="outside"))
        fig.update_layout(template="sih", height=220, yaxis=dict(range=[0, 1]),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Average risk by project type**")
        pt = df.groupby("project_type")["risk_score"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=pt.index, y=pt.values, marker_color="#e67e22",
                               text=[f"{v:.2f}" for v in pt.values], textposition="outside"))
        fig.update_layout(template="sih", height=220, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.markdown("**Top districts by average risk**")
        d = df.groupby("district")["risk_score"].mean().nlargest(12).sort_values()
        fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color="#8e44ad",
                               text=[f"{v:.2f}" for v in d.values], textposition="outside"))
        fig.update_layout(template="sih", height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Heat map — delay probability by district × stage**")
    heat = df.groupby("district")[[f"{s}_prob" for s in STAGES]].mean()
    heat["_m"] = heat.mean(axis=1)
    heat = heat.sort_values("_m", ascending=False).drop(columns="_m")
    fig = go.Figure(go.Heatmap(z=heat.values, x=STAGES, y=heat.index, colorscale="YlOrRd",
                               colorbar=dict(title="P(delay)"),
                               text=np.round(heat.values, 2), texttemplate="%{text}"))
    fig.update_layout(template="sih", height=max(320, 14 * len(heat)),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Historical mean overrun per stage** (completed projects)")
    hist = pd.read_parquet(ROOT / "data" / "generated" / "stage_timelines_historical.parquet")
    hm = hist.groupby("stage")["delay_days"].mean().reindex(STAGES)
    fig = go.Figure(go.Bar(x=hm.index, y=hm.values, marker_color="#c0392b",
                           text=[f"{v:.0f}d" for v in hm.values], textposition="outside"))
    fig.update_layout(template="sih", height=240, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Compare view
# --------------------------------------------------------------------------- #

def view_compare(df: pd.DataFrame) -> None:
    st.subheader("Comparative analytics")
    mode = st.radio("Compare", ["Districts", "Projects"], horizontal=True)
    col = "district" if mode == "Districts" else "project_id"
    entities = sorted(df[col].unique())
    if len(entities) < 2:
        st.info("Need at least two entities to compare.")
        return
    a = st.selectbox("Entity A", entities, index=0)
    others = [e for e in entities if e != a]
    b = st.selectbox("Entity B", others, index=0)

    def summary(e):
        sub = df[df[col] == e]
        return {
            "n": len(sub),
            "risk": sub["risk_score"].mean(),
            "red": int((sub["risk_level"] == "RED").sum()),
            "yel": int((sub["risk_level"] == "YELLOW").sum()),
            "grn": int((sub["risk_level"] == "GREEN").sum()),
            "overrun": sub["expected_overrun_days"].mean(),
            "maxprob": sub["max_delay_prob"].mean(),
            "stages": [float(sub[f"{s}_prob"].mean()) for s in STAGES],
        }

    sa, sb = summary(a), summary(b)

    kpi_row([
        (f"{a} avg risk", f"{sa['risk']:.3f}", None, ui.NAVY),
        (f"{b} avg risk", f"{sb['risk']:.3f}", None, ui.STEEL),
        ("Parcels (A/B)", f"{sa['n']:,} / {sb['n']:,}", None, ui.NAVY),
        ("Avg overrun (A/B)", f"{sa['overrun']:.0f} / {sb['overrun']:.0f} d", None, ui.STEEL),
    ])

    fig = go.Figure()
    fig.add_trace(go.Bar(name=a, x=STAGES, y=sa["stages"], marker_color="#2E75B6"))
    fig.add_trace(go.Bar(name=b, x=STAGES, y=sb["stages"], marker_color="#E81123"))
    fig.update_layout(barmode="group", template="sih", height=300,
                      yaxis=dict(range=[0, 1], title="P(delay)"), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for e, s in ((a, sa), (b, sb)):
        fig.add_trace(go.Bar(name=e, x=["RED", "YELLOW", "GREEN"],
                             y=[s["red"], s["yel"], s["grn"]],
                             marker_color=[COLORS["RED"], COLORS["YELLOW"], COLORS["GREEN"]]))
    fig.update_layout(barmode="group", template="sih", height=280,
                      yaxis_title="parcels", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    theme = st.session_state.get("theme", "Light")
    inject_css(theme)
    setup_plotly(theme)
    hero(ui.NAME, ui.TAGLINE, ui.SUB)

    df = load_portfolio()
    artifacts = load_artifacts_cached()

    with st.sidebar:
        brand_sidebar()
        st.markdown(
            "<div style='margin:-4px 0 12px 2px; font-size:.78rem; color:#5a6b7b;'>"
            "States: " + " · ".join(f"<b>{s}</b>" for s in ui.STATES) + "</div>",
            unsafe_allow_html=True)
        st.markdown("## Role (mock)")
        role = st.selectbox("Select role", ["Admin", "Officer", "Viewer"])
        is_admin = role == "Admin"
        is_viewer = role == "Viewer"
        st.session_state["role"] = role

        st.markdown("## Navigation")
        nav = ["Portfolio", "Project", "Detail", "Map"]
        if not is_viewer:
            nav = ["Portfolio", "Project", "New Project", "Detail", "What-if", "Alerts",
                   "Area of Interest", "Trends", "Compare", "Map"]
        # initialize nav once; nav is NOT widget-bound, so nav_to() can update it safely
        if "nav" not in st.session_state:
            st.session_state["nav"] = nav[0]
        current_nav = st.session_state["nav"]
        if current_nav not in nav:
            current_nav = nav[0]
            st.session_state["nav"] = current_nav

        # unkeyed radio: index always follows nav (so programmatic navigation stays in
        # sync), and a user click updates nav afterwards (no StreamlitAPIException).
        selected = st.radio("Go to", nav, index=nav.index(current_nav))
        if selected != current_nav:
            st.session_state["nav"] = selected

        # theme toggle (keyed to session_state so CSS/plotly re-theme on change)
        st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme")

        if is_admin and st.button("Refresh portfolio cache"):
            refresh()
        if is_admin and st.button("Reset user-created projects"):
            user_projects.reset_user_data()
            load_portfolio.clear()
            st.rerun()
        st.caption(f"Logged in as **{role}**")

    page = st.session_state["nav"]
    if page == "Portfolio":
        view_portfolio(df)
    elif page == "Project":
        view_project(df)
    elif page == "New Project":
        view_new_project(artifacts)
    elif page == "Detail":
        view_detail(df, artifacts)
    elif page == "What-if":
        view_whatif(df, artifacts)
    elif page == "Alerts":
        view_alerts(df)
    elif page == "Area of Interest":
        view_area(df)
    elif page == "Trends":
        view_trends(df)
    elif page == "Compare":
        view_compare(df)
    elif page == "Map":
        view_map(df)

    footer(metrics_summary_footer())


if __name__ == "__main__":
    main()
