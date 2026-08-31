"""SIH26017 Streamlit dashboard (main demo path).

Offline-capable: reads the precomputed portfolio cache and calls predict.py directly
(no network, no tiles). Six views: Portfolio, Detail, What-if, Alerts, Map, plus a
mock role-switcher with functional gating.

Run:
  .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402
from predict import load_artifacts, score_parcel, risk_level  # noqa: E402

PORTFOLIO_PATH = ROOT / "data" / "generated" / "portfolio_scores.parquet"

COLORS = {"RED": "#e74c3c", "YELLOW": "#f1c40f", "GREEN": "#2ecc71"}
EMOJI = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}

st.set_page_config(page_title="SIH26017 — Land Delay Early Warning", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #

@st.cache_data
def load_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_PATH.exists():
        predict.refresh_portfolio()
    return pd.read_parquet(PORTFOLIO_PATH)


@st.cache_resource
def load_artifacts_cached() -> dict:
    return load_artifacts()


@st.cache_data
def load_districts() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "generated" / "districts.parquet")


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


# --------------------------------------------------------------------------- #
# View: Portfolio
# --------------------------------------------------------------------------- #

def view_portfolio(df: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("Portfolio risk table")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Live parcels", f"{len(df):,}")
    k2.metric("High risk (RED)", f"{(df['risk_level'] == 'RED').sum():,}")
    k3.metric("Medium (YELLOW)", f"{(df['risk_level'] == 'YELLOW').sum():,}")
    k4.metric("Low (GREEN)", f"{(df['risk_level'] == 'GREEN').sum():,}")
    k5.metric("Avg risk", f"{df['risk_score'].mean():.2f}")

    level = st.selectbox("Group by", ["district", "village", "project_id"], index=0)
    g = df.groupby(level).agg(
        n_parcels=("parcel_id", "size"),
        avg_risk=("risk_score", "mean"),
        wavg_risk=("risk_score", lambda s: np.average(s, weights=df.loc[s.index, "area_sqm"])),
        avg_overrun=("expected_overrun_days", "mean"),
        red_count=("risk_level", lambda s: (s == "RED").sum()),
    ).reset_index().sort_values("wavg_risk", ascending=False)

    g["level"] = g["wavg_risk"].map(risk_level)
    g["emoji"] = g["level"].map(EMOJI)
    disp = g[["emoji", level, "n_parcels", "wavg_risk", "avg_overrun", "red_count", "level"]]
    disp.columns = ["", level, "n_parcels", "area-wtd risk", "avg overrun (d)", "RED count", "level"]
    st.dataframe(disp, width='stretch', hide_index=True)

    sel = st.selectbox("Drill into group", g[level].tolist())
    subset = df[df[level] == sel].sort_values("risk_score", ascending=False)
    st.markdown(f"**{len(subset)} parcels in {level} `{sel}`** (top 50 by risk)")
    cols = ["parcel_id", "risk_level", "risk_score", "expected_overrun_days",
            "current_stage", "overrun_while_ongoing_days", "court_stay", "compensation_status"]
    st.dataframe(subset[cols].head(50), width='stretch', hide_index=True)

    if not subset.empty:
        if st.button("Open top-risk parcel in Detail / What-if", key="open_top"):
            st.session_state["parcel_id"] = subset.iloc[0]["parcel_id"]
            st.rerun()


# --------------------------------------------------------------------------- #
# View: Detail
# --------------------------------------------------------------------------- #

def _pick_parcel(df: pd.DataFrame) -> str:
    if "parcel_id" in st.session_state:
        default_idx = df.index[df["parcel_id"] == st.session_state["parcel_id"]]
    else:
        default_idx = df.index[df["risk_score"] == df["risk_score"].max()]
    idx = int(default_idx[0]) if len(default_idx) else 0
    options = df["parcel_id"].tolist()
    return st.selectbox("Parcel", options, index=idx, key="parcel_selector")


def _stage_bars(contract: dict) -> go.Figure:
    stages = contract["stages"]
    names = list(stages.keys())
    probs = [stages[s]["delay_prob"] for s in names]
    overruns = [stages[s]["expected_overrun"] for s in names]
    statutory = [stages[s]["statutory_days"] for s in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="delay probability", x=names, y=probs, marker_color="#3498db",
                         yaxis="y", text=[f"{p:.0%}" for p in probs], textposition="outside"))
    fig.add_trace(go.Bar(name="expected overrun (days)", x=names, y=overruns, marker_color="#e67e22",
                         yaxis="y2", opacity=0.85))
    fig.add_trace(go.Scatter(name="statutory days", x=names, y=statutory, yaxis="y2",
                             mode="lines+markers", line=dict(dash="dot", color="#555"),
                             marker=dict(size=6)))
    fig.update_layout(
        barmode="group", template="plotly_white", height=380,
        yaxis=dict(title="P(delay)", range=[0, 1]),
        yaxis2=dict(title="days", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _shap_bars(contract: dict) -> go.Figure:
    tf = contract["top_factors"]
    names = [t[0] for t in tf][::-1]
    vals = [t[1] for t in tf][::-1]
    colors = [COLORS["RED"] if v >= 0 else "#3498db" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(template="plotly_white", height=340,
                      xaxis_title="|SHAP| impact (higher = bigger driver)")
    return fig


def view_detail(df: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("Parcel detail")
    pid = _pick_parcel(df)
    row = df[df["parcel_id"] == pid].iloc[0]
    features = {c: row[c] for c in predict.DEFAULTS}
    timeline = load_live_timeline(pid)
    contract = score_parcel(features, artifacts=artifacts, timeline=timeline, parcel_id=pid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{contract['risk_score']:.3f}")
    c2.metric("Level", contract["risk_level"])
    c3.metric("Expected overrun", f"{contract['expected_overrun_days']:.0f} days")
    c4.metric("Already overrun (ongoing)",
              f"{contract['overrun_while_ongoing_days']:.0f} d"
              if contract["overrun_while_ongoing_days"] else "—")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Per-stage delay probability vs statutory clock**")
        st.plotly_chart(_stage_bars(contract), width='stretch')
    with right:
        st.markdown("**Why is this parcel at risk? (SHAP)**")
        st.plotly_chart(_shap_bars(contract), width='stretch')
        st.markdown("**Recommended actions**")
        for a in contract["recommended_actions"]:
            badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}[a["priority_label"]]
            st.markdown(f"- {badge} **{a['factor']}** — {a['action']}")


# --------------------------------------------------------------------------- #
# View: What-if
# --------------------------------------------------------------------------- #

def view_whatif(df: pd.DataFrame, artifacts: dict) -> None:
    st.subheader("What-if simulator")
    pid = _pick_parcel(df)
    row = df[df["parcel_id"] == pid].iloc[0]
    features = {c: row[c] for c in predict.DEFAULTS}

    c1, c2 = st.columns(2)
    new_court_stay = c1.selectbox("Court stay", [0, 1], index=int(features["court_stay"]),
                                  help="Toggle to simulate clearing an active court stay.")
    new_comp = c2.selectbox("Compensation status", ["paid", "partial", "pending"],
                            index=["paid", "partial", "pending"].index(features["compensation_status"]),
                            help="Simulate disbursing compensation.")

    changed = (new_court_stay != int(features["court_stay"])) or (new_comp != features["compensation_status"])

    before = score_parcel(features, artifacts=artifacts, parcel_id=pid)
    mutated = {**features, "court_stay": new_court_stay, "compensation_status": new_comp}
    after = score_parcel(mutated, artifacts=artifacts, parcel_id=pid)

    b1, b2, b3 = st.columns(3)
    b1.metric("Risk score (before)", f"{before['risk_score']:.3f}",
              delta=None)
    b2.metric("Risk score (after)", f"{after['risk_score']:.3f}",
              delta=f"{after['risk_score'] - before['risk_score']:+.3f}")
    b3.metric("Expected overrun (after)", f"{after['expected_overrun_days']:.0f} d",
              delta=f"{after['expected_overrun_days'] - before['expected_overrun_days']:+.0f} d")

    if not changed:
        st.info("Change the court-stay or compensation toggles to see the risk move live.")
    else:
        st.success(f"Level {before['risk_level']} → {after['risk_level']}")


# --------------------------------------------------------------------------- #
# View: Alerts
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
        st.dataframe(a.head(200), width='stretch', hide_index=True)
        st.markdown(f"Alert types: {a['alert_type'].value_counts().to_dict()}")


# --------------------------------------------------------------------------- #
# View: Map (offline plotly scatter)
# --------------------------------------------------------------------------- #

def view_map(df: pd.DataFrame) -> None:
    st.subheader("District risk hotspot map (offline)")
    districts = load_districts()
    agg = df.groupby("district").agg(
        n=("parcel_id", "size"),
        wavg_risk=("risk_score", lambda s: np.average(s, weights=df.loc[s.index, "area_sqm"])),
    ).reset_index()
    agg["level"] = agg["wavg_risk"].map(risk_level)
    m = districts.merge(agg, on="district", how="left")

    fig = go.Figure()
    for lvl, color in COLORS.items():
        sub = m[m["level"] == lvl]
        fig.add_trace(go.Scatter(
            x=sub["lon"], y=sub["lat"], mode="markers+text",
            marker=dict(size=sub["n"].clip(8, 30), color=color, line=dict(width=1, color="#333")),
            text=sub["district"], textposition="top center", name=lvl,
            customdata=sub[["wavg_risk", "n"]].round(2).values,
            hovertemplate="%{text}<br>area-wtd risk: %{customdata[0]}<br>parcels: %{customdata[1]}",
        ))
    fig.update_layout(template="plotly_white", height=520,
                      xaxis_title="longitude", yaxis_title="latitude",
                      showlegend=True,
                      title="Himachal Pradesh — district risk (area-weighted)")
    st.plotly_chart(fig, width='stretch')
    st.caption("Offline lat/lon scatter (no tile server) — chosen over Folium to keep the demo fully offline.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.title("SIH26017 — Predictive Analytics for Land Acquisition Delays")
    st.caption("Early-warning system · RFCTLARR 2013 · Himachal Pradesh pilot")

    df = load_portfolio()
    artifacts = load_artifacts_cached()

    with st.sidebar:
        st.markdown("## Role (mock)")
        role = st.selectbox("Select role", ["Admin", "Officer", "Viewer"])
        is_admin = role == "Admin"
        is_viewer = role == "Viewer"

        st.markdown("## Navigation")
        nav = ["Portfolio", "Detail", "Map"]
        if not is_viewer:
            nav = ["Portfolio", "Detail", "What-if", "Alerts", "Map"]
        page = st.radio("Go to", nav)

        if is_admin:
            if st.button("Refresh portfolio cache"):
                refresh()
        st.caption(f"Logged in as **{role}**")

    if page == "Portfolio":
        view_portfolio(df, artifacts)
    elif page == "Detail":
        view_detail(df, artifacts)
    elif page == "What-if":
        view_whatif(df, artifacts)
    elif page == "Alerts":
        view_alerts(df)
    elif page == "Map":
        view_map(df)


if __name__ == "__main__":
    main()
