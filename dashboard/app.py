"""GRAMPULSE — AI-powered early-warning dashboard.

Run:
    streamlit run dashboard/app.py

The dashboard uses the existing GRAMPULSE analytical pipeline and adds:
- poster-style KPI dashboard
- interactive risk geography when village coordinates are supplied
- village drill-down with anomaly markers
- DMS component visualization
- ranked alerts and explainability
- optional VB-G RAM G employment-demand CSV upload
- optional village coordinates CSV upload
"""

import os
import sys
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from run_pipeline import run  # noqa: E402
from config import ACT_EFFECTIVE_DATE, ACT_NAME, GUARANTEED_DAYS, PROGRAM_SHORT_NAME  # noqa: E402

st.set_page_config(
    page_title="GRAMPULSE | VB-G RAM G Early Warning",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_ORDER = ["LOW", "MODERATE", "HIGH", "EXTREME"]
RISK_COLORS = {
    "LOW": "#22c55e",
    "MODERATE": "#eab308",
    "HIGH": "#f97316",
    "EXTREME": "#dc2626",
}

st.markdown(
    """
    <style>
    .stApp { background: #071a33; }
    [data-testid="stHeader"] { background: #071a33; }
    [data-testid="stMainBlockContainer"] { background: #071a33; }
    [data-testid="stSidebar"] { background: #0b1f3a; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .hero {
        background: linear-gradient(135deg,#09203f,#174a75);
        padding: 26px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(9,32,63,.18);
    }
    .hero h1 { margin: 0; font-size: 38px; letter-spacing: .5px; }
    .hero p { margin: 6px 0 0; opacity: .88; font-size: 16px; }
    .kpi {
        background: #0d2747;
        border: 1px solid #1e4770;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,.22);
    }
    .kpi-title { color:#a9bfd6; font-size:13px; font-weight:600; }
    .kpi-value { color:#ffffff; font-size:28px; font-weight:800; margin-top:4px; }
    .risk-card {
        border-radius: 14px;
        padding: 14px 16px;
        color: white;
        min-height: 94px;
    }
    .section-title { font-size: 24px; font-weight: 800; color:#ffffff; margin: 12px 0; }
    .small-muted { color:#a9bfd6; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _default_data_path():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    current = os.path.join(base, "vb_gram_g_raw.csv")
    legacy = os.path.join(base, "mgnrega_raw.csv")
    return current if os.path.exists(current) else legacy


@st.cache_data(show_spinner="Running GRAMPULSE analytics...")
def load_pipeline(input_path: str, mtime: float):
    return run(input_path)


def kpi(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="small-muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk):
    return f"{risk}"


def sidebar():
    st.sidebar.markdown("# 🌱 GRAMPULSE")
    st.sidebar.caption("Predict Before the Crisis")
    st.sidebar.divider()

    default_path = _default_data_path()
    uploaded = st.sidebar.file_uploader("VB-G RAM G employment-demand CSV", type=["csv"])

    if uploaded:
        data = uploaded.getvalue()
        tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "_uploaded.csv"))
        with open(tmp_path, "wb") as f:
            f.write(data)
        input_path = tmp_path
        st.sidebar.success("Custom VB-G RAM G dataset loaded")
    else:
        input_path = default_path
        st.sidebar.info("Using bundled sample data")

    coords = st.sidebar.file_uploader(
        "Village coordinates CSV (optional)",
        type=["csv"],
        help="Columns required: village, latitude, longitude. Optional: village_id, district, block.",
    )

    coords_df = None
    if coords:
        try:
            coords_df = pd.read_csv(BytesIO(coords.getvalue()))
            required = {"village", "latitude", "longitude"}
            if not required.issubset(coords_df.columns):
                st.sidebar.error("Coordinates CSV needs village, latitude and longitude columns.")
                coords_df = None
            else:
                st.sidebar.success(f"Loaded {len(coords_df)} coordinate records")
        except Exception as exc:
            st.sidebar.error(f"Could not read coordinates: {exc}")

    return input_path, coords_df


def overview(latest):
    st.markdown('<div class="section-title">Distress Situation Overview</div>', unsafe_allow_html=True)
    counts = latest["risk"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    extreme_high = int(counts["EXTREME"] + counts["HIGH"])
    avg_dms = latest["dms"].mean() if len(latest) else 0

    cols = st.columns(6)
    vals = [
        ("Villages analyzed", len(latest), "Current latest week"),
        ("Extreme", counts["EXTREME"], "Immediate review"),
        ("High", counts["HIGH"], "Priority action"),
        ("Moderate", counts["MODERATE"], "Monitor closely"),
        ("Low", counts["LOW"], "Routine monitoring"),
        ("High + Extreme", extreme_high, "Active alerts"),
    ]
    for col, (title, value, subtitle) in zip(cols, vals):
        with col:
            kpi(title, f"{value}", subtitle)

    left, right = st.columns([1.15, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=RISK_ORDER,
            y=[counts[r] for r in RISK_ORDER],
            marker_color=[RISK_COLORS[r] for r in RISK_ORDER],
            text=[counts[r] for r in RISK_ORDER],
            textposition="outside",
        ))
        fig.update_layout(
            title="Current risk distribution",
            height=360,
            margin=dict(l=20, r=20, t=55, b=20),
            yaxis_title="Villages",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### System signal")
        st.metric("Average DMS", f"{avg_dms:.1f} / 100")
        st.progress(min(max(avg_dms / 100, 0), 1))
        st.info(
            "GRAMPULSE combines employment-demand growth speed, historical intensity and anomaly signals "
            "into the Distress Momentum Score (DMS)."
        )

        alerts = latest[latest["risk"].isin(["HIGH", "EXTREME"])].sort_values("dms", ascending=False)
        if not alerts.empty:
            st.markdown("### 🚨 Top active alerts")
            for _, row in alerts.head(5).iterrows():
                st.markdown(
                    f"**{row['risk_emoji']} {row['village']}** — DMS **{row['dms']:.1f}** ({row['risk']})"
                )


def risk_map(latest, coords_df):
    st.markdown('<div class="section-title">Geographic Risk Map</div>', unsafe_allow_html=True)

    if coords_df is None:
        st.warning(
            "No village coordinates were supplied. Upload a CSV containing `village`, `latitude`, "
            "and `longitude` in the sidebar to activate the geographic risk map."
        )
        st.markdown("**Expected format:**")
        st.code("village,latitude,longitude\nThenmalai,11.53,78.60\nNallur,11.72,78.14")
    else:
        map_df = latest.merge(coords_df, on="village", how="inner", suffixes=("", "_coord"))
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"])

        if map_df.empty:
            st.error("No village names in the coordinates file matched the current dataset.")
        else:
            fig = go.Figure()
            for risk in RISK_ORDER:
                part = map_df[map_df["risk"] == risk]
                if part.empty:
                    continue
                fig.add_trace(go.Scattergeo(
                    lat=part["latitude"],
                    lon=part["longitude"],
                    mode="markers",
                    name=risk,
                    text=part["village"],
                    customdata=part[["dms", "demand", "recommendation"]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "DMS: %{customdata[0]:.1f}<br>"
                        "Employment demand: %{customdata[1]:.0f}<br>"
                        "Risk: " + risk + "<extra></extra>"
                    ),
                    marker=dict(size=12 if risk in ["HIGH", "EXTREME"] else 9,
                                color=RISK_COLORS[risk], line=dict(width=1, color="white")),
                ))
            fig.update_geos(
                scope="asia",
                showcountries=True,
                showsubunits=True,
                center=dict(lat=11.0, lon=78.5),
                projection_scale=7,
            )
            fig.update_layout(height=540, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Administrative drill-down")
    c1, c2, c3, c4 = st.columns(4)
    state = c1.selectbox("State", ["All"] + sorted(latest["state"].dropna().unique().tolist()))
    f = latest if state == "All" else latest[latest["state"] == state]
    district = c2.selectbox("District", ["All"] + sorted(f["district"].dropna().unique().tolist()))
    f = f if district == "All" else f[f["district"] == district]
    block = c3.selectbox("Block", ["All"] + sorted(f["block"].dropna().unique().tolist()))
    f = f if block == "All" else f[f["block"] == block]
    risks = c4.multiselect("Risk", RISK_ORDER, default=RISK_ORDER)
    f = f[f["risk"].isin(risks)].sort_values("dms", ascending=False)

    display = f[["village", "panchayat", "block", "district", "demand", "dms", "risk"]].copy()
    display["dms"] = display["dms"].round(1)
    st.dataframe(display, use_container_width=True, hide_index=True)


def village_detail(result, latest):
    st.markdown('<div class="section-title">Village Detail & Early-Warning Signal</div>', unsafe_allow_html=True)
    names = latest.sort_values("dms", ascending=False)["village"].tolist()
    selected = st.selectbox("Village", names)
    row = latest[latest["village"] == selected].iloc[0]
    history = result[result["village_id"] == row["village_id"]].sort_values("date")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Distress Momentum Score", f"{row['dms']:.1f}", f"{row['risk']} risk")
    with c2:
        kpi("Current demand", f"{row['demand']:.0f}", "latest week")
    with c3:
        growth = row["weekly_growth"]
        kpi("Weekly growth", f"{growth:.1f}%" if pd.notna(growth) else "—", "speed signal")
    with c4:
        kpi("Consecutive growth", f"{int(row['consecutive_growth_weeks'])}", "weeks")

    st.markdown("### Why is this village at risk now?")
    st.info(row["explanation"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["demand"], mode="lines+markers", name="Weekly employment demand"
    ))
    anomalies = history[history["anomaly_detected"] == True]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["demand"], mode="markers",
            name="Detected anomaly", marker=dict(size=12, symbol="x", color=RISK_COLORS["EXTREME"])
        ))
    fig.update_layout(height=410, title=f"{selected} — historical demand trend", yaxis_title="Employment demand")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### DMS composition")
    scores = [float(row["speed_score"]), float(row["intensity_score"]), float(row["abnormality_score"])]
    fig2 = go.Figure(go.Bar(
        x=scores,
        y=["Speed", "Intensity", "Abnormality"],
        orientation="h",
        text=[f"{x:.1f}" for x in scores],
        textposition="outside",
    ))
    fig2.update_xaxes(range=[0, 105], title="Score (0–100)")
    fig2.update_layout(height=250, margin=dict(l=20, r=30, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recommended government action")
    if row["risk"] == "EXTREME":
        st.error(row["recommendation"])
    elif row["risk"] == "HIGH":
        st.warning(row["recommendation"])
    else:
        st.info(row["recommendation"])


def ranking(latest):
    st.markdown('<div class="section-title">Village Ranking</div>', unsafe_allow_html=True)
    risk = st.multiselect("Filter risk", RISK_ORDER, default=RISK_ORDER, key="ranking_risk")
    ranked = latest[latest["risk"].isin(risk)].sort_values("dms", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    ranked["DMS"] = ranked["dms"].round(1)
    ranked["Weekly growth"] = ranked["weekly_growth"].round(1)
    ranked["Risk"] = ranked["risk"]
    out = ranked[["Rank", "village", "district", "block", "DMS", "Weekly growth", "Risk"]]
    st.dataframe(out, use_container_width=True, hide_index=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranking CSV", csv, "grampulse_rankings.csv", "text/csv")


def explainability(latest):
    st.markdown('<div class="section-title">Explainability & Alerts</div>', unsafe_allow_html=True)
    flagged = latest[latest["risk"].isin(["HIGH", "EXTREME"])].sort_values("dms", ascending=False)
    if flagged.empty:
        st.success("No HIGH or EXTREME villages are currently flagged.")
        return

    for _, row in flagged.iterrows():
        with st.expander(f"{row['risk_emoji']} {row['village']} — DMS {row['dms']:.1f} — {row['risk']}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Speed", f"{row['speed_score']:.1f}")
            c2.metric("Intensity", f"{row['intensity_score']:.1f}")
            c3.metric("Abnormality", f"{row['abnormality_score']:.1f}")
            st.markdown("**Why flagged**")
            st.write(row["explanation"])
            st.markdown("**Recommended action**")
            st.write(row["recommendation"])

    st.download_button(
        "Download active alerts",
        flagged.to_csv(index=False).encode("utf-8"),
        "grampulse_alerts.csv",
        "text/csv",
    )


def main():
    input_path, coords_df = sidebar()
    if not os.path.exists(input_path):
        st.error("VB-G RAM G employment-demand input data was not found.")
        st.stop()

    mtime = os.path.getmtime(input_path)
    result, latest = load_pipeline(input_path, mtime)

    st.markdown(
        f"""
        <div class="hero">
            <h1>🌱 GRAMPULSE</h1>
            <p>AI-Powered Early Warning System for Rural Distress using VB-G RAM G employment-demand data</p>
            <p><b>Predict Before the Crisis.</b> Detect unusual employment-demand momentum before visible distress becomes a crisis.</p>
            <p><b>Current framework:</b> {PROGRAM_SHORT_NAME} • {ACT_NAME} • In force from {ACT_EFFECTIVE_DATE.strftime("%d %B %Y")} • {GUARANTEED_DAYS} days statutory employment guarantee</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📊 Overview",
        "🗺️ Risk Map",
        "🏘️ Village Detail",
        "🏆 Ranking",
        "🚨 Explainability & Alerts",
    ])
    with tabs[0]:
        overview(latest)
    with tabs[1]:
        risk_map(latest, coords_df)
    with tabs[2]:
        village_detail(result, latest)
    with tabs[3]:
        ranking(latest)
    with tabs[4]:
        explainability(latest)

    st.divider()
    st.caption(
        "GRAMPULSE is an early-warning analytics prototype. A risk score indicates an unusual demand pattern "
        "that warrants administrative review; it does not by itself establish that a village is in distress."
    )


if __name__ == "__main__":
    main()
