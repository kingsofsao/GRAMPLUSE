"""GRAMPULSE — AI-powered early-warning dashboard.

Run:
    streamlit run dashboard/app.py

The dashboard uses the GRAMPULSE analytical pipeline and provides:
- dark-blue poster-style KPI dashboard
- VB-G RAM G employment-demand CSV upload with validation
- historical MGNREGA compatibility for baseline/trend analysis
- interactive risk geography when village coordinates are supplied
- village drill-down with anomaly markers
- DMS component visualization
- ranked alerts and explainability
- downloadable ranking/alert outputs

Expected employment-demand CSV columns:
    state, district, block, panchayat, village, date, demand
"""

import hashlib
import os
import sys
import tempfile
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from the project's src directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (  # noqa: E402
    ACT_EFFECTIVE_DATE,
    ACT_NAME,
    EMPLOYMENT_DEMAND_LABEL,
    GUARANTEED_DAYS,
    PROGRAM_SHORT_NAME,
)
from run_pipeline import run  # noqa: E402


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="GRAMPULSE | VB-G RAM G Early Warning",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "state",
    "district",
    "block",
    "panchayat",
    "village",
    "date",
    "demand",
}

RISK_ORDER = ["LOW", "MODERATE", "HIGH", "EXTREME"]

RISK_COLORS = {
    "LOW": "#22c55e",
    "MODERATE": "#eab308",
    "HIGH": "#f97316",
    "EXTREME": "#dc2626",
}


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #071a33;
    }

    [data-testid="stHeader"] {
        background: #071a33;
    }

    [data-testid="stMainBlockContainer"] {
        background: #071a33;
    }

    [data-testid="stSidebar"] {
        background: #0b1f3a;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .hero {
        background: linear-gradient(135deg, #09203f, #174a75);
        padding: 26px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, .18);
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        letter-spacing: .5px;
    }

    .hero p {
        margin: 6px 0 0;
        opacity: .88;
        font-size: 16px;
    }

    .kpi {
        background: #0d2747;
        border: 1px solid #1e4770;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, .22);
        min-height: 105px;
    }

    .kpi-title {
        color: #a9bfd6;
        font-size: 13px;
        font-weight: 600;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
    }

    .risk-card {
        border-radius: 14px;
        padding: 14px 16px;
        color: white;
        min-height: 94px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin: 12px 0;
    }

    .small-muted {
        color: #a9bfd6;
        font-size: 13px;
    }

    .upload-help {
        background: #0d2747;
        border: 1px solid #1e4770;
        border-radius: 10px;
        padding: 10px 12px;
        margin: 8px 0 12px 0;
        color: #dbeafe;
        font-size: 12px;
    }

    .success-box {
        background: #063b35;
        border: 1px solid #0f766e;
        border-radius: 10px;
        padding: 10px 12px;
        color: #d1fae5;
        font-size: 13px;
    }

    .warning-box {
        background: #4a3207;
        border: 1px solid #a16207;
        border-radius: 10px;
        padding: 10px 12px;
        color: #fef3c7;
        font-size: 13px;
    }

    code {
        color: #dbeafe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def _default_data_path():
    """Return the preferred bundled current-data file, with legacy fallback."""
    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    )
    current = os.path.join(base, "vb_gram_g_raw.csv")
    legacy = os.path.join(base, "mgnrega_raw.csv")
    return current if os.path.exists(current) else legacy


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for validation without changing the original file."""
    copy = df.copy()
    copy.columns = [str(c).strip().lower() for c in copy.columns]
    return copy


def validate_employment_csv(file_bytes: bytes):
    """Validate an employment-demand CSV before sending it to the pipeline.

    Returns:
        (True, dataframe, "") when valid.
        (False, dataframe_or_none, error_message) when invalid.
    """
    try:
        if not file_bytes:
            return False, None, "The uploaded CSV is empty."

        df = pd.read_csv(BytesIO(file_bytes))

        if df.empty:
            return False, df, "The uploaded CSV contains no data rows."

        normalized = _normalize_column_names(df)
        missing = sorted(REQUIRED_COLUMNS - set(normalized.columns))

        if missing:
            actual = ", ".join(str(c) for c in df.columns)
            required = ", ".join(sorted(REQUIRED_COLUMNS))
            missing_text = ", ".join(missing)
            message = (
                "Invalid VB-G RAM G employment-demand CSV. "
                f"Missing required column(s): {missing_text}. "
                f"Required columns are: {required}. "
                f"Uploaded columns were: {actual}."
            )
            return False, df, message

        # Validate the two fields most likely to cause downstream failures.
        if normalized["date"].dropna().empty:
            return False, df, "The 'date' column contains no usable values."

        numeric_demand = pd.to_numeric(normalized["demand"], errors="coerce")
        if numeric_demand.notna().sum() == 0:
            return False, df, "The 'demand' column contains no numeric values."

        return True, df, ""

    except pd.errors.EmptyDataError:
        return False, None, "The uploaded CSV is empty."
    except pd.errors.ParserError as exc:
        return False, None, f"The uploaded CSV could not be parsed: {exc}"
    except Exception as exc:
        return False, None, f"Could not validate the uploaded CSV: {exc}"


def _save_uploaded_csv(file_bytes: bytes) -> str:
    """Save a validated upload to a unique temporary CSV path."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    )
    os.makedirs(base_dir, exist_ok=True)

    path = os.path.join(base_dir, f"_uploaded_{digest}.csv")
    if not os.path.exists(path):
        with open(path, "wb") as file:
            file.write(file_bytes)
    return path


def _sample_csv_bytes() -> bytes:
    """Return a small valid template for users who need the input format."""
    sample = pd.DataFrame(
        [
            {
                "state": "Tamil Nadu",
                "district": "Salem",
                "block": "Attur",
                "panchayat": "Example Panchayat",
                "village": "Example Village",
                "date": "2026-07-05",
                "demand": 120,
            },
            {
                "state": "Tamil Nadu",
                "district": "Salem",
                "block": "Attur",
                "panchayat": "Example Panchayat",
                "village": "Example Village",
                "date": "2026-07-12",
                "demand": 145,
            },
        ]
    )
    return sample.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner="Running GRAMPULSE analytics...")
def load_pipeline(input_path: str, mtime: float):
    """Run the analytical pipeline with Streamlit caching."""
    return run(input_path)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


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


def sidebar():
    """Render sidebar controls and return input path + coordinates dataframe."""
    st.sidebar.markdown("# 🌱 GRAMPULSE")
    st.sidebar.caption("Predict Before the Crisis")
    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Employment-demand data upload
    # ------------------------------------------------------------------
    default_path = _default_data_path()

    st.sidebar.markdown("### VB-G RAM G employment-demand data")
    st.sidebar.markdown(
        """
        <div class="upload-help">
        <b>Required CSV columns</b><br>
        state, district, block, panchayat, village, date, demand
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.sidebar.file_uploader(
        "Upload employment-demand CSV",
        type=["csv"],
        key="employment_csv",
        help=(
            "Upload village-level weekly/daily employment-demand data. "
            "Do not upload rankings.csv or grampulse_alerts.csv here."
        ),
    )

    input_path = default_path
    upload_is_valid = True

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        valid, uploaded_df, error_message = validate_employment_csv(file_bytes)

        if valid:
            input_path = _save_uploaded_csv(file_bytes)
            st.sidebar.success(
                f"Valid dataset loaded: {len(uploaded_df):,} rows"
            )

            # Show a compact validation summary.
            normalized = _normalize_column_names(uploaded_df)
            village_count = normalized["village"].nunique(dropna=True)
            st.sidebar.caption(
                f"{village_count:,} villages • {normalized['date'].notna().sum():,} dated rows"
            )
        else:
            upload_is_valid = False
            st.sidebar.error("Invalid employment-demand CSV")
            st.sidebar.caption(error_message)
            st.sidebar.download_button(
                "Download valid CSV template",
                data=_sample_csv_bytes(),
                file_name="vb_gram_g_employment_demand_template.csv",
                mime="text/csv",
                key="download_template",
            )

    if uploaded is None:
        if os.path.basename(default_path).lower() == "vb_gram_g_raw.csv":
            st.sidebar.info("Using bundled VB-G RAM G sample data")
        else:
            st.sidebar.warning(
                "Bundled VB-G RAM G data was not found; using the legacy MGNREGA file."
            )

    # ------------------------------------------------------------------
    # Coordinates upload
    # ------------------------------------------------------------------
    st.sidebar.markdown("### Village coordinates")
    coords = st.sidebar.file_uploader(
        "Village coordinates CSV (optional)",
        type=["csv"],
        key="coordinates_csv",
        help="Required: village, latitude, longitude. Optional: village_id, district, block.",
    )

    coords_df = None
    if coords is not None:
        try:
            coords_df = pd.read_csv(BytesIO(coords.getvalue()))
            coords_df.columns = [str(c).strip().lower() for c in coords_df.columns]
            required = {"village", "latitude", "longitude"}

            if not required.issubset(coords_df.columns):
                missing = sorted(required - set(coords_df.columns))
                st.sidebar.error(
                    "Coordinates CSV is missing: " + ", ".join(missing)
                )
                coords_df = None
            else:
                coords_df["latitude"] = pd.to_numeric(
                    coords_df["latitude"], errors="coerce"
                )
                coords_df["longitude"] = pd.to_numeric(
                    coords_df["longitude"], errors="coerce"
                )
                coords_df = coords_df.dropna(
                    subset=["village", "latitude", "longitude"]
                )
                st.sidebar.success(
                    f"Loaded {len(coords_df):,} coordinate records"
                )
        except Exception as exc:
            st.sidebar.error(f"Could not read coordinates: {exc}")
            coords_df = None

    # If the user supplied an invalid employment file, do not run the
    # analytical pipeline. This prevents the raw ValueError traceback shown
    # in the deployed app when an output file such as grampulse_alerts.csv
    # is accidentally uploaded as input.
    if not upload_is_valid:
        input_path = None

    return input_path, coords_df


# -----------------------------------------------------------------------------
# Dashboard sections
# -----------------------------------------------------------------------------


def overview(latest):
    st.markdown(
        '<div class="section-title">Distress Situation Overview</div>',
        unsafe_allow_html=True,
    )

    if latest.empty:
        st.warning("No village-level results are available for the current dataset.")
        return

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
        fig.add_trace(
            go.Bar(
                x=RISK_ORDER,
                y=[counts[r] for r in RISK_ORDER],
                marker_color=[RISK_COLORS[r] for r in RISK_ORDER],
                text=[counts[r] for r in RISK_ORDER],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Current risk distribution",
            height=360,
            margin=dict(l=20, r=20, t=55, b=20),
            yaxis_title="Villages",
            showlegend=False,
            paper_bgcolor="#071a33",
            plot_bgcolor="#071a33",
            font=dict(color="#ffffff"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### System signal")
        st.metric("Average DMS", f"{avg_dms:.1f} / 100")
        st.progress(min(max(avg_dms / 100, 0), 1))
        st.info(
            "GRAMPULSE combines employment-demand growth speed, historical intensity "
            "and anomaly signals into the Distress Momentum Score (DMS)."
        )

        alerts = latest[latest["risk"].isin(["HIGH", "EXTREME"])].sort_values(
            "dms", ascending=False
        )
        if not alerts.empty:
            st.markdown("### 🚨 Top active alerts")
            for _, row in alerts.head(5).iterrows():
                st.markdown(
                    f"**{row['risk_emoji']} {row['village']}** — "
                    f"DMS **{row['dms']:.1f}** ({row['risk']})"
                )


def risk_map(latest, coords_df):
    st.markdown(
        '<div class="section-title">Geographic Risk Map</div>',
        unsafe_allow_html=True,
    )

    if coords_df is None:
        st.warning(
            "No village coordinates were supplied. Upload a CSV containing "
            "`village`, `latitude`, and `longitude` in the sidebar to activate the map."
        )
        st.markdown("**Expected format:**")
        st.code(
            "village,latitude,longitude\n"
            "Thenmalai,11.53,78.60\n"
            "Nallur,11.72,78.14"
        )
    else:
        map_df = latest.merge(
            coords_df,
            on="village",
            how="inner",
            suffixes=("", "_coord"),
        )
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"])

        if map_df.empty:
            st.error(
                "No village names in the coordinates file matched the current dataset. "
                "Check spelling and village naming consistency."
            )
        else:
            fig = go.Figure()
            for risk in RISK_ORDER:
                part = map_df[map_df["risk"] == risk]
                if part.empty:
                    continue

                fig.add_trace(
                    go.Scattergeo(
                        lat=part["latitude"],
                        lon=part["longitude"],
                        mode="markers",
                        name=risk,
                        text=part["village"],
                        customdata=part[
                            ["dms", "demand", "recommendation"]
                        ],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "DMS: %{customdata[0]:.1f}<br>"
                            f"{EMPLOYMENT_DEMAND_LABEL}: %{{customdata[1]:.0f}}<br>"
                            f"Risk: {risk}<br>"
                            "<extra></extra>"
                        ),
                        marker=dict(
                            size=12 if risk in ["HIGH", "EXTREME"] else 9,
                            color=RISK_COLORS[risk],
                            line=dict(width=1, color="white"),
                        ),
                    )
                )

            fig.update_geos(
                scope="asia",
                showcountries=True,
                showsubunits=True,
                center=dict(lat=11.0, lon=78.5),
                projection_scale=7,
                bgcolor="#071a33",
            )
            fig.update_layout(
                height=540,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#071a33",
                font=dict(color="#ffffff"),
                legend=dict(font=dict(color="#ffffff")),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Administrative drill-down")
    c1, c2, c3, c4 = st.columns(4)

    state_options = ["All"] + sorted(
        latest["state"].dropna().astype(str).unique().tolist()
    )
    state = c1.selectbox("State", state_options, key="map_state")
    filtered = latest if state == "All" else latest[latest["state"] == state]

    district_options = ["All"] + sorted(
        filtered["district"].dropna().astype(str).unique().tolist()
    )
    district = c2.selectbox("District", district_options, key="map_district")
    filtered = (
        filtered
        if district == "All"
        else filtered[filtered["district"] == district]
    )

    block_options = ["All"] + sorted(
        filtered["block"].dropna().astype(str).unique().tolist()
    )
    block = c3.selectbox("Block", block_options, key="map_block")
    filtered = filtered if block == "All" else filtered[filtered["block"] == block]

    risks = c4.multiselect(
        "Risk",
        RISK_ORDER,
        default=RISK_ORDER,
        key="map_risks",
    )
    filtered = filtered[filtered["risk"].isin(risks)].sort_values(
        "dms", ascending=False
    )

    display_columns = [
        "village",
        "panchayat",
        "block",
        "district",
        "demand",
        "dms",
        "risk",
    ]
    display = filtered[display_columns].copy()
    display["dms"] = display["dms"].round(1)
    st.dataframe(display, use_container_width=True, hide_index=True)


def village_detail(result, latest):
    st.markdown(
        '<div class="section-title">Village Detail & Early-Warning Signal</div>',
        unsafe_allow_html=True,
    )

    if latest.empty:
        st.warning("No village results are available.")
        return

    names = latest.sort_values("dms", ascending=False)["village"].tolist()
    selected = st.selectbox("Village", names, key="village_detail")
    row = latest[latest["village"] == selected].iloc[0]
    history = result[result["village_id"] == row["village_id"]].sort_values("date")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Distress Momentum Score", f"{row['dms']:.1f}", f"{row['risk']} risk")
    with c2:
        kpi("Current demand", f"{row['demand']:.0f}", "latest week")
    with c3:
        growth = row["weekly_growth"]
        kpi(
            "Weekly growth",
            f"{growth:.1f}%" if pd.notna(growth) else "—",
            "speed signal",
        )
    with c4:
        consecutive = row.get("consecutive_growth_weeks", 0)
        consecutive = 0 if pd.isna(consecutive) else int(consecutive)
        kpi("Consecutive growth", f"{consecutive}", "weeks")

    st.markdown("### Why is this village at risk now?")
    st.info(row["explanation"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["demand"],
            mode="lines+markers",
            name="Weekly employment demand",
        )
    )

    anomalies = history[history["anomaly_detected"] == True]  # noqa: E712
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["date"],
                y=anomalies["demand"],
                mode="markers",
                name="Detected anomaly",
                marker=dict(
                    size=12,
                    symbol="x",
                    color=RISK_COLORS["EXTREME"],
                ),
            )
        )

    fig.update_layout(
        height=410,
        title=f"{selected} — historical demand trend",
        yaxis_title=EMPLOYMENT_DEMAND_LABEL,
        paper_bgcolor="#071a33",
        plot_bgcolor="#071a33",
        font=dict(color="#ffffff"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### DMS composition")
    scores = [
        float(row["speed_score"]),
        float(row["intensity_score"]),
        float(row["abnormality_score"]),
    ]
    fig2 = go.Figure(
        go.Bar(
            x=scores,
            y=["Speed", "Intensity", "Abnormality"],
            orientation="h",
            text=[f"{x:.1f}" for x in scores],
            textposition="outside",
        )
    )
    fig2.update_xaxes(range=[0, 105], title="Score (0–100)")
    fig2.update_layout(
        height=250,
        margin=dict(l=20, r=30, t=20, b=20),
        paper_bgcolor="#071a33",
        plot_bgcolor="#071a33",
        font=dict(color="#ffffff"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recommended government action")
    if row["risk"] == "EXTREME":
        st.error(row["recommendation"])
    elif row["risk"] == "HIGH":
        st.warning(row["recommendation"])
    else:
        st.info(row["recommendation"])


def ranking(latest):
    st.markdown(
        '<div class="section-title">Village Ranking</div>',
        unsafe_allow_html=True,
    )

    if latest.empty:
        st.warning("No ranking data is available.")
        return

    risk = st.multiselect(
        "Filter risk",
        RISK_ORDER,
        default=RISK_ORDER,
        key="ranking_risk",
    )
    ranked = latest[latest["risk"].isin(risk)].sort_values(
        "dms", ascending=False
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    ranked["DMS"] = ranked["dms"].round(1)
    ranked["Weekly growth"] = ranked["weekly_growth"].round(1)
    ranked["Risk"] = ranked["risk"]

    out = ranked[
        ["Rank", "village", "district", "block", "DMS", "Weekly growth", "Risk"]
    ]
    st.dataframe(out, use_container_width=True, hide_index=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranking CSV",
        csv,
        "grampulse_rankings.csv",
        "text/csv",
        key="download_rankings",
    )


def explainability(latest):
    st.markdown(
        '<div class="section-title">Explainability & Alerts</div>',
        unsafe_allow_html=True,
    )

    if latest.empty:
        st.warning("No alert data is available.")
        return

    flagged = latest[latest["risk"].isin(["HIGH", "EXTREME"])].sort_values(
        "dms", ascending=False
    )

    if flagged.empty:
        st.success("No HIGH or EXTREME villages are currently flagged.")
        return

    for _, row in flagged.iterrows():
        with st.expander(
            f"{row['risk_emoji']} {row['village']} — "
            f"DMS {row['dms']:.1f} — {row['risk']}"
        ):
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
        key="download_alerts",
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main():
    input_path, coords_df = sidebar()

    # Invalid uploads are stopped before run_pipeline.py is called. This is
    # the key fix for the deployed error caused by uploading grampulse_alerts.csv
    # into the employment-demand uploader.
    if input_path is None:
        st.error("The uploaded employment-demand CSV cannot be processed.")
        st.info(
            "Upload a valid VB-G RAM G employment-demand CSV with these columns: "
            "state, district, block, panchayat, village, date, demand."
        )
        st.stop()

    if not os.path.exists(input_path):
        st.error("VB-G RAM G employment-demand input data was not found.")
        st.stop()

    # ------------------------------------------------------------------
    # Run the pipeline safely. Any known ValueError is shown as a clean
    # application message instead of an uncaught Streamlit traceback.
    # ------------------------------------------------------------------
    try:
        mtime = os.path.getmtime(input_path)
        result, latest = load_pipeline(input_path, mtime)
    except ValueError as exc:
        st.error("The employment-demand dataset is not compatible with GRAMPULSE.")
        st.warning(str(exc))
        st.info(
            "Required columns: state, district, block, panchayat, village, date, demand."
        )
        st.stop()
    except Exception:
        st.error(
            "GRAMPULSE could not process the current dataset. "
            "Please check the Streamlit deployment logs for the detailed error."
        )
        st.stop()

    if result.empty or latest.empty:
        st.error("The pipeline completed but produced no village-level results.")
        st.stop()

    # ------------------------------------------------------------------
    # Hero
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tabs = st.tabs(
        [
            "📊 Overview",
            "🗺️ Risk Map",
            "🏘️ Village Detail",
            "🏆 Ranking",
            "🚨 Explainability & Alerts",
        ]
    )

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
        "GRAMPULSE is an early-warning analytics prototype. A risk score indicates an unusual "
        "employment-demand pattern that warrants administrative review; it does not by itself "
        "establish that a village is in distress."
    )


if __name__ == "__main__":
    main()
