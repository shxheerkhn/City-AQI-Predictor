from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aqi_predictor.configs.settings import AppConfig, load_config
from aqi_predictor.monitoring.explainability import explain_prediction
from aqi_predictor.prediction_service.service import AQIPredictionService
from aqi_predictor.training_pipeline.train import run_training_pipeline
from aqi_predictor.utils.logging import configure_logging, get_logger


# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor – Karachi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging()
LOGGER = get_logger(__name__)
CONFIG = load_config()
SERVICE = AQIPredictionService(CONFIG)

KARACHI_LABEL = "Karachi, Pakistan"
HISTORY_METRICS = [
    "aqi", "pm25", "pm10", "temperature", "humidity",
    "wind_speed", "pressure", "rainfall", "co", "no2", "so2", "o3",
]
METRIC_LABELS: dict[str, str] = {
    "aqi":         "AQI",
    "pm25":        "PM2.5 (µg/m³)",
    "pm10":        "PM10 (µg/m³)",
    "temperature": "Temperature (°C)",
    "humidity":    "Humidity (%)",
    "wind_speed":  "Wind Speed (m/s)",
    "pressure":    "Pressure (hPa)",
    "rainfall":    "Rainfall (mm)",
    "co":          "CO (mg/m³)",
    "no2":         "NO₂ (µg/m³)",
    "so2":         "SO₂ (µg/m³)",
    "o3":          "O₃ (µg/m³)",
}
AQI_BANDS: list[tuple[int, int, str, str]] = [
    (0,   50,  "Good",                    "#22c55e"),
    (51,  100, "Moderate",                "#eab308"),
    (101, 150, "Unhealthy for Sensitive", "#f97316"),
    (151, 200, "Unhealthy",               "#ef4444"),
    (201, 300, "Very Unhealthy",          "#a855f7"),
    (301, 500, "Hazardous",               "#7f1d1d"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def aqi_category(value: float) -> tuple[str, str]:
    for lo, hi, label, color in AQI_BANDS:
        if lo <= value <= hi:
            return label, color
    return "Hazardous", "#7f1d1d"


def city_display_name(city_id: str, config: AppConfig) -> str:
    for city in config.cities:
        if city.city_id == city_id:
            if city.city_id == "karachi":
                return KARACHI_LABEL
            return f"{city.city_id.replace('_', ' ').title()}, {city.country}"
    return city_id.replace("_", " ").title()


def preferred_city_ids(config: AppConfig) -> list[str]:
    return [city.city_id for city in config.cities] or ["karachi"]


def _base_layout(height: int = 380) -> dict:
    return dict(
        hovermode="x unified",
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.8)",
    )


# ── CSS + Hero ────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29,78,216,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(15,118,110,0.10), transparent 24%),
                linear-gradient(180deg,#f7fbff 0%,#edf3f8 100%);
        }
        .hero {
            padding:1.4rem 1.7rem; border-radius:22px;
            background:linear-gradient(135deg,rgba(15,23,42,.98) 0%,rgba(29,78,216,.96) 52%,rgba(15,118,110,.96) 100%);
            color:white; margin-bottom:1rem;
            box-shadow:0 22px 50px rgba(15,23,42,.22);
            border:1px solid rgba(255,255,255,.08);
        }
        .hero h1{color:white;margin-bottom:.2rem;font-weight:800;letter-spacing:-.02em}
        .hero p{color:rgba(255,255,255,.88);margin-top:0;font-size:1.02rem}
        .section-card{
            border-radius:16px;padding:.9rem 1rem;
            background:rgba(255,255,255,.78);
            border:1px solid rgba(15,23,42,.08);
            box-shadow:0 8px 22px rgba(15,23,42,.05);
            margin-bottom:.8rem;
        }
        .band-row{display:flex;flex-wrap:wrap;gap:.45rem;margin:.6rem 0 .2rem}
        .band-pill{
            padding:.22rem .75rem;border-radius:999px;
            font-size:.75rem;font-weight:600;
            opacity:.42;transition:all .15s;
        }
        .band-pill.active{opacity:1;box-shadow:0 2px 10px rgba(0,0,0,.18);transform:scale(1.07)}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
          <h1>🌿 Pearls AQI Predictor</h1>
          <p>Real-time AQI forecasting for Karachi &nbsp;·&nbsp; 3-day outlook
             &nbsp;·&nbsp; History exploration &nbsp;·&nbsp; Model explainability</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aqi_band_legend(current_aqi: float) -> None:
    active_label, _ = aqi_category(current_aqi)
    pills = "".join(
        f'<span class="band-pill {"active" if name == active_label else ""}" '
        f'style="background:{color}28;color:{color};border:1.5px solid {color}55">'
        f'{name} ({lo}–{hi})</span>'
        for lo, hi, name, color in AQI_BANDS
    )
    st.markdown(f'<div class="band-row">{pills}</div>', unsafe_allow_html=True)


# ── Onboarding ────────────────────────────────────────────────────────────────

def render_onboarding(
    selected_city: str,
    bundle_error: str | None,
    history_error: str | None,
) -> None:
    st.info(
        "The dashboard is ready, but a trained champion model and Karachi history "
        "are required before live forecasts appear."
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        '<div class="section-card"><strong>What to run next</strong><br/>'
        "1. Ingest Karachi weather and pollutant data.<br/>"
        "2. Train the models and register a champion.<br/>"
        "3. Refresh this dashboard to see the forecast.</div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="section-card"><strong>Current focus</strong><br/>'
        f'City: <code>{city_display_name(selected_city, CONFIG)}</code><br/>'
        f'Horizon: <code>{CONFIG.forecast_horizon_hours}h</code><br/>'
        f'Threshold: <code>{CONFIG.hazard_aqi_threshold:.0f}</code><br/>'
        f'Source: <code>{CONFIG.external_api_provider}</code></div>',
        unsafe_allow_html=True,
    )
    st.code(
        "python -m aqi_predictor.feature_pipeline.cli\n"
        "python -m aqi_predictor.training_pipeline.train\n"
        "python -m aqi_predictor.reporting.cli",
        language="bash",
    )
    if bundle_error:
        st.caption(f"Model status: {bundle_error}")
    if history_error:
        st.caption(f"History status: {history_error}")


# ── Top metric bar ────────────────────────────────────────────────────────────

def render_metric_cards(
    history: pd.DataFrame,
    forecast_payload: dict[str, Any],
) -> None:
    latest_row = history.sort_values("timestamp").tail(1).iloc[0]
    fdf = pd.DataFrame(forecast_payload["points"])
    current_aqi = float(latest_row["aqi"])
    next_aqi    = float(fdf.iloc[0]["aqi"])
    delta       = next_aqi - current_aqi
    label, _    = aqi_category(current_aqi)

    cols = st.columns(6)
    cols[0].metric("City",           KARACHI_LABEL)
    cols[1].metric("Current AQI",    f"{current_aqi:.1f}",
                   delta=f"{delta:+.1f} next hr")
    cols[2].metric("Category",       label)
    cols[3].metric("Hazard Hours",   str(int(forecast_payload["hazard_count"])),
                   delta=f"of {int(forecast_payload['forecast_hours'])}h",
                   delta_color="inverse")
    cols[4].metric("Horizon",        f"{int(forecast_payload['forecast_hours'])}h")
    cols[5].metric("Champion",       forecast_payload["model_name"])

    render_aqi_band_legend(current_aqi)


def build_forecast_day_frames(forecast_payload: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    fdf = pd.DataFrame(forecast_payload["points"]).copy()
    if fdf.empty:
        return []
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])
    start_time = fdf["timestamp"].min()
    fdf["forecast_day"] = ((fdf["timestamp"] - start_time).dt.total_seconds() // 86400).astype(int) + 1
    day_frames: list[tuple[str, pd.DataFrame]] = []
    for day_number, frame in fdf.groupby("forecast_day", sort=True):
        if int(day_number) > 3:
            continue
        day_start = start_time + pd.Timedelta(days=int(day_number) - 1)
        label = f"Day {int(day_number)} · {day_start.strftime('%a, %d %b')}"
        day_frames.append((label, frame.reset_index(drop=True)))
    return day_frames


def render_current_day_panel(history: pd.DataFrame) -> None:
    latest_row = history.sort_values("timestamp").tail(1).iloc[0]
    current_aqi = float(latest_row["aqi"])
    current_label, current_color = aqi_category(current_aqi)
    panel_cols = st.columns(4)
    panel_cols[0].metric("Current AQI", f"{current_aqi:.1f}")
    panel_cols[1].metric("AQI Category", current_label)
    panel_cols[2].metric("Observation Time", pd.Timestamp(latest_row["timestamp"]).strftime("%Y-%m-%d %H:%M"))
    panel_cols[3].metric("City", KARACHI_LABEL)
    render_aqi_band_legend(current_aqi)

    st.markdown(
        f"""
        <div class="section-card">
          <strong>Current day conditions</strong><br/>
          AQI is <span style="color:{current_color};font-weight:700">{current_label}</span> right now for Karachi.
          The table below shows the latest observed weather and pollutant readings for today.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_snapshot = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(latest_row["timestamp"]),
                "aqi": float(latest_row["aqi"]),
                "temperature": float(latest_row.get("temperature", 0.0)),
                "humidity": float(latest_row.get("humidity", 0.0)),
                "wind_speed": float(latest_row.get("wind_speed", 0.0)),
                "pressure": float(latest_row.get("pressure", 0.0)),
                "rainfall": float(latest_row.get("rainfall", 0.0)),
                "pm25": float(latest_row.get("pm25", 0.0)),
                "pm10": float(latest_row.get("pm10", 0.0)),
                "co": float(latest_row.get("co", 0.0)),
                "no2": float(latest_row.get("no2", 0.0)),
                "so2": float(latest_row.get("so2", 0.0)),
                "o3": float(latest_row.get("o3", 0.0)),
            }
        ]
    )
    st.dataframe(current_snapshot, use_container_width=True)


def render_next_three_day_summary(forecast_payload: dict[str, Any]) -> None:
    day_frames = build_forecast_day_frames(forecast_payload)
    if not day_frames:
        st.info("No forecast points are available for the next 3 days yet.")
        return

    st.subheader("Next 3 Days Forecast")
    st.caption("The forecast below is grouped by calendar day so you can compare each day explicitly.")

    for label, frame in day_frames[:3]:
        avg_aqi = float(frame["aqi"].mean())
        min_aqi = float(frame["aqi"].min())
        max_aqi = float(frame["aqi"].max())
        hazard_hours = int(frame["hazard_flag"].sum())
        day_label, day_color = aqi_category(avg_aqi)
        st.markdown(
            f"""
            <div class="section-card">
              <strong>{label}</strong><br/>
              Expected air quality for this 24-hour window is
              <span style="color:{day_color};font-weight:700">{day_label}</span>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        cols[0].metric("Average AQI", f"{avg_aqi:.1f}")
        cols[1].metric("Min / Max", f"{min_aqi:.1f} / {max_aqi:.1f}")
        cols[2].metric("Hazard Hours", str(hazard_hours))
        cols[3].metric("Day Status", day_label)
        st.dataframe(
            frame[["timestamp", "aqi", "lower_bound", "upper_bound", "hazard_flag", "model_name"]],
            use_container_width=True,
        )


# ── Overview tab ──────────────────────────────────────────────────────────────

def render_overview_tab(
    history: pd.DataFrame,
    forecast_payload: dict[str, Any],
) -> None:
    latest_row  = history.sort_values("timestamp").tail(1).iloc[0]
    current_aqi = float(latest_row["aqi"])
    fdf         = pd.DataFrame(forecast_payload["points"])
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City", KARACHI_LABEL)
    c2.metric("Current AQI", f"{current_aqi:.1f}")
    c3.metric("Latest Observation",
              pd.Timestamp(latest_row["timestamp"]).strftime("%Y-%m-%d %H:%M"))
    c4.metric("Forecast Horizon", f"{int(forecast_payload['forecast_hours'])}h")

    first_fc = float(fdf.iloc[0]["aqi"])
    if first_fc >= CONFIG.hazard_aqi_threshold:
        st.error(f"⚠️ Immediate AQI ({first_fc:.1f}) exceeds hazard threshold "
                 f"({CONFIG.hazard_aqi_threshold:.0f}).")
    elif int(forecast_payload["hazard_count"]) > 0:
        st.warning(f"⚡ Hazardous AQI predicted in "
                   f"{forecast_payload['hazard_count']} hour(s).")
    else:
        st.success("✅ No hazardous AQI levels detected in the selected horizon.")

    st.subheader("Current Day Data")
    render_current_day_panel(history)

    # Combined history + forecast chart
    st.subheader("History + Forecast Overview")
    hist72 = history.sort_values("timestamp").tail(72).copy()
    hist72["timestamp"] = pd.to_datetime(hist72["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist72["timestamp"], y=hist72["aqi"],
        name="History (72 h)", mode="lines",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fdf["timestamp"], fdf["timestamp"][::-1]]),
        y=pd.concat([fdf["upper_bound"], fdf["lower_bound"][::-1]]),
        fill="toself", fillcolor="rgba(16,185,129,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band",
    ))
    fig.add_trace(go.Scatter(
        x=fdf["timestamp"], y=fdf["aqi"],
        name="Forecast", mode="lines+markers",
        line=dict(color="#10b981", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig.add_hline(
        y=CONFIG.hazard_aqi_threshold,
        line_width=1.5, line_dash="dash", line_color="#dc2626",
        annotation_text=f"Hazard ({CONFIG.hazard_aqi_threshold:.0f})",
        annotation_position="top right",
    )
    fig.update_layout(**_base_layout(360))
    st.plotly_chart(fig, use_container_width=True)

    # Pollutant snapshot
    st.subheader("Current Pollutant Snapshot")
    poll_data = {
        "PM2.5":  float(latest_row.get("pm25", 0)),
        "PM10":   float(latest_row.get("pm10", 0)),
        "CO×10":  float(latest_row.get("co", 0)) * 10,
        "NO₂":    float(latest_row.get("no2", 0)),
        "SO₂":    float(latest_row.get("so2", 0)),
        "O₃":     float(latest_row.get("o3", 0)),
    }
    poll_df = pd.DataFrame({
        "Pollutant": list(poll_data.keys()),
        "Value":     list(poll_data.values()),
    })
    pf = px.bar(
        poll_df, x="Pollutant", y="Value", color="Pollutant",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Pollutant concentrations (CO ×10 for visibility)",
    )
    pf.update_layout(showlegend=False, **_base_layout(260))
    st.plotly_chart(pf, use_container_width=True)

    # Weather mini-cards
    st.subheader("Current Weather Conditions")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("🌡 Temperature", f"{float(latest_row.get('temperature', 0)):.1f} °C")
    w2.metric("💧 Humidity",    f"{float(latest_row.get('humidity', 0)):.1f} %")
    w3.metric("💨 Wind Speed",  f"{float(latest_row.get('wind_speed', 0)):.1f} m/s")
    w4.metric("🌧 Rainfall",    f"{float(latest_row.get('rainfall', 0)):.2f} mm")


# ── Forecast tab ──────────────────────────────────────────────────────────────

def render_forecast_tab(
    history: pd.DataFrame,
    forecast_payload: dict[str, Any],
    hazard_threshold: float,
) -> None:
    fdf = pd.DataFrame(forecast_payload["points"])
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])
    fdf["direction"] = fdf["aqi"].diff().fillna(0.0)
    fdf["category"]  = fdf["aqi"].apply(lambda v: aqi_category(v)[0])

    st.subheader("Forecast Explorer")

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    chart_type   = ctrl1.radio("Chart type", ["Line", "Area"], horizontal=True, key="fc_type")
    show_ci      = ctrl2.toggle("Confidence interval", value=True, key="fc_ci")
    shade_hazard = ctrl3.toggle("Shade hazard hours", value=True, key="fc_shade")
    hazard_only  = ctrl4.toggle("Hazard hours only",  value=False, key="fc_honly")

    plot_df = fdf[fdf["hazard_flag"]] if hazard_only else fdf

    fig = go.Figure()

    if show_ci and not plot_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([plot_df["timestamp"], plot_df["timestamp"][::-1]]),
            y=pd.concat([plot_df["upper_bound"], plot_df["lower_bound"][::-1]]),
            fill="toself", fillcolor="rgba(239,68,68,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band",
        ))

    marker_cfg = dict(
        size=7,
        color=plot_df["aqi"].tolist() if not plot_df.empty else [],
        colorscale=[
            [0.00, "#22c55e"], [0.17, "#eab308"], [0.33, "#f97316"],
            [0.50, "#ef4444"], [0.67, "#a855f7"], [1.00, "#7f1d1d"],
        ],
        cmin=0, cmax=300,
        showscale=True,
        colorbar=dict(title="AQI", thickness=12, len=0.7),
    )

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["aqi"],
        fill="tozeroy" if chart_type == "Area" else "none",
        fillcolor="rgba(59,130,246,0.10)",
        mode="lines+markers", name="AQI Forecast",
        line=dict(color="#3b82f6", width=2),
        marker=marker_cfg,
    ))

    if shade_hazard:
        for _, row in fdf[fdf["hazard_flag"]].iterrows():
            fig.add_vrect(
                x0=row["timestamp"] - pd.Timedelta(minutes=30),
                x1=row["timestamp"] + pd.Timedelta(minutes=30),
                fillcolor="rgba(220,38,38,0.11)", line_width=0,
            )

    fig.add_hline(
        y=hazard_threshold, line_width=1.5, line_dash="dash", line_color="#dc2626",
        annotation_text=f"Hazard ({hazard_threshold:.0f})",
        annotation_position="top right",
    )
    fig.update_layout(**_base_layout(420))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("First forecast AQI", f"{float(fdf.iloc[0]['aqi']):.1f}")
    m2.metric("Hazard risk hours",  str(int(forecast_payload["hazard_count"])))

    render_next_three_day_summary(forecast_payload)

    csv_fc = fdf[
        ["timestamp", "aqi", "lower_bound", "upper_bound", "hazard_flag", "category", "model_name"]
    ].to_csv(index=False)
    st.download_button(
        "⬇️ Download forecast CSV",
        data=csv_fc, file_name="karachi_forecast.csv", mime="text/csv",
    )

    st.dataframe(
        fdf[["timestamp", "aqi", "lower_bound", "upper_bound",
             "hazard_flag", "category", "model_name"]],
        use_container_width=True,
    )


# ── History tab ───────────────────────────────────────────────────────────────

def render_history_tab(
    history: pd.DataFrame,
    selected_metric: str,
    hazard_threshold: float,
) -> None:
    hf = history.copy()
    hf["timestamp"] = pd.to_datetime(hf["timestamp"])
    hf = hf.sort_values("timestamp")
    display = hf.tail(720)

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    overlay_metric = ctrl1.selectbox(
        "Overlay second metric",
        options=["None"] + [m for m in HISTORY_METRICS if m != selected_metric],
        format_func=lambda m: METRIC_LABELS.get(m, m) if m != "None" else "None",
        key="hist_overlay",
    )
    show_ma = ctrl2.toggle("Moving average", value=False, key="hist_ma")
    ma_window = (
        ctrl3.slider("MA window (hours)", 6, 72, 24, 6, key="hist_maw")
        if show_ma else 24
    )

    primary_label = METRIC_LABELS.get(selected_metric, selected_metric.upper())
    st.subheader(f"Historical {primary_label} Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display["timestamp"], y=display[selected_metric],
        name=primary_label, mode="lines",
        line=dict(color="#3b82f6", width=1.8),
    ))

    if show_ma:
        ma = display[selected_metric].rolling(ma_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=display["timestamp"], y=ma,
            name=f"{ma_window}h Moving Avg", mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))

    if overlay_metric and overlay_metric != "None" and overlay_metric in display.columns:
        ol_label = METRIC_LABELS.get(overlay_metric, overlay_metric.upper())
        fig.add_trace(go.Scatter(
            x=display["timestamp"], y=display[overlay_metric],
            name=ol_label, mode="lines",
            line=dict(color="#10b981", width=1.6),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(
                title=ol_label,
                overlaying="y", side="right", showgrid=False,
            )
        )

    if selected_metric == "aqi":
        fig.add_hline(
            y=hazard_threshold, line_width=1.5, line_dash="dash", line_color="#dc2626",
            annotation_text=f"Hazard ({hazard_threshold:.0f})",
        )

    fig.update_layout(**_base_layout(390))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Min",     f"{float(hf[selected_metric].min()):.1f}")
    s2.metric("Max",     f"{float(hf[selected_metric].max()):.1f}")
    s3.metric("Mean",    f"{float(hf[selected_metric].mean()):.1f}")
    s4.metric("Std Dev", f"{float(hf[selected_metric].std(ddof=0)):.1f}")

    csv_h = display[
        ["timestamp", "aqi", "pm25", "pm10",
         "temperature", "humidity", "wind_speed", "source"]
    ].to_csv(index=False)
    st.download_button(
        "⬇️ Download history CSV",
        data=csv_h, file_name="karachi_history.csv", mime="text/csv",
    )

    st.dataframe(
        hf.tail(50)[
            ["timestamp", "aqi", "pm25", "pm10",
             "temperature", "humidity", "wind_speed", "source"]
        ],
        use_container_width=True,
    )


# ── Explainability tab ────────────────────────────────────────────────────────

def render_explainability_tab(bundle: Any, history: pd.DataFrame) -> None:
    st.subheader("Model Explainability")
    latest_row = history.tail(1)
    if latest_row.empty:
        st.info("No history available for explanation.")
        return

    with st.spinner("Computing feature attributions…"):
        explanation = explain_prediction(bundle.model_artifact, history, latest_row)

    if explanation.direction_summary:
        st.markdown(
            f'<div class="section-card"><strong>Prediction drivers</strong><br/>'
            f'{explanation.direction_summary}</div>',
            unsafe_allow_html=True,
        )

    max_feat  = max(5, min(20, len(explanation.global_importance)))
    n_features = st.slider(
        "Features to display", 5, max_feat, min(10, max_feat), key="exp_n"
    )

    tab_g, tab_l, tab_raw = st.tabs(["Global Importance", "Local Impact", "Raw Tables"])

    with tab_g:
        if not explanation.global_importance.empty:
            gfig = px.bar(
                explanation.global_importance.head(n_features),
                x="importance", y="feature", orientation="h",
                title=f"Top {n_features} global feature importances",
                color="importance",
                color_continuous_scale=["#93c5fd", "#1e40af"],
            )
            gfig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                **_base_layout(360),
            )
            st.plotly_chart(gfig, use_container_width=True)
        else:
            st.info("Global importance not available for this model family.")

    with tab_l:
        if not explanation.local_summary.empty:
            lfig = px.bar(
                explanation.local_summary.head(n_features),
                x="impact", y="feature", orientation="h",
                title=f"Top {n_features} local prediction drivers",
                color="impact",
                color_continuous_scale=["#22c55e", "#f9fafb", "#ef4444"],
                color_continuous_midpoint=0,
            )
            lfig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                **_base_layout(360),
            )
            st.plotly_chart(lfig, use_container_width=True)
        else:
            st.info("Local impact not available for this model family.")

    with tab_raw:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Global Feature Importance**")
            st.dataframe(explanation.global_importance.head(20), use_container_width=True)
        with r2:
            st.markdown("**Local Prediction Impact**")
            st.dataframe(explanation.local_summary.head(20), use_container_width=True)


# ── Operations tab ────────────────────────────────────────────────────────────

def render_operations_tab(metrics: dict[str, Any], bundle: Any) -> None:
    st.subheader("Model & Service Operations")

    m  = metrics["metrics"]
    mc = st.columns(4)
    mc[0].metric("RMSE", f"{m.get('rmse', 0.0):.3f}")
    mc[1].metric("MAE",  f"{m.get('mae',  0.0):.3f}")
    mc[2].metric("R²",   f"{m.get('r2',   0.0):.3f}")
    mc[3].metric("MAPE", f"{m.get('mape', 0.0):.2f} %")

    ops = metrics.get("service_metrics", {})
    oc  = st.columns(4)
    oc[0].metric("Total Requests",   str(int(ops.get("total_requests", 0))))
    oc[1].metric("Forecast Calls",   str(int(ops.get("forecast_requests", 0))))
    oc[2].metric("Prediction Calls", str(int(ops.get("prediction_requests", 0))))
    oc[3].metric("Errors",           str(int(ops.get("error_requests", 0))))

    # Normalized radar
    rmse_s = max(0.0, 1.0 - m.get("rmse", 0) / 100.0)
    mae_s  = max(0.0, 1.0 - m.get("mae",  0) /  50.0)
    r2_s   = max(0.0, float(m.get("r2", 0)))
    mape_s = max(0.0, 1.0 - m.get("mape", 0) / 100.0)
    cats   = ["RMSE", "MAE", "R²", "MAPE"]
    scores = [rmse_s, mae_s, r2_s, mape_s]

    radar = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.18)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=7, color="#3b82f6"),
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model performance (normalized 0–1, higher = better)",
        height=300,
        margin=dict(l=50, r=50, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(radar, use_container_width=True)

    st.caption(
        f"Model: **{bundle.model_metadata['model_name']}** | "
        f"Version: {bundle.model_metadata['model_version']} | "
        f"Trained: {bundle.model_metadata['trained_at']}"
    )
    st.caption(
        f"Avg latency: {float(ops.get('average_latency_ms', 0.0)):.1f} ms | "
        f"Last error: {ops.get('last_error') or 'none'}"
    )
    with st.expander("Full model metadata"):
        st.dataframe(pd.DataFrame([bundle.model_metadata]), use_container_width=True)


# ── Setup tab ─────────────────────────────────────────────────────────────────

def render_setup_tab(
    bundle_error: str | None,
    history_error: str | None,
    selected_city: str,
) -> None:
    st.subheader("Setup and Recovery")
    st.markdown(
        f'<div class="section-card"><strong>Karachi workflow</strong><br/>'
        f'Dashboard is centered on <code>{city_display_name(selected_city, CONFIG)}</code> '
        f'and expects a registered champion model.</div>',
        unsafe_allow_html=True,
    )
    st.code(
        "python -m aqi_predictor.feature_pipeline.cli\n"
        "python -m aqi_predictor.training_pipeline.train\n"
        "python -m aqi_predictor.reporting.cli\n"
        "streamlit run aqi_predictor/dashboard/app.py",
        language="bash",
    )
    if bundle_error:
        st.warning(f"Model status: {bundle_error}")
    if history_error:
        st.warning(f"History status: {history_error}")


def bootstrap_demo_model_if_needed(selected_city: str) -> None:
    if not CONFIG.auto_bootstrap_demo:
        return
    if st.session_state.get("karachi_bootstrap_complete"):
        return
    if st.session_state.get("karachi_bootstrap_attempted"):
        return

    try:
        st.session_state["karachi_bootstrap_attempted"] = True
        with st.spinner("Bootstrapping Karachi data and model..."):
            run_training_pipeline(CONFIG)
        st.session_state["karachi_bootstrap_complete"] = True
        st.success(
            f"Karachi model ready. You can now view the next {CONFIG.forecast_horizon_hours} hours of predictions."
        )
        st.rerun()
    except Exception as exc:
        LOGGER.warning("Karachi bootstrap failed: %s", exc)
        st.info(
            "Automatic bootstrap could not complete. "
            "Run the training pipeline once, or set AQI_AUTO_BOOTSTRAP_DEMO=true if you want the dashboard to build itself on startup."
        )


# ── Reports tab ──────────────────────────────────────────────────────────────

def render_reports_tab(store: Any) -> None:
    """Render model reports and comparison tab."""
    st.subheader("📊 Model Reports & Performance")
    
    try:
        comparison_df = store.get_model_comparison(city_scope="global")
        
        if not comparison_df.empty:
            st.subheader("Model Comparison Ranking")
            st.dataframe(
                comparison_df[["rank", "model_name", "model_family", "rmse", "mae", "r2", "mape", "is_champion"]].head(10),
                use_container_width=True,
                hide_index=True,
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("RMSE Comparison")
                fig_rmse = px.bar(
                    comparison_df.head(10),
                    x="model_name",
                    y="rmse",
                    color="is_champion",
                    color_discrete_map={True: "#10b981", False: "#6b7280"},
                    labels={"model_name": "Model", "rmse": "RMSE"},
                )
                fig_rmse.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                st.subheader("R² Comparison")
                fig_r2 = px.bar(
                    comparison_df.head(10),
                    x="model_name",
                    y="r2",
                    color="is_champion",
                    color_discrete_map={True: "#10b981", False: "#6b7280"},
                    labels={"model_name": "Model", "r2": "R²"},
                )
                fig_r2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            history_df = store.get_model_history(city_scope="global", limit=20)
            if not history_df.empty:
                st.subheader("Model History Timeline")
                history_df["trained_at_datetime"] = pd.to_datetime(history_df["trained_at"])
                history_df = history_df.sort_values("trained_at_datetime", ascending=False)
                
                for _, row in history_df.head(10).iterrows():
                    metrics = row["metrics"]
                    champion_badge = "🏆 Champion" if row["is_champion"] else ""
                    st.markdown(
                        f"**{row['model_name']}** ({row['model_family']}) {champion_badge}\n"
                        f"- Trained: {pd.Timestamp(row['trained_at']).strftime('%Y-%m-%d %H:%M')}\n"
                        f"- RMSE: {metrics.get('rmse', 0):.4f} | MAE: {metrics.get('mae', 0):.4f} | R²: {metrics.get('r2', 0):.4f}"
                    )
        else:
            st.info("No model history available yet. Run the training pipeline to generate models.")
    
    except Exception as exc:
        st.error(f"Failed to load model reports: {exc}")


# ── EDA tab ───────────────────────────────────────────────────────────────────

def render_eda_tab(report_dir: Path) -> None:
    """Render EDA analysis and visualizations tab."""
    st.subheader("🔬 Exploratory Data Analysis")
    
    try:
        eda_report_path = report_dir / "eda" / "eda_report.json"
        
        if eda_report_path.exists():
            with open(eda_report_path, "r") as f:
                eda_report = json.load(f)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "Univariate", "Bivariate", "Time Series"])
            
            with tab1:
                st.subheader("Data Quality Summary")
                dq = eda_report.get("data_quality", {})
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Rows", f"{dq.get('total_rows', 0):,}")
                col2.metric("Total Columns", dq.get("total_columns", 0))
                col3.metric("Memory (MB)", f"{dq.get('memory_usage_mb', 0):.2f}")
                col4.metric("Duplicates", f"{dq.get('duplicates', {}).get('count', 0):,}")
                
                st.subheader("Missing Values by Column")
                missing_data = dq.get("missing_values", {})
                if missing_data:
                    missing_df = pd.DataFrame([
                        {"Column": col, "Missing": v["count"], "Percentage": f"{v['percentage']:.2f}%"}
                        for col, v in missing_data.items()
                        if v["count"] > 0
                    ])
                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No missing values detected!")
            
            with tab2:
                st.subheader("Univariate Statistics")
                uni = eda_report.get("univariate", {})
                
                if uni:
                    selected_feature = st.selectbox(
                        "Select feature to analyze:",
                        options=list(uni.keys())
                    )
                    
                    if selected_feature in uni:
                        stats_data = uni[selected_feature]
                        if isinstance(stats_data, dict) and "mean" in stats_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean", f"{stats_data.get('mean', 0):.4f}")
                            col2.metric("Median", f"{stats_data.get('median', 0):.4f}")
                            col3.metric("Std Dev", f"{stats_data.get('std', 0):.4f}")
                            col4.metric("IQR", f"{stats_data.get('iqr', 0):.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Min", f"{stats_data.get('min', 0):.4f}")
                                st.metric("Q25", f"{stats_data.get('q25', 0):.4f}")
                            with col2:
                                st.metric("Q75", f"{stats_data.get('q75', 0):.4f}")
                                st.metric("Max", f"{stats_data.get('max', 0):.4f}")
                            
                            st.metric("Skewness", f"{stats_data.get('skewness', 0):.4f}")
            
            with tab3:
                st.subheader("Bivariate Analysis")
                biv = eda_report.get("bivariate", {})
                
                if biv:
                    high_corr = biv.get("high_correlations", [])
                    if high_corr:
                        st.subheader("High Correlations (|r| > 0.7)")
                        high_corr_df = pd.DataFrame(high_corr)
                        st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Feature-Target Correlations")
                    target_corrs = {k: v for k, v in biv.items() if k.startswith("target_correlation_")}
                    if target_corrs:
                        target_df = pd.DataFrame([
                            {"Feature": k.replace("target_correlation_", ""), "Correlation": v}
                            for k, v in target_corrs.items()
                        ]).sort_values("Correlation", key=abs, ascending=False)
                        st.dataframe(target_df, use_container_width=True, hide_index=True)
            
            with tab4:
                st.subheader("Time Series Patterns")
                ts = eda_report.get("time_series", {})
                
                if ts:
                    date_info = ts.get("date_range", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Start", date_info.get("start", "N/A")[:10])
                    col2.metric("End", date_info.get("end", "N/A")[:10])
                    col3.metric("Days Covered", date_info.get("days_covered", 0))
                    
                    hourly_mean = ts.get("hourly_mean_aqi", {})
                    if hourly_mean:
                        st.subheader("Hourly Average AQI")
                        fig = px.line(
                            x=list(map(int, hourly_mean.keys())),
                            y=list(map(float, hourly_mean.values())),
                            labels={"x": "Hour of Day", "y": "Average AQI"},
                            title="AQI by Hour of Day",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("📊 EDA report not yet available. Run the training pipeline to generate analysis.")
    
    except Exception as exc:
        st.error(f"Failed to load EDA analysis: {exc}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

render_header()

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    city_options  = preferred_city_ids(CONFIG)
    selected_city = st.selectbox(
        "City",
        options=city_options,
        format_func=lambda cid: city_display_name(cid, CONFIG),
        index=0,
    )
    forecast_hours = st.slider(
        "Forecast horizon (hours)", min_value=1, max_value=72, value=72, step=1
    )
    history_window = st.slider(
        "History window (hours)", min_value=24, max_value=720, value=336, step=24
    )
    selected_metric = st.selectbox(
        "Primary history metric",
        options=HISTORY_METRICS,
        format_func=lambda m: METRIC_LABELS.get(m, m.upper()),
        index=0,
    )
    show_data_table = st.toggle("Show recent data table", value=True)
    st.divider()
    refresh_requested = st.button("🔄 Refresh now", use_container_width=True)
    st.divider()
    st.caption("📍 City: Karachi, Pakistan")
    st.caption(f"⚠️ Hazard threshold: AQI {CONFIG.hazard_aqi_threshold:.0f}")
    st.caption(f"🕐 Horizon: {forecast_hours}h  |  Window: {history_window}h")
    st.caption(f"🛰️ Source: {CONFIG.external_api_provider}")

if refresh_requested:
    st.rerun()

# ── Data loading ──────────────────────────────────────────────────────────────

bundle = None
bundle_error: str | None = None
history_error: str | None = None
history = pd.DataFrame()
forecast_payload: dict[str, Any] | None = None
metrics_payload: dict[str, Any] = {}

try:
    bundle = SERVICE.get_bundle()
except Exception as exc:
    bundle_error = str(exc)

try:
    history = SERVICE.latest_history(selected_city, window_hours=history_window)
except Exception as exc:
    history_error = str(exc)

# ── Live context bar ──────────────────────────────────────────────────────────

st.markdown(
    f'<div class="section-card">'
    f'<strong>Live Context</strong> &nbsp;·&nbsp; '
    f'Horizon: <code>{forecast_hours}h</code> &nbsp;·&nbsp; '
    f'Window: <code>{history_window}h</code> &nbsp;·&nbsp; '
    f'Metric: <code>{selected_metric}</code>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Main content ──────────────────────────────────────────────────────────────

if bundle is None:
    bootstrap_demo_model_if_needed(selected_city)
    try:
        bundle = SERVICE.get_bundle()
        bundle_error = None
    except Exception as exc:
        bundle_error = str(exc)
    try:
        history = SERVICE.latest_history(selected_city, window_hours=history_window)
        history_error = None
    except Exception as exc:
        history_error = str(exc)

if bundle is None:
    render_onboarding(selected_city, bundle_error, history_error)
    if not history.empty:
        tab1, tab2 = st.tabs(["📈 History", "🔧 Setup"])
        with tab1:
            render_history_tab(history, selected_metric, CONFIG.hazard_aqi_threshold)
            if show_data_table:
                st.dataframe(history.tail(50), use_container_width=True)
        with tab2:
            render_setup_tab(bundle_error, history_error, selected_city)
else:
    try:
        forecast_payload = SERVICE.forecast(selected_city, forecast_hours)
    except Exception as exc:
        bundle_error = f"Forecast unavailable: {exc}"
        LOGGER.exception("Dashboard forecast failed")

    if not history.empty and forecast_payload is not None:
        try:
            metrics_payload = SERVICE.metrics()
        except Exception as exc:
            metrics_payload = {}
            st.caption(f"Model metrics unavailable: {exc}")

        render_metric_cards(history, forecast_payload)

        ov, fc, ht, ex, op, rp, ed, su = st.tabs([
            "🏠 Overview",
            "📡 Forecast",
            "📈 History",
            "🧠 Explainability",
            "⚙️ Operations",
            "📊 Reports",
            "🔬 EDA",
            "🔧 Setup",
        ])

        with ov:
            render_overview_tab(history, forecast_payload)
        with fc:
            render_forecast_tab(history, forecast_payload, CONFIG.hazard_aqi_threshold)
        with ht:
            render_history_tab(history, selected_metric, CONFIG.hazard_aqi_threshold)
            if show_data_table:
                st.dataframe(history.tail(50), use_container_width=True)
        with ex:
            render_explainability_tab(bundle, history)
        with op:
            if metrics_payload:
                render_operations_tab(metrics_payload, bundle)
            else:
                st.info("No model metrics are available yet.")
        with rp:
            render_reports_tab(SERVICE.store)
        with ed:
            render_eda_tab(CONFIG.report_dir)
        with su:
            render_setup_tab(bundle_error, history_error, selected_city)

    else:
        render_onboarding(selected_city, bundle_error, history_error)
        if not history.empty:
            st.subheader("History Explorer")
            render_history_tab(history, selected_metric, CONFIG.hazard_aqi_threshold)
            if show_data_table:
                st.dataframe(history.tail(50), use_container_width=True)
