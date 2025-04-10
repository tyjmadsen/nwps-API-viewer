from typing import Any, Optional, cast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from logic import interpolate_stage_cached, parse_timeseries, preprocess_stageflow

# --- Default Configs ---
DEFAULT_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "shortRange": {"label": "SRF", "color": "green"},
    "mediumRange": {"label": "MRF", "color": "orange"},
    "mediumRangeBlend": {"label": "MR-B", "color": "mediumpurple"},
    "analysisAssimilation": {"label": "AA", "color": "gray"},
}

DEFAULT_COLOR_THEME: dict[str, str] = {
    "observed": "firebrick",
    "official_forecast": "royalblue",
    "threshold_action": "#FFFF00",
    "threshold_minor": "#FFA500",
    "threshold_moderate": "#FF0000",
    "threshold_major": "#DA00FF",
}


def add_ensemble_percentile_bands(
    fig: Figure,
    percentiles: dict[str, pd.Series],
    rating_df: pd.DataFrame,
    forecast_init_time: Optional[pd.Timestamp] = None,
) -> None:
    colors = {
        "band_05_95": "rgba(70,130,180,0.08)",
        "band_10_90": "rgba(70,130,180,0.15)",
        "band_25_75": "rgba(70,130,180,0.35)",
        "line": "rgb(70,130,180)",
    }

    labels_with_colors = [
        ("p25", "p75", "25–75%", "band_25_75"),
        ("p10", "p90", "10–90%", "band_10_90"),
        ("p05", "p95", "5–95%", "band_05_95"),
    ]

    legend_labels = {
        "25–75%": "Most Likely (25-75%)",
        "10–90%": "More Likely (10-90%)",
        "5–95%": "Less Likely (5-95%)",
    }

    # Median with all bands in tooltip
    fig.add_trace(
        go.Scatter(
            x=percentiles["p50"].index,
            y=percentiles["p50"],
            name="HEFS Median (P50)",
            mode="lines",
            line=dict(
                color=colors["line"],
                width=2,
                dash="dashdot",
                shape="spline",
                smoothing=1.3,
            ),
            customdata=np.column_stack([
                percentiles["p05"],
                percentiles["p95"],
                percentiles["p10"],
                percentiles["p90"],
                percentiles["p25"],
                percentiles["p75"],
            ]),
            hovertemplate=(
                "%{x|%b %d %H:%M}<br>"
                "<b>Median</b>: %{y:.1f} ft<br>"
                "5–95%: %{customdata[0]:.1f} – %{customdata[1]:.1f} ft<br>"
                "10–90%: %{customdata[2]:.1f} – %{customdata[3]:.1f} ft<br>"
                "25–75%: %{customdata[4]:.1f} – %{customdata[5]:.1f} ft"
                "<extra></extra>"
            ),
        )
    )


    for low, high, label, color_key in labels_with_colors:
        lower = percentiles[low].dropna()
        upper = percentiles[high].dropna()
        if lower.empty or upper.empty:
            continue

        x_vals = upper.index
        legend_name = legend_labels[label]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor=colors[color_key],
                line=dict(width=0),
                name=legend_name,
                showlegend=True,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=upper,
                mode="markers",
                name=label,
                marker=dict(
                    size=12,
                    color=colors[color_key],
                    line=dict(width=0),
                    opacity=1,
                ),
                customdata=np.column_stack([lower, upper]),
                hovertemplate=(
                    "<b>%{text}</b>:<br>%{customdata[0]:.1f} – "
                    "%{customdata[1]:.1f} ft<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.85)", font=dict(color="white")
                ),
                showlegend=False,
                visible="legendonly",
                text=legend_name,
            )
        )


def add_flood_threshold_shading(
    fig: Figure,
    flood_lines: dict[str, tuple[float, str]],
    show_thresholds: list[str],
    zoom_start: Any,
    zoom_end: Any,
    upper_limit: float,
) -> None:
    soft_flood_colors = {
        "action": "rgba(255, 255, 0, 0.15)",
        "minor": "rgba(255, 165, 0, 0.15)",
        "moderate": "rgba(255, 0, 0, 0.15)",
        "major": "rgba(218, 0, 255, 0.15)",
    }

    label_colors = {
        "action": "#FFEB3B",
        "minor": "#FFB74D",
        "moderate": "#FF6666",
        "major": "#E066FF",
    }

    sorted_cats = sorted(
        [(cat, flood_lines[cat][0]) for cat in show_thresholds if cat in flood_lines],
        key=lambda x: x[1],
    )

    for i, (cat, y0) in enumerate(sorted_cats):
        if i + 1 < len(sorted_cats):
            y1 = sorted_cats[i + 1][1]
        elif cat == "major":
            y1 = upper_limit
        else:
            continue
    
        fig.add_shape(
            type="rect",
            xref="x",  # ← change this
            yref="y",
            x0=zoom_start,  # ← match data range
            x1=zoom_end,
            y0=y0,
            y1=y1,
            fillcolor=soft_flood_colors.get(cat, "rgba(200,200,200,0.1)"),
            line_width=0,
            layer="below",
        )


    for cat, y_val in sorted(
        [(cat, flood_lines[cat][0]) for cat in flood_lines],
        key=lambda x: x[1],
    ):
        fig.add_hline(
            y=y_val,
            line_dash="dash",
            line_color=soft_flood_colors.get(cat, "gray"),
            annotation_text=cat.title(),
            annotation_position="top left",
            annotation_font_color=label_colors.get(cat, "white"),
        )

    if sorted_cats and sorted_cats[-1][0] == "major":
        last_cat, last_val = sorted_cats[-1]
        fig.add_shape(
            type="rect",
            xref="x",  # ⬅️ anchor to actual x-axis range
            yref="y",
            x0=zoom_start,
            x1=zoom_end,
            y0=last_val,
            y1=upper_limit,
            fillcolor=soft_flood_colors.get(last_cat, "rgba(150,150,150,0.15)"),
            line_width=0,
            layer="below",
        )



def _add_observed_trace(fig: Figure, observed_df: pd.DataFrame, COLOR_THEME: dict[str, str]) -> None:
    fig.add_trace(
        go.Scatter(
            x=observed_df.index,
            y=observed_df["stage_ft"],
            mode="markers",
            name="Observed Stage",
            marker=dict(color=COLOR_THEME["observed"], size=6),
            hovertemplate="%{x|%b %d %H:%M}<br>Stage: %{y:.1f} ft<extra></extra>",
        ),
        secondary_y=False,
    )


def _add_forecast_trace(
    fig: Figure, forecast_df: pd.DataFrame, forecast_flow: pd.Series, COLOR_THEME: dict[str, str]
) -> None:
    if not forecast_df.empty:
        forecast_color = "#B266FF"  # electric purple
        border_color = "white"

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["stage_ft"],
                mode="lines+markers",
                name="Official Forecast",
                line=dict(color=forecast_color, width=5),
                marker=dict(
                    size=5,
                    symbol="square",
                    color=forecast_color,
                    line=dict(width=1.5, color=border_color)
                ),
                customdata=forecast_flow.to_numpy().reshape(-1, 1),
                hovertemplate=(
                    "%{x|%b %d %H:%M}<br>"
                    "<b>Forecast:</b> %{y:.1f} ft / %{customdata[0]:.0f} cfs"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )


def _add_nwm_model_traces(
    fig: Figure,
    model_selection: list[str],
    nwm_data: dict[str, Any],
    rating_df: pd.DataFrame,
    MODEL_CONFIG: dict[str, dict[str, str]],
    COLOR_THEME: dict[str, str],
) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    rating_dict = rating_df.to_dict("list")

    for model_key in MODEL_CONFIG:
        if model_key not in model_selection:
            continue

        config = MODEL_CONFIG[model_key]
        raw = nwm_data.get(model_key, {}).get("series" if model_key != "mediumRange" else "mean", {})
        if not raw:
            continue

        raw_ref = raw.get("referenceTime")
        ref_time = pd.to_datetime(raw_ref) if raw_ref is not None else pd.Timestamp.now()
        raw_data = raw.get("data", [])
        if not raw_data:
            continue

        map_used = None
        for key in ["flow", "primary", "value"]:
            if key in raw_data[0]:
                map_used = {key: "flow_cfs"}
                break

        if map_used:
            df = preprocess_stageflow(raw_data, map_used, rating_dict)
            if not df.empty and "flow_cfs" in df.columns:
                df["stage_ft"] = interpolate_stage_cached(df["flow_cfs"], rating_dict)

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["stage_ft"],
                        mode="lines",
                        name=f"{config['label']} ({ref_time.strftime('%m/%d %H:%M')})",
                        line=dict(color=config["color"], dash="dash"),
                        customdata=df["flow_cfs"].to_numpy().reshape(-1, 1),
                        hovertemplate=(
                            f"%{{x|%b %d %H:%M}}<br>{config['label']}: "
                            "%{y:.1f} ft / %{customdata[0]:.0f} cfs<extra></extra>"
                        ),
                    ),
                    secondary_y=False,
                )

                if not df["stage_ft"].empty:
                    peak_time = df["stage_ft"].idxmax()
                    peaks.append(
                        {
                            "Model": config["label"],
                            "Reference Time (UTC)": ref_time.strftime("%Y-%m-%d %H:%M"),
                            "Peak Time (UTC)": peak_time.strftime("%Y-%m-%d %H:%M"),
                            "Peak Stage (ft)": round(df.loc[peak_time]["stage_ft"], 2),
                            "Peak Flow (cfs)": round(df.loc[peak_time]["flow_cfs"], 0),
                        }
                    )

    return peaks


def _compute_yaxis_range(
    observed_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_selection: list[str],
    nwm_data: dict[str, Any],
    rating_df: pd.DataFrame,
    flood_lines: dict[str, tuple[float, str]],
    show_thresholds: list[str],
) -> tuple[float, float]:
    all_stages: list[float] = []

    if not observed_df.empty:
        all_stages += observed_df["stage_ft"].tolist()
    if not forecast_df.empty:
        all_stages += forecast_df["stage_ft"].tolist()

    for model in model_selection:
        raw = nwm_data.get(model, {}).get("series" if model != "mediumRange" else "mean", {})
        df = parse_timeseries(raw.get("data", []), {"flow": "flow_cfs"})
        if not df.empty:
            df["stage_ft"] = interpolate_stage_cached(df["flow_cfs"], rating_df)
            all_stages += df["stage_ft"].tolist()

    threshold_vals = [flood_lines[k][0] for k in show_thresholds if k in flood_lines]
    all_stages += threshold_vals

    if not all_stages:
        return 10.0, 0.0

    stage_min = min(all_stages)
    stage_max = max(all_stages)
    upper_limit = max(stage_max * 1.15, stage_max + 1.0)
    yaxis_min = max(0.0, stage_min - 0.5)

    if upper_limit - yaxis_min < 2.5:
        yaxis_min = max(0.0, upper_limit - 2.5)

    return upper_limit, yaxis_min


def _apply_layout(
    fig: Figure,
    metadata: dict[str, Any],
    nws_id: str,
    zoom_start: Any,
    zoom_end: Any,
    upper_limit: float,
    yaxis_min: float,
) -> None:
    site_name = metadata.get("name", "Unknown Site")
    title_text = (
        f"<b>💧 {site_name}</b><br>"
        f"<span style='font-size:14px; color:#aaa;'>NWS ID: {nws_id}, Reach ID: {metadata.get('reachId', 'N/A')}</span>"
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18),
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
        ),
        xaxis_title="Date/Time",
        yaxis=dict(title="Stage (ft)", showgrid=True, gridcolor="#333", range=[yaxis_min, upper_limit]),
        yaxis2=dict(title="Flow (cfs)", overlaying="y", side="right", showgrid=False),
        template="plotly_white" if st.session_state.get("theme") == "Light" else "plotly_dark",
        height=750,
        margin=dict(l=60, r=40, t=80, b=100),
        dragmode="pan",
        hovermode="x unified",
        xaxis=dict(tickformat="%b %d", rangeslider=dict(visible=True), range=[zoom_start, zoom_end]),
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5, traceorder="normal"),
        hoverlabel=dict(bgcolor="#1f1f1f", font_size=13, font_color="white"),
    )


def build_hydrograph(
    observed_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_flow: pd.Series,
    rating_df: pd.DataFrame,
    nwm_data: dict[str, Any],
    flood_lines: dict[str, tuple[float, str]],
    show_thresholds: list[str],
    model_selection: list[str],
    zoom_start: Any,
    zoom_end: Any,
    metadata: dict[str, Any],
    nws_id: str,
    MODEL_CONFIG: Optional[dict[str, dict[str, str]]] = None,
    COLOR_THEME: Optional[dict[str, str]] = None,
) -> tuple[Figure, list[dict[str, Any]]]:
    MODEL_CONFIG = MODEL_CONFIG or DEFAULT_MODEL_CONFIG
    COLOR_THEME = COLOR_THEME or DEFAULT_COLOR_THEME

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    _add_observed_trace(fig, observed_df, COLOR_THEME)
    _add_forecast_trace(fig, forecast_df, forecast_flow, COLOR_THEME)
    peaks = _add_nwm_model_traces(fig, model_selection, nwm_data, rating_df, MODEL_CONFIG, COLOR_THEME)
    upper_limit, yaxis_min = _compute_yaxis_range(
        observed_df, forecast_df, model_selection, nwm_data, rating_df, flood_lines, show_thresholds
    )
    add_flood_threshold_shading(fig, flood_lines, show_thresholds, zoom_start, zoom_end, upper_limit)
    _apply_layout(fig, metadata, nws_id, zoom_start, zoom_end, upper_limit, yaxis_min)

    return fig, peaks
