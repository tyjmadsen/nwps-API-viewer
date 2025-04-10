# --- Streamflow Hydrograph Viewer App ---
import streamlit as st
from streamlit import session_state as ss
from streamlit.components.v1 import html
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import logging
from logger import logger
import plotly.express as px

# --- Module Imports ---
from api import (
    list_all_gauges,
    get_metadata,
    get_stageflow,
    get_ratings,
    get_nwm_streamflow,
    get_latest_hefs_headers,
    fetch_hefs_timeseries
)

from logic import (
    load_data,
    parse_timeseries,
    interpolate_flow,
    interpolate_stage_cached,
    compute_percentile_bands,
    compute_exceedance_probabilities,
    compute_and_interpolate_percentiles,
    preprocess_stageflow
)

from plots import build_hydrograph, add_ensemble_percentile_bands
from ui import build_sidebar_controls, flood_lines_from_metadata, render_gauge_map, render_gauge_map_esri

from fragments import (
    render_hydrograph,
    render_ensemble_bands,
    render_peak_table,
    render_rating_curve
)

logger.setLevel(logging.DEBUG)

# --- Flood Stage Color Mapping ---
COLOR_THEME = {
    "threshold_action": "#FFFF00",   # yellow
    "threshold_minor": "#FFA500",    # orange
    "threshold_moderate": "#FF0000", # red
    "threshold_major": "#DA00FF",    # purple
}

FIM_CATEGORY_COLORS = {
    "Low": "#C2E699",  # soft green
    "Near": "#78C679",  # green
    "Minor": COLOR_THEME["threshold_minor"],
    "Moderate": COLOR_THEME["threshold_moderate"],
    "Major": COLOR_THEME["threshold_major"],
}


def show_fim_map(nws_id: str, lat: float, lon: float, selected_layer_ids: list[int]):
    fim_url = f"https://mapservices.weather.noaa.gov/static/rest/services/NWS_FIM/FIM_{nws_id.lower()}/MapServer"

    # Fallback to default layers if none selected
    if not selected_layer_ids:
        selected_layer_ids = [27, 28, 29]  # Minor, Moderate, Major

    st.write("📋 Selected FIM Layers:", selected_layer_ids)

    # Format the list as 'show:' string for dynamicMapLayer
    layer_list = "show:" + ",".join(str(layer_id) for layer_id in selected_layer_ids)

    iframe_html = f"""
    <div id=\"map\" style=\"height: 100vh; width: 100%;\"></div>
    <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.3/dist/leaflet.css\" />
    <script src=\"https://unpkg.com/leaflet@1.9.3/dist/leaflet.js\"></script>
    <script src=\"https://unpkg.com/esri-leaflet@3.0.4/dist/esri-leaflet.js\"></script>
    <script>
      var map = L.map('map').setView([{lat}, {lon}], 13);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 18
      }}).addTo(map);

      var fimLayer = L.esri.dynamicMapLayer({{
        url: '{fim_url}',
        layers: '{layer_list}',
        fromCache: false,
        opacity: 0.65
      }}).addTo(map);

      fimLayer.on('error', function(err) {{
        console.log('❌ ESRI FIM load error:', err);
        alert('Error loading FIM layer. Check console for details.');
      }});
    </script>
    """

    html(iframe_html, height=700)

def get_contrasting_text_color(hex_color):
    """Return black or white depending on background brightness for readability."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#000" if brightness > 160 else "#fff"

# --- Define card layout wrapper ---
def sidebar_card_block(title: str, body_fn):
    bg_color = "#ffffff" if theme == "Light" else "#1a1a2e"
    title_color = "#0b5394" if theme == "Light" else "#4da6ff"
    
    st.sidebar.markdown(f"""
    <div style="background-color: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
        <h4 style="color: {title_color}; margin-top: 0;">{title}</h4>
    """, unsafe_allow_html=True)


def get_color_theme():
    return {
        "observed": "firebrick",
        "official_forecast": "royalblue",
        "threshold_action": "#FFD700" if st.session_state["theme"] == "Light" else "#FFFF00",
        "threshold_minor": "#FF8C00" if st.session_state["theme"] == "Light" else "#FFA500",
        "threshold_moderate": "#DC143C" if st.session_state["theme"] == "Light" else "#FF0000",
        "threshold_major": "#8A2BE2" if st.session_state["theme"] == "Light" else "#DA00FF",
    }


# --- Setup ---
st.set_page_config(layout="wide", page_title="Streamflow Viewer")

def theme_toggle_buttons():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Dark"

    theme = st.session_state["theme"]

    # Show Light button first (left), then Dark
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🌞 Light", key="theme_light_btn"):
            st.session_state["theme"] = "Light"

    with col2:
        if st.button("🌙 Dark", key="theme_dark_btn"):
            st.session_state["theme"] = "Dark"

    # Apply stronger style targeting
    st.markdown("""
        <style>
        /* Basic button style */
        [data-testid="stSidebar"] button[kind="secondary"] {
            border-radius: 10px;
            border: 2px solid transparent;
            background-color: #ffffff !important;
            color: #000000 !important;
            font-weight: 600;
            padding: 0.3rem 0.6rem;
        }
    
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: #999999;
        }
    
        /* Force all text and icons (emoji) inside buttons to black */
        [data-testid="stSidebar"] button[kind="secondary"] * {
            color: #000000 !important;
            fill: #000000 !important;
        }
    
        /* Selected button style (just show a border, don't mess with colors) */
        [data-testid="stSidebar"] button[aria-pressed="true"] {
            border: 2px solid #ff4b4b !important;
        }
        </style>
    """, unsafe_allow_html=True)


def apply_custom_css():
    if st.session_state.get("theme") == "Light":
        st.markdown("""
            <style>
            body {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .stSidebar {
                background-color: #f3f3f3 !important;
                color: #111 !important;
            }
            .stSidebar .css-1v0mbdj, .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
                color: #111 !important;
            }
            .stRadio > div > label,
            .stCheckbox > div > label {
                color: #111 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body {
                background-color: #0D1B2A !important;
                color: #e6f1ff !important;
            }
    
            .stSidebar {
                background-color: #1B263B !important;
                color: #e6f1ff !important;
            }
    
            /* General text and headers */
            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
            .stSidebar p, .stSidebar label, .stSidebar span {
                color: #e6f1ff !important;
            }
    
            /* Checkbox and radio labels */
            .stCheckbox > div > label,
            .stRadio > div > label {
                color: #e6f1ff !important;
            }
    
            /* Help icons and secondary text */
            .stSidebar .css-15zrgzn,
            .stSidebar .css-1aehpvj,
            .stSidebar .css-1v4eu6x,
            .stSidebar .css-1kyxreq {
                color: #aaa !important;
            }
    
            /* Force all icons and emojis to inherit brighter color */
            .stSidebar span, .stSidebar div {
                color: #e6f1ff !important;
            }
            </style>
        """, unsafe_allow_html=True)

theme_toggle_buttons()
theme = st.session_state["theme"]
apply_custom_css()


# --- Sidebar Header / Branding ---
sidebar_bg = "#f9f9f9" if theme == "Light" else "linear-gradient(90deg, #0D1B2A, #1B263B)"
text_color = "#111" if theme == "Light" else "#e6f1ff"
subtext_color = "#333" if theme == "Light" else "#bbb"
title_color = "#0b5394" if theme == "Light" else "#4da6ff"

st.sidebar.markdown(f"""
<div style="
    background: {sidebar_bg};
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3);
    color: {text_color};
">
    <h2 style="margin: 0; font-size: 1.45rem; color: {title_color};">📈 NWPS API Viewer</h2>
    <p style="font-size: 0.85rem; color: {subtext_color}; margin-top: 0.3rem;">
        Visualize observed and forecast river stage and flow.
    </p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar: Site Selection ---
st.sidebar.markdown("""
<div style="margin-bottom: -2.9rem;">
  <h3 style="margin: 0 0 -0.7rem 0; padding: 0;">📍 Site Selection</h3>
</div>
<style>
[data-testid="stSidebar"] section[data-testid="stTextInput"] {
    margin-top: -2.4rem !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

use_gauge_search = ss.get("use_gauge_search", False)

if use_gauge_search:
    try:
        with st.spinner("📡 Loading gauges..."):
            gauges = list_all_gauges()

        gauge_query = st.sidebar.text_input("Type gauge name or ID")

        filtered_gauges = [
            g for g in gauges
            if gauge_query.upper() in g["lid"].upper() or gauge_query.lower() in g["name"].lower()
        ] if gauge_query else gauges[:15]

        if filtered_gauges:
            display_options = [f"{g['name']} ({g['lid']})" for g in filtered_gauges]
            selected_label = st.sidebar.selectbox("Matching gauges:", display_options, key="match_select")
            nws_id = selected_label.split("(")[-1].replace(")", "")
        else:
            nws_id = st.sidebar.text_input(
                label="",
                value="",
                key="manual_fallback",
                placeholder="Enter NWS ID",
                help="Enter NWS ID manually if no match found"
            )
    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed to load gauge list.\n\n{e}")
        nws_id = st.sidebar.text_input(
            label="",
            value="",
            key="manual_fallback_error",
            placeholder="Enter NWS ID",
            help="Enter NWS ID manually (e.g., CDFP1)"
        )
else:
    nws_id = st.sidebar.text_input(
        label="",
        value="CDFP1",
        key="manual_input",
        placeholder="Enter NWS ID",
        help="Enter NWS ID (e.g., CDFP1)"
    )


# --- Sidebar: Advanced Settings ---
with st.sidebar.expander("🛠️ Advanced Settings"):
    previous_value = ss.get("use_gauge_search", False)
    new_value = st.checkbox(
        "🔍 Query All Gauges",
        value=previous_value,
        help="Enable a searchable list of all NWPS gauges. May take a moment to load."
    )

    # Update and rerun only if value changed
    if new_value != previous_value:
        ss.use_gauge_search = new_value
        st.rerun()

    bypass_cache = st.checkbox("Bypass cache (force fresh reload)")

# --- Main Logic ---
if nws_id:
    try:
        time.sleep(0.2)

        # Step 1: Load core data
        with st.spinner("🔄 Loading data..."):
            if nws_id not in st.session_state or bypass_cache:
                st.session_state[nws_id] = load_data(
                    nws_id,
                    model_selection=[],  # we apply filter later
                    bypass=bypass_cache,
                    get_metadata=get_metadata,
                    get_stageflow=get_stageflow,
                    get_ratings=get_ratings,
                    get_nwm_streamflow=get_nwm_streamflow
                )

            metadata, stream_data, rating_df, nwm_data = st.session_state[nws_id]

        # Step 2: Flood lines & sidebar controls
        flood_lines = flood_lines_from_metadata(metadata)
        show_thresholds, model_selection, show_ensemble_flag, show_hefs_risk_table, show_fim = build_sidebar_controls(flood_lines)
        # Persist the ensemble toggle in session state
        if "show_ensemble" not in st.session_state:
            st.session_state["show_ensemble"] = show_ensemble_flag
        elif st.session_state.show_ensemble != show_ensemble_flag:
            st.session_state.show_ensemble = show_ensemble_flag
            st.rerun()  # Force a rerun when toggled to avoid loss on rerender
        show_ensemble = st.session_state.show_ensemble


        # Step 3: Date range
        now = datetime.now(timezone.utc)
        
        # -- Set zoom range early (using session value or default) for internal logic
        days_range = ss.get("zoom_slider_main", 6)
        
        zoom_start = now - timedelta(days=2)
        zoom_end = now + timedelta(days=days_range)

        # Step 4: Parse time series
        
        forecast_data = stream_data.get("forecast") or {}
        observed_data = stream_data.get("observed") or {}
        
        observed_raw = observed_data.get("data", [])
        forecast_raw = forecast_data.get("data") if isinstance(forecast_data, dict) else []


        # Step 4: Parse time series + use session cache

        obs_key = f"{nws_id}_observed_df"
        fcst_key = f"{nws_id}_forecast_df"
        flow_key = f"{nws_id}_forecast_flow"
        
        rating_dict = rating_df.to_dict("list")
        
        # --- Observed Timeseries
        if obs_key not in st.session_state:
            st.session_state[obs_key] = preprocess_stageflow(
                observed_raw, {"primary": "stage_ft", "secondary": "flow_kcfs"}, rating_dict
            )
        observed_df = st.session_state[obs_key]
        
        # --- Forecast Timeseries
        if fcst_key not in st.session_state:
            st.session_state[fcst_key] = preprocess_stageflow(
                forecast_raw, {"primary": "stage_ft", "secondary": "flow_kcfs"}, rating_dict
            )
        forecast_df = st.session_state[fcst_key]
        
        # --- Forecast Flow Interpolated
        if flow_key not in st.session_state:
            if not forecast_df.empty and "stage_ft" in forecast_df.columns:
                st.session_state[flow_key] = forecast_df["flow_kcfs"]
            else:
                st.session_state[flow_key] = pd.Series()
        forecast_flow = st.session_state[flow_key]



        # Step 5: Safety net if no data
        has_nwm_data = any(
            nwm_data.get(model, {}).get("series" if model != "mediumRange" else "mean", {}).get("data")
            for model in model_selection
        )

        if observed_df.empty and forecast_df.empty and not has_nwm_data:
            st.warning("⚠️ No observed or forecast model data available for this location.")
            st.stop()

        # Step 6: Forecast summary block
        forecast_issued_raw = forecast_data.get("issuedTime", "N/A")
        try:
            forecast_dt = pd.to_datetime(forecast_issued_raw)
            forecast_issued = "N/A" if forecast_dt.year < 1677 else forecast_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            forecast_issued = "N/A"

        forecast_status = (
            "<span style='background-color:#ff4444;padding:2px 6px;border-radius:6px;color:white;'>⚠️ No official forecast</span>"
            if forecast_df.empty else f"<code style='background:#333;padding:2px 6px;border-radius:6px;'>{forecast_issued}</code>"
        )


        hefs_init = ""
        ensemble_meta = []
        ensemble_df = pd.DataFrame()
        percentiles_interp_key = f"{nws_id}_percentiles_interp"

        if show_ensemble:
            with st.spinner("🕒 Checking for HEFS availability..."):
                ensemble_meta_raw = get_latest_hefs_headers(nws_id)
                ensemble_meta = [h for h in ensemble_meta_raw if h.get("id")]

            if ensemble_meta:
                init_date = ensemble_meta[0].get("forecast_date_date")
                init_time = ensemble_meta[0].get("forecast_date_time")
                hefs_init_dt = pd.to_datetime(f"{init_date} {init_time}")
                hefs_init = hefs_init_dt.strftime("%Y-%m-%d %H:%M UTC")

                ensemble_key = f"{nws_id}_ensemble_df"
                if ensemble_key not in st.session_state:
                    st.session_state[ensemble_key] = fetch_hefs_timeseries(ensemble_meta)

                ensemble_df = st.session_state[ensemble_key]

                if percentiles_interp_key not in st.session_state:
                    logger.info(f"💡 ensemble_df shape before percentiles: {ensemble_df.shape}")
                    logger.info(f"ensemble_df columns: {ensemble_df.columns.tolist()}")
                    logger.info(f"ensemble_df head:\n{ensemble_df.head()}")
                
                    st.session_state[percentiles_interp_key] = compute_and_interpolate_percentiles(
                        ensemble_df,
                        rating_df.to_dict("list")
                    )
                
                    percentiles = st.session_state[percentiles_interp_key]
                    logger.info(f"✅ Interpolated percentiles keys: {list(percentiles.keys())}")
                    for k, s in percentiles.items():
                        logger.info(f"Percentile {k}: {s.notna().sum()} non-NaN out of {len(s)}")

            else:
                st.info("⚠️ No HEFS ensemble forecast data available.")
                show_ensemble = False


        if hefs_init and ensemble_meta:
            # Extract years from ensemble_member_index (e.g., "1953", "1954", ..., "2017")
            indices = []
            for h in ensemble_meta:
                try:
                    idx = int(float(h.get("ensemble_member_index", 0)))
                    if not pd.isna(idx):
                        indices.append(idx)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid ensemble_member_index: {h.get('ensemble_member_index')}")

            if indices:
                span_start = min(indices)
                span_end = max(indices)
                count = len(indices)
                hefs_display = (
                    f"<code>{hefs_init}</code> "
                    f"({count} members, {span_start}-{span_end})"
                )
            else:
                hefs_display = f"<code>{hefs_init}</code>"
        else:
            hefs_display = ""

        latest_obs = observed_df.index.max().strftime("%Y-%m-%d %H:%M UTC") if not observed_df.empty else "N/A"

       
        # Build metadata card (concatenated for full safety)
        panel_bg = "#f9f9f9" if theme == "Light" else "linear-gradient(90deg, #0D1B2A, #1B263B)"
        panel_text = "#111" if theme == "Light" else "#e6f1ff"
        accent = "#0b5394" if theme == "Light" else "#a0cfff"
        
        panel_html = (
            f"<div style=\"background: {panel_bg}; padding: 1.25rem 1.75rem; "
            f"border-radius: 14px; color: {panel_text}; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 1.5rem;\">"
            f"<h3 style=\"margin: 0 0 0rem 0; font-size: 1.6rem;\">"
            f"🌊 {metadata['name']} <span style=\"font-size: 0.9rem; color: {accent};\">({nws_id})</span></h3>"
            f"<div style=\"font-size: 0.95rem;\">"
            f"<p style=\"margin: 0.1rem 0;\">📍 <strong>Latest Observation:</strong> "
            f"<code style='color:#fff;background:#333;padding:2px 6px;border-radius:6px;'>{latest_obs}</code></p>"
            f"<p style=\"margin: 0.1rem 0;\">🕓 <strong>Forecast Issued:</strong> {forecast_status}</p>"
        )


        # Print metadata HTML card first
        st.markdown(panel_html, unsafe_allow_html=True)
               
        # Add HEFS block separately
        if hefs_display:
            st.markdown(
                f"<p style='margin: 0.3rem 0;'>🕒 <strong>HEFS Runtime:</strong> {hefs_display}</p></div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("</div></div>", unsafe_allow_html=True)
            
        # --- Render Zoom Forecast Slider just below metadata card ---
        days_range = st.slider(
            "🔍 Zoom Forecast (Days)",
            min_value=1,
            max_value=14,
            value=days_range,
            key="zoom_slider_main",
            help="Adjust how far into the future the forecast is shown."
        )

        # Step 7: Gauge map
        show_map = st.sidebar.checkbox("🗺️ Map", value=False, help="Show gauge location and basin.")
        
        if show_map:
            with st.expander("🗺️ View Gauge Location Map", expanded=True):
                render_gauge_map(metadata, nwm_data, nws_id, show_fim)


        # Step 8: Build and render hydrograph including ensemble bands (if any)
        percentiles_interp = st.session_state.get(percentiles_interp_key)
        forecast_init_time = hefs_init 
        
        # --- Build the hydrograph (returns fig and peaks)
        fig, peaks = render_hydrograph(
            observed_df,
            forecast_df,
            forecast_flow,
            rating_df,
            nwm_data,
            flood_lines,
            show_thresholds,
            model_selection,
            zoom_start,
            zoom_end,
            metadata,
            nws_id,
        )
             
        # Save it if you need to reference it again
        st.session_state["ensemble_figure"] = fig
        
        # Assuming you have ensemble_df (HEFS ensemble data), rating_df (your rating curve), and flood_lines available.
        if show_ensemble and not ensemble_df.empty:
            # Define forecast period boundaries
            try:
                forecast_start = ensemble_df.index[0]
                period1_end = forecast_start + pd.Timedelta(days=3)
                period2_start = period1_end
                period2_end = ensemble_df.index[-1]
            
                prob_period1 = compute_exceedance_probabilities(ensemble_df, rating_df, flood_lines, forecast_start, period1_end)
                prob_period2 = compute_exceedance_probabilities(ensemble_df, rating_df, flood_lines, period2_start, period2_end)
                logger.debug(f"📊 Exceedance inputs:\nprob_period1 = {prob_period1}\nprob_period2 = {prob_period2}")
            
                def get_color(prob1, prob2):
                    max_val = max(prob1, prob2)
                    if max_val >= 50: return "#FF0000"
                    elif max_val >= 20: return "#FF8000"
                    elif max_val >= 5: return "#FFD700"
                    return "#AAAAAA"
            
                def get_shade(prob1, prob2):
                    max_val = max(prob1, prob2)
                    if max_val >= 50: return "🟥"
                    elif max_val >= 20: return "🟧"
                    elif max_val >= 5: return "🟨"
                    return "⬛️"
                
                def get_risk_label(p1, p2):
                    max_p = max(p1 or 0, p2 or 0)
                    if max_p >= 50:
                        return "<span style='color:#FF3333;'>🟥 High</span>"
                    elif max_p >= 20:
                        return "<span style='color:#FFA500;'>🟧 Medium</span>"
                    elif max_p >= 5:
                        return "<span style='color:#FFD700;'>🟨 Low</span>"
                    return "<span style='color:#AAAAAA;'>⬛️ Unlikely</span>"

                
                # Build the lines
                lines = [
                    " Category    D1–3  D4–10  10 Day Risk",
                    f"🟨 Action   {prob_period1.get('action', 0):>3.0f}%   {prob_period2.get('action', 0):>3.0f}%   {get_risk_label(prob_period1.get('action', 0), prob_period2.get('action', 0))}",
                    f"🟧 Minor    {prob_period1.get('minor', 0):>3.0f}%   {prob_period2.get('minor', 0):>3.0f}%   {get_risk_label(prob_period1.get('minor', 0), prob_period2.get('minor', 0))}",
                    f"🟥 Moderate {prob_period1.get('moderate', 0):>3.0f}%   {prob_period2.get('moderate', 0):>3.0f}%   {get_risk_label(prob_period1.get('moderate', 0), prob_period2.get('moderate', 0))}",
                    f"🟪 Major    {prob_period1.get('major', 0):>3.0f}%   {prob_period2.get('major', 0):>3.0f}%   {get_risk_label(prob_period1.get('major', 0), prob_period2.get('major', 0))}",
                ]
                
                # Combine into HTML with <br> line breaks
                mini_table_text = (
                    "<b style='font-size:26px;'>HEFS 10 Day Flood Risk</b><br>"
                    "<span style='font-size:12px; color:#bbb; display:block; margin:2px 0;'>"
                    "Days 1–3 & 4–10: Probs of exceeding flood categories."
                    "</span><br>"
                    "<span style='font-family:monospace; white-space:pre-wrap;'>"
                    + "<br>".join(lines) +
                    "</span><br><span style='font-size:11px; color:#ccc;'>"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⬛️ &lt;5%&nbsp;&nbsp;&nbsp;&nbsp;🟨 ≥5%&nbsp;&nbsp;&nbsp;&nbsp;🟧 ≥20%&nbsp;&nbsp;&nbsp;&nbsp;🟥 ≥50%</span>"
                )

                if show_hefs_risk_table:
                    fig.add_annotation(
                        text=mini_table_text,
                        x=0.94,
                        y=1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        bordercolor="#1E293B",
                        borderwidth=1,
                        borderpad=6,
                        bgcolor="rgba(11, 15, 25, 0.9)",
                        font=dict(size=13, color="#FFFFFF"),
                        align="left"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Skipping exceedance mini-table due to error: {e}", exc_info=True)

         
        # --- If ensemble is toggled and available, add to same figure
        if show_ensemble and percentiles_interp:
            add_ensemble_percentile_bands(fig, percentiles_interp, rating_df, forecast_init_time)
        
        
        
        # --- Now render the figure
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, key="main_hydrograph")
        
        # Step 8z: Display Clean Threshold Table Below Hydrograph
        threshold_order = ["action", "minor", "moderate", "major"]
        header_cells = ""
        value_cells = ""
        
        for level in threshold_order:
            if level in flood_lines:
                val, bg_color = flood_lines[level]
                logger.debug(f"🔎 Checking threshold '{level}': raw value = {val}, bg = {bg_color}")
        
                try:
                    numeric_val = float(val)
                    if pd.isna(numeric_val):
                        logger.warning(f"⚠️ Threshold '{level}' is NaN. Skipping...")
                        continue  # 🔁 Skip this level
                except (ValueError, TypeError) as e:
                    logger.error(f"❌ Failed to convert threshold '{level}' value '{val}' → float: {e}")
                    continue  # 🔁 Skip this level too
        
                text_color = get_contrasting_text_color(bg_color)
                header_cells += (
                    f"<th style='padding: 6px 10px; background-color:{bg_color}; color:{text_color}; "
                    f"border-radius: 4px 4px 0 0;'>{level.title()}</th>"
                )
                value_cells += f"<td style='padding: 6px 10px; color: #eaeaea;'>{numeric_val:.2f}</td>"




        
        threshold_table = (
            "<div style='margin: .1rem 0;'>"
            "<table style='font-size: 1rem; border-collapse: collapse; width: 100%; text-align: center;'>"
            "<thead><tr>" + header_cells + "</tr></thead>"
            "<tbody><tr style='background-color: #111827;'>" + value_cells + "</tr></tbody>"
            "</table></div>"
        )
        
        st.markdown(threshold_table, unsafe_allow_html=True)
        
        
        # Step 9: Show peak table and Rating curve preview (lazy)
        tab1, tab2 = st.tabs(["📊 Forecast Peaks", "📉 Rating Curve"])
        with tab1:
            render_peak_table(peaks)  # ✅ Use the correct variable name
        with tab2:
            render_rating_curve(rating_df)
   
    

# =============================================================================
#         # === Simple FIM Layer Viewer (Optional)
#         with st.expander("🛰️ View FIM Flood Layer Map", expanded=True):
#             st.markdown("Customize which flood severity layers to overlay on the map.")
#         
#             fim_layers = {
#                 "Low": 25,
#                 "Near": 26,
#                 "Minor": 27,
#                 "Moderate": 28,
#                 "Major": 29,
#             }
#         
#             selected_categories = st.multiselect(
#                 "Select Flood Severity Categories",
#                 options=list(fim_layers.keys()),
#                 default=["Minor", "Moderate", "Major"]
#             )
#         
#             selected_layer_ids = [fim_layers[cat] for cat in selected_categories]
#         
#             # Pull lat/lon from loaded metadata (fallback to known coords if needed)
#             lat = metadata.get("lat", 30.429)
#             lon = metadata.get("lon", -91.198)
#             st.write("📋 Selected FIM Layers:", selected_layer_ids)
# 
#         
#             show_fim_map(nws_id, lat, lon, selected_layer_ids)
# =============================================================================




    except Exception as e:
        import traceback
        st.error(f"❌ App error: {e}")
        st.text(traceback.format_exc())  # 🔍 show full traceback in the UI
        logger.exception("Unhandled exception in main app block")



