import streamlit as st
from streamlit import session_state as ss
import folium
from streamlit.components.v1 import html
from typing import Any, Dict, List, Tuple

from api import get_usgs_basin_geojson, get_reach_centroid

# --- Flood Threshold Colors (match hydrograph_app.py) ---
COLOR_THEME: Dict[str, str] = {
    "threshold_action": "#FFFF00",
    "threshold_minor": "#FFA500",
    "threshold_moderate": "#FF0000",
    "threshold_major": "#DA00FF",
}


@st.fragment
def render_lazy_map(
    metadata: Dict[str, Any], nwm_data: Dict[str, Any], nws_id: str
) -> None:
    render_gauge_map(metadata, nwm_data, nws_id)


def flood_lines_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Tuple[float, str]]:
    cats = metadata.get("flood", {}).get("categories", {})
    return {
        k: (v["stage"], COLOR_THEME[f"threshold_{k}"])
        for k, v in cats.items()
        if v.get("stage")
    }


def build_sidebar_controls(flood_lines: Dict[str, Tuple[float, str]]) -> Tuple[List[str], int, List[str], bool, bool, bool]:
    from streamlit import session_state as ss

    # --- Flood Thresholds Block ---
    st.sidebar.markdown("""
    <div style="margin-bottom: -2rem;">
        <h4 style="margin-bottom: -0.5rem;">⚠️ Flood Thresholds</h4>
    </div>
    <style>
    /* Tighten spacing on multiselect container in sidebar */
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] {
        margin-top: -1.4rem;
        margin-bottom: -0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    threshold_order = ["action", "minor", "moderate", "major"]
    threshold_labels = {
        "action": "🟡 Action",
        "minor": "🟠 Minor",
        "moderate": "🔴 Moderate",
        "major": "🟣 Major"
    }
    raw_selection = st.sidebar.multiselect(
        label="",
        options=threshold_order,
        format_func=lambda k: threshold_labels[k],
        default=["action", "minor"],
        key="flood_threshold_select"
    )
    max_idx = max((threshold_order.index(k) for k in raw_selection), default=1)
    show_thresholds = threshold_order[:max_idx + 1]


    # --- Ensemble Settings ---
    st.sidebar.markdown("### 📈 Forecast Settings")
    with st.sidebar.container():
        col1, col2 = st.columns(2)
        with col1:
            show_ensemble = st.checkbox("HEFS Ensembles", value=False, key="show_ensemble_checkbox")
        with col2:
            show_hefs_risk_table = st.checkbox("Risk Table", value=True, disabled=not show_ensemble, key="show_risk_table_checkbox")

    # --- Model Selection ---
    with st.sidebar.expander("🧪 Forecast Models"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ All Models"):
                ss.model_selection = ["shortRange", "mediumRange", "mediumRangeBlend", "analysisAssimilation"]
        with col2:
            if st.button("🚫 Clear"):
                ss.model_selection = []

        if "model_selection" not in ss:
            ss.model_selection = ["shortRange", "mediumRange", "mediumRangeBlend", "analysisAssimilation"]

        model_selection = st.multiselect(
            "Select NWM Models:",
            options=["shortRange", "mediumRange", "mediumRangeBlend", "analysisAssimilation"],
            default=ss.model_selection,
            key="model_selection"
        )

    # --- Optional FIM Toggle ---
    show_fim = st.sidebar.checkbox("🛰️ Show FIM Layer", value=True)

    return show_thresholds, model_selection, show_ensemble, show_hefs_risk_table, show_fim

def render_gauge_map_esri(nws_id: str, lat: float, lon: float) -> None:
    fim_service_url = f"https://mapservices.weather.noaa.gov/static/rest/services/NWS_FIM/FIM_{nws_id.lower()}/MapServer"
    
    iframe_html = f"""
    <div id="map" style="height: 480px; width: 100%;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://unpkg.com/esri-leaflet@3.0.4/dist/esri-leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{lat}, {lon}], 13);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19
        }}).addTo(map);

        L.esri.dynamicMapLayer({{
            url: '{fim_service_url}',
            layers: [0, 27, 28, 29],  // Minor, Moderate, Major (categorical)
            opacity: 0.65
        }}).addTo(map);
    </script>
    """
    html(iframe_html, height=500)

def render_gauge_map(
    metadata: Dict[str, Any], nwm_data: Dict[str, Any], nws_id: str, show_fim: bool
) -> None:
    try:
        lat = metadata.get("lat") or nwm_data.get("reach", {}).get("latitude")
        lon = metadata.get("lon") or nwm_data.get("reach", {}).get("longitude")

        if lat is None or lon is None:
            st.info("📍 Gauge location not available.")
            return

        fmap = folium.Map(
            location=[lat, lon],
            zoom_start=10,
            tiles="CartoDB positron",
            control_scale=True,
        )
        fim = metadata.get("fim", {})
        if show_fim and fim.get("fim_available"):
            render_gauge_map_esri(nws_id, lat, lon)
            return  # Skip Folium if FIM (ESRI) is shown



        folium.Marker(
            location=[lat, lon],
            tooltip=f"{metadata['name']} ({nws_id})",
            icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
        ).add_to(fmap)

        usgs_id = metadata.get("usgsId")
        if usgs_id:
            basin_geojson = get_usgs_basin_geojson(usgs_id)
            if basin_geojson:
                folium.GeoJson(
                    data=basin_geojson,
                    name="Upstream Basin",
                    tooltip="USGS NLDI Upstream Basin",
                    style_function=lambda feature: {
                        "fillColor": "#4a89dc",
                        "color": "#4a89dc",
                        "weight": 2,
                        "fillOpacity": 0.2,
                    },
                ).add_to(fmap)

        upstream = nwm_data.get("reach", {}).get("route", {}).get("upstream", [])
        for reach in upstream:
            rid = reach.get("reachId")
            ulat, ulon = get_reach_centroid(rid)
            if ulat and ulon:
                folium.Marker(
                    location=[ulat, ulon],
                    tooltip=f"Upstream Reach: {rid}",
                    icon=folium.Icon(color="green", icon="arrow-up", prefix="fa"),
                ).add_to(fmap)
                folium.PolyLine(
                    [(ulat, ulon), (lat, lon)], color="green", weight=2, dash_array="4"
                ).add_to(fmap)

        downstream = nwm_data.get("reach", {}).get("route", {}).get("downstream", [])
        for reach in downstream:
            rid = reach.get("reachId")
            dlat, dlon = get_reach_centroid(rid)
            if dlat and dlon:
                folium.Marker(
                    location=[dlat, dlon],
                    tooltip=f"Downstream Reach: {rid}",
                    icon=folium.Icon(color="red", icon="arrow-down", prefix="fa"),
                ).add_to(fmap)
                folium.PolyLine(
                    [(lat, lon), (dlat, dlon)], color="red", weight=2, dash_array="4"
                ).add_to(fmap)

        folium.LayerControl().add_to(fmap)
        html(fmap._repr_html_(), height=350)

    except Exception as e:
        st.warning(f"❌ Could not render map: {e}")
