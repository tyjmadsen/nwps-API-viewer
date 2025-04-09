# fragments.py

import streamlit as st
import pandas as pd
from logger import logger
from plots import build_hydrograph, add_ensemble_percentile_bands


@st.fragment
def render_hydrograph(
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
    show_ensemble=False,
    percentiles_interp=None,
    forecast_init_time=None,
):
    try:
        fig, peaks = build_hydrograph(
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

        
        # Save once and render once
        st.session_state["ensemble_figure"] = fig
        #st.plotly_chart(fig, use_container_width=True, key="hydrograph_plot")

        return fig, peaks
    except Exception as e:
        import traceback
        logger.error("❌ Exception in render_hydrograph:\n" + traceback.format_exc())
        st.error("🚫 Could not render hydrograph.")
        return None, []



@st.fragment
def render_ensemble_bands(percentiles_interp, rating_df, forecast_init_time):
    try:
        if not percentiles_interp or all(s.empty for s in percentiles_interp.values()):
            st.info("📉 No ensemble forecast data available.")
            return

        fig = st.session_state.get("ensemble_figure")
        if fig:
            # Modify fig in-place
            add_ensemble_percentile_bands(fig, percentiles_interp, rating_df, forecast_init_time)

            # ✅ Ensure this gets a consistent unique key
            st.plotly_chart(fig, use_container_width=True, key="ensemble_chart")
    except Exception as e:
        logger.warning(f"Ensemble bands failed: {e}", exc_info=True)
        st.error("🚫 Could not render ensemble bands.")

@st.fragment
def render_peak_table(peaks):
    try:
        if not peaks:
            st.info("📉 No forecast peaks found.")
            return
        df = pd.DataFrame(peaks)
        st.markdown("### 📊 Forecast Peaks")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        logger.warning(f"Peak table failed: {e}")
        st.error("🚫 Could not render peak table.")


@st.fragment
def render_rating_curve(rating_df):
    try:
        if rating_df.empty:
            st.info("📉 Rating curve not available.")
            return
        st.line_chart(rating_df.set_index("flow")["stage"])
    except Exception as e:
        logger.warning(f"Rating curve failed: {e}")
        st.error("🚫 Could not render rating curve.")
