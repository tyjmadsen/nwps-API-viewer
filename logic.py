import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from logger import logger

@st.cache_data(hash_funcs={dict: lambda x: hash(frozenset(x.items()))})
def parse_timeseries(
    data: List[Dict[str, Any]], column_map: Dict[str, str]
) -> pd.DataFrame:
    if not isinstance(data, list) or not data:
        logger.warning("parse_timeseries: input data is not a list or is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(data).rename(columns=column_map)
    logger.info(f"parse_timeseries: Dataframe shape after renaming: {df.shape}")

    if "validTime" not in df.columns:
        logger.warning("parse_timeseries: Missing 'validTime' in data columns.")
        st.warning("⚠️ Missing validTime in timeseries data.")
        return pd.DataFrame()

    try:
        df = df[df["validTime"] != "0001-01-01T00:00:00Z"]
        df["validTime"] = pd.to_datetime(df["validTime"], errors="coerce")
        df = df.dropna(subset=["validTime"])
    except Exception as e:
        logger.exception("parse_timeseries: Failed to parse validTime.")
        st.warning("⚠️ Could not parse validTime.")
        return pd.DataFrame()

    logger.info(f"parse_timeseries: Returning cleaned dataframe with shape {df.shape}")
    return df.set_index("validTime").sort_index().replace(-999, pd.NA).dropna(how="any")


@st.cache_data(ttl=3600)
def preprocess_stageflow(
    raw_data: List[Dict[str, Any]],
    column_map: Dict[str, str],
    rating_df_dict: Dict[str, List[Any]],
) -> pd.DataFrame:
    if not raw_data:
        logger.warning("preprocess_stageflow: raw_data is empty.")
        return pd.DataFrame()

    df = parse_timeseries(raw_data, column_map)
    if df.empty:
        logger.warning("preprocess_stageflow: Parsed dataframe is empty.")
        return df

    rating_df = pd.DataFrame(rating_df_dict)
    if rating_df.empty:
        logger.warning("preprocess_stageflow: rating_df is empty.")
        return df

    if "stage_ft" in df.columns:
        df["flow_kcfs"] = interpolate_flow(df["stage_ft"], rating_df)
        logger.info("preprocess_stageflow: Interpolated flow from stage.")
    elif "flow_cfs" in df.columns:
        df["stage_ft"] = interpolate_stage_cached(df["flow_cfs"], rating_df_dict)
        logger.info("preprocess_stageflow: Interpolated stage from flow.")

    return df


@st.cache_data(
    hash_funcs={
        list: lambda x: hash(tuple(x)),
        dict: lambda x: hash(
            frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in x.items())
        ),
    }
)
def interpolate_stage_cached(
    flow_list: pd.Series, rating_df_dict: Dict[str, List[Any]]
) -> pd.Series:
    flow = pd.Series(flow_list)
    rating_df = pd.DataFrame(rating_df_dict)
    rating_df = rating_df.sort_values("flow")

    if rating_df.empty:
        logger.warning("interpolate_stage_cached: Rating dataframe is empty.")
        return pd.Series([np.nan] * len(flow), index=flow.index)

    logger.info("interpolate_stage_cached: Interpolating stage from flow.")
    return pd.Series(
        np.interp(
            flow.to_numpy(), rating_df["flow"].to_numpy(), rating_df["stage"].to_numpy()
        ),
        index=flow.index,
    )


def interpolate_flow(stage: pd.Series, rating_df: pd.DataFrame) -> pd.Series:
    if rating_df.empty:
        logger.warning("interpolate_flow: Rating dataframe is empty.")
        return pd.Series([np.nan] * len(stage), index=stage.index)

    logger.info("interpolate_flow: Interpolating flow from stage.")
    return pd.Series(
        np.interp(
            stage.to_numpy(),
            rating_df["stage"].to_numpy(),
            rating_df["flow"].to_numpy(),
        ),
        index=stage.index,
    )


def parse_hefs_member_response(
    response_json: Dict[str, Any], header_id: str
) -> pd.Series:
    events = response_json.get("events", [])
    if not events:
        logger.warning(f"parse_hefs_member_response: No events for header {header_id}.")
        return pd.Series(dtype="float64")

    df = pd.DataFrame(events)
    if "date" not in df.columns or "time" not in df.columns:
        logger.warning(f"parse_hefs_member_response: Missing 'date' or 'time' in header {header_id}.")
        return pd.Series(dtype="float64")

    df["validTime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["validTime"]).set_index("validTime")
    logger.info(f"parse_hefs_member_response: Parsed member {header_id} with {len(df)} points.")
    return df["value"].rename(f"member_{header_id}")


@st.cache_data(ttl=3600)
def compute_and_interpolate_percentiles(
    ensemble_df: pd.DataFrame, rating_df_dict: Dict[str, List[Any]]
) -> Dict[str, pd.Series]:
    if ensemble_df.empty:
        logger.warning("compute_and_interpolate_percentiles: Ensemble dataframe is empty.")
        return {}

    logger.info(f"compute_and_interpolate_percentiles: Starting with shape {ensemble_df.shape}")
    percentiles = compute_percentile_bands(ensemble_df)

    rating_df = pd.DataFrame(rating_df_dict)
    interp = {}
    for key, flow_series in percentiles.items():
        interp[key] = pd.Series(
            np.interp(
                flow_series.to_numpy(),
                rating_df["flow"].to_numpy(),
                rating_df["stage"].to_numpy(),
            ),
            index=flow_series.index,
        )
        logger.info(f"Interpolated {key}: {flow_series.notna().sum()} valid points.")
        logger.info(f"✅ Interpolated percentiles keys: {list(interp.keys())}")

    return interp

def compute_exceedance_probabilities(
    ensemble_df: pd.DataFrame,
    rating_df: pd.DataFrame,
    flood_lines: dict,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp
) -> dict:
    logger.info("🧮 Starting compute_exceedance_probabilities")
    logger.debug(f"➡️  Period: {period_start} to {period_end}")
    logger.debug(f"➡️  Flood lines: {flood_lines.keys()}")

    subset = ensemble_df.loc[period_start:period_end]
    logger.debug(f"📊 Subset shape: {subset.shape}")

    results = {}
    for level, (threshold_value, _) in flood_lines.items():
        logger.debug(f"🔎 Processing level '{level}' with threshold {threshold_value}")
        exceedance_flags = []

        for col in subset.columns:
            flow_series = subset[col]

            try:
                valid_mask = flow_series.notna()
                if valid_mask.sum() == 0:
                    logger.warning(f"⚠️ All values NaN for member {col} at level '{level}'")
                    stage_series = np.array([])
                else:
                    stage_series = np.interp(
                        flow_series[valid_mask].astype(float).to_numpy(),
                        rating_df["flow"].astype(float).to_numpy(),
                        rating_df["stage"].astype(float).to_numpy()
                    )

                flag = np.any(stage_series > threshold_value) if stage_series.size else False
                exceedance_flags.append(flag)
            except Exception as e:
                logger.error(f"❌ Error interpolating for member {col} at level '{level}': {e}")
                exceedance_flags.append(False)

        probability = np.mean(exceedance_flags) * 100
        logger.debug(f"✅ {level} exceedance probability: {probability:.2f}%")
        results[level] = probability

    logger.info(f"🎯 Exceedance probabilities calculated: {results}")
    return results



def compute_percentile_bands(df: pd.DataFrame) -> Dict[str, pd.Series]:
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return {
            k: pd.Series(dtype="float64")
            for k in ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
        }

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Step 1: Compute raw percentiles at native 6-hour time points
    raw_percentiles = {
        "p05": df.quantile(0.05, axis=1),
        "p10": df.quantile(0.10, axis=1),
        "p25": df.quantile(0.25, axis=1),
        "p50": df.quantile(0.50, axis=1),
        "p75": df.quantile(0.75, axis=1),
        "p90": df.quantile(0.90, axis=1),
        "p95": df.quantile(0.95, axis=1),
    }

    # Step 2: Resample to hourly for smooth plotting
    interpolated = {
        k: s.resample("1h").interpolate("time") for k, s in raw_percentiles.items()
    }

    return interpolated

def load_data(
    nws_id: str,
    model_selection: List[str],
    bypass: bool = False,
    get_metadata=None,
    get_stageflow=None,
    get_ratings=None,
    get_nwm_streamflow=None,
) -> tuple[dict, dict, pd.DataFrame, dict]:
    if bypass:
        st.toast("Bypassing cache — fresh data loaded.", icon="🔄")
        for fn in [get_metadata, get_stageflow, get_ratings, get_nwm_streamflow]:
            if fn and hasattr(fn, "clear"):
                fn.clear()

    metadata = get_metadata(nws_id)
    stream_data = get_stageflow(nws_id)
    ratings = get_ratings(nws_id)
    reach_id = metadata.get("reachId")
    nwm_data = get_nwm_streamflow(reach_id)

    rating_df = (
        pd.DataFrame(ratings.get("data", []))
        if isinstance(ratings, dict)
        else pd.DataFrame()
    )
    if not {"stage", "flow"}.issubset(rating_df.columns):
        logger.warning("load_data: Incomplete rating curve (missing 'stage' or 'flow').")
        rating_df = pd.DataFrame()

    logger.info(f"load_data: Metadata received for {nws_id}. Stream data shape: {stream_data.get('data', []) if isinstance(stream_data, dict) else 'unknown'}")
    return metadata, stream_data, rating_df, nwm_data
