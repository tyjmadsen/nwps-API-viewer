import requests
import pandas as pd
import streamlit as st
from logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any


session = requests.Session()
session.headers.update(
    {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "User-Agent": "StreamflowApp/1.0",
    }
)

adapter = requests.adapters.HTTPAdapter(
    max_retries=3, pool_connections=100, pool_maxsize=100
)
session.mount("https://", adapter)

BASE_URL = "https://api.water.noaa.gov/nwps/v1/gauges"
REACH_BASE_URL = "https://testing-api.water.noaa.gov/nwps/v1/reaches"
HEFS_URL = "https://api.water.noaa.gov/hefs/v1/headers/"


def chunked(iterable: List[Any], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


@st.cache_data(ttl=900)
def list_all_gauges() -> List[Dict[str, str]]:
    try:
        r = session.get(BASE_URL, timeout=30)
        r.raise_for_status()
        gauges = r.json().get("gauges", [])
        return sorted(
            [
                {"lid": g["lid"], "name": g["name"]}
                for g in gauges
                if g.get("lid") and g.get("name")
            ],
            key=lambda x: x["lid"],
        )
    except Exception as e:
        st.warning(f"❌ Failed to fetch gauge list.\n{e}")
        return []


@st.cache_data(ttl=600)
def get_metadata(nws_id: str) -> Dict[str, Any]:
    r = session.get(f"{BASE_URL}/{nws_id}")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def get_stageflow(nws_id: str) -> Dict[str, Any]:
    r = session.get(f"{BASE_URL}/{nws_id}/stageflow")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def get_ratings(nws_id: str) -> Dict[str, Any]:
    r = session.get(f"{BASE_URL}/{nws_id}/ratings?onlyTenths=false")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def get_nwm_streamflow(reach_id: str) -> Dict[str, Any]:
    try:
        r = session.get(f"{REACH_BASE_URL}/{reach_id}/streamflow", timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        st.warning(f"⚠️ Could not fetch NWM streamflow for reach {reach_id}.")
        return {}


@st.cache_data(ttl=300)
def get_latest_hefs_headers(nws_id: str) -> List[Dict[str, Any]]:
    url = f"{HEFS_URL}?location_id={nws_id}&parameter_id=QINE&ordering=-forecast_date_date,-forecast_date_time&limit=100"
    r = session.get(url)
    r.raise_for_status()
    headers = r.json().get("results", [])

    if not headers:
        return []

    df = pd.DataFrame(headers).dropna(
        subset=["forecast_date_date", "forecast_date_time"]
    )
    df["runtime"] = pd.to_datetime(
        df["forecast_date_date"] + " " + df["forecast_date_time"], errors="coerce"
    )

    latest_runtime = df["runtime"].max()
    if pd.isna(latest_runtime):
        return []

    matched = df[abs(df["runtime"] - latest_runtime) < pd.Timedelta(minutes=1)]
    return matched.to_dict(orient="records")


@st.cache_data(ttl=300)
def fetch_single_hefs_member(header_id: str) -> Optional[pd.Series]:
    logger.info(f"fetch_single_hefs_member: Trying to fetch header_id {header_id}")
    url = f"https://api.water.noaa.gov/hefs/v1/ensembles/{header_id}/"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        events = data.get("events", [])
        if not events:
            return None

        df = pd.DataFrame(events)
        if {"date", "time", "value"}.issubset(df.columns):
            df["validTime"] = pd.to_datetime(
                df["date"] + " " + df["time"], errors="coerce"
            )
            df = df.dropna(subset=["validTime"]).set_index("validTime")
            logger.info(
                f"fetch_single_hefs_member: Success for header_id {header_id}, {len(df)} points"
            )
            return df["value"].rename(f"member_{header_id}")
    except Exception as e:
        st.warning(f"Error fetching member {header_id}: {e}")
        logger.warning(
            f"fetch_single_hefs_member: No usable data for header_id {header_id}"
        )
    return None


@st.cache_data(ttl=600)
def fetch_hefs_timeseries(
    headers: List[Dict[str, Any]], chunk_size: int = 25
) -> pd.DataFrame:

    def safe_fetch(header_id: str) -> pd.Series:
        try:
            ts = fetch_single_hefs_member(header_id)
            if ts is not None and not ts.empty:
                return ts
            return pd.Series(dtype="float64")

        except Exception:
            logger.exception(f"safe_fetch: Exception fetching member {header_id}")
            return pd.Series(dtype="float64")

    header_ids = [h.get("id") for h in headers if h.get("id")]
    total = len(header_ids)
    all_flows = []

    if total == 0:
        st.warning("⚠️ No HEFS headers to fetch.")
        logger.warning("fetch_hefs_timeseries: No header IDs found in input.")
        return pd.DataFrame()

    progress = st.progress(0, text="📡 Fetching HEFS ensemble members...")
    completed = 0

    for chunk in chunked(header_ids, chunk_size):
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            futures = {executor.submit(safe_fetch, hid): hid for hid in chunk}
            for future in as_completed(futures):
                ts = future.result()
                if ts is not None and not ts.empty:
                    all_flows.append(ts)
                completed += 1
                progress.progress(
                    completed / total, text=f"📡 Fetching... {completed}/{total}"
                )

    progress.empty()

    if not all_flows:
        st.warning("⚠️ No valid HEFS ensemble member timeseries retrieved.")
        logger.warning("fetch_hefs_timeseries: No HEFS members returned any data")
        return pd.DataFrame()

    try:
        df_out = pd.concat(all_flows, axis=1)
        return df_out
    except Exception as e:
        logger.exception(
            f"fetch_hefs_timeseries: Failed to concatenate HEFS members — {e}"
        )
        return pd.DataFrame()

    df_out = pd.concat(all_flows, axis=1)
    return df_out


@st.cache_data(ttl=600)
def get_reach_centroid(reach_id: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        r = session.get(f"{REACH_BASE_URL}/{reach_id}")
        r.raise_for_status()
        data = r.json()
        return data.get("latitude"), data.get("longitude")
    except Exception:
        return None, None


@st.cache_data(ttl=600)
def get_usgs_basin_geojson(usgs_id: str) -> Optional[Dict[str, Any]]:
    try:
        base_url = "https://api.water.usgs.gov/nldi/linked-data"
        url = f"{base_url}/nwissite/USGS-{usgs_id}/basin?f=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"⚠️ Could not load basin from NLDI for USGS-{usgs_id}: {e}")
        return None
