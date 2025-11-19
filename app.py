import math
from datetime import timedelta

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from textwrap import wrap

API_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_INFOS = [
    ("ECMWF IFS 0.25°", "ecmwf_ifs025", "ECMWF の全球モデル。0.25°グリッドで広域の雲量傾向をつかみやすい。"),
    ("NOAA GFS 0.25°", "gfs_seamless", "米国 NOAA の全球予報システム。世界中の雲量・風を広く捉えます。"),
    ("ICON Global 0.25°", "icon_global", "ドイツ気象庁（DWD）の ICON グローバルモデル。高解像度の全球予報。"),
    ("JMA GSM 20km", "jma_gsm", "気象庁 GSM。約20km メッシュで数日先の傾向を把握できます。"),
    ("JMA MSM 5km", "jma_msm", "気象庁 MSM。5km メッシュで日本域の短期予測に強みがあります。"),
]

FORECAST_MODELS = {name: code for name, code, _ in MODEL_INFOS}


def round_coord(value: float) -> float:
    return round(value, 4)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model: str) -> pd.DataFrame:
    params = {
        "latitude": round_coord(lat),
        "longitude": round_coord(lon),
        "hourly": "cloudcover",
        "forecast_days": 7,
        "timezone": "auto",
        "models": model,
    }
    resp = requests.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    clouds = hourly.get("cloudcover")
    if not times or not clouds:
        raise ValueError("Open-Meteo API から雲量データを取得できませんでした。")

    df = pd.DataFrame({"time": pd.to_datetime(times), "cloud_cover": clouds})
    timezone = payload.get("timezone", "UTC")
    try:
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(timezone)
        else:
            df["time"] = df["time"].dt.tz_convert(timezone)
    except (TypeError, ValueError):
        df["time"] = df["time"].dt.tz_localize(None)
        timezone = "UTC"
    df["timezone"] = timezone
    return df


def filter_next_hours(df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    if df.empty:
        return df
    tzinfo = df["time"].dt.tz
    now = pd.Timestamp.now(tz=tzinfo)
    cutoff = now + timedelta(hours=hours)
    filtered = df[(df["time"] >= now) & (df["time"] <= cutoff)].copy()
    if tzinfo is not None:
        filtered["time"] = filtered["time"].dt.tz_localize(None)
    return filtered


def prepare_chart_data(timeseries: pd.DataFrame) -> pd.DataFrame:
    chart_df = timeseries.melt("time", var_name="model", value_name="cloud_cover")
    chart_df["cloud_cover"] = pd.to_numeric(chart_df["cloud_cover"], errors="coerce")
    chart_df = chart_df.dropna(subset=["cloud_cover"])
    return chart_df


def build_line_chart(chart_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("time:T", title="日時", scale=alt.Scale(nice=False, clamp=True)),
            y=alt.Y("cloud_cover:Q", title="雲量 (%)", scale=alt.Scale(domain=[0, 100], clamp=True)),
            color=alt.Color("model:N", title="モデル"),
            tooltip=[
                alt.Tooltip("time:T", title="日時"),
                alt.Tooltip("model:N", title="モデル"),
                alt.Tooltip("cloud_cover:Q", title="雲量 (%)"),
            ],
        )
        .configure_mark(strokeWidth=3)
    )


def update_selected_location(lat: float, lon: float) -> None:
    st.session_state.selected_location = {"lat": lat, "lon": lon}
    st.session_state.lat_input = lat
    st.session_state.lon_input = lon
    st.session_state.pop("latest_data", None)
    try:
        geocoder = Nominatim(user_agent="cloud_cover_app", timeout=5)
        location = geocoder.reverse((lat, lon), language="ja")
        st.session_state.selected_place_name = location.address if location else "地名を取得できませんでした"
    except Exception:
        st.session_state.selected_place_name = "地名を取得できませんでした"


def main() -> None:
    st.set_page_config(page_title="雲量比較ダッシュボード", layout="wide")
    st.title("雲量比較ダッシュボード")
    st.caption("Open-Meteo の各モデルから雲量を比較します。")

    if "selected_location" not in st.session_state:
        st.session_state.selected_location = {"lat": 35.0, "lon": 139.0}
    if "selected_place_name" not in st.session_state:
        st.session_state.selected_place_name = "未取得"
    if "lat_input" not in st.session_state:
        st.session_state.lat_input = st.session_state.selected_location["lat"]
    if "lon_input" not in st.session_state:
        st.session_state.lon_input = st.session_state.selected_location["lon"]

    current_lat = st.session_state.selected_location["lat"]
    current_lon = st.session_state.selected_location["lon"]

    col_map, col_inputs = st.columns([2, 1])
    with col_map:
        st.subheader("地図から地点を選択")
        st.write("マップをクリックすると選択中の座標が更新されます。")
        map_fig = folium.Map(location=[current_lat, current_lon], zoom_start=5, control_scale=True)
        folium.LatLngPopup().add_to(map_fig)
        folium.Marker([current_lat, current_lon], tooltip=f"{current_lat:.3f}, {current_lon:.3f}").add_to(map_fig)
        map_state = st_folium(map_fig, height=420, key="forecast_map", returned_objects=["last_clicked"])
        if map_state and map_state.get("last_clicked"):
            lat_click = map_state["last_clicked"].get("lat")
            lon_click = map_state["last_clicked"].get("lng")
            if lat_click is not None and lon_click is not None:
                st.info(f"クリックした地点: {lat_click:.4f}, {lon_click:.4f}")
                update_selected_location(lat_click, lon_click)
                st.rerun()
        st.caption(
            f"現在の座標: {st.session_state.selected_location['lat']:.4f}, "
            f"{st.session_state.selected_location['lon']:.4f}"
        )
        st.caption("\n".join(wrap(st.session_state.get("selected_place_name", "未取得") or "未取得", 25)))

    with col_inputs:
        st.subheader("緯度・経度を直接入力")
        lat_value = st.number_input(
            "緯度", min_value=-90.0, max_value=90.0, value=float(st.session_state.lat_input), step=0.1
        )
        lon_value = st.number_input(
            "経度", min_value=-180.0, max_value=180.0, value=float(st.session_state.lon_input), step=0.1
        )
        if not math.isclose(lat_value, st.session_state.selected_location["lat"], abs_tol=1e-4) or not math.isclose(
            lon_value, st.session_state.selected_location["lon"], abs_tol=1e-4
        ):
            update_selected_location(lat_value, lon_value)
            st.rerun()
        st.metric("緯度", f"{st.session_state.selected_location['lat']:.4f}")
        st.metric("経度", f"{st.session_state.selected_location['lon']:.4f}")
        st.text_area(
            "地名",
            "\n".join(wrap(st.session_state.get("selected_place_name", "未取得") or "未取得", 25)),
            height=80,
        )

    st.markdown("---")

    refresh = st.button("雲量比較を更新", type="primary")
    must_refresh = refresh or "latest_data" not in st.session_state

    if must_refresh:
        lat = st.session_state.selected_location["lat"]
        lon = st.session_state.selected_location["lon"]
        with st.spinner("Open-Meteo API から雲量データを取得しています..."):
            base_times = None
            metadata = []
            for display_name, model_id in FORECAST_MODELS.items():
                try:
                    df = fetch_forecast(lat, lon, model_id)
                    df = filter_next_hours(df)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{display_name} の取得に失敗しました: {exc}")
                    continue

                renamed = df.rename(columns={"cloud_cover": display_name})
                if base_times is None:
                    base_times = renamed[["time", display_name]]
                else:
                    base_times = base_times.merge(renamed[["time", display_name]], on="time", how="outer")

                tz_label = df["timezone"].iloc[0] if "timezone" in df.columns and not df.empty else "不明"
                metadata.append({"モデル": display_name, "データ件数": len(df), "タイムゾーン": tz_label})

            if base_times is None or base_times.empty:
                st.warning("有効なデータが取得できませんでした。地点を変更するか時間をおいて再試行してください。")
                return

            base_times = base_times.sort_values("time").reset_index(drop=True)
            st.session_state.latest_data = {"timeseries": base_times, "metadata": metadata}

    if "latest_data" not in st.session_state:
        st.info("まず地点を選択し、データを取得してください。")
        return

    ts_df = st.session_state.latest_data["timeseries"]
    metadata = st.session_state.latest_data["metadata"]

    st.subheader("モデル別データ状況")
    st.table(pd.DataFrame(metadata))

    st.subheader("48 時間の雲量推移")
    chart_df = prepare_chart_data(ts_df)
    if chart_df.empty:
        st.info("各モデルで有効な雲量データを取得できませんでした。")
    else:
        st.altair_chart(build_line_chart(chart_df), use_container_width=True)

    st.subheader("詳細データ")
    st.dataframe(ts_df, use_container_width=True, height=360)

    st.markdown("---")
    st.subheader("モデルの概要")
    for name, _, desc in MODEL_INFOS:
        st.markdown(f"**{name}**: {desc}")


if __name__ == "__main__":
    main()
