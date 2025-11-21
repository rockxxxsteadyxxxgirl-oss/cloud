"""Open-Meteo から複数モデルの雲量を取得して比較する Streamlit アプリ."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, List, Optional

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from textwrap import wrap

# openmeteo_requests が入っていれば利用し、無ければ requests フォールバック
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
except ImportError:  # pragma: no cover - ライブラリ無い場合のフォールバック
    openmeteo_requests = None

API_URL = "https://api.open-meteo.com/v1/forecast"

# モデル一覧
MODEL_INFOS: List[Dict[str, str]] = [
    {
        "display_name": "ECMWF IFS 0.25°",
        "code": "ecmwf_ifs025",
        "desc": "ECMWF の全球モデル。0.25°グリッドで広域の雲量傾向を把握。",
    },
    {
        "display_name": "ECMWF IFS (デフォルト)",
        "code": "ecmwf_ifs",
        "desc": "ECMWF のデフォルト解像度版。0.25°との違いを比較できます。",
    },
    {
        "display_name": "NOAA GFS 0.25°",
        "code": "gfs_seamless",
        "desc": "米国 NOAA の全球予報システム。広域の雲量・風をカバー。",
    },
    {
        "display_name": "ICON Global 0.25°",
        "code": "icon_global",
        "desc": "ドイツ気象庁（DWD）の ICON グローバルモデル。",
    },
    {
        "display_name": "Météo-France Seamless",
        "code": "meteofrance_seamless",
        "desc": "フランス気象局のシームレス予報モデル。欧州域の比較用。",
    },
    {
        "display_name": "UKMO Seamless",
        "code": "ukmo_seamless",
        "desc": "英国気象庁(UKMO)のシームレス予報モデル。英国・欧州の比較用。",
    },
    {
        "display_name": "JMA Seamless",
        "code": "jma_seamless",
        "desc": "気象庁のシームレス予報モデル。日本域細網での比較用。",
    },
    {
        "display_name": "JMA GSM 20km",
        "code": "jma_gsm",
        "desc": "気象庁 GSM。約20km メッシュで数日先の傾向把握に。",
    },
    {
        "display_name": "JMA MSM 5km",
        "code": "jma_msm",
        "desc": "気象庁 MSM。5km メッシュで日本域の短期予測に強み。",
    },
]

FORECAST_MODELS = {m["display_name"]: m["code"] for m in MODEL_INFOS}


_OM_CLIENT: Optional["openmeteo_requests.Client"] = None


def get_openmeteo_client() -> Optional["openmeteo_requests.Client"]:
    """openmeteo_requests が入っていればキャッシュ＋リトライ付きクライアントを返す。"""
    global _OM_CLIENT  # noqa: PLW0603
    if openmeteo_requests is None:
        return None
    if _OM_CLIENT is None:
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        _OM_CLIENT = openmeteo_requests.Client(session=retry_session)
    return _OM_CLIENT


def round_coord(value: float) -> float:
    return round(value, 4)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model: str) -> pd.DataFrame:
    """総雲量を取得（cloudcover に加え層別雲量で補完）。"""
    client = get_openmeteo_client()

    if client is not None:
        params = {
            "latitude": round_coord(lat),
            "longitude": round_coord(lon),
            "hourly": "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high",
            "forecast_days": 7,
            "timezone": "auto",
            "models": model,
        }
        responses = client.weather_api(API_URL, params=params)
        if not responses:
            raise ValueError("Open-Meteo API から雲量データを取得できませんでした。")
        resp = responses[0]
        hourly = resp.Hourly()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_localize(None)
        total = pd.Series(hourly.Variables(0).ValuesAsNumpy())
        low = pd.Series(hourly.Variables(1).ValuesAsNumpy())
        mid = pd.Series(hourly.Variables(2).ValuesAsNumpy())
        high = pd.Series(hourly.Variables(3).ValuesAsNumpy())
        timezone = resp.Timezone()
    else:
        params = {
            "latitude": round_coord(lat),
            "longitude": round_coord(lon),
            "hourly": "cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
            "forecast_days": 7,
            "timezone": "auto",
            "models": model,
        }
        resp = requests.get(API_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        hourly = payload.get("hourly") or {}
        times = hourly.get("time")
        if not times:
            raise ValueError("Open-Meteo API から雲量データを取得できませんでした。")
        timezone = payload.get("timezone", "UTC")
        times = pd.to_datetime(times)
        total = pd.to_numeric(pd.Series(hourly.get("cloudcover")), errors="coerce")
        low = pd.to_numeric(pd.Series(hourly.get("cloudcover_low")), errors="coerce")
        mid = pd.to_numeric(pd.Series(hourly.get("cloudcover_mid")), errors="coerce")
        high = pd.to_numeric(pd.Series(hourly.get("cloudcover_high")), errors="coerce")

    has_layer_data = not (
        low.empty or mid.empty or high.empty or low.isna().all() or mid.isna().all() or high.isna().all()
    )

    candidate = total
    max_val = candidate.max(skipna=True)
    has_fraction = ((candidate % 1) != 0).any()

    # cloudcover が欠落/二値のみなら層別の最大を総雲量に使用
    if (candidate.isna().all() or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
        candidate = pd.concat([low, mid, high], axis=1).max(axis=1)
        max_val = candidate.max(skipna=True)
        has_fraction = ((candidate % 1) != 0).any()

    # 0〜1 の実数を含む場合だけ百分率化
    if max_val is not None and max_val <= 1 and has_fraction:
        candidate = candidate * 100

    df = pd.DataFrame({"time": times, "cloud_cover": candidate, "timezone": timezone})
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


def build_line_chart(chart_df: pd.DataFrame, *, mobile: bool = False) -> alt.Chart:
    axis_values = None
    if not chart_df.empty:
        start = chart_df["time"].min().floor("H")
        end = chart_df["time"].max().ceil("H")
        hourly = pd.date_range(start, end, freq="1H")
        axis_values = [
            {"year": ts.year, "month": ts.month, "date": ts.day, "hours": ts.hour, "minutes": ts.minute}
            for ts in hourly
        ]
    # スマホでも崩れにくいサイズに調整（スクロールはコンテナ側で実施）
    height = int(360 * 1.2)
    width = 900
    # X軸のみズーム・パンを許可（Y軸は 0-100 に固定）
    x_zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "time:T",
                title="日時 (日付＋時刻)",
                scale=alt.Scale(nice=False, clamp=True),
                axis=alt.Axis(
                    format="%m/%d %H:%M",
                    values=axis_values,
                    labelOverlap=False,
                    labelAngle=-80 if mobile else -45,
                    labelFontSize=9 if mobile else 12,
                    titleFontSize=12 if mobile else 14,
                ),
            ),
            y=alt.Y(
                "cloud_cover:Q",
                title="雲量 (%)",
                scale=alt.Scale(domain=[0, 100], clamp=True, nice=False),
            ),
            color=alt.Color(
                "model:N",
                title="モデル",
                legend=alt.Legend(
                    orient="bottom",
                    columns=2 if mobile else 3,
                    labelFontSize=10 if mobile else 12,
                ),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="日時"),
                alt.Tooltip("model:N", title="モデル"),
                alt.Tooltip("cloud_cover:Q", title="雲量 (%)"),
            ],
        )
        .properties(height=height, width=width)
        .configure_mark(strokeWidth=3)
        .configure_view(strokeWidth=0, continuousWidth=width, continuousHeight=height)
        .configure(padding={"top": 0, "left": 0, "right": 0, "bottom": 0})
        .add_selection(x_zoom)
    )


def render_responsive_chart(chart: alt.Chart, *, mobile: bool = False) -> None:
    """スクロール可能な枠内にチャートを収め、詳細データと似た見た目にする。"""
    container_style = (
        "width:100%; max-width:100%; overflow:auto; max-height:720px;"
        "padding:0 8px 8px 8px; margin:0;"
        "border:1px solid #dfe3eb; border-radius:8px; background:#ffffff;"
        "box-shadow: 0 1px 3px rgba(0,0,0,0.04);"
    )
    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)


def update_selected_location(lat: float, lon: float) -> None:
    st.session_state.selected_location = {"lat": lat, "lon": lon}
    st.session_state.lat_input = lat
    st.session_state.lon_input = lon
    st.session_state.pop("latest_data", None)
    try:
        geocoder = Nominatim(user_agent="cloud_cover_app", timeout=5)
        location = geocoder.reverse((lat, lon), language="ja")
        st.session_state.selected_place_name = location.address if location else "地名を取得できませんでした"
    except Exception:  # pragma: no cover - ネットワーク例外
        st.session_state.selected_place_name = "地名を取得できませんでした"


def fetch_ip_location() -> Optional[tuple[float, float]]:
    """IP ベースで概略の現在地を取得（モバイル向けの簡易ボタン用）。"""
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception:  # pragma: no cover - ネットワーク例外
        return None


def init_session_state() -> None:
    defaults = {
        "selected_location": {"lat": 38.1363, "lon": 140.4495},
        "selected_place_name": "未取得",
        "lat_input": 38.1363,
        "lon_input": 140.4495,
        "saved_locations": [],
        "save_label": "",
        "saved_select": "",
        "geo_applied": False,
        "map_center": {"lat": 38.1363, "lon": 140.4495},
        "map_zoom": 10,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_query_params() -> Dict[str, str]:
    """Streamlit バージョン差分を吸収して query params を取得."""
    if hasattr(st, "query_params"):
        return st.query_params
    return st.experimental_get_query_params()


def parse_geo_params(qp: Dict[str, str]) -> Optional[tuple[float, float]]:
    """geo_lat/geo_lon を dict または QueryParams から安全に取り出し float に変換."""
    def first(val):
        if isinstance(val, list):
            return val[0] if val else None
        return val

    lat_raw = first(qp.get("geo_lat"))
    lon_raw = first(qp.get("geo_lon"))
    if lat_raw is None or lon_raw is None:
        return None
    try:
        return float(lat_raw), float(lon_raw)
    except (TypeError, ValueError):
        return None


def clear_query_params() -> None:
    """query params をクリア."""
    if hasattr(st, "query_params"):
        st.query_params.clear()
    else:
        st.experimental_set_query_params()


def render_gps_button() -> None:
    """ブラウザの geolocation API を使うボタン（HTTPS または localhost 必須）。"""
    components.html(
        """
        <div style="margin-top:8px;">
          <button onclick="navigator.geolocation.getCurrentPosition(
              pos => {
                const lat = pos.coords.latitude.toFixed(6);
                const lon = pos.coords.longitude.toFixed(6);
                const url = new URL(window.location.href);
                url.searchParams.set('geo_lat', lat);
                url.searchParams.set('geo_lon', lon);
                window.location.href = url.toString();
              },
              err => { alert('位置情報を取得できません: ' + err.message + '\\n(HTTPSまたはlocalhostが必要です)'); },
              {enableHighAccuracy:true, timeout:10000}
          );"
          style="width:100%;padding:8px;background:#0068c9;color:white;border:none;border-radius:4px;cursor:pointer;">
          ブラウザ位置情報(GPS)を使用
          </button>
        </div>
        """,
        height=60,
    )


def main() -> None:
    st.set_page_config(page_title="雲量比較ダッシュボード", layout="wide")
    st.title("雲量比較ダッシュボード")
    st.caption("Open-Meteo の各モデルで総雲量を比較します。")

    init_session_state()

    # クエリパラメータに geo_lat/geo_lon があれば 1 回だけ反映してクリア
    qp = get_query_params()
    geo = parse_geo_params(qp)
    if not st.session_state.geo_applied and geo:
        lat_q, lon_q = geo
        update_selected_location(lat_q, lon_q)
        st.session_state.geo_applied = True
        st.success(f"ブラウザ位置情報を反映しました: {lat_q:.4f}, {lon_q:.4f}")
        clear_query_params()
        st.rerun()

    with st.sidebar:
        mobile_mode = False  # モバイル表示モードを非表示・無効化

    current_lat = st.session_state.selected_location["lat"]
    current_lon = st.session_state.selected_location["lon"]

    if mobile_mode:
        map_container = st.container()
        input_container = st.container()
    else:
        map_container, input_container = st.columns([5, 1])

    with map_container:
        st.subheader("地図から地点を選択")
        st.write("マップをクリックすると選択中の座標が更新されます。")
        # 前回の中心・ズームを維持
        map_fig = folium.Map(
            location=[st.session_state.map_center["lat"], st.session_state.map_center["lon"]],
            zoom_start=st.session_state.map_zoom,
            control_scale=True,
        )
        folium.LatLngPopup().add_to(map_fig)
        folium.Marker(
            [current_lat, current_lon],
            tooltip=f"選択中: {current_lat:.3f}, {current_lon:.3f}",
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(map_fig)
        map_state = st_folium(
            map_fig,
            width=None if mobile_mode else 1100,
            height=420 if mobile_mode else 540,
            key="forecast_map",
            returned_objects=["last_clicked"],
        )
        if map_state:
            # 表示中の中心のみ保存（ズームは固定）
            center = map_state.get("center")
            if center:
                st.session_state.map_center = {"lat": center.get("lat", current_lat), "lon": center.get("lng", current_lon)}

            if map_state.get("last_clicked"):
                lat_click = map_state["last_clicked"].get("lat")
                lon_click = map_state["last_clicked"].get("lng")
                if lat_click is not None and lon_click is not None:
                    st.info(f"クリックした地点: {lat_click:.4f}, {lon_click:.4f}")
                    update_selected_location(lat_click, lon_click)
                    st.session_state.map_center = {"lat": lat_click, "lon": lon_click}
                    st.session_state.map_zoom = 13
                    st.rerun()
        st.caption(f"現在の座標: {st.session_state.selected_location['lat']:.4f}, "
                   f"{st.session_state.selected_location['lon']:.4f}")
        st.caption("\n".join(wrap(st.session_state.get("selected_place_name", "未取得") or "未取得", 25)))

    with input_container:
        st.subheader("緯度・経度を直接入力")
        lat_value = st.number_input("緯度", min_value=-90.0, max_value=90.0,
                                    value=float(st.session_state.lat_input), step=0.1)
        lon_value = st.number_input("経度", min_value=-180.0, max_value=180.0,
                                    value=float(st.session_state.lon_input), step=0.1)
        if not math.isclose(lat_value, st.session_state.selected_location["lat"], abs_tol=1e-4) or not math.isclose(
            lon_value, st.session_state.selected_location["lon"], abs_tol=1e-4
        ):
            update_selected_location(lat_value, lon_value)
            st.rerun()
        st.metric("緯度", f"{st.session_state.selected_location['lat']:.4f}")
        st.metric("経度", f"{st.session_state.selected_location['lon']:.4f}")
        st.text_area(
            "推定された地名",
            "\n".join(wrap(st.session_state.get("selected_place_name", "未取得") or "未取得", 25)),
            height=80,
        )

        st.subheader("地点登録（最大20件）")
        st.text_input("地点ラベル (省略可)", key="save_label")
        if st.button("現在の地点を登録", use_container_width=True):
            label = st.session_state.save_label.strip() or f"地点 {len(st.session_state.saved_locations) + 1}"
            saved = st.session_state.saved_locations
            replaced = False
            for loc in saved:
                if loc["name"] == label:
                    loc["lat"] = st.session_state.selected_location["lat"]
                    loc["lon"] = st.session_state.selected_location["lon"]
                    replaced = True
                    break
            if not replaced:
                if len(saved) >= 20:
                    saved.pop(0)
                saved.append(
                    {
                        "name": label,
                        "lat": st.session_state.selected_location["lat"],
                        "lon": st.session_state.selected_location["lon"],
                    }
                )
            st.success(f"「{label}」を登録しました。")

        if st.session_state.saved_locations:
            options = [f"{loc['name']} ({loc['lat']:.2f}, {loc['lon']:.2f})" for loc in st.session_state.saved_locations]
            saved_choice = st.selectbox("登録済み地点", options=options, key="saved_select")
            if st.button("選択した地点を呼び出す", use_container_width=True):
                idx = options.index(saved_choice)
                target = st.session_state.saved_locations[idx]
                update_selected_location(target["lat"], target["lon"])
                st.info(f"{target['name']} を読み込みました。")
                st.rerun()
        else:
            st.info("まだ登録された地点はありません。")

    st.markdown("---")

    refresh = st.button("雲量を再取得", type="primary")
    must_refresh = refresh or "latest_data" not in st.session_state

    if must_refresh:
        lat = st.session_state.selected_location["lat"]
        lon = st.session_state.selected_location["lon"]
        with st.spinner("Open-Meteo API から雲量データを取得しています..."):
            base_times = None
            metadata = []
            for info in MODEL_INFOS:
                display_name, model_code = info["display_name"], info["code"]
                try:
                    df = fetch_forecast(lat, lon, model_code)
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

    with st.container():
        st.markdown(
            "<div style='margin:0; padding:0; font-size:1.6rem; font-weight:700;'>48 時間の雲量推移</div>",
            unsafe_allow_html=True,
        )
        chart_df = prepare_chart_data(ts_df)
        if chart_df.empty:
            st.info("各モデルで有効な雲量データを取得できませんでした。")
        else:
            render_responsive_chart(build_line_chart(chart_df, mobile=mobile_mode), mobile=mobile_mode)

    st.subheader("詳細データ")
    st.dataframe(ts_df, use_container_width=True, height=360)

    st.markdown("---")
    st.subheader("モデルの概要")
    for info in MODEL_INFOS:
        st.markdown(f"**{info['display_name']}**: {info['desc']}")


if __name__ == "__main__":
    main()
