#!/usr/bin/env python
"""
Open-Meteo の複数モデルで雲量を比較するシンプルな Streamlit アプリ。

実行例:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

API_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_INFOS: List[Dict[str, str]] = [
    {"display_name": "ECMWF IFS 0.25°", "code": "ecmwf_ifs025"},
    {"display_name": "ECMWF IFS", "code": "ecmwf_ifs"},
    {"display_name": "NOAA GFS 0.25°", "code": "gfs_seamless"},
    {"display_name": "ICON Global 0.25°", "code": "icon_global"},
    {"display_name": "Météo-France Seamless", "code": "meteofrance_seamless"},
    {"display_name": "UKMO Seamless", "code": "ukmo_seamless"},
    {"display_name": "JMA Seamless", "code": "jma_seamless"},
    {"display_name": "JMA GSM 20km", "code": "jma_gsm"},
    {"display_name": "JMA MSM 5km", "code": "jma_msm"},
]


def round_coord(value: float) -> float:
    return round(value, 4)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model: str) -> Tuple[pd.DataFrame, str]:
    """総雲量を取得。cloudcover が 0/1 だけなら層別雲量で補完し百分率化。"""
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
        raise ValueError("Open-Meteo から雲量データを取得できませんでした。")
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

    if (candidate.isna().all() or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
        candidate = pd.concat([low, mid, high], axis=1).max(axis=1)
        max_val = candidate.max(skipna=True)
        has_fraction = ((candidate % 1) != 0).any()

    if max_val is not None and max_val <= 1 and has_fraction:
        candidate = candidate * 100

    df = pd.DataFrame({"time": times, "cloud_cover": candidate})
    return df, timezone


def filter_next_hours(df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz=df["time"].dt.tz)
    cutoff = now + timedelta(hours=hours)
    filtered = df[(df["time"] >= now) & (df["time"] <= cutoff)].copy()
    filtered["time"] = filtered["time"].dt.tz_localize(None)
    return filtered


def prepare_chart_data(timeseries: pd.DataFrame) -> pd.DataFrame:
    chart_df = timeseries.melt("time", var_name="model", value_name="cloud_cover")
    chart_df["cloud_cover"] = pd.to_numeric(chart_df["cloud_cover"], errors="coerce")
    return chart_df.dropna(subset=["cloud_cover"])


def prepare_layer_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    chart_df = df.melt("time", var_name="layer", value_name="cloud_cover")
    chart_df["cloud_cover"] = pd.to_numeric(chart_df["cloud_cover"], errors="coerce")
    return chart_df.dropna(subset=["cloud_cover"])


def build_line_chart(chart_df: pd.DataFrame) -> alt.Chart:
    axis_values = None
    if not chart_df.empty:
        start = chart_df["time"].min().floor("H")
        end = chart_df["time"].max().ceil("H")
        axis_values = pd.date_range(start, end, freq="1H").to_pydatetime().tolist()

    return (
        alt.Chart(chart_df)
        .mark_line(point=False, strokeWidth=2.4)
        .encode(
            x=alt.X(
                "time:T",
                title="日時",
                axis=alt.Axis(
                    format="%m/%d %H:%M",
                    labelAngle=-45,
                    labelFontSize=11,
                    titleFontSize=12,
                    values=axis_values,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(
                "cloud_cover:Q",
                title="雲量 (%)",
                scale=alt.Scale(domain=[0, 100], clamp=True),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12, grid=True),
            ),
            color=alt.Color(
                "model:N",
                title="モデル",
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=len(MODEL_INFOS),  # 1行に並べる
                    labelFontSize=11,
                    titleFontSize=12,
                ),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="日時"),
                alt.Tooltip("model:N", title="モデル"),
                alt.Tooltip("cloud_cover:Q", title="雲量 (%)"),
            ],
        )
        .properties(height=640, width=2400)
        .configure_view(strokeWidth=0)
    )


def build_layer_chart(chart_df: pd.DataFrame, title_suffix: str) -> alt.Chart:
    axis_values = None
    if not chart_df.empty:
        start = chart_df["time"].min().floor("H")
        end = chart_df["time"].max().ceil("H")
        axis_values = pd.date_range(start, end, freq="1H").to_pydatetime().tolist()

    return (
        alt.Chart(chart_df)
        .mark_line(point=False, strokeWidth=2.4)
        .encode(
            x=alt.X(
                "time:T",
                title=f"日時 ({title_suffix})",
                axis=alt.Axis(
                    format="%m/%d %H:%M",
                    labelAngle=-45,
                    labelFontSize=11,
                    titleFontSize=12,
                    values=axis_values,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(
                "cloud_cover:Q",
                title="雲量 (%)",
                scale=alt.Scale(domain=[0, 100], clamp=True),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12, grid=True),
            ),
            color=alt.Color(
                "layer:N",
                title="雲の層",
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=4,
                    labelFontSize=11,
                    titleFontSize=12,
                ),
                scale=alt.Scale(
                    domain=["総雲量", "下層雲", "中層雲", "上層雲"],
                    range=["#1f78b4", "#33a02c", "#fb9a99", "#6a3d9a"],
                ),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="日時"),
                alt.Tooltip("layer:N", title="層"),
                alt.Tooltip("cloud_cover:Q", title="雲量 (%)"),
            ],
        )
        .properties(height=480, width=2000)
        .configure_view(strokeWidth=0)
    )


def geocode_place(query: str) -> Optional[Tuple[float, float, Optional[str]]]:
    if not query.strip():
        return None
    try:
        geocoder = Nominatim(user_agent="cloud_cover_simple_app", timeout=5)
        result = geocoder.geocode(query)
        if result is None:
            return None
        return float(result.latitude), float(result.longitude), result.address
    except Exception:
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        geocoder = Nominatim(user_agent="cloud_cover_simple_app", timeout=5)
        result = geocoder.reverse((lat, lon), language="ja")
        if result is None:
            return None
        return result.address
    except Exception:
        return None


def load_models(lat: float, lon: float) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    frames: List[pd.DataFrame] = []
    metadata: List[Dict[str, str]] = []
    for info in MODEL_INFOS:
        display_name, model_code = info["display_name"], info["code"]
        total_label = f"{display_name} (Total cloud)"
        df, tz = fetch_forecast(lat, lon, model_code)
        df = filter_next_hours(df)
        renamed = df.rename(columns={"cloud_cover": total_label})
        frames.append(renamed[["time", total_label]])
        metadata.append({"モデル": total_label, "データ件数": len(df), "タイムゾーン": tz})

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged, metadata


def normalize_cloud(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    max_val = series.max(skipna=True)
    has_fraction = ((series % 1) != 0).any()
    if max_val is not None and max_val <= 1 and has_fraction:
        series = series * 100
    return series


def fetch_layered_forecast(lat: float, lon: float, model: str) -> pd.DataFrame:
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
        raise ValueError("Open-Meteo から雲量データを取得できませんでした。")
    times = pd.to_datetime(times)

    total = normalize_cloud(pd.Series(hourly.get("cloudcover")))
    low = normalize_cloud(pd.Series(hourly.get("cloudcover_low")))
    mid = normalize_cloud(pd.Series(hourly.get("cloudcover_mid")))
    high = normalize_cloud(pd.Series(hourly.get("cloudcover_high")))

    has_layer_data = not (low.empty or mid.empty or high.empty or low.isna().all() or mid.isna().all() or high.isna().all())
    max_val = total.max(skipna=True)
    has_fraction = ((total % 1) != 0).any()
    if (total.isna().all() or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
        total = pd.concat([low, mid, high], axis=1).max(axis=1)
        total = normalize_cloud(total)

    df = pd.DataFrame(
        {
            "time": times,
            "総雲量": total,
            "下層雲": low,
            "中層雲": mid,
            "上層雲": high,
        }
    )
    return filter_next_hours(df)


CACHE_FILE = Path(".saved_locations.json")


def load_saved_locations_from_disk() -> List[Dict[str, object]]:
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_saved_locations_to_disk(locations: List[Dict[str, object]]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(locations, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def init_state() -> None:
    defaults = {
        "lat": 35.6812,
        "lon": 139.7671,
        "data": None,
        "metadata": None,
        "last_click": None,
        "place_name": "未取得",
        "trigger_fetch": False,
        "saved_locations": [],
        "save_label": "",
        "selected_saved": "",
        "layer_data": None,
        "layer_model": "",
        "model_diagnostics": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.get("saved_locations"):
        disk_locations = load_saved_locations_from_disk()
        if disk_locations:
            st.session_state.saved_locations = disk_locations


def render_saved_locations(saved: List[Dict[str, object]]) -> None:
    if saved:
        options = [f"{loc['name']} ({loc['lat']:.3f}, {loc['lon']:.3f})" for loc in saved]
        choice = st.selectbox("登録済み地点", options=options, key="selected_saved")

        if st.button("選択した地点を呼び出す"):
            idx = options.index(choice)
            target = saved[idx]
            st.session_state.lat = target["lat"]
            st.session_state.lon = target["lon"]
            st.session_state.place_name = target.get("place_name") or target["name"]
            st.session_state.last_click = (target["lat"], target["lon"])
            st.session_state.trigger_fetch = True
            st.success(f"{target['name']} を読み込みました。")

        if st.button("選択した地点を削除する", type="secondary"):
            idx = options.index(choice)
            target = saved[idx]
            st.session_state.saved_locations = [loc for i, loc in enumerate(saved) if i != idx]
            save_saved_locations_to_disk(st.session_state.saved_locations)
            st.success(f"{target['name']} を削除しました。")
            st.rerun()
    else:
        st.info("登録済みの地点はまだありません。")

    st.markdown("**登録地点の一覧 / エクスポート**")
    saved_df = pd.DataFrame(saved)[["name", "lat", "lon", "place_name"]] if saved else pd.DataFrame(
        columns=["name", "lat", "lon", "place_name"]
    )
    st.dataframe(saved_df.rename(columns={"name": "ラベル", "lat": "緯度", "lon": "経度", "place_name": "地名"}), height=240)
    json_str = json.dumps(saved if saved else [], ensure_ascii=False, indent=2)
    st.download_button(
        "登録地点をJSON出力",
        data=json_str.encode("utf-8"),
        file_name="saved_locations.json",
        mime="application/json",
        disabled=not bool(saved),
    )

    st.markdown("**JSON インポート**")
    uploaded = st.file_uploader("登録地点のJSONを読み込み", type=["json"])
    if uploaded and st.button("JSONをインポート"):
        try:
            uploaded.seek(0)
            imported = json.load(uploaded)
            if not isinstance(imported, list):
                raise ValueError("JSONは地点のリスト形式にしてください。")

            cleaned: List[Dict[str, object]] = []
            for item in imported:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("label") or "").strip()
                lat = item.get("lat")
                lon = item.get("lon")
                if not name or lat is None or lon is None:
                    continue
                place_name = str(item.get("place_name") or item.get("name") or name)
                cleaned.append({"name": name, "lat": float(lat), "lon": float(lon), "place_name": place_name})

            if not cleaned:
                raise ValueError("有効な地点データが見つかりませんでした。")

            merged = {loc["name"]: loc for loc in st.session_state.saved_locations}
            for loc in cleaned:
                merged[loc["name"]] = loc
            merged_list = list(merged.values())
            if len(merged_list) > 20:
                merged_list = merged_list[-20:]
            st.session_state.saved_locations = merged_list
            save_saved_locations_to_disk(st.session_state.saved_locations)
            st.success(f"JSONから {len(cleaned)} 件取り込みました。")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"インポートに失敗しました: {exc}")


def render_sidebar() -> None:
    st.subheader("地点の指定")
    query = st.text_input("地名/住所で検索（任意）", key="query_input", placeholder="例: 東京駅")
    if st.button("地名から検索"):
        result = geocode_place(query)
        if result:
            lat, lon, name = result
            st.session_state.lat, st.session_state.lon = lat, lon
            st.session_state.last_click = (lat, lon)
            st.session_state.place_name = name or query
            st.session_state.trigger_fetch = True
            st.success(f"座標を更新: {lat:.4f}, {lon:.4f}")
        else:
            st.error("地名を特定できませんでした。")

    st.session_state.lat = st.number_input(
        "緯度", min_value=-90.0, max_value=90.0, value=float(st.session_state.lat), step=0.1
    )
    st.session_state.lon = st.number_input(
        "経度", min_value=-180.0, max_value=180.0, value=float(st.session_state.lon), step=0.1
    )
    if st.button("この地点の雲量を取得", type="primary"):
        st.session_state.trigger_fetch = True

    st.markdown("---")
    st.subheader("地点の登録・呼び出し")
    st.text_input("登録名", key="save_label", placeholder="例: 自宅/職場/観測点")
    if st.button("現在の地点を登録"):
        label = st.session_state.save_label.strip() or f"地点 {len(st.session_state.saved_locations) + 1}"
        saved = list(st.session_state.saved_locations)
        replaced = False
        for loc in saved:
            if loc["name"] == label:
                loc.update(
                    {"lat": st.session_state.lat, "lon": st.session_state.lon, "place_name": st.session_state.place_name}
                )
                replaced = True
                break
        if not replaced:
            if len(saved) >= 20:
                saved.pop(0)
            saved.append(
                {
                    "name": label,
                    "lat": st.session_state.lat,
                    "lon": st.session_state.lon,
                    "place_name": st.session_state.place_name,
                }
            )
        st.session_state.saved_locations = saved
        save_saved_locations_to_disk(st.session_state.saved_locations)
        st.success(f"「{label}」を保存しました。")

    saved = st.session_state.saved_locations
    render_saved_locations(saved)


def main() -> None:
    st.set_page_config(page_title="雲量比較", layout="wide")
    st.title("雲量比較")
    st.caption("Open-Meteo の複数モデルで直近 48 時間の雲量を比較します。")

    init_state()

    with st.sidebar:
        render_sidebar()

    tab_compare, tab_manage = st.tabs(["比較モード", "モデルの雲量グラフ"])

    with tab_compare:
        st.subheader("地図で地点を選択")
        selected_lat = st.session_state.lat
        selected_lon = st.session_state.lon
        map_fig = folium.Map(location=[selected_lat, selected_lon], zoom_start=13, control_scale=True)
        folium.Marker(
            [selected_lat, selected_lon],
            tooltip="選択中の地点",
            popup=st.session_state.place_name,
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(map_fig)
        map_state = st_folium(map_fig, width=900, height=420, key="map", returned_objects=["last_clicked"])

        if map_state and map_state.get("last_clicked"):
            lat_click = map_state["last_clicked"].get("lat")
            lon_click = map_state["last_clicked"].get("lng")
            if lat_click is not None and lon_click is not None:
                new_click = (float(lat_click), float(lon_click))
                if st.session_state.last_click != new_click:
                    st.session_state.last_click = new_click
                    st.session_state.lat, st.session_state.lon = new_click
                    st.session_state.place_name = reverse_geocode(*new_click) or "未取得"
                    st.session_state.trigger_fetch = True
                    st.rerun()
                else:
                    st.info(f"地図で選択: {lat_click:.4f}, {lon_click:.4f}")

        st.caption(f"現在の座標: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
        st.caption(f"推定された地名: {st.session_state.place_name}")

        if st.session_state.trigger_fetch:
            st.session_state.trigger_fetch = False
            try:
                with st.spinner("Open-Meteo からデータ取得中..."):
                    ts_df, metadata = load_models(st.session_state.lat, st.session_state.lon)
                st.session_state.data = ts_df
                st.session_state.metadata = metadata
                st.success("データを更新しました。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"取得に失敗しました: {exc}")

        if st.session_state.get("data") is None:
            st.info("地図をクリックするか、サイドバーで地点を指定して雲量を取得してください。")
            return

        ts_df = st.session_state.data
        metadata = st.session_state.metadata or []

        st.subheader("48 時間の雲量推移")
        chart_df = prepare_chart_data(ts_df)
        if chart_df.empty:
            st.info("有効な雲量データがありません。")
        else:
            st.altair_chart(build_line_chart(chart_df), use_container_width=False)

        st.subheader("詳細データ")
        st.dataframe(ts_df, use_container_width=True, height=360)

        st.subheader("モデル別データ状況")
        st.table(pd.DataFrame(metadata))

    with tab_manage:
        st.subheader("モデルの雲量グラフ（登録地点から選択）")
        saved = st.session_state.saved_locations
        if not saved:
            st.info("登録済みの地点がありません。サイドバーまたは比較モードで地点を登録してください。")
        else:
            loc_options = [f"{loc['name']} ({loc['lat']:.3f}, {loc['lon']:.3f})" for loc in saved]
            choice = st.selectbox("登録地点を選択", options=loc_options, key="manage_select")
            model_options = [m["display_name"] for m in MODEL_INFOS]
            model_choice = st.selectbox("モデルを選択", options=model_options, key="manage_model_select")

            if st.button("選択した地点とモデルの雲量を表示", key="manage_fetch"):
                idx = loc_options.index(choice)
                target = saved[idx]
                model_code = next(m["code"] for m in MODEL_INFOS if m["display_name"] == model_choice)
                try:
                    with st.spinner("Open-Meteo からデータ取得中..."):
                        layer_df = fetch_layered_forecast(target["lat"], target["lon"], model_code)
                        layer_df = filter_next_hours(layer_df)
                    st.session_state.layer_data = layer_df
                    st.session_state.layer_model = model_choice
                    st.session_state.lat = target["lat"]
                    st.session_state.lon = target["lon"]
                    st.session_state.place_name = target.get("place_name") or target["name"]
                    st.session_state.last_click = (target["lat"], target["lon"])
                    st.success(f"{target['name']} / {model_choice} のデータを更新しました。")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"取得に失敗しました: {exc}")

            if st.button("この地点で全モデル検証＆JSON出力", key="manage_diag"):
                idx = loc_options.index(choice)
                target = saved[idx]
                diagnostics: List[Dict[str, object]] = []
                for info in MODEL_INFOS:
                    model_code = info["code"]
                    label = info["display_name"]
                    entry: Dict[str, object] = {"model": label, "code": model_code}
                    try:
                        df = fetch_layered_forecast(target["lat"], target["lon"], model_code)
                        df = filter_next_hours(df)
                        entry["status"] = "success"
                        entry["rows"] = len(df)
                        entry["time_start"] = df["time"].min().isoformat() if not df.empty else None
                        entry["time_end"] = df["time"].max().isoformat() if not df.empty else None
                        if not df.empty:
                            export_df = df.copy()
                            export_df["time"] = export_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
                            entry["data"] = export_df.fillna("").to_dict(orient="records")
                    except Exception as exc:  # noqa: BLE001
                        entry["status"] = "error"
                        entry["error"] = str(exc)
                    diagnostics.append(entry)
                st.session_state.model_diagnostics = diagnostics
                st.success("全モデルの検証が完了しました。下のJSONをダウンロードできます。")
                diag_json = json.dumps(diagnostics, ensure_ascii=False, indent=2)
                st.download_button(
                    "検証結果をJSONダウンロード",
                    data=diag_json.encode("utf-8"),
                    file_name="model_diagnostics.json",
                    mime="application/json",
                    key="diag_download",
                )

            layer_df = st.session_state.get("layer_data")
            if layer_df is not None and not layer_df.empty:
                st.caption(
                    f"現在の座標: {st.session_state.lat:.4f}, {st.session_state.lon:.4f} / "
                    f"推定地名: {st.session_state.place_name}"
                )
                chart_df = prepare_layer_chart_data(layer_df)
                st.subheader(f"{st.session_state.layer_model} の層別雲量（48 時間）")
                st.altair_chart(build_layer_chart(chart_df, st.session_state.layer_model), use_container_width=False)
                st.subheader("詳細データ")
                st.dataframe(layer_df, use_container_width=True, height=360)
            else:
                st.info("地点とモデルを選択して「選択した地点とモデルの雲量を表示」を押してください。")


if __name__ == "__main__":
    main()
