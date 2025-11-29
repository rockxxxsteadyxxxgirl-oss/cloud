#!/usr/bin/env python
from __future__ import annotations

import json
import random
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

# GPS 取得用（任意）
try:
    from streamlit_geolocation import geolocation
except ImportError:
    geolocation = None

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

# プリセット
DEFAULT_PRESETS = [
    {
        "name": "星空観測メイン",
        "models": ["JMA MSM 5km", "JMA GSM 20km", "ECMWF IFS 0.25°", "ECMWF IFS"],
    },
    {
        "name": "高速チェック（軽量）",
        "models": ["JMA MSM 5km", "ECMWF IFS 0.25°"],
    },
    {
        "name": "全球モデル比較",
        "models": [
            "ECMWF IFS 0.25°",
            "NOAA GFS 0.25°",
            "ICON Global 0.25°",
            "Météo-France Seamless",
            "UKMO Seamless",
            "JMA Seamless",
        ],
    },
]

CACHE_FILE = Path(".saved_locations.json")
CONFIG_FILE = Path(".cloud_viewer_config.json")

DARK_BG_COLOR = "#020617"   # ほぼ黒の紺
DARK_TEXT_COLOR = "#e5e7eb"  # 明るいグレー
LIGHT_BG_COLOR = "linear-gradient(180deg, #6bb9ff 0%, #9fd7ff 45%, #e8f7ff 100%)"
LIGHT_TEXT_COLOR = "#1f2937"


def round_coord(value: float) -> float:
    """API へ投げる座標の丸め精度（5 桁）"""
    return round(value, 5)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model: str) -> Tuple[pd.DataFrame, str]:
    """Open-Meteo から総雲量（＋層別雲量）を取得し、0〜100% に正規化した DataFrame を返す。"""
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
        low.empty or mid.empty or high.empty
        or low.isna().all() or mid.isna().all() or high.isna().all()
    )

    candidate = total
    max_val = candidate.max(skipna=True)
    has_fraction = ((candidate % 1) != 0).any()

    # 総雲量が 0/1 しかない or 欠損 → 層別の最大値で代用
    if (candidate.isna().all()
        or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
        candidate = pd.concat([low, mid, high], axis=1).max(axis=1)
        max_val = candidate.max(skipna=True)
        has_fraction = ((candidate % 1) != 0).any()

    # 0〜1 の小数なら % に変換
    if max_val is not None and max_val <= 1 and has_fraction:
        candidate = candidate * 100

    df = pd.DataFrame({"time": times, "cloud_cover": candidate})
    return df, timezone


def filter_next_hours(df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    """直近 hours 時間だけに絞る。"""
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

    chart = (
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
                    columns=len(MODEL_INFOS),
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
        .properties(height=420)
        .configure_view(strokeWidth=0)
    )
    return chart


def build_layer_chart(chart_df: pd.DataFrame, title_suffix: str) -> alt.Chart:
    axis_values = None
    if not chart_df.empty:
        start = chart_df["time"].min().floor("H")
        end = chart_df["time"].max().ceil("H")
        axis_values = pd.date_range(start, end, freq="1H").to_pydatetime().tolist()

    chart = (
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
        .properties(height=420)
        .configure_view(strokeWidth=0)
    )
    return chart


def parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    """「38.12, 140.44」などの文字列を緯度・経度にパース。全角カンマ/スペースにも対応。"""
    s = text.strip()
    if not s:
        return None
    s = s.replace("，", ",").replace("、", ",")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def geocode_place(query: str) -> Optional[Tuple[float, float, Optional[str]]]:
    """
    地名/住所 または "緯度, 経度" を受け取り、(lat, lon, 名前) を返す。
    """
    if not query.strip():
        return None

    # まずは「緯度, 経度」として解釈を試みる
    coords = parse_latlon(query)
    if coords is not None:
        lat, lon = coords
        try:
            geocoder = Nominatim(user_agent="cloud_cover_simple_app", timeout=5)
            result = geocoder.reverse((lat, lon), language="ja")
            name = result.address if result is not None else None
        except Exception:
            name = None
        if not name:
            name = f"{lat:.4f}, {lon:.4f}"
        return lat, lon, name

    # 通常の地名検索
    try:
        geocoder = Nominatim(user_agent="cloud_cover_simple_app", timeout=5)
        result = geocoder.geocode(query)
        if result is None:
            return None
        return float(result.latitude), float(result.longitude), result.address
    except Exception:
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """逆ジオコーディングで地名を取得（失敗したら None）。"""
    try:
        geocoder = Nominatim(user_agent="cloud_cover_simple_app", timeout=5)
        result = geocoder.reverse((lat, lon), language="ja")
        if result is None:
            return None
        return result.address
    except Exception:
        return None


def load_models(lat: float, lon: float) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """全モデルの総雲量を取得・マージして 1 つの DataFrame にまとめる。"""
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
    """0〜1 の小数で来た雲量を 0〜100% に直すヘルパー。"""
    series = pd.to_numeric(series, errors="coerce")
    max_val = series.max(skipna=True)
    has_fraction = ((series % 1) != 0).any()
    if max_val is not None and max_val <= 1 and has_fraction:
        series = series * 100
    return series


def fetch_layered_forecast(lat: float, lon: float, model: str) -> pd.DataFrame:
    """層別雲量（総雲量＋下層・中層・上層）の 48h 分を取得。"""
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

    has_layer_data = not (
        low.empty or mid.empty or high.empty
        or low.isna().all() or mid.isna().all() or high.isna().all()
    )
    max_val = total.max(skipna=True)
    has_fraction = ((total % 1) != 0).any()
    if (total.isna().all()
        or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
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


def load_saved_locations_from_disk() -> List[Dict[str, object]]:
    """ローカル JSON から登録地点を読み込む。"""
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_saved_locations_to_disk(locations: List[Dict[str, object]]) -> None:
    """登録地点をローカル JSON に保存。"""
    try:
        CACHE_FILE.write_text(
            json.dumps(locations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_config_from_disk() -> Dict[str, object]:
    """モデル選択やプリセットの設定をローカル JSON から読み込む。"""
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config_to_disk(config: Dict[str, object]) -> None:
    """モデル選択やプリセットの設定をローカル JSON に保存。"""
    try:
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def init_state() -> None:
    """Session State の初期化。"""
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
        "selected_models": None,
        "theme_mode": "dark",   # ダーク / ライト
        "bg_pattern": None,     # 1〜3 のランダム背景パターン
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 登録地点の復元
    if not st.session_state.get("saved_locations"):
        disk_locations = load_saved_locations_from_disk()
        if disk_locations:
            st.session_state.saved_locations = disk_locations

    # 表示モデル選択の復元
    if st.session_state.selected_models is None:
        all_names = [m["display_name"] for m in MODEL_INFOS]
        cfg = load_config_from_disk()
        selected = cfg.get("selected_models")
        if isinstance(selected, list):
            selected = [name for name in selected if name in all_names]
        if not selected:
            selected = all_names
        st.session_state.selected_models = selected


def apply_theme_css(mode: str) -> None:
    """
    Streamlit の見た目を壊さないシンプルなテーマ切替。
    - ライト: 明るいグレー背景
    - ダーク: 濃紺背景
    """
    is_dark = (mode == "dark")

    if is_dark:
        bg = "#020617"   # 濃紺
        fg = "#e5e7eb"   # 明るいグレー
    else:
        bg = "#f9fafb"   # 明るいグレー
        fg = "#111827"   # ほぼ黒

    css = f"""
    <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: {bg} !important;
      color: {fg} !important;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .stApp {{
      background: {bg} !important;
      color: {fg} !important;
    }}

    /* 文字色だけ最低限合わせる（レイアウトは素のまま） */
    .stMarkdown, .stText, .stCaption, .stDataFrame, .stTable, label, span, p, h1, h2, h3, h4 {{
      color: {fg} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)




def render_saved_locations(saved: List[Dict[str, object]]) -> None:
    """登録地点の一覧＋JSON入出力 UI。"""
    if saved:
        options = [f"{loc['name']} ({loc['lat']:.4f}, {loc['lon']:.4f})" for loc in saved]
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
    st.dataframe(
        saved_df.rename(columns={"name": "ラベル", "lat": "緯度", "lon": "経度", "place_name": "地名"}).style.format(
            {"緯度": "{:.4f}", "経度": "{:.4f}"}
        ),
        height=240,
    )

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


def render_control_panel() -> None:
    """上部の「地点の指定・登録」フォーム部分。"""
    st.subheader("地点の指定")

    query = st.text_input(
        "地名/住所 または '緯度, 経度'（任意）",
        key="query_input",
        placeholder="例: 東京駅 / 38.1297, 140.4445",
    )

    if st.button("地名/座標から検索"):
        result = geocode_place(query)
        if result:
            lat, lon, name = result
            st.session_state.lat, st.session_state.lon = lat, lon
            st.session_state.last_click = (lat, lon)
            st.session_state.place_name = name or query
            st.session_state.trigger_fetch = True
            st.success(f"座標を更新: {lat:.4f}, {lon:.4f}")
        else:
            st.error("地名/座標を特定できませんでした。")

    st.session_state.lat = st.number_input(
        "緯度", min_value=-90.0, max_value=90.0, value=float(st.session_state.lat), step=0.00001
    )
    st.session_state.lon = st.number_input(
        "経度", min_value=-180.0, max_value=180.0, value=float(st.session_state.lon), step=0.00001
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("この地点の雲量を取得", type="primary"):
            st.session_state.trigger_fetch = True

    with c2:
        if geolocation is not None:
            loc = geolocation("📍 GPS から現在地を取得")
            if loc:
                try:
                    lat = float(loc["latitude"])
                    lon = float(loc["longitude"])
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.session_state.last_click = (lat, lon)
                    st.session_state.place_name = reverse_geocode(lat, lon) or "現在地（推定）"
                    st.session_state.trigger_fetch = True
                    st.success(f"現在地を取得しました: {lat:.4f}, {lon:.4f}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"現在地の取得に失敗しました: {exc}")
        else:
            st.caption("※ GPS取得には `pip install streamlit-geolocation` と HTTPS 接続が必要です。")

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
                    {"lat": st.session_state.lat, "lon": st.session_state.lon,
                     "place_name": st.session_state.place_name}
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

    init_state()

    # テーマ選択（シンプル版）
    mode_label = st.radio(
        "テーマ",
        ["🌙 ダークモード", "☀ ライトモード"],
        horizontal=True,
        index=0 if st.session_state.theme_mode == "dark" else 1,
    )
    st.session_state.theme_mode = "dark" if "ダーク" in mode_label else "light"

    # シンプルCSS適用
    apply_theme_css(st.session_state.theme_mode)


    st.title("雲量比較")
    st.caption("Open-Meteo の複数モデルで直近 48 時間の雲量を比較します。")

    with st.expander("地点の指定・登録（タップで開閉）", expanded=True):
        render_control_panel()

    tab_compare, tab_manage = st.tabs(["比較モード", "モデルの雲量グラフ"])

    # === 比較モード ===
    with tab_compare:
        st.subheader("地図で地点を選択")

        selected_lat = st.session_state.lat
        selected_lon = st.session_state.lon

        tiles = "CartoDB dark_matter" if st.session_state.theme_mode == "dark" else "OpenStreetMap"

        map_fig = folium.Map(
            location=[selected_lat, selected_lon],
            zoom_start=13,
            control_scale=True,
            tiles=tiles,
        )
        folium.Marker(
            [selected_lat, selected_lon],
            tooltip="選択中の地点",
            popup=st.session_state.place_name,
            icon=folium.Icon(color="red", icon="map-marker"),
        ).add_to(map_fig)

        # use_container_width=True でスマホ幅でも自動調整
        map_state = st_folium(
            map_fig,
            height=420,
            key="map",
            returned_objects=["last_clicked"],
            use_container_width=True,
        )

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
            st.info("地図をクリックするか、上部フォームで地点を指定して雲量を取得してください。")
            return

        ts_df = st.session_state.data
        metadata = st.session_state.metadata or []

        # --- モデル選択＆プリセット ---
        all_display_names = [m["display_name"] for m in MODEL_INFOS]
        if not st.session_state.get("selected_models"):
            st.session_state.selected_models = all_display_names

        cfg = load_config_from_disk()
        raw_presets = cfg.get("presets") or []
        presets: List[Dict[str, object]] = []
        for p in raw_presets:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            models = [m for m in (p.get("models") or []) if m in all_display_names]
            if name and models:
                presets.append({"name": name, "models": models})

        changed = False
        for dp in DEFAULT_PRESETS:
            name = dp["name"]
            base_models = dp.get("models") or []
            models = [m for m in base_models if m in all_display_names]
            if not models:
                continue
            if any(p["name"] == name for p in presets):
                continue
            presets.append({"name": name, "models": models})
            changed = True

        if changed:
            cfg["presets"] = presets
            save_config_to_disk(cfg)

        with st.expander("モデルプリセット（保存 / 読み込み）", expanded=False):
            st.caption("よく使うモデルの組み合わせをプリセットとして保存しておけます。")

            st.markdown("**おすすめプリセット（ワンクリック適用）**")
            c_q1, c_q2, c_q3 = st.columns(3)

            def apply_preset_by_name(preset_name: str) -> None:
                target = next((p for p in presets if p["name"] == preset_name), None)
                if not target:
                    st.warning(f"プリセット「{preset_name}」が見つかりませんでした。")
                    return
                models = target["models"]
                st.session_state.selected_models = models
                cfg2 = load_config_from_disk()
                cfg2["selected_models"] = models
                cfg2["presets"] = presets
                save_config_to_disk(cfg2)
                st.success(f"プリセット「{preset_name}」を適用しました。")
                st.rerun()

            with c_q1:
                if st.button("星空観測メイン", key="quick_preset_main"):
                    apply_preset_by_name("星空観測メイン")
            with c_q2:
                if st.button("高速チェック（軽量）", key="quick_preset_fast"):
                    apply_preset_by_name("高速チェック（軽量）")
            with c_q3:
                if st.button("全球モデル比較", key="quick_preset_global"):
                    apply_preset_by_name("全球モデル比較")

            st.markdown("---")

            preset_names = [p["name"] for p in presets]
            col_p1, col_p2 = st.columns([2, 1])

            with col_p1:
                preset_select = st.selectbox(
                    "プリセット一覧",
                    options=["（未選択）"] + preset_names,
                    key="preset_select",
                )

            with col_p2:
                if st.button("プリセットを読み込む", key="preset_apply") and preset_select != "（未選択）":
                    apply_preset_by_name(preset_select)

            new_name = st.text_input(
                "新しく保存 / 上書きするプリセット名",
                key="preset_name",
                placeholder="例: 星空観測用 / 軽量モード など",
            )
            if st.button("現在の選択をプリセットとして保存", key="preset_save"):
                if not new_name.strip():
                    st.error("プリセット名を入力してください。")
                else:
                    name = new_name.strip()
                    current_models = st.session_state.selected_models or all_display_names

                    new_presets: List[Dict[str, object]] = []
                    replaced = False
                    for p in presets:
                        if p["name"] == name:
                            new_presets.append({"name": name, "models": current_models})
                            replaced = True
                        else:
                            new_presets.append(p)
                    if not replaced:
                        new_presets.append({"name": name, "models": current_models})

                    if len(new_presets) > 20:
                        new_presets = new_presets[-20:]

                    cfg["selected_models"] = current_models
                    cfg["presets"] = new_presets
                    save_config_to_disk(cfg)
                    st.success(f"プリセット「{name}」を保存しました。")
                    st.rerun()

            if st.button("選択中のプリセットを削除", key="preset_delete") and preset_select != "（未選択）":
                new_presets = [p for p in presets if p["name"] != preset_select]
                cfg["presets"] = new_presets
                save_config_to_disk(cfg)
                st.success(f"プリセット「{preset_select}」を削除しました。")
                st.rerun()

        # --- グラフ本体 ---
        st.subheader("48 時間の雲量推移")

        selected_display = st.multiselect(
            "グラフに表示するモデル",
            options=all_display_names,
            default=st.session_state.selected_models,
            help="表示したいモデルだけを選択できます（選択内容はローカルに保存されます）。",
        )

        if not selected_display:
            st.warning("少なくとも1つのモデルを選択してください。（一時的に全モデルを表示します）")
            selected_display = all_display_names

        st.session_state.selected_models = selected_display
        cfg3 = load_config_from_disk()
        cfg3["selected_models"] = selected_display
        cfg3["presets"] = presets
        save_config_to_disk(cfg3)

        columns_to_use = ["time"]
        for name in selected_display:
            col = f"{name} (Total cloud)"
            if col in ts_df.columns:
                columns_to_use.append(col)

        filtered_ts = ts_df[columns_to_use].copy()
        chart_df = prepare_chart_data(filtered_ts)

        if chart_df.empty:
            st.info("有効な雲量データがありません。")
        else:
            st.altair_chart(build_line_chart(chart_df), use_container_width=True)

        st.subheader("詳細データ")
        st.dataframe(filtered_ts, use_container_width=True, height=360)

        st.subheader("モデル別データ状況")
        selected_set = set(selected_display)
        filtered_meta = []
        for row in metadata:
            model_label = row.get("モデル", "")
            base_name = model_label.split(" (Total cloud)")[0]
            if base_name in selected_set:
                filtered_meta.append(row)

        st.table(pd.DataFrame(filtered_meta))

    # === モデルの雲量グラフ ===
    with tab_manage:
        st.subheader("モデルの雲量グラフ（登録地点から選択）")
        saved = st.session_state.saved_locations

        if not saved:
            st.info("登録済みの地点がありません。上部のフォームまたは比較モードで地点を登録してください。")
        else:
            loc_options = [f"{loc['name']} ({loc['lat']:.4f}, {loc['lon']:.4f})" for loc in saved]
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
                    f"推定された地名: {st.session_state.place_name}"
                )

                chart_df = prepare_layer_chart_data(layer_df)
                st.subheader(f"{st.session_state.layer_model} の層別雲量（48 時間）")
                st.altair_chart(
                    build_layer_chart(chart_df, st.session_state.layer_model),
                    use_container_width=True,
                )

                st.subheader("詳細データ")
                st.dataframe(layer_df, use_container_width=True, height=360)
            else:
                st.info("地点とモデルを選択して「選択した地点とモデルの雲量を表示」を押してください。")


if __name__ == "__main__":
    main()
