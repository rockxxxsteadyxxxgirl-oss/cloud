#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from PIL import Image
from streamlit_folium import st_folium
from urllib.parse import urlencode

# =========================================
# 定数 / モデル定義
# =========================================

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


def round_coord(value: float) -> float:
    """API へ投げる座標の丸め精度（5 桁）"""
    return round(value, 5)


# =========================================
# 座標 / ジオコーディング系
# =========================================

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

    coords = parse_latlon(query)
    if coords is not None:
        lat, lon = coords
        try:
            geocoder = Nominatim(user_agent="cloud_cover_app", timeout=5)
            result = geocoder.reverse((lat, lon), language="ja")
            name = result.address if result is not None else None
        except Exception:
            name = None
        if not name:
            name = f"{lat:.5f}, {lon:.5f}"
        return lat, lon, name

    try:
        geocoder = Nominatim(user_agent="cloud_cover_app", timeout=5)
        result = geocoder.geocode(query)
        if result is None:
            return None
        return float(result.latitude), float(result.longitude), result.address
    except Exception:
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """逆ジオコーディングで地名を取得（失敗したら None）。"""
    try:
        geocoder = Nominatim(user_agent="cloud_cover_app", timeout=5)
        result = geocoder.reverse((lat, lon), language="ja")
        if result is None:
            return None
        return result.address
    except Exception:
        return None


# =========================================
# ローカル JSON 設定
# =========================================

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


# =========================================
# Streamlit 状態 & テーマ
# =========================================

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
        "theme_mode": "dark",
        "last_layer_model_choice": None,
        "meteo_df": None,
        "timelapse_index": 0,
        "contour_grid": None,
        "contour_lats": None,
        "contour_lons": None,
        "contour_model": None,
        "contour_time_idx": None,
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
    """ダーク/ライトテーマの CSS（現在はダーク固定）。"""
    is_dark = (mode == "dark")
    if is_dark:
        bg = "#020617"
        fg = "#e5e7eb"
    else:
        bg = "#f9fafb"
        fg = "#111827"

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

    .stMarkdown, .stText, .stCaption, .stDataFrame, .stTable, label, span, p, h1, h2, h3, h4 {{
      color: {fg} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================================
# Open-Meteo API ラッパー
# =========================================

def normalize_cloud(series: pd.Series) -> pd.Series:
    """0〜1 の小数で来た雲量を 0〜100% に直すヘルパー。"""
    series = pd.to_numeric(series, errors="coerce")
    max_val = series.max(skipna=True)
    has_fraction = ((series % 1) != 0).any()
    if max_val is not None and max_val <= 1 and has_fraction:
        series = series * 100
    return series


def filter_next_hours(df: pd.DataFrame, hours: int = 72) -> pd.DataFrame:
    """直近 hours 時間だけに絞る。"""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz=df["time"].dt.tz)
    cutoff = now + timedelta(hours=hours)
    filtered = df[(df["time"] >= now) & (df["time"] <= cutoff)].copy()
    filtered["time"] = filtered["time"].dt.tz_localize(None)
    return filtered


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, model: str) -> Tuple[pd.DataFrame, str]:
    """Open-Meteo から総雲量を取得し、0〜100% に正規化した DataFrame を返す。"""
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

    if (candidate.isna().all()
        or (max_val is not None and max_val <= 1 and not has_fraction)) and has_layer_data:
        candidate = pd.concat([low, mid, high], axis=1).max(axis=1)
        max_val = candidate.max(skipna=True)
        has_fraction = ((candidate % 1) != 0).any()

    if max_val is not None and max_val <= 1 and has_fraction:
        candidate = candidate * 100

    df = pd.DataFrame({"time": times, "cloud_cover": candidate})
    return df, timezone


def load_models(lat: float, lon: float) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """全モデルの総雲量を取得・マージして 1 つの DataFrame にまとめる。"""
    frames: List[pd.DataFrame] = []
    metadata: List[Dict[str, str]] = []
    for info in MODEL_INFOS:
        display_name, model_code = info["display_name"], info["code"]
        total_label = f"{display_name} (Total cloud)"
        df, tz = fetch_forecast(lat, lon, model_code)
        df = filter_next_hours(df, hours=72)
        renamed = df.rename(columns={"cloud_cover": total_label})
        frames.append(renamed[["time", total_label]])
        metadata.append({"モデル": total_label, "データ件数": len(df), "タイムゾーン": tz})

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged, metadata


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_layered_forecast(lat: float, lon: float, model: str) -> pd.DataFrame:
    """層別雲量（総雲量＋下層・中層・上層）の 72h 分を取得。"""
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
    return filter_next_hours(df, hours=72)


def build_forecast_url(
    lat: float,
    lon: float,
    model_code: str,
    hourly: str = "cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
) -> str:
    """デバッグ用に Open-Meteo API URL を構築して返す。"""
    params = {
        "latitude": round_coord(lat),
        "longitude": round_coord(lon),
        "hourly": hourly,
        "forecast_days": 7,
        "timezone": "auto",
        "models": model_code,
    }
    return f"{API_URL}?{urlencode(params)}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_full_meteo(lat: float, lon: float, model: str) -> pd.DataFrame:
    """雲量＋気温＋湿度＋風向/風速を 48h 分取得。"""
    params = {
        "latitude": round_coord(lat),
        "longitude": round_coord(lon),
        "hourly": ",".join(
            [
                "cloudcover",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
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
        raise ValueError("Open-Meteo から気象要素データを取得できませんでした。")
    times = pd.to_datetime(times)

    df = pd.DataFrame(
        {
            "time": times,
            "cloudcover": normalize_cloud(pd.Series(hourly.get("cloudcover"))),
            "temp_2m": pd.to_numeric(pd.Series(hourly.get("temperature_2m")), errors="coerce"),
            "rh_2m": pd.to_numeric(pd.Series(hourly.get("relative_humidity_2m")), errors="coerce"),
            "wind_speed_10m": pd.to_numeric(pd.Series(hourly.get("wind_speed_10m")), errors="coerce"),
            "wind_dir_10m": pd.to_numeric(pd.Series(hourly.get("wind_direction_10m")), errors="coerce"),
        }
    )
    df = filter_next_hours(df, hours=48)
    return df


# =========================================
# 等値線マップ用（実験）
# =========================================

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_single_time_cloud(lat: float, lon: float, model: str, target_time_idx: int = 0) -> Tuple[pd.Timestamp, float]:
    """指定地点で cloudcover を取得し、target_time_idx の値を返す（実験用）。"""
    params = {
        "latitude": round_coord(lat),
        "longitude": round_coord(lon),
        "hourly": "cloudcover",
        "forecast_days": 2,
        "timezone": "auto",
        "models": model,
    }
    resp = requests.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    vals = hourly.get("cloudcover")
    if not times or not vals:
        raise ValueError("cloudcover データが取得できませんでした。")
    times = pd.to_datetime(times)
    series = pd.to_numeric(pd.Series(vals), errors="coerce")
    series = series * (100.0 if series.max(skipna=True) <= 1.0 else 1.0)
    idx = min(max(target_time_idx, 0), len(series) - 1)
    return times[idx], float(series.iloc[idx])


def generate_cloud_grid(
    center_lat: float,
    center_lon: float,
    model: str,
    radius_km: float = 50.0,
    grid_size: int = 13,
    target_time_idx: int = 0,
):
    """
    中心座標から半径 radius_km の正方形領域を grid_size x grid_size に等分し、
    各格子点で cloudcover を取得して 2D 配列として返す（実験用）。
    ※ API コールが多くなるので注意。
    """
    deg_per_km = 1.0 / 111.0
    delta_deg = radius_km * deg_per_km
    lats = np.linspace(center_lat - delta_deg, center_lat + delta_deg, grid_size)
    lons = np.linspace(center_lon - delta_deg, center_lon + delta_deg, grid_size)

    grid = np.zeros((grid_size, grid_size), dtype=float)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            _, val = _fetch_single_time_cloud(lat, lon, model, target_time_idx)
            grid[i, j] = val

    return lats, lons, grid


def grid_to_image(grid: np.ndarray, vmin: float = 0, vmax: float = 100) -> Image.Image:
    """
    雲量グリッドを 0〜255 の擬似カラー画像（RGBA）に変換する簡易関数。
    ここでは単純なグレースケールを使用。
    """
    norm = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    arr = (norm * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L").convert("RGBA")
    return img


# =========================================
# Altair 用ヘルパー / 解析
# =========================================

def prepare_chart_data(timeseries: pd.DataFrame) -> pd.DataFrame:
    """複数モデル時系列用に wide→long 変換。"""
    chart_df = timeseries.melt("time", var_name="model", value_name="cloud_cover")
    chart_df["cloud_cover"] = pd.to_numeric(chart_df["cloud_cover"], errors="coerce")
    return chart_df.dropna(subset=["cloud_cover"])


def prepare_layer_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """層別雲量用に wide→long 変換。"""
    chart_df = df.melt("time", var_name="layer", value_name="cloud_cover")
    chart_df["cloud_cover"] = pd.to_numeric(chart_df["cloud_cover"], errors="coerce")
    return chart_df.dropna(subset=["cloud_cover"])


def build_line_chart(chart_df: pd.DataFrame) -> alt.Chart:
    """72h / 複数モデル比較用ラインチャート。"""
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
    """層別雲量用ラインチャート。"""
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


def analyze_vertical_cloud(layer_df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """
    層別雲量 DataFrame から簡易的な統計を出す。
    対象列: 「総雲量」「下層雲」「中層雲」「上層雲」
    """
    if layer_df is None or layer_df.empty:
        return pd.DataFrame()

    df = layer_df.copy()
    df = df.sort_values("time")
    if "time" in df.columns:
        t0 = df["time"].min()
        t1 = t0 + pd.Timedelta(hours=hours)
        df = df[(df["time"] >= t0) & (df["time"] <= t1)]

    if df.empty:
        return pd.DataFrame()

    layers: List[str] = [c for c in ["総雲量", "下層雲", "中層雲", "上層雲"] if c in df.columns]

    records = []
    for col in layers:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mean_val = s.mean()
        frac_clear = (s < 30).sum() / len(s)

        altitude_hint = ""
        if col == "下層雲":
            altitude_hint = "〜約 2km"
        elif col == "中層雲":
            altitude_hint = "約 2〜7km"
        elif col == "上層雲":
            altitude_hint = "約 7km 以上"
        elif col == "総雲量":
            altitude_hint = "全高度"

        records.append(
            {
                "層": col,
                "想定高度帯": altitude_hint,
                "平均雲量(%)": round(mean_val, 1),
                "雲量30%未満の時間割合": f"{frac_clear * 100:.1f}%",
            }
        )

    return pd.DataFrame(records)


def render_weather_multi_chart(df: pd.DataFrame, model_name: str) -> None:
    """雲量＋気温＋湿度＋風速＋風向の同時可視化チャート群を描画。"""
    if df is None or df.empty:
        st.info("気象要素データがありません。")
        return

    st.caption(f"{model_name} による 48 時間の雲量＋気象要素")

    base = alt.Chart(df).encode(
        x=alt.X(
            "time:T",
            title="日時",
            axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, labelFontSize=10, titleFontSize=11),
        )
    )

    # 雲量 + 湿度
    cloud_line = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("cloudcover:Q", title="雲量 / 湿度 (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.value("#60a5fa"),
        tooltip=[
            alt.Tooltip("time:T", title="日時"),
            alt.Tooltip("cloudcover:Q", title="雲量(%)"),
        ],
    )

    rh_line = base.mark_line(strokeDash=[4, 3], strokeWidth=1.6).encode(
        y="rh_2m:Q",
        color=alt.value("#f97316"),
        tooltip=[
            alt.Tooltip("time:T", title="日時"),
            alt.Tooltip("rh_2m:Q", title="湿度(%)"),
        ],
    )

    chart1 = (
        alt.layer(cloud_line, rh_line)
        .resolve_scale(y="shared")
        .properties(height=260)
        .configure_view(strokeWidth=0)
    )

    # 気温 + 風速
    temp_line = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("temp_2m:Q", title="気温 (°C)"),
        color=alt.value("#ef4444"),
        tooltip=[
            alt.Tooltip("time:T", title="日時"),
            alt.Tooltip("temp_2m:Q", title="気温(°C)"),
        ],
    )

    wind_line = base.mark_line(strokeDash=[2, 2], strokeWidth=1.6).encode(
        y=alt.Y("wind_speed_10m:Q", title="風速 (m/s)"),
        color=alt.value("#22c55e"),
        tooltip=[
            alt.Tooltip("time:T", title="日時"),
            alt.Tooltip("wind_speed_10m:Q", title="風速(m/s)"),
        ],
    )

    chart2 = (
        alt.layer(temp_line, wind_line)
        .resolve_scale(y="independent")
        .properties(height=260)
        .configure_view(strokeWidth=0)
    )

    # 風向
    wind_dir = base.mark_line(strokeWidth=1.6).encode(
        y=alt.Y("wind_dir_10m:Q", title="風向 (°)", scale=alt.Scale(domain=[0, 360])),
        color=alt.value("#a855f7"),
        tooltip=[
            alt.Tooltip("time:T", title="日時"),
            alt.Tooltip("wind_dir_10m:Q", title="風向(°)"),
        ],
    )
    chart3 = wind_dir.properties(height=200).configure_view(strokeWidth=0)

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)
    st.altair_chart(chart3, use_container_width=True)


# =========================================
# UI: 登録地点 / 上部フォーム
# =========================================

def render_saved_locations(saved: List[Dict[str, object]]) -> None:
    """登録地点の一覧＋JSON入出力 UI。"""
    if saved:
        options = [f"{loc['name']} ({loc['lat']:.5f}, {loc['lon']:.5f})" for loc in saved]
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
    else:
        st.info("登録済みの地点はまだありません。")

    st.markdown("**登録地点の一覧 / エクスポート**")
    saved_df = (
        pd.DataFrame(saved)[["name", "lat", "lon", "place_name"]]
        if saved
        else pd.DataFrame(columns=["name", "lat", "lon", "place_name"])
    )
    st.dataframe(
        saved_df.rename(columns={"name": "ラベル", "lat": "緯度", "lon": "経度", "place_name": "地名"}).style.format(
            {"緯度": "{:.5f}", "経度": "{:.5f}"}
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
                cleaned.append(
                    {
                        "name": name,
                        "lat": float(lat),
                        "lon": float(lon),
                        "place_name": place_name,
                    }
                )

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
        except Exception as exc:  # noqa: BLE001
            st.error(f"インポートに失敗しました: {exc}")


def render_control_panel() -> None:
    """上部の「地点の指定・登録」フォーム部分。"""
    st.subheader("地点の指定")

    query = st.text_input(
        "地名/住所 または '緯度, 経度'（任意）",
        key="query_input",
        placeholder="例: 東京駅 / 38.12970, 140.44450",
    )

    if st.button("地名/座標から検索"):
        result = geocode_place(query)
        if result:
            lat, lon, name = result
            st.session_state.lat, st.session_state.lon = lat, lon
            st.session_state.last_click = (lat, lon)
            st.session_state.place_name = name or query
            st.session_state.trigger_fetch = True
            st.success(f"座標を更新: {lat:.5f}, {lon:.5f}")
        else:
            st.error("地名/座標を特定できませんでした。")

    st.session_state.lat = st.number_input(
        "緯度",
        min_value=-90.0,
        max_value=90.0,
        value=float(st.session_state.lat),
        step=0.00001,
        format="%.5f",
    )
    st.session_state.lon = st.number_input(
        "経度",
        min_value=-180.0,
        max_value=180.0,
        value=float(st.session_state.lon),
        step=0.00001,
        format="%.5f",
    )

    c1, _ = st.columns(2)
    with c1:
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
                    {
                        "lat": st.session_state.lat,
                        "lon": st.session_state.lon,
                        "place_name": st.session_state.place_name,
                    }
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


# =========================================
# UI セクション
# =========================================

def render_map_section() -> None:
    """上部の Folium 地図（地点選択）と、地点指定/登録パネル。"""
    st.subheader("地図で地点を選択")

    selected_lat = st.session_state.lat
    selected_lon = st.session_state.lon

    tiles = "OpenStreetMap"
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

    map_state = st_folium(
        map_fig,
        height=380,
        key="map_main",
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
                st.info(f"地図で選択: {lat_click:.5f}, {lon_click:.5f}")

    st.caption(f"現在の座標: {st.session_state.lat:.5f}, {st.session_state.lon:.5f}")
    st.caption(f"推定された地名: {st.session_state.place_name}")

    with st.expander("地点の指定・登録（タップで開閉）", expanded=False):
        render_control_panel()


def render_multimodel_section() -> None:
    """72時間の雲量推移（複数モデル比較）。"""
    st.markdown("---")
    st.subheader("72 時間の雲量推移（複数モデル比較）")

    ts_df = st.session_state.get("data")
    metadata = st.session_state.get("metadata") or []

    if ts_df is None:
        st.info("地図をクリックするか、上部フォームで地点を指定して雲量を取得してください。")
        return

    all_display_names = [m["display_name"] for m in MODEL_INFOS]
    if not st.session_state.get("selected_models"):
        st.session_state.selected_models = all_display_names

    cfg = load_config_from_disk()
    raw_presets = cfg.get("presets") or []
    presets: List[dict] = []
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

    # プリセット UI
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

                new_presets: List[dict] = []
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

        if st.button("選択中のプリセットを削除", key="preset_delete") and preset_select != "（未選択）":
            new_presets = [p for p in presets if p["name"] != preset_select]
            cfg["presets"] = new_presets
            save_config_to_disk(cfg)
            st.success(f"プリセット「{preset_select}」を削除しました。")

    # グラフ描画
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

    st.subheader("詳細データ（比較用）")
    with st.expander("テーブルを表示", expanded=False):
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


def render_layered_section() -> None:
    """層別雲量グラフ＋API URL＋モデル検証 JSON 出力。"""
    st.markdown("---")
    st.subheader("モデルの雲量グラフ（現在の地点・層別）")

    target_lat: float = st.session_state.lat
    target_lon: float = st.session_state.lon
    target_label: str = st.session_state.place_name or "現在の地点"

    st.caption(
        f"現在の地点: {target_lat:.5f}, {target_lon:.5f} / "
        f"推定された地名: {target_label}"
    )

    model_options = [m["display_name"] for m in MODEL_INFOS]
    model_choice = st.selectbox("モデルを選択", options=model_options, key="manage_model_select")

    auto_fetch = False
    prev_choice = st.session_state.get("last_layer_model_choice")
    if prev_choice is None:
        st.session_state.last_layer_model_choice = model_choice
    elif model_choice != prev_choice:
        st.session_state.last_layer_model_choice = model_choice
        auto_fetch = True

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        manual_clicked = st.button("選択したモデルの層別雲量を表示", key="manage_fetch")

    with col_b2:
        if st.button("この地点で全モデル検証＆JSON出力", key="manage_diag"):
            diagnostics = []
            for info in MODEL_INFOS:
                model_code = info["code"]
                label = info["display_name"]
                entry = {"model": label, "code": model_code}
                entry["url"] = build_forecast_url(
                    target_lat,
                    target_lon,
                    model_code,
                    hourly="cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
                )
                try:
                    df = fetch_layered_forecast(target_lat, target_lon, model_code)
                    df = filter_next_hours(df, hours=72)
                    entry["status"] = "success"
                    entry["rows"] = len(df)
                    entry["time_start"] = df["time"].min().isoformat() if not df.empty else None
                    entry["time_end"] = df["time"].max().isoformat() if not df.empty else None
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
            )

    if auto_fetch or manual_clicked:
        try:
            model_code = next(m["code"] for m in MODEL_INFOS if m["display_name"] == model_choice)
            with st.spinner("Open-Meteo からデータ取得中..."):
                layer_df = fetch_layered_forecast(target_lat, target_lon, model_code)
            st.session_state.layer_data = layer_df
            st.session_state.layer_model = model_choice
            st.session_state.layer_api_url = build_forecast_url(
                target_lat,
                target_lon,
                model_code,
                hourly="cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
            )
            if not auto_fetch:
                st.success(f"{target_label} / {model_choice} の層別データを更新しました。")
        except Exception as exc:
            st.error(f"取得に失敗しました: {exc}")

    layer_df = st.session_state.get("layer_data")
    if layer_df is not None and not layer_df.empty:
        chart_df = prepare_layer_chart_data(layer_df)
        st.subheader(f"{st.session_state.layer_model} の層別雲量（72 時間）")
        st.altair_chart(
            build_layer_chart(chart_df, st.session_state.layer_model),
            use_container_width=True,
        )

        url_text = st.session_state.get("layer_api_url")
        if url_text:
            st.markdown("**この層別グラフの取得に使用した Open-Meteo API URL**")
            st.code(url_text, language="text")

        with st.expander("詳細データ（層別）テーブル", expanded=False):
            st.dataframe(layer_df, use_container_width=True, height=360)
    else:
        st.info("モデルを選択後に切り替えると自動で雲量を取得します。（必要に応じてボタンでも更新できます）")


def render_vertical_analysis_section() -> None:
    """高度依存の雲量解析（簡易統計）。"""
    st.markdown("---")
    st.subheader("高度依存の雲量解析（簡易）")

    layer_df = st.session_state.get("layer_data")
    if layer_df is None or layer_df.empty:
        st.info("まず上で層別データを取得してください。")
        return

    hours_for_analysis = st.slider("解析対象とする時間範囲（先頭からの時間）", 6, 72, 24, step=6)
    analysis_df = analyze_vertical_cloud(layer_df, hours=hours_for_analysis)
    if analysis_df.empty:
        st.info("解析対象のデータが不足しています。")
        return

    st.caption(
        f"※ ここでは簡易的に、層別の平均雲量と「雲量 < 30%」の時間割合を算出しています。（対象: 先頭 {hours_for_analysis} 時間）"
    )
    st.table(analysis_df)


def render_meteo_section() -> None:
    """雲量＋気温・湿度・風向/風速＋48hタイムラプス風ビュー。"""
    st.markdown("---")
    st.subheader("雲量＋気温・湿度・風速・風向の同時可視化（48 時間）")

    model_options = [m["display_name"] for m in MODEL_INFOS]
    meteo_model_name = st.selectbox(
        "気象要素を取得するモデルを選択",
        options=model_options,
        key="meteo_model_select",
    )
    meteo_model_code = next(m["code"] for m in MODEL_INFOS if m["display_name"] == meteo_model_name)

    if st.button("48時間分の気象要素を取得・更新", key="fetch_meteo"):
        try:
            with st.spinner("Open-Meteo から気象要素データ取得中..."):
                meteo_df = fetch_full_meteo(st.session_state.lat, st.session_state.lon, meteo_model_code)
            st.session_state["meteo_df"] = meteo_df
            st.success("気象要素データを更新しました。")
        except Exception as exc:
            st.error(f"取得に失敗しました: {exc}")

    meteo_df = st.session_state.get("meteo_df")
    if meteo_df is None or meteo_df.empty:
        st.info("「気象要素を取得・更新」ボタンで 48 時間分のデータを取得すると、ここに表示されます。")
        return

    render_weather_multi_chart(meteo_df, meteo_model_name)

    # スライダースクラブ型タイムラプス
    st.subheader("48 時間タイムラプス風ビュー（スクラブ）")
    total_frames = len(meteo_df)
    if total_frames <= 0:
        return

    idx = st.slider(
        "タイムラプス上の時刻（インデックス）",
        0,
        max(0, total_frames - 1),
        st.session_state.get("timelapse_index", 0),
    )
    st.session_state.timelapse_index = idx

    current_row = meteo_df.iloc[idx]
    st.caption(f"選択中の時刻: {current_row['time']}")

    c_tm1, c_tm2, c_tm3, c_tm4 = st.columns(4)
    with c_tm1:
        st.metric("雲量(%)", f"{current_row['cloudcover']:.0f}")
    with c_tm2:
        st.metric("気温(°C)", f"{current_row['temp_2m']:.1f}")
    with c_tm3:
        st.metric("湿度(%)", f"{current_row['rh_2m']:.0f}")
    with c_tm4:
        st.metric("風速(m/s)", f"{current_row['wind_speed_10m']:.1f}")

    st.caption("※ スライダーを左右に動かして、48時間分の変化をタイムラプスのように確認できます。")


def render_contour_section() -> None:
    """予測モデルの等値線マップ（雲量グリッド・実験的）。"""
    st.markdown("---")
    st.subheader("予測モデルの等値線マップ（雲量グリッド・実験的）")

    target_lat: float = st.session_state.lat
    target_lon: float = st.session_state.lon
    st.caption(f"中心座標: {target_lat:.5f}, {target_lon:.5f}")

    model_options = [m["display_name"] for m in MODEL_INFOS]
    contour_model_name = st.selectbox(
        "グリッドを計算するモデル",
        options=model_options,
        key="contour_model_select",
    )
    contour_model_code = next(m["code"] for m in MODEL_INFOS if m["display_name"] == contour_model_name)

    radius_km = st.slider("半径（km）", 10, 100, 50, step=10)
    grid_size = st.selectbox("グリッドサイズ", options=[9, 13, 17], index=1)
    time_idx = st.slider("対象時刻インデックス（0=最初の時刻）", 0, 24, 0)

    if st.button("等値線マップを生成（API 多数アクセス・実験）", key="contour_generate"):
        try:
            with st.spinner("雲量グリッドを計算中...（少し時間がかかる場合があります）"):
                lats, lons, grid = generate_cloud_grid(
                    target_lat,
                    target_lon,
                    contour_model_code,
                    radius_km=radius_km,
                    grid_size=grid_size,
                    target_time_idx=time_idx,
                )
            st.session_state.contour_grid = grid
            st.session_state.contour_lats = lats
            st.session_state.contour_lons = lons
            st.session_state.contour_model = contour_model_name
            st.session_state.contour_time_idx = time_idx
            st.success("グリッド計算が完了しました。下のマップにオーバーレイを表示します。")
        except Exception as exc:
            st.error(f"グリッド計算に失敗しました: {exc}")

    grid = st.session_state.get("contour_grid")
    lats = st.session_state.get("contour_lats")
    lons = st.session_state.get("contour_lons")
    if grid is None or lats is None or lons is None:
        st.info("等値線マップを表示するには、上のボタンでグリッドを生成してください。")
        return

    # グレースケール画像へ変換
    img = grid_to_image(grid, vmin=0, vmax=100)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = buf.getvalue()

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    m = folium.Map(
        location=[target_lat, target_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Marker(
        [target_lat, target_lon],
        tooltip="中心座標",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=b64,
        bounds=bounds,
        opacity=0.6,
        name="Cloud Grid",
        interactive=False,
        cross_origin=False,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st.caption(
        f"モデル: {st.session_state.contour_model} / グリッド: {grid.shape[0]}×{grid.shape[1]} / "
        f"半径: 約 {radius_km} km / 時刻インデックス: {st.session_state.contour_time_idx}"
    )
    st_folium(m, height=420, key="map_contour", use_container_width=True)


# =========================================
# メイン
# =========================================

def main() -> None:
    st.set_page_config(page_title="雲量比較・気象解析ビューア（単一ファイル版）", layout="wide")

    init_state()

    # テーマはとりあえずダーク固定
    st.session_state.theme_mode = "dark"
    apply_theme_css(st.session_state.theme_mode)

    st.title("雲量比較・気象解析ビューア（単一ファイル完全版）")
    st.caption("Open-Meteo の複数モデルで直近 72 時間の雲量と、気象要素・等値線マップを総合的にチェックします。")

    # 上部：地図＋地点指定パネル
    render_map_section()

    # 地図やフォームから指定された地点でデータ取得
    if st.session_state.trigger_fetch:
        st.session_state.trigger_fetch = False
        try:
            with st.spinner("Open-Meteo から雲量データ取得中..."):
                ts_df, metadata = load_models(st.session_state.lat, st.session_state.lon)
            st.session_state.data = ts_df
            st.session_state.metadata = metadata
            st.success("雲量データを更新しました。")
        except Exception as exc:  # noqa: BLE001
            st.error(f"取得に失敗しました: {exc}")

    # 中段：72h 複数モデル比較
    render_multimodel_section()

    # 下段：層別雲量グラフ / 高度依存解析 / 気象要素 / 等値線マップ
    render_layered_section()
    render_vertical_analysis_section()
    render_meteo_section()
    render_contour_section()


if __name__ == "__main__":
    main()
