#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open-Meteo の複数モデルで雲量を比較するスマホ向け Streamlit アプリ
ダークモード UI 固定・地図は OpenStreetMap（ライトタイル）
ズーム：初回 6、地点選択後は自動で 13 へズームイン

実行例:
    streamlit run app2.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

# Optional: GPS
try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None  # GPS なしで動作可

# =========================
# 定数・モデル定義
# =========================

API_URL = "https://api.open-meteo.com/v1/forecast"

APP_DIR = Path(".")
SAVED_LOCATIONS_PATH = APP_DIR / ".saved_locations.json"
CONFIG_PATH = APP_DIR / ".cloud_viewer_config.json"

# モデル一覧
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

DISPLAY_TO_CODE = {m["display_name"]: m["code"] for m in MODEL_INFOS}
CODE_TO_DISPLAY = {m["code"]: m["display_name"] for m in MODEL_INFOS}

# おすすめプリセット
DEFAULT_PRESETS: Dict[str, List[str]] = {
    "星空観測メイン": [
        "JMA MSM 5km",
        "JMA GSM 20km",
        "ECMWF IFS 0.25°",
        "ECMWF IFS",
    ],
    "高速チェック（軽量）": [
        "JMA MSM 5km",
        "ECMWF IFS 0.25°",
    ],
    "全球モデル比較": [
        "ECMWF IFS 0.25°",
        "NOAA GFS 0.25°",
        "ICON Global 0.25°",
        "Météo-France Seamless",
        "UKMO Seamless",
        "JMA Seamless",
    ],
}
# 全モデル（ALL）プリセットを追加
DEFAULT_PRESETS["ALL"] = [m["display_name"] for m in MODEL_INFOS]


# =========================
# データクラス
# =========================

@dataclass
class ModelMeta:
    display_name: str
    code: str
    rows: int
    timezone: Optional[str]


# =========================
# ユーティリティ（テーマ・CSS）
# =========================

def apply_theme_css() -> None:
    """ダークモード固定の CSS 適用（スマホ向け余白調整＋タブ＋ボタン＋入力＋Expander＋プルダウン全般）"""

    # ダークモード配色
    bg = "#020617"          # アプリ全体の背景色（ほぼ黒に近い濃紺）
    fg = "#e5e7eb"          # 基本文字色（明るいグレー）

    tab_active_bg = "#111827"    # タブ：選択中タブ背景
    tab_inactive_bg = "#020617"  # タブ：未選択タブ背景
    tab_border = "#4b5563"       # タブ：枠線・下線色

    btn_bg = "#111827"      # ボタン通常時の背景色
    btn_fg = "#e5e7eb"      # ボタン文字色
    btn_hover = "#1f2937"   # ボタン hover 背景

    input_bg = "#020617"    # テキスト/数値/セレクト入力枠の背景
    input_fg = "#e5e7eb"    # テキスト/数値/セレクト入力枠の文字

    border = "#4b5563"      # 各種コンポーネント共通ボーダー色

    exp_header_bg = "#020617"   # Expander ヘッダー背景

    tag_bg = "#111827"      # マルチセレクトのタグ背景
    tag_fg = "#e5e7eb"      # マルチセレクトのタグ文字

    option_bg = "#020617"       # プルダウン選択肢の背景（ダーク）
    option_hover_bg = "#111827" # プルダウン選択肢 hover 背景

    st.markdown(
        f"""
        <style>
        /* 余白調整（スマホ向け） */
        .block-container {{
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}

        /* アプリ全体の背景・文字色を強制上書き */
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {bg} !important;
            color: {fg} !important;
        }}

        /* メインコンテンツ内のテキスト色も統一 */
        .block-container, .block-container * {{
            color: {fg} !important;
        }}

        /* タブ（st.tabs）の配色を上書き */
        .stTabs [role="tablist"] button {{
            background-color: {tab_inactive_bg} !important;
            color: {fg} !important;
            border: 1px solid {tab_border} !important;
            border-bottom: 1px solid {tab_border} !important;
            padding: 0.35rem 0.8rem !important;
            font-size: 0.9rem !important;
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            background-color: {tab_active_bg} !important;
            color: {fg} !important;
            border-bottom: 2px solid #3b82f6 !important;
        }}

        /* ボタンの配色 */
        [data-testid="stButton"] button {{
            background-color: {btn_bg} !important;
            color: {btn_fg} !important;
            border: 1px solid {border} !important;
            border-radius: 0.5rem !important;
            padding: 0.25rem 0.8rem !important;
            font-size: 0.9rem !important;
        }}
        [data-testid="stButton"] button:hover {{
            background-color: {btn_hover} !important;
        }}

        /* テキスト入力・数値入力・セレクトボックスの配色（枠部分） */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stMultiSelect"] div[role="combobox"] {{
            background-color: {input_bg} !important;
            color: {input_fg} !important;
            border: 1px solid {border} !important;
            border-radius: 0.5rem !important;
        }}

        /* セレクト系の中のテキスト */
        [data-testid="stSelectbox"] div[role="combobox"] * ,
        [data-testid="stMultiSelect"] div[role="combobox"] * {{
            color: {input_fg} !important;
        }}

        /* ▼▼ マルチセレクトのタグチップ部分の配色 ▼▼ */
        [data-testid="stMultiSelect"] [data-baseweb="tag"],
        [data-testid="stMultiSelect"] [data-baseweb="tag"] div {{
            background-color: {tag_bg} !important;
            color: {tag_fg} !important;
            border-radius: 999px !important;
            border: 1px solid {border} !important;
        }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span {{
            color: {tag_fg} !important;
        }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg *,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{
            stroke: {tag_fg} !important;
        }}

        /* ▼▼ MultiSelect 個別の listbox ▼▼ */
        [data-testid="stMultiSelect"] ul[role="listbox"],
        [data-testid="stMultiSelect"] div[role="listbox"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
            border: 1px solid {border} !important;
        }}
        [data-testid="stMultiSelect"] ul[role="listbox"] li,
        [data-testid="stMultiSelect"] div[role="listbox"] li,
        [data-testid="stMultiSelect"] div[role="listbox"] div[role="option"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
        }}
        [data-testid="stMultiSelect"] ul[role="listbox"] li:hover,
        [data-testid="stMultiSelect"] div[role="listbox"] li:hover,
        [data-testid="stMultiSelect"] div[role="listbox"] div[role="option"]:hover {{
            background-color: {option_hover_bg} !important;
            color: {input_fg} !important;
        }}

        /* ▼▼ Selectbox 個別の listbox ▼▼ */
        [data-testid="stSelectbox"] ul[role="listbox"],
        [data-testid="stSelectbox"] div[role="listbox"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
            border: 1px solid {border} !important;
        }}
        [data-testid="stSelectbox"] ul[role="listbox"] li,
        [data-testid="stSelectbox"] div[role="listbox"] li,
        [data-testid="stSelectbox"] div[role="listbox"] div[role="option"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
        }}
        [data-testid="stSelectbox"] ul[role="listbox"] li:hover,
        [data-testid="stSelectbox"] div[role="listbox"] li:hover,
        [data-testid="stSelectbox"] div[role="listbox"] div[role="option"]:hover {{
            background-color: {option_hover_bg} !important;
            color: {input_fg} !important;
        }}

        /* ▼▼ 全プルダウン共通（保険で広めに指定） ▼▼ */
        .stApp ul[role="listbox"],
        .stApp div[role="listbox"],
        .stApp [data-baseweb="menu"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
        }}
        .stApp ul[role="listbox"] li,
        .stApp div[role="listbox"] li,
        .stApp ul[role="listbox"] div[role="option"],
        .stApp div[role="listbox"] div[role="option"],
        .stApp [data-baseweb="menu"] li,
        .stApp [data-baseweb="menu"] div[role="option"] {{
            background-color: {option_bg} !important;
            color: {input_fg} !important;
        }}
        .stApp ul[role="listbox"] li:hover,
        .stApp div[role="listbox"] li:hover,
        .stApp ul[role="listbox"] div[role="option"]:hover,
        .stApp div[role="listbox"] div[role="option"]:hover,
        .stApp [data-baseweb="menu"] li:hover,
        .stApp [data-baseweb="menu"] div[role="option"]:hover {{
            background-color: {option_hover_bg} !important;
            color: {input_fg} !important;
        }}

        /* Expander（地点の指定・登録）のヘッダー部分 */
        [data-testid="stExpander"] details > summary {{
            background-color: {exp_header_bg} !important;
            color: {fg} !important;
            border: 1px solid {border} !important;
            border-radius: 0.75rem !important;
            padding: 0.4rem 0.8rem !important;
            font-weight: 600 !important;
        }}
        [data-testid="stExpander"] details > summary svg {{
            stroke: {fg} !important;
        }}
        [data-testid="stExpander"] details > summary svg * {{
            stroke: {fg} !important;
        }}

        /* ▼▼ 緯度・経度（st.number_input）の +/- ボタン配色 ▼▼ */
        [data-testid="stNumberInput"] button {{
            background-color: {btn_bg} !important;
            color: {btn_fg} !important;
            border: 1px solid {border} !important;
            border-radius: 0.4rem !important;
        }}
        [data-testid="stNumberInput"] button:hover {{
            background-color: {btn_hover} !important;
            color: {btn_fg} !important;
        }}
        [data-testid="stNumberInput"] button svg,
        [data-testid="stNumberInput"] button svg * {{
            stroke: {btn_fg} !important;
        }}

        /* ヘッダー／フッターは非表示のまま */
        [data-testid="stHeader"], header, footer {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def init_session_state() -> None:
    # 初期地点（日本の真ん中あたり）
    st.session_state.setdefault("lat", 36.0)
    st.session_state.setdefault("lon", 138.0)
    st.session_state.setdefault("place_name", "初期地点（日本付近）")
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("trigger_fetch", False)
    st.session_state.setdefault("data", None)
    st.session_state.setdefault("metadata", [])
    st.session_state.setdefault("saved_locations", [])
    st.session_state.setdefault("save_label", "")
    st.session_state.setdefault("selected_saved", None)
    st.session_state.setdefault("layer_data", None)
    st.session_state.setdefault("layer_model", None)
    st.session_state.setdefault("model_diagnostics", None)
    st.session_state.setdefault("selected_models", [])
    st.session_state.setdefault("multiselect_models", [])  # モデル選択用 UI 値
    st.session_state.setdefault("map_zoom", 6)             # 地図のズームレベル（初期は 6）


# ジオコーディング
_GEOLocator = Nominatim(user_agent="cloud-viewer-app", timeout=5)


def geocode_place(text: str) -> Optional[Tuple[float, float, str]]:
    """地名 or 'lat, lon' 文字列 → (lat, lon, place_name)"""
    text = text.strip()
    if not text:
        return None

    # "lat, lon" 形式
    if "," in text:
        try:
            lat_str, lon_str = [t.strip() for t in text.split(",", 1)]
            lat_val = float(lat_str)
            lon_val = float(lon_str)
            loc = _GEOLocator.reverse(f"{lat_val}, {lon_val}", language="ja")
            name = loc.address if loc else f"{lat_val:.4f}, {lon_val:.4f}"
            return lat_val, lon_val, name
        except Exception:
            pass

    # 地名
    try:
        loc = _GEOLocator.geocode(text, language="ja")
        if not loc:
            return None
        return loc.latitude, loc.longitude, loc.address
    except Exception:
        return None


def reverse_geocode(lat: float, lon: float) -> str:
    """座標 → 地名（失敗したら座標文字列）"""
    try:
        loc = _GEOLocator.reverse(f"{lat}, {lon}", language="ja")
        if loc and loc.address:
            return loc.address
    except Exception:
        pass
    return f"{lat:.4f}, {lon:.4f}"


def normalize_cloud(series: pd.Series) -> pd.Series:
    """雲量を 0〜100% に正規化"""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s

    unique = s.dropna().unique()
    # 0/1 バイナリ
    if len(unique) <= 2 and set(unique).issubset({0, 1}):
        return s * 100.0

    max_val = s.max()
    # 0〜1 小数
    if max_val <= 1.0000001:
        return s * 100.0

    # 既に 0〜100 とみなす
    return s


def filter_next_hours(df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    """現在時刻から hours 時間先までのレコードに絞る"""
    if "time" not in df.columns or df.empty:
        return df

    now = pd.Timestamp.now()
    end = now + timedelta(hours=hours)
    mask = (df["time"] >= now) & (df["time"] <= end)
    filtered = df.loc[mask].copy()
    return filtered


# =========================
# Open-Meteo 取得
# =========================

def fetch_forecast_single_model(
    lat: float, lon: float, model_code: str
) -> Tuple[pd.DataFrame, ModelMeta]:
    """1 モデル分の総雲量を取得し DataFrame を返す"""
    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "hourly": "cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
        "forecast_days": 7,
        "timezone": "auto",
        "models": model_code,
    }

    r = requests.get(API_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    total_raw = pd.Series(hourly.get("cloudcover", []))
    low_raw = pd.Series(hourly.get("cloudcover_low", []))
    mid_raw = pd.Series(hourly.get("cloudcover_mid", []))
    high_raw = pd.Series(hourly.get("cloudcover_high", []))

    # 総雲量を正規化
    total = normalize_cloud(total_raw)

    # 「0/1 しかない」or 欠損が多いときは層別の max から補完
    uniq_raw = total_raw.dropna().unique()
    is_binary_0_1 = len(uniq_raw) <= 2 and set(uniq_raw).issubset({0, 1})
    if is_binary_0_1 or total.dropna().empty:
        low = normalize_cloud(low_raw)
        mid = normalize_cloud(mid_raw)
        high = normalize_cloud(high_raw)
        total = pd.concat([low, mid, high], axis=1).max(axis=1)

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            CODE_TO_DISPLAY.get(model_code, model_code): total,
        }
    )

    tz = data.get("timezone")
    meta = ModelMeta(
        display_name=CODE_TO_DISPLAY.get(model_code, model_code),
        code=model_code,
        rows=len(df),
        timezone=tz,
    )
    return df, meta


def load_models(lat: float, lon: float, model_display_names: List[str]) -> Tuple[pd.DataFrame, List[ModelMeta]]:
    """複数モデルの総雲量を time をキーに outer merge"""
    all_df: Optional[pd.DataFrame] = None
    metadata: List[ModelMeta] = []

    for disp in model_display_names:
        code = DISPLAY_TO_CODE.get(disp)
        if not code:
            continue
        try:
            df_m, meta = fetch_forecast_single_model(lat, lon, code)
        except Exception:
            # 失敗したモデルはスキップ（メタに rows=0 で入れておく）
            meta = ModelMeta(display_name=disp, code=code, rows=0, timezone=None)
            metadata.append(meta)
            continue

        metadata.append(meta)
        if all_df is None:
            all_df = df_m
        else:
            all_df = pd.merge(all_df, df_m, on="time", how="outer")

    if all_df is None:
        all_df = pd.DataFrame(columns=["time"])

    all_df.sort_values("time", inplace=True)
    all_df.reset_index(drop=True, inplace=True)
    all_df = filter_next_hours(all_df, hours=48)

    return all_df, metadata


def fetch_layered_forecast(
    lat: float, lon: float, model_code: str
) -> pd.DataFrame:
    """層別雲量（総・下層・中層・上層）を取得して 48h に絞る"""
    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "hourly": "cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high",
        "forecast_days": 7,
        "timezone": "auto",
        "models": model_code,
    }

    r = requests.get(API_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    total_raw = pd.Series(hourly.get("cloudcover", []))
    low_raw = pd.Series(hourly.get("cloudcover_low", []))
    mid_raw = pd.Series(hourly.get("cloudcover_mid", []))
    high_raw = pd.Series(hourly.get("cloudcover_high", []))

    total = normalize_cloud(total_raw)
    low = normalize_cloud(low_raw)
    mid = normalize_cloud(mid_raw)
    high = normalize_cloud(high_raw)

    uniq_raw = total_raw.dropna().unique()
    is_binary_0_1 = len(uniq_raw) <= 2 and set(uniq_raw).issubset({0, 1})
    if is_binary_0_1 or total.dropna().empty:
        total = pd.concat([low, mid, high], axis=1).max(axis=1)

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "Total cloud": total,
            "Low cloud": low,
            "Mid cloud": mid,
            "High cloud": high,
        }
    )

    df = filter_next_hours(df, hours=48)
    return df


# =========================
# グラフ生成（Altair）
# =========================

def prepare_chart_data(timeseries: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    cols = ["time"] + [m for m in models if m in timeseries.columns]
    df = timeseries[cols].copy()
    melted = df.melt("time", var_name="model", value_name="cloud_cover")
    return melted


def build_line_chart(melted: pd.DataFrame) -> alt.Chart:
    """モデル比較（凡例をグラフ下部に配置）"""
    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "time:T",
                title="時刻",
                axis=alt.Axis(
                    labelAngle=-45,
                    format="%m/%d %H:%M",
                    tickCount=12,
                ),
            ),
            y=alt.Y(
                "cloud_cover:Q",
                title="雲量 (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "model:N",
                title="モデル",
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    labelLimit=180,
                ),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="時刻", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("model:N", title="モデル"),
                alt.Tooltip("cloud_cover:Q", title="雲量", format=".1f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    return chart


def prepare_layer_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["time", "Total cloud", "Low cloud", "Mid cloud", "High cloud"]
    df = df[cols].copy()
    melted = df.melt("time", var_name="layer", value_name="cloud_cover")
    return melted


def build_layer_chart(melted: pd.DataFrame, title_suffix: str) -> alt.Chart:
    """層別雲量グラフ（凡例も下部に配置）"""
    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "time:T",
                title="時刻",
                axis=alt.Axis(
                    labelAngle=-45,
                    format="%m/%d %H:%M",
                    tickCount=12,
                ),
            ),
            y=alt.Y(
                "cloud_cover:Q",
                title="雲量 (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "layer:N",
                title="層",
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    labelLimit=180,
                ),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="時刻", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("layer:N", title="層"),
                alt.Tooltip("cloud_cover:Q", title="雲量", format=".1f"),
            ],
        )
        .properties(
            height=420,
            title=f"層別雲量（{title_suffix}）",
        )
        .interactive()
    )
    return chart


# =========================
# 地点保存・読み込み
# =========================

def save_current_location(name: str) -> None:
    locs: List[Dict] = st.session_state.get("saved_locations", [])
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    place_name = st.session_state.get("place_name", "")

    # 空なら自動名称
    if not name:
        name = f"地点 {len(locs) + 1}"

    # 上書き or 追加
    replaced = False
    for loc in locs:
        if loc.get("name") == name:
            loc.update({"lat": lat, "lon": lon, "place_name": place_name})
            replaced = True
            break

    if not replaced:
        locs.append({"name": name, "lat": lat, "lon": lon, "place_name": place_name})
        # 最大 20 件
        if len(locs) > 20:
            locs = locs[-20:]

    st.session_state["saved_locations"] = locs
    save_json(SAVED_LOCATIONS_PATH, locs)


def render_saved_locations_ui() -> None:
    st.subheader("登録済み地点", anchor=False)
    locs: List[Dict] = st.session_state.get("saved_locations", [])
    if not locs:
        st.caption("※ まだ登録された地点はありません。")
        return

    names = [loc["name"] for loc in locs]
    idx = 0
    if st.session_state.get("selected_saved") in names:
        idx = names.index(st.session_state["selected_saved"])

    selected_name = st.selectbox(
        "登録済み地点を選択",
        names,
        index=idx,
        key="selected_saved",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("選択した地点を呼び出す"):
            for loc in locs:
                if loc["name"] == selected_name:
                    st.session_state["lat"] = loc["lat"]
                    st.session_state["lon"] = loc["lon"]
                    st.session_state["place_name"] = loc.get("place_name", selected_name)
                    st.session_state["map_zoom"] = 13       # 呼び出し時にズームアップ
                    st.session_state["trigger_fetch"] = True
                    st.success(f"地点「{selected_name}」を反映しました。")
                    break
    with col2:
        if st.button("選択した地点を削除する"):
            new_locs = [loc for loc in locs if loc["name"] != selected_name]
            st.session_state["saved_locations"] = new_locs
            save_json(SAVED_LOCATIONS_PATH, new_locs)
            st.success(f"地点「{selected_name}」を削除しました。")

    # 一覧テーブル
    df = pd.DataFrame(locs)
    if not df.empty:
        df["lat"] = df["lat"].map(lambda v: f"{v:.4f}")
        df["lon"] = df["lon"].map(lambda v: f"{v:.4f}")
        st.dataframe(df, use_container_width=True, hide_index=True)

    # JSON エクスポート
    export_json = json.dumps(locs, ensure_ascii=False, indent=2)
    st.download_button(
        "JSON エクスポート（saved_locations.json）",
        data=export_json.encode("utf-8"),
        file_name="saved_locations.json",
        mime="application/json",
    )

    # JSON インポート
    uploaded = st.file_uploader("JSON インポート", type=["json"], key="loc_json_uploader")
    if uploaded is not None:
        if st.button("JSON をインポートしてマージ"):
            try:
                imported = json.loads(uploaded.read().decode("utf-8"))
                if not isinstance(imported, list):
                    raise ValueError
                merged = locs + imported
                # name, lat, lon, place_name を持つものだけ
                cleaned = []
                seen = set()
                for item in merged:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    lat = item.get("lat")
                    lon = item.get("lon")
                    if name is None or lat is None or lon is None:
                        continue
                    key = (name, float(lat), float(lon))
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(
                        {
                            "name": name,
                            "lat": float(lat),
                            "lon": float(lon),
                            "place_name": item.get("place_name", name),
                        }
                    )
                if len(cleaned) > 20:
                    cleaned = cleaned[-20:]
                st.session_state["saved_locations"] = cleaned
                save_json(SAVED_LOCATIONS_PATH, cleaned)
                st.success("インポートとマージが完了しました。")
            except Exception:
                st.error("JSON の形式が不正です。")


# =========================
# プリセット設定 UI
# =========================

def load_config_to_state() -> None:
    """起動時に設定ファイルから選択モデル＆プリセットを読み込み"""
    config = load_json(CONFIG_PATH, {})
    st.session_state["user_presets"] = config.get("presets", [])

    sel_models = config.get("selected_models")
    if sel_models:
        filtered = [m for m in sel_models if m in DISPLAY_TO_CODE]
    else:
        filtered = DEFAULT_PRESETS["星空観測メイン"]

    st.session_state["selected_models"] = filtered
    # multiselect も同期（まだ UI は描画していないので安全）
    if not st.session_state.get("multiselect_models"):
        st.session_state["multiselect_models"] = filtered


def save_config_from_state() -> None:
    config = {
        "selected_models": st.session_state.get("selected_models", []),
        "presets": st.session_state.get("user_presets", []),
    }
    save_json(CONFIG_PATH, config)


def _apply_preset(models: List[str], message: str) -> None:
    """プリセット適用時の共通処理（状態同期＋再取得フラグON）"""
    filtered = [m for m in models if m in DISPLAY_TO_CODE]
    if not filtered:
        return
    # multiselect_models は「次回 multiselect 描画時の default」として使う
    st.session_state["selected_models"] = filtered
    st.session_state["multiselect_models"] = filtered
    save_config_from_state()
    st.session_state["trigger_fetch"] = True
    st.success(message)


def render_presets_ui() -> None:
    st.subheader("比較モード・プリセット", anchor=False)

    # おすすめプリセット一覧を説明付きで表示
    st.caption("■ おすすめプリセットの内容")
    st.markdown(
        f"- **星空観測メイン**: {', '.join(DEFAULT_PRESETS['星空観測メイン'])}"
    )
    st.markdown(
        f"- **高速チェック（軽量）**: {', '.join(DEFAULT_PRESETS['高速チェック（軽量）'])}"
    )
    st.markdown(
        f"- **全球モデル比較**: {', '.join(DEFAULT_PRESETS['全球モデル比較'])}"
    )
    st.markdown(
        f"- **ALL（全モデル）**: {', '.join(DEFAULT_PRESETS['ALL'])}"
    )

    st.markdown("---")

    # おすすめプリセットボタン
    st.caption("■ おすすめプリセット（ワンタップ適用）")
    c1, c2, c3 = st.columns(3)
    if c1.button("星空観測メイン"):
        _apply_preset(DEFAULT_PRESETS["星空観測メイン"], "「星空観測メイン」を適用しました。")
    if c2.button("高速チェック"):
        _apply_preset(DEFAULT_PRESETS["高速チェック（軽量）"], "「高速チェック（軽量）」を適用しました。")
    if c3.button("全球モデル比較"):
        _apply_preset(DEFAULT_PRESETS["全球モデル比較"], "「全球モデル比較」を適用しました。")

    # ALL プリセットボタン
    st.write("")
    if st.button("ALL（全モデル）を適用"):
        _apply_preset(DEFAULT_PRESETS["ALL"], "ALL（全モデル）を適用しました。")

    st.markdown("---")

    # ユーザー定義プリセット
    st.caption("■ ユーザー定義プリセット")
    presets: List[Dict] = st.session_state.get("user_presets", [])
    preset_names = [p["name"] for p in presets] if presets else []

    if preset_names:
        selected_idx = 0
        selected_preset_name = st.selectbox(
            "プリセット一覧",
            preset_names,
            index=selected_idx,
            key="preset_select",
        )

        # 選択中プリセットの中身を表示
        for p in presets:
            if p["name"] == selected_preset_name:
                st.caption("このプリセットに含まれるモデル： " + ", ".join(p.get("models", [])))
                break

        col1, col2 = st.columns(2)
        with col1:
            if st.button("プリセットを読み込む"):
                for p in presets:
                    if p["name"] == selected_preset_name:
                        _apply_preset(
                            p.get("models", []),
                            f"プリセット「{selected_preset_name}」を適用しました。",
                        )
                        break
        with col2:
            if st.button("選択中のプリセットを削除"):
                new_presets = [p for p in presets if p["name"] != selected_preset_name]
                st.session_state["user_presets"] = new_presets
                save_config_from_state()
                st.success(f"プリセット「{selected_preset_name}」を削除しました。")

    # 新規保存
    st.markdown("#### 現在のモデル選択をプリセットとして保存")
    new_name = st.text_input("新しく保存 / 上書きするプリセット名", key="preset_new_name")
    if st.button("プリセットを保存 / 上書き"):
        if not new_name.strip():
            st.error("プリセット名を入力してください。")
        else:
            # 現在の UI の選択状態から保存
            models = st.session_state.get("multiselect_models") or st.session_state.get("selected_models", [])
            if not models:
                st.error("現在のモデル選択が空です。")
            else:
                presets = st.session_state.get("user_presets", [])
                # 上書き or 追加
                replaced = False
                for p in presets:
                    if p["name"] == new_name:
                        p["models"] = list(models)
                        replaced = True
                        break
                if not replaced:
                    presets.append({"name": new_name, "models": list(models)})
                    if len(presets) > 20:
                        presets = presets[-20:]
                st.session_state["user_presets"] = presets
                save_config_from_state()
                st.success(f"プリセット「{new_name}」を保存しました。")


# =========================
# コントロールパネル
# =========================

def render_control_panel() -> None:
    st.subheader("地点の指定", anchor=False)
    st.caption("地名／住所 または '緯度, 経度' を入力して検索できます。")

    # 地名／座標入力
    q = st.text_input("地名/住所 または '緯度, 経度'", key="place_query")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("地名/座標から検索"):
            result = geocode_place(q)
            if result is None:
                st.error("ジオコーディングに失敗しました。表記を変えて再度お試しください。")
            else:
                lat, lon, name = result
                st.session_state["lat"] = lat
                st.session_state["lon"] = lon
                st.session_state["place_name"] = name
                st.session_state["map_zoom"] = 13      # 検索で地点更新時にズームアップ
                st.session_state["trigger_fetch"] = True
                st.success(f"地点を更新しました：{name}")
    with col2:
        # GPS (Optional)
        if streamlit_geolocation is not None:
            if st.button("📍 GPS で取得"):
                try:
                    loc = streamlit_geolocation()
                    if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                        lat = float(loc["latitude"])
                        lon = float(loc["longitude"])
                        st.session_state["lat"] = lat
                        st.session_state["lon"] = lon
                        name = reverse_geocode(lat, lon)
                        st.session_state["place_name"] = name
                        st.session_state["map_zoom"] = 13   # GPS 取得時もズームアップ
                        st.session_state["trigger_fetch"] = True
                        st.success("現在地を反映しました。")
                    else:
                        st.error("現在地が取得できませんでした。位置情報の権限を確認してください。")
                except Exception as e:
                    st.error(f"GPS 取得中にエラーが発生しました: {e}")
        else:
            st.caption("※ GPS 機能を使うには `pip install streamlit-geolocation` が必要です。")

    # 緯度・経度手動入力
    st.markdown("#### 緯度・経度（手動調整）")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_val = st.number_input(
            "緯度 (Latitude)",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state["lat"]),
            step=0.0001,
            format="%.4f",
        )
    with col_lon:
        lon_val = st.number_input(
            "経度 (Longitude)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state["lon"]),
            step=0.0001,
            format="%.4f",
        )

    if st.button("この地点の雲量を取得"):
        st.session_state["lat"] = float(lat_val)
        st.session_state["lon"] = float(lon_val)
        st.session_state["place_name"] = reverse_geocode(lat_val, lon_val)
        st.session_state["map_zoom"] = 13              # 手動座標指定時もズームアップ
        st.session_state["trigger_fetch"] = True
        st.success("指定した座標で取得します。")

    st.markdown("---")

    # 地点の保存
    st.subheader("地点の登録", anchor=False)
    save_label = st.text_input("登録名（空の場合は自動で命名）", key="save_label")
    if st.button("現在の地点を登録"):
        save_current_location(save_label)
        st.success("現在の地点を登録しました。")

    # 保存済み地点 UI
    render_saved_locations_ui()


# =========================
# 地図（タイトル直下に表示）
# =========================

def render_map_and_click() -> None:
    """folium 地図で地点選択（タイトル直下で使用）"""
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]

    # 地図タイル：常にライトモード相当（OpenStreetMap）を使用
    tiles = "OpenStreetMap"

    zoom = st.session_state.get("map_zoom", 6)

    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles)
    folium.Marker(
        [lat, lon],
        tooltip="現在の地点",
        icon=folium.Icon(color="red", icon="cloud"),
    ).add_to(m)

    # スマホ向けに高さ小さめ
    map_data = st_folium(
        m,
        width=None,
        height=260,
        key="map",
    )

    last_clicked = map_data.get("last_clicked") if map_data else None
    if last_clicked:
        clicked_lat = float(last_clicked.get("lat"))
        clicked_lon = float(last_clicked.get("lng"))
        prev = st.session_state.get("last_click")
        # クリック位置が変わったら更新
        if not prev or (abs(prev[0] - clicked_lat) > 1e-6 or abs(prev[1] - clicked_lon) > 1e-6):
            st.session_state["last_click"] = (clicked_lat, clicked_lon)
            st.session_state["lat"] = clicked_lat
            st.session_state["lon"] = clicked_lon
            st.session_state["place_name"] = reverse_geocode(clicked_lat, clicked_lon)
            st.session_state["map_zoom"] = 13       # 地図クリック時もズームアップ
            st.session_state["trigger_fetch"] = True
            st.rerun()


# =========================
# 比較モードタブ
# =========================

def render_compare_tab() -> None:
    st.markdown("### 🔍 比較モード（総雲量）")
    st.caption("※ 上の地図で地点を選んでから、比較するモデルを設定してください。")
    st.markdown("---")

    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    all_models = [m["display_name"] for m in MODEL_INFOS]

    # 先にプリセット UI
    with st.expander("比較モードのプリセット設定（タップして開く）", expanded=False):
        render_presets_ui()

    # モデル選択（マルチセレクト）
    st.subheader("比較するモデルの選択", anchor=False)

    # セッションに値がなければデフォルトを設定
    if not st.session_state.get("multiselect_models"):
        base = st.session_state.get("selected_models") or DEFAULT_PRESETS["星空観測メイン"]
        st.session_state["multiselect_models"] = [m for m in base if m in all_models]

    selected_models = st.multiselect(
        "モデルを選択",
        options=all_models,
        default=st.session_state["multiselect_models"],
        key="multiselect_models",
        help="複数選択して総雲量を比較します。",
    )

    # UI → 内部状態へ同期
    st.session_state["selected_models"] = selected_models
    save_config_from_state()

    # データ取得トリガ
    if st.session_state.get("trigger_fetch"):
        if not selected_models:
            st.warning("少なくとも 1 つのモデルを選択してください。")
        else:
            with st.spinner("Open-Meteo から雲量データ取得中..."):
                df, metadata = load_models(lat, lon, selected_models)
                st.session_state["data"] = df
                st.session_state["metadata"] = [m.__dict__ for m in metadata]
        st.session_state["trigger_fetch"] = False

    # 48 時間の雲量推移グラフ
    data: Optional[pd.DataFrame] = st.session_state.get("data")
    if data is None or data.empty or len(data.columns) <= 1:
        st.info("データがまだありません。「地図クリック」や「地点の指定」→ モデル選択後に取得されます。")
        return

    available_models = [c for c in data.columns if c != "time"]
    show_models = [m for m in selected_models if m in available_models] or available_models

    st.subheader("直近 48 時間の雲量推移", anchor=False)
    st.caption("下のチェックボックスで表示するモデルを絞り込めます。")

    show_models = st.multiselect(
        "グラフに表示するモデル",
        options=available_models,
        default=show_models,
        key="chart_models",
    )

    if not show_models:
        st.warning("少なくとも 1 つのモデルを選択してください。")
    else:
        melted = prepare_chart_data(data, show_models)
        chart = build_line_chart(melted)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### 詳細データ（時間別・モデル別）")
    st.dataframe(
        data[["time"] + available_models],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### モデル別データ状況")
    meta_list = st.session_state.get("metadata", [])
    if meta_list:
        meta_df = pd.DataFrame(meta_list)
        meta_df = meta_df[meta_df["display_name"].isin(available_models)]
        st.dataframe(meta_df, use_container_width=True, hide_index=True)
    else:
        st.caption("※ メタデータは取得されていません。")


# =========================
# モデル別タブ（層別雲量）
# =========================

def render_layer_tab() -> None:
    st.markdown("### 📊 モデル別の層別雲量")

    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    place_name = st.session_state.get("place_name", "")
    st.caption(f"現在の地点：{place_name}（{lat:.4f}, {lon:.4f}）")

    all_models = [m["display_name"] for m in MODEL_INFOS]
    default_models = st.session_state.get("selected_models") or DEFAULT_PRESETS["星空観測メイン"]
    default_first = next((m for m in default_models if m in all_models), all_models[0])

    selected_model = st.selectbox(
        "対象モデル",
        options=all_models,
        index=all_models.index(default_first),
        key="layer_model_select",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("選択した地点とモデルの雲量を表示"):
            code = DISPLAY_TO_CODE.get(selected_model)
            if code:
                with st.spinner("層別雲量を取得中..."):
                    df_layer = fetch_layered_forecast(lat, lon, code)
                    st.session_state["layer_data"] = df_layer
                    st.session_state["layer_model"] = selected_model

    with col2:
        if st.button("この地点で全モデル検証＆JSON出力"):
            results = []
            with st.spinner("全モデルの層別雲量を簡易チェック中..."):
                for disp in all_models:
                    code = DISPLAY_TO_CODE.get(disp)
                    if not code:
                        continue
                    try:
                        df_layer = fetch_layered_forecast(lat, lon, code)
                        if df_layer.empty:
                            results.append(
                                {
                                    "model_display": disp,
                                    "model_code": code,
                                    "rows": 0,
                                    "time_start": None,
                                    "time_end": None,
                                    "error": None,
                                }
                            )
                        else:
                            results.append(
                                {
                                    "model_display": disp,
                                    "model_code": code,
                                    "rows": len(df_layer),
                                    "time_start": df_layer["time"].min().isoformat(),
                                    "time_end": df_layer["time"].max().isoformat(),
                                    "error": None,
                                }
                            )
                    except Exception as e:
                        results.append(
                            {
                                "model_display": disp,
                                "model_code": code,
                                "rows": 0,
                                "time_start": None,
                                "time_end": None,
                                "error": str(e),
                            }
                        )

            st.session_state["model_diagnostics"] = results
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                "model_diagnostics.json をダウンロード",
                data=json_str.encode("utf-8"),
                file_name="model_diagnostics.json",
                mime="application/json",
            )

    # 層別雲量グラフ
    df_layer: Optional[pd.DataFrame] = st.session_state.get("layer_data")
    layer_model_name: Optional[str] = st.session_state.get("layer_model")

    if df_layer is None or df_layer.empty:
        st.info("層別雲量はまだ表示されていません。「選択した地点とモデルの雲量を表示」を押してください。")
        return

    melted = prepare_layer_chart_data(df_layer)
    title_suffix = layer_model_name or selected_model
    chart = build_layer_chart(melted, title_suffix=title_suffix)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### 層別の詳細データ")
    st.dataframe(df_layer, use_container_width=True, hide_index=True)


# =========================
# メイン
# =========================

def main():
    st.set_page_config(
        page_title="雲量比較（Open-Meteo）",
        page_icon="☁️",
        layout="wide",
    )

    init_session_state()
    load_config_to_state()
    apply_theme_css()  # ダークモード CSS を一括で適用

    st.title("雲量比較ビューア")
    st.caption("Open-Meteo の複数モデルで直近 48 時間の雲量を比較します。スマホ表示前提の簡易ビューアです。")

    # タイトル直下に地図（ライトタイル）
    st.caption("地図をタップして地点を選択できます。")
    render_map_and_click()

    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    place_name = st.session_state.get("place_name", "")
    st.caption(f"現在の地点：{place_name}（{lat:.4f}, {lon:.4f}）")

    # その下に「地点の指定・登録」エクスパンダ
    with st.expander("地点の指定・登録（タップで開閉）", expanded=True):
        render_control_panel()

    # さらに下にタブ
    tab1, tab2 = st.tabs(["🔍 比較モード", "📊 モデル別グラフ"])
    with tab1:
        render_compare_tab()
    with tab2:
        render_layer_tab()


if __name__ == "__main__":
    main()
