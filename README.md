# 雲量比較ダッシュボード

Open-Meteo API を利用し、ECMWF / GFS / ICON / JMA GSM / JMA MSM の各モデルが予測する雲量を比較表示する Streamlit アプリです。

## 主な機能
- 地図クリックまたは緯度・経度入力で地点を指定
- Open-Meteo モデルコード（例: `ecmwf_ifs025`）で 48 時間の雲量を取得
- Altair ラインチャートで雲量推移を描画（線を太めに設定）
- モデル別のデータ件数やタイムゾーンをテーブル表示
- Nominatim による逆ジオコーディングで地名を複数行表示

## セットアップ
```bash
pip install -r requirements.txt
```

## 実行方法
```bash
streamlit run app.py
```
実行後、ブラウザで `http://localhost:8501` を開いてください。

## 利用モデル
| モデル名 | Open-Meteo コード | 説明 |
| --- | --- | --- |
| ECMWF IFS 0.25° | `ecmwf_ifs025` | ECMWF の全球モデル。広域の雲量傾向をつかみやすい。 |
| NOAA GFS 0.25° | `gfs_seamless` | 米国 NOAA の全球予報システム。世界中の雲量・風を広く扱う。 |
| ICON Global 0.25° | `icon_global` | ドイツ気象庁（DWD）の高解像度全球モデル。 |
| JMA GSM 20km | `jma_gsm` | 気象庁 GSM。約 20km メッシュで数日先の傾向を把握できる。 |
| JMA MSM 5km | `jma_msm` | 気象庁 MSM。5km メッシュで日本域の短期予測が得意。 |

Streamlit Community Cloud へのデプロイ時は `requirements.txt` に `geopy` など必要な依存関係が含まれていることを確認してください。
