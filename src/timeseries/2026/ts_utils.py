"""
NODASS 2026 — time series workshop shared helpers.

設計原則
--------
1. 每個資料載入函式都「先試線上、失敗就用本機快取」(network-first, fallback to
   ./data)，這樣課堂網路不穩時不會整堂卡住。
2. 函式小、單一職責，方便對應未來 Jupyter notebook 的一個 cell。
3. 相容 pandas 3.0 / numpy 2.x / Python 3.13。

資料來源
--------
- NOAA PSL Niño 3.4 月距平: https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data
- ODB MHW API: https://eco.odb.ntu.edu.tw/api/mhw
- 龍洞浮標 (CWA / 國海院 NAMR 提供)
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"

NOAA_NINO34_URL = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
ODB_MHW_API = "https://eco.odb.ntu.edu.tw/api/mhw"


# ---------------------------------------------------------------------------
# 1. Niño 3.4 SST 距平 (NOAA)：線上優先、失敗用本機快取
# ---------------------------------------------------------------------------
def _parse_noaa_nino34(text: str) -> pd.DataFrame:
    """把 NOAA 純文字格式 (每行: year 12個月值) 轉成月頻 Series。"""
    rows = []
    for line in text.strip().split("\n")[1:]:  # 跳過第一行表頭
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue  # 檔尾的缺值說明行
        for i, val in enumerate(parts[1:13]):
            if "-99.99" in val:  # 缺值
                break
            rows.append((pd.Timestamp(year=year, month=i + 1, day=15), float(val)))
    return pd.DataFrame(rows, columns=["date", "ssta"]).set_index("date")


def load_noaa_nino34(start="1950", end="2025", timeout=20) -> pd.DataFrame:
    """
    回傳 NOAA Niño 3.4 月距平 DataFrame(index=date, 欄位 'ssta')。
    先嘗試線上抓取，失敗則讀 data/noaa_nino34_anom.csv。
    """
    cache = DATA_DIR / "noaa_nino34_anom.csv"
    df = None
    try:
        resp = requests.get(NOAA_NINO34_URL, timeout=timeout)
        resp.raise_for_status()
        df = _parse_noaa_nino34(resp.text)
        df.to_csv(cache)  # 更新快取
    except Exception as exc:  # noqa: BLE001 — 課堂用，任何網路錯誤都退回快取
        warnings.warn(f"NOAA 線上抓取失敗 ({exc})，改用本機快取 {cache.name}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)

    y0, y1 = int(str(start)[:4]), int(str(end)[:4])
    return df.loc[f"{y0}":f"{y1}"]


# ---------------------------------------------------------------------------
# 2. Niño 3.4 區域平均 SST / 距平 (由 ODB MHW API 預先彙整，存成本機 CSV)
# ---------------------------------------------------------------------------
def load_nino34_sst() -> pd.Series:
    """Niño 3.4 區域月平均「原始」海表溫 (含季節循環)。index=date。"""
    df = pd.read_csv(DATA_DIR / "nino34_sst_1982-2024.csv",
                     index_col=0, parse_dates=True)
    return df["sst"]


def load_nino34_anomaly_odb() -> pd.Series:
    """Niño 3.4 區域月平均 SST 距平 (來自 ODB MHW API)。index=date。"""
    df = pd.read_csv(DATA_DIR / "nino34_mean_1982-2024.csv",
                     index_col=0, parse_dates=True)
    return df.iloc[:, 0].rename("sst_anomaly")


# ---------------------------------------------------------------------------
# 3. 距平 (anomaly) = 觀測 − 月氣候平均：本工作坊的核心觀念
# ---------------------------------------------------------------------------
def monthly_climatology(series: pd.Series, base=("1982", "2011")) -> pd.Series:
    """
    由一段「基準期 (climatology base period)」算 12 個月的長期月平均。
    回傳 index 為月份 (1..12) 的 Series。
    """
    base_slice = series.loc[base[0]:base[1]]
    return base_slice.groupby(base_slice.index.month).mean()


def to_anomaly(series: pd.Series, clim: pd.Series | None = None,
               base=("1982", "2011")) -> pd.Series:
    """把原始序列轉成距平：每個月減掉它所屬月份的長期平均。"""
    if clim is None:
        clim = monthly_climatology(series, base=base)
    return series - series.index.month.map(clim)


# ---------------------------------------------------------------------------
# 4. 教學用模擬訊號 (有已知的趨勢 + 季節 + 噪音，方便驗證 STL 拆得對不對)
# ---------------------------------------------------------------------------
def fetch_mhw_data(lon0, lat0, lon1, lat1, start, end, timeout=40) -> pd.DataFrame:
    """直接打 ODB MHW API，回傳逐格點 DataFrame。API 手冊見 ODB_MHW_API。"""
    params = dict(lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1,
                  start=start, end=end, append="sst,sst_anomaly,level")
    resp = requests.get(ODB_MHW_API, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def load_enso_map(date="2015-12") -> pd.DataFrame:
    """
    太平洋 SST 距平格點 (跨換日線)。線上優先 (分兩段抓避開 API bbox 限制)，
    失敗則讀 data/enso_map_{date}.csv。回傳含 lon/lat/sst_anomaly 的 DataFrame。
    """
    cache = DATA_DIR / f"enso_map_{date}.csv"
    start = f"{date}-01"
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    lat0, lat1 = -25, 25
    try:
        # API 的 lon0..lon1 無法直接跨換日線，所以分東、西兩段抓再接起來
        west = fetch_mhw_data(135, lat0, 179.999, lat1, start, end)
        east = fetch_mhw_data(-179.999, lat0, -60, lat1, start, end)
        df = pd.concat([west, east], ignore_index=True)
        df = df[["lon", "lat", "sst_anomaly"]]   # 只留繪圖需要的欄位，快取較小
        df.to_csv(cache, index=False)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ODB API 抓取失敗 ({exc})，改用本機快取 {cache.name}")
        df = pd.read_csv(cache)
    return df


def load_cholera() -> pd.DataFrame:
    """坦尚尼亞年霍亂病例 (已去趨勢)。欄位 year, detrended_cholera_cases。"""
    return pd.read_csv(DATA_DIR / "tanzania_cholera_detrended.csv")


# ---------------------------------------------------------------------------
# 5. 龍洞浮標 + 湧升指數 (Upwelling Index, UI)
# ---------------------------------------------------------------------------
CWA_SITES = {"Longdong": (25.0969, 121.9222)}  # (lat, lon)

# 物理常數 (Huang et al. 2021, https://www.mdpi.com/2072-4292/13/2/170)
_RHO_AIR = 1.22     # 空氣密度 kg/m^3
_RHO_W = 1026.0     # 海水密度 kg/m^3


def load_buoy(site="Longdong") -> pd.DataFrame:
    """龍洞浮標時頻資料 (欄位 site, Wind, Wind_Dir, SST, Hs)，index 為 datetime。"""
    df = pd.read_csv(DATA_DIR / f"buoy/cwa_buoy_{site}_2010-2024.csv",
                     index_col=0, parse_dates=True)
    return df[df["site"] == site] if "site" in df.columns else df


def _drag_coef(V):
    """風阻力係數經驗公式 (Huang et al. 2021)。"""
    return (0.8 + 0.065 * V) * 1e-3


def upwelling_index(df, site="Longdong", coast_angle=18.0) -> pd.Series:
    """
    由浮標風速/風向算「湧升指數」UI (m^2/s)：風應力沿岸分量 / (海水密度 × 科氏參數)。
    UI 為正 = 離岸 Ekman 輸送 = 有利湧升 (把底層冷水帶上來)。

    公式完全依 Huang et al. 2021 (Remote Sensing 13:170) eq 1–4：
        UI = ρa·Cd·V²·cos(α−β) / (f·ρw),  Cd=(0.8+0.065V)×1e-3,
        f = 2ω·sin(φ),  ρa=1.22, ρw=1026, ω=7.2921e-5
    論文岸線角 β：北段 18°、中段 15°、南段 20°（龍洞屬北段 → 18°）。

    Parameters
    ----------
    df : 含欄位 'Wind'(m/s)、'Wind_Dir'(度, 風『來向』) 的 DataFrame
    coast_angle : 岸線走向 β (度, 自真北順時針；台灣東北段=18°)
    """
    lat = CWA_SITES[site][0]
    V = df["Wind"]
    alpha = (df["Wind_Dir"] + 180.0) % 360.0           # 風來向 → 風去向
    cos_ab = np.cos(np.deg2rad(alpha - coast_angle))   # 投影到離岸方向
    tau = _RHO_AIR * _drag_coef(V) * V ** 2 * cos_ab    # 風應力沿岸分量
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))         # 科氏參數
    return tau / (_RHO_W * f)


def lagged_xcorr(x: pd.Series, y: pd.Series, max_lag=20):
    """
    x 與『平移後的 y』的落後互相關。lag>0 表示用『較晚的 y』對齊 x
    (即 x 領先 y)。回傳 (lags, r, p, best_lag)。先把兩者標準化 (z-score)。
    """
    from scipy.stats import pearsonr
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    lags, rs, ps = [], [], []
    for lag in range(-max_lag, max_lag + 1):
        ys = yz.shift(-lag)               # lag>0: y 往前移 → 比較『未來的 y』
        m = xz.notna() & ys.notna()
        if m.sum() > 5:
            r, p = pearsonr(xz[m], ys[m])
            lags.append(lag); rs.append(r); ps.append(p)
    rs = np.array(rs)
    best_lag = lags[int(np.argmax(np.abs(rs)))]
    return np.array(lags), rs, np.array(ps), best_lag


def make_synthetic_series(n=120, freq="MS", start="2010-01-01",
                          slope=0.04, season_amp=2.0, noise=0.5,
                          seed=2026) -> pd.DataFrame:
    """
    回傳 DataFrame(index=日期)，欄位:
      - y      : 觀測 = 趨勢 + 季節 + 噪音
      - trend  : 真正的趨勢 (ground truth)
      - season : 真正的季節
      - signal : 趨勢 + 季節 (無噪音)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range(start, periods=n, freq=freq)
    trend = 20 + slope * t
    season = season_amp * np.sin(2 * np.pi * t / 12)
    e = noise * rng.standard_normal(n)
    return pd.DataFrame(
        {"y": trend + season + e, "trend": trend, "season": season,
         "signal": trend + season},
        index=idx,
    )
