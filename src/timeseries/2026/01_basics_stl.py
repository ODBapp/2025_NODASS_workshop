# %% [markdown]
# # A. 時間序列基礎 + STL 分解
# 對應敘事文件: narrative/01_basics_stl.md
# 主軸：**訊號常藏在「殘差／距平」裡，而不是原始序列。** 這個觀念會用三種尺度反覆出現。
#
# 註：圖上文字一律用英文，以免 Colab 預設字型缺中文而變方塊；中文說明放在 .md 與註解。

# %%
# A0. 載入套件與共用工具
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

import ts_utils as ts

# %% [markdown]
# ## A1. 先用「模擬訊號」建立直覺
# 自己造一條 = 趨勢 + 季節 + 噪音 的序列，因為**知道正確答案**，
# 才能檢查待會 STL 拆得對不對。

# %%
sim = ts.make_synthetic_series(n=120, slope=0.04, season_amp=2.0, noise=0.5)

plt.figure(figsize=(12, 3))
plt.plot(sim.index, sim["y"], label="Observed  y = trend + season + noise", color="steelblue")
plt.plot(sim.index, sim["signal"], label="True signal (trend + season)", color="crimson", lw=2)
plt.title("Synthetic time series")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A2. STL 分解：把序列拆成 趨勢 / 季節 / 殘差
# STL = Seasonal-Trend decomposition using LOESS。`period=12` 表示一年 12 個月。
# `robust=True` 會降低離群值 (outlier) 的權重，讓擬合不易被極端值帶歪。

# %%
stl_sim = STL(sim["y"], period=12, robust=True).fit()
fig = stl_sim.plot()
fig.set_size_inches(12, 8)
fig.suptitle("STL decomposition of the synthetic series", y=1.00)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A3. 驗證：STL 拆出來的趨勢/季節，是否接近我們設定的真值？
# 把 STL 的 trend、seasonal 疊回造資料時的 ground truth。

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(sim.index, sim["trend"], label="True trend", color="crimson", lw=2)
axes[0].plot(sim.index, stl_sim.trend, label="STL-estimated trend", color="black", ls="--")
axes[0].set_title("Trend")
axes[0].legend(loc="upper left")

axes[1].plot(sim.index, sim["season"], label="True seasonal", color="crimson", lw=2)
axes[1].plot(sim.index, stl_sim.seasonal, label="STL-estimated seasonal", color="black", ls="--")
axes[1].set_title("Seasonal")
axes[1].legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A4. 真實資料：原始海溫 vs 距平（合併看）
# 換上真的海洋資料：**Niño 3.4 區域月平均海溫**（赤道中太平洋，判斷聖嬰/反聖嬰的關鍵區）。
#
# - 上圖：原始海溫（實線）疊上「每個月該有的長期平均」(climatology，虛線)。
#   有趣的是——赤道附近**季節變化其實很小**（虛線幾乎是平的，全年只差約 1.3°C），
#   但原始值卻上下大幅擺盪 (約 5°C)：那些大擺盪**不是季節，而是真正的異常 (ENSO)**。
# - 下圖：距平 = 原始 − 月氣候平均。把基準拉到 0 之後，就能用**一條統一門檻**
#   (±0.5°C，即 ONI) 來定義聖嬰/反聖嬰——這正是下一段 (B) 的基礎；原始海溫沒辦法這樣設門檻。

# %%
sst = ts.load_nino34_sst()
clim = ts.monthly_climatology(sst, base=("1982", "2011"))   # 12 個月長期平均
baseline = sst.index.month.map(clim)                        # 把月平均「鋪」回時間軸
anom = ts.to_anomaly(sst, clim)                             # 距平
print("季節 (climatology) 全年振幅僅 %.2f°C，原始值跨度卻有 %.2f°C"
      % (clim.max() - clim.min(), sst.max() - sst.min()))

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(sst.index, sst, color="darkorange", lw=1, label="Raw SST")
axes[0].plot(sst.index, baseline, color="gray", ls="--", lw=1.2,
             label="Monthly climatology (expected)")
axes[0].set_title("Niño 3.4 raw SST vs monthly climatology  (season is weak near the equator)")
axes[0].set_ylabel("SST (°C)")
axes[0].legend(loc="upper left")

axes[1].axhline(0, color="gray", lw=0.6)
axes[1].fill_between(anom.index, anom, 0, where=anom >= 0, color="tab:red", alpha=0.5)
axes[1].fill_between(anom.index, anom, 0, where=anom < 0, color="tab:blue", alpha=0.5)
axes[1].axhline(0.5, color="red", ls=":", lw=0.8)
axes[1].axhline(-0.5, color="blue", ls=":", lw=0.8)
axes[1].set_title("Anomaly = raw − monthly climatology   (±0.5°C dotted = ENSO threshold, next section)")
axes[1].set_ylabel("SST anomaly (°C)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A5. STL 也能做到——而且自動處理趨勢
# 直接把「原始海溫」丟給 STL：seasonal 對應手算的氣候季節，
# residual 則類似「距平再去掉長期趨勢」。比較兩種做法殊途同歸。
#
# 一個重要旋鈕：`trend=` 是趨勢的平滑視窗 (月)。預設視窗較短，趨勢太有彈性，
# 會把我們想看的 ENSO 年際訊號也「吃進趨勢」。把它加大 (這裡 181≈15 年)，
# 趨勢變硬、只保留長期／多十年的緩變 (不一定單調)，ENSO 就留在殘差裡——殘差因此更貼近手算距平。

# %%
stl_sst = STL(sst, period=12, trend=181, robust=True).fit()
fig = stl_sst.plot()
fig.set_size_inches(12, 8)
fig.suptitle("STL decomposition of Niño 3.4 raw SST", y=1.00)
plt.tight_layout()
plt.show()

# %%
# 把「手算距平」與「STL 殘差」疊起來看：兩者形狀相近 (STL 又多扣掉了趨勢)
plt.figure(figsize=(12, 3))
plt.axhline(0, color="gray", lw=0.6)
plt.plot(anom.index, anom, label="Manual anomaly (raw − monthly mean)", color="seagreen", alpha=0.8)
plt.plot(sst.index, stl_sst.resid, label="STL residual (trend also removed)", color="black", alpha=0.7)
plt.title("Anomaly vs STL residual: the signal lives in the residual")
plt.ylabel("°C")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
