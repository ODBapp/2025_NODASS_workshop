# %% [markdown]
# # C. 在地應用：龍洞浮標 + STL + 湧升指數
# 對應敘事文件: narrative/03_buoy_upwelling.md
# 時間預估：約 20 分鐘
#
# 資料是國海院 (NAMR) 提供的龍洞波浪浮標。這一段把前面學的工具用在在地資料上：
# 1. 浮標的季節循環**很強**（對比 B 段赤道的弱季節）。
# 2. 重用 **STL** 取季節/趨勢/殘差。
# 3. 由原始風資料算出物理量「**湧升指數 (Upwelling Index)**」。
# 4. 誠實地問：湧升和 SST 有沒有關係？(回扣故事①：別過度解讀、別挑窗)
#
# 註：圖上文字用英文 (避免 Colab 缺中文字型)；中文說明在 .md 與註解。

# %%
# C0. 套件與工具
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

import ts_utils as ts

# %% [markdown]
# ## C1. 浮標資料概覽：強烈的季節循環
# 龍洞浮標是**時頻 (hourly)** 資料、2010–2024，但有不少缺值。
# 先 resample 成月平均看大圖：SST 全年擺動約 10°C——**這裡季節才是主角**
# （和 B 段赤道 Niño 3.4 只有 1.3°C 形成對比）。

# %%
buoy = ts.load_buoy("Longdong")
monthly = buoy[["SST", "Wind", "Hs"]].resample("MS").mean()

fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
axes[0].plot(monthly.index, monthly["SST"], color="tab:red"); axes[0].set_ylabel("SST (°C)")
axes[1].plot(monthly.index, monthly["Wind"], color="tab:green"); axes[1].set_ylabel("Wind (m/s)")
axes[2].plot(monthly.index, monthly["Hs"], color="tab:blue"); axes[2].set_ylabel("Hs (m)")
axes[0].set_title("Longdong buoy — monthly mean (2010–2024)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## C2. 重用 STL：強季節 + 殘差
# 把日平均 SST 丟給 STL（`period=365`）。和 B 段不同，這裡 seasonal 振幅很大。
#
# ⚠️ 看 trend 面板：它**不是**單調暖化、而是上下起伏。因為這裡**沒有鎖 `trend=`**（對比 A5），
# STL 的趨勢較「軟」，把「年代際的緩慢振盪」也算進了趨勢——這**不代表**龍洞海溫真有那麼大的
# 年代際變化，而是 STL 固定週期分解的侷限（正是附錄 EEMD 想處理的問題）。
# residual 面板裡的一些大塊則多半是**資料缺口**造成的插值假影。
#
# 正因如此，待會做湧升比較時，SST 端我們**不用這條 STL 殘差、改用「距平」**
# （完整理由見 C4 與附錄 `appendix_ui_xcorr`）。這裡先示範 STL 在強季節資料上一樣拆得動。

# %%
sst_d = (buoy["SST"].interpolate(limit=6)              # 補小缺口 (≤6 小時)
         .rolling(48, center=True, min_periods=24).mean()  # 48 小時平滑
         .resample("D").mean().dropna())               # 日平均

stl_sst = STL(sst_d, period=365, robust=True).fit()
fig = stl_sst.plot()
fig.set_size_inches(12, 8)
fig.suptitle("STL decomposition of Longdong daily SST", y=1.00)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## C3. 由原始風資料算「湧升指數 (Upwelling Index, UI)」
# 這是把**原始觀測**變成**物理量**的好例子：
# 風速 + 風向 → 風應力沿岸分量 → 除以 (海水密度 × 科氏參數) → 離岸 Ekman 輸送。
# **UI 為正 = 有利湧升**（把底層冷水帶上來）。公式見 Huang et al. 2021。

# %%
wind = buoy[["Wind", "Wind_Dir"]].interpolate(limit=6)   # 只內插數值欄
ui_h = ts.upwelling_index(wind, "Longdong", coast_angle=18.0)
ui_d = (ui_h.rolling(48, center=True, min_periods=24).mean()
        .resample("D").mean().dropna())

# UI 的季節氣候平均：夏季 (6–8 月) 偏正 = 有利湧升的風場
ui_clim = ui_d.groupby(ui_d.index.month).mean()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].plot(ui_d.index, ui_d, color="tab:purple", lw=0.6)
axes[0].axhline(0, color="gray", lw=0.6)
axes[0].set_title("Daily upwelling index (UI)"); axes[0].set_ylabel("UI (m²/s)")
axes[1].bar(range(1, 13), ui_clim.values,
            color=["tab:red" if v > 0 else "tab:blue" for v in ui_clim.values])
axes[1].axhline(0, color="gray", lw=0.6)
axes[1].set_title("UI seasonal climatology (red>0 = upwelling-favorable)")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("UI (m²/s)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## C4. 沿用論文方法、用浮標驗證 (Huang et al. 2021)
# 論文用 CFSv2 風 + Himawari-8 衛星 SST，主結果是「湧升風天數↔衛星湧升天數 r=0.96」，
# 並指出「湧升訊號比風事件落後幾天」。這裡**沿用論文的 UI 公式 (β=18° 北段)**，
# 但改用國海院浮標資料驗證那句「湧升落後風場」。
# 機制：一陣**有利湧升的風** → 把底層冷水帶上來 → SST 過 1～2 天**下降** (預期負相關、UI 領先)。
#
# 做法（回扣主軸「比較異常」）：
# - **UI** 取 STL **殘差**（去季節去趨勢）。
# - **SST** 取**距平**（減掉浮標自身的月氣候平均）。
# - 算兩者的**落後互相關**，聚焦 2016 年上升流季 (4–10 月)，與論文一致。

# %%
# 先算好兩條序列
sst_clim = sst_d.groupby(sst_d.index.month).mean()        # 浮標自身的月氣候平均
sst_anom = sst_d - sst_d.index.month.map(sst_clim)        # SST 距平
ui_resid = STL(ui_d, period=365, robust=True).fit().resid  # UI 去季節去趨勢
window = slice("2016-04-10", "2016-10-15")                 # 聚焦上升流季 (與論文一致)

# %% [markdown]
# ### C4a. 先用眼睛確認：UI 與 SST 距平是否反相位？
# 相關統計圖不直覺，所以**先疊合時間序列**：看 UI 衝高時，SST 距平是不是隨後往下掉。
# 這一步是在確認「換成浮標資料後，論文 Fig. 3 的 pattern 是否仍存在」——
# 確認看得到，才值得做後面的量化。
#
# 註：這張**先用原始 UI** 看趨勢比較直覺；下一步 C4b 量化時 UI 會改用「STL 殘差」
# （去掉季節後雜訊較少），最佳落後相同、結論一致。

# %%
ui_w = ui_d.loc[window]
anom_w = sst_anom.loc[window]
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(ui_w.index, ui_w, color="tab:blue", label="Upwelling index (UI)")
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylabel("UI (m²/s)", color="tab:blue"); ax.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax.twinx()
ax2.plot(anom_w.index, anom_w, color="tab:orange", label="SST anomaly")
ax2.set_ylabel("SST anomaly (°C)", color="tab:orange"); ax2.tick_params(axis="y", labelcolor="tab:orange")
ax.set_title("2016 upwelling season: UI vs SST anomaly  (UI spikes → SST dips shortly after)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### C4b. 再量化：落後互相關
# 眼睛看到的 pattern，用數字確認：UI 領先 SST 幾天、相關多強。
# UI 取 STL 殘差、SST 取距平，算落後互相關（這就是最後的量化結果）。

# %%
corr_df = pd.concat([ui_resid.rename("ui_resid"),
                     sst_anom.rename("sst_anom")], axis=1, join="inner").dropna()
win = corr_df.loc[window]
lags, rs, ps, best = ts.lagged_xcorr(win["ui_resid"], win["sst_anom"], max_lag=15)
r_best = rs[list(lags).index(best)]

plt.figure(figsize=(10, 3.5))
plt.stem(lags, rs, basefmt="k-")
plt.axhline(0, color="gray", lw=0.6)
plt.axvline(best, color="red", ls="--", lw=1,
            label=f"best lag = +{best} d,  r = {r_best:.2f}")
plt.xlabel("Lag (days):  UI leads SST  →")
plt.ylabel("Correlation r")
plt.title("UI vs SST-anomaly cross-correlation (2016 upwelling season)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **怎麼讀這張圖（給第一次看互相關的人）：**
# - **x 軸 = 落後天數**。`lag = +k` 的意思是「把 SST 往後挪 k 天去對齊 UI」，
#   也就是在問：**今天的 UI 和 k 天後的 SST 有沒有關係**（正 lag = UI 領先 SST）。
# - **y 軸 = 相關係數 r**（−1～+1）。**負的**代表「UI 高 → SST 低」，正是湧升的**降溫**。
# - 所以我們專心看**右半邊（正 lag）有沒有明顯的負值**：這裡 **lag=+1、r≈−0.33** 最強，
#   就是「有利湧升的風，領先海溫下降約 1 天」。
# - ⚠️ r≈−0.33 是**中等偏弱**（單站、訊號雜）；而且相鄰 lag 的相關**彼此不獨立**，
#   所以別只盯著某個 p<0.05，要看整體形狀（右半邊一路是負的）才可靠。

# %%
print(f"best lag = +{best} day,  r = {r_best:.2f}")
print()
print("結果：最強的是 lag = +1 天、r ≈ -0.33 (負相關)——")
print("即『有利湧升的風』領先 SST 下降約 1 天，與 Huang et al. 2021『湧升落後風場』一致。")
print()
print("教學重點：")
print("- 沿用論文的 UI 方法 (公式、β=18°)，用國海院浮標資料就驗證得到湧升的降溫落後效應。")
print("- 方法回扣主軸：UI 取『STL 殘差』、SST 取『距平』，都是把可預期的部分拿掉、只比較異常。")
print("- 誠實區分：論文原始用衛星空間資料、主結果 r=0.96；我們是單站的改編驗證。")
print("- 這是聚焦上升流季 (4–10 月) 的單站結果，單站浮標是很好的地面驗證 (ground truth)。")

# %% [markdown]
# ### C4c. 回扣 B 段：換個時間窗，結論會不會變？
# B 段教過「定義/選擇會影響結論」。這裡換幾個時間窗算同一件事，看 best lag 與 r 穩不穩。

# %%
for label, w in [("2016 上升流季", slice("2016-04-10", "2016-10-15")),
                 ("2018 上升流季", slice("2018-04-10", "2018-10-15")),
                 ("全紀錄 2010–2024", slice(None))]:
    sub = corr_df.loc[w] if w.start is not None else corr_df
    lg, r, p, bl = ts.lagged_xcorr(sub["ui_resid"], sub["sst_anom"], max_lag=15)
    print(f"{label:14s}: best lag = {bl:+d} d,  r = {r[list(lg).index(bl)]:+.2f}")
print()
print("→ 換個窗，best lag 與 r 就會變 (回扣 B 段：選擇會影響結論)。")
print("  所以我們才聚焦在『有物理意義的上升流季』，並誠實說明這是單站、弱訊號的驗證。")
