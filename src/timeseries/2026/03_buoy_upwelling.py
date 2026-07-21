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
sst_d = ts.to_daily(buoy["SST"].interpolate(limit=6),  # 補小缺口 (≤6 小時)
                    min_hours=12)                      # 日平均，當日不足 12 小時就捨棄

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
ui_d = ts.to_daily(ui_h, min_hours=12)                   # 日平均，不做額外平滑 (見下方說明)

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
# ### 插曲：要不要「再平滑一下」？——一個很常見、但會反咬自己的誘惑
# 浮標是逐時資料、看起來很毛躁，直覺會想在取日平均前先做個滑動平均，例如
# `ui_h.rolling(48, center=True).mean().resample("D").mean()`。我們實測過，**別這樣做**。
#
# **1) 它比看起來更強力。** 48 小時方波再跟日平均的 24 小時方波卷積，等效濾波跨距是
# **3 天**（72h 梯形），不是 2 天。龍洞資料上，日間變動被削掉 31%、尖峰降低 23%，
# 上升流季 UI>4 的天數從 4 天剩 1 天——湧升風本來就是**離散事件**，尖峰正是訊號本身。
#
# **2) 它讓相關係數變好看，卻讓結論變模糊。** 這是最反直覺的地方：
#
# | 做法 | best lag | r | lag+2 ÷ lag+1 | 有效樣本數 |
# |---|---|---|---|---|
# | 不額外平滑（本教材） | +1 | **−0.22** | **0.15**（峰銳利） | 91 |
# | UI+SST 都 48h 平滑 | +1 | −0.33 | 0.61（峰很鈍） | 60 |
#
# 平滑後 r 從 −0.22「進步」到 −0.33，但 lag 0/+1/+2 的相關變成 −0.21/−0.33/−0.20——
# 三根幾乎一樣高，**你根本分不出峰在哪**。而 C4 整段要主張的正是「UI 領先 SST 約 1 天」，
# 把峰磨鈍等於親手削弱自己的結論。而且平滑會灌入共同的自相關，有效樣本數掉 34%，
# 若照原始 N 去讀 p 值就會**高估顯著性**。
#
# **3) 「防資料缺口」不需要靠平滑。** 龍洞 96% 的日子有 ≥18 小時資料，不足 12 小時的
# 稀疏日只佔 1.4%。直接數點數（`ts.to_daily(..., min_hours=12)`）目的單純、也不動訊號。
#
# 教訓：**平滑讓圖變漂亮、相關變強，代價是把你要證明的東西一起抹掉。**
# 做任何平滑前先問：我抹掉的，會不會正好是我要找的訊號？

# %% [markdown]
# ## C3.5 熱身：用「已知答案」的模擬序列，先搞懂落後互相關
# C4 要對真實資料算「**落後互相關 (lagged cross-correlation)**」。在那之前，先重用 A1 的模擬函數
# 自己造**一對**序列——因為延遲幾個月、方向為何**都是我們自己設定的**，才能檢驗這個方法讀不讀得出來
# （和 A1「先用模擬驗證 STL」是同一個精神）。
#
# 設定（模仿湧升的物理）：
# - **Driver X**＝驅動者（想成湧升指數）：直接重用 `ts.make_synthetic_series`（趨勢＋季節＋噪音）。
# - **Response Y**＝反應者（想成 SST 距平）：對 **3 個月前**的 X 反應、**方向相反**（X 高 → Y 低），
#   再加上 Y 自己的噪音。

# %%
# 兩個「已知的正確答案」—— 待會要用互相關把它們找回來
LAG_TRUE = 3      # Y 比 X 慢 3 個月反應
COUPLING = -0.8   # 方向相反：X 高 → 3 個月後 Y 低（模仿 湧升風 → 降溫）

sim2 = ts.make_synthetic_series(n=120, slope=0.04, season_amp=2.0, noise=0.5)  # 重用 A1
driver = sim2["y"].rename("driver")                     # X：驅動者（像湧升指數）

rng = np.random.default_rng(2027)
response = (COUPLING * driver.shift(LAG_TRUE)           # 對「3 個月前的 X」反應
            + 0.5 * rng.standard_normal(len(driver))    # 再加 Y 自己的噪音
            ).rename("response")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(driver.index, driver, color="tab:blue")
ax.set_ylabel("Driver X", color="tab:blue"); ax.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax.twinx()
ax2.plot(response.index, response, color="tab:orange")
ax2.set_ylabel("Response Y", color="tab:orange"); ax2.tick_params(axis="y", labelcolor="tab:orange")
ax.set_title(f"Synthetic pair:  Y(t) = {COUPLING} * X(t-{LAG_TRUE}) + noise   (true lag = {LAG_TRUE} months)")
fig.tight_layout()
plt.show()
# 眼睛先看：橘線 (Y) 的谷，總是跟在藍線 (X) 的峰後面約 3 格——這就是待會要「量」出來的延遲

# %% [markdown]
# ### 「落後相關」其實就是：搬一搬，再算相關
# 把 Y 整條往回搬 `lag` 個月（讓「X」對齊「lag 個月後的 Y」），算一個普通的相關係數 r；
# 每個 lag 都做一次、比較哪個 lag 的 **|r| 最大**——這就是 `ts.lagged_xcorr()` 做的全部事情，
# 沒有更多魔法。先手動掃 0～6 個月做給你看（`#` 越長 = |r| 越大）：

# %%
# 「落後相關」＝ 把 Y 往回搬 lag 個月、對齊、算普通的 Pearson r —— 如此而已
for lag in range(0, 7):
    shifted = response.shift(-lag)               # 對齊「lag 個月後的 Y」
    m = driver.notna() & shifted.notna()         # 兩邊都有值的月份才算
    r = driver[m].corr(shifted[m])
    print(f"lag=+{lag} 月  r = {r:+.2f}  {'#' * int(abs(r) * 20)}")

# %% [markdown]
# ### 陷阱：季節會製造「到處都相關」的假象
# 上面掃出 lag=+3 最強（r≈−0.94），看起來成功了？**別急。**
# X 和 Y 都帶著強季節，而任何兩條有季節的序列，每隔一個週期（12 個月）就會「自動」對齊一次——
# 高相關可以完全來自共同的季節，跟真正的因果耦合無關。
#
# 把 lag 掃寬到 ±20 個月就會現形（下圖左）：除了 +3 之外，**−9、+15 的相關幾乎一樣強**。
# 光看原始序列，你分不出哪一根才是真的延遲。
#
# 解法是 A 段的老朋友：先用 **STL 把季節和趨勢拿掉、只留殘差**，再算互相關（下圖右）——
# 假峰消失，只剩 +3 一根乾淨的負峰。
#
# 圖上的**灰帶**是「顯著門檻」：`ts.r_critical(n)` 算出在 α=0.05 下 |r| 要多大才跟 0 分得出來。
# **落在灰帶裡的都不算數**——這比只看「哪根最高」可靠得多，因為最高的那根也可能只是雜訊。
# （公式就是相關係數的 t 檢定，兩行 scipy；資料量夠時不需要 bootstrap。）

# %%
# 重用 A 段的 STL：把季節+趨勢拿掉、只留殘差，再算一次互相關來對照
drv_resid = STL(driver, period=12, robust=True).fit().resid
rsp_resid = STL(response.dropna(), period=12, robust=True).fit().resid

lags_raw, r_raw, _, best_raw = ts.lagged_xcorr(driver, response, max_lag=20)
lags_res, r_res, _, best_res = ts.lagged_xcorr(drv_resid, rsp_resid, max_lag=20)

# 顯著帶：|r| 要超過這條線，才跟 0 分得出來（灰帶內 = 可能只是雜訊）
rc = ts.r_critical(len(drv_resid.dropna()))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for ax, (lg, rr, bl, ttl) in zip(axes, [
        (lags_raw, r_raw, best_raw, "Raw series (season & trend still inside)"),
        (lags_res, r_res, best_res, "STL residuals (season & trend removed)")]):
    ax.axhspan(-rc, rc, color="gray", alpha=0.15,
               label=f"not significant (|r| < {rc:.2f})")
    ax.stem(lg, rr, basefmt="k-")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(LAG_TRUE, color="green", ls=":", lw=1.5, label=f"true lag = +{LAG_TRUE}")
    ax.axvline(bl, color="red", ls="--", lw=1,
               label=f"best lag = {bl:+d},  r = {rr[list(lg).index(bl)]:.2f}")
    ax.set_xlabel("Lag (months):  X leads Y  →")
    ax.set_title(ttl)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Correlation r")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 對照表：這個熱身 ↔ C4 的真實案例
#
# | 熱身（模擬） | C4（真實資料） |
# |---|---|
# | Driver X（的 STL 殘差） | 湧升指數 UI（的 STL 殘差） |
# | Response Y | SST 距平 |
# | true lag：我們自己設 +3 個月 | 未知——互相關就是用來「估」它 |
# | 方向相反（COUPLING < 0） | 湧升 → 降溫，也預期**負**相關 |
#
# 兩個誠實提醒：
# 1. 殘差版的 r≈−0.5，比我們設定的耦合 −0.8 弱——因為兩條序列各自的噪音還在（本來就該在）。
#    所以待會 C4 看到 r≈−0.22 不要驚訝：**真實資料裡的「強訊號」就長這樣**。
# 2. 原始序列的 −0.94 雖然大，但它和 −9、+15 的假峰難分真假——**大的 r 不等於可信的 r**。
#    先去掉可預期的部分（季節、趨勢），才能明確指認真正的延遲。

# %% [markdown]
# ### 🧪 小練習：訊號要多弱，才會被雜訊淹沒？
# 把上面造 `response` 那格的噪音 `0.5 * rng.standard_normal(...)` 改大，重跑這一段：
#
# | Y 的噪音 | best lag | r | 還找得到真答案嗎 |
# |---|---|---|---|
# | 0.5（預設） | +3 | −0.48 | ✅ 遠高於灰帶 |
# | 0.75 | +3 | −0.33 | ✅ 仍明確 |
# | 1.0 | +3 | −0.26 | ⚠️ 剛好撐住 |
# | **1.5** | **+15** | **+0.19** | ❌ **真訊號消失，假峰勝出** |
#
# noise=1.5 那列最值得看：最強的變成 lag **+15、而且是正相關**——
# 「Y 領先 X 15 個月又同向」在我們的設定裡根本不可能，它純粹是雜訊剛好對上，
# 卻仍「勉強超過」顯著門檻。
#
# ⚠️ **為什麼會這樣？多重比較。** 我們一次掃了 41 個 lag，每個都用 α=0.05 檢定，
# 純靠運氣就會期望有 **2 個左右**冒出灰帶。所以「有一根超出灰帶」本身**不是**強證據。
#
# 那 C4 的結果憑什麼可信？因為那根峰落在**事先就預測好的位置與方向**
# （湧升風領先降溫 → 正 lag、負相關）。**事前預測**和**事後從 41 根裡挑最高的**，
# 證據力天差地遠——這也是讀 paper 時最該提防的地方。

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

# 顯著門檻要用「有效樣本數」，不是天真的 len()：時間序列相鄰點彼此相關
n_eff = ts.n_effective(win["ui_resid"], win["sst_anom"])
rc = ts.r_critical(n_eff)
print(f"N = {len(win)} 天，但有效樣本數只有 {n_eff:.0f} (自相關讓獨立樣本變少)")
print(f"→ 顯著門檻 |r| > {rc:.3f}   (若天真用 N={len(win)} 會鬆成 {ts.r_critical(len(win)):.3f})")

plt.figure(figsize=(10, 3.5))
plt.axhspan(-rc, rc, color="gray", alpha=0.15,
            label=f"not significant (|r| < {rc:.2f}, N_eff={n_eff:.0f})")
plt.stem(lags, rs, basefmt="k-")
plt.axhline(0, color="gray", lw=0.6)
plt.axvline(best, color="red", ls="--", lw=1,
            label=f"best lag = +{best} d,  r = {r_best:.2f}")
plt.xlabel("Lag (days):  UI leads SST  →")
plt.ylabel("Correlation r")
plt.title("UI vs SST-anomaly cross-correlation (2016 upwelling season)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **怎麼讀這張圖（給第一次看互相關的人）：**
# - **x 軸 = 落後天數**。`lag = +k` 的意思是「把 SST 往後挪 k 天去對齊 UI」，
#   也就是在問：**今天的 UI 和 k 天後的 SST 有沒有關係**（正 lag = UI 領先 SST）。
# - **y 軸 = 相關係數 r**（−1～+1）。**負的**代表「UI 高 → SST 低」，正是湧升的**降溫**。
# - 所以我們專心看**右半邊（正 lag）有沒有明顯的負值**：這裡 **lag=+1、r≈−0.22** 最強，
#   就是「有利湧升的風，領先海溫下降約 1 天」。
# - **注意這根峰有多「尖」**：lag 0/+1/+2 是 −0.12 / −0.22 / −0.03，峰只有一根、兩側掉很快。
#   這正是我們前面**刻意不做額外平滑**的回報——平滑過的版本 r 會「進步」到 −0.33，
#   但三根會變成 −0.21/−0.33/−0.20，反而看不出峰在哪（見 C3 的插曲）。
# - **灰帶 = 顯著門檻**。這裡有個關鍵細節：雖然有 171 天資料，但相鄰兩天的 UI（和 SST）
#   本來就相似，**有效樣本數只有約 100**（少了 4 成）。若天真地用 N=171 查表，門檻會鬆成
#   0.150；改用有效樣本數後是 **0.197**。我們的 |r|=0.216 **剛好越過**——通過了，但不寬裕。
# - 31 根 lag 裡**只有 +1 這根**探出灰帶，位置與方向都符合「湧升風領先降溫」的物理預期。
# - ⚠️ 但要誠實：掃這麼多 lag，純靠運氣也會有一兩根冒出灰帶（多重比較）。
#   這個結果之所以可信，是因為它落在**事先預測好的位置與方向**，而不是事後挑最高的那根。
#   r≈−0.22 仍屬**偏弱**（單站、訊號雜），別過度宣稱。

# %%
print(f"best lag = +{best} day,  r = {r_best:.2f}")
print()
print(f"結果：最強的是 lag = +{best} 天、r = {r_best:.2f} (負相關)——")
print("即『有利湧升的風』領先 SST 下降約 1 天，與 Huang et al. 2021『湧升落後風場』一致。")
print()
print("教學重點：")
print("- 沿用論文的 UI 方法 (公式、β=18°)，用國海院浮標資料就驗證得到湧升的降溫落後效應。")
print("- 方法回扣主軸：UI 取『STL 殘差』、SST 取『距平』，都是把可預期的部分拿掉、只比較異常。")
print("- 不做額外平滑：平滑會讓 r 變好看 (-0.22 → -0.33)，卻讓相關峰變鈍、落後判不出來。")
print("- 誠實區分：論文原始用衛星空間資料、主結果 r=0.96；我們是單站的改編驗證。")
print("- 這是聚焦上升流季 (4–10 月) 的單站結果，單站浮標是很好的地面驗證 (ground truth)。")

# %% [markdown]
# ### C4c. 回扣 B 段：換個時間窗，結論會不會變？
# B 段教過「定義/選擇會影響結論」。這裡換幾個時間窗算同一件事，看 best lag 與 r 穩不穩，
# 並且**每個窗都附上自己的顯著門檻**——你會看到一件反直覺的事：
# 三個窗的結果**統計上全都「顯著」**，但其中兩個根本講不通。

# %%
for label, w in [("2016 上升流季", slice("2016-04-10", "2016-10-15")),
                 ("2018 上升流季", slice("2018-04-10", "2018-10-15")),
                 ("全紀錄 2010–2024", slice(None))]:
    sub = corr_df.loc[w] if w.start is not None else corr_df
    lg, r, p, bl = ts.lagged_xcorr(sub["ui_resid"], sub["sst_anom"], max_lag=15)
    rb = r[list(lg).index(bl)]
    ne = ts.n_effective(sub["ui_resid"], sub["sst_anom"])
    rc = ts.r_critical(ne)
    verdict = "超過門檻" if abs(rb) > rc else "落在灰帶內"
    print(f"{label:14s}: best lag = {bl:+3d} d,  r = {rb:+.2f}   "
          f"(N_eff={ne:.0f}, 門檻={rc:.3f} → {verdict})")
print()
print("→ 換個窗，best lag 與 r 就會變 (回扣 B 段：選擇會影響結論)。")
print()
print("⚠️ 最重要的一點：這三個結果『統計上』全都超過門檻，但只有 2016 那個講得通。")
print("  - 2018: lag = -12 天、正相關 → 意思是『SST 領先 UI 12 天且同向』，物理上荒謬。")
print("  - 全紀錄: r 只有 -0.10，卻因為樣本數上千而輕鬆「顯著」——")
print("    資料夠多時，再微小的相關都會顯著，但『顯著』不等於『重要』。")
print()
print("所以：統計檢定只能擋掉『太小的 r』，擋不掉『沒有物理意義的 r』。")
print("最後把關的永遠是機制與事前預測，不是 p 值。")
