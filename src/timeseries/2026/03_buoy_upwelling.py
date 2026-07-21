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
from scipy import stats
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
# ## C4. 借用論文的方法，換一個地點試試看 (Huang et al. 2021)
# 論文用 CFSv2 風 + Himawari-8 衛星 SST，研究**台灣東岸**的沿岸湧升。
# 我們借它的 UI 公式，改用國海院龍洞浮標，問一個論文其實**沒有直接量過**的問題：
# **有利湧升的風吹起來之後，SST 隔多久才降下來？**
#
# ⚠️ 先講清楚三件事，免得把「借用方法」誤讀成「重現論文」——這三點本身就是教學重點：
#
# **1. 論文從頭到尾沒有量過「落後幾天」。** 全文沒有互相關、沒有 lag 估計。只有兩句沾到邊：
# - 方法 §2.3：因為湧升訊號**可能**落後於風事件（深層水上湧需要時間），所以他們把每個風事件的
#   判定窗**往後延 3 天**——這是方法上的**預留假設**，不是量測結果。
# - 結果 §3.1：湧升訊號常在風事件**結束之後**還持續數日——講的是**風停後的衰減**，
#   不是**風起後的落後**。
#
#   我們要量的 lag 是**起始落後**，和論文那句是兩個不同的量。所以這一段不是「驗證論文」，
#   而是**把論文假設的機制，實際量出來**——這反而比重現更有價值。
#
# **2. SST 距平的定義不一樣（最深的方法落差）。**
# - 論文：`SST_A = 周圍 20 km 環帶均溫 − 湧升區塊均溫` → **空間**距平（「這一塊比旁邊冷多少」）
# - 我們：`今天 − 該月氣候平均` → **時間**距平（「今天比往年這時候冷多少」）
#
#   單站浮標**在結構上就算不出**論文那個量。而且時間距平會把所有讓該點降溫的原因
#   （鋒面、降雨、混合、黑潮擺動）全部收進來，不只湧升——這是我們 r 偏弱的根本原因之一。
#
# **3. 龍洞根本不在論文的研究區內。** 龍洞浮標 (25.10°N, 121.92°E) 在**東北角**；
#   論文研究的是**東岸**，而且論文開宗明義就把台灣**東北**外海的湧升劃給另一套機制的既有文獻
#   （氣旋式冷丘 / 黑潮入侵），說「東岸沿岸湧升」才是它要補的空白。
#   連 β=18° 都是論文**東岸北段**測點的岸線走向，不是龍洞的。
#   → 方法可以搬，但搬到不同機制主導的海域，訊號會弱掉。**這件事要誠實講，不要假裝沒發生。**
#
# 另外，論文的招牌數字 **r = 0.96** 是「風事件天數 ↔ 湧升天數」的**事件層級散布圖**
# (n=50，計數對計數)，**不是時間序列相關**——所以不能拿我們的 r 去跟它比大小，量的是不同東西。
# (論文另一個 r=0.80 是 n=3 個湧升中心、p<0.1；同一件事在事件層級只有 r=0.40，論文自己說偏弱。)
#
# **機制預期**：一陣**有利湧升的風** → 把底層冷水帶上來 → SST 隨後**下降**（負相關、UI 領先）。
#
# 做法（回扣主軸「比較異常」）：
# - **UI** 取 STL **殘差**（去季節去趨勢）。
# - **SST** 取**距平**（減掉浮標自身的月氣候平均）。
# - 三步走：先看 2016 單一上升流季 (C4a/C4b) → 檢查它到底穩不穩 (C4c) →
#   最後用論文自己的事件判準做合成 (C4d)。**結論會在這三步之間反轉一次，這正是重點。**

# %%
# 先算好兩條序列
# ⚠️ 氣候平均一定要用「浮標自己」的資料算，不能借 OISST 的——理由見下一格
sst_clim = sst_d.groupby(sst_d.index.month).mean()        # 浮標自身的月氣候平均
sst_anom = sst_d - sst_d.index.month.map(sst_clim)        # SST 距平
ui_resid = STL(ui_d, period=365, robust=True).fit().resid  # UI 去季節去趨勢
window = slice("2016-04-10", "2016-10-15")                 # 聚焦上升流季 (與論文一致)

# %% [markdown]
# ### 插曲：氣候平均的基準該用誰的？——一個比「平滑」更隱蔽的陷阱
# 上面用**浮標自己 2010–2024** 的月平均當基準。有兩個很自然的質疑，值得分開回答，
# 因為**一個無關緊要，另一個會直接毀掉結論**。
#
# **質疑一：不該用「同一段資料」的平均，應該用固定基期（例如 1982–2011）。**
# 原則正確——氣候學慣例確實是用固定基期（WMO 標準normals、ONI 的 30 年基期都是）。
# 但**在這個分析裡幾乎沒有差別**。實測：把來源固定成浮標、只改基期，
#
# | 基準 | lag 0 | lag +1 |
# |---|---|---|
# | 全紀錄 2010–2024（本教材） | r=−0.111, p=0.0030 | r=−0.107, p=0.0068 |
# | 固定基期 2010–2019 | r=−0.111, p=0.0030 | r=−0.107, p=0.0068 |
# | 固定基期 2011–2020 | r=−0.117, p=0.0021 | r=−0.112, p=0.0050 |
#
# 為什麼沒差？每個月份格子都有 **380–450 天**資料，月平均早就收斂（誤差約 0.07°C）；
# 而我們要找的 25 個事件只占月平均極小一部分，不存在「訊號被自己扣掉」的問題。
# （基期**真正要緊**的場合是：算長期趨勢、或要和別的研究比較距平的絕對值。）
#
# **質疑二：那乾脆借 `ts.oisst_monthly_climatology()` 算好的 OISST 1982–2011 基準？**
# **千萬不要。** 這會毀掉結果——而且原因不是基期，是**量測對象根本不同**：
#
# | 月 | 浮標 2010–24 | OISST 82–11 | 差 |
# |---|---|---|---|
# | 2 | 18.87 | 20.95 | **−2.08** |
# | 5 | 23.96 | 25.16 | −1.20 |
# | 8 | 28.27 | 27.89 | **+0.38** |
# | 11 | 22.00 | 23.81 | −1.81 |
#
# 差值不是常數偏移：全年振幅 **2.92°C**、上升流季內 2.19°C，而且**會變號**。
# 根源是**季節振幅不一樣**——浮標 9.40°C vs OISST 6.94°C。
# 浮標是**岸邊淺水的一個點**；OISST 是 **25 km 網格、混進了外海黑潮**。
# 淺水升降溫快、外海被黑潮撐住，兩者的季節循環在物理上就是不同的東西。
#
# 所以拿 OISST 的氣候平均去扣浮標，**扣不掉浮標的季節循環**，
# 反而在上升流季內留下約 2.2°C 的殘餘季節。後果：
#
# | | lag 0 | lag +1 | lag +4 |
# |---|---|---|---|
# | 浮標自身基準（本教材） | −0.111 (p=**0.003**) | −0.107 (p=**0.007**) | +0.087 |
# | 改用 OISST 基準 | −0.054 (p=0.093) | −0.045 (p=0.227) | **+0.138** (p=0.0006) |
#
# **真訊號被打掉，lag +4 的假峰反而變強**——那正是灌進去的殘餘季節在製造假相關。
# 距平的標準差也從 1.442 漲到 1.740，多出來的變異就是沒扣乾淨的季節。
#
# **原則：要扣掉的氣候平均，必須是「同一個量測對象」的氣候平均。**
# 基期選哪一段是次要的；**跨來源相減才是致命的**。
#
# 對照 A 段就懂了：A4 用 `base=("1982","2011")` 算 Niño 3.4 距平是**對的**，
# 因為那裡序列和氣候平均**都來自同一套 NOAA/OISST 資料**，內部一致。
# 浮標這裡若改用 OISST，就變成拿 A 的尺去量 B 了。
#
# 💡 附帶一提：OISST 在這裡其實有個**正當**用途，只是不是當基準——
# 論文的距平是**空間**距平（比周圍海域冷多少）。若要逼近論文，應該拿浮標去比
# **同一天**的外海 OISST，而不是比它的長期平均。那需要逐日抓 OISST，可以當作延伸練習。

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
#   看起來就是「有利湧升的風，領先海溫下降約 1 天」。
#   （**先別急著相信這個「1 天」**——C4c 會發現 lag 0 和 +1 其實分不出高下。）
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
print(f"2016 這個窗的結果：最強的是 lag = +{best} 天、r = {r_best:.2f} (負相關)——")
print("方向符合『有利湧升的風 → 隨後降溫』的事前預期。")
print()
print("方法回扣主軸：UI 取『STL 殘差』、SST 取『距平』，都是把可預期的部分拿掉、只比較異常。")
print("不做額外平滑：平滑會讓 r 變好看 (-0.22 → -0.33)，卻讓相關峰變鈍、落後判不出來。")
print()
print("⚠️ 但先別下結論——這只是『一個窗、一年』。")
print("   C4c 會用同樣的方法掃過全部 15 個上升流季，你會看到單一窗有多不可靠。")

# %% [markdown]
# ### C4c. 回扣 B 段：換個時間窗，結論會不會變？
# B 段教過「定義/選擇會影響結論」。C4b 只看了 2016 一年——現在把**完全一樣**的計算
# 套到 2010–2024 每一個上升流季，看 best lag 與 r 穩不穩。
#
# 先看兩個「隨手換窗」的例子，每個都附上自己的顯著門檻。你會看到一件反直覺的事：
# **統計上全都「顯著」，但其中兩個根本講不通。**

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
    print(f"{label:16s}: best lag = {bl:+3d} d,  r = {rb:+.2f}   "
          f"(N_eff={ne:.0f}, 門檻={rc:.3f} → {verdict})")
print()
print("⚠️ 這三個結果『統計上』全都超過門檻，但只有 2016 那個講得通。")
print("  - 2018: lag = -12 天、正相關 → 意思是『SST 領先 UI 12 天且同向』，物理上荒謬。")
print("  - 全紀錄: r 只有 -0.10，卻因為樣本數上千而輕鬆「顯著」——")
print("    資料夠多時，再微小的相關都會顯著，但『顯著』不等於『重要』。")
print()
print("所以：統計檢定只能擋掉『太小的 r』，擋不掉『沒有物理意義的 r』。")

# %% [markdown]
# ### 全部掃一遍：2016 其實是「剛好對的那一年」
# 上面只挑了三個窗，容易被說是刻意選的。乾脆把 15 個上升流季**全部**跑一次，
# 每年各自挑自己 |r| 最大的 lag，看看落在哪裡。

# %%
SEASON_YEARS = range(2010, 2025)


def season_window(yr):
    return slice(f"{yr}-04-10", f"{yr}-10-15")


season_curves = {}          # 年 -> 該年的整條 r(lag) 曲線
print(f"{'年':>6} {'N':>5} {'best lag':>9} {'r':>7}   物理上講得通？")
for yr in SEASON_YEARS:
    sub = corr_df.loc[season_window(yr)]
    if len(sub) < 120:      # 資料不足約 4 個月的季節跳過 (2024 只有 79 天)
        print(f"{yr:>6} {len(sub):>5}   —— 資料不足，跳過")
        continue
    lg, r, p, bl = ts.lagged_xcorr(sub["ui_resid"], sub["sst_anom"], max_lag=15)
    season_curves[yr] = r
    rb = r[list(lg).index(bl)]
    ok = "✅" if (0 <= bl <= 2 and rb < 0) else "❌"
    print(f"{yr:>6} {len(sub):>5} {bl:>+9d} {rb:>+7.2f}   {ok}")

lags_c = lg                                        # 所有年份的 lag 軸都一樣
R = pd.DataFrame(season_curves, index=lags_c).T    # 列 = 年, 欄 = lag
n_ok = sum(1 for yr in R.index
           if (0 <= R.columns[np.argmax(np.abs(R.loc[yr].values))] <= 2
               and R.loc[yr].values[np.argmax(np.abs(R.loc[yr].values))] < 0))
print()
print(f"→ {len(R)} 個上升流季裡，best lag 落在 0~+2 天且為負(講得通)的只有 {n_ok} 年。")
print("   2016 正是那少數幾年之一。**如果我們只報 C4b，就等於挑了對的那一年。**")

# %% [markdown]
# ### 那怎麼辦？不是「再換一個窗」，而是**跨年合成**
# 逐年各挑各的 best lag，等於每年都做一次「從 31 根裡挑最高」的多重比較——難怪亂跳。
#
# 正確的做法是把**每一年的整條 r(lag) 曲線疊起來平均**：
# 真訊號每年都在同一個 lag、同一個方向，會**疊加**；
# 雜訊的假峰每年落在不同位置，會**互相抵消**。
# 這樣就不必挑年份，也不必挑 lag。

# %%
mean_r = R.mean(axis=0)
se_r = R.std(axis=0, ddof=1) / np.sqrt(len(R))

plt.figure(figsize=(10, 3.8))
plt.axhline(0, color="gray", lw=0.6)
plt.fill_between(R.columns, mean_r - se_r, mean_r + se_r,
                 color="tab:blue", alpha=0.25, label="±1 SE (across seasons)")
plt.plot(R.columns, mean_r, color="tab:blue", marker="o", ms=3.5,
         label=f"mean over {len(R)} upwelling seasons")
plt.axvline(0, color="red", ls="--", lw=1)
plt.axvline(1, color="red", ls="--", lw=1)
plt.xlabel("Lag (days):  UI leads SST  →")
plt.ylabel("Mean correlation r")
plt.title(f"Composite UI–SST cross-correlation ({len(R)} upwelling seasons, 2010–2023)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# 對每個 lag 做單樣本 t 檢定：跨年的平均 r 跟 0 分得出來嗎？
print(f"{'lag':>5} {'平均 r':>8} {'SE':>7} {'t':>7} {'p':>8}  負值年數")
for lag in range(-2, 7):
    v = R[lag].values
    t, p = stats.ttest_1samp(v, 0)
    star = " ←" if p < 0.01 else ""
    print(f"{lag:>+5d} {v.mean():>+8.3f} {v.std(ddof=1)/np.sqrt(len(v)):>7.3f} "
          f"{t:>+7.2f} {p:>8.4f}  {int((v < 0).sum())}/{len(v)}{star}")

# 更穩健的檢查：不看 r 的大小、只看「有幾年是負的」(符號檢定，不受離群年份影響)
print()
for lag in [0, 1]:
    k, n = int((R[lag] < 0).sum()), len(R)
    p_sign = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"符號檢定 lag={lag:+d}: {k}/{n} 年為負,  p = {p_sign:.4f}")

# %% [markdown]
# ### 合成的結果：訊號回來了，而且比單一年份可信得多
# 疊完 14 個上升流季之後，曲線變得很乾淨：
#
# | lag | 平均 r | t | p | 負值年數 |
# |---|---|---|---|---|
# | 0 | **−0.111** | −3.63 | **0.003** | 11/14 |
# | +1 | **−0.107** | −3.21 | **0.007** | 11/14 |
# | +2 | −0.001 | −0.05 | 0.96 | 9/14 |
# | +4 | +0.087 | +3.47 | 0.004 | 3/14 |
#
# 三件事值得注意：
#
# **1. 方向是跨年可複製的。** lag 0 和 +1 都是負的、都通過檢定，14 年裡有 11 年同號
# （符號檢定 p=0.029）。單一年份的 r 會亂跳，但**平均起來方向很穩**——
# 這比 C4b 那個「剛好越過門檻」的 −0.22 可信得多，儘管數字更小 (−0.11)。
#
# **2. 但「領先 1 天」這個說法要收回一點。** lag 0 (−0.111) 和 lag +1 (−0.107) **分不出高下**。
# 日資料本來就無法解析 1 天以內的差別，所以誠實的講法是「**同日到 1 天內**」，
# 而不是 C4b 看起來的「就是 1 天」。**單一年份的峰位置，很大一部分是雜訊決定的。**
#
# **3. lag +3~+6 翻成正的，是有物理意義的。** 這代表「今天 UI 高 → 4 天後 SST 偏暖」，
# 也就是**湧升風鬆弛之後的回溫**。它不是雜訊——下一節 C4d 會直接看到這個回復過程。
#
# ⚠️ 最後仍要誠實：r ≈ −0.11 是**很弱**的相關。它撐得住的是「方向」與「時間尺度」，
# 撐不住「湧升是龍洞 SST 變化的主要原因」——後者需要的證據遠比這多。

# %% [markdown]
# ### C4d. 換個角度：用論文自己的「事件」判準（回扣 B 段）
# 互相關是把兩條**連續序列**逐點相乘，湧升風其實是**離散事件**——
# 一陣風吹起來、持續幾天、然後停。B 段教過「把序列抽象成事件」，這裡正好再用一次，
# 而且這是**最貼近論文的做法**：論文根本沒算互相關，它是先定義「顯著湧升風事件」，
# 再去看事件期間 SST 有沒有出現湧升訊號。
#
# 論文 §2.2 的事件判準（我們取最單純的連續版）：
# 1. 連續 **≥5 天** UI > 0；
# 2. 且**前 5 天累積 UI ≥ 2.0** m²/s（同時要求「夠久」和「夠強」）。
#
# （論文另有「中間可以斷 1–3 天」的細則，這裡略過不影響教學。）
#
# 然後做**事件合成 (superposed epoch analysis)**：把每個事件的**起風日**對齊成 day 0，
# 看事件前後 SST 距平平均怎麼走。這是大氣海洋領域處理「離散事件 → 反應」的標準工具。

# %%
def find_wind_events(ui_daily, years, min_days=5, cum5_min=2.0):
    """
    論文 §2.2 的簡化版：找「顯著的有利湧升風事件」，回傳每個事件的起風日。
    - 連續 >= min_days 天 UI > 0
    - 且前 5 天累積 UI >= cum5_min (m^2/s)
    """
    onsets = []
    for yr in years:
        s = ui_daily.loc[f"{yr}-04-10":f"{yr}-10-15"].dropna()
        pos = s > 0
        run = (pos != pos.shift()).cumsum()          # 連續同號編號 (同 B2 的手法)
        for _, g in s.groupby(run):
            if (g > 0).all() and len(g) >= min_days and g.iloc[:5].sum() >= cum5_min:
                onsets.append(g.index[0])
    return onsets


onsets = find_wind_events(ui_d, SEASON_YEARS)
print(f"2010–2024 上升流季共找到 {len(onsets)} 個顯著湧升風事件")
print("前幾個事件的起風日：",
      ", ".join(d.strftime("%Y-%m-%d") for d in onsets[:6]), "...")

# %%
# 事件合成：把每個事件的起風日對齊成 day 0，取 SST 距平的平均
OFFSETS = np.arange(-7, 15)
M = np.array([[sst_anom.get(t0 + pd.Timedelta(days=int(k)), np.nan) for k in OFFSETS]
              for t0 in onsets], dtype=float)
n_obs = np.sum(~np.isnan(M), axis=0)
mean_ssta = np.nanmean(M, axis=0)
se_ssta = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n_obs)

plt.figure(figsize=(10, 4))
plt.axhline(0, color="gray", lw=0.6)
plt.axvline(0, color="tab:green", ls="--", lw=1.5, label="Wind event onset (day 0)")
plt.fill_between(OFFSETS, mean_ssta - se_ssta, mean_ssta + se_ssta,
                 color="tab:orange", alpha=0.25, label="±1 SE")
plt.plot(OFFSETS, mean_ssta, color="tab:orange", marker="o", ms=4,
         label=f"Mean SST anomaly (n={len(onsets)} events)")
plt.xlabel("Days relative to onset of upwelling-favorable wind event")
plt.ylabel("SST anomaly (°C)")
plt.title("Superposed epoch: SST response to upwelling-favorable wind events (Longdong)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

for k, m, s, n in zip(OFFSETS, mean_ssta, se_ssta, n_obs):
    if -7 <= k <= 8:
        mark = "  <-- 起風" if k == 0 else ("  <-- 最低點" if k == 2 else "")
        print(f"day {k:+3d}:  SSTA = {m:+.3f} ± {s:.3f} °C  (n={n}){mark}")

# %%
# 量化：事件後 (+1~+7 天) 比事件前 (-7~-3 天) 冷了多少？（同一批事件配對比較）
pre = np.nanmean(M[:, (OFFSETS >= -7) & (OFFSETS <= -3)], axis=1)
post = np.nanmean(M[:, (OFFSETS >= 1) & (OFFSETS <= 7)], axis=1)
ok = ~np.isnan(pre) & ~np.isnan(post)
t_ev, p_ev = stats.ttest_rel(post[ok], pre[ok])

print(f"事件後(+1~+7天) 減 事件前(-7~-3天) 的 SST 距平變化：")
print(f"  Δ = {(post[ok] - pre[ok]).mean():+.3f} °C   "
      f"(配對 t = {t_ev:+.2f}, p = {p_ev:.4f}, n = {ok.sum()} 個事件)")
print(f"  {int((post[ok] < pre[ok]).sum())}/{ok.sum()} 個事件在起風後降溫")

# %% [markdown]
# ### C4d 的結果：這是整段最清楚的一張圖
# 事件合成畫出來的形狀，就是教科書上的湧升反應：
#
# | 相對天數 | 平均 SST 距平 |
# |---|---|
# | −7 ~ −1（起風前） | **約 +0.3 ~ +0.4 °C**（偏暖） |
# | 0（起風日） | +0.06 |
# | +1 | **−0.55** |
# | **+2** | **−0.71 ← 最低點** |
# | +4 | −0.58 |
# | +6 以後 | 回到 0 附近 |
#
# 整個擺盪幅度約 **1.0 °C**，配對檢定 Δ = **−0.65 °C**（t = −2.24, p = 0.035，23 個事件中 15 個降溫）。
#
# **為什麼這張比互相關好看那麼多？**
# 互相關假設「UI 每高一單位、SST 就低固定比例」，是**線性、逐點**的假設；
# 但真實的湧升是**閾值型的離散事件**——風要夠強夠久才會把冷水抽上來，
# 弱風的日子再多也對不出關係，反而把 r 稀釋掉。
# 換成事件框架，等於只問「該發生的時候有沒有發生」，訊號就浮出來了。
# **這正是 B 段「把序列抽象成事件」在真實資料上的回報。**
#
# 同時它也解釋了 C4c 的兩個現象：
# - 最低點在 **day +2**（不是 +1）→ 難怪互相關在 lag 0~+1 分不出高下，日資料解析度就這樣。
# - **day +6 回到 0** → 就是 C4c 看到 lag +3~+6 翻正的那個回溫。
#
# ⚠️ 仍要誠實的地方：
# - 只有 **25 個事件**，單日的 SE 有 ±0.3~0.4，撐住結論的是**形狀的連貫性**，不是單日的顯著性。
# - p = 0.035 是「還可以」，不是「鐵證」。
# - 起風前偏暖 (+0.4 °C) 有一部分可能是**天氣系統本身**造成的（西南風來之前常是晴朗高溫），
#   不全是湧升的功勞——**單站資料沒辦法把這兩者分開**，這是本例的天花板。
#
# ### 那 C 段最後到底能說什麼？
# 可以說的：**在龍洞，有利湧升的風事件之後 1–2 天，海溫確實會出現約 0.5–1 °C 的降溫，
# 並在約一週內回復；這個方向在 14 個上升流季裡是可複製的。**
#
# 不能說的：不能說湧升是龍洞海溫變化的主因（r 只有 −0.11）；
# 不能說我們「重現」了 Huang et al. 2021（地點、SST 距平定義、分析方法都不同）；
# 也不能拿我們的 r 去和論文的 0.96 比大小。
#
# **這就是整個工作坊最想教的事：把結論修剪到證據真正撐得住的大小。**
