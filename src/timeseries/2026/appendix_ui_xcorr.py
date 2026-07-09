# %% [markdown]
# # 附錄. UI–SST 互相關的「參數版」：為什麼 SST 用距平、不再做 STL？
# 對應敘事文件: narrative/appendix_ui_xcorr.md
# **這一段不在課堂時間內，回答 lecture 裡的一個問題、並示範「處理方式會影響結論」。**
#
# 主文 C 段的選擇是：**UI 取 STL 殘差、SST 取距平**。常被問到：
# > 「為什麼 UI 要做 STL，SST 卻只用距平、不也做 STL？」
#
# 答案：**SST 距平已經減過一次月氣候平均**，等於把季節與大部分長期趨勢去掉了；
# 再對它做 STL 等於**重複去趨勢、會把訊號裡的資訊也削掉**（over-processing）。
# UI 沒有先做距平，所以才需要 STL。這一頁用資料把這件事攤開來看。
#
# 註：圖上文字用英文；中文說明在 .md 與註解。

# %%
# U0. 套件、資料、共用片段
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

import ts_utils as ts

buoy = ts.load_buoy("Longdong")
wind = buoy[["Wind", "Wind_Dir"]].interpolate(limit=6)
ui_d = (ts.upwelling_index(wind, "Longdong", 18.0)
        .rolling(48, center=True, min_periods=24).mean().resample("D").mean().dropna())
sst_d = (buoy["SST"].interpolate(limit=6)
         .rolling(48, center=True, min_periods=24).mean().resample("D").mean().dropna())

ui_resid = STL(ui_d, period=365, robust=True).fit().resid           # UI：去季節去趨勢
sst_anom = sst_d - sst_d.index.month.map(sst_d.groupby(sst_d.index.month).mean())  # SST：距平 (減一次)
sst_resid_raw = STL(sst_d, period=365, robust=True).fit().resid     # SST：改用 STL 殘差 (對照組)
sst_anom_stl = STL(sst_anom.dropna(), period=365, robust=True).fit().resid  # SST：距平又再 STL (雙重)

WINDOW = slice("2016-04-10", "2016-10-15")


def xcorr_best(x, y, max_lag=15, window=WINDOW):
    """對齊兩條序列、取窗、回傳 (lags, r, best_lag, r_at_best)。"""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna().loc[window]
    lags, rs, ps, best = ts.lagged_xcorr(d["x"], d["y"], max_lag=max_lag)
    return lags, rs, best, rs[list(lags).index(best)]


def stem_panel(ax, lags, rs, best, title):
    ax.stem(lags, rs, basefmt="k-")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(best, color="red", ls="--", lw=1)
    ax.set_title(f"{title}\nbest lag={best:+d}, r={rs[list(lags).index(best)]:+.2f}", fontsize=10)
    ax.set_xlabel("Lag (d): UI leads SST →"); ax.set_ylabel("r")

# %% [markdown]
# ## U1. 主文用法：UI 殘差 vs SST 距平
# 這是 C 段的結果，當基準線：lag=+1 天、r≈−0.33（湧升領先 SST 降溫）。

# %%
lA, rA, bA, vA = xcorr_best(ui_resid, sst_anom)
print(f"[A] UI_resid vs SST_anom        : best lag={bA:+d}, r={vA:+.2f}  ← 主文用法")

# %% [markdown]
# ## U2. 對照組：如果 SST 也改用「STL 殘差」(而不是距平) 會怎樣？
# 把 SST 改成 `STL(raw SST).resid`，其餘不變，比較兩者。

# %%
lB, rB, bB, vB = xcorr_best(ui_resid, sst_resid_raw)
print(f"[B] UI_resid vs SST_resid (STL) : best lag={bB:+d}, r={vB:+.2f}  ← 退化")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
stem_panel(axes[0], lA, rA, bA, "A) SST = anomaly  (correct)")
stem_panel(axes[1], lB, rB, bB, "B) SST = STL residual of raw SST")
plt.tight_layout()
plt.show()

# %% [markdown]
# 結果：**A（距平）乾淨地在 lag=+1 給出 −0.33**；
# **B（SST 改用 STL 殘差）卻退化**（最佳落後跑到邊界、符號還變正）。
# 原因：`STL(raw SST)` 的趨勢較有彈性，會把訊號處理得和距平不同（呼應 A 段 `trend=` 的討論）。
# → **要把 SST 的季節拿掉，用「減月氣候平均（距平）」比用 STL 乾淨。**

# %% [markdown]
# ## U3. 那「距平之後再做一次 STL」呢？——多餘
# 既然距平已經去過季節，對距平再做 STL 應該幾乎沒差。驗證看看。

# %%
lC, rC, bC, vC = xcorr_best(ui_resid, sst_anom_stl)
print(f"[C] UI_resid vs STL(SST_anom)   : best lag={bC:+d}, r={vC:+.2f}  ← 和 A 幾乎一樣")
print()
print("→ 距平又再 STL，結果和 A 幾乎相同：因為距平本來就沒有季節了，再 STL 幾乎沒東西可拿掉。")
print("  所以對 SST『做一次距平』就夠了；再 STL 是多餘的（而且有過度刪除訊號的風險）。")

# %% [markdown]
# ## U4. 另一個常見開關：一階差分 (differencing)
# 差分 = 用「逐日變化量」取代「水準值」，可去掉時間自相關，讓 lag 更銳利；
# 但它**改變了問的問題**（變成比較「變化量」而非「異常值」），也會移除較長尺度的 pattern。

# %%
lD, rD, bD, vD = xcorr_best(ui_resid.diff(), sst_anom.diff())
print(f"[D] diff(UI_resid) vs diff(SST_anom): best lag={bD:+d}, r={vD:+.2f}")
print()
print("差分後相關更強 (約 -0.51)，lag 仍是 +1——短期『風變強→隔天海溫掉』看得更清楚。")
print("但它測的是『日變化量』而非『異常水準』，較長期的關係被差掉了，所以主文預設不開差分，")
print("保持『比較異常值』的直覺。要不要差分是一個『取捨』，沒有絕對對錯。")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
stem_panel(axes[0], lA, rA, bA, "A) levels: UI_resid vs SST_anom")
stem_panel(axes[1], lD, rD, bD, "D) differenced (day-to-day change)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## U5. 小結
# | 變體 | best lag | r | 解讀 |
# |---|---|---|---|
# | A 主文：UI殘差 vs SST距平 | +1 | −0.33 | 乾淨、可解釋 ✓ |
# | B SST 改用 STL 殘差 | 邊界 | +0.27 | 退化、符號反 ✗ |
# | C SST 距平又再 STL | +1 | −0.33 | 和 A 一樣 → 多餘 |
# | D 一階差分 | +1 | −0.51 | 更銳利，但改成比「變化量」、取捨 |
#
# **重點**：怎麼「去掉可預期的部分」是一個會影響結論的選擇——
# SST 用距平（減一次）剛剛好；再 STL 多餘、改用 STL-of-raw 反而更差。
# 這正是整堂的主軸再現一次：**處理/定義的選擇，會改變你看到什麼。**

# %%
print("摘要：")
for tag, b, v in [("A 主文 (距平)", bA, vA), ("B SST也STL", bB, vB),
                  ("C 距平再STL", bC, vC), ("D 差分", bD, vD)]:
    print(f"  {tag:16s}: best lag={b:+d}, r={v:+.2f}")
