# %% [markdown]
# # 附錄. EEMD：當 STL 不夠用時的資料驅動分解
# 對應敘事文件: narrative/appendix_eemd.md
# **這一段不在課堂時間內，留給有興趣的學員自學。難度明顯比 A/B/C 高。**
#
# 主旨：
# - **STL 必須先指定一個固定週期** (例如 period=12)，只給你「一個季節 + 一個趨勢」。
#   如果資料其實有**多個**週期，STL 沒辦法把它們分開，趨勢也會被汙染。
# - **EEMD (Ensemble Empirical Mode Decomposition)** 是**資料驅動**的：不必指定週期，
#   它把序列拆成多個 IMF (本徵模態)，每個 IMF 大致對應一個時間尺度。
#   再把「最慢的幾個 IMF + 殘差」加起來，就能抽出**長期趨勢**。
#
# 流程：① 人工合成例 (STL 抓不到、EEMD 抓得到長期趨勢) → ② 真實案例 Niño 3.4 SSTA
# 與龍洞 Hs (趨勢都有點微弱，但 EEMD 仍抓得出來)。
#
# 註：圖上文字用英文 (避免 Colab 缺中文字型)；中文說明在 .md 與註解。

# %%
# E0. 套件與工具
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess
from PyEMD import EEMD

import ts_utils as ts


# ---- 共用小工具 (放在這裡讓附錄自成一檔) ----
def eemd_decompose(series, trials=100, noise_width=0.2, seed=2026):
    """對 Series 做 EEMD，回傳 (對齊後的 series, IMFs 陣列, 殘差陣列)。先補小缺口。

    - `parallel=False` (單行程)：EEMD 預設會開多行程，在 macOS / Colab / 一般 .py 腳本裡
      常因為缺少 `if __name__=='__main__'` 守衛而報 RuntimeError。單行程最穩健、這個資料量也夠快。
    - `noise_seed(seed)` 固定亂數：EEMD 靠「加隨機噪音再平均」，不固定種子每次結果會略有不同；
      固定後學員每次跑都得到相同的圖。
    """
    s = series.interpolate("time").dropna()
    x = s.values
    eemd = EEMD(trials=trials, noise_width=noise_width, parallel=False)
    eemd.noise_seed(seed)
    eemd.eemd(x, np.arange(len(x)))
    imfs, residual = eemd.get_imfs_and_residue()
    return s, imfs, residual


def imf_peak_periods(imfs, dt=1.0):
    """用 FFT 找每個 IMF 的主週期 (與 dt 同單位)。"""
    n = imfs.shape[1]
    freqs = np.fft.rfftfreq(n, d=dt)
    rows = []
    for i, imf in enumerate(imfs, start=1):
        power = np.abs(np.fft.rfft(imf)) ** 2
        k = np.argmax(power[1:]) + 1               # 跳過 DC (第 0 項)
        f = freqs[k]
        rows.append({"IMF": i, "peak_period": (1 / f if f > 0 else np.inf)})
    return pd.DataFrame(rows)


def extract_long_term(imfs, residual, periods, threshold):
    """把『主週期 > threshold』的 IMF 連同殘差加起來 = 長期成分。"""
    long_component = residual.copy()
    picked = []
    for i, (imf, p) in enumerate(zip(imfs, periods), start=1):
        if p > threshold:
            long_component = long_component + imf
            picked.append(i)
    return long_component, picked


def plot_imfs(index, imfs, residual, title):
    """把所有 IMF 與殘差逐層畫出來。"""
    n = imfs.shape[0]
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 1.7 * (n + 1)), sharex=True)
    for i in range(n):
        axes[i].plot(index, imfs[i], lw=0.8)
        axes[i].set_ylabel(f"IMF{i+1}")
    axes[-1].plot(index, residual, "k", lw=1.2)
    axes[-1].set_ylabel("Residual")
    fig.suptitle(title, y=1.00)
    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## E1. 人工合成例：放進「多個」週期 + 趨勢 + 噪音
# 故意做一條同時有 12 個月、43 個月、約 12 年 (151 月) 三種週期，外加線性趨勢與噪音的序列。
# 因為是我們自己造的，**知道正確答案**。

# %%
rng = np.random.default_rng(41)
n = 600
t = np.arange(n)
idx = pd.date_range("1970-01-01", periods=n, freq="MS")
trend_true = -2 + 0.003 * t                                   # 線性趨勢 (真值)
y = (trend_true
     + 0.8 * np.sin(2 * np.pi * t / 12)                       # 年週期
     + 0.5 * np.cos(2 * np.pi * t / 43 + rng.uniform(0, np.pi))   # 中期 ~3.6 年
     + 0.6 * np.sin(2 * np.pi * t / 151 + rng.uniform(0, np.pi))  # 長期 ~12.6 年
     + 0.5 * rng.standard_normal(n))                          # 噪音
sim = pd.Series(y, index=idx)

plt.figure(figsize=(12, 3))
plt.plot(idx, sim, lw=0.9, label="Composite series")
plt.plot(idx, trend_true, "r", lw=2, label="True linear trend")
plt.title("Synthetic series: 3 periodicities + linear trend + noise")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## E2. STL 的侷限：只給「一個」季節 + 一個趨勢
# STL 必須指定 period=12。它會把 12 個月當季節，**其他週期 (43、151 月) 沒地方放**，
# 只好被擠進 trend 和 residual——所以 STL 的「趨勢」其實混進了那條 12 年的慢振盪，不乾淨。

# %%
stl = STL(sim, period=12, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(12, 8)
fig.suptitle("STL (period=12): other cycles get smeared into trend/residual", y=1.00)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## E3. EEMD：資料驅動，把每個尺度分開
# 不指定週期，EEMD 自己把序列拆成多個 IMF。

# %%
sim_s, imfs, residual = eemd_decompose(sim, trials=100)
plot_imfs(sim_s.index, imfs, residual, "EEMD decomposition of the synthetic series")

# %% [markdown]
# ## E4. 每個 IMF 的主週期：EEMD 把放進去的尺度都找回來了
# 用 FFT 找各 IMF 的主週期，應該會看到接近 12、43、151 月的成分。
#
# 📌 注意：最慢的那幾個 IMF 因為**不足一個完整週期**，FFT 能解析的最低頻率只有 1/N，
# 所以它們的「主週期」會頂到序列長度本身 (≈ N，這裡 N=600 個月)。這不是 bug——
# 「週期 ≈ N」正好就是把它們歸類為『長期成分』的依據 (見 E5)。

# %%
periods_df = imf_peak_periods(imfs, dt=1.0)   # 月資料 dt=1 → 週期單位是「月」
print(periods_df.round(1).to_string(index=False))

# %% [markdown]
# ## E5. 重建長期趨勢：挑「最慢的 IMF + 殘差」，再 LOWESS 平滑
# 把主週期 > N/3 個月的 IMF 連同殘差加起來，就是長期成分；再用 LOWESS 平滑成趨勢線。
# 和真值 (紅線) 比，EEMD 抽出的長期趨勢比 STL 的 trend 乾淨許多。

# %%
periods = periods_df["peak_period"].values
long_c, picked = extract_long_term(imfs, residual, periods, threshold=n / 3)
long_smooth = lowess(long_c, np.arange(len(long_c)), frac=0.3, return_sorted=False)
print("挑出的長期 IMF (1-based):", picked)

plt.figure(figsize=(12, 3))
plt.plot(sim_s.index, trend_true, "r", lw=2, label="True trend")
plt.plot(sim_s.index, stl.trend.values, color="gray", ls="--", label="STL trend (contaminated)")
plt.plot(sim_s.index, long_smooth, "b", lw=2, label="EEMD long-term (IMFs>N/3 + resid, LOWESS)")
plt.title("Long-term trend: EEMD recovers it cleaner than STL")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## E6. 真實案例①：Niño 3.4 SSTA 的長期趨勢 (微弱暖化)
# 拿真的 Niño 3.4 月距平，用同一套 EEMD 流程抽長期趨勢。
# 趨勢很微弱 (本來就該如此——距平已扣掉季節)，但 EEMD 仍抓得出緩慢的變化。

# %%
ssta = ts.load_noaa_nino34("1950", "2025")["ssta"]
ssta_s, imf_ssta, res_ssta = eemd_decompose(ssta, trials=100)
per_ssta = imf_peak_periods(imf_ssta, dt=1.0)["peak_period"].values
long_ssta, picked_ssta = extract_long_term(imf_ssta, res_ssta, per_ssta,
                                           threshold=imf_ssta.shape[1] / 3)
smooth_ssta = lowess(long_ssta, np.arange(len(long_ssta)), frac=0.2, return_sorted=False)

# 不畫 raw：raw 的尺度太大會把微弱趨勢蓋掉。改畫「長期成分 (含 EEMD 殘差)」+ LOWESS 趨勢，
# 小尺度才看得出微弱的長期變化，也印證 EEMD 殘差裡仍帶著慢趨勢。
plt.figure(figsize=(12, 3))
plt.plot(ssta_s.index, long_ssta, color="orange", lw=0.6, alpha=0.5,
         label="Long-term component (slow IMFs + EEMD residual)")
plt.plot(ssta_s.index, smooth_ssta, "r", lw=2.5, label="LOWESS trend")
plt.axhline(0, color="gray", lw=0.5)
plt.title("Niño 3.4 SST anomaly — EEMD long-term component & trend (weak)")
plt.ylabel("°C")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## E7. 真實案例②：龍洞浮標示性波高 Hs 的長期趨勢
# 同樣手法用在龍洞的 Hs (significant wave height) 日資料上。趨勢一樣偏微弱。

# %%
buoy = ts.load_buoy("Longdong")
hs = (buoy["Hs"].rolling("72h").mean().resample("D").mean()
      .interpolate("time", limit=2).dropna())
hs_s, imf_hs, res_hs = eemd_decompose(hs, trials=50)
per_hs = imf_peak_periods(imf_hs, dt=1.0)["peak_period"].values   # 日資料 → 週期單位「天」
long_hs, picked_hs = extract_long_term(imf_hs, res_hs, per_hs,
                                       threshold=imf_hs.shape[1] / 3)
smooth_hs = lowess(long_hs, np.arange(len(long_hs)), frac=0.4, return_sorted=False)

# 同樣不畫 raw：只畫長期成分 (含殘差) + LOWESS 趨勢
plt.figure(figsize=(12, 3))
plt.plot(hs_s.index, long_hs, color="orange", lw=0.5, alpha=0.4,
         label="Long-term component (slow IMFs + EEMD residual)")
plt.plot(hs_s.index, smooth_hs, "b", lw=2.5, label="LOWESS trend")
plt.title("Longdong significant wave height — EEMD long-term component & trend (weak)")
plt.ylabel("Hs (m)")
plt.legend(); plt.tight_layout(); plt.show()

# %%
print("EEMD 抽出的長期趨勢都很微弱——這本身就是誠實的結果：")
print("- Niño 3.4 SSTA 已是距平，長期趨勢本來就小。")
print("- 龍洞 Hs 15 年間沒有強烈的單調變化。")
print("重點是『方法』：EEMD 不必預設週期就能分離多尺度、再重建長期趨勢，")
print("這是 STL (需固定週期) 做不到的。但 EEMD 較慢、結果對參數 (trials, noise_width) 較敏感。")
