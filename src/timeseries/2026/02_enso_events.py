# %% [markdown]
# # B. 把序列變成「事件」，再談相關（與相關 ≠ 因果）
# 對應敘事文件: narrative/02_enso_events.md
# 時間預估：約 25 分鐘
#
# 承接 A 段主軸：距平讓 ENSO 訊號浮現。這一段把連續的距平序列**抽象成離散事件**
# (聖嬰/反聖嬰)，再用事件去談相關——並親手示範「定義方式會翻轉統計結論」。
#
# 註：圖上文字一律英文 (避免 Colab 缺中文字型)；中文說明在 .md 與註解。

# %%
# B0. 套件與工具
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import ts_utils as ts

# %% [markdown]
# ## B1. 先看 Niño 3.4 在哪裡：太平洋 SST 距平地圖
# 用 cartopy 畫一張跨換日線的赤道太平洋距平圖（2015-12，超強聖嬰）。
# 技巧：經度轉 0–360 後資料是連續的（不會在 180° 斷開），
# 投影用 `PlateCarree(central_longitude=180)` 把太平洋擺中間，資料用 `transform=PlateCarree()`。

# %%
def plot_enso_map(date="2015-12", note="strong El Niño"):
    df = ts.load_enso_map(date)            # 線上優先、失敗用快取
    df["lon360"] = df["lon"] % 360         # 轉 0–360：跨太平洋變連續
    grid = df.pivot_table(index="lat", columns="lon360", values="sst_anomaly")
    LON, LAT = np.meshgrid(grid.columns.values, grid.index.values)

    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 4.5))
    ax = plt.axes(projection=proj)
    ax.set_extent([135, 300, -25, 25], crs=data_crs)

    im = ax.contourf(LON, LAT, grid.values, levels=np.linspace(-3, 3, 21),
                     cmap="RdYlBu_r", extend="both", transform=data_crs)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray", zorder=3)
    ax.coastlines("110m", lw=0.6, zorder=4)

    # Niño 區域框 (經度用 0–360)
    boxes = {"Niño 4": (160, 210, -5, 5), "Niño 3.4": (190, 240, -5, 5),
             "Niño 3": (210, 270, -5, 5), "Niño 1+2": (270, 280, -10, 0)}
    for name, (a, b, c, d) in boxes.items():
        lw = 2.0 if name == "Niño 3.4" else 1.0
        ax.plot([a, b, b, a, a], [c, c, d, d, c], transform=data_crs, lw=lw, label=name)

    gl = ax.gridlines(draw_labels=True, lw=0.3, ls="--")
    gl.top_labels = gl.right_labels = False
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02,
                 label="SST anomaly (°C)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_title(f"Pacific SST anomaly — {date}  ({note})")
    # 注意：cartopy 的 gridline labels 與 plt.tight_layout() 會衝突，這裡不要用 tight_layout
    plt.show()


# 想換成反聖嬰範例可改成: plot_enso_map("2007-12", note="La Niña")
plot_enso_map("2015-12", note="strong El Niño")

# %% [markdown]
# ## B2. 連續序列 → 指數 → 事件
# 三步驟把「距平」變成大家講的「聖嬰年/反聖嬰年」：
# 1. **ONI** = Niño 3.4 距平的 3 個月移動平均（NOAA 官方定義）。
# 2. **門檻**：ONI ≥ +0.5°C 偏暖、≤ −0.5°C 偏冷。
# 3. **事件**：要連續 ≥ 5 個月超過門檻，才算一次聖嬰/反聖嬰事件（濾掉短暫雜訊）。

# %%
ssta = ts.load_noaa_nino34("1950", "2025")   # NOAA Niño3.4 月距平 (線上優先/快取)
ssta["oni"] = ssta["ssta"].rolling(3, center=True).mean()


def classify_enso_events(df, min_months=5, threshold=0.5):
    """把 ONI 序列切成 聖嬰/反聖嬰 事件 (連續 ≥ min_months 個月超過門檻)。"""
    def phase_of(x):
        if x >= threshold:
            return "El Niño"
        if x <= -threshold:
            return "La Niña"
        return "Neutral"

    d = df.copy()
    d["phase"] = d["oni"].apply(phase_of)
    d["run"] = (d["phase"] != d["phase"].shift()).cumsum()   # 連續同相位編號
    events = []
    for _, g in d.groupby("run"):
        ph = g["phase"].iloc[0]
        if ph in ("El Niño", "La Niña") and len(g) >= min_months:
            peak = g["oni"].max() if ph == "El Niño" else g["oni"].min()
            events.append({"start": g.index[0], "end": g.index[-1],
                           "phase": ph, "peak_oni": peak})
    return pd.DataFrame(events)


events = classify_enso_events(ssta)
print(events.tail(6).to_string(index=False))

# %% [markdown]
# ## B3. 事件長條圖：把整段歷史的聖嬰/反聖嬰標出來

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 1]})
# 上：ONI 曲線 + 門檻
axes[0].plot(ssta.index, ssta["oni"], color="cornflowerblue", lw=1)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].axhline(0.5, color="red", ls=":", lw=0.8)
axes[0].axhline(-0.5, color="blue", ls=":", lw=0.8)
axes[0].set_ylabel("ONI (°C)")
axes[0].set_title("Oceanic Niño Index (3-month running mean of Niño 3.4 anomaly)")

# 下：把事件期間塗成長條 (紅=聖嬰, 藍=反聖嬰)
for _, ev in events.iterrows():
    color = "tab:red" if ev["phase"] == "El Niño" else "tab:blue"
    vals = ssta.loc[ev["start"]:ev["end"], "oni"]   # 直接用真實索引切片 (day=15)
    axes[1].bar(vals.index, vals.values, width=25, color=color, alpha=0.7)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_ylabel("ENSO events\n(red=El Niño, blue=La Niña)")
axes[1].xaxis.set_major_locator(mdates.YearLocator(10))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## B4. 用事件談相關——以及一個刻意的陷阱
# 經典問題：聖嬰會不會影響遠方的疾病？這裡用一篇研究的坦尚尼亞**霍亂**年資料
# (已去趨勢) 對上 ENSO 事件
# （[Anyamba et al. 2019, *Sci Rep* 9:1930](https://doi.org/10.1038/s41598-018-38034-z)）。
#
# 關鍵教學點：**怎麼把「跨年的 DJF (12–1–2月) 事件」指派給哪一年，是人為決定。**
# 以「2006年12月～2007年2月 ONI 突破門檻」的聖嬰為例，兩種指派都說得通：
#
# - **定義 A（一般慣例）**：以事件「起始年」命名 → **2006 是聖嬰年**，
#   於是 2006 全年的霍亂就對上 2006 的聖嬰。
# - **定義 B（影響年觀點）**：聖嬰峰值在冬季，強降雨、積水、排水系統癱瘓等
#   公衛衝擊多半出現在**峰值之後的那一年** → 把同一個 DJF 改算給 **2007**，
#   讓「聖嬰年」對齊「受影響的那一年」。
#
# 兩種都合理——但它們只差一個年份的對齊選擇。下面親手做兩份，看結論會不會一樣。

# %%
cholera = ts.load_cholera()


def annual_enso_phase(ssta_df, years, assign_to_next_year):
    """
    把跨年的冬季 DJF (12–1–2 月) ONI 指派成「某一年 yr」的 ENSO 相位。
    以「2006-12 ~ 2007-02 的 ONI 突破門檻」這次聖嬰為例：

    assign_to_next_year=False（定義 A，一般慣例）:
        yr 的相位 = DJF(今年12月, 明年1月, 明年2月) → 2006 算聖嬰年
        （事件以「起始年」命名；把 2006 全年霍亂對上 2006 聖嬰）。
    assign_to_next_year=True （定義 B，影響年觀點）:
        yr 的相位 = DJF(去年12月, 今年1月, 今年2月) → 同一個 DJF 改算給 2007
        （聖嬰峰值在冬季，對降雨/公衛的衝擊多落在「隔年」，故對齊受影響的那一年）。
    """
    rows = []
    for yr in years:
        base = yr - 1 if assign_to_next_year else yr
        djf_months = [pd.Timestamp(base, 12, 15),
                      pd.Timestamp(base + 1, 1, 15),
                      pd.Timestamp(base + 1, 2, 15)]
        try:
            oni = ssta_df.loc[djf_months, "oni"].mean()
        except KeyError:
            oni = np.nan
        phase = ("El Niño" if oni >= 0.5 else "La Niña" if oni <= -0.5 else "Neutral")
        rows.append({"year": yr, "oni_djf": oni, "enso_phase": phase})
    return pd.DataFrame(rows)


# 兩種指派各做一份合併表
merged_A = cholera.merge(annual_enso_phase(ssta, cholera["year"], False), on="year")
merged_B = cholera.merge(annual_enso_phase(ssta, cholera["year"], True), on="year")

order = ["El Niño", "Neutral", "La Niña"]
palette = {"El Niño": "#f8766d", "La Niña": "#619cff", "Neutral": "lightgray"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (mdf, title) in zip(
        axes, [(merged_A, "Definition A: DJF → that year"),
               (merged_B, "Definition B: same DJF → next year")]):
    sns.boxplot(data=mdf, x="enso_phase", y="detrended_cholera_cases", order=order,
                hue="enso_phase", palette=palette, legend=False,
                showfliers=False, width=0.5, ax=ax)
    sns.stripplot(data=mdf, x="enso_phase", y="detrended_cholera_cases", order=order,
                  color="black", size=4, jitter=0.12, ax=ax)
    means = mdf.groupby("enso_phase")["detrended_cholera_cases"].mean()
    ax.set_title(f"{title}\nEl Niño mean = {means.get('El Niño', float('nan')):.0f}")
    ax.set_xlabel(""); ax.axhline(0, color="gray", lw=0.6)
axes[0].set_ylabel("Detrended cholera cases")
plt.suptitle("Same data, two reasonable definitions → opposite conclusions", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# 把兩種定義下「聖嬰年平均霍亂」並排印出來，數字最有說服力
for label, mdf in [("A: DJF→that year", merged_A), ("B: DJF→next year", merged_B)]:
    means = mdf.groupby("enso_phase")["detrended_cholera_cases"].mean().round(0)
    print(label, "->", means.to_dict())
print("\n結論：只是換了 DJF 指派給哪一年，聖嬰年的霍亂訊號就從『明顯偏高』變成『幾乎沒差』。")
print("讀 paper 時務必確認：作者怎麼定義事件、怎麼對齊時間？相關 ≠ 因果。")
