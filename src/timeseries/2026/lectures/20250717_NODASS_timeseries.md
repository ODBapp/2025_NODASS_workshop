# 時間序列資料處理

### 以海洋熱浪與氣象浮標資料為例

翁其羽 — 國科會海洋學門資料庫 (ODB)
2025/07/17 @ 諾大師 — 海洋大數據競賽

> 本檔由 `lectures/20250717_NODASS_timeseries.pdf`（PowerPoint 匯出）轉寫而成。
> 文字為逐頁精確轉寫、方便直接複製到 Jupyter notebook 的 markdown cell；
> 每頁附上原始投影片圖 `images/slideNN.png`（整頁 render，含所有圖表）。
> 抽出的內嵌小圖多因透明度/CMYK 而不可靠，故一律以整頁圖為準。

---

## 投影片 1 — 封面

**時間序列資料處理 — 以海洋熱浪與氣象浮標資料為例**

翁其羽

![slide01](images/slide01.png)

---

## 投影片 2 — ODB Open API

網址：<https://api.odb.ntu.edu.tw/hub>

- **海洋研究船長期蒐集台灣周遭海域之船測資料**
  - **ODB CTD API**：溫鹽深儀（CTD）暨附屬探針資料：溫度、鹽度、溶氧、透光度、螢光度（39 yrs, 18 million records）
  - **ODB SADCP API**：船載都卜勒流剖儀（SADCP）海流 u、v 流速、流向資料（185 million records）
- **海洋科學家貢獻之資料集，或整理自其他公開文獻、合作計畫等**
  - **ODB ECO API**：海洋生物出現記錄（主要為浮游動物、仔稚魚等，172,691 records）與微塑膠資料集（23,189 records）
- **轉製海洋科學文獻方法、或由其他全球公開資料庫編譯**
  - **Marine Heatwaves (MHW) API**：全球海洋熱浪：1982- 迄今月均海表溫及距平值、熱浪級數等（533.95 million records）
  - **GEBCO (2026 Grid) Bathymetry API**：全球地形高程資料，15 弧秒（約 400 m）解析（3.732 billion records）
  - **Tide API**：TPXO10-atlas-v2 全球潮汐模型（874.96 million records）
  - **WOA23 API**：World Ocean Atlas 2023：溫度、鹽度、營養鹽、溶氧相關等長期平均資料（324.26 million records）
  - **GLODAP API**：GLODAP v2.2023（近 20 種水文化學參數）（1.49M global bottle data from 1,116 cruises, 1972–2021）
- **工具型 API**
  - **OGC WMS/WMTS Capabilities API**：查詢支援開放地理空間協會（OGC）WMS/WMTS 標準之地理圖層資訊

![slide02](images/slide02.png)

---

## 投影片 3 — Time-series Lecture Targets

- 找到適當的資料來源：**Open API**、網頁、文獻，並檢視其時空模式（**patterns**）
- 處理時間尺度或精細度不同的時間序列關係
- 初探時間序列分解（**decomposition**）

**資料來源：**
- 中央氣象署氣象浮標資料（**NODASS** 提供）、**ODB** 海洋熱浪 **MHW API**

**附註：**
- 時間有限，科普（如聖嬰、**Ekman transport**）、程式指令細節、套件安裝，請自行 **ask google or AI**

![slide03](images/slide03.png)

---

## 投影片 4 — The Oceanic Niño Index (ONI)

**The Oceanic Niño Index (ONI)**：running 3-month mean SST anomaly for the Niño 3.4 region (5°N–5°S, 120°–170°W)

**El Niño / La Niña event**：5 consecutive overlapping 3-month periods > / < ±0.5 °C

圖：Sea Surface Temperature Anomaly（Jan 2009）與 NIÑO 3.4 SST Anomaly Timeline & Disease Outbreaks。
by Helen-Nicole Kostis，from <https://svs.gsfc.nasa.gov/4765>

![slide04](images/slide04.png)

---

## 投影片 5 — ONI, El Niño and La Niña

- Niño Index Regions (SST)（from climatedataguide.ucar.edu）
- (a) SST Anomalies in DJF 1997/1998、(b) SST Anomalies in DJF 1973/1974
  （from www.cwa.gov.tw/V8/C/C/Knowledge/knowledge_4-1.html）
- El Niño / Normal / La Niña Conditions 三種大氣—海洋環流示意（from <https://rsiasacademy.com/el-nino-upsc/>）
- Oceanic Niño Index (ONI) time-series：各次事件疊合曲線（Developing → Maturity），紅色 1997/98、藍色 1973/74
- 聖嬰現象 (El Niño) / 反聖嬰現象 (La Niña) / 中性 (Neutral) 之台灣氣候意義
  （資料來源 NOAA Climate.gov；<https://climate.cwa.gov.tw/ENSO?subpage=Intro>）

**中性 (Neutral)**：不屬於聖嬰、也不為反聖嬰現象的氣候狀態；或是海洋符合聖嬰／反聖嬰現象，但大氣壓力變化卻不符合（反之亦然）。

![slide05](images/slide05.png)

---

## 投影片 6 — Result：利用 ODB MHW API 檢視 Niño 3.4 SST Anomaly pattern

- 左：Open API of Marine Heatwaves 查詢介面
  （<https://api.odb.ntu.edu.tw/hub/swagger?node=odb_mhw_v1>）
- 中：SST Anomalies in Central Pacific (Niño 3.4) — Dec 2007，標出 Niño 1+2 / 3 / 4 / 3.4 區域框
- 右：SST Anomalies Time Series at Niño 3.4 region
  - 上：Niño 3.4 SST Anomaly **from NOAA**
    （<https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data>）
  - 下：Niño 3.4 SST Anomaly **from MHW API**

兩者形狀一致，示範用 ODB MHW API 可重現 NOAA 的 Niño 3.4 距平序列。

![slide06](images/slide06.png)

---

## 投影片 7 — El Niño and Disease Outbreaks

- 時間序列轉化為適當的**事件**資料，常有助於統計分析，尤其面對非線性、不同時間規模或精細度的資料。
- 事件發生的**頻繁**與否、如何**分類**都可以著力，但須合理。
- 有相關性或顯著差別，並不能推演為具有**因果性（causality）**。

圖：Tanzania 霍亂案例 vs ENSO（Nov–Jan），以及 Cholera distribution by ENSO phase (2000–2017) 的箱型圖。
紅字「說好的 match 呢？」點出相關不必然成立。
From Fig.4. Anyamba, A., Chretien, JP., Britch, S.C. et al. *Global Disease Outbreaks Associated with the 2015–2016 El Niño Event.* Sci Rep 9, 1930 (2019). <https://doi.org/10.1038/s41598-018-38034-z>

![slide07](images/slide07.png)

---

## 投影片 8 — Time-series Decomposition by STL

- **STL**：Seasonal and Trend decomposition using LOESS
- 為何須了解時間序列中的趨勢、季節性與殘差（**residuals**）？
- **STL** 有何限制？

程式示意（模擬 趨勢 + 季節 + 噪音，再用 STL 拆解）：

```python
# 1. Simulate trend + seasonality + noise
np.random.seed(2024)
n = 100
t = np.arange(1, n+1)
time_index = pd.date_range("2000-01-01", periods=n, freq="ME")

mu = -5 + 0.1 * t                    # 線性趨勢
ss = np.sin(2 * np.pi * t / 12)      # 每 12 期的季節性
e  = 0.5 * np.random.randn(n)        # 隨機噪音
y  = mu + ss + e

df = pd.DataFrame({"y": y, "signal": mu + ss}, index=time_index)
```

```python
from statsmodels.tsa.seasonal import STL

stl = STL(df["y"],
          period=12,
          robust=True).fit()
fig = stl.plot()
```

圖：Time-series with Trend and Seasonality、以及 Seasonal Decomposition of SST（y / Trend / Season / Resid 四面板）。

![slide08](images/slide08.png)

---

## 投影片 9 — Time-series Decomposition by EEMD（進階參考）

- **EEMD**：Ensemble empirical mode decomposition (EMD)
- EMD 這種分解技術，不依賴於穩態假設，將訊號分解為具有不同振盪模式的基函數。
- 傳統傅立葉（**FFT**）告訴你有哪些固定頻率存在，就好像把原始訊號拆解成一個個固定但不同頻率的正弦波組成。但從左例可以看到：真實世界的振盪很少是純正弦波，而是調幅調頻的複合式振盪 + 隨機變數。
- **IMF（Intrinsic Mode Function）** 代表一個特徵時間尺度（模態）的振盪成分。
- 迭代尋找每個 IMF：連接訊號的極大、極小值形成上下包絡線（**envelope**），移除包絡線平均後，檢查是否符合 IMF 條件。
- **IMF 條件**：極值點數量與過零點數量相等或至多差一，且上下包絡線的平均趨近零。
- 若符合，得到 IMF₁，繼續迭代找下一個 IMF。IMF₁：最快的振盪、IMF₂：次快…
- 為什麼？
  - 原始訊號 = 快振盪 + 慢變化的基線
  - 移除包絡平均 = 移除慢變化基線
  - 結果 = 在該時間尺度上的「純」振盪
- 剩下來的殘差，仍具有長期趨勢成份（含單調趨勢）。
- **EEMD 改進**：每次拆解 IMF，加入白噪音後找尋上下包絡線平均，解決 EMD 的模態混合問題，使每個 IMF 更純淨地代表單一時間尺度。

**Colab 程式碼**：
<https://colab.research.google.com/github/ODBapp/2025_NODASS_workshop/blob/main/src/timeseries/EEMD01.ipynb>

![slide09](images/slide09.png)

---

## 投影片 10 — 氣象浮標資料處理

- 氣象浮標為連續觀測之時間序列資料，處理上需注意缺值、離群值與在時間尺度上的處理。從不同的資料變數月平均可知具有明顯的季節性耦合，尤其是複合性的週期訊號，在處理相關性分析也須留意。
- 和更長時間規模的 **Niño 3.4 SST** 距平值相比，顯示氣象浮標事件 **SST** 距平值受季風因素影響有顯著的季節性週期，和 **Niño 3.4 SST** 距平值差異甚大。
- 進階參考範例中使用 **EEMD** 處理示性波高（**Significant wave height**）趨勢、週期性可和 **STL** 結果比較。
- 以下將使用文獻中利用風速、風向等資料計算湧升流的湧升指數（**Upwelling Index, UI**）為例。

圖：Longdong 浮標 Monthly-mean values (2010–2024) 的 Wind speed / SST；以及 Niño 3.4 SST Anomaly (NOAA) 與 Longdong SST Anomaly 對比。

![slide10](images/slide10.png)

---

## 投影片 11 — 氣象浮標資料：由風速、風向計算湧升指數

The daily composited wind data were first calculated from the 6-hourly data. The daily UI (in m² s⁻¹) was then calculated using the following equations:

$$ \mathrm{UI} = \frac{\tau}{f\,\rho_w}\cos(\alpha - \beta) \tag{1} $$

$$ \tau = \rho_a\,C_d\,V^2 \tag{2} $$

$$ f = 2\,\omega\,\sin(\varphi) \tag{3} $$

$$ C_d = (0.8 + 0.065\,V)\times 10^{-3} \tag{4} $$

其中 τ 為風應力，ρ_a 為空氣密度（1.22 kg m⁻³），ρ_w 為海水密度（1026 kg m⁻³），C_d 為風阻力係數（由經驗式計算，Huang et al. 2021），f 為科氏參數（s⁻¹），α 為風向，β 為岸線走向（general shoreline orientation），ω 為地球自轉角速度（7.2921×10⁻⁵ rad s⁻¹），φ 為緯度，V 為風速（m s⁻¹）。

- 計算風應力 τ，並考慮柯氏力所造成 **Ekman** 作用而形成離岸流，帶動湧升流；意謂 **Ekman** 作用把表水推離岸 → 較冷的深層水上湧。計算所得之湧升指數（**Upwelling Index, UI**）正值代表湧升。

對應的實作（`ts_utils.upwelling_index` 同源）：

```python
RHO_AIR = 1.22      # air density in kg m-3
RHO_W   = 1026.0    # seawater density in kg m-3
def drag_coef(V):                     # 阻力係數的經驗公式
    return (0.8 + 0.065 * V) * 1e-3   # Huang et al. 2021

def calc_ui(df: pd.DataFrame, site_name: str, coast_angle: float) -> pd.Series:
    """由浮標 dataframe 產生月平均 Upwelling Index (UI)，需手動指定 site 與岸線角度。

    Parameters
    ----------
    df : DataFrame
        需含欄位 'Wind'(m/s)、'Wind_Dir'(deg, dir-from)
    site_name : str
        浮標所屬站點名稱，會自動帶入 CWB_SITES 對應的緯度
    coast_angle : float
        岸線走向角度 (deg, 順時針自真北)；例：文獻中台灣東部海域北段 ~= 18°
    Returns
    -------
    Series (DatetimeIndex, 月頻率): upwelling index 單位為 m²/s
    """
    lat = CWA_SITES[site_name][0]
    V   = df["Wind"]
    a   = (df["Wind_Dir"] + 180.) % 360   # 風向 → 風去向

    # cos(a-b) is the projection vector of t: wind stress
    cos_ab = np.cos(np.deg2rad(a - coast_angle))
    Cd  = drag_coef(V)
    tau = (RHO_AIR * Cd * V**2) * cos_ab            # t: wind stress · cos(a-b)

    f  = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))    # Coriolis parameter (s-1)
    ui = tau / (RHO_W * f)                          # m² s⁻¹ (與論文一致)
    return ui
```

by Paul Webb（岸線—風—湧升示意圖）

![slide11](images/slide11.png)

---

## 投影片 12 — 氣象浮標資料 → 逐日湧升指數

- 湧升流為較冷的深層水上湧，則湧升指數與 **SST** 的關係會是？如何分析？

圖：Upwelling Index (2016 / 2017 / 2018) 逐日序列（左），以及與 Huang et al. 2021 Fig.3 的比對（右）。
From Huang et al., 2021 Fig.3

![slide12](images/slide12.png)

---

## 投影片 13 — 逐日湧升指數 vs SST Anomalies

- 湧升指數 **UI**（以 **STL** 移除趨勢、季節性）領先 **SST** 距平值 **+1 lag** 顯著負相關。
- 為何 **UI** 用 **STL**，但 **SST** 距平值不以此處理？
- 但其實並不是每一段逐日 **UI** 都和 **SST** 距平值有這麼顯著關係，**SST** 仍受其他許多外在因素影響。

圖三面板：
- Upwelling and SST Anomalies Overlay（UI vs SST Anomalies）
- STL Residuals and Lagged SST Anomalies Overlay（ui_resid (z) vs anom_resid (lag=−1)）
- Cross-correlation of Residuals（**Max r = −0.38 at lag −1**）

![slide13](images/slide13.png)

---

## 投影片 14 — Take Home Messages

- 開源函式庫 **+ AI + Vibe coding** 已經使資料處理技術不再是最大門檻
  - 了解資料的本質與如何應用，才能驅動 **AI** 幫忙做出有用的事
  - 做為入門的第一步：找到適合的科學文獻參考，幫助頗大！
- 初探時間序列：
  - 時間序列的相關性、因果關係，往往存在錯誤陷阱與迷思
    - 時間拉長，真實世界的時間訊號往往都具非線性、複雜因子，所謂的相關性往往不復存在
  - 將連續的時間序列，轉換成指數、事件，常有助於對影響因子的統計分析

![slide14](images/slide14.png)
