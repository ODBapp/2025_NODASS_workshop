# NODASS 2026 — 時間序列教材（2026 改版）

Ocean Data Bank (ODB, IONTU) 為國海院 NODASS 2026 工作坊準備的**時間序列**單元。
這是 2025 版（`../timeseries01.ipynb`、`../EEMD01.ipynb`，純程式碼無說明）的重寫版：
**程式 + 中文教學敘事分離**、加上 fallback 資料與可重現環境，方便學員自學與複習。

> 對象：高中到研究所、程式程度不一的學員（海洋資料專題競賽培訓）。
> 課堂在 Google Colab 上一步步帶；本資料夾是底稿，講師做投影片時會再微調。

---

## 一條主軸貫穿全課

> **訊號常藏在「殘差／距平」裡，而不是原始序列。**

這個觀念用三種尺度反覆出現：**距平**（減氣候平均）→ **STL 殘差**（再去趨勢）→ **事件抽象**（把序列變事件再比較）。

---

## 課程結構（課堂約 60–70 分鐘 + 自學附錄）

| 段 | 主題 | 教的核心技能 | 程式 | 敘事 |
|---|---|---|---|---|
| **A** | 基礎 + STL 分解 | 時間序列三要素、距平、STL（`period`/`robust`/`trend`） | `01_basics_stl.py` | `narrative/01_basics_stl.md` |
| **B** | ENSO 事件 + 相關≠因果 | 把序列抽象成事件；**定義會翻轉結論** | `02_enso_events.py` | `narrative/02_enso_events.md` |
| **C** | 龍洞浮標 + 湧升指數 | 原始觀測→物理量；重用 STL；復現 SCI 論文關係 | `03_buoy_upwelling.py` | `narrative/03_buoy_upwelling.md` |
| 附錄1 | EEMD（**不計課堂時間**） | 資料驅動分解 vs 固定週期；抽長期趨勢 | `appendix_eemd.py` | `narrative/appendix_eemd.md` |
| 附錄2 | UI–SST 互相關參數版（**不計課堂時間**） | 為何 SST 用距平不再 STL；處理方式會影響結論 | `appendix_ui_xcorr.py` | `narrative/appendix_ui_xcorr.md` |

`ts_utils.py`：共用的 fallback-aware 資料載入器與輔助函式（`load_*`、`upwelling_index`、`lagged_xcorr`…）。

---

## 怎麼跑（uv + Python 3.13）

本資料夾是一個獨立的 uv 專案，`.venv` 就建在這裡。

```bash
cd src/timeseries/2026
uv sync                      # 第一次：依 pyproject.toml 建好 .venv 並裝套件
uv run python 01_basics_stl.py   # 跑任一段（會用 plt.show() 顯示圖）
```

- 套件：numpy / pandas / scipy / statsmodels / matplotlib / seaborn / requests / cartopy（+ EMD-signal，只給 EEMD 附錄用）。
- **資料抓取「線上優先、失敗用本機快取」**：課堂網路不穩也能跑（NOAA Niño3.4、ODB MHW API、浮標、霍亂等）。本機快取在 `data/`。

### Colab 用法（給學員）
課堂 notebook `timeseries2026.ipynb` 已由 A/B/C 三個 `.py` 依序組裝好，上傳 Colab 即可用；記得一併上傳 `ts_utils.py`（線上抓資料失敗時另需 `data/` 快取）。Colab 已預載大部分套件，cartopy/EMD-signal 視需要 `!pip install`。圖上文字是英文（Colab 預設字型無中文），中文說明在 markdown。

---

## 檔案格式約定

- `.py` 用 `# %%` / `# %% [markdown]`（jupytext「percent」格式）分格，方便一格對應 notebook 一個 cell。
- 中文教學敘事獨立放在 `narrative/*.md`，標題與程式分段一一對應。
- 課堂 notebook `timeseries2026.ipynb` 由 `uv run python build_notebook.py` 從 01/02/03 三個 `.py` 依序組裝（markdown 格 → markdown cell、程式格 → code cell，一格對一格不合併）；改了 `.py` 後重跑即可重新產生（notebook 上的手動微調會被覆蓋）。
- **圖上文字一律英文**（matplotlib 在 Colab 無 CJK 字型，中文會變方塊）；中文在 `.md` 與程式註解。

---

## 圖檔索引（`figs/`）

`figs/` 是**執行時自動產生的預覽圖**（已 gitignore，非必要產物）。敘事 `.md` 會引用這些檔名方便對照：

| 檔名 | 內容 |
|---|---|
| `01_01`–`01_06` | A：模擬訊號 / STL / STL vs 真值 / 原始SST vs 距平 / STL of SST / 距平 vs STL殘差 |
| `02_01`–`02_03` | B：太平洋距平地圖(cartopy) / ONI+事件長條 / 霍亂兩種定義箱型圖 |
| `03_01`–`03_05` | C：浮標概覽 / STL / 湧升指數 / UI×SST疊合 / UI–SST 互相關 |
| `A_01`–`A_06` | 附錄1 EEMD：合成例 / STL / IMF / EEMD vs STL趨勢 / SSTA / Hs |
| `U_01`–`U_02` | 附錄2：SST距平 vs STL殘差對照 / 差分對照 |

要重新產生預覽圖，可在 `MPLBACKEND=Agg` 下把 `plt.show` 接到 `savefig`（開發用）。

---

## 資料來源

- NOAA PSL Niño 3.4 月距平：<https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data>
- ODB 海洋熱浪 API：<https://eco.odb.ntu.edu.tw/api/mhw>
- 龍洞波浪浮標：CWA / 國海院（NAMR）提供，**僅供課程使用**。
- 湧升指數方法：Huang et al. 2021, *Remote Sensing* 13:170，<https://www.mdpi.com/2072-4292/13/2/170>
- 霍亂 vs ENSO 範例：digitized from Nature Sci. Rep. (2019) `s41598-018-38034-z`。

> 資料若需正式使用，請逕向國海院（NAMR）或中央氣象署（CWA）申請。
