# 2025_NODASS_workshop
Lectures and codebase for 2025 NODASS Workshop (2025-07-16 ~ 2025-07-18)

### 📖 Overview

   本課程與 Python 程式碼由 ODB 技術員寫作與維持，著重於海洋資料整理，包含地理資訊格式、海洋生物資料標準化、時間序列等，與海洋科學資料分析初探，以海洋熱浪與氣象浮標為基本範例。

### 📁 Directory structure

```markdown

/2025_NODASS_workshop  
├── lectures/               # lecture slides (PDFs)
├── src/                    # source code and data
│   ├── bio/                # data for biological lecture
│   ├── geo/                # data for geographic lecture
│   ├── mhw/                # data for marine heatwaves lecture
│   ├── timeseries/         # data for time series lecture
│   └── ODB.ipynb           # code for this workshop
└── requirements.txt        # only needed when running the code locally

```

### 📝 Note

 [工作坊官網](https://sites.google.com/view/nodassbigdata/index)

 [Colab 連結](https://colab.research.google.com/github/ODBapp/2025_NODASS_workshop/blob/main/src/odb.ipynb)

#### `requirements.txt`

-If you are running the code on your local machine, install dependencies with:

```bash
pip install -r requirements.txt
```
-If you are using Google Colab, you do not need to install anything manually — all required packages will be installed automatically within the notebook.


### 📊 Data source

- Ocean Data Bank (ODB): https://www.odb.ntu.edu.tw/  
- NODASS, NAMR: https://nodass.namr.gov.tw/  
- CWA API: https://opendata.cwa.gov.tw/dist/opendata-swagger.html  

> **Note**
> - Buoy data provided from NAMR (國海院) are used only for this training workshop. Please acquire the data from NAMR or CWA (中央氣象署) directly.
> - The dataset 台灣深海魚類多樣性的調查研究  used in the biology lecture, is provided by NAMR (國海院) and can be accessed through NODASS.

