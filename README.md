# [2024_發電量預測競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/36)
> TEAM_5781，名次 32/500 (6.4%)

## Environment
`python3.8.13`

## Installation
`pip install -r requirements.txt`

## Directory

```bash
.
├── README.md
├── external_data
│   └── AQX (空氣品質資料)
│   └── CWB (氣象歷史資料)
├── src
│   └── LGB.py (模型套件)
│   └── main.py (主檔案)
│   └── preprocessing.py (資料前處理，執行完畢後，會輸出訓練與測試資料，提供給 main.py)
└── └── utils.py (preprocessing.py 使用函式)
```

### 簡介
[2024_發電量預測競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/36)

## Usage

### 調整參數

若要進行任何參數調整，請至`config.ini`中改寫參數。

### 情境說明

透過每位消費者的線上購買資料訓練模型，準確推薦消費者可能想購買的商品。

### API 說明
#### module_api.py
1. `module_api.py`會於初始化時，載入`config.ini`參數與`module.py`推薦系統
2. `module.py`會於初始化時，載入`edge_model`模型，並匯入`node_df.csv`、`edge_df.csv`
3. I :  
   item : str，購買商品，空值請回傳空字串，舉例 : '22726'  
4. O :  
   dic : dict，推薦內容包含3個商品推薦(StockCode)，皆不可為空值  
   舉例 : {'StockCode': ["22494","21417","16254"]}  

```bash
cd ~/Recommandation_System/service
python module_api.py
```


