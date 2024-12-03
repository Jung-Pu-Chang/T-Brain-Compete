# 2024_TBrain_根據區域微氣候資料預測發電量競賽
> [2024_TBrain_發電量預測競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/36)  
> TEAM_5781，排名如下   
  
|         |     AE    |  排名  |
| :------ | :-------: | :----: |
| Public  | 631438.44 | 30/500 |
| Private | 729789.1  | 32/500 |

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
│   └── main.py (主檔案，透過 LGB.py 訓練 preprocessing.py 輸出的訓練資料，預測測試資料)
│   └── preprocessing.py (資料前處理，執行後，會輸出訓練與測試資料)
└── └── utils.py (preprocessing.py 使用函式)
```

## 作法簡介
> 預測目標 : Power(mW)  
> 演算法 : LightGBM + Dart + regression_l1    
> 競賽特徵使用 : 僅使用時間   
> 外部特徵使用 : AQX、CWB、pvlib、meteostat  


## 參考資料
> [A Review of Solar Forecasting Techniques and the Role of Artificial Intelligence](https://www.mdpi.com/2673-9941/4/1/5)   
> [AIdea: Solar PV Forecast - Surplux](https://github.com/siang-chang/aidea-solar-energy-surplux?tab=readme-ov-file)  
> [pvlib](https://github.com/pvlib/pvlib-python)  
> [臺灣歷史氣象觀測資料庫](https://github.com/Raingel/historical_weather)  





