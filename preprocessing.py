import sys
sys.path.append(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電')
import pandas as pd
import os
import glob
from datetime import datetime
from module import Feature_Engineering as fe
import warnings
warnings.filterwarnings('ignore')

'內部特徵工程'
files = glob.glob(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\raw\*.csv')
df = pd.concat([pd.read_csv(fp).assign(file_name=os.path.basename(fp)) for fp in files])
df.columns = ['地點', 'DateTime', '風速(m/s)_raw', '氣壓(hpa)_raw', '溫度(°C)_raw',
              '濕度(%)_raw', '亮度(Lux)_raw', '發電量(mW)','檔名']
df['key'] = df['DateTime'].astype(str).str[:16 ]+ '_' + df['地點'].astype(str)
df = df.drop_duplicates(subset=['key'],keep='first')
df['month'] = df['DateTime'].astype(str).str[5:7].astype(int)
df['day'] = df['DateTime'].astype(str).str[8:11].astype(int)
df['hour'] = df['DateTime'].astype(str).str[11:13].astype(int)
df['minutes'] = df['DateTime'].astype(str).str[14:16].astype(int)

test = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\upload(no answer).csv', encoding='utf-8',header=0)
test['地點'] = test['序號'].astype(str).str[12:14].astype(int)
test['month'] = test['序號'].astype(str).str[4:6].astype(int)
test['day'] = test['序號'].astype(str).str[6:8].astype(int)
test['hour'] = test['序號'].astype(str).str[8:10].astype(int)
test['minutes_min'] = test['序號'].astype(str).str[10:12].astype(int)
test['key'] = (test['序號'].astype(str).str[:4] + '-' +
               test['序號'].astype(str).str[4:6] + '-' + test['序號'].astype(str).str[6:8] + ' ' + 
               test['序號'].astype(str).str[8:10] + ':' + test['序號'].astype(str).str[10:12] + '_' +
               test['地點'].astype(str))
test['DateTime'] = (test['序號'].astype(str).str[:4] + '-' +
                    test['序號'].astype(str).str[4:6] + '-' + test['序號'].astype(str).str[6:8] + ' ' + 
                    test['序號'].astype(str).str[8:10] + ':' + test['序號'].astype(str).str[10:12])
'時間 EDA'
time = pd.concat([df[['地點', 'DateTime']],test[['地點', 'DateTime']]],axis=0)
time = time.sort_values(by = 'DateTime',ascending=True) 
start = datetime.strptime(time['DateTime'].head(1).iloc[0][:10], '%Y-%m-%d')
end = datetime.strptime(time['DateTime'].tail(1).iloc[0][:10], '%Y-%m-%d')
time = time.groupby('地點').agg(min_time=('DateTime', 'min'), max_time=('DateTime', 'max')).reset_index()

'位置'
location = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\locations_data.csv', encoding='big5',header=0)
location['地點'] = location['比賽代號']
df = pd.merge(df, location[['緯度', '經度', '地點']], on="地點",how='left')

'特徵建構 : 外部特徵'
pvlib_df = []
for i in range(1,18) : 
    out = location[location["地點"] == i] 
    lat = out['緯度'].head(1).iloc[0]
    lon = out['經度'].head(1).iloc[0]
    start_2 = time.loc[time['地點']== i]['min_time'].head(1).iloc[0][:16]
    end_2 = time.loc[time['地點']== i]['max_time'].head(1).iloc[0][:16]
    out = fe.feature_from_pvlib(lat, lon, start_2, end_2, freq='T')
    out['地點'] = i
    pvlib_df.append(out) #合併
    print("地點 ", i)
pvlib_df = pd.concat(pvlib_df,axis=0)
pvlib_df['airmass'] = pvlib_df['airmass'].fillna(0)

meteo_df = fe.feature_from_meteostat(25.09108, 121.5598, start, end)

'1~14 cwb data : 只到小時'
climate_1 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\hourly_C0Z100_2024-01-01_2024-10-31.csv', encoding='utf-8',header=0)
climate_1['key'] = climate_1['﻿觀測時間(hour)'].astype(str).str[:13]
climate_1.isna().sum()
climate_1 = climate_1.drop(['日照時數(hr)','TxSoil5cm'], axis=1)

'15~17 cwb data : 只到小時'
climate_2 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\hourly_466990_2024-01-01_2024-10-31.csv', encoding='utf-8',header=0)
climate_2['key'] = climate_2['﻿觀測時間(hour)'].astype(str).str[:13]
climate_2.isna().sum()
climate_2 = climate_2.drop(['TxSoil200cm','TxSoil300cm','TxSoil500cm','VaporPressure'], axis=1)
climate_2 = climate_2.rename(columns={'Cloud Amount': 'Cloud_Amount'})

'合併'
pvlib_df['key'] = pvlib_df['DateTime'].astype(str).str[:16] + '_' + pvlib_df['地點'].astype(str)
df = pd.merge(df, pvlib_df, on="key",how='left')

df['key'] = df['key'].str[:13]
test['key'] = test['key'].str[:13]
df = pd.merge(df, meteo_df, on="key",how='left')
test = pd.merge(test, meteo_df, on="key",how='left')

df_1 = pd.merge(df.loc[df['地點_x']<=14], climate_1, on="key",how='left')
df_2 = pd.merge(df.loc[df['地點_x']>=15], climate_2, on="key",how='left')
test_1 = pd.merge(test.loc[test['地點']<=14], climate_1, on="key",how='left')
test_2 = pd.merge(test.loc[test['地點']>=15], climate_2, on="key",how='left')

'特徵轉換 : 10分鐘平均'
feature_name = ['風速(m/s)_raw', '氣壓(hpa)_raw', '溫度(°C)_raw',
                '濕度(%)_raw', '亮度(Lux)_raw',
                'apparent_zenith', 'zenith', 'apparent_elevation',
                'elevation', 'azimuth', 'equation_of_time', 'ghi', 
                'dni', 'dhi', 'airmass', 'cloud_cover', '發電量(mW)',
                '地點_x', 'month', 'day', 'hour']
train_1 = []
for i in range(1,15) : # 1~14
    out = fe.resample_10T(i, df_1, feature_name)
    train_1.append(out) 
    print("地點 ", i)
train_1 = pd.concat(train_1,axis=0)
df_1['key'] = df_1['DateTime_x'].astype(str).str[:16] + '_' + df_1['地點_x'].astype(int).astype(str)
feature_train = ['key', 'wspd', 'pres', 'temp', 'rhum',
                 '測站氣壓(hPa)', '氣溫(℃)', '相對溼度(%)', '風速(m/s)', '風向(360degree)',
                 '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)', '天全空日射量(MJ/㎡)',
                 'TxSoil0cm', 'TxSoil20cm', 'TxSoil50cm', 'TxSoil100cm', 'TxSoil10cm']
train_1 = pd.merge(train_1, df_1[feature_train], on="key",how='left')
train_1 = train_1.dropna()

train_2 = []

for i in range(15,18) : # 15~17
    out = fe.resample_10T(i, df_2, feature_name)
    train_2.append(out) 
    print("地點 ", i)
train_2 = pd.concat(train_2,axis=0)
df_2['key'] = df_2['DateTime_x'].astype(str).str[:16] + '_' + df_2['地點_x'].astype(int).astype(str)
feature_name = ['key', 'wspd', 'pres', 'temp', 'rhum',
                '測站氣壓(hPa)', '海平面氣壓(hPa)', '氣溫(℃)', '露點溫度(℃)', '相對溼度(%)', '風速(m/s)',
                '風向(360degree)', '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)',
                '降水時數(hr)', '日照時數(hr)', '天全空日射量(MJ/㎡)', 'TxSoil0cm', 'TxSoil5cm',
                'TxSoil10cm', 'TxSoil20cm', 'TxSoil50cm', 'TxSoil100cm', 'Visb', 'UVI',
                'Cloud_Amount', 'TxSoil30cm']
train_2 = pd.merge(train_2, df_2[feature_name], on="key",how='left')
train_2 = train_2.dropna()

train_1.to_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\train_1.csv', encoding='big5',index=False) 
train_2.to_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\train_2.csv', encoding='big5',index=False) 

'test 串 pvlib 平均 30 = 30~39'
pvlib_df_10 = []
for i in range(1,18) : # 15~17
    out = pvlib_df.loc[pvlib_df['地點']==i]
    out['DateTime'] = pd.to_datetime(out['DateTime'])
    out.set_index('DateTime', inplace=True)
    key = out[['key']].resample('10T').min().reset_index()
    out = out[['apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
               'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi', 'airmass',
               'cloud_cover']].resample('10T').mean().reset_index()
    out = pd.concat([key[['key']],out],axis=1) 
    pvlib_df_10.append(out) 
    print("地點 ", i)
pvlib_df_10 = pd.concat(pvlib_df_10,axis=0)

test_1['key'] = test_1['DateTime'].astype(str) + '_' + test_1['地點'].astype(str)
test_1 = pd.merge(test_1, pvlib_df_10, on="key",how='left')
test_2['key'] = test_2['DateTime'].astype(str) + '_' + test_2['地點'].astype(str)
test_2 = pd.merge(test_2, pvlib_df_10, on="key",how='left')

test_1.isna().sum()
test_2.isna().sum()

test_1.to_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\test_1.csv', encoding='utf-8',index=False) 
test_2.to_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\test_2.csv', encoding='utf-8',index=False) 

#%%
'chk'
print('空值檢查 : ', train.isna().sum())
len(clean.loc[clean['地點']==17])
chk = clean[clean['結清年'].isnull()]