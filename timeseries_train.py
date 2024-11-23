import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
import sys
sys.path.append(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電')
from multi_prophet import MultiProphet
from prophet import Prophet
from scipy import stats # boxcox & yeojohnson 

'test time 都於區間中 : sparse time series'
train = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\train.csv', encoding='big5',header=0)
test = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\test.csv', encoding='big5',header=0)
test = test.rename(columns={'minutes': 'minutes_min'})
test['DateTime'] = pd.to_datetime(test['key'].str.replace(r'^\d+-', '', regex=True))
time_train = train.groupby('地點').agg(min_time=('DateTime', 'min'), max_time=('DateTime', 'max')).reset_index()
time_test = test.groupby('地點').agg(min_time=('DateTime', 'min'), max_time=('DateTime', 'max')).reset_index()

def df_to_model(place) : 
    trainX = train.loc[train['地點']==place].reset_index(drop=True)
    trainX['ds'] = pd.to_datetime(trainX['DateTime']) 
    #trainX['y'] = trainX['發電量(mW)']
    fitted_data, fitted_lambda = stats.yeojohnson(trainX['發電量(mW)'].to_numpy()) 
    trainX['y'] = pd.DataFrame(fitted_data)
    
    testX = test.loc[test['地點']==place].reset_index(drop=True)
    testX['ds'] = pd.to_datetime(testX['DateTime'])
    return trainX, testX, fitted_lambda

def variate_prophet(df_train, df_test) : 
    model = Prophet()
    model.fit(df_train[['ds', 'y']])
    forecast = model.predict(df_test[['ds']])
    forecast.loc[forecast['yhat'] < 0, 'yhat'] = 0
    return forecast[['yhat']]

def internal_variate_prophet(df_train, df_test, feature_name) : 
    ''' prophet 序列預測 內部特徵 
        '風速(m/s)', '氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)' '''
    
    df_train['y'] = df_train[f'{feature_name}']
    model = Prophet()
    model.fit(df_train[['ds', 'y']])
    forecast = model.predict(df_test[['ds']])
    forecast.loc[forecast['yhat'] < 0, 'yhat'] = 0
    forecast = forecast.rename(columns={'yhat': f'{feature_name}'})
    return forecast[[f'{feature_name}']].reset_index(drop=True)

def external_variate_prophet(df_train, df_test) : 
    model = Prophet() #weekly_seasonality
    regressors = [
    'apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
    'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi',
    #'風速(m/s)', '氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)',
    'airmass', 'cloud_cover', 'wspd', 'pres', 'temp', 'rhum']
    
    for regressor in regressors:
        model.add_regressor(regressor)
    
    model.fit(df_train[['ds', 'y', 'apparent_zenith', 'zenith', 
                        'apparent_elevation', 'elevation',
                        'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi',
                        #'風速(m/s)', '氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)',
                        'airmass', 'cloud_cover', 'wspd', 'pres', 'temp', 'rhum']])
    
    forecast = model.predict(df_test[['ds','apparent_zenith', 'zenith', 
                                      'apparent_elevation', 'elevation',
                                      'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi',
                                      #'風速(m/s)', '氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)',
                                      'airmass', 'cloud_cover', 'wspd', 'pres', 'temp', 'rhum']])
    
    forecast.loc[forecast['yhat'] < 0, 'yhat'] = 0
    return forecast[['yhat']]

'加入 內部特徵會爆'
upload = []
for i in range(1,17) : # 先跑 1~17
    trainX, testX, fitted_lambda = df_to_model(i)
    forecast = external_variate_prophet(trainX, testX)
    forecast = np.power((fitted_lambda * forecast[['yhat']]) + 1, 1 / fitted_lambda) - 1
    out = pd.concat([testX[['序號']], forecast],axis=1)
    upload.append(out) #合併
    print("區域代號 ", i)

upload = pd.concat(upload,axis=0)
upload.columns = ['序號', '答案']
#%%
'17 單變量'
trainX, testX, fitted_lambda = df_to_model(17)
forecast = variate_prophet(trainX, testX)
forecast = np.power((fitted_lambda * forecast[['yhat']]) + 1, 1 / fitted_lambda) - 1
out = pd.concat([testX[['序號']], forecast],axis=1)
out.columns = ['序號', '答案']

'big5無法讀取'
upload = pd.concat([upload,out],axis=0)
upload.to_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\upload.csv',encoding='utf-8',index=False)

#%%


'特徵篩選 : Sequential Backward Selection、Mutual Information、芙蓉'
'lightGBM 分組跑 > Prophet 分組跑'
