import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
from fancyimpute import IterativeImputer # mice
import sys
sys.path.append(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電')
from LGB import LightGBM
from scipy import stats # boxcox & yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

train_1 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\train_1.csv', encoding='big5',header=0)
train_2 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\train_2.csv', encoding='big5',header=0)
test_1 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\test_1.csv', encoding='utf-8',header=0)
test_2 = pd.read_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\test_2.csv', encoding='utf-8',header=0)

params = {
          'boosting_type': 'dart', #生成方式 gbdt, dart, rf 
          'n_estimators' : 3000,
          'learning_rate': 0.05,
          'reg_alpha' : 0.5, # L1 正規化 : 大 = 篩選特徵，減少極端值
          'reg_lambda' : 0.5, # L2 正規化 : 大 = 避免過適，減少極端值
          'n_jobs' : -1, #執行所有CPU
          'random_state' : 7,
          'verbose' : 0
          }

scoring = {
           'r2_score' : make_scorer(r2_score), 
           'mae' : make_scorer(mean_absolute_error),
           'mape' : make_scorer(mean_absolute_percentage_error), 
           }

'內部特徵爛，外部特徵 scaler'
def feature_KNN_fill(place, train_data, test_data) : 
    trainX = train_data.loc[train_data['地點']==place].reset_index(drop=True)
    use = test_data.loc[test_data['地點']==place].reset_index(drop=True)
    use[['氣壓(hpa)_raw', '溫度(°C)_raw', '濕度(%)_raw', '亮度(Lux)_raw']] = np.nan
    trainX = trainX.drop(['DateTime', '發電量(mW)','key',], axis=1)
    testX = use.drop(['序號','答案','DateTime_x', 'DateTime_y',
                      'time', '﻿觀測時間(hour)', 'key'], axis=1)
    fill_data = pd.concat([trainX, testX],axis=0).reset_index(drop=True)
    #imputer = KNNImputer(n_neighbors=2) # 5，neighbor
    imputer = IterativeImputer() # mice
    imputed_data = imputer.fit_transform(fill_data) 
    imputed_df = pd.DataFrame(imputed_data, columns=fill_data.columns)
    
    missing_mask = fill_data[['氣壓(hpa)_raw', '溫度(°C)_raw', '濕度(%)_raw', '亮度(Lux)_raw']].isna()
    imputed_mask = pd.DataFrame(imputed_data, columns=fill_data.columns)[['氣壓(hpa)_raw', '溫度(°C)_raw', '濕度(%)_raw', '亮度(Lux)_raw']].isna()
    補值標記 = missing_mask != imputed_mask
    imputed_df['補值標記'] = 補值標記.any(axis=1)
    testX = imputed_df.loc[imputed_df['補值標記']==True].reset_index(drop=True)
    testX = pd.concat([use[['序號','答案']], testX],axis=1)
    testX = testX.drop(['補值標記'], axis=1)
    return testX

def external_scaler(place, train_data, test_data):
    trainX = train_data.loc[train_data['地點']==place].reset_index(drop=True)
    trainX['type'] = 'train'
    trainY = trainX[['發電量(mW)']]
    testX = test_data.loc[test_data['地點']==place].reset_index(drop=True)
    testX['type'] = 'test'
    testY = testX[['序號']]
    if place <= 14: 
        trans_feature = ['type', 'apparent_zenith', 'zenith', 'apparent_elevation', 
                         'elevation', 'azimuth', 'equation_of_time', 'ghi',
                         'dni', 'dhi', 'airmass', 'cloud_cover', 'month', 'day', 'hour', 
                         'minutes_min', 'wspd', 'pres', 'temp', 'rhum',
                         '測站氣壓(hPa)', '氣溫(℃)', '相對溼度(%)',
                         '風速(m/s)', '風向(360degree)', '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)',
                         '天全空日射量(MJ/㎡)', 'TxSoil0cm', 'TxSoil20cm', 'TxSoil50cm', 'TxSoil100cm',
                         'TxSoil10cm']
    else :
        trans_feature = ['type', 'apparent_zenith', 'zenith', 'apparent_elevation', 
                         'elevation', 'azimuth', 'equation_of_time', 'ghi',
                         'dni', 'dhi', 'airmass', 'cloud_cover', 'month', 'day', 'hour', 
                         'minutes_min', 'wspd', 'pres', 'temp', 'rhum', 
                         '測站氣壓(hPa)', '海平面氣壓(hPa)',
                         '氣溫(℃)', '露點溫度(℃)', '相對溼度(%)', '風速(m/s)', '風向(360degree)', '最大陣風(m/s)',
                         '最大陣風風向(360degree)', '降水量(mm)', '降水時數(hr)', '日照時數(hr)', '天全空日射量(MJ/㎡)',
                         'TxSoil0cm', 'TxSoil5cm', 'TxSoil10cm', 'TxSoil20cm', 'TxSoil50cm',
                         'TxSoil100cm', 'Visb', 'UVI', 'Cloud_Amount', 'TxSoil30cm']
        
    use = pd.concat([trainX[trans_feature], testX[trans_feature]],axis=0)
    scaler = RobustScaler() 
    #scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(use.drop(['type'], axis=1)))

    use = pd.concat([use[['type']].reset_index(drop=True), scaled_data],axis=1)
    use.columns = trans_feature
    trainX = use.loc[use['type']=='train'].drop(['type'], axis=1)
    trainX = pd.concat([trainX.reset_index(drop=True),trainY],axis=1)
    testX = use.loc[use['type']=='test'].drop(['type'], axis=1)
    testX = pd.concat([testX.reset_index(drop=True),testY],axis=1)
    return trainX, testX

def train_LGB(trainX) : 
    trainY = trainX[['發電量(mW)']]
    trainX = trainX.drop(['發電量(mW)'], axis=1)
    train_X_fs, feature_name = LightGBM.permutation_selection(trainX, trainY, 
                                                              params = params,
                                                              imp = 0, # 無特徵篩選
                                                              isClassifier=False)
    model, cv, cv_idx = LightGBM.build_model(train_X_fs, trainY, 
                                             params = params, scoring = scoring, 
                                             fold_time = 2, isClassifier=False)
    return model, feature_name, cv

def LGB_prediction(feature_name, model, testX) : 
    pred = testX[feature_name]
    pred = pd.DataFrame(model.predict(pred)).rename(columns={0: '答案'})
    pred = pd.concat([testX[['序號']], pred],axis=1)
    return pred

'LGB * 17'
upload = []
cv_final = []
for i in range(1,15) : # 先跑 1~1
    #testX = feature_KNN_fill(i, train_1, test_1)
    #trainX, testX = external_scaler(i, train_1, testX)
    trainX, testX = external_scaler(i, train_1, test_1)
    model, feature_name, cv = train_LGB(trainX) 
    cv['地點'] = i
    cv_final.append(cv) #合併
    out = LGB_prediction(feature_name, model, testX)
    upload.append(out) #合併
    print("區域代號 ", i)

upload = pd.concat(upload,axis=0)
cv_final = pd.concat(cv_final,axis=0)

upload_2 = []
cv_final_2 = []
for i in range(15,18) : # 先跑 1~1
    #testX = feature_KNN_fill(i, train_2, test_2)
    #trainX, testX = external_scaler(i, train_2, testX)
    trainX, testX = external_scaler(i, train_2, test_2)
    model, feature_name, cv = train_LGB(trainX) 
    cv['地點'] = i
    cv_final_2.append(cv) #合併
    out = LGB_prediction(feature_name, model, testX)
    upload_2.append(out) #合併
    print("區域代號 ", i)

upload_2 = pd.concat(upload_2,axis=0)
cv_final_2 = pd.concat(cv_final_2,axis=0)

out = pd.concat([upload, upload_2],axis=0)
out.loc[out['答案'] < 0, '答案'] = 0 
'big5無法讀取'
out.to_csv(r'C:\Users\user\Desktop\比賽\Tbrain_2024_發電\data\upload.csv',encoding='utf-8',index=False)

#%%
'內部特徵轉換 : scaler + pca + knn補'
def external_scaler(place):
    trainX = train.loc[train['地點']==place].reset_index(drop=True)
    trainX['type'] = 'train'
    trainY = trainX[['發電量(mW)']]
    testX = test.loc[test['地點']==place].reset_index(drop=True)
    testX['type'] = 'test'
    testY = testX[['序號']]
    trans_feature = ['type', 'apparent_zenith', 'zenith', 'apparent_elevation', 
                     'elevation', 'azimuth', 'equation_of_time', 'ghi',
                     'dni', 'dhi', 'airmass', 'cloud_cover', 'wspd', 'pres', 
                     'temp', 'rhum', 'month', 'day', 'hour', 'minutes_min']
    use = pd.concat([trainX[trans_feature], testX[trans_feature]],axis=0)
    scaler = RobustScaler() 
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(use.drop(['type'], axis=1)))

    use = pd.concat([use[['type']].reset_index(drop=True), scaled_data],axis=1)
    use.columns = trans_feature
    trainX = use.loc[use['type']=='train'].drop(['type'], axis=1)
    trainX = pd.concat([trainX.reset_index(drop=True),trainY],axis=1)
    testX = use.loc[use['type']=='test'].drop(['type'], axis=1)
    testX = pd.concat([testX.reset_index(drop=True),testY],axis=1)
    return trainX, testX


def y_trans(trainX):
    fitted_data, fitted_lambda = stats.yeojohnson(trainX['發電量(mW)'].to_numpy()) 
    trainX['發電量(mW)'] = pd.DataFrame(fitted_data)
    return trainX, fitted_lambda

def feature_transform(place) : # mle : 4壓3最佳
    use = train.loc[train['地點']==place].reset_index(drop=True)
    scaler = StandardScaler() # 轉標準常態(0,1)
    scaled_data = scaler.fit_transform(use[['氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)']])  
    pca = PCA(n_components=3, svd_solver='full', random_state=10) 
    pca = pd.DataFrame(pca.fit_transform(scaled_data))
    trainX = pd.concat([use, pca],axis=1)
    trainX = trainX.rename(columns={0: 'PCA_1', 1: 'PCA_2', 2: 'PCA_3'})
    return trainX

def feature_KNN_fill(place, final_use) : 
    use = test.loc[test['地點']==place].reset_index(drop=True)
    use[['PCA_1', 'PCA_2', 'PCA_3']] = np.nan
    final_use = final_use.drop(['DateTime', '發電量(mW)','氣壓(hpa)',
                                '溫度(°C)', '濕度(%)', '亮度(Lux)',
                                'minutes', 'minutes_max'], axis=1)
    use_2 = use.drop(['key', '序號','答案'], axis=1)
    fill_data = pd.concat([final_use, use_2],axis=0).reset_index(drop=True)
    imputer = KNNImputer(n_neighbors=2)
    imputed_data = imputer.fit_transform(fill_data) 
    imputed_df = pd.DataFrame(imputed_data, columns=fill_data.columns)
    
    missing_mask = fill_data[['PCA_1', 'PCA_2', 'PCA_3']].isna()
    imputed_mask = pd.DataFrame(imputed_data, columns=use_2.columns)[['PCA_1', 'PCA_2', 'PCA_3']].isna()
    補值標記 = missing_mask != imputed_mask
    imputed_df['補值標記'] = 補值標記.any(axis=1)
    testX = imputed_df.loc[imputed_df['補值標記']==True].reset_index(drop=True)
    testX = pd.concat([use[['序號','答案']], testX],axis=1)
    testX = testX.drop(['補值標記'], axis=1)
    return testX

'17用補的'
train_17 = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\raw\L17_Train.csv', encoding='utf-8',header=0)
train_17['month'] = train_17['DateTime'].astype(str).str[5:7].astype(int)
train_17['day'] = train_17['DateTime'].astype(str).str[8:10].astype(int)
train_17['hour'] = train_17['DateTime'].astype(str).str[11:13].astype(int)
train_17['minutes_min'] = train_17['DateTime'].astype(str).str[14:16].astype(int)
train_17['DateTime'] = pd.to_datetime(train_17['DateTime'])
train_17.set_index('DateTime', inplace=True)
train_17 = train_17.resample('10T').mean().reset_index()
train_17 = train_17[['Power(mW)', 'month', 'day', 'hour', 'minutes_min']]
train_17 = train_17.dropna()

test_17 = test.loc[test['地點']==17]
test_17['Power(mW)'] = np.nan
test_17 = test_17[['Power(mW)', 'month', 'day', 'hour', 'minutes_min']]
pred_17 = pd.concat([train_17, test_17],axis=0).reset_index(drop=True)

'knn & mice'
imputer = KNNImputer(n_neighbors=2)
#imputer = IterativeImputer() # mice

imputed_data = imputer.fit_transform(pred_17) 
imputed_df = pd.DataFrame(imputed_data, columns=pred_17.columns)
imputed_df = imputed_df.iloc[9822:10974,:].reset_index(drop=True)

upload_17 = test.loc[test['地點']==17].reset_index(drop=True)
upload_17 = pd.concat([imputed_df, upload_17],axis=1).reset_index(drop=True)
upload_17['答案'] = upload_17['Power(mW)']
upload_17 = upload_17[['序號','答案']]
upload = pd.concat([upload, upload_17], axis=0)

upload.loc[upload['答案'] < 0, '答案'] = 0 # chk < 0