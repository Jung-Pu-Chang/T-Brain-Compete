import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_absolute_percentage_error
from LGB import LightGBM
from sklearn.preprocessing import RobustScaler

train_1 = pd.read_csv('../data/train_1.csv', encoding='big5',header=0)
train_2 = pd.read_csv('../data/train_2.csv', encoding='big5',header=0)
test_1 = pd.read_csv('../data/test_1.csv', encoding='utf-8',header=0)
test_2 = pd.read_csv('../data/test_2.csv', encoding='utf-8',header=0)

params = {
          'boosting_type': 'dart', 
          'n_estimators' : 4000,
          'learning_rate': 0.05,
          'reg_alpha' : 0.5, 
          'reg_lambda' : 0.5, 
          'n_jobs' : -1, 
          'random_state' : 7,
          'verbose' : 0,
          'objective' : 'regression_l1' #default = regression
          }

scoring = {
           'r2_score' : make_scorer(r2_score), 
           'mae' : make_scorer(mean_absolute_error),
           'mape' : make_scorer(mean_absolute_percentage_error), 
           }

'內部特徵爛，外部特徵 scaler'
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
                         'TxSoil10cm', 'temp_county', 'rh']
    else :
        trans_feature = ['type', 'apparent_zenith', 'zenith', 'apparent_elevation', 
                         'elevation', 'azimuth', 'equation_of_time', 'ghi',
                         'dni', 'dhi', 'airmass', 'cloud_cover', 'month', 'day', 'hour', 
                         'minutes_min', 'wspd', 'pres', 'temp', 'rhum', 
                         '測站氣壓(hPa)', '海平面氣壓(hPa)',
                         '氣溫(℃)', '露點溫度(℃)', '相對溼度(%)', '風速(m/s)', '風向(360degree)', '最大陣風(m/s)',
                         '最大陣風風向(360degree)', '降水量(mm)', '降水時數(hr)', '日照時數(hr)', '天全空日射量(MJ/㎡)',
                         'TxSoil0cm', 'TxSoil5cm', 'TxSoil10cm', 'TxSoil20cm', 'TxSoil50cm',
                         'TxSoil100cm', 'Visb', 'UVI', 'Cloud_Amount', 'TxSoil30cm', 'temp_county', 'rh']
        
    use = pd.concat([trainX[trans_feature], testX[trans_feature]],axis=0)
    scaler = RobustScaler() 
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
for i in range(1,15) : 
    trainX, testX = external_scaler(i, train_1, test_1)
    model, feature_name, cv = train_LGB(trainX) 
    cv['地點'] = i
    cv_final.append(cv) 
    out = LGB_prediction(feature_name, model, testX)
    upload.append(out) 
    print("區域代號 ", i)

upload = pd.concat(upload,axis=0)
cv_final = pd.concat(cv_final,axis=0)

upload_2 = []
cv_final_2 = []
for i in range(15,18) : 
    trainX, testX = external_scaler(i, train_2, test_2)
    model, feature_name, cv = train_LGB(trainX) 
    cv['地點'] = i
    cv_final_2.append(cv) 
    out = LGB_prediction(feature_name, model, testX)
    upload_2.append(out) 
    print("區域代號 ", i)

upload_2 = pd.concat(upload_2,axis=0)
cv_final_2 = pd.concat(cv_final_2,axis=0)

out = pd.concat([upload, upload_2],axis=0)
out.loc[out['答案'] < 0, '答案'] = 0 

out.to_csv('../data/upload.csv',encoding='utf-8',index=False)

