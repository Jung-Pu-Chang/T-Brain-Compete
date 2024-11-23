
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
#from fancyimpute import IterativeImputer # mice
import sys
sys.path.append(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電')
from LGB import LightGBM
from scipy import stats # boxcox & yeojohnson

train = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\train.csv', encoding='big5',header=0)
test = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\test.csv', encoding='big5',header=0)
test = test.rename(columns={'minutes': 'minutes_min'})

train = train[['地點', '發電量(mW)', 'apparent_zenith', 'zenith', 'apparent_elevation',
               'elevation', 'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi',
               'airmass', 'cloud_cover', 'wspd', 'pres', 'temp', 'rhum', 'month',
               'day', 'hour', 'minutes_min', '氣壓(hpa)', '溫度(°C)', '濕度(%)', '亮度(Lux)']]



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

def train_LGB(place) : 
    use = train.loc[train['地點']==place]
    trainX = use.drop(['地點', '發電量(mW)'], axis=1)
    trainY = use[['發電量(mW)']]
    train_X_fs, feature_name = LightGBM.permutation_selection(trainX, trainY, 
                                                              params = params,
                                                              imp = 0, # 無特徵篩選
                                                              isClassifier=False)
    model, cv, cv_idx = LightGBM.build_model(train_X_fs, trainY, 
                                             params = params, scoring = scoring, 
                                             fold_time = 10, isClassifier=False)
    return model, feature_name, cv

def LGB_prediction(place, feature_name, model) : 
    use = test.loc[test['地點']==place].reset_index(drop=True)
    testX = use[feature_name]
    pred = pd.DataFrame(model.predict(testX)).rename(columns={0: '答案'})
    pred = pd.concat([use[['序號']], pred],axis=1)
    return pred
    
'1~16 直接 LGB'
upload = []
cv_final = []
for i in range(1,17) : # 先跑 1~16
    model, feature_name, cv = train_LGB(i) 
    cv['地點'] = i
    cv_final.append(cv) #合併
    out = LGB_prediction(i, feature_name, model)
    upload.append(out) #合併
    print("區域代號 ", i)

upload = pd.concat(upload,axis=0)
cv_final = pd.concat(cv_final,axis=0)

backup = upload

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

'inverse(Y)'

forecast = np.power((fitted_lambda * forecast[['yhat']]) + 1, 1 / fitted_lambda) - 1

'big5無法讀取'
upload.to_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\upload.csv',encoding='utf-8',index=False)
