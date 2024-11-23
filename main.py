import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
#from fancyimpute import IterativeImputer # mice
import sys
sys.path.append(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電')
from LGB import LightGBM

'EDA'
train = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\train.csv', encoding='big5',header=0)
test = pd.read_csv(r'D:\Users\Pu_chang\Desktop\競賽\Tbrain_2024_發電\data\test.csv', encoding='big5',header=0)
test = test.rename(columns={'minutes': 'minutes_min'})
test['DateTime'] = pd.to_datetime(test['key'].str.replace(r'^\d+-', '', regex=True))
train_EDA = train.groupby('地點').agg(min_time=('DateTime', 'min'), max_time=('DateTime', 'max'), max_y=('發電量(mW)', 'max'), min_y=('發電量(mW)', 'min'), mean_y=('發電量(mW)', 'mean')).reset_index()
test_EDA = test.groupby('地點').agg(min_time=('DateTime', 'min'), max_time=('DateTime', 'max')).reset_index()
#%%

train = train[['地點', '發電量(mW)', 'apparent_zenith', 'zenith', 'apparent_elevation',
               'elevation', 'azimuth', 'equation_of_time', 'ghi', 'dni', 'dhi',
               'airmass', 'cloud_cover', 'wspd', 'pres', 'temp', 'rhum', 'month',
               'day', 'hour', 'minutes_min']]


params = {
          'boosting_type': 'dart', #生成方式 gbdt, dart, rf 
          'n_estimators' : 1000,
          'learning_rate': 0.05,
          'n_jobs' : -1, #執行所有CPU
          'random_state' : 7,
          'verbose' : 0
          }

scoring = {
           'r2_score' : make_scorer(r2_score), 
           'mae' : make_scorer(mean_absolute_error),
           'mape' : make_scorer(mean_absolute_percentage_error), 
           }
