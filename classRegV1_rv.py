#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:17:44 2023

@author: itachi
"""
from ohlc import AmberData
import asyncio
import aiohttp
import logging
import json
import datetime as dt
import pandas as pd
import requests
import time
from datetime import datetime
import numpy as np
from warnings import simplefilter
from create_rvFeatures import generate_rv_features
    #%% get data
data_start = datetime.now()

now = datetime.now()
start = dt.datetime(2018, 1, 1)
end = dt.datetime.combine(dt.date.today(), dt.time.min)


ticker = "BTCUSDT"
interval_in = 'hours'
exchange_in="binance", 
changes =[1,3,6,12,24]
ewm_days = [3,7,21]
vol_windows=[1,3,7,15,30,60]
future=True
day_step = 1
if interval_in == "hours":
    day_step = 24
elif interval_in == "minutes":
    day_step = 24*60
hour_step = day_step/24
df = generate_rv_features(start, end, ticker, interval_in,exchange_in="binance", changes =[1,3,6,12,24], ewm_days = [3,7,21],vol_windows=[1,3,7,15,30,60], future=True)
data_end = datetime.now()
print("Gather RV", data_end-data_start)
#%%get up down or chop based on sdev
df['closeChg%_forward24HR'] = df['closeChg%24HR'].shift(-int(day_step))
df['closeChg%_forward1HR'] = df['closeChg%1HR'].shift(-int(hour_step))
df_test = df[df['dt']> datetime(2022,12,1)]
df_train = df[df['dt'] <= datetime(2022,12,1)] 
std_24 = df_train['closeChg%_forward24HR'].std()
std_1 = df_train['closeChg%_forward1HR'].std()
def helperLabel(x, std_val):
  if abs(x) >= std_val:
      if x > 0:
          return 1
      else:
          return -1
  else:
      return 0
df['UpDownPred24HR'] = df['closeChg%_forward24HR'].apply(lambda x: helperLabel(x, std_24))
df['UpDownPred1HR'] = df['closeChg%_forward1HR'].apply(lambda x: helperLabel(x, std_1))
df_test = df[df['dt']> datetime(2022,12,1)]
df_train = df[df['dt'] <= datetime(2022,12,1)] 
df_test.reset_index(drop = True, inplace=True)  
df_train.reset_index(drop = True, inplace=True)  

#%% train classifcation
start_class = datetime.now()
from pycaret.classification import *


exp_clf24 = setup(df_train,target='UpDownPred24HR',
        ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR'],session_id=11,
        profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True) 
end_setup_class= datetime.now()
models_class = compare_models(turbo=True, n_select =4)
model_df_class = pull()
best_models_class = model_df_class.iloc[0:4]
end_class = datetime.now()
#%%
print("compare_models", end_class-start_class)
evaluate_model(models_class[0])
#%%

#et = create_model(best_models_class.index[0])
best_class= create_model(best_models_class.index[0])
# rf = create_model('rf')
# cat = create_model('catboost')
#%%
#blend1 = blend_models(estimator_list = [et, rf, cat])
#%% save classification model
save_model(best_class,"24hr_up_down")

#%% train regression no class
from pycaret.regression import *
end2 = datetime.now()

bad_cols = ['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR']


a=setup(df_train,target='closeChg%_forward24HR',
        ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR',  'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR', 'UpDownPred24HR']
        ,session_id=12,profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True);
models = compare_models(exclude = ['tr', 'lightgbm'],turbo=True, n_select =4)
#pull error df
model_df = pull()
#get top 4 models to train
best_models = model_df.iloc[0:4]
#%% get class labels
classifier = load_model("24hr_up_down");
class_pred = predict_model(classifier, df_train)
#get state labels
df_train_reg['state_class'] = class_pred['Label']

#%% get class labels
from pycaret.clustering import *
c = setup(df_train_reg, normalize = True)
#%%
a=setup(df_train_reg,target='closeChg%_forward24HR',
        ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR',  'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR', 'UpDownPred24HR' ]
        ,session_id=12,profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True, bin_numeric_features='state_class');
models = compare_models(exclude = ['tr', 'lightgbm'],turbo=True, n_select =4)
#pull error df
model_df = pull()
#get top 4 models to train
best_models = model_df.iloc[0:4]
#%%
