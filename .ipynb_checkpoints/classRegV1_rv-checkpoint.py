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
#%%
df['closeChg%_forward24HR'] = df['closeChg%24HR'].shift(-int(day_step))
df['closeChg%_forward1HR'] = df['closeChg%1HR'].shift(-int(hour_step))
#get up down or chop based on sdev
std_24 = df['closeChg%_forward24HR'].std()
std_1 = df['closeChg%_forward1HR'].std()
def helperLabel(x, std_val):
  if abs(x) >= std_val:
      if x > 0:
          return 1
      else:
          return -1
  else:
      return 0
#%%train test
df_test = df[df['dt']> datetime(2022,12,1)]
df_train = df[df['dt'] <= datetime(2022,12,1)] 
    
#%% classify
start_class = datetime.now()
from pycaret.classification import *


exp_clf24 = setup(df,target='UpDownPred24HR',
        ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR'],session_id=11,
        profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True) 
end_setup_class= datetime.now()
models_class = compare_models(turbo=True, n_select =4)
model_df_class = pull()
best_models_class = model_df_class.iloc[0:4]
end_class = datetime.now()
#%%
evaluate_model(models_class[0])
#%%
et = create_model('et')
rf = create_model('rf')
cat = create_model('catboost')
#%%
blend1 = blend_models(estimator_list = [et, rf, cat])
#%%
class_predictions = predict_model(et, data= df_train)

train_ts = df_train.copy(deep=True)




