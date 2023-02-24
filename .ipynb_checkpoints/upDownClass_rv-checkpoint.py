#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:46:04 2023

@author: moussa
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
def create_changePCT(df, shift, column, time):
    df[column+"Chg"+time] = (futures_candle_df[column]-shift[column])
    df[column+"Chg%"+time] = df[column+"Chg"+time]/shift[column]
    return df
    
#%%
data_start = datetime.now()
amberdata = AmberData()
now = datetime.now()
start = dt.datetime(2018, 1, 1)
end = dt.datetime.combine(dt.date.today(), dt.time.min)

# spot_candles = amberdata.get_spot_candles(pair='btc_usdt', exchange='binance', interval='hours', start=start, end=end)
# spot_candle_df = pd.DataFrame(spot_candles)
# print(spot_candle_df)
ticker = "BTCUSDT"
exchange_in = 'binance'
interval_in = 'hours'
shift_val = 24
futures_candles = amberdata.get_futures_candles(instrument=ticker, exchange=exchange_in, interval=interval_in, start=start, end=end)
futures_candle_df = pd.DataFrame(futures_candles)
print(futures_candle_df)
end = datetime.now()
print('raw data', end-now)
#futures_candle_df['close_forward24'] = futures_candle_df['close'].shift(-1)
futures_candle_df['volume_24HR'] = futures_candle_df['volume'].rolling(24).sum()
# futures_candle_df['volume_24HR$'] = futures_candle_df['volume_24HR'].rolling(24).sum()*futures_candle_df['close']
# futures_candle_df['volume_$'] = futures_candle_df['volume'].rolling(24).sum()*futures_candle_df['close']



futures_candle_df['volume_$'] = futures_candle_df['volume']*futures_candle_df['close']
futures_candle_df['volume_24HR$'] = futures_candle_df['volume_$'].rolling(24).sum()

#make % changes on 24hrs
#%%
shift_df = futures_candle_df.copy(deep=True).shift(shift_val)
#24hr raw feature changes
futures_candle_df = create_changePCT(futures_candle_df, shift_df, "close", "24HR")
futures_candle_df= create_changePCT(futures_candle_df, shift_df, "open", "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, "high", "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, "low", "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume_24HR', "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume_24HR$', "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume', "24HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume_$', "24HR")
#1hr raw feature changes

shift_df1hr = futures_candle_df.copy(deep=True).shift(1)
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, "close", "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, "open", "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, "high", "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, "low", "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, 'volume_24HR', "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df1hr, 'volume_24HR$', "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume', "1HR")
futures_candle_df = create_changePCT(futures_candle_df, shift_df, 'volume_$', "1HR")
#3hr raw feature changes
# shift_df3hr = futures_candle_df.copy(deep=True).shift(3)
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, "close", "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, "open", "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, "high", "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, "low", "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, 'volume_24HR', "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, 'volume_24HR$', "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, 'volume', "3HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df3hr, 'volume_$', "3HR")
# #6hr raw feature changes
# shift_df6hr = futures_candle_df.copy(deep=True).shift(6)
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, "close", "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, "open", "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, "high", "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, "low", "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, 'volume_24HR', "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, 'volume_24HR$', "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, 'volume', "6HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df6hr, 'volume_$', "6HR")
# #12hr raw feature changes
# shift_df12hr = futures_candle_df.copy(deep=True).shift(12)
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, "close", "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, "open", "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, "high", "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, "low", "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, 'volume_24HR', "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, 'volume_24HR$', "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, 'volume', "12HR")
# futures_candle_df = create_changePCT(futures_candle_df, shift_df12hr, 'volume_$', "12HR")
#%%

#for prediction 24hr ahead
futures_candle_df['closeChg%_forward24HR'] = futures_candle_df['closeChg%24HR'].shift(-24)
futures_candle_df['closeChg%_forward1HR'] = futures_candle_df['closeChg%1HR'].shift(-1)
#used parkinson and garman-klass
futures_candle_df['hl_log_sqr'] = np.log(futures_candle_df["high"]/futures_candle_df["low"])**2
#used in garman-klass
futures_candle_df['co_log_sqr'] = np.log(futures_candle_df["close"]/futures_candle_df["open"])**2
#used in rogers-satchell
futures_candle_df['hc_log'] = np.log(futures_candle_df["high"]/futures_candle_df["close"])
futures_candle_df['ho_log'] = np.log(futures_candle_df["high"]/futures_candle_df["open"])
futures_candle_df['lc_log'] = np.log(futures_candle_df["low"]/futures_candle_df["close"])
futures_candle_df['lo_log'] = np.log(futures_candle_df["low"]/futures_candle_df["open"])


#%%
#get EWM based on halflives
day = '3D'
futures_candle_df['close_ewm'+day] = futures_candle_df['close'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['open_ewm'+day] = futures_candle_df['open'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['high_ewm'+day] = futures_candle_df['high'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['low_ewm'+day] = futures_candle_df['low'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hl_log_sqr_ewm'+day] = futures_candle_df['hl_log_sqr'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['co_log_sqr_ewm'+day] = futures_candle_df['co_log_sqr'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hc_log_ewm'+day] = futures_candle_df['hc_log'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['ho_log_ewm'+day] = futures_candle_df['ho_log'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lc_log_ewm'+day] = futures_candle_df['lc_log'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lo_log_ewm'+day] = futures_candle_df['lo_log'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_ewm'+day] = futures_candle_df['volume'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR_ewm'+day] = futures_candle_df['volume_24HR'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_$_ewm'+day] = futures_candle_df['volume_$'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR$_ewm'+day] = futures_candle_df['volume_24HR$'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()

day = '7D'
futures_candle_df['close_ewm'+day] = futures_candle_df['close'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['open_ewm'+day] = futures_candle_df['open'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['high_ewm'+day] = futures_candle_df['high'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['low_ewm'+day] = futures_candle_df['low'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hl_log_sqr_ewm'+day] = futures_candle_df['hl_log_sqr'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['co_log_sqr_ewm'+day] = futures_candle_df['co_log_sqr'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hc_log_ewm'+day] = futures_candle_df['hc_log'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['ho_log_ewm'+day] = futures_candle_df['ho_log'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lc_log_ewm'+day] = futures_candle_df['lc_log'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lo_log_ewm'+day] = futures_candle_df['lo_log'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_ewm'+day] = futures_candle_df['volume'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR_ewm'+day] = futures_candle_df['volume_24HR'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_$_ewm'+day] = futures_candle_df['volume_$'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR$_ewm'+day] = futures_candle_df['volume_24HR$'].ewm(halflife='3 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()



day = '21D'
futures_candle_df['close_ewm'+day] = futures_candle_df['close'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['open_ewm'+day] = futures_candle_df['open'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['high_ewm'+day] = futures_candle_df['high'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['low_ewm'+day] = futures_candle_df['low'].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hl_log_sqr_ewm'+day] = futures_candle_df['hl_log_sqr'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['co_log_sqr_ewm'+day] = futures_candle_df['co_log_sqr'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['hc_log_ewm'+day] = futures_candle_df['hc_log'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['ho_log_ewm'+day] = futures_candle_df['ho_log'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lc_log_ewm'+day] = futures_candle_df['lc_log'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['lo_log_ewm'+day] = futures_candle_df['lo_log'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_ewm'+day] = futures_candle_df['volume'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR_ewm'+day] = futures_candle_df['volume_24HR'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_$_ewm'+day] = futures_candle_df['volume_$'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
futures_candle_df['volume_24HR$_ewm'+day] = futures_candle_df['volume_24HR$'].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
#%% make ewm % changes
shifts = [1, 24]
for col in futures_candle_df.columns:
    if "_ewm" in col:
        for s in shifts:
            shift_df_temp = futures_candle_df.copy(deep=True).shift(s)
            futures_candle_df = create_changePCT(futures_candle_df, shift_df_temp, col, str(s)+"HR")
            
            
futures_candle_df.dropna(inplace=True)
futures_candle_df.reset_index(drop = True, inplace=True)  
#%% make vols


vol_windows = [1*24,3*24,5*24,7*24, 15*24, 30*24, 60*24]
for x in vol_windows:
    #make close2close
    futures_candle_df['vol_c2c_'+str(x/24)+"D"] = (futures_candle_df['closeChg%24HR'].rolling(x).std()*((365)**.5))*100

    
    
    #make parkinson
    futures_candle_df['vol_park_'+str(x/24)+"D"] = (futures_candle_df['hl_log_sqr'].rolling(x).sum()*(1.0 / (4.0*(x)* np.log(2.0))))**.5
    futures_candle_df['vol_park_'+str(x/24)+"D"] = (futures_candle_df['vol_park_'+str(x/24)+"D"]*((365*24)**.5))*100
    
    
    
    #make garman-klass
    mult_one= 1/(2*x)
    mult_two = (2*np.log(2)-1)/x
    futures_candle_df['vol_garman_'+str(x/24)+"D"] = futures_candle_df['hl_log_sqr']-mult_two*futures_candle_df['co_log_sqr']
    futures_candle_df['vol_garman_'+str(x/24)+"D"] = (mult_one*futures_candle_df['vol_garman_'+str(x/24)+"D"].rolling(x).sum())**.5
    futures_candle_df['vol_garman_'+str(x/24)+"D"] = (futures_candle_df['vol_garman_'+str(x/24)+"D"]*((365*24)**.5))*100
    #rogers-stachell
    mult_one = 1/x
    futures_candle_df['vol_rogers_'+str(x/24)+"D"] = (futures_candle_df['hc_log']*futures_candle_df['ho_log'])+ (futures_candle_df['lc_log']*futures_candle_df['lo_log'])
    futures_candle_df['vol_rogers_'+str(x/24)+"D"] = (mult_one*futures_candle_df['vol_rogers_'+str(x/24)+"D"].rolling(x).sum())**.5
    futures_candle_df['vol_rogers_'+str(x/24)+"D"] = (futures_candle_df['vol_rogers_'+str(x/24)+"D"]*((365*24)**.5))*100


    #make EWM base don halflives
    futures_candle_df['vol_c2c_'+str(x/24)+"D"+"_ewm7D"] = futures_candle_df['vol_c2c_'+str(x/24)+"D"].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_c2c_'+str(x/24)+"D"+"_ewm21D"] = futures_candle_df['vol_c2c_'+str(x/24)+"D"].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_park_'+str(x/24)+"D"+"_ewm7D"] = futures_candle_df['vol_park_'+str(x/24)+"D"].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_park_'+str(x/24)+"D"+"_ewm21D"] = futures_candle_df['vol_park_'+str(x/24)+"D"].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_garman_'+str(x/24)+"D"+"_ewm7D"] = futures_candle_df['vol_garman_'+str(x/24)+"D"].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_garman_'+str(x/24)+"D"+"_ewm21D"] = futures_candle_df['vol_garman_'+str(x/24)+"D"].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_rogers_'+str(x/24)+"D"+"_ewm7D"] = futures_candle_df['vol_rogers_'+str(x/24)+"D"].ewm(halflife='7 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    futures_candle_df['vol_rogers_'+str(x/24)+"D"+"_ewm21D"] = futures_candle_df['vol_rogers_'+str(x/24)+"D"].ewm(halflife='21 days', times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
#24hr and 1hr changes  
shifts = [1, 24]
for col in futures_candle_df.columns:
    if "vol_" in col:
        for s in shifts:
            shift_df_temp = futures_candle_df.copy(deep=True).shift(s)
            futures_candle_df = create_changePCT(futures_candle_df, shift_df_temp, col, str(s)+"HR")   
 

#%%get Add Lagged variables
# lags = [1,6,12, 24]

# for col in futures_candle_df.columns:
#     if col not in ['dt', 'exchange', 'timestamp', 'instrument']:
#         for lag in lags:
#             futures_candle_df[col+"_lag"+str(lag)] = futures_candle_df[col].shift(lag)
#%%get pred
futures_candle_df['closeChg%_forward24HR'] = futures_candle_df['closeChg%24HR'].shift(-24)
futures_candle_df['closeChg%_forward1HR'] = futures_candle_df['closeChg%1HR'].shift(-1)
#get up down or chop based on sdev
std_24 = futures_candle_df['closeChg%_forward24HR'].std()
std_1 = futures_candle_df['closeChg%_forward1HR'].std()
for index, row in futures_candle_df.iterrows():
    if abs(row['closeChg%_forward24HR'])>=  std_24:
        if row['closeChg%_forward24HR'] > 0:
            futures_candle_df.at[index,'UpDownPred24HR' ] =1
        else:
            futures_candle_df.at[index,'UpDownPred24HR' ] =-1
    else:
        futures_candle_df.at[index,'UpDownPred24HR' ] =0
    if abs(row['closeChg%_forward1HR'])>=  std_1:
        if row['closeChg%_forward1HR'] > 0:
            futures_candle_df.at[index,'UpDownPred1HR' ] =1
        else:
            futures_candle_df.at[index,'UpDownPred1HR' ] =-1
    else:
        futures_candle_df.at[index,'UpDownPred1HR' ] =0



futures_candle_df.dropna(inplace=True)
futures_candle_df.reset_index(drop = True, inplace=True)  


#%% classification 24hr
start_class = datetime.now()
from pycaret.classification import *
#number_columns = 100/(len(futures_candle_df.columns))
exp_clf24 = setup(futures_candle_df,target='UpDownPred24HR',
        ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR'],session_id=11,
        profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True) 
end_setup_class= datetime.now()
models_class = compare_models(turbo=True, n_select =4)
model_df_class = pull()
best_models_class = model_df_class .iloc[0:4]

end_class = datetime.now()
print('class_setup_config', (end_setup_class - start_class))
#%% regression time using pycaret
# from pycaret.regression import *
# end2 = datetime.now()

# bad_cols = ['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR']
# #select 100 features
# number_columns = 100/(len(futures_candle_df.columns))
# a=setup(futures_candle_df,target='closeChg%_forward24HR',
#         ignore_features=['dt', 'exchange', 'timestamp', 'instrument', 'closeChg%_forward1HR',  'closeChg%_forward1HR', 'closeChg%_forward24HR', 'UpDownPred1HR'],session_id=11,
#         profile=False,  use_gpu=True,  normalize = True,  remove_multicollinearity=True, n_features_to_select=number_columns);



# end_setup = datetime.now()
# #get best models
# models = compare_models(exclude = ['tr', 'lightgbm'],turbo=True, n_select =4)
# #pull error df
# model_df = pull()
# #get top 4 models to train
# best_models = model_df.iloc[0:4]
# #train best models


# end_final = datetime.now()
# print('class_setup_config', (end_setup_class - start_class))
# print('compare_modles', (end_class-end_setup_class))
# print('model compare', (end_final-end_setup))
# print('model_setup', (end_setup-end2))
# print('model compare', (end_final-end_setup))

# print('total_time', (end_final-end))
# model_lst = []

# print(best_models)
#%% regression time using pycaret
# model_lst = []
# model_perform = []
# for x in best_models.index:
#     #train model
#     temp_model = create_model(x)
#     #save error
#     df_1 = pull()
#     #tune model
#     tuned_model = tune_model(temp_model, choose_better= True)
#     #save error
#     df_2 = pull()
    
#     #choose best model based on r2 and save
#     if df_1.loc["Mean"]['R2'] > df_2.loc["Mean"]['R2']:
#         model_lst.append(temp_model)
#         model_perform.append(df_1)
#     else:
#         model_lst.append(tuned_model)
#         model_perform.append(df_2)
    





# direction_correct = 0
# for i in range(len(df)):
#     if (regression_predict_best_blend_tuned.loc[i, 'IndexPrice'] > regression_predict_best_blend_tuned.loc[i, 'IndexPrice_Lag1']) and (regression_predict_best_blend_tuned.loc[i, 'prediction_label'] > regression_predict_best_blend_tuned.loc[i, 'IndexPrice_Lag1']):
#         direction_correct += 1
#     elif (regression_predict_best_blend_tuned.loc[i, 'IndexPrice'] < regression_predict_best_blend_tuned.loc[i, 'IndexPrice_Lag1']) and (regression_predict_best_blend_tuned.loc[i, 'prediction_label'] < regression_predict_best_blend_tuned.loc[i, 'IndexPrice_Lag1']):
#         direction_correct += 1

# direction_accuracy = direction_correct / len(df) * 100
# print(f'Direction accuracy: {direction_accuracy:.2f}%')
#%%plot stuff


# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default='browser'

# fig = px.line(futures_candle_df,y=[futures_candle_df['vol_c2c_'+str(30.0)+"D"], 
#                                     futures_candle_df['vol_garman_'+str(30.0)+"D"],futures_candle_df['vol_rogers_'+str(30.0)+"D"] ,
#                                     futures_candle_df['vol_park_'+str(30.0)+"D"]],  x=futures_candle_df['dt'], 
#               labels= ['C2C', 'garman', 'rogers', 'park'])

# fig.show()
    
    




