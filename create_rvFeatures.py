
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
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
def create_changePCT(df, shift, column, time):
    

    df[column+"Chg"+time] = (df[column]-shift[column])
    df[column+"Chg%"+time] = df[column+"Chg"+time]/shift[column]
    return df


def generate_rv_features(start, end, ticker, interval_in,exchange_in="binance", changes =[1,3,6,12,24], ewm_days = [3,7,21],vol_windows=[1,3,7,15,30,60], future=True):
#%%
    """
    changes in hours
    ewm_days in days
    vol_windows in days
    future bool for spot or future
    
    """
    amberdata = AmberData()
    #get futures or spot
    if future:
        futures_candles = amberdata.get_futures_candles(instrument=ticker, exchange=exchange_in, interval=interval_in, start=start, end=end)
    else:
        futures_candles = amberdata.get_spot_candles(instrument=ticker, exchange=exchange_in, interval=interval_in, start=start, end=end)
    futures_candle_df = pd.DataFrame(futures_candles)
    #get day step for rows
    day_step = 1
    if interval_in == "hours":
        day_step = 24
    elif interval_in == "minutes":
        day_step = 24*60
    hour_step = day_step/24
    futures_candle_df['volume_24HR'] = futures_candle_df['volume'].rolling(day_step).sum()
    
    
    futures_candle_df['volume_$'] = futures_candle_df['volume']*futures_candle_df['close']
    futures_candle_df['volume_24HR$'] = futures_candle_df['volume_$'].rolling(day_step).sum()

    #make % changes on 24hrs
    #%% Make raw feature changes
    raw_columns = ["close", "open", "high", "low",'volume_24HR', 'volume_24HR$', 'volume','volume_$']
    for c in changes:
        name = str(c)+"HR"

        shift_amount = int(hour_step*c)

        shift_df = futures_candle_df.shift(shift_amount)
        for col in raw_columns:
            futures_candle_df = create_changePCT(futures_candle_df, shift_df, col, name)
        


    #%%
    #used parkinson and garman-klass
    futures_candle_df['hl_log_sqr'] = np.log(futures_candle_df["high"]/futures_candle_df["low"])**2
    #used in garman-klass
    futures_candle_df['co_log_sqr'] = np.log(futures_candle_df["close"]/futures_candle_df["open"])**2
    #used in rogers-satchell
    futures_candle_df['hc_log'] = np.log(futures_candle_df["high"]/futures_candle_df["close"])
    futures_candle_df['ho_log'] = np.log(futures_candle_df["high"]/futures_candle_df["open"])
    futures_candle_df['lc_log'] = np.log(futures_candle_df["low"]/futures_candle_df["close"])
    futures_candle_df['lo_log'] = np.log(futures_candle_df["low"]/futures_candle_df["open"])

    


#%% make vols

    
    #make rv
    for x in vol_windows:
        ##x is days
        #make close2close
        days = 24*x
        window_size = day_step*x
        #make close2close annualized
        futures_candle_df['vol_c2c_'+str(x)+"D"] = (futures_candle_df['closeChg%24HR'].rolling(window_size).std()*((365)**.5))*100
        #make parkinson
        futures_candle_df['vol_park_'+str(x)+"D"] = (futures_candle_df['hl_log_sqr'].rolling(window_size).sum()*(1.0 / (4.0*(window_size)* np.log(2.0))))**.5
        futures_candle_df['vol_park_'+str(x)+"D"] = (futures_candle_df['vol_park_'+str(x)+"D"]*((365*day_step)**.5))*100
        #make garman-klass
        mult_one= 1/(2*window_size)
        mult_two = (2*np.log(2)-1)/window_size
        futures_candle_df['vol_garman_'+str(x)+"D"] = futures_candle_df['hl_log_sqr']-mult_two*futures_candle_df['co_log_sqr']
        futures_candle_df['vol_garman_'+str(x)+"D"] = (mult_one*futures_candle_df['vol_garman_'+str(x)+"D"].rolling(window_size).sum())**.5
        futures_candle_df['vol_garman_'+str(x)+"D"] = (futures_candle_df['vol_garman_'+str(x)+"D"]*((365*day_step)**.5))*100
        #rogers-stachell
        mult_one = 1/window_size
        futures_candle_df['vol_rogers_'+str(x)+"D"] = (futures_candle_df['hc_log']*futures_candle_df['ho_log'])+ (futures_candle_df['lc_log']*futures_candle_df['lo_log'])
        futures_candle_df['vol_rogers_'+str(x)+"D"] = (mult_one*futures_candle_df['vol_rogers_'+str(x)+"D"].rolling(window_size).sum())**.5
        futures_candle_df['vol_rogers_'+str(x)+"D"] = (futures_candle_df['vol_rogers_'+str(x)+"D"]*((365*day_step)**.5))*100
        ###########################

    
        
    #make rv changes
    for col in futures_candle_df.columns:
        if "vol_" in col:
            for c in changes:
                name = str(c)+"HR"
                shift_amount = int(hour_step*c)
                shift_df = futures_candle_df.shift(shift_amount)
                futures_candle_df = create_changePCT(futures_candle_df, shift_df, col, name)
                    
    
    #%% make ewm % changes              

    raw_columns = ["close", "open", "high", "low",'volume_24HR', 'volume_24HR$', 'volume','volume_$','hl_log_sqr',  'co_log_sqr','hc_log', 'ho_log', 'lc_log', 'lo_log' ]
    for col in futures_candle_df.columns:
        if "vol_" in col:
            if "Chg" not in col:
                raw_columns.append(col)
    
    for ewm_d in ewm_days:
        for col in raw_columns:
            day = str(ewm_d)+"D"
            day_half = str(ewm_d)+ " days"
            futures_candle_df[col+"_ewm"+day] = futures_candle_df[col].ewm(halflife=day_half, times=pd.DatetimeIndex(futures_candle_df['dt'])).mean()
    #%% make ewm % changes
    for col in futures_candle_df.columns:
        if "_ewm" in col:
            for c in changes:
                name = str(c)+"HR"
                shift_amount = int(hour_step*c)
                shift_df = futures_candle_df.shift(shift_amount)
                futures_candle_df = create_changePCT(futures_candle_df, shift_df, col, name)

                 
                 
    futures_candle_df.dropna(inplace=True)
    futures_candle_df.reset_index(drop = True, inplace=True)  
    return futures_candle_df

#%% test
# data_start = datetime.now()

# now = datetime.now()
# start = dt.datetime(2018, 1, 1)
# end = dt.datetime.combine(dt.date.today(), dt.time.min)


# ticker = "BTCUSDT"
# interval_in = 'hours'
# exchange_in="binance", 
# changes =[1,3,6,12,24]
# ewm_days = [3,7,21]
# vol_windows=[1,3,7,15,30,60]
# future=True

# df = generate_rv_features(start, end, ticker, interval_in,exchange_in="binance", changes =[1,3,6,12,24], ewm_days = [3,7,21],vol_windows=[1,3,7,15,30,60], future=True)
# data_end = datetime.now()
# print("Gather RV", data_end-data_start)




