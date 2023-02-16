#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:09:01 2023

@author: moussa
"""

import datetime as dt
from datetime import date, timedelta, datetime, timezone
import json
import requests
import time
import pandas as pd
import math
from tqdm import tqdm
import time
import os
from dateutil import tz
import numpy as np
from option_class1 import Option, getOptionData
pd.options.mode.chained_assignment = None  # default='warn'
#pull data from api
def pullHistoric(ts, underlying, fields_in=None, instruments_in =None):
    data = requests.get('http://localhost:8895/deribit_tickers', json={'instruments': instruments_in, 'ts': ts, 'fields': fields_in,'underlying': underlying}).json()
    #print(data['result'])
    data_output = pd.DataFrame(data['result'])
    #data_output['dt']= data_output['dt'].apply(lambda x: dt.datetime.fromtimestamp(x))
    #data_delta = data_output[~data_output['greeks'].notnull()]
    #data_options = data_output[data_output['greeks'].notnull()]
    #data_options['delta'] = data_options['greeks'].apply(lambda x: x['delta'])
    return data_output
#make expiry info helper function
def id_maker(x):
    #x = "ETH 2020-01-17 Call $130.00"
    date_string = x.split(" ")[1].split("-")
    month = date_string[1]
    year = date_string[0]
    day = date_string[-1]
    expiry = datetime(int(year), int(month), int(day), 8, 0, 0, tzinfo=timezone.utc)
    return expiry

#def unique_id_maker()
#test chooser to structure data in output
#['Unique ID', 'Mark ID', 'Cpty', 'Direction', 'Size', 'TradeIV', 'Option Type', 'Expiry', 'Strike','Spot','mark_iv', 'Div', 'D2E', 'MTM Currency', 'COIN']

class TradeOptionLeg:

    def __init__(self, info):
        self.unique_id = info['Unique ID']
        self.settlement_type = info['Settlement Type']
        self.mark_id = info['Mark ID']
        self.cpty = info['Cpty']
        self.direction = info['Direction']
        self.size = info['Size']
        self.trade_px = info['TradePX']
        self.trade_iv = info['TradeIV']
        self.option_type = info['Option Type']
        self.expiry = info['Expiry']
        self.strike = info['Strike']
        #self.spot = info['Spot']
        #self.mark_iv = info['mark_iv']
        #self.div = info['Div']
        self.d2e = info['D2E']
        self.mtm_currency = info['MTM Currency']
        self.coin = info['COIN']
        self.update_time = info['UpdateTime']


class TradeOptionAgg:

    def __init__(self, info, fees = 0):
        self.unique_id = info.unique_id
        self.mark_id = info.mark_id
        self.settlement_type = info.settlement_type

        self.buy_size = 0
        self.sell_size = 0
        self.buy_px = 0
        self.sell_px = 0
        self.buy_dpx = 0
        self.sell_dpx =0
        self.option_type = info.option_type
        self.expiry = info.expiry
        self.strike = info.strike
        self.spot = 0
        self.mark_iv = 0
        self.div = 0
        self.d2e = info.d2e
        self.mtm_currency = info.mtm_currency
        self.coin = info.coin
        self.fees_per_trade = fees
        self.cur_size = 0
        if info.direction =="Buy":
            self.buy_size = info.size
            self.buy_px = info.trade_px
            self.cur_size = info.cur_size
        else:
            self.sell_size = info.size
            self.sell_px = info.trade_px
            self.cur_size = info.size*-1
        self.trade_info = [info]
        #####maybe change fees for real pms
        self.fees_paid = self.fees_per_trade*((self.buy_size*self.buy_px)+(self.sell_size*self.sell_px))
        self.update_time = info.update_time
        #update $ price
        if self.mtm_currency !="USD":
            self.buy_dpx = self.buy_px*self.spot
            self.sell_dpx = self.sell_px*self.spot
        else:
            self.buy_dpx = self.buy_px
            self.sell_dpx = self.sell_px
        self.expired = 0
        self.info_dict = {}
        self.premium_delta = 0
        if self.mtm_currency != 'USD':
            self.premium_delta = (self.buy_size*self.buy_px)*-1+(self.sell_size*self.sell_px)
    def updateTrade(self, info):
        if info.unique_id == self.unique_id:
            if info.direction== "Buy":
                old_pxqty = (self.buy_size*self.buy_px)
                trade_pxqty = (info.size*info.trade_px)
                cur_pxqty = old_pxqty+trade_pxqty
                cur_size = (self.buy_sizer+info.size)
                avg_px = cur_pxqty/cur_size
                self.buy_size = cur_size
                self.buy_px = avg_px

            else:
                old_pxqty = (self.sell_size*self.sell_px)
                trade_pxqty = (info.size*info.trade_px)
                cur_pxqty = old_pxqty+trade_pxqty
                cur_size = (self.sell_size+info.size)
                avg_px = cur_pxqty/cur_size
                self.sell_size = cur_size
                self.sell_px = avg_px
            #update to current for pricing
            #self.spot = info.spot
            #self.mark_iv = info.mark_iv
            #self.div = info.div
            self.d2e = info.d2e
            self.update_time = info.update_time
            #add trades
            self.trade_info.append(info)
            #update fees paid and cur_size
            self.fees_paid = self.fees_per_trade*((self.buy_size*self.buy_px)+(self.sell_size*self.sell_px))
            self.cur_size = self.buy_size-self.sell_size
            #update $ prices
            if self.mtm_currency !="USD":
                self.buy_dpx = self.buy_px*self.spot
                self.sell_dpx = self.sell_px*self.spot
                self.premium_delta = (self.buy_size*self.buy_px)*-1+(self.sell_size*self.sell_px)
            else:
                self.buy_dpx = self.buy_px
                self.sell_dpx = self.sell_px

            return
        else:
            print("Wrong Ticker to Agg in Options Update")
            return
    def updateMarks(self, spot, iv, div, time):
        self.spot = spot
        self.mark_iv = iv
        self.div = div
        self.time = time
        return
    def createDict(self):
        temp = {}
        temp['unique_id'] = self.unique_id
        temp['mark_id'] = self.mark_id
        temp['settlement_type'] = self.settlement_type

        temp['buy_size'] = self.buy_size
        temp['sell_size'] = self.sell_size
        temp['buy_px'] = self.buy_size
        temp['sell_px'] = self.sell_px
        temp['buy_dpx'] = self.buy_dpx
        temp['sell_dpx'] =self.sell_dpx
        temp['option_type'] = self.option_type
        temp['expiry']= self.expiry
        temp['strike'] = self.strike
        temp['spot'] = self.spot
        temp['mark_iv'] = self.mark_iv
        temp['div'] = self.div
        temp['d2e'] = self.d2e
        temp['mtm_currency'] = self.mtm_currency
        temp['coin'] = self.coin
        temp['cur_size']=self.cur_size 
        temp['update_time'] = self.update_time
        temp['fees_paid'] = self.fees_paid
        temp['expired'] = self.expired
        temp['premium_delta'] = self.premium_delta
        self.info_dict = temp
        return
    
     
class DeltaTrade:
    def __init__(self, info, fees=0):
        self.unique_id = info['Unique ID']
        self.mark_id = info['Mark ID']
        self.settlement = info['Settlement']
        self.type = info['Type']
        self.coin = info['COIN']
        self.expiry = info['Expiry']
        self.buy_size = 0
        self.buy_px = 0

        self.sell_size =0
        self.sell_px = 0


        self.fees_per_trade = fees

        self.trade_info = [info]
        self.update_time = info['TradeTime']
        if info['TradeDirection'] == "Buy":
            self.buy_size = info['TradeSize']
            self.buy_px = info['TradePx']

        else:
            self.sell_size = info['TradeSize']
            self.sell_px = info['TradePx']

        self.fees_paid = self.fees_per_trade*((self.buy_size*self.buy_px)+(self.sell_size*self.sell_px))
        self.cur_delta = self.buy_size - self.sell_size

        self.mark_px = 0
        self.pnl = 0
        self.expired = 0
        self.info_dict = {}

        
    def updateTrade(self, info):
        #calc average of buy sells in line with only know n-1
        if info['Unique ID'] == self.unique_id:
            if info['TradeDirection'] == "Buy":
                old_pxqty = (self.buy_size*self.buy_px)
                trade_pxqty = (info['TradeSize']*info['TradePx'])
                cur_pxqty = old_pxqty+trade_pxqty
                cur_qty = (self.buy_size+info['TradeSize'])
                avg_px = cur_pxqty/cur_qty
                self.buy_size = cur_qty
                self.buy_px = avg_px

            else:
                old_pxqty = (self.sell_size*self.buy_px)
                trade_pxqty = (info['TradeSize']*info['TradePx'])
                cur_pxqty = old_pxqty+trade_pxqty
                cur_qty = (self.sell_size+info['TradeSize'])
                avg_px = cur_pxqty/cur_qty
                self.sell_size = cur_qty   
                self.size =cur_qty
                self.sell_px = avg_px

            self.update_time = info['TradeTime']
            self.trade_info.append(info)
            self.fees_paid = self.fees_per_trade*((self.buy_size*self.buy_px)+(self.sell_size*self.sell_px))
            self.cur_delta = self.buy_size-self.sell_size
            return
        else:
            print("Wrong Ticker to Agg in Delta Update")
            return
    def MTM(self, mark, time):
        #update avg trade buy sell pnl based on mark and save pnl and mark
        pnl_sum = 0
        if self.buy_size !=0:
            pnl_sum += (mark-self.buy_px)*self.buy_size
        if self.sell_size !=0:
            pnl_sum += (self.sell_px-mark)*self.sell_size
        self.pnl = pnl_sum
        self.mark_px = mark
        self.update_time = time
        return
    def createDict(self):
        temp = {}
        temp['unique_id'] = self.unique_id
        temp['mark_id'] = self.mark_id
        temp['settlement_type'] = self.settlement

        temp['buy_size'] = self.buy_size
        temp['sell_size'] = self.sell_size
        temp['buy_px'] = self.buy_size
        temp['sell_px'] = self.sell_px
        temp['expiry']= self.expiry
        temp['mark_px']= self.mark_px
        temp['expired'] = self.expired
        temp['fees_paid'] = self.fees_paid
        temp['cur_size'] = self.cur_delta
        temp['PNL'] = self.pnl
        temp['coin'] = self.coin

        self.info_dict = temp
        return
    


from collections import defaultdict       
class Portfolio:
    def __init__(self,ts, posOptions_dict, posDelta_dict):
        self.options_positions = posOptions_dict
        self.delta_positions = posDelta_dict
        self.now_options = ts
        self.now_delta = ts
        self.delta_mark_dict = {}
        #iv, div, spot, time
        self.options_mark_dict = {}
        #save basis interpolated function for price otc futures
        self.basis_interp_dict = {}
        #indexed by coin
        self.pnl_dict = None
        self.options_risk = None
        self.options_risk_hist = {}
        self.delta_risk = None
        self.delta_risk_hist = {}
        #sumamry table
        self.summary_table = None
    # def summary_risk(self):
    #     unique_coins = set(self.options_risk['coin'].unique(), self.delta_risk['coin'].unique())
    #     print(unique_coins)
    #     data= []
    #     for uc in unique_coins:
    #         temp_delta = self.delta_risk[self.delta_risk['coin']==uc]
    #         temp_options = self.options_risk[self.options_risk['coin']==uc]
    #         delta = temp_delta['cur_size'].sum()+(temp_options['Unit Delta']+temp_options['Premium Delta']).sum()
            
    def price_delta(self, ts, delta_prices):
        table_delta = []
        for key, value in self.delta_positions.items():
            self.delta_positions[key].MTM(delta_prices[value.mark_id], ts)
            self.delta_positions[key].createDict()
            temp_delta = self.delta_positions[key].info_dict
            table_delta.append(temp_delta)
        table_delta = pd.DataFrame(table_delta)
        self.now_delta = ts
        self.delta_marks_dict = delta_prices
        self.delta_risk = table_delta
        self.delta_risk_hist[ts] = table_delta
        return
    def price_options(self, ts, option_marks, store = True):
        #require port spot marks and vol marks to be updated to be updated assumes are
        table_options = []
        already_priced = {}
        for key, value in self.options_positions.items():
            #update spot and time
            self.options_positions[key].spot = self.delta_marks_dict[self.options_positions[key].coin]
            self.options_positions[key].update_time = ts
            if self.options_positions[key].mark_id in already_priced.keys():
                temp_marks = already_priced[self.options_positions[key].mark_id]
                self.options_positions.updateMarks(temp_marks)
            else:
                spot = self.delta_marks_dict[self.options_positions[key].coin]
                vol = option_marks.loc[self.options_positions[key].mark_id]['mark_iv']
                div =  option_marks.loc[self.options_positions[key].mark_id]['Div']
                already_priced[self.options_positions[key].mark_id] = [spot, vol, div, ts]
                self.options_positions[key].updateMarks( spot, vol, div, ts)
            value.createDict()
            temp_op = value.info_dict
            table_options.append(temp_op)
        option_price_dict = {}
        table_options = pd.DataFrame(table_options)
        #need net position
        self.now_options = ts
        table_priced_options = self.price_options_helper(table_options.copy(deep=True), ts)
        self.options_risk = table_priced_options
        self.options_risk_hist[ts] = table_priced_options
        return

    

    def price_options_helper(self, df,now, new_spot=False):
        df1 = df[df['option_type']== 'Call'].copy()
        sub_call = pd.DataFrame()
        if df1.shape[0] > 0:
            temp_call = Option(df['mark_id'], 'call', df1['strike'], df1['expiry'], df['d2e']/365, size =df1['cur_size'], 
                               dividend_rate=df1['div'], volatility=df1['mark_iv'], underlying_price = df1['spot'])
            column_names , ret_data = getOptionData(temp_call)
            sub_call = pd.DataFrame(data=dict(zip(column_names,ret_data)))
            df_1 = df.iloc[:,df.columns.isin(sub_call.columns)==False]
            sub_call = df_1.join(sub_call)
        df2 = df[df['option_type']== 'Put'].copy()
        sub_put = pd.DataFrame()
        if df2.shape[0] > 0:
            temp_put = Option(df2['mark_id'], 'put', df2['strike'], df2['expiry'], df2['d2e']/365, size = df2['cur_size'], 
                                dividend_rate=df2['div'], volatility=df2['mark_iv'], underlying_price = df2['spot'])
            column_names , ret_data = getOptionData(temp_put)
            sub_put = pd.DataFrame(data=dict(zip(column_names,ret_data)))
            df_2 = df.iloc[:,df.columns.isin(sub_put.columns)==False]
            sub_put = df_2.join(sub_put)
        df_5 = pd.concat([sub_call, sub_put], ignore_index=True)
        df_5['Unit Delta'] = df_5['Delta']*df_5['cur_size']
        #df['PNL'] = (df['']-df['TradePX$'])*df['Pos']
        return df_5

def chooser_simple(single_date, underlying, data, direction ="Sell", size = 1):
    size = 1
    direction = "Sell"
    #date set up
    options = data[data['greeks_gamma'].notnull()]
    delta =data[~data['greeks_gamma'].notnull()]
    perp_data = data[data['product_id']==underlying+"-USD"]

    delta_data = dict()
    delta_data['Prod ID'] = perp_data['product_id'].values[0]
    delta_data['Spot'] =perp_data['index_price'].values[0]
    delta_data['Bid'] =perp_data['best_bid_price'].values[0]
    delta_data['Ask'] = perp_data['best_ask_price'].values[0]
    delta_data['Unique ID'] = "Deribit "+perp_data['product_id'].values[0] +" Perp None CASH"
    delta_data['Mark ID'] = perp_data['instrument_name'].values[0]
    delta_data['Type'] = "Perp"
    delta_data['Settlement'] = "CASH"
    delta_data['Expiry'] = None
    #SIMPLE CHOICE FOR ATM



    options['Expiry'] = options['product_id'].apply(id_maker)
    options['Strike'] = options['product_id'].apply(lambda x: float(x.split(" ")[-1][1:]))
    #make unique_id
   

    options['Option Type'] = options['product_id'].apply(lambda x: "Call" if x.split("-")[-1] else "Put")
    options['Cpty'] = "Deribit"

    options['COIN'] = underlying
    options['MTM Currency'] = underlying
    #need to get forward
    options['Future Prem$'] = (options['underlying_price']-options['index_price'])
    options['D2E'] = (options['Expiry']-single_date).apply(timedelta.total_seconds)/(60*60*24)
    options['Div'] = (options['Future Prem$'])/(options['index_price'])*(365/options['D2E'])*-1
    options['Settlement Type'] = "CASH"
    #make IDs
    options['Unique ID'] = options['Cpty']+" " + options['product_id']+" Option "+options['Expiry'].astype(str).apply(lambda x: x.split("+")[0].split(" ")[1]) + " CASH "+ options['MTM Currency']
    options['Mark ID'] = options['Unique ID'].apply(lambda x: " ".join(x.split(" ")[1:7]))


    #input spot ref
    options['Spot'] = delta_data['Spot']
    options['UpdateTime'] = single_date
    options['mark_iv'] /= 100
  

    #get trades
    #trades = options[(options['product_id'])=="ETH 2020-06-26 Call $160.00"]
    options['abs_delta'] = options['greeks_delta'].apply(lambda x: abs(x))
    trades = options[(options['abs_delta']>.45)&(options['abs_delta']< .55)]
    trades['Direction'] = direction
    trades['Size'] = 1
    trades['cur_size'] = 1
    trades['TradeIV'] = trades['bid_iv']
    trades['TradePX'] = trades['best_bid_price']
    trades['Hedge Units'] = (trades['greeks_delta']*trades['Size'])

    trades['Premium Paid'] = (trades['TradePX']*trades['cur_size'])*-1
    
    trades['Effective Delta Hedge'] = trades['Hedge Units']+trades['Premium Paid']   
    trades['UpdateTime'] = single_date
    
    #set up initial delta hedge
    delta_data['TradePx'] = delta_data['Ask']
    delta_data['TradeSize'] = abs(trades['Effective Delta Hedge'].sum())
    delta_data['TradeDirection'] = "Buy"
    delta_data['TradeTime'] = single_date
    delta_data['COIN'] = underlying
    #return df of trades and dict for spot ref pricing and also hedging and all options for pricing
    return trades[['Unique ID', 'Mark ID', 'Cpty', 'Direction', 'Size', 'TradePX', 'TradeIV', 'Option Type', 'Expiry', 'Strike','Spot','mark_iv', 'Div', 'D2E', 'MTM Currency', 'Settlement Type',  'COIN', 'UpdateTime']], delta_data, options
#%%
def getExpiry(date_time, currency):
    date = str(date_time).split("+")[0]
    
start = datetime(2020,1,2, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2020,1,3, 0, 0, 0, tzinfo=timezone.utc)
#instruments = ['ETH-20JAN23-1400-C']

now = datetime.now()
trade_dict_options = {}
trade_dict_delta = {}
mark_id_iv_dict = {}
mark_id_delta_dict = {}
port = Portfolio(None, {}, {})
for single_date in tqdm(pd.date_range(start, end, freq='30T')):
    #print(single_date)
    ts = single_date.timestamp()
    fields = None
    underlying = 'ETH'
    #get data
    data = pullHistoric(ts, underlying, fields_in = fields)
    delta_data_raw = data[data['mark_iv'].isnull()]
    spot = {}
    for index, row in delta_data_raw.iterrows():
        spot[row['instrument_name']] = row['mark_price']
        

    trades, delta_info, all_options = chooser_simple(single_date,'ETH', data)
    all_options['mark_id']= all_options['Mark ID']
    all_options.index = all_options['mark_id']

    spot['ETH'] = delta_info['Spot']
    

    
    #port
    


    new_trade_dict = {}
    for index, row in trades.iterrows():
        new_trade = TradeOptionLeg(row)
        
        
        if new_trade.unique_id in port.options_positions.keys():
            port.options_positions[new_trade.unique_id].updateTrade(new_trade)
            #for marking purposes

        else:
            port.options_positions[new_trade.unique_id] = TradeOptionAgg(new_trade)
        
    if delta_info['TradeSize'] != 0:
        if delta_info['Unique ID'] in trade_dict_delta.keys():
            port.delta_positions[delta_info['Unique ID']].updateTrade(delta_info)
        else:
            port.delta_positions[delta_info['Unique ID']] = DeltaTrade(delta_info)
    #port order
    #1
    port.delta_marks_dict = spot
    #2
    port.price_options(single_date, all_options)
    #3
    port.price_delta(ts, spot)

x = port.options_risk
y = port.delta_risk
end = datetime.now()
print(end-now)

x
#%%
# requests.get('http://localhost:8895/deribit_settlements', json={'underlying': 'ETH', 'date': '2022-12-06 08:00:00'}).json()['result']
# {'type': 'settlement', 'timestamp': 1670313600032, 'session_profit_loss': 7425.492660455, 'profit_loss': -244.91858944, 'position': 10541259, 'mark_price': 1260.34, 'instrument_name': 'ETH-PERPETUAL', 'index_price': 1260.35, 'funding': -0.049261112, 'rtime': '2022-12-06T08:06:08.945000', 'usrt': 1670313968945990, 'dt': '2022-12-06T08:00:00.032000', 'usdt': 1670313600032000, 'product_id': 'ETH-USD', 'key': 1}
x =requests.get('http://localhost:8895/deribit_settlements', json={'underlying': 'ETH', 'date': '2023-1-19 08:00:00'}).json()['result']