#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:45:15 2023

@author: moussa
"""
import csv
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine
import os
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict 
from scipy.interpolate import interp1d
from scipy.stats import norm
import pickle 
import warnings
import time
import pymongo
import logging
class Option:
    def __init__(self,
                 underlying_pair,
                 option_type,
                 strike,
                 expiry,
                 time_left, 
                 size=1,
                 interest_rate=0,
                 dividend_rate =0, 
                 volatility=None,
                 underlying_price=None,
                 time=datetime.utcnow(),
                 exchange_symbol=None):
        self.underlying_pair = underlying_pair
#         if not option_type == 'call' and not option_type == 'put':
#             raise ValueError('Expected "call" or "put", got ' + option_type)
        self.option_type = option_type
        self.strike = strike.astype(float)
        self.expiry = expiry
        self.interest_rate = interest_rate
        self.dividend_rate = dividend_rate.astype(float)
        if volatility is not None:
            self.vol = volatility
        else:
            self.vol = None
        self.underlying_price = underlying_price.astype(float)
        self.time = time
        self.exchange_symbol = exchange_symbol
        self.time_left = time_left.astype(float)
        self.size = size.astype(float)
        self.d1 = None
        self.d2 = None
        self.theo = None
        self.theo_btc = None
        self.delta = None
        self.dollar_delta = None
        self.unit_delta = None
        self.gamma = None
        self.dollar_gamma = None
        self.theta = None
        self.dollar_theta = None
        self.vega = None
        self.dollar_vega = None
        self.wvega = None
        self.present_value = None
        self.mid_market = None
        self.best_bid = None
        self.best_ask = None
        self.gamma_theta = None
        self.normalized_strike = None
        self.time = time


    def __str__(self):
        return self.underlying_pair + " " + str(self.strike) + " " + self.option_type + " with expiry " + str(self.expiry)


    def set_underlying_price(self, underlying_price):
        self.underlying_price = underlying_price

    def get_metadata(self, timestamp=None):
        if timestamp is None:
            timestamp = str(self.time())
        return {
            'timestamp': timestamp,
            'expiry': str(self.expiry)[:10],
            'type': self.option_type,
            'strike': str(self.strike),
            'delta': str(self.delta),
            'gamma': str(self.gamma),
            'theta': str(self.theta),
            'wvega': str(self.wvega),
            'vega': str(self.vega),
            'vol': str(self.vol),
            'best_bid': str(self.best_bid),
            'best_ask': str(self.best_ask),
            'exchange_symbol': self.exchange_symbol
        }

    def set_time(self, time):
        self.time = time

    def set_vol(self, vol):
        self.vol = vol

    def set_mid_market(self, mid_market=None):
        if mid_market is not None:
            self.mid_market = mid_market
        else:
            if self.best_bid is not None and self.best_ask is not None:
                self.mid_market = (self.best_bid + self.best_ask) / 2
        logging.info("Set mid market: " + str(self.mid_market))

    def calc_greeks(self, verbose=False):
        self.calc_theo()
        self.calc_delta()
        self.calc_gamma()
        self.calc_theta()
        self.calc_vega()
        
        try:
            self.gamma_theta = self.dollar_gamma/self.dollar_theta
        except:
            self.gamma_theta = 0
        self.normalized_strike = np.log(self.strike/self.underlying_price)/((self.time_left**(1/2))*self.vol)

        if verbose: 
            print(str(self))
            
            print("Position: ", self.size)
            print("Theo Val: ", round(self.theo, 3))
            print("Delta: ", round(self.delta, 3))
            print("Gamma: ", round(self.gamma,3))
            print("Theta: ", round(self.theta, 3))
            print("Vega: ", round(self.vega, 3))
            print("Unit Delta: ", round(self.unit_delta, 3))
            print("$Delta: ", round(self.dollar_delta, 3))
            print("$Gamma: ", round(self.dollar_gamma, 3))
            print("$Theta: ", round(self.dollar_theta, 3))
            print("$Vega: ", round(self.dollar_vega, 3))
            
    def calc_theo(self, time_left=None, store=True):
        
        if not time_left:
            time_left = self.time_left
        d1 = (np.log(self.underlying_price / self.strike) + time_left*(self.interest_rate-self.dividend_rate +((self.vol ** 2)/2.0)))/(self.vol * np.sqrt(time_left))
        d2 = d1 - (self.vol * np.sqrt(time_left))
        theo = None
        if self.option_type == "call":
            theo = self.underlying_price *np.exp(-1*self.dividend_rate*time_left)*norm.cdf(d1) - self.strike*np.exp(-1*self.interest_rate*time_left)*norm.cdf(d2)
        elif self.option_type == "put":
            theo = -1*self.underlying_price *np.exp(-1*self.dividend_rate*time_left)*norm.cdf(-1*d1) + self.strike*np.exp(-1*self.interest_rate*time_left)*norm.cdf(-1*d2)
        if store:
            self.d1 = d1
            self.d2 = d2
            self.theo = theo.astype(float)
            self.theo_btc = self.theo/self.underlying_price
        return theo

    def calc_delta(self):
        delta = None
        if self.option_type == 'call':
            delta = np.exp(-self.dividend_rate*self.time_left)*norm.cdf(self.d1)
        elif self.option_type == 'put':
            delta = np.exp(-self.dividend_rate*self.time_left)*(norm.cdf(self.d1)-1)
        self.delta = delta
        self.unit_delta = delta*self.size
        self.dollar_delta = self.unit_delta*self.underlying_price
        return delta

    def calc_gamma(self):
        try:
            self.gamma = np.exp(-1*self.dividend_rate*self.time_left)/(self.underlying_price*self.vol*np.sqrt(self.time_left))*np.exp(-1*(self.d1**2)/2)/np.sqrt(2*np.pi)
            self.dollar_gamma = self.gamma*(self.underlying_price**2)/100*self.size
        except:
            self.gamma = 0
            self.dollar_gamma = 0
        return self.gamma

    def calc_theta(self, time_change=1/365):
        try:
            if self.option_type == "call":
                self.theta = (-(self.underlying_price*np.exp(-1*(self.d1**2)/2)/np.sqrt(2*np.pi)*self.vol*np.exp(-self.dividend_rate*self.time_left)/(2*np.sqrt(self.time_left)))-(self.interest_rate*self.strike*np.exp(-self.interest_rate*self.time_left)*norm.cdf(self.d2))+(self.dividend_rate*self.underlying_price*np.exp(-self.dividend_rate*self.time_left)*norm.cdf(self.d1)))/365
            else:
                self.theta = (-(self.underlying_price*np.exp(-1*(self.d1**2)/2)/np.sqrt(2*np.pi)*self.vol*np.exp(-self.dividend_rate*self.time_left)/(2*np.sqrt(self.time_left)))+(self.interest_rate*self.strike*np.exp(-self.interest_rate*self.time_left)*norm.cdf(-1*self.d2))-(self.dividend_rate*self.underlying_price*np.exp(-self.dividend_rate*self.time_left)*norm.cdf(-1*self.d1)))/365
            self.dollar_theta = self.theta*self.size
        except:
            self.theta = 0
            self.dollar_theta = 0
        return self.theta

    def calc_vega(self, vol_change=.01):
        try:
            original_vol = self.vol
            original_theo = self.theo
            self.vega = np.exp(-1*(self.d1**2)/2)/np.sqrt(2*np.pi)*np.exp(-1*self.dividend_rate*self.time_left)*self.underlying_price*np.sqrt(self.time_left)/100
            self.dollar_vega = self.vega*self.size
        except:
            self.vega = 0
            self.dollar_vega = 0
        return self.vega

    # Weighted vega = vega / atm_vega
    def calc_wvega(self, atm_vega):
        self.wvega = self.calc_vega() / atm_vega
        return self.wvega

    # Price in BTC
    def calc_implied_vol(self, btc_price=None, num_iterations=1000, accuracy=.00001, low_vol=0, high_vol=10):
        if btc_price is None:
            usd_price = self.mid_market
        else:
            usd_price = btc_price * self.underlying_price
        self.calc_theo()
        for i in range(num_iterations):
            if self.theo > usd_price + accuracy:
                high_vol = self.vol
            elif self.theo < usd_price - accuracy:
                low_vol = self.vol
            else:
                break
            self.vol = low_vol + (high_vol - low_vol) / 2.0
            self.calc_theo()
        return self.vol
    
def instrument_name_to_product_id(exchange, instrument_name):

    components = instrument_name.split('-')

    if len(components) == 4:    # option
        # ETH-24JUN22-2600-P => ETH 2022-06-24 Put $2,600.00
        underlying = components[0]
        regex = re.compile("([0-9]+)([A-Z]+)([0-9]+)")
        matches = regex.match(components[1])
        date = dt.datetime.strptime(' '.join([
            matches.group(1), 
            matches.group(2).capitalize(), 
            matches.group(3)
        ]), '%d %b %y').date().isoformat()
        option_type = 'Put' if components[3] == 'P' else 'Call'
        strike = '${:,.2f}'.format(float(components[2]))
        product_id = f'{exchange} {underlying} {date} {option_type} {strike} Option'

    elif components[1] == 'PERPETUAL':   # perp
        # ETH-PERPETUAL => ETH-USD
        underlying = components[0]
        product_id = f'{exchange} {underlying} Perp'

    else:  # future
        # ETH-24JUN22 => ETH 2022-06-24 Future
        underlying = components[0]
        regex = re.compile("([0-9]+)([A-Z]+)([0-9]+)")
        matches = regex.match(components[1])
        date = dt.datetime.strptime(' '.join([
            matches.group(1), 
            matches.group(2).capitalize(), 
            matches.group(3)
        ]), '%d %b %y').date().isoformat()
        product_id = f'{exchange} {underlying} {date} Future'

    return product_id

def getOptionData(option_obj):
    try:
        option_obj.calc_greeks()
        
        ret_data = []
        names = []
        name = option_obj.underlying_pair
        ret_data.append(name)
        names.append("Mark ID")
        
        option_type = option_obj.option_type
        ret_data.append(option_type)
        names.append("option_type")
        
        strike = option_obj.strike
        ret_data.append(strike)
        names.append("Strike")
        
        normalized_strike = option_obj.normalized_strike
        ret_data.append(normalized_strike)
        names.append("Normalized Strike")
        
        expiry = option_obj.expiry
        ret_data.append(expiry)
        names.append("Expiry")
        
        time_left = option_obj.time_left*365
        ret_data.append(time_left)
        names.append("Days to Expiry")
        
    
        
        dividend_rate = option_obj.dividend_rate
        ret_data.append(dividend_rate)
        names.append("Div Rate")
        
        iv = option_obj.vol
        ret_data.append(iv)
        names.append("IV")
    
        gamma_theta = option_obj.gamma_theta
        ret_data.append(gamma_theta)
        names.append("$Gamma/$Theta")
    
    
        spot = option_obj.underlying_price
        ret_data.append(spot)
        names.append("Spot Mark")
        
        px_usd = option_obj.theo
        ret_data.append(px_usd)
        names.append("pxUSD")
        
        px_btc = option_obj.theo/spot
        ret_data.append(px_btc)
        names.append("pxCOIN")
        
        # all_data.at[index, "D1"]= temp_option.d1
        # all_data.at[index, "D2"]= temp_option.d2
        d1 = option_obj.d1
        ret_data.append(d1)
        names.append("D1")
        
        d2 = option_obj.d2
        ret_data.append(d2)
        names.append("D2")
        
        delta = option_obj.delta
        ret_data.append(delta)
        names.append("Delta")
        
        gamma = option_obj.gamma
        ret_data.append(gamma)
        names.append("Gamma")
        
        theta = option_obj.theta
        ret_data.append(theta)
        names.append("Theta")
        
        vega = option_obj.vega
        ret_data.append(vega)
        names.append("Vega")
        
        dollar_delta = option_obj.dollar_delta
        ret_data.append(dollar_delta)
        names.append("$Delta")
        
        dollar_gamma = option_obj.dollar_gamma
        ret_data.append(dollar_gamma)
        names.append("$Gamma")
        
        dollar_theta = option_obj.dollar_theta
        ret_data.append(dollar_theta)
        names.append("$Theta")
        
        dollar_vega = option_obj.dollar_vega
        ret_data.append(dollar_vega)
        names.append("$Vega")
    except:
        ret_data = []
        names = []
        name = option_obj.underlying_pair
        ret_data.append(name)
        names.append("Mark ID")
        
        option_type = option_obj.option_type
        ret_data.append(option_type)
        names.append("option_type")
        
        strike = option_obj.strike
        ret_data.append(strike)
        names.append("Strike")
        
        normalized_strike = option_obj.normalized_strike
        ret_data.append(normalized_strike)
        names.append("Normalized Strike")
        
        expiry = option_obj.expiry
        ret_data.append(expiry)
        names.append("Expiry")
        
        time_left = option_obj.time_left*365
        ret_data.append(time_left)
        names.append("Days to Expiry")
        
    
        
        dividend_rate = option_obj.dividend_rate
        ret_data.append(dividend_rate)
        names.append("Div Rate")
        
        iv = option_obj.vol
        ret_data.append(iv)
        names.append("IV")
    
        gamma_theta = 0
        ret_data.append(gamma_theta)
        names.append("$Gamma/$Theta")
    
    
        spot = option_obj.underlying_price
        ret_data.append(spot)
        names.append("Spot Mark")
        
        px_usd = 0
        ret_data.append(px_usd)
        names.append("pxUSD")
        
        px_btc = 0
        ret_data.append(px_btc)
        names.append("pxCOIN")
        
        # all_data.at[index, "D1"]= temp_option.d1
        # all_data.at[index, "D2"]= temp_option.d2
        d1 = 0
        ret_data.append(d1)
        names.append("D1")
        
        d2 = 0
        ret_data.append(d2)
        names.append("D2")
        
        delta = 0
        ret_data.append(delta)
        names.append("Delta")
        
        gamma = 0
        ret_data.append(gamma)
        names.append("Gamma")
        
        theta = 0
        ret_data.append(theta)
        names.append("Theta")
        
        vega = 0
        ret_data.append(vega)
        names.append("Vega")
        
        dollar_delta = 0
        ret_data.append(dollar_delta)
        names.append("$Delta")
        
        dollar_gamma = 0
        ret_data.append(dollar_gamma)
        names.append("$Gamma")
        
        dollar_theta = 0
        ret_data.append(dollar_theta)
        names.append("$Theta")
        
        dollar_vega = 0
        ret_data.append(dollar_vega)
        names.append("$Vega")
    return names, ret_data

#%% Class FwdBasis

class FwdBasis:
    def __init__(self, currency, minimum_starting=0):
        '''
        curency: str
            Choose between "BTC" or "ETH"            
        minimum_starting: float
            The minimum number of days used to fit the basis data. By default this value is set to 0 days.
        '''
        self.currency = currency.upper()        
        self.minimum_starting = minimum_starting
        # Download new data everytime the FwdBasis object is called
        self.fitted_function = self.fit()
        
    def fit(self):        
        """
        Downloads and fits the data from Deribit. 
        """
        r = requests.get(url="https://deribit.com/api/v2/public/" + \
                         "get_book_summary_by_currency?currency=" \
                        + self.currency +'&kind=option')        
        content = json.loads(r.text)

        df = pd.DataFrame(content["result"])
        # Deribit options expire at 8AM UTC so need to add 8 hour offset      
        df['expiry_timestamp'] = [(pd.to_datetime(df.instrument_name[i].split("-")[1]) + pd.Timedelta("8 hours")).timestamp() for i in range(len(df))]
        df['days_maturity'] = round((df.expiry_timestamp - df.creation_timestamp/1000)/(60*60*24), 5)
        df = df[['underlying_price', "estimated_delivery_price", 'underlying_index', 'days_maturity']]
        # Calculate difference between underlying price and spot/delivery price
        df['basis_diff'] = df.underlying_price -  df.estimated_delivery_price
        # Interpolate from minimum days to 365 days - can change this up to whichever range is required
        df['ann_basis'] = (((df.underlying_price/df.estimated_delivery_price)**(365/df.days_maturity))-1)*100        
        df = df[['days_maturity', 'ann_basis']].sort_values("days_maturity").reset_index(drop=True)
        df = df.drop_duplicates("days_maturity").reset_index(drop=True)
        df = df[df.days_maturity >= self.minimum_starting].reset_index(drop=True)                
        # Interpolate the basis
        f = interp1d(df.days_maturity, df.ann_basis, fill_value="extrapolate")
        return f
    
    # def return_vals(self, maturities, year_diff=True):        
    #     """
    #     maturities: list
    #         A list of float values indicating the days to maturity. ie: [30, 60, 90]
    #         This will return the interpolated basis values for these given maturities. 
    #     """
    #     expiry_dict = defaultdict(int)
    #     m =1
    #     if year_diff == True:
    #         m = 365
    #     expiry_dict= dict(zip(maturities, self.fitted_function(np.array(maturities)*m)/100))
    #     return expiry_dict

#%% price Options/ initialize/ fix

def price_options(_df,now, new_spot=False):
    _df['TradePX$']= np.where(_df['Asset']!= 'USD',_df['Px']*_df['Spot Mark'], _df['Px'])
    
    df = _df[_df['Option Type']== 'C'].copy()

    temp_call = Option(df['Mark ID'], 'call', df['Strike'], df['Expiry'], df['day_diff']/365, size = df['Pos'], 
                       dividend_rate=df['Div Rate'], volatility=df['IV'], underlying_price = df['Spot Mark'])
    column_names , ret_data = getOptionData(temp_call)
    sub_call = pd.DataFrame(data=dict(zip(column_names,ret_data)))
    df = df.iloc[:,df.columns.isin(sub_call.columns)==False]
    sub_call = df.join(sub_call)

    df = _df[_df['Option Type']== 'P'].copy()
    temp_put = Option(df['Mark ID'], 'put', df['Strike'], df['Expiry'], df['day_diff']/365, size = df['Pos'], 
                       dividend_rate=df['Div Rate'], volatility=df['IV'], underlying_price = df['Spot Mark'])
    column_names , ret_data = getOptionData(temp_put)
    sub_put = pd.DataFrame(data=dict(zip(column_names,ret_data)))
    df = df.iloc[:,df.columns.isin(sub_put.columns)==False]
    sub_put = df.join(sub_put)

    df = pd.concat([sub_call, sub_put], ignore_index=True)
    df['Unit Delta'] = df['Delta']*df['Pos']
    df['PNL'] = (df['pxUSD']-df['TradePX$'])*df['Pos']
    return df