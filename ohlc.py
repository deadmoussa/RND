import asyncio
import aiohttp
import logging
import json
import datetime as dt
import pandas as pd
import requests
import time

# see documentation: https://docs.amberdata.io/reference/reference-getting-started
class AmberData:
    URL = 'https://web3api.io/api'
    WSS = 'wss://ws.web3api.io'
    QUOTES = ['USD', 'USDC', 'USDT']
    INTERVALS = {
        'minutes': dt.timedelta(days=1),
        'hours': dt.timedelta(days=30),
        'days': dt.timedelta(days=365),
        'weeks': dt.timedelta(days=(365 * 10))
    }
    
    def __init__(self):
        self.key = 'UAKeb75c7488ce4acf005daec2655ab6ebb'

    def header(self):
        headers = {
            'x-api-key': self.key
        }
        return headers

    def _request(self, method, path, params=None):
        if params is None:
            params = {}
        try:
            req = requests.request(method=method, url=self.URL + path, params=params, headers=self.header())
            response = req.json()
            return response.get('payload', dict())
        except Exception as e:
            logging.warning('amberdata exception %s %s', e, req)
            print(path, params, 'exception', e, 'waiting 30 seconds...')
            time.sleep(10)
            return self._request(method, path, params)

    @staticmethod
    def flatten_markets(pair_dict):
        flattened = list()
        for pair, pair_data in pair_dict.items():
            for exchange, exchange_data in pair_data.items():
                flattened_data = dict()
                flattened_data['pair'] = pair
                flattened_data['exchange'] = exchange
                for metric, metric_dates in exchange_data.items():
                    for date in metric_dates:
                        if not metric_dates[date]: continue
                        metric_dates[date] = dt.datetime.fromtimestamp(metric_dates[date] / 1e3)
                    flattened_data[metric] = metric_dates
                flattened.append(flattened_data)
        return flattened

    def get_spot_markets(self, pair=None, exchange=None, time_format='ms', include_dates=True):
        params = {
            'includeDates': str(include_dates).lower(),
            'timeFormat': time_format
        }
        if pair:
            params['pair'] = pair.lower()
        if exchange:
            params['exchange'] = exchange.lower()
        data = self._request('GET', '/v2/market/pairs', params=params)
        return self.flatten_markets(data)

    def get_base_spot_markets(self, base=None, exchange=None, time_format='ms'):
        markets = []
        for quote in self.QUOTES:
            quote_markets = self.get_spot_markets(pair=f"{base}_{quote}", exchange=exchange, time_format=time_format)
            markets += quote_markets
        return markets

    def get_oldest_spot_market(self, base, exchange=None):
        markets = self.get_base_spot_markets(base=base, exchange=exchange)
        if not markets:
            markets = self.get_base_spot_markets(base=base, exchange=None)
        if not markets:
            return dict()
        now = dt.datetime.utcnow()
        oldest = None
        for market in markets:
            if 'ohlc' not in market: continue
            if market['ohlc'].get('startDate') is None or market['ohlc']['endDate'] is None: continue
            if now - market['ohlc']['endDate'] > dt.timedelta(hours=1): continue
            if oldest is None or market['ohlc']['startDate'] < oldest['ohlc']['startDate']: oldest = market
        return oldest

    

    @staticmethod
    def flatten_spot_candles(pair, candles):
        if not candles.get('data'): return dict()
        candle_list = list()
        columns = candles['metadata']['columns']
        for exchange, exchange_candles in candles['data'].items():
            for exchange_candle in exchange_candles:
                candle_data = dict(zip(columns, exchange_candle))
                candle_data['pair'] = pair
                candle_data['exchange'] = exchange
                candle_data['dt'] = dt.datetime.fromtimestamp(candle_data['timestamp'] / 1e3)
                candle_list.append(candle_data)
        return candle_list

    def get_interval_spot_candles(self, pair, exchange, interval, start, end=None, time_format='ms'):
        if end is None:
            end = start + self.INTERVALS[interval]
        params = {
            'exchange': exchange,
            'timeFormat': time_format,
            'startDate': start.isoformat(),
            'endDate': min(start + self.INTERVALS[interval], end).isoformat(),
            'timeInterval': interval
        }
        candles = self._request('GET', f'/v2/market/spot/ohlcv/{pair}/historical', params=params)
        return self.flatten_spot_candles(pair, candles)

    def get_spot_candles(self, pair, exchange, interval, start=None, end=None, time_format='ms'):
        end_time = dt.datetime.utcnow() if end is None else end
        start_time = end_time - self.INTERVALS[interval] if start is None else start
        # print(start_time, end_time)
        candles = []
        while start_time <= end_time:
            interval_candles = self.get_interval_spot_candles(pair, exchange, interval, start_time, end_time, time_format)
            candles += interval_candles
            start_time += self.INTERVALS[interval]
        return candles

    @staticmethod
    def flatten_futures_candles(instrument, candles):
        if not candles.get('data'): return dict()
        candle_list = list()
        # print(candles)
        # columns = candles['metadata']['columns']
        for candle_data in candles['data']:
            # for exchange_candle in exchange_candles:
            # candle_data = dict(zip(columns, exchange_candle))
            candle_data['instrument'] = instrument
            # candle_data['exchange'] = exchange
            candle_data['dt'] = dt.datetime.fromtimestamp(candle_data['timestamp'] / 1e3)
            candle_list.append(candle_data)
        return candle_list

    def get_interval_futures_candles(self, instrument, exchange, interval, start, time_format='ms'):
        params = {
            'exchange': exchange,
            'timeFormat': time_format,
            'startDate': start.isoformat(),
            'endDate': (start + self.INTERVALS[interval]).isoformat(),
            'timeInterval': interval
        }
        candles = self._request('GET', f'/v2/market/futures/ohlcv/{instrument}/historical', params=params)
        return self.flatten_futures_candles(instrument, candles)

    def get_futures_candles(self, instrument, exchange, interval, start=None, end=None, time_format='ms'):
        end_time = dt.datetime.utcnow() if end is None else end
        start_time = end_time - self.INTERVALS[interval] if start is None else start

        candles = []
        while start_time <= end_time:
            interval_candles = self.get_interval_futures_candles(instrument, exchange, interval, start_time, time_format)
            candles += interval_candles
            start_time += self.INTERVALS[interval]
        return candles

    @staticmethod
    def flatten_spot_trades(pair, trades):
        if not trades.get('data'): return dict()
        trade_list = list()
        columns = trades['metadata']['columns']
        for trade in trades['data']:
            trade_data = dict(zip(columns, trade))
            trade_data['pair'] = pair
            trade_data['dt'] = dt.datetime.fromtimestamp(trade_data['timestamp'] / 1e3)
            trade_list.append(trade_data)
        return trade_list

    def get_interval_spot_trades(self, pair, exchange, interval, start, time_format='ms', flatten=True):
        if isinstance(interval, str):
            interval_delta = self.INTERVALS[interval]
        else:
            interval_delta = dt.timedelta(minutes=interval)

    def get_spot_price(self, pair, exchange=None):
        params = {}
        if exchange is not None:
            params['exchange'] = exchange
        spot = self._request('GET', f'/v2/market/spot/prices/pairs/{pair}/latest/', params=params)
        spot['dt'] = dt.datetime.fromtimestamp(spot['timestamp'] / 1e3)
        for field in ['price', 'volume']:
            try:
                spot[field] = float(spot[field])
            except Exception as e:
                print(e, type(e), spot)
        return spot

    def get_interval_funding_rates(self, instrument, exchange, interval, start, time_format='ms'):
        params = {
            'exchange': exchange,
            'timeFormat': time_format,
            'startDate': start.isoformat(),
            'endDate': (start + self.INTERVALS[interval]).isoformat(),
            'timeInterval': interval
        }
        data = self._request('GET', f'/v2/market/futures/funding-rates/{instrument}/historical', params=params)
        funding_rates = data['data']
        for rate in funding_rates:
            rate['dt'] = dt.datetime.fromtimestamp(rate['timestamp'] / 1e3)
            rate['instrument'] = instrument
        return funding_rates

    def get_spot_trades(self, pair, exchange, interval, start=None, end=None, time_format='ms'):
        end_time = dt.datetime.utcnow() if end is None else end
        start_time = end_time - self.INTERVALS[interval] if start is None else start

        trades = []
        while start_time <= end_time:
            interval_trades = self.get_interval_spot_trades(pair, exchange, interval, start_time, time_format)
            trades += interval_trades
            start_time += self.INTERVALS[interval]
        return trades

    def get_funding_rates(self, instrument, exchange, interval='hours', start=None, end=None, time_format='ms'):
        end_time = dt.datetime.utcnow() if end is None else end
        start_time = end_time - self.INTERVALS[interval] if start is None else start

        funding_rates = []
        while start_time <= end_time:
            interval_funding_rates = self.get_interval_funding_rates(instrument, exchange, interval, start_time, time_format)
            funding_rates += interval_funding_rates
            start_time += self.INTERVALS[interval]
        return funding_rates

    def get_latest_funding_rates(self, exchange, instrument=None, time_format='ms'):
        params = {
            'timeFormat': time_format
        }
        if instrument is not None:
            params['instrument'] = instrument
        funding_rates = self._request('GET', f'/v2/market/futures/funding-rates/exchange/{exchange}/latest', params=params)
        for rate in funding_rates:
            rate['dt'] = dt.datetime.fromtimestamp(rate['timestamp'] / 1e3)
            rate['exchange'] = exchange
        return funding_rates



