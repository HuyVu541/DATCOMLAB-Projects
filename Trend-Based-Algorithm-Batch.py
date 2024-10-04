import vnstock as vns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt

starts = pd.date_range('2014-01-01', '2024-07-01', freq = '6MS')
ends = pd.date_range('2019-01-01', '2024-07-01', freq = '6MS')

tickers = ['ACB', 'BVH', 'CTG', 'FPT', 'GAS', 'HPG', 'MBB', 'MSN', 'SHB', 'SSI', 'STB', 'VCB', 'VIC', 'VNM']
dfs = {}
for day in range(len(ends)):
    day_start = pd.to_datetime(starts[day]).strftime('%Y-%m-%d')
    day_end = pd.to_datetime(ends[day]).strftime('%Y-%m-%d')
    for ticker in tickers:
        df = vns.stock_historical_data(symbol = ticker, start_date=day_start, 
                            end_date=day_end, resolution='1D', type='stock')
        dfs[(day_start, day_end, ticker)] = df


for day in range(len(ends)):
    day_start = pd.to_datetime(starts[day]).strftime('%Y-%m-%d')
    day_end = pd.to_datetime(ends[day]).strftime('%Y-%m-%d')
    print(f'From {day_start} to {day_end}:')
    for ticker in tickers:
        print(f'Current ticker: {ticker}\tStrategy: {s[0]}')
        df = dfs[(day_start, day_end, ticker)]
        exec(open("Trading Strategy 3.py").read())