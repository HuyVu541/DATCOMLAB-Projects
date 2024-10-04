import vnstock as vns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from add_indicators import add_mfi, add_rsi, add_macd, add_bollinger_bands, psar, add_psar, psar_trend, calculate_indicators
from cosine import cosine
from itertools import combinations
from Trend_Detection import add_trend
from add_signals import add_RSI_signal, add_MFI_signal, add_MACD_signal, add_BOLL_signal, add_PSAR_signal, calculate_all_signals


strategies_num = 2

df = vns.stock_historical_data(symbol = ticker, start_date="2014-07-01", 
                            end_date='2024-07-01', resolution='1D', type='stock')

df_len = len(df)
train_length = 0.8
start_global = round(df_len * train_length)
df_train = df[:start_global]
df_test = df[start_global:]

signal_dict = {'MFI': 'MFI_signal', 'MACD': 'MACD_signal', 'BOLL': 'BOLL_signal', 'RSI': 'RSI_signal', 'PSAR': 'PSAR_signal'}

def calculate_signal(data, index, signal_list):
    temp = 0
    for signal in signal_list:
        temp += data.loc[index, signal_dict[signal]]
    return temp

def advance_day(day, money, debt, period = 20):
   if day == period:
      day = 0
      money += 1000000
      debt += 1000000
   return money, debt

class Results:
    def __init__(self):
        self.profit = 0
        self.buy_points = []
        self.sell_points = []
        self.assets = []
        self.debt = 0

def train_helper(data, upper_limit, lower_limit, decay, strategies, start = 0, budget = 100000000, current_date = 0, advance = False):
    
    # Creating intial parameters
    current_date = start
    results = Results()
    stocks_holding = 0
    signal = 0
    # If using advance_day()
    debt = 0
    days_passed = 0
    
    while current_date < len(data):
        
        # Decaying the signal
        if signal >= 0:
            signal = max(0, signal - decay)
        else:
            signal = min(0, signal + decay)

        current_price = data.loc[current_date, 'close']
        signal = signal + calculate_signal(data, current_date, strategies)

        if signal >= upper_limit:
            stocks_to_buy = (budget * 3/4) // current_price  #Spends 3/4 the budget to buy
            if stocks_to_buy > 0:
                budget -= stocks_to_buy * current_price
                stocks_holding += stocks_to_buy
                results.buy_points.append(current_date)
        
        elif signal <= lower_limit:
            stocks_sold = stocks_holding
            if stocks_sold > 0:
                budget += stocks_sold * current_price
                stocks_holding -= stocks_sold
                results.sell_points.append(current_date)

        results.assets.append(budget + stocks_holding * current_price)
        
        if advance:
            days_passed, budget, debt = advance_day(days_passed, budget, debt)
        
        current_date += 1
        # Moving forward one day

    results.assets.append(budget + stocks_holding * current_price)
    
    results.profit = (results.assets[-1] - results.assets[0]) / results.assets[0] * 100

    return results

signals = ['MFI', 'MACD', 'BOLL', 'RSI', 'PSAR']
signal_list = []
for i in range(1, len(signals)):
    signal_list += combinations(signals, i)
signal_list = [list(i) for i in signal_list]

def train_parameter(data_train):
    profit_and_loss = []

    # train_results_dict = {}
    
    signals = ['MFI', 'MACD', 'BOLL', 'RSI', 'PSAR']
    signal_list = []
    for i in range(1, len(signals) + 1):
        signal_list += combinations(signals, i)
    signal_list = [list(i) for i in signal_list]

    for s in signal_list:
        print(f'Current strategy: {[s]}')
        # For testing parameters
        for i in range(1,41):
            # for j in range(0, min(i, len(s)*5 + 1)):
            for j in range(0, i):
                temp = train_helper(data_train, i/10, -i/10, j/10, s).profit

                profit_and_loss.append([s, i/10, -i/10, j/10, temp])
        
        # train_results = sorted(profit_and_loss, key= lambda x: (x[-1], -x[1]), reverse = True)
        train_results = sorted(profit_and_loss, key= lambda x: (x[-1]), reverse = True)

        # str_strategy = '+'.join(s)
        # train_results_dict[str_strategy] = train_results
        boll_results = []
        rsi_results = []

        for i in range(len(train_results)):
            if 'BOLL' in train_results[i][0]:
                boll_results.append(train_results[i][:4])
            elif 'RSI' in train_results[i][0]:
                rsi_results.append(train_results[i][:4])
            
    return boll_results, rsi_results

boll_strategies, rsi_strategies = train_parameter(df_train)

def choose_parameters(trend):
    if trend == 'Up':
        return boll_strategies[:strategies_num]
    else:
        return rsi_strategies[:strategies_num]
# Tra ve 4 cai profit cao nhat sao cho dung trend    

def backtest(data, total_budget = 10000000, current_date = 0, advance = False):
    
    # Creating intial parameters
    current_date = 0
    current_trend = data.loc[current_date, 'trend']
    parameters = choose_parameters(current_trend)
    
    total_budget = 0
    total_stocks_holding = 0

    for i in parameters:
        i.append(0)
        i.append(total_budget/strategies_num)
    # Parameter: [strategy, upper_limit, lower_limit, decay, individual, budget, individual stocks_holding]

    results = Results()
    
    # If using advance_day()
    total_debt = 0
    days_passed = 0
    
    while current_date < len(data):
        
        # Decaying the signal
        if signal >= 0:
            signal = max(0, signal - decay)
        else:
            signal = min(0, signal + decay)

        # Checking for changes in trend
        if data.loc[current_date, 'trend'] != current_trend:
            # When trend changes
            total_budget = sum([i[5] for i in parameters])
            total_stocks_holding = sum([i[6] for i in parameters])
            current_trend = data.loc[current_date, 'trend']
            parameters = choose_parameters(data, current_date, current_trend)
            for i in parameters:
                i.append(0)
                i.append(total_budget/4)
        else:
            current_trend = data.loc[current_date, 'trend']

        current_price = data.loc[current_date, 'close']

        for parameter_set in parameters:
            strategies, upper_limit, lower_limit, decay, signal, individual_budget, individual_stock_holding = parameter_set
        
            signal = signal + calculate_signal(data, current_date, strategies)

            if signal >= upper_limit:
                stocks_to_buy = (individual_budget * 3/4) // current_price  #Spends 3/4 the budget to buy
                if stocks_to_buy > 0:
                    individual_budget -= stocks_to_buy * current_price
                    individual_stock_holding += stocks_to_buy
                    results.buy_points.append(current_date)
            
            elif signal <= lower_limit:
                stocks_sold = individual_stock_holding
                if stocks_sold > 0:
                    budget += stocks_sold * current_price
                    individual_stock_holding -= stocks_sold
                    results.sell_points.append(current_date)

            parameter_set = strategies, upper_limit, lower_limit, decay, signal, individual_budget, individual_stock_holding

        total_budget = sum([i[5] for i in parameters])
        total_stocks_holding = sum([i[6] for i in parameters])
        
        results.assets.append(total_budget + total_stocks_holding * current_price)
        
        if advance:
            days_passed += 1
            for i in parameters:
                parameters[5], total_debt = advance_day(budget, total_debt)
        
        current_date += 1
        # Moving forward one day

    results.assets.append(budget + total_stocks_holding * current_price)
    results.debt = total_debt
    results.profit = (results.assets[-1] - results.assets[0]) / results.assets[0] * 100
    return results

test_results = backtest(df_test)

f = open(f'Pattern Matching Results - {ticker}.txt', 'a')
f.write(f'{day_start} to {day_end}: Profit: {test_results.profit}% >< \
    {(df_test['close'].iloc[-1] - df_test['close'].iloc[0])/df_test['close'].iloc[0] * 100}% \n')
f.close()