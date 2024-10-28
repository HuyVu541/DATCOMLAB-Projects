import vnstock as vns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import copy
from add_indicators import calculate_indicators
from itertools import combinations
from add_signals import calculate_all_signals


df = vns.stock_historical_data(symbol = ticker, start_date=day_start, 
                            end_date=day_end, resolution='1D', type='stock')

df = calculate_indicators(df)
df = calculate_all_signals(df)

df_len = len(df)
train_length = 0.6
start_global = round(df_len * train_length)
df_train = df[:start_global]
df_test = df[start_global:]

# signal_dict = {'MFI': 'MFI_signal', 'MACD': 'MACD_signal', 'BOLL': 'BOLL_signal', 'RSI': 'RSI_signal', 'PSAR': 'PSAR_signal'}
signal_dict = {'MFI': 'MFI_signal', 'MACD': 'MACD_signal', 'BOLL': 'BOLL_signal', 'RSI': 'RSI_signal'}

def calculate_signal(data, index, signal_list):
    temp = 0
    for signal in signal_list:
        temp += data.loc[index, signal_dict[signal]]
    return temp

def advance_day(day, bonus, debt, period = 20):
    if day % period == 0:
        bonus += 1000000
        debt += 1000000
    else:
        bonus = 0
    return bonus, debt


class Results:
    def __init__(self):
        self.profit = 0
        self.buy_points = []
        self.sell_points = []
        self.assets = []
        self.debt = 0

def train_helper(data, upper_limit, lower_limit, decay, strategies, budget = 10000000, current_date = 0, advance = False):
     
    # Creating intial parameters
    results = Results()
    stocks_holding = 0
    signal = 0

    # If using advance_day()
    debt = 0
    days_passed = 0
    bonus = 0
    
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
            days_passed += 1
            bonus, debt = advance_day(days_passed, bonus, debt)
            budget += bonus

        # Moving forward one day
        current_date += 1

    results.assets.append(budget + stocks_holding * current_price)
    
    results.profit = (results.assets[-1] - results.assets[0]) / results.assets[0] * 100

    return results


def train_parameter(data_train, rank = 1):
    signals = list(signal_dict.keys())
    signal_list = []
    for i in range(1, len(signals) + 1):
        signal_list += combinations(signals, i)
    signal_list = [list(i) for i in signal_list]

    boll_results = []
    rsi_results = []
    other_results = []

    for s in signal_list:
        print(f'Current strategy: {[s]}')
        profit_and_loss = []

        # For testing parameters
        for i in range(1,41):
            # for j in range(0, min(i, len(s)*5 + 1)):
            for j in range(0, i):
                temp = train_helper(data_train, i/10, -i/10, j/10, s, advance=True).profit
                profit_and_loss.append([s, i/10, -i/10, j/10, temp])
        
        if 'BOLL' in s:
            boll_results += profit_and_loss
        if 'RSI' in s:
            rsi_results += profit_and_loss
        if 'BOLL' not in s and 'RSI' not in s:
            other_results += profit_and_loss

    boll_results = copy.deepcopy(sorted(boll_results, key= lambda x: (x[-1]), reverse = True))
    rsi_results = copy.deepcopy(sorted(rsi_results, key= lambda x: (x[-1]), reverse = True))
    other_results = copy.deepcopy(sorted(other_results, key= lambda x: (x[-1]), reverse = True))

    prev_boll = None
    boll_rank = 0
    prev_rsi = None
    rsi_rank = 0
    prev_other = None
    other_rank = 0

    for i in range(len(boll_results)):
        if boll_results[i][-1] != prev_boll:
            prev_boll = boll_results[i][-1]
            boll_rank += 1
        boll_results[i].append(boll_rank)

    for i in range(len(rsi_results)):
        if rsi_results[i][-1] != prev_rsi:
            prev_rsi = rsi_results[i][-1]
            rsi_rank += 1
        rsi_results[i].append(rsi_rank)

    for i in range(len(other_results)):
        if other_results[i][-1] != prev_other:
            prev_other = other_results[i][-1]
            other_rank += 1
        other_results[i].append(other_rank)

    boll_valid = []
    rsi_valid = []
    other_valid = []

    for boll in boll_results:
        if boll[-1] <= rank:
            boll_valid.append(boll[:4])
        else:
            break
    
    for rsi in rsi_results:
        if rsi[-1] <= rank:
            rsi_valid.append(rsi[:4])
        else:
            break

    for other in other_results:
        if other[-1] <= rank:
            other_valid.append(other[:4])
        else:
            break

    return boll_valid, rsi_valid, other_valid


boll_strategies, rsi_strategies, other_strategies = train_parameter(df_train)

def choose_parameters(trend):
    if trend == 'Up':
        return copy.deepcopy(boll_strategies)
    else:
        return copy.deepcopy(rsi_strategies)

    
def backtest(data, total_budget = 10000000, current_date = 0, advance = False):

    # Creating intial parameters
    current_trend = data.loc[current_date, 'trend']
    parameters = copy.deepcopy(choose_parameters(current_trend))
    
    total_stocks_holding = 0

    for i in range(len(parameters)):
        parameters[i].append(0)
        parameters[i].append(0)
        parameters[i].append(total_budget // (len(parameters) - i))
        total_budget -= total_budget // (len(parameters) - i)
    # Parameter: [strategy, upper_limit, lower_limit, decay, signal, individual stocks_holding, individual_budget]

    results = Results()
    
    # If using advance_day()
    total_debt = 0
    days_passed = 0
    bonus = 0
        
    while current_date < len(data):
        # Checking for changes in trend
        if data.loc[current_date, 'trend'] != current_trend:

            total_budget = sum([i[-1] for i in parameters]) + bonus
            total_stocks_holding = sum([i[-2] for i in parameters])
            previous_signal = np.mean([i[4] for i in parameters]) 
            current_trend = data.loc[current_date, 'trend']

            parameters = copy.deepcopy(choose_parameters(current_trend))
            for i in range(len(parameters)):
                parameters[i].append(previous_signal)
                parameters[i].append(total_stocks_holding // (len(parameters) - i))
                total_stocks_holding -= total_stocks_holding // (len(parameters) - i)
                parameters[i].append(total_budget // (len(parameters) - i))
                total_budget -= total_budget // (len(parameters) - i)
        
        else:
            for i in range(len(parameters)):
                parameters[i][-1] = parameters[i][-1] + bonus // (len(parameters) - i)
                bonus -= bonus // (len(parameters) - i)

        current_price = data.loc[current_date, 'close']

        for i in range(len(parameters)):
            strategies, upper_limit, lower_limit, decay, signal, individual_stock_holding, individual_budget = parameters[i]
            
             # Decaying the signal
            if signal >= 0:
                signal = max(0, signal - decay)
            else:
                signal = min(0, signal + decay)

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
                    individual_budget += stocks_sold * current_price
                    individual_stock_holding -= stocks_sold
                    results.sell_points.append(current_date)

            parameters[i] = [strategies, upper_limit, lower_limit, decay, signal, individual_stock_holding, individual_budget]
        
        total_budget = sum([i[-1] for i in parameters])
        total_stocks_holding = sum([i[-2] for i in parameters])
        results.assets.append(total_budget + total_stocks_holding * current_price)

        if advance:
            days_passed += 1
            bonus, total_debt = advance_day(days_passed, bonus, total_debt)
            
        # Moving forward one day
        current_date += 1

    results.assets.append(total_budget + total_stocks_holding * current_price)
    results.debt = total_debt
    results.profit = (results.assets[-1] - results.assets[0] - total_debt) / results.assets[0] * 100
    return results 

df_test = df[start_global:].reset_index(drop = True) 
test_results = backtest(df_test, advance = True) 

f = open(f'Pattern Matching Results - {ticker}.txt', 'a')
f.write(f"{df_test['time'].iloc[0]} to {df_test['time'].iloc[-1]}: Profit: {test_results.profit}% >< \
    {(df_test['close'].iloc[-1] - df_test['close'].iloc[0])/df_test['close'].iloc[0] * 100}% {boll_strategies} {rsi_strategies}\n")
f.close()
    