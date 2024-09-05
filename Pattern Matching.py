import vnstock as vns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from MFI import add_mfi
from RSI import add_rsi
from MACD import add_macd
from cosine import cosine
from bollinger_bands import add_bollinger_bands

df = vns.stock_historical_data(symbol = ticker, start_date="2014-01-01", 
                            end_date='2024-06-01', resolution='1D', type='stock')


def calculate_indicators(data):
    add_mfi(data)
    
    data = data[['time', 'close', 'MFI']]

    # data.set_index('time', inplace = True)

    data = data.dropna()

    data['sma10'] = data['close'].rolling(window = 10, min_periods = 1).mean()
    
    data['sma50'] = data['close'].rolling(window = 50, min_periods = 1).mean()
    
    data['ema'] = data['close'].ewm(span = 10).mean()
    
    add_bollinger_bands(data)

    add_rsi(data)

    add_macd(data)

    data.dropna(inplace = True)
    
    data.reset_index(inplace = True, drop = True)

    return data

df = calculate_indicators(df)

df_len = len(df)
train_length = 0.8
start_global = round(df_len * train_length)
df_train = df[:start_global]
df_test = df[start_global:]

def RSI_signal(data, index):
    if data['RSI'].iloc[index] <= 70 and data['RSI'].iloc[index - 1] >= 70:
        return -1
    elif data['RSI'].iloc[index] >= 30 and data['RSI'].iloc[index - 1] <= 30:
        return 1
    else:
        return 0

def MFI_signal(data, index):
    if data['MFI'].iloc[index] <= 80 and data['MFI'].iloc[index - 1] >= 80:
        return -1
    elif data['MFI'].iloc[index] >= 20 and data['MFI'].iloc[index - 1] <= 20:
        return 1
    else:
        return 0

def MACD_signal(data, index):
    if data['MACD'].iloc[index] >= data['Signal_Line'].iloc[index] and \
        data['MACD'].iloc[index - 1] <= data['Signal_Line'].iloc[index - 1]:
        return -1
    elif data['MACD'].iloc[index] <= data['Signal_Line'].iloc[index] and \
        data['MACD'].iloc[index - 1] >= data['Signal_Line'].iloc[index - 1]:
        return 1    
    else:
        return 0

def bollinger_bands_signal(data, index):
    if data['close'].iloc[index] <= data['LB'].iloc[index]:\
        #and data['close'].iloc[index - 1] >= data['LB'].iloc[index - 1]\
        return 1
    elif data['close'].iloc[index] >= data['UB'].iloc[index] and \
        data['close'].iloc[index - 1] <= data['UB'].iloc[index - 1]:
        return -1
    else:
        return 0

def pattern_matching(start, window, indicator):
    df_temp = df_test[indicator].iloc[start-window + 1:start + 1]
    df_window = df_train[indicator].copy()  
    df_window -= df_window.iloc[0]
    cosines = []
    for i in range(len(df_window)-window):
        df_compare = df_window.iloc[i:i+window].copy()
        df_compare -= df_compare.iloc[0]
        cosines.append(cosine(df_temp, df_compare, window))
    return cosines.index(max(cosines))

def matching_signal(data, start):
    pattern_start = pattern_matching(start, window, indicator)
    predicted_earnings = data[indicator].iloc[pattern_start + window*2 - 1] - data[indicator].iloc[pattern_start + window - 1]
    
    if predicted_earnings > 0:
        return 0.25
    else:
        return -0.25

signal_dict = {'MFI': MFI_signal, 'MACD': MACD_signal, 'bollinger_bands': bollinger_bands_signal, 'RSI': RSI_signal, 'matching': matching_signal}

def calculate_signal(data, index, signal_list):
    temp = 0
    for signal in signal_list:
        temp += signal_dict[signal](data, index)
    return temp

# For testing parameters

profit_and_loss = []

for i in range(1,51):
    for j in range(1,i):
        indicator = 'close'
        start = 50  # Is the current day, must be equal to or higher than window
        window = 50 # Looks into the past
        df_matching = df_train.copy()
        budget = 10000000  # How much money to start with, doesnt really matter
        total_assets = [10000000]
        stocks_holding = 0
        buy_points = []
        sell_points = []
        signal = 0
        signal_list = ['MFI', 'MACD', 'bollinger_bands', 'RSI']

        upper_signal = i/10
        lower_signal = -i/10
        decay = j/10
        
        for k in range(len(df_matching) - start):
            current_price = df_matching['close'].iloc[start]
            signal = signal + calculate_signal(df_matching, start, signal_list)
            if signal >= upper_signal:
                stocks_to_buy = (budget * 3/4) // current_price  #Spends half the budget to buy
                
                if stocks_to_buy > 0:
                    budget -= stocks_to_buy * current_price
                    stocks_holding += stocks_to_buy
                    buy_points.append(start)
                    
            elif (signal <= lower_signal) and (stocks_holding > 0):
                sell_points.append(start)
                stocks_sold = stocks_holding
                budget += stocks_sold * current_price
                stocks_holding -= stocks_sold 

            total_assets.append(budget + stocks_holding*current_price)
            start += 1
            
            if signal >= 0:
                signal = max(0, signal - decay)
            else:
                signal = min(0, signal + decay)
            # Decaying the signal
        current_price = df_matching['close'].iloc[-1]
        total_assets.append(budget + stocks_holding * current_price)

        profit_and_loss.append([((total_assets[-1] - total_assets[0])/total_assets[0]) * 100, i/10, j/10])

train_results = sorted(profit_and_loss, key= lambda x: x[0], reverse = True)

limit = train_results[0][1]
trained_decay = train_results[0][2]

start = 50  # At which point to start investing
df_matching = df_test.copy()
budget = 1000000  # How much money to start with, doesnt really matter
total_assets = [1000000]
window = 50
stocks_holding = 0
buy_points = []
sell_points = []
signal = 0
signal_list = ['MFI', 'MACD', 'bollinger_bands', 'RSI']
upper_signal = limit
lower_signal = -limit
decay = trained_decay
for k in range(len(df_matching) - start):
   current_price = df_matching['close'].iloc[start]
   signal = signal + calculate_signal(df_matching, start, signal_list)
   if signal >= upper_signal:
      stocks_to_buy = (budget * 3/4) // current_price  #Spends half the budget to buy
      
      if stocks_to_buy > 0:
         budget -= stocks_to_buy * current_price
         stocks_holding += stocks_to_buy
         buy_points.append(start)
         
   elif (signal <= lower_signal) and (stocks_holding > 0):
      
      sell_points.append(start)
      stocks_sold = stocks_holding
      budget += stocks_sold * current_price
      stocks_holding -= stocks_sold

   total_assets.append(budget + stocks_holding*current_price)
   start += 1

   if signal >= 0:
      signal = max(0, signal - decay)
   else:
      signal = min(0, signal + decay)
   # Decaying the signal


current_price = df_matching['close'].iloc[-1]
total_assets.append(budget + stocks_holding * current_price)

print(f'Profit/loss is: {(total_assets[-1] - total_assets[0])/(total_assets[0]) * 100}%')

print(f'Stock grew by: {(df_test['close'].iloc[-1] - df_test['close'].iloc[0])/df_test['close'].iloc[0] * 100}%')
print(f'From {df_test['time'].iloc[0]} to {df_test['time'].iloc[-1]}')

f = open('Pattern Matching Results.txt')
f.write(f'{ticker}: {(total_assets[-1] - total_assets[0])/(total_assets[0]) * 100}% >< \
    {(df_test['close'].iloc[-1] - df_test['close'].iloc[0])/df_test['close'].iloc[0] * 100}% \n \
    From {df_test['time'].iloc[0]} to {df_test['time'].iloc[-1]}\n')
f.close()