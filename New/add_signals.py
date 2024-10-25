import numpy as np

def add_RSI_signal(data):
    data['RSI_signal'] = 0
    data.loc[data.iloc[np.where((data['RSI'] >= 70) & (data['RSI'].shift(1) <= 70))].index, ['RSI_signal']] = -1
    data.loc[data.iloc[np.where((data['RSI'] <= 30) & (data['RSI'].shift(1) >= 30))].index, ['RSI_signal']] = 1

def add_MFI_signal(data):
    data['MFI_signal'] = 0 
    data.loc[data.iloc[np.where((data['MFI'] >= 80) & (data['MFI'].shift(1) <= 80))].index, ['MFI']] = -1
    data.loc[data.iloc[np.where((data['MFI'] <= 20) & (data['MFI'].shift(1) >= 20))].index, ['MFI']] = 1

def add_MACD_signal(data):
    data['MACD_signal'] = 0
    data.loc[data.iloc[np.where((data['MACD'] >= data['Signal_Line']) & \
                                (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)))].index, ['MACD_signal']] = 1
    data.loc[data.iloc[np.where((data['MACD'] <= data['Signal_Line']) & \
                                (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)))].index, ['MACD_signal']] = -1

def add_BOLL_signal(data):
    data['BOLL_signal'] = 0
    data.loc[data.iloc[np.where((data['close'] <= data['LB']) & \
                                (data['close'].shift(1) >= data['LB'].shift(1)))].index, ['BOLL_signal']] = 1
    data.loc[data.iloc[np.where((data['close'] >= data['UB']) & \
                                (data['close'].shift(1) <= data['UB'].shift(1)))].index, ['BOLL_signal']] = 1
    data.loc[data.iloc[np.where((data['close'] >= data['LB']) & \
                                (data['close'].shift(1) <= data['LB'].shift(1)))].index, ['BOLL_signal']] = -1
    data.loc[data.iloc[np.where((data['close'] <= data['UB']) & \
                                (data['close'].shift(1) >= data['UB'].shift(1)))].index, ['BOLL_signal']] = -1
    
def add_PSAR_signal(data):
    data['PSAR_signal'] = 0
    data.loc[data.iloc[np.where((data['PSAR'] > data['close']) & \
                                (data.shift(1)['PSAR'] < data.shift(1)['close']))].index, ['PSAR_signal']] = -1
    data.loc[data.iloc[np.where((data['PSAR'] < data['close']) & \
                                (data.shift(1)['PSAR'] > data.shift(1)['close']))].index, ['PSAR_signal']] = 1
                                
 
def calculate_all_signals(data):
    add_RSI_signal(data)
    add_MFI_signal(data)
    add_MACD_signal(data)
    add_BOLL_signal(data)
    add_PSAR_signal(data)
    return data