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

#Strong weak trend
def add_ADX_signal(data):
    data['ADX_signal'] = 0
    data.loc[(data['ADX'] < 25), 'ADX_signal'] = -1 #Weak trend
    data.loc[(data['ADX'] >= 25) & (data['ADX'] < 50), 'ADX_signal'] = 1 #Strong trend
    data.loc[(data['ADX'] >= 50) & (data['ADX'] < 75), 'ADX_signal'] = 2 #Very strong trend
    data.loc[(data['ADX'] >= 75), 'ADX_signal'] = 3 #Extremely strong trend

def add_SO_signal(data):
    data['SO_signal'] = 0
    data.loc[(data['%K'] <= 20) & (data['%K'] >= data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)), 'SO_signal'] = 1
    data.loc[(data['%K'] >= 80) & (data['%K'] <= data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)), 'SO_signal'] = -1

def add_WILL_signal(data):
    data['WILL_signal'] = 0
    data.loc[(data['%R'] >= -80) & (data['%R'].shift(1) <= -80), 'WILL_signal'] = 1
    data.loc[(data['%R'] <= -20) & (data['%R'].shift(1) >= -20), 'WILL_signal'] = -1
    
# Adjust signal 
def add_OBV_signal(data):
    data['OBV_signal'] = 0
    data.loc[(data['OBV'] > data['OBV'].shift(1)) & (data['close'] > data['close'].shift(1)), 'OBV_signal'] = 1
    data.loc[(data['OBV'] < data['OBV'].shift(1)) & (data['close'] < data['close'].shift(1)), 'OBV_signal'] = -1

def add_VWAP_signal(data):
    data['VWAP_signal'] = 0
    data.loc[(data['close'] >= data['VWAP']) & (data['close'].shift(1) <= data['VWAP'].shift(1)), 'VWAP_signal'] = 1
    data.loc[(data['close'] <= data['VWAP']) & (data['close'].shift(1) >= data['VWAP'].shift(1)), 'VWAP_signal'] = -1

#Not sure strategy
def add_ICHI_signal(data):
    data['ICHI_signal'] = 0
    data.loc[(data['close'] > data['senkouA']) & (data['close'] > data['senkouB']) & 
             (data['tenkan'] > data['kijun']) & (data['chikou'] > data['close']), 'ICHI_signal'] = 1
    data.loc[(data['close'] < data['senkouA']) & (data['close'] < data['senkouB']) & 
             (data['tenkan'] < data['kijun']) & (data['chikou'] < data['close']), 'ICHI_signal'] = -1

     
def calculate_all_signals(data):
    add_RSI_signal(data)
    add_MFI_signal(data)
    add_MACD_signal(data)
    add_BOLL_signal(data)
    add_PSAR_signal(data)
    add_ADX_signal(data)
    add_SO_signal(data)
    add_WILL_signal(data)
    add_OBV_signal(data)
    add_VWAP_signal(data)
    # add_ICHI_signal(data)
    return data