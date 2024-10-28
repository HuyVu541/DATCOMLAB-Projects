import numpy as np
import pandas as pd
from Trend_Detection import add_trend

def add_rsi(data):
    rsi_period = 14
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

def add_macd(data):
    data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()

    # Calculate the 26-period EMA
    data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()

    # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
    data['MACD'] = data['EMA12'] - data['EMA26']

    # Calculate the 9-period EMA of MACD (Signal Line)
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data.drop(['EMA12', 'EMA26'], axis = 1)

def gain(x):
    return ((x > 0) * x).sum()

def loss(x):
    return ((x < 0) * x).sum()

# Calculate money flow index
def add_mfi(data, n=14):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign

    # Calculate gain and loss using vectorized operations
    positive_mf = np.where(signed_mf > 0, signed_mf, 0)
    negative_mf = np.where(signed_mf < 0, -signed_mf, 0)

    mf_avg_gain = pd.Series(positive_mf).rolling(window=n).sum()
    mf_avg_loss = pd.Series(negative_mf).rolling(window=n).sum()

    data['MFI'] = (100 - 100 / (1 + mf_avg_gain / mf_avg_loss)).to_numpy()

def psar(barsdata, iaf = 0.02, inc=0.02,maxaf = 0.2):
    length = len(barsdata)
    dates = list(barsdata['time'])
    high = list(barsdata['high'])
    low = list(barsdata['low'])
    close = list(barsdata['close'])
    psar = close[0:len(close)]
    cpsar = close[0:len(close)]
    bb = close[0:len(close)]
    EP =close[0:len(close)]
    AF=close[0:len(close)]
    rev =close[0:len(close)]
    #psarbull = [None] * length
    #psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2,length):
        
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        cpsar[i]=psar[i]
        
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = max(hp,high[i])
                lp = low[i]
                EP[i]=lp
                af = iaf
                AF[i]=af
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = min(lp,low[i])
                hp = high[i]
                EP[i]=hp
                af = iaf
                AF[i]=af
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + inc, maxaf)
                    EP[i] =hp
                    AF[i]=af
                    #psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
                else:
                    AF[i]=AF[i-1]
                    EP[i]=EP[i-1]
                
                    
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
                    
                
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + inc, maxaf)
                    EP[i]=lp
                    AF[i]=af
                    #psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
                else:
                    AF[i]=AF[i-1]
                    EP[i]=EP[i-1]
                
                   
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
                
                    
        bb[i]=bull
        rev[i]=reverse

    return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar,"bb":bb,"cpsar":cpsar,"ep":EP,"af":AF,"rev":rev}

def add_psar(data):
    data['PSAR'] = psar(data)['psar']
    data['rev'] = psar(data)['rev']

def add_bollinger_bands(data):

    data['SMA'] = data['close'].rolling(window=20).mean()

    # Calculate the 20-period Standard Deviation (SD)
    data['SD'] = data['close'].rolling(window=20).std()

    # Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB)
    data['UB'] = data['SMA'] + 2 * data['SD']
    data['LB'] = data['SMA'] - 2 * data['SD']

    data.drop('SD', axis = 1)

def add_atr(data):
    atr_period = 14
    TR = np.maximum(data['high'] - data['low'], abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1)))
    first_ATR = TR.rolling(window=atr_period).mean().dropna().iloc[0]
    ATR = pd.Series(index=TR.index)
    ATR.iloc[0] = first_ATR
    for i in range(1, len(ATR)):
        ATR[i] = (ATR[i-1] * (atr_period - 1) + TR[i])/atr_period
    data['ATR'] = ATR

def add_adx(data):
    adx_period = 14
    plus_DM = data['high'].diff()
    minus_DM = data['low'].shift(1) - data['low']
    plus_DM = pd.Series(np.where((plus_DM > minus_DM) & (plus_DM > 0), plus_DM, 0))
    minus_DM = pd.Series(np.where((plus_DM < minus_DM) & (minus_DM > 0), minus_DM, 0))

    ATR = data['ATR'].copy()
    plus_DI = plus_DM.rolling(window=adx_period).mean() / ATR * 100
    minus_DI = minus_DM.rolling(window=adx_period).mean() / ATR * 100

    DX = np.abs((plus_DI - minus_DI) / (plus_DI + minus_DI))*100
    first_ADX = DX.rolling(window=adx_period).mean().dropna().iloc[0]
    idx_first_ADX = DX.rolling(window=adx_period).mean().dropna().index[0]
    ADX = pd.Series(index=DX.index)
    ADX.iloc[idx_first_ADX - 1] = first_ADX
    for i in range(idx_first_ADX, len(DX)):
        ADX[i] = (ADX[i-1] * (adx_period - 1) + DX[i])/adx_period
    data['ADX'] = ADX

def add_so(data):
    so_period = 14
    D_period = 3
    L14 = data['low'].rolling(window=so_period).min()
    H14 = data['high'].rolling(window=so_period).max()
    data['%K'] = (data['close'] - L14) / (H14 - L14) * 100
    data['%D'] = data['%K'].rolling(window = D_period).mean()

def add_williams_r(data):
    will_period = 14
    L14 = data['low'].rolling(window=will_period).min()
    H14 = data['high'].rolling(window=will_period).max()
    data['%R'] = (H14 - data['close']) / (H14 - L14) * -100

def add_pivotpoint(data): #For stop loss maybe
    data['p'] = (data['high'] + data['low'] + data['close']) / 3
    data['r1'] = 2*data['p'] - data['low']
    data['r2'] = data['p'] + (data['high'] - data['low'])
    data['s1'] = 2*data['p'] - data['high']
    data['s2'] = data['p'] - (data['high'] - data['low'])

def add_obv(data):
    data['OBV'] = 0
    data['OBV'] = np.where(data['close'] > data['close'].shift(1), 
                           data['volume'], 
                           np.where(data['close'] < data['close'].shift(1), 
                                    -data['volume'], 
                                    0)).cumsum()
            
def add_VWAP(data):
    cum_price_vol = (data['p'] * data['volume']).cumsum()
    cum_vol = data['volume'].cumsum()
    data['VWAP'] = cum_price_vol/cum_vol

def add_ichimoku(data):
    tenkan_period = 9
    kijun_period = 26
    senkou_period = 52
    data['tenkan'] = (data['high'].rolling(window=tenkan_period).max() + data['low'].rolling(window=tenkan_period).min()) / 2
    data['kijun'] = (data['high'].rolling(window=kijun_period).max() + data['low'].rolling(window=kijun_period).min()) / 2
    data['senkouA'] = ((data['tenkan'] + data['kijun']) / 2).shift(kijun_period)
    data['senkouB'] = ((data['high'].rolling(window=senkou_period).max() + data['low'].rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    data['chikou'] = data['close'].shift(-kijun_period)

def calculate_indicators(data):

    data.dropna(inplace = True)
    data.reset_index(inplace = True, drop = True) 

    add_mfi(data)
    add_psar(data)

    data = data[['time', 'high', 'low', 'close', 'volume', 'MFI', 'PSAR', 'rev']].copy()

    add_bollinger_bands(data)
    add_rsi(data)
    add_macd(data)
    add_atr(data)
    add_adx(data)
    add_so(data)
    add_williams_r(data)
    add_pivotpoint(data)
    add_obv(data)
    add_VWAP(data)
    add_ichimoku(data)

    data['sma10'] = data['close'].rolling(window = 10).mean()
    data['sma20'] = data['close'].rolling(window = 20).mean()
    data['sma50'] = data['close'].rolling(window = 50).mean()
    data['sma200'] = data['close'].rolling(window = 200).mean()
    data['ema'] = data['close'].ewm(span = 10, adjust=False).mean()

    add_trend(data)

    return data