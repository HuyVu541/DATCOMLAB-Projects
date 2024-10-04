import numpy as np
import pandas as pd

def add_rsi(data):
    rsi_period = 10
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

    mf_avg_gain = pd.Series(positive_mf).rolling(n, min_periods=1).sum()
    mf_avg_loss = pd.Series(negative_mf).rolling(n, min_periods=1).sum()

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

def psar_trend(data):
    data['psar_trend'] = 'None'
    
    data.loc[data.iloc[np.where(data['close'] > data['PSAR'])].index, 'psar_trend'] = 'Up'
    data.loc[data.iloc[np.where(data['close'] < data['PSAR'])].index, 'psar_trend'] = 'Down'

def add_bollinger_bands(data):

    data['SMA'] = data['close'].rolling(window=20).mean()

    # Calculate the 20-period Standard Deviation (SD)
    data['SD'] = data['close'].rolling(window=20).std()

    # Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB)
    data['UB'] = data['SMA'] + 2 * data['SD']
    data['LB'] = data['SMA'] - 2 * data['SD']

    data.drop('SD', axis = 1)


def calculate_indicators(data):
    add_mfi(data)

    add_psar(data)

    data = data[['time', 'close', 'MFI', 'PSAR', 'rev']]

    # data.set_index('time', inplace = True)

    data = data.dropna()

    add_bollinger_bands(data)

    add_rsi(data)

    add_macd(data)

    data.dropna(inplace = True)
    
    data.reset_index(inplace = True, drop = True)

    data['sma10'] = data['close'].rolling(window = 10, min_periods = 1).mean()
    
    data['sma20'] = data['close'].rolling(window = 20, min_periods = 1).mean()

    data['sma50'] = data['close'].rolling(window = 50, min_periods = 1).mean()

    data['sma200'] = data['close'].rolling(window = 200, min_periods = 1).mean()
    
    data['ema'] = data['close'].ewm(span = 10).mean()

    add_trend(data)

    psar_trend(data)

    return data