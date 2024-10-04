import pandas as pd
import numpy as np

def add_trend(data, fast = 'sma20', slow = 'sma50'):
    data['trend'] = 'None'
    data.loc[data.iloc[np.where((data[fast] > data[slow]) & \
                                (data[fast].shift(1) < data[slow].shift(1)))].index, 'trend'] = 'Up'
    # Ở ngày trước đó sma50 đang thấp hơn, hôm nay cao hơn
    data.loc[data.iloc[np.where((data[fast] < data[slow]) & \
                                (data[fast].shift(1) > data[slow].shift(1)))].index, 'trend'] = 'Down'
    # Ở ngày trước đó sma50 đang cao hơn, hôm nay thấp hơn
    checkpoint = 0
    for i in range(len(data)):
        if data.loc[i, 'trend'] == 'Up':
            data.loc[checkpoint : i-1, 'trend'] = 'Down'
            checkpoint = i
        elif data.loc[i, 'trend'] == 'Down':
            data.loc[checkpoint : i-1, 'trend'] = 'Up'
            checkpoint = i
    data.loc[checkpoint:, 'trend'] = data.loc[checkpoint, 'trend']

    return data