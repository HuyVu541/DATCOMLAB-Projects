import pandas as pd

def add_bollinger_bands(data):
    data['SMA'] = data['close'].rolling(window=20).mean()

    # Calculate the 20-period Standard Deviation (SD)
    data['SD'] = data['close'].rolling(window=20).std()

    # Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB)
    data['UB'] = data['SMA'] + 2 * data['SD']
    data['LB'] = data['SMA'] - 2 * data['SD']

        data.drop('SD', axis = 1)