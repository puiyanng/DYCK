'''
- May be interesting: Interest rates: https://quant.stackexchange.com/questions/20911/how-to-predict-daily-range-of-forex
Feature transformation:
- For every DJIA day make Close = next Open
- Generate Y: BUY, SELL, HOLD at time t signals if p(t + 1) is higher, lower or the same as p(t), lets call it "Signal"
- Put together the S&P OHLC, S&P vol and closing prices of all the potential features in one file
- Normalize the data: Calculate daily log returns on the price data
- Put all of the above in one file

Data integration:
- It integrates all data S&P500, DJI, FTSE, N225, GOLD, and exchange rates as a single csv file.
- Keeps in mind that it's just raw data, meaning that missing data has not been handled. The missing data can be due to different rest days across different markets. (US, UK, Japan)

Feature selection:
- Choose the top 4 indeces using:
- Perform correlation matrix analysis
- PCA
- Make a call in case of conflict between the above 2 methods i.e. pick what we think is better


'''

import pandas as pd
import download_data as dd
import numpy as np
import ta


class Signals:
    def HOLD(self): return int(0)

    def BUY(self): return int(1)

    def SELL(self): return int(2)

    def SIGNAL(self):
        return "Signal"


SIGNALS = Signals()


# def equalize_close_open()

def generate_y(df, col_name):
    diff = df[col_name].diff(periods=-1)
    diff.values[diff.values == 0] = SIGNALS.HOLD()
    diff.values[diff.values > 0] = SIGNALS.SELL()
    diff.values[diff.values < 0] = SIGNALS.BUY()
    return diff


def log_returns(df, col_name):
    ratio = df[col_name] / df[col_name].shift(1)
    return np.log(ratio)


def standardize(df, col_name):
    col = df[col_name]
    mean = col.mean()
    std = col.std()
    return ((col - mean) / std)


# correlation matrix
def corr_matrix():
    df = pd.read_csv('data_id_5y.csv', index_col=0)
    # forward fill
    df = df.ffill(axis=0)
    # all close prices
    df = df.filter(regex="Close")
    matrix = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
    return (matrix)


def normalize_data(data, save=False):
    data = data.ffill(axis=0)
    for col in data.columns:
        if col != "Date" and col != "Volume" and col != "Signal":
            data[col] = log_returns(data, col)
            data[col] = data[col].replace(to_replace=0, method='ffill')

    data = data.dropna()
    if save:
        data.to_csv(dd.file_name("data_normalized"))
    return data


# Non-normalized data should be fed to this function
def add_technical_indicators(raw_data):
    return ta.add_all_ta_features(raw_data, "Open", "High", "Low", "Close", "Volume", fillna=True)
