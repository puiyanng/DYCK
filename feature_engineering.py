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
    
    
def data_integrate():
    data_1d_5y = pd.read_csv('data_1d_5y.csv') # Please type the path of the original data by Danish
    data_1d_5y['Date'] = pd.to_datetime(data_1d_5y['Date'], dayfirst=True)

    data_1d_5y.set_index('Date', inplace=True)
    data_1d_5y.drop(['Unnamed: 0'], axis=1, inplace=True)
    # data_1d_5y.head(5)


    dji = pd.read_csv('DJI_1d_5y.csv') # US data
    dji['Date'] = pd.to_datetime(dji['Date'], dayfirst=True)
    dji = dji[['Date', 'Close']]
    dji.rename(columns={'Close': 'DJI_Close'}, inplace=True)
    dji.set_index('Date', inplace=True)
    #dji.head(10)
    data_1d_5y = data_1d_5y.join(dji)
    # data_1d_5y.head(5)


    ftse = pd.read_csv('FTSE_1d_5y.csv') # UK data
    ftse['Date'] = pd.to_datetime(ftse['Date'], dayfirst=True)
    ftse = ftse[['Date', 'Close']]
    ftse.rename(columns={'Close': 'FTSE_Close'}, inplace=True)
    ftse.set_index('Date', inplace=True)
    # ftse.head(10)
    data_1d_5y = data_1d_5y.join(ftse)
    # data_1d_5y.head(5)


    gold = pd.read_csv('GOLD_1d_5y.csv') # Gold Price
    gold['Date'] = pd.to_datetime(gold['Dates'], dayfirst=True)
    gold = gold[['Date', 'Close']]
    gold.rename(columns={'Close': 'GOLD_Close'}, inplace=True)
    gold.set_index('Date', inplace=True)
    #gold.head(5)
    data_1d_5y = data_1d_5y.join(gold)
    # data_1d_5y.head(5)


    N225 = pd.read_csv('N225_1d_5y.csv') # Japan Data
    N225['Date'] = pd.to_datetime(N225['Date'], dayfirst=True)
    N225 = N225[['Date', 'Close']]
    N225.rename(columns={'Close': 'N225_Close'}, inplace=True)
    N225.set_index('Date', inplace=True)
    # N225.head(5)
    data_1d_5y = data_1d_5y.join(N225)
    # data_1d_5y.head(5)


    currency = ['CNY', 'EUR', 'GBP', 'JPY']
    for cur in currency:
        currency_df = pd.read_csv(f"USD{cur}_1d_5y.csv") # Currency
        currency_df['Date'] = pd.to_datetime(currency_df['Date'], dayfirst=True)
        currency_df = currency_df[['Date', 'Close']]
        currency_df.rename(columns={'Close': f"USD{cur}_Close"}, inplace=True)
        currency_df.set_index('Date', inplace=True)
        data_1d_5y = data_1d_5y.join(currency_df)
    # data_1d_5y.head(5)
    data_1d_5y[data_1d_5y.isna().any(axis=1)]
    data_1d_5y.reset_index(inplace=True)
    data_1d_5y.to_csv('data_1d_5y.csv', index=True)


# def harmonize_dates():
SP_file_name = dd.file_name("SP", dd.interval_period)
SP = pd.read_csv(SP_file_name)
print(standardize(SP, "Volume"))
# SP[SIGNALS.SIGNAL()] = generate_y(SP, "Close")
# SP.to_csv(SP_file_name)


# correlation matrix
def corr_matrix():
    df = pd.read_csv('data_id_5y.csv',index_col=0)
    #forward fill
    df = df.ffill(axis=0)
    #all close prices
    df = df.filter(regex="Close")
    matrix = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
    return(matrix)

