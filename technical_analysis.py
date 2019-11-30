
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path
mpl.style.use('seaborn')


df = pd.read_csv('data/DJI_1d_10y.csv', sep=',')
df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)

plt.plot(df[2000:2500].Close)
plt.plot(df[2200:2500].volatility_bbh, label='High BB')
plt.plot(df[2200:2500].volatility_bbl, label='Low BB')
plt.plot(df[2200:2500].volatility_bbm, label='EMA BB')
plt.title('Bollinger Bands')
plt.legend()
plt.savefig('Charts/BB.png')

plt.plot(df[2000:2500].Close)
plt.plot(df[2000:2500].volatility_kcc, label='Central KC')
plt.plot(df[2000:2500].volatility_kch, label='High KC')
plt.plot(df[2000:2500].volatility_kcl, label='Low KC')
plt.title('Keltner Channel')
plt.legend()

plt.plot(df[2000:2500].trend_macd, label='MACD')
plt.plot(df[2000:2500].trend_macd_signal, label='MACD Signal')
plt.plot(df[2000:2500].trend_macd_diff, label='MACD Difference')
plt.title('MACD, MACD Signal and MACD Difference')
plt.legend()
plt.savefig('Charts/MACD.png')


outpath = "Charts/"
for col in df.columns:
    plt.plot(df[col])
    plt.title(col)
    plt.savefig(path.join(outpath, "graph{0}.png".format(col)))


df.to_csv('technical_analysis_results.csv')





