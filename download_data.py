import yfinance as yf


def file_name(fx, interval, period):
    return "data/" + fx + "_" + interval + "_" + period + ".csv"


data_files = []

usd_to_fx = ["JPY=X", "GBP=X", "EUR=X"]
# NOTE: hourly data is not available beyond the past 730 days
intervals_periods = [("1h", "2y"), ("1d", "5y")]

for fx in usd_to_fx:
    for i_p in intervals_periods:
        data = yf.download(
            tickers=fx,
            period=i_p[1],
            interval=i_p[0],
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )
        # Please follow this naming convention across the project
        file = file_name(fx, i_p[0], i_p[1])
        data_files.append(file)
        data.to_csv(file)
