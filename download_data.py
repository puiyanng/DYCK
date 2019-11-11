import yfinance as yf

# NOTE => DO NOT JUST RE RUN THIS FILE, THIS WILL (OBVIOUSLY) RE-DOWNLOAD AND UPDATE THE DATA

# Note: ("CL=F", "OIL"), ("GC=F", "GOLD") prices aren't available for enough days, so I used Bloomberg data for gold, trying to get oil;s

instrument_to_name_map = [("^GSPC", "SP"), ("^NYA", "NYSE"), ("^DJI", "DJI"), ("^RUT", "RUSS")
    , ("^IXIC", "NASDAQ"), ("^FTSE", "FTSE"), ("JPY=X", "USDJPY"), ("GBP=X", "USDGBP"), ("EUR=X", "USDEUR"),
                          ("^N225", "N225"), ("CNY=X", "USDCNY")]

interval_period = ("1d", "5y")

def file_name(inst):
    return "data/" + inst + "_" + interval_period[0] + "_" + interval_period[1] + ".csv"




def download():
    for inst in instrument_to_name_map:
        data = yf.download(
            tickers=inst[0],
            period=interval_period[1],
            interval=interval_period[0],
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )
        # Please follow this naming convention across the project
        file = file_name(inst[1])
        data.to_csv(file)
