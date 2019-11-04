import pandas as pd
import download_data as dd
import feature_engineering as fe

data = pd.read_csv(dd.file_name("data", dd.interval_period))

data["Signal"] = fe.generate_y(data, "DJI_Close")
data.to_csv(dd.file_name("data", dd.interval_period))
