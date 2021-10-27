import datetime
import numpy as np
import pandas as pd

from data import data_fetch
from crypto_model import features, predict


def driver(lag=20):
    # Get data
    date = datetime.datetime.now() - datetime.timedelta(days=lag)
    df = data_fetch.data_fetch(start_date=date.strftime("%Y-%m-%d"),
                               interval=6)
    # Features build
    feat_df = features.main(df)
    print(feat_df.head())
    # Model prediction
    output = predict.main(feat_df)
    output_date = output['date'].values[0]
    output.to_csv(f"output/output_{pd.to_datetime(output_date).strftime('%Y_%m_%d-%I_%M_%S_%p')}.csv",
                  index=False)

    return output
