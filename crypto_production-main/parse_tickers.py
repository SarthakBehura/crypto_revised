import numpy as np
import pandas as pd


def valid(chunks, coins):
    for chunk in chunks:
        mask = chunk['ticker'].isin(coins)
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            # break


def main(file, tickers):
    chunksize = 10 ** 5
    chunks = pd.read_csv(file, chunksize=chunksize)
    df = pd.concat(valid(chunks, tickers))
    return df


if __name__ == "__main__":
    tickers = []
    # tickers.append('BTCUSDT')
    output_df = main('filepath for features file', tickers)
    output_df.to_csv('all_data_v1.csv', index=False)

