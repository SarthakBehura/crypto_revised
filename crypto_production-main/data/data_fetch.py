import datetime
import json
import time

import numpy as np
import pandas as pd
import multiprocessing as mp
from binance.client import Client

from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, TICKERS_1, TICKERS_2, TICKERS, USDT_TICKERS

DATA_COL = ['start_time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'Ignore']
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

completed = []
batch = 1


def data_fetch(start_date="05 July, 2021",
               output_file=None,
               interval=6,
               coins=None):
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    ls = []
    for symbol in coins:
        print(f'data fetching for ticker {symbol}')
        if interval == 6:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_6HOUR, start_date)
        elif interval == 30:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, start_date)
        elif interval == 60:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_date)
        elif interval == 15:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, start_date)
        else:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date)
        print('data fetched')
        # storing data in panadas dataframe object
        klines = pd.DataFrame(klines)
        klines['ticker'] = symbol

        klines.rename(columns={0: 'start_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close',
                               5: 'volume', 6: 'close_time', 7: 'quote_asset_volume', 8: 'number_of_trades',
                               9: 'taker_buy_base_asset_volume', 10: 'taker_buy_quote_asset_volume',
                               11: 'Ignore'}, inplace=True)

        ls.append(klines)

    # pandas concatenate list of tickers
    ls = pd.concat(ls, ignore_index=True)
    # writing to csv
    if output_file is not None:
        ls.to_csv(output_file, index=False)
    return ls


def data_fetch_one(symbol, start_date):
    # ls = pd.DataFrame(columns=[*DATA_COL, 'ticker'])
    print(f'data fetching for ticker {symbol}')
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, start_date)
    print('data fetched')
    klines = pd.DataFrame(klines, columns=DATA_COL)
    klines['ticker'] = symbol
    # print(klines)
    return klines


def main(currencies):
    print(len(currencies))
    ls = []
    counter = 1
    global completed
    global batch
    try:
        for coin in currencies:
            # coi = list(batch)
            final_df = data_fetch_one(symbol=coin,
                                      start_date='01 Jan 2019')
            ls.append(final_df)
            # final_df.to_csv(f'batch_{idx + 1}.csv', index=False)
            if counter % 50 == 0:
                # pandas concatenate list of tickers
                ls = pd.concat(ls, ignore_index=True)
                ls.to_csv(f'batch_{batch}.csv', index=False)
                ls = []
                batch += 1
                time.sleep(60)
            counter += 1
            completed.append(coin)
        if len(ls) != 0:
            # pandas concatenate list of tickers
            ls = pd.concat(ls, ignore_index=True)
            ls.to_csv(f'batch_{batch}.csv', index=False)
            batch += 1
    except Exception as e:
        print('Coins completed: ')
        print(completed)
        time.sleep(60)
        print('Reinvoking main')
        global client
        client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

        new = list(set(TICKERS.copy()) - set(completed))
        main(new)


if __name__ == "__main__":
    # final_df = data_fetch(start_date=datetime.datetime.now().strftime("%Y-%m-%d"))
    # print(batches)

    main(USDT_TICKERS)
    # final_df['start_time'] = final_df['start_time'].apply(lambda x: datetime.datetime.fromtimestamp(int(x) / 1000))
    # final_df['close_time'] = final_df['close_time'].apply(lambda x: datetime.datetime.fromtimestamp(int(x) / 1000))

    # print(final_df.tail())

    # Append all batches
    # ls = []
    # for i in range(1, 4):
    #     temp = pd.read_csv(f'batch_{i}.csv')
    #     ls.append(temp)
    # #
    # ls = pd.concat(ls, ignore_index=True)
    # ls.to_csv('all_data_historic.csv', index=False)
