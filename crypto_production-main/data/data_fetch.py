import datetime
import pandas as pd
from binance.client import Client

from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, TICKERS


def data_fetch(start_date="05 July, 2021",
               output_file=None,
               interval=6,
               coins=None):
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    if not coins:
        coins = TICKERS
    ls = []
    for symbol in coins:
        print(f'data fetching for ticker {symbol}')

        if interval == 6:
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_6HOUR, start_date)
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


if __name__ == "__main__":
    # final_df = data_fetch(start_date=datetime.datetime.now().strftime("%Y-%m-%d"))
    final_df = data_fetch(output_file='all_data.csv')
    final_df['start_time'] = pd.to_datetime(final_df['start_time'], unit='ms')
    final_df['close_time'] = pd.to_datetime(final_df['close_time'], unit='ms')

    final_df['start_time'] = final_df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    final_df['close_time'] = final_df['close_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    print(final_df.tail())
