import datetime
import time
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
# from crypto_model import fund_features as fdf
import fund_features as fdf
from config import FEATURES

warnings.filterwarnings('ignore')


def main(df):
    all_data = df[['close_time', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                   'number_of_trades', 'quote_asset_volume', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume']].copy()
    index_data = all_data[all_data['ticker'] == 'BTCUSDT'][['close_time', 'close']].copy()
    index_data.rename(columns={'close': 'index_close'}, inplace=True)
    all_data = all_data[all_data['ticker'] != 'BTCUSDT'].copy()
    all_data = all_data.merge(index_data, on='close_time', how='left')
    all_data.rename(columns={'close_time': 'date'}, inplace=True)
    all_data['date'] = pd.to_datetime(all_data['date'], unit='ms')
    all_data['date'] = all_data['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    # all_data = all_data[all_data['date'] < datetime.datetime.utcnow()].copy()
    # Append derived features to main dataframe
    print('Building derived features')
    _derived(all_data)

    # Build fund features
    print('Building fund features')
    batches_of_funds = fdf.multiprocessing(all_data['ticker'].unique().tolist())
    all_data = fdf.feature_generator(batches_of_funds, all_data)

    # Other features
    print('Building other technicals')
    tech_df = _technicals(all_data)

    # tech_df = tech_df[tech_df['date'] == all_data['date'].max()]
    # all_data = all_data[all_data['date'] == all_data['date'].max()]
    regexp = '^(?!.*_DROP)'
    all_data = all_data.merge(tech_df, on=['date', 'ticker'], how='left',
                              suffixes=('', '_DROP')).filter(regex=regexp)
    # all_data = all_data.merge(tech_df, on=['date', 'ticker'], how='left')
    return_cols = list(set(FEATURES + ['open', 'high', 'low', 'close', 'volume', 'index_close']))
    # all_data['CKSPl_10_3_20'], all_data['CKSPs_10_3_20'] = all_data.ta.cksp()
    # not_present = [x for x in return_cols if x not in all_data.columns.tolist()]
    # return_cols = [x for x in return_cols if x in all_data.columns.tolist()]
    # print(not_present)
    all_data.ta.cksp(p=10, x=1, q=9, append=True)
    all_data.ta.amat(append=True)
    all_data.rename(columns={'CKSPl_10_1.0_9': 'CKSPl_10_1_9',
                             'CKSPs_10_1.0_9': 'CKSPs_10_1_9',
                             'AMATe_LR_8_21_2': 'AMATe_LR_2'}, inplace=True)
    all_data['HW-UPPER'] = 0
    return all_data[['date', 'ticker', *return_cols]]
    # return all_data


def _derived(all_data):
    # Build derived features
    # VWAP
    all_data['vwap'] = all_data['quote_asset_volume'] / all_data['volume']

    # Bull Candle
    mask = all_data['vwap'] > all_data['close']
    all_data['bull_candle'] = np.where(mask, 1, 0)

    # Bear Candle
    mask = all_data['vwap'] < all_data['close']
    all_data['bear_candle'] = np.where(mask, 1, 0)

    # Volatility
    all_data['volatility'] = (all_data['high'] - all_data['high']) / all_data['close']

    # Market Order percentage
    all_data['market_order_per'] = all_data['taker_buy_quote_asset_volume'] / all_data['quote_asset_volume']

    # Market Order percentage movement
    all_data['market_order_per_move'] = all_data.groupby(['ticker'])['market_order_per'].transform(lambda x:
                                                                                                   x / x.shift(1) - 1)

    # Average Trade Size
    all_data['avg_trade_size'] = all_data['quote_asset_volume'] / all_data['number_of_trades']

    # Average Trade Size movement
    all_data['avg_trade_size_move'] = all_data.groupby(['ticker'])['avg_trade_size'].transform(lambda x:
                                                                                               x / x.shift(1) - 1)


def _technicals(all_data):
    # Build ta all strategy features for all crypto
    ls = []
    for coin in all_data['ticker'].unique():
        print(f"Coin: {coin}")
        df = all_data[all_data['ticker'] == coin].copy().reset_index(drop=True)
        df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)

        # Calculate Returns and append to the df DataFrame
        df.ta.log_return(cumulative=True, append=True)
        df.ta.percent_return(cumulative=True, append=True)

        # Runs and appends all indicators to the current DataFrame by default
        # The resultant DataFrame will be large.
        df.ta.strategy("all", verbose=True, timed=True)

        df.reset_index(drop=True)
        ls.append(df)

    final_df = pd.concat(ls, ignore_index=True)

    ignore_cols = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'quote_asset_volume',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'index_close', 'vwap',
                   'bull_candle', 'bear_candle', 'volatility', 'market_order_per', 'market_order_per_move',
                   'avg_trade_size', 'avg_trade_size_move']

    return final_df.drop(columns=ignore_cols)


if __name__ == "__main__":
    # final_df = data_fetch(start_date=datetime.datetime.now().strftime("%Y-%m-%d"))
    feat_df = pd.read_csv('../data/all_data.csv')
    final_df = main(feat_df)
    final_df.to_csv('../data/all_features.csv', index=False)
    print(final_df.tail())
    # print(feat_df.ta.above(append=True))
    # print(feat_df)
