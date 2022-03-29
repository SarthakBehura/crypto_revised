import datetime
import time
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from feature_utils import fund_features, apply_parallel, \
    technicals_with_lag, time_features, interaction_features, reduce_mem_usage
from config import FEATURES

warnings.filterwarnings('ignore')


def main(df, df2):
    all_data = df[['close_time', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                   'number_of_trades', 'quote_asset_volume', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume']].copy()
    # index_data = all_data[all_data['ticker'] == 'BTCUSDT'][['close_time', 'close']].copy()
    index_data = df2[['close_time', 'close']].copy()
    #
    index_data.rename(columns={'close': 'index_close'}, inplace=True)
    # # all_data = all_data[all_data['ticker'] != 'BTCUSDT'].copy()
    all_data = all_data.merge(index_data, on='close_time', how='left')
    all_data.rename(columns={'close_time': 'date'}, inplace=True)
    #
    all_data['date'] = all_data['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x) / 1000))

    # all_data = all_data[all_data['date'] >= '2019-12-01']
    all_data.rename(columns={'final_index_close': 'index_close',
                             'close_time': 'date'}, inplace=True)
    print(all_data.shape)
    print(all_data['ticker'].nunique())
    # all_data = all_data[all_data['date'] < datetime.datetime.utcnow()].copy()
    # Append derived features to main dataframe
    print('Building derived features')
    _derived(all_data)
    _lag(all_data)

    # print('Building time based features')
    time_features(all_data)

    print('Building interaction features')
    interaction_features(all_data)
    _lag(all_data, feat_for_lag=['close_by_open', 'high_by_open', 'high_by_close',
                                 'low_by_open', 'low_by_close', 'high_minus_low',
                                 'close_minus_open'])

    # Build fund features
    print('Building fund features')
    fund_df = apply_parallel(all_data.groupby('ticker'), fund_features)
    # reduce_mem_usage(fund_df)

    # TI with lag
    print('Building Technicals with lag')
    ti_df = apply_parallel(all_data.groupby(['ticker']), technicals_with_lag).reset_index()
    # reduce_mem_usage(ti_df)

    # Other features
    print('Building other technicals')
    tech_df = _technicals(all_data)
    # reduce_mem_usage(tech_df)

    # tech_df = tech_df[tech_df['date'] == all_data['date'].max()]
    # all_data = all_data[all_data['date'] == all_data['date'].max()]
    regexp = '^(?!.*_DROP)'
    all_data = all_data.merge(tech_df, on=['date', 'ticker'], how='left',
                              suffixes=('', '_DROP')).filter(regex=regexp)

    all_data = all_data.merge(fund_df, on=['date', 'ticker'], how='left',
                              suffixes=('', '_DROP')).filter(regex=regexp)

    all_data = all_data.merge(ti_df, on=['date', 'ticker'], how='left',
                              suffixes=('', '_DROP')).filter(regex=regexp)
    # reduce_mem_usage(all_data)

    # all_data.to_csv('../data/all_data.csv', index=False)
    # tech_df.to_csv('../data/tech_df.csv', index=False)
    # fund_df.to_csv('../data/fund_df.csv', index=False)
    # ti_df.to_csv('../data/ti_df.csv', index=False)

    return all_data


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
    all_data['volatility'] = (all_data['high'] - all_data['low']) / all_data['close']

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


def _lag(all_data, feat_for_lag=None):
    if feat_for_lag is None:
        feat_for_lag = ['vwap', 'bull_candle', 'bear_candle', 'volatility', 'market_order_per_move',
                        'avg_trade_size_move', 'open', 'high', 'low', 'close', 'volume']
    for i in feat_for_lag:
        for j in range(1, 16):
            all_data[f'{i}_lag{j}'] = all_data.sort_values(by=['ticker',
                                                               'date']).groupby(['ticker'])[i].transform(
                lambda x: x.shift(j))


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

    final = pd.concat(ls, ignore_index=True)
    # Future data leak features
    drop = ['DPO_20', 'ICS_26', 'QQEs_14_5_4.236', 'SUPERTl_7_3.0', 'PSARs_0.02_0.2',
            'HILOs_13_21', 'PSARl_0.02_0.2', 'HILOl_13_21', 'SUPERTs_7_3.0', 'EOM_14_100000000']
    final = final.drop(columns=drop)
    return final


def merge(df, file):
    df2 = pd.read_csv(file)
    regexp = '^(?!.*_DROP)'
    df = df.merge(df2, on=['date', 'ticker'], how='left',
                  suffixes=('', '_DROP')).filter(regex=regexp)
    return df


if __name__ == "__main__":
    # final_df = data_fetch(start_date=datetime.datetime.now().strftime("%Y-%m-%d"))
    feat_df = pd.read_csv('../data/raw_data_subset_usdt_1hour.pkl')
    feat_2 = pd.read_csv('../data/ETHUSDT_15min_candle_test.csv')

    # print(feat_df.shape)
    # print(feat_df['ticker'].nunique())

    # final_df = main(feat_df, [])
    # final_df.to_csv('../data/usdt_all_features_v2.csv', index=False)
    # print(final_df.tail())
    all_data = main(feat_df, feat_2)
    # all_data = pd.read_csv('../data/all_data.csv')
    # tech_df = pd.read_csv('../data/tech_df.csv')

    # all_data = merge(all_data, '../data/tech_df.csv')
    # all_data = merge(all_data, '../data/fund_df.csv')
    # all_data = merge(all_data, '../data/ti_df.csv')
    #
    all_data.to_pickle('../data/all_features_usdt_1hour_newindex.pkl')
