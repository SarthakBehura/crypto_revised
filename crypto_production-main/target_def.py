import datetime
import numpy as np
import pandas as pd


def main(raw_data, candle=15):
    all_data = raw_data[['close_time', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    index_data = all_data[all_data['ticker'] == 'BTCUSDT'][['close_time', 'close']].copy()
    index_data.rename(columns={'close': 'index_close'}, inplace=True)
    all_data = all_data[all_data['ticker'] != 'BTCUSDT'].copy()
    all_data = all_data.merge(index_data, on='close_time', how='left')
    all_data.rename(columns={'close_time': 'date'}, inplace=True)

    all_data['date'] = all_data['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x) / 1000))
    # Get half an hour returns for the next 6 hours
    if candle == 15:
        for i in range(1, 25):
            all_data[f'next_{i * 15}min_return'] = all_data.sort_values(by=['ticker', 'date']).groupby(['ticker'])['close'].transform(
                lambda x: (x.shift(-i) / x.shift(-(i - 1))) - 1)
            all_data[f'next_{i * 15}min_index_return'] = all_data.sort_values(by=['ticker',
                                                                                  'date']).groupby(['ticker'])['index_close'].transform(
                lambda x: (x.shift(-i) / x.shift(-(i - 1))) - 1)
            all_data[f'next_{i * 15}min_rel_return'] = all_data[f'next_{i * 15}min_return'] - \
                                                       all_data[f'next_{i * 15}min_index_return']

        # Percentile of individual 30 minute returns
        mean_columns = []
        check_columns = []
        for i in range(1, 25):
            all_data[f'next_perc_{i * 15}min_rel_return'] = all_data.groupby(['date'])[
                f'next_{i * 15}min_rel_return'].rank(
                pct=True)
            mean_columns.append(f'next_perc_{i * 15}min_rel_return')
            check_columns.append(f'next_{i * 15}min_rel_return')

        all_data['next_mean_continuous'] = all_data[mean_columns[:2]].mean(axis=1)
        all_data['next_perc30_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:4]].mean(axis=1)
        all_data['next_perc60_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:6]].mean(axis=1)
        all_data['next_perc90_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:8]].mean(axis=1)
        all_data['next_perc2_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:12]].mean(axis=1)
        all_data['next_perc3_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns].mean(axis=1)
        all_data['next_perc6_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)
    else:
        for i in range(1, 13):
            all_data[f'next_{i * 30}min_return'] = all_data.sort_values(by=['ticker', 'date']).groupby(['ticker'])['close'].transform(
                lambda x: (x.shift(-i) / x.shift(-(i - 1))) - 1)
            all_data[f'next_{i * 30}min_index_return'] = all_data.sort_values(by=['ticker',
                                                                                  'date']).groupby(['ticker'])['index_close'].transform(
                lambda x: (x.shift(-i) / x.shift(-(i - 1))) - 1)
            all_data[f'next_{i * 30}min_rel_return'] = all_data[f'next_{i * 30}min_return'] - \
                                                       all_data[f'next_{i * 30}min_index_return']

        # Percentile of individual 30 minute returns
        mean_columns = []
        check_columns = []
        for i in range(1, 13):
            all_data[f'next_perc_{i * 30}min_rel_return'] = all_data.groupby(['date'])[
                f'next_{i * 30}min_rel_return'].rank(
                pct=True)
            mean_columns.append(f'next_perc_{i * 30}min_rel_return')
            check_columns.append(f'next_{i * 30}min_rel_return')

        all_data['next_mean_continuous'] = all_data[mean_columns[:1]].mean(axis=1)
        all_data['next_perc30_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:2]].mean(axis=1)
        all_data['next_perc60_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:3]].mean(axis=1)
        all_data['next_perc90_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:4]].mean(axis=1)
        all_data['next_perc2_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns[:6]].mean(axis=1)
        all_data['next_perc3_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

        all_data['next_mean_continuous'] = all_data[mean_columns].mean(axis=1)
        all_data['next_perc6_conti_return'] = all_data.groupby(['date'])['next_mean_continuous'].rank(pct=True)

    # check = np.sign(all_data[check_columns]) >= 0
    # all_data['next_all_12_pos'] = (check.all(axis=1)).astype(int)
    #
    # check = np.sign(all_data[check_columns]) < 0
    # all_data['next_all_12_neg'] = (check.all(axis=1)).astype(int)

    ''' Type 1 Target variable classification'''
    # Class 1: Top 25% in consistent performance
    # Class 2: 25%-50%
    # Class 3: 50%-75%
    # Class 4: Bottom 25%
    # all_data['beat_miss_multiclass'] = 4
    # condition_2 = all_data['next_perc6_conti_return'] >= 0.75
    # all_data['beat_miss_multiclass'] = np.where(condition_2, 1, all_data['beat_miss_multiclass'])
    # condition_3 = ((all_data['next_perc6_conti_return'] < 0.75) & (all_data['next_perc6_conti_return'] >= 0.50))
    # all_data['beat_miss_multiclass'] = np.where(condition_3, 2, all_data['beat_miss_multiclass'])
    # condition_3 = ((all_data['next_perc6_conti_return'] < 0.50) & (all_data['next_perc6_conti_return'] >= 0.25))
    # all_data['beat_miss_multiclass'] = np.where(condition_3, 3, all_data['beat_miss_multiclass'])

    ''' Type 2 Target variable classification'''
    # Binary Class: Top 25% of continuous relative returns for next 6 hours
    all_data['beat_miss_binary30'] = np.where(all_data['next_perc30_conti_return'] >= 0.75, 1, 0)
    all_data['beat_miss_binary60'] = np.where(all_data['next_perc60_conti_return'] >= 0.75, 1, 0)
    all_data['beat_miss_binary90'] = np.where(all_data['next_perc90_conti_return'] >= 0.75, 1, 0)
    all_data['beat_miss_binary2'] = np.where(all_data['next_perc2_conti_return'] >= 0.75, 1, 0)
    all_data['beat_miss_binary3'] = np.where(all_data['next_perc3_conti_return'] >= 0.75, 1, 0)
    all_data['beat_miss_binary6'] = np.where(all_data['next_perc6_conti_return'] >= 0.75, 1, 0)
    # return all_data[['date', 'ticker', 'beat_miss_binary', 'beat_miss_multiclass']]
    return all_data[['date', 'ticker', 'beat_miss_binary30', 'beat_miss_binary60', 'beat_miss_binary90',
                     'beat_miss_binary2', 'beat_miss_binary3', 'beat_miss_binary6']]


if __name__ == "__main__":
    feat_df = pd.read_csv('data/all_data_historic.csv')

    # feat_df = feat_df[~(feat_df['ticker'].isin(remove_ticks))]
    final_df = main(feat_df, candle=30)
    # print(final_df['beat_miss_binary30'].value_counts(normalize=True))
    # print(final_df['beat_miss_multiclass'].value_counts(normalize=True))
    final_df.to_csv('data/target_v3.csv', index=False)
