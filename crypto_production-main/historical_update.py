import numpy as np
import pandas as pd


def main(df, topk=3):
    # coins = df['ticker'].unique().tolist()
    df['latest_return'] = df.sort_values(by=['date',
                                             'ticker']).groupby(['ticker'])['close'].apply(
        lambda x: x.shift(-1) / x - 1)

    df['latest_index_return'] = df.sort_values(by=['date',
                                                   'ticker']).groupby(['ticker'])['index_close'].apply(
        lambda x: x.shift(-1) / x - 1)
    # print(df[['date','ticker','close','latest_return','index_close','latest_index_return']].head())
    df = df.sort_values(by=['date', 'per_prob_score'], ascending=False).groupby(['date']).head(topk)
    df = df.sort_values(by=['date', 'ticker'])
    return df


if __name__ == "__main__":
    historic_df = pd.read_csv('output/historical_predictions.csv')
    historic_df['date'] = pd.to_datetime(historic_df['date'])
    output = main(historic_df)
    output.to_csv('output/historical_output.csv', index=False)
    # print(historic_df.tail())
