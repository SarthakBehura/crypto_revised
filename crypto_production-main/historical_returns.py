import numpy as np
import datetime
import pandas as pd


def main(prediction_file='output/historical_predictions.csv',
         ohlcv_file='data/all_data.csv'):
    pred_df = pd.read_csv(prediction_file)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df = pred_df[pred_df['date'] < '2021-10-27']
    # print(pred_df.head())
    ohlcv_df = pd.read_csv(ohlcv_file)
    ohlcv_df = ohlcv_df[['close_time', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    ohlcv_df.rename(columns={'close_time': 'date'}, inplace=True)
    ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'], unit='ms')
    ohlcv_df['date'] = ohlcv_df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    ls = []
    for idx, row in pred_df.iterrows():
        try:
            current_date = row['date']
            current_ticker = row['ticker']
            current_index = row['index_close']
            current_close = row['close']
            close_date = current_date + datetime.timedelta(hours=6)
            # print(current_date, close_date, current_ticker)
            new_open = ohlcv_df[(ohlcv_df['date'] == close_date) & (ohlcv_df['ticker'] == current_ticker)]['open'].values[0]
            new_index = ohlcv_df[(ohlcv_df['date'] == close_date) & (ohlcv_df['ticker'] == 'BTCUSDT')]['open'].values[0]
            ticker_return = (new_open / current_close) - 1
            index_return = (new_index / current_index) - 1

            temp = current_date, ticker_return, index_return
            ls.append(temp)
        except Exception as e:
            print(current_date, close_date, current_ticker)
            continue
    ls = pd.DataFrame(ls, columns=['date', 'return', 'index_return'])
    ls['relative_returns'] = ls['return'] - ls['index_return']

    # print((ls['return'] > 0).mean())
    print(f"Beat Ratio for every insight: {(ls['relative_returns'] > 0).mean()}")
    temp = ls.groupby(['date'])['relative_returns'].mean().reset_index()
    # print(temp.head())
    print(f"Average Accuracy: {(temp['relative_returns'] > 0).mean()}")
    print(f"Average Relative Returns: {round(temp['relative_returns'].mean()*100, 4)}")

    temp = ls.groupby(['date'])['return'].mean().reset_index()
    print(f"Average Returns: {round(ls['return'].mean() * 100, 4)}")
    # ls.to_csv('historical_returns.csv', index=False)


if __name__ == "__main__":
    main()
