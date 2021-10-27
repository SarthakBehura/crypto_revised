import os
import glob
import datetime

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from data import data_fetch
from crypto_model import model

sched = BlockingScheduler()


@sched.scheduled_job('interval', seconds=60)
def timed_job():
    # get data for latest coins
    files_path = os.path.join('output', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)

    data = pd.read_csv((files[0]))
    coins = data['ticker'].unique().tolist()
    coins.append('BTCUSDT')
    df = data_fetch.data_fetch(start_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                               interval=1,
                               coins=coins)

    all_data = df[['close_time', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    index_data = all_data[all_data['ticker'] == 'BTCUSDT'][['close_time', 'close']].copy()
    index_data.rename(columns={'close': 'index_close'}, inplace=True)
    all_data = all_data[all_data['ticker'] != 'BTCUSDT'].copy()
    all_data = all_data.merge(index_data, on='close_time', how='left')
    all_data.rename(columns={'close_time': 'date'}, inplace=True)
    all_data['date'] = pd.to_datetime(all_data['date'], unit='ms')
    all_data['date'] = all_data['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    all_data = all_data[all_data['date'] == all_data.date.max()]

    data = data.merge(all_data, on=['ticker'], how='left')
    data = data[['date_x', 'ticker', 'Probability', 'per_prob_score', 'close_x', 'close_y',
                 'index_close_x', 'index_close_y']]
    data['close_y'] = data['close_y'].astype(np.float64)
    data['index_close_y'] = data['index_close_y'].astype(np.float64)
    data['latest_return'] = data.apply(lambda x: (x['close_x'] / x['close_y']) - 1, axis=1)
    data['latest_index_return'] = data.apply(lambda x: (x['index_close_x'] / x['index_close_y']) - 1, axis=1)
    data = data[['date_x', 'ticker', 'Probability', 'close_x', 'index_close_x', 'per_prob_score',
                 'latest_return', 'latest_index_return']].copy()
    data.rename(columns={'date_x': 'date',
                         'close_x': 'close',
                         'index_close_x': 'index_close'}, inplace=True)
    data.to_csv(files[0], index=False)
    print(data.tail())


@sched.scheduled_job('interval', seconds=60 * 60 * 6)
def scheduled_job():
    final_output = model.driver(lag=20)


sched.start()
