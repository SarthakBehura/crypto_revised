import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed

import ti_utils


def apply_parallel(df_grouped, func):
    ret_lst = Parallel(n_jobs=mp.cpu_count()-1)(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(ret_lst)


def beta_ric(temp, i):
    sec_re = 'returns_1_days*100'
    ind_re = 'index_returns_1_days*100'
    return temp[[sec_re, ind_re]].rolling((i * 30)).cov().unstack()[sec_re] \
               [ind_re] / temp[ind_re].rolling((i * 30)).var()


def _drawdown(vec):
    maximums = np.maximum.accumulate(vec)
    drawdowns = 1 - vec / maximums
    return np.max(drawdowns)


def max_draw_down(x, i):
    return x.rolling((i * 30), min_periods=1).apply(_drawdown)


def corr(x, y, i):
    return (x.rolling(window=(i * 30)).corr(y)) ** 2


def lag_feat(df):
    try:
        my_days = list(range(1, 10))
        other_days = [1, 2, 3, 7, 15, 30, 60]
        days = my_days + [15, 30, 60]
        for i in days:
            if i in other_days:
                df[f'RI_lag_{i}d'] = (df['close'] / df['close'].shift(i)) - 1
                df[f'Index_RI_lag_{i}d'] = (df['index_close'] / df['index_close'].shift(i)) - 1
                df[f'rel_lag_{i}d'] = df[f'RI_lag_{i}d'] - df[f'Index_RI_lag_{i}d']
            if i in my_days:
                df[f'rel_lag_1d_lag_{i}'] = df['rel_lag_1d'].shift(i)
                df[f'RI_lag_1d_lag_{i}'] = df['RI_lag_1d'].shift(i)
    except Exception as e:
        print('Error in calculating lag features')


def common_features_ric(df):
    try:
        months = [0, 1, 2, 3, 6, 9, 12]
        df['returns_1_days*100'] = df['RI_lag_1d'] * 100
        df['index_returns_1_days*100'] = df['Index_RI_lag_1d'] * 100
        for i in months:
            if i == 0:
                df['RI_returns_1_days'] = (df['close'] / df['close'].shift(1)) - 1
                df['index_returns_1_days'] = (df['index_close'] / df['index_close'].shift(i * 30)) - 1
                df['RI_rel_returns_1_days'] = df['RI_returns_1_days'] - df['index_returns_1_days']
                continue

            df[f'RI_returns_{i * 30}_days'] = (df['close'] / df['close'].shift(i * 30)) - 1
            df[f'index_returns_{i * 30}_days'] = (df['index_close'] / df['index_close'].shift(i * 30)) - 1
            df[f'RI_rel_returns_{i * 30}_days'] = df[f'RI_returns_{i * 30}_days'] - df[f'index_returns_{i * 30}_days']

            df[f'RI_std_{i}m'] = df['RI_returns_1_days'].rolling(window=(30 * i)).std(ddof=0) * np.sqrt(252)
            df[f'RI_std_{i}m_sortino'] = df['RI_returns_1_days'].rolling(window=(30 * i)).apply(
                lambda x: x[x < 0].std(ddof=0)) * np.sqrt(252)

            df[f'RI_downside_deviation_{i}m'] = (df[f'RI_std_{i}m_sortino']).copy()
            df[f'RI_tracking_error_{i}_m'] = (df['rel_lag_1d'].rolling(window=(30 * i)).std(ddof=0)) * np.sqrt(252)
            df[f'RI_sharpe_{i}m'] = df[f'RI_returns_{i * 30}_days'] / df[f'RI_std_{i}m']
            df[f'RI_sharpe_{i}m_sortino'] = df[f'RI_returns_{i * 30}_days'] / (df[f'RI_std_{i}m_sortino'])
            df[f'RI_max_drawdown_{i}months'] = max_draw_down(df['close'], i)
            df[f'RI_information_ratio_{i}_m'] = (df[f'RI_rel_returns_{i * 30}_days'] / df[f'RI_tracking_error_{i}_m'])
            df[f'RI_r_square_{i}_m'] = corr(df['RI_lag_1d'], df['Index_RI_lag_1d'], i)
            df[f'RI_beta_{i}m'] = beta_ric(df, i)
            df[f'RI_volitality_{i}m'] = (df[f'RI_std_{i}m']) ** 2
    except Exception as e:
        print('Error in calculating fund common features')
        raise e


def fund_features(df):
    df = df.sort_values(by=['date'])
    # req = df.Date.max() - datetime.timedelta(days=500)
    # temp = df[df['Date'] > req].copy()
    temp = df.copy()
    lag_feat(temp)
    common_features_ric(temp)

    df['RI_returns_360_days'] = (df['close'] / df['close'].shift(360)) - 1
    df['RI_avg_rolling_return_1_year'] = df['RI_returns_360_days'].rolling(720, min_periods=1).mean()
    df['RI_fy1_return'] = (df['close'].shift(360) / df['close'].shift(720)) - 1

    quarters = [1, 2, 3, 4, 5, 6]
    for i in quarters:
        df[f'RI_q{i}_return'] = (df['close'].shift(90 * (i - 1)) / df['close'].shift(90 * i)) - 1

    regexp = '^(?!.*_DROP)'
    df = df.merge(temp, on=['date', 'ticker'], how='left', suffixes=('', '_DROP')).filter(regex=regexp)

    return df


def technicals_with_lag(df):
    """
    tech_with_lag : calculates technical features
    Args:1. DataFrame with high , open low, close, volume, vwap
    Returns: DataFrame with the technical values
    """
    try:
        data = df.sort_values(by=['date'])

        for i in range(1, 16):
            ti_utils.get_CCI_single(data, 'close', i)
            ti_utils.get_CMO_single(data, 'close', i)
            ti_utils.get_DPO_single(data, 'close', i)
            ti_utils.get_mfi_single(data, i)
            ti_utils.get_ROC_single(data, 'close', i)
            ti_utils.get_TRIX_single(data, 'close', i)
            ti_utils.get_williamR_single(data, 'close', i)
            ti_utils.get_kst_single(data, 'close', i)
            ti_utils.get_BB_MAV_single(data, 'close', i)

        ti_utils.get_RSI_smooth(data, 'close', range(1, 16))
        # SDF features: KDJK, RSV, DMI
        ti_utils.get_sdf_feats(data, range(1, 16))
        return data
    except Exception as e:
        print(f'Error in generating technical lag features: {e}')


def time_features(data):
    data['hour'] = data['date'].dt.hour
    data['minute'] = data['date'].dt.minute
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['day_name'] = data['date'].dt.day_name()
    data['quarter'] = data['date'].dt.quarter
    data['is_month_end'] = data['date'].dt.is_month_end
    data['is_month_start'] = data['date'].dt.is_month_start
    data['is_year_start'] = data['date'].dt.is_year_start
    data['is_year_end'] = data['date'].dt.is_year_end


def interaction_features(data):
    data['close_by_open'] = data['close'] / data['open']
    data['high_by_open'] = data['high'] / data['open']
    data['high_by_close'] = data['high'] / data['close']
    data['low_by_open'] = data['low'] / data['open']
    data['low_by_close'] = data['low'] / data['close']
    data['high_minus_low'] = (data['high'] - data['low']) / data['close']
    data['close_minus_open'] = (data['close'] - data['open']) / data['close']