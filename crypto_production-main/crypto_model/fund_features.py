import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

from math import ceil

# predefined values
returns_days = [-30, 1, 30, 60, 90, 180, 270, 360]
months = [1, 2, 3, 6, 9, 12]
sqrt = [252, 252, 252, 252, 252, 252]
map_sqrt = dict(zip(months, sqrt))
quarters = range(1, 9)
years = [1, 2]
avg_return_months = range(1, 25)


def data_resample(data_df):
    """
    data_resample : resamples data for 15 minute each in day.
    input : dataframe
    output : dataframe with resampled data
    """
    # resample
    data_df.index = data_df.date
    data_df = data_df.groupby(['ticker']).apply(lambda x: x.resample('6H', on='date').mean().ffill()).reset_index(
        [1]).reset_index()
    data_df.index = data_df.date
    data_df = data_df.groupby(['ticker']).apply(lambda x: x.resample('6H', on='date').mean().bfill()).reset_index(
        [1]).reset_index()
    return data_df


# def data_resample_ric(data_df):
#     """
#     data_resample : resamples data for 15 minute each in day.
#     input : dataframe
#     output : dataframe with resampled data
#     """
#     # resample
#     data_df.index = data_df.date
#     data_df = data_df.resample('6H').mean().ffill().reset_index()
#     data_df.index = data_df.date
#     data_df = data_df.resample('6H').mean().bfill().reset_index()
#     return data_df

    # common_features


def common_features(df, returns, stdev, map_sqrt, col, index_col):
    """
        common_features : generates commonly used features such as returns, std dev for differnt days
        input : dataframe,list for the returns days, months for std dev, dict of months to sqrt mapping, column name, index column name
        output : dataframe with returns and std dev values
    """
    returns_compute = returns.copy()
    returns_compute.remove(-30)
    for i in returns_compute:
        df[f'{col}_returns_{i}_days'] = df.sort_values(by=['ticker', 'date']).groupby(['ticker'])[col].transform(
            lambda x: (x / x.shift(i)) - 1)
        df[f'index_returns_{i}_days'] = df.sort_values(by=['ticker', 'date']).groupby('ticker')[index_col].transform(
            lambda x: (x / x.shift(i)) - 1)
        df[f'{col}_rel_returns_{i}_days'] = df[f'{col}_returns_{i}_days'] - df[f'index_returns_{i}_days']
    for i in stdev:
        std_df = df.set_index('date').groupby('ticker').rolling(window=(30 * i)).std(ddof=0)[
            f'{col}_returns_1_days'].reset_index().rename(columns={f'{col}_returns_1_days': f'{col}_std_{i}m'})
        std_df_sortino = df.set_index('date').groupby('ticker')[f'{col}_returns_1_days'].rolling(window=(30 * i)).apply(
            lambda x: x[x < 0].std(ddof=0)).reset_index().rename(
            columns={f'{col}_returns_1_days': f'{col}_std_{i}m_sortino'})
        df = df.merge(std_df, on=['date', 'ticker'], how='left')
        df = df.merge(std_df_sortino, on=['date', 'ticker'], how='left')
        df[f'{col}_std_{i}m'] = (df[f'{col}_std_{i}m'] * np.sqrt(map_sqrt[i]))
        df[f'{col}_std_{i}m_sortino'] = (df[f'{col}_std_{i}m_sortino'] * np.sqrt(map_sqrt[i]))
    return df


def common_features_ric(df, col, index_col):
    """
        common_features : generates commonly used features such as returns, std dev for differnt days
        input : dataframe,list for the returns days, months for std dev, dict of months to sqrt mapping, column name, index column name
        output : dataframe with returns and std dev values
    """
    df.sort_values(by=['date'], inplace=True)
    returns_compute = returns_days.copy()
    returns_compute.remove(-30)
    for i in returns_compute:
        df[f'{col}_returns_{i}_days'] = df[[col]].apply(lambda x: (x / x.shift(i)) - 1)
        df[f'index_returns_{i}_days'] = df[[index_col]].apply(lambda x: (x / x.shift(i)) - 1)
        df[f'{col}_rel_returns_{i}_days'] = df[f'{col}_returns_{i}_days'] - df[f'index_returns_{i}_days']
    for i in months:
        df[f'{col}_std_{i}m'] = df[f'{col}_returns_1_days'].rolling(window=(30 * i)).std(ddof=0)
        df[f'{col}_std_{i}m_sortino'] = df[f'{col}_returns_1_days'].rolling(window=(30 * i)).apply(
            lambda x: x[x < 0].std(ddof=0))
        df[f'{col}_std_{i}m'] = (df[f'{col}_std_{i}m'] * np.sqrt(map_sqrt[i]))
        df[f'{col}_std_{i}m_sortino'] = (df[f'{col}_std_{i}m_sortino'] * np.sqrt(map_sqrt[i]))


# next 30 days returns
def next_30_days(df, col, index_col):
    """
    next_30_days : generates next 30 days returns
    input : dataframe, column name, index column name
    output : dataframe with returns values
    """
    df[f'{col}_returns_next_30_days'] = df.sort_values(by=['ticker', 'date']).groupby(['ticker'])[col].transform(
        lambda x: (x.shift(-30) / x) - 1)
    df[f'index_returns_next_30_days'] = df.sort_values(by=['ticker', 'date']).groupby('ticker')[index_col].transform(
        lambda x: (x.shift(-30) / x) - 1)
    df[f'rel_returns_next_30_days'] = df[f'{col}_returns_next_30_days'] - df[f'index_returns_next_30_days']


def next_30_days_ric(df, col, index_col):
    """
    next_30_days : generates next 30 days returns
    input : dataframe, column name, index column name
    output : dataframe with returns values
    """
    df.sort_values(by='date', inplace=True)
    df[f'{col}_returns_next_30_days'] = df[[col]].apply(lambda x: (x.shift(-30) / x) - 1)
    df['index_returns_next_30_days'] = df[[index_col]].apply(lambda x: (x.shift(-30) / x) - 1)
    df['rel_returns_next_30_days'] = df[f'{col}_returns_next_30_days'] - df[f'index_returns_next_30_days']


# volitality
def volitality(df, months, col):
    """
    volitality : calculates volatility
    input : dataframe,list of months,column name,
    output : dataframe with volitality values
    """
    for i in months:
        df[f'{col}_volitality_{i}m'] = (df[f'{col}_std_{i}m']) ** 2


def volitality_ric(df, col):
    """
    volitality : calculates volatility
    input : dataframe,list of months,column name,
    output : dataframe with volitality values
    """
    for i in months:
        df[f'{col}_volitality_{i}m'] = (df[f'{col}_std_{i}m']) ** 2


# sharpe   
def sharpe(df, months, col):
    """
    sharpe : calculates sharpe
    input : dataframe,list of months,column name,
    output : dataframe with sharpe values
    """
    for i in months:
        df[f'{col}_sharpe_{i}m'] = df[f'{col}_returns_{(i * 30)}_days'] / df[f'{col}_std_{i}m']


def sharpe_ric(df, col):
    """
    sharpe : calculates sharpe
    input : dataframe,list of months,column name,
    output : dataframe with sharpe values
    """
    for i in months:
        df[f'{col}_sharpe_{i}m'] = df[f'{col}_returns_{(i * 30)}_days'] / df[f'{col}_std_{i}m']


# sortino_sharpe
def sortino_sharpe(df, months, col):
    """
    sortino_sharpe : calculates sortino sharpe
    input : dataframe,list of months,column name,
    output : dataframe with sortino sharpe values
    """
    for i in months:
        df[f'{col}_sharpe_{i}m_sortino'] = df[f'{col}_returns_{(i * 30)}_days'] / (df[f'{col}_std_{i}m_sortino'])


def sortino_sharpe_ric(df, col):
    """
    sortino_sharpe : calculates sortino sharpe
    input : dataframe,list of months,column name,
    output : dataframe with sortino sharpe values
    """
    for i in months:
        df[f'{col}_sharpe_{i}m_sortino'] = df[f'{col}_returns_{(i * 30)}_days'] / (df[f'{col}_std_{i}m_sortino'])


# quarterly returns        
def quarterly_returns(df, quarters, col):
    """
    quarterly_returns : calculates quarterly returns
    input : dataframe,list of quarters,column name,
    output : dataframe with quarterly returns values
    """
    for i in quarters:
        df[f'{col}_q{i}_return'] = df.sort_values(by=['ticker', 'date']).groupby(['ticker'])[col].transform(
            lambda x: (x.shift(90 * (i - 1)) / x.shift(90 * i)) - 1)


def quarterly_returns_ric(df, col):
    """
    quarterly_returns : calculates quarterly returns
    input : dataframe,list of quarters,column name,
    output : dataframe with quarterly returns values
    """
    for i in quarters:
        df[f'{col}_q{i}_return'] = df[[col]].apply(lambda x: (x.shift(90 * (i - 1)) / x.shift(90 * i)) - 1)


# yearly_returns
def yearly_returns(df, years, col):
    """
    yearly_returns : calculates yearly returns
    input : dataframe,list of years,column name,
    output : dataframe with yearly returns values
    """
    for i in years:
        df[f'{col}_fy{i}_return'] = df.sort_values(by=['ticker', 'date']).groupby(['ticker'])[col].transform(
            lambda x: (x.shift(360 * (i - 1)) / x.shift(360 * i)) - 1)


def yearly_returns_ric(df, col):
    """
    yearly_returns : calculates yearly returns
    input : dataframe,list of years,column name,
    output : dataframe with yearly returns values
    """
    for i in years:
        df[f'{col}_fy{i}_return'] = df[[col]].apply(lambda x: (x.shift(360 * (i - 1)) / x.shift(360 * i)) - 1)


# avg_rolling_return_1yr
def avg_rolling_return_1yr(df, col):
    """
    yearly_returns : calculates avg rollling 1 year return
    input : dataframe,column name
    output : dataframe with avg rollling 1 year return returns values
    """
    df[f'{col}_avg_rolling_return_1_year'] = df.sort_values(by=['ticker', 'date']).groupby('ticker')[
        f'{col}_returns_360_days'].rolling(720, min_periods=1).mean().reset_index(0, drop=True)


def avg_rolling_return_1yr_ric(df, col):
    """
    yearly_returns : calculates avg rollling 1 year return
    input : dataframe,column name
    output : dataframe with avg rollling 1 year return returns values
    """
    df[f'{col}_avg_rolling_return_1_year'] = df[f'{col}_returns_360_days'].rolling(720, min_periods=1).mean()


# max_draw_down
def _drawdown(vec):
    maximums = np.maximum.accumulate(vec)
    drawdowns = 1 - vec / maximums
    return np.max(drawdowns)


def max_draw_down(df, months, col):
    """
    max_draw_down : calculates max drawdown
    input : dataframe,list of months,column name
    output : dataframe with avg rollling 1 year return returns values
    """
    for i in months:
        df[f'{col}_max_drawdown_{i}months'] = df.groupby('ticker')[col].rolling((i * 30), min_periods=1).apply(
            _drawdown).reset_index(drop=True)


def max_draw_down_ric(df, col):
    """
    max_draw_down : calculates max drawdown
    input : dataframe,list of months,column name
    output : dataframe with avg rollling 1 year return returns values
    """
    for i in months:
        df[f'{col}_max_drawdown_{i}months'] = df[col].rolling((i * 30), min_periods=1).apply(_drawdown)


# tracking_error
def tracking_error(df, months, col):
    """
    tracking_error : calculates tracking error
    input : dataframe,list of months,column name
    output : dataframe with tracking error values
    """
    new_months = months.copy()
#     new_months.append(-1)
    for i in new_months:
        if i == -1:
            risk = df.set_index('date').groupby('ticker').rolling(window=(-i * 30)).std(ddof=0)[
                f'{col}_rel_returns_1_days'].shift(i * 30).reset_index().rename(
                columns={f'{col}_rel_returns_1_days': 'std_returns_diff'})
            df = df.merge(risk, on=['date', 'ticker'], how='left')
            df[f'{col}_tracking_error_{i}_m'] = df['std_returns_diff'] * np.sqrt(252)
            df = df.drop(columns=['std_returns_diff'])
        else:
            risk = df.set_index('date').groupby('ticker').rolling(window=(i * 30)).std(ddof=0)[
                f'{col}_rel_returns_1_days'].reset_index().rename(
                columns={f'{col}_rel_returns_1_days': 'std_returns_diff'})
            df = df.merge(risk, on=['date', 'ticker'], how='left')
            df[f'{col}_tracking_error_{i}_m'] = df['std_returns_diff'] * np.sqrt(252)
            df = df.drop(columns=['std_returns_diff'])
    return df


def tracking_error_ric(df, col):
    """
    tracking_error : calculates tracking error
    input : dataframe,list of months,column name
    output : dataframe with tracking error values
    """
    new_months = months.copy()
    new_months.append(-1)
    for i in new_months:
        if i == -1:
            df['std_returns_diff'] = df[f'{col}_rel_returns_1_days'].rolling(window=(-i * 30)).std(ddof=0).shift(i * 30)
            df[f'{col}_tracking_error_{i}_m'] = df['std_returns_diff'] * np.sqrt(252)
            df = df.drop(columns=['std_returns_diff'])
        else:
            df['std_returns_diff'] = df[f'{col}_rel_returns_1_days'].rolling(window=(i * 30)).std(ddof=0)
            df[f'{col}_tracking_error_{i}_m'] = df['std_returns_diff'] * np.sqrt(252)
            df = df.drop(columns=['std_returns_diff'])
    return df


# upcapture_downcapture
def upcapture_downcapture(df, returns_days, col):
    """
    upcapture_downcapture : calculates upcapture and downcapture values
    input : dataframe,list of returns days,column name
    output : dataframe with upcapture and down capture values
    """
    new_returns_days = returns_days.copy()
    new_returns_days.remove(-30)
    new_returns_days.remove(1)
    ls = pd.DataFrame()
    for i in new_returns_days:
        ls1 = []
        for j in df['date'].unique():
            j = pd.to_datetime(j)
            mask = ((df['date'] > (j - datetime.timedelta(days=i))) & (df['date'] <= j))
            subset_df = df.loc[mask]
            df_pos = subset_df[subset_df['index_returns_1_days'] > 0]
            up_df = df_pos.groupby('ticker')[f'{col}_returns_1_days'].sum() / df_pos.groupby('ticker')[
                f'index_returns_1_days'].sum()
            up_df = pd.DataFrame(up_df).reset_index([0]).rename(columns={0: f'{col}_up_capture_{i}_days'})
            df_neg = subset_df[subset_df['index_returns_1_days'] < 0]
            down_df = df_neg.groupby('ticker')[f'{col}_returns_1_days'].sum() / df_neg.groupby('ticker')[
                f'index_returns_1_days'].sum()
            down_df = pd.DataFrame(down_df).reset_index([0]).rename(columns={0: f'{col}_down_capture_{i}_days'})
            temp_df = up_df.merge(down_df, on=['ticker'], how='outer')
            temp_df['date'] = pd.to_datetime(j)
            ls1.append(temp_df)
        if len(ls) == 0:
            ls = pd.concat(ls1)
        else:
            temp_ls = pd.concat(ls1)
            ls = ls.merge(temp_ls, on=['ticker', 'date'], how='left')
    return ls


def _capture(row, return_days, df, col):
    mask = ((df['date'] > (row['date'] - datetime.timedelta(days=return_days))) & (df['date'] <= row['date']))
    subset_df = df.loc[mask]
    df_pos = subset_df[subset_df['index_returns_1_days'] > 0]
    up_df = df_pos[f'{col}_returns_1_days'].sum() / df_pos[f'index_returns_1_days'].sum()
    df_neg = subset_df[subset_df['index_returns_1_days'] < 0]
    down_df = df_neg[f'{col}_returns_1_days'].sum() / df_neg[f'index_returns_1_days'].sum()


def upcapture_downcapture_ric(df, col):
    """
    upcapture_downcapture : calculates upcapture and downcapture values
    input : dataframe,list of returns days,column name
    output : dataframe with upcapture and down capture values
    """
    new_returns_days = returns_days.copy()
    new_returns_days.remove(-30)
    new_returns_days.remove(1)
    ls = pd.DataFrame()
    for i in new_returns_days:
        ls1 = []
        for j in df['date'].unique():
            j = pd.to_datetime(j)
            mask = ((df['date'] > (j - datetime.timedelta(days=i))) & (df['date'] <= j))
            subset_df = df.loc[mask]
            df_pos = subset_df[subset_df['index_returns_1_days'] > 0][['ticker',
                                                                       f'{col}_returns_1_days',
                                                                       'index_returns_1_days']]
            df_pos[f'{col}_up_capture_{i}_days'] = df_pos[f'{col}_returns_1_days'].sum() / df_pos['index_returns_1_days'].sum()
            df_neg = subset_df[subset_df['index_returns_1_days'] < 0][['ticker',
                                                                       f'{col}_returns_1_days',
                                                                       'index_returns_1_days']]
            df_neg[f'{col}_down_capture_{i}_days'] = df_neg[f'{col}_returns_1_days'].sum() / df_neg['index_returns_1_days'].sum()
            temp_df = df_pos.merge(df_neg, on=['ticker'], how='outer')
            temp_df['date'] = j
            ls1.append(temp_df)
        if len(ls) == 0:
            ls = pd.concat(ls1)
        else:
            temp_ls = pd.concat(ls1)
            ls = ls.merge(temp_ls, on=['ticker', 'date'], how='left')
    return ls


# beta feature 1 year
def beta_feature(df, months, col):
    """
    beta_feature : calculates beta values
    input : dataframe,list of months,column name
    output : dataframe with beta values
    """
    df['returns_1_days*100'] = df[f'{col}_returns_1_days'] * 100
    df['index_returns_1_days*100'] = df['index_returns_1_days'] * 100
    for i in months:
        df = df.set_index('date')
        df_cov = \
            df.groupby(['ticker'])[[f'returns_1_days*100', 'index_returns_1_days*100']].rolling(
                (i * 30)).cov().unstack()[
                f'returns_1_days*100']['index_returns_1_days*100']
        df_var = df.groupby(['ticker'])['index_returns_1_days*100'].rolling((i * 30)).var()
        df_cov = df_cov.reset_index()
        df_cov.rename(columns={'index_returns_1_days*100': 'covar'}, inplace=True)
        df_var = df_var.reset_index()
        df_var.rename(columns={'index_returns_1_days*100': 'var'}, inplace=True)
        merge_df = df_cov.merge(df_var)
        merge_df[f'{col}_beta_{i}m'] = merge_df['covar'] / merge_df['var']
        df = df.reset_index()
        df = df.merge(merge_df[['date', 'ticker', f'{col}_beta_{i}m']].copy(), on=['date', 'ticker'])
    df = df.drop(columns=['returns_1_days*100', 'index_returns_1_days*100'])
    return df


def beta_feature_ric(df, col):
    """
    beta_feature : calculates beta values
    input : dataframe,list of months,column name
    output : dataframe with beta values
    """
    df['returns_1_days*100'] = df[f'{col}_returns_1_days'] * 100
    df['index_returns_1_days*100'] = df['index_returns_1_days'] * 100
    for i in months:
        df = df.set_index('date')
        df_cov = df[[f'returns_1_days*100', 'index_returns_1_days*100']].rolling((i * 30)).cov() \
            .unstack()[f'returns_1_days*100']['index_returns_1_days*100']
        df_var = df['index_returns_1_days*100'].rolling((i * 30)).var()
        df_cov = df_cov.reset_index()
        df_cov.rename(columns={'index_returns_1_days*100': 'covar'}, inplace=True)
        df_var = df_var.reset_index()
        df_var.rename(columns={'index_returns_1_days*100': 'var'}, inplace=True)
        merge_df = df_cov.merge(df_var)
        merge_df[f'{col}_beta_{i}m'] = merge_df['covar'] / merge_df['var']
        df = df.reset_index()
        df = df.merge(merge_df[['date', f'{col}_beta_{i}m']], on=['date'])
    df = df.drop(columns=['returns_1_days*100', 'index_returns_1_days*100'])
    return df


# information ratio 1 year
def information_ratio_1yr(df, months, col):
    """
    information_ratio_1yr : calculates information ratios values
    input : dataframe,list of months,column name
    output : dataframe with information ratio values
    """
    for i in months:
        df[f'{col}_information_ratio_{i}_m'] = (
                df[f'{col}_rel_returns_{(i * 30)}_days'] / df[f'{col}_tracking_error_{i}_m'])


def information_ratio_1yr_ric(df, col):
    """
    information_ratio_1yr : calculates information ratios values
    input : dataframe,list of months,column name
    output : dataframe with information ratio values
    """
    for i in months:
        df[f'{col}_information_ratio_{i}_m'] = (
                df[f'{col}_rel_returns_{(i * 30)}_days'] / df[f'{col}_tracking_error_{i}_m'])


# downside_deviation
def downside_deviation(df, months, col):
    """
    downside_deviation : calculates downside deviation values
    input : dataframe,list of months,column name
    output : dataframe with downside deviation values
    """
    for i in months:
        df[f'{col}_downside_deviation_{i}m'] = (df[f'{col}_std_{i}m_sortino'])


def downside_deviation_ric(df, col):
    """
    downside_deviation : calculates downside deviation values
    input : dataframe,list of months,column name
    output : dataframe with downside deviation values
    """
    for i in months:
        df[f'{col}_downside_deviation_{i}m'] = (df[f'{col}_std_{i}m_sortino'])


# r square
def correlation(x, i, col):
    """
    correlation : calculates correlation of returns and index returns
    input : dataframe,rolling window ,column name
    output : dataframe with correlated values
    """
    return pd.DataFrame(x[f'{col}_returns_1_days'].rolling(window=(i * 30)).corr(x['index_returns_1_days']))


def r_square(df, months, col):
    """
    r_square : calculates r_square is square of correlated values
    input : dataframe,list of months ,column name
    output : dataframe with squared correlated values
    """
    for i in months:
        df[f'{col}_r_square_{i}_m'] = df.sort_values(by=['ticker', 'date']).groupby(['ticker'])[
            [f'{col}_returns_1_days', 'index_returns_1_days']].apply(correlation, i=i, col=col)
        df[f'{col}_r_square_{i}_m'] = df[f'{col}_r_square_{i}_m'] ** 2


def r_square_ric(df, col):
    """
    r_square : calculates r_square is square of correlated values
    input : dataframe,list of months ,column name
    output : dataframe with squared correlated values
    """
    for i in months:
        df[f'{col}_r_square_{i}_m'] = df[f'{col}_returns_1_days'].rolling(window=(i * 30)).corr(df['index_returns_1_days'])
        df[f'{col}_r_square_{i}_m'] = df[f'{col}_r_square_{i}_m'] ** 2


# generate_features
def generate_features(funds_nav_table, column,
                      index_column):
    """
    generate_features : final function to generate all features
    input : dataframe,list of returns days,list of months ,dict of months to sqrt mapping,
            list of quarters, list of years, list of avg returns for months,column name, index column name
    output : dataframe with all features 
    """
    # prep and resample data
    data_df = data_resample(funds_nav_table)
    print('Data Resampled')
    # functions
    data_df = common_features(data_df, returns_days, months, map_sqrt, column, index_column)
    print("Common Features built")
    # next 30 days returns
    next_30_days(data_df, column, index_column)
    print("Next 30 days features built")
    # volitality
    volitality(data_df, months, column)
    print("Volatility computed")
    # sharpe    
    sharpe(data_df, months, column)
    print("Sharpe computed")
    # sortino sharpe
    sortino_sharpe(data_df, months, column)
    print("Sortino computed")
    # quarterly returns
    quarterly_returns(data_df, quarters, column)
    print("Quarterly returns computed")
    # yearly returns
    yearly_returns(data_df, years, column)
    print("Yearly returns computed")
    # avg rolling 1 year returns
    avg_rolling_return_1yr(data_df, column)
    print("Rolling return computed")
    # max draw down
    max_draw_down(data_df, months, column)
    print("Drawdown computed")
    # tracking error
    data_df = tracking_error(data_df, months, column)
    print("Tracking error computed")
    data_df = data_df.rename(columns={f'{column}tracking_error_-1_m': f"{column}_tracking_error_next_1_m"})
    # upcapture downcapture
    upcapture_downcapture_df = upcapture_downcapture(data_df, returns_days, column)
    print("Upcapture Downcapture computed")
    data_df = data_df.merge(upcapture_downcapture_df, on=['ticker', 'date'], how='left')
    # beta feature
    data_df = beta_feature(data_df, months, column)
    print("Beta computed")
    # information ratio 
    information_ratio_1yr(data_df, months, column)
    print("Information Ratio computed")
    # downside deviation
    downside_deviation(data_df, months, column)
    print("Downside deviation computed")
    # r square
    r_square(data_df, months, column)
    print("R Squared computed")
    return data_df


def generate_crypto(tickers):
    for ticker in tickers:
        yield ticker


def multiprocessing(equities_ids):
    try:
        batch_size = 1
        num_batches = ceil(len(equities_ids) / batch_size)
        chunks = []
        for i in range(num_batches):
            chunks.append(equities_ids[i * batch_size: (i + 1) * batch_size])
        pool_size = (2 * mp.cpu_count()) - 1
        batches_of_funds = chunks
        m = mp.Manager()
        result_queue = m.Queue()
        pool = mp.pool.ThreadPool(pool_size)
        thread_pool_counter = 1
        monthly_df = pd.DataFrame()
        return batches_of_funds
    except Exception as e:
        print(f'Error in multiprocessing: {e}')
        raise e


def _feature_generator(output_q,
                       tickers,
                       data_df: pd.DataFrame,
                       column='close',
                       index_column='index_close'):
    ls = []
    tick = tickers[0]
    # for tick in tickers:
    data = data_df[data_df['ticker'] == tick].copy()
    # Resampling
#     data = data_resample_ric(data)
#     data['ticker'] = tick
    # Common features
    common_features_ric(data, column, index_column)
    print('Common features built')
    # Next 30 day returns
    next_30_days_ric(data, column, index_column)
    print('Next 30 day done')
    # volitality
    volitality_ric(data, column)
    print('Volatility done')
    # sharpe
    sharpe_ric(data, column)
    print('Sharpe computed')
    # sortino sharpe
    sortino_sharpe_ric(data, column)
    print('Sortino computed')
    # quarterly returns
    quarterly_returns_ric(data, column)
    print('Quarterly returns computed')
    # yearly returns
    yearly_returns_ric(data, column)
    print('Yearly returns computed')
    # avg rolling 1 year returns
    avg_rolling_return_1yr_ric(data, column)
    # max draw down
    max_draw_down_ric(data, column)
    print('Drawdown computed')
    # tracking error
    data = tracking_error_ric(data, column)
    data = data.rename(columns={f'{column}tracking_error_-1_m': f"{column}_tracking_error_next_1_m"})
    print('Tracking error computed')
    # upcapture downcapture
    # upcapture_downcapture_df = upcapture_downcapture_ric(data, column)
    # data = data.merge(upcapture_downcapture_df, on=['ticker', 'date'], how='left')
    # beta feature
    data = beta_feature_ric(data, column)
    print('Beta computed')
    data['ticker'] = tick
    # information ratio
    information_ratio_1yr_ric(data, column)
    print('Information ratio computed')
    # downside deviation
    downside_deviation_ric(data, column)
    print('Downside deviation computed')
    # r square
    r_square_ric(data, column)
    print('R squared computed')
    print(f'All Features for RICS {tick} complete')
    output_q.put(data)


def feature_generator(batches_of_funds,
                      df: pd.DataFrame,
                      column='close',
                      index_column='index_close'):
    pool_size = (5 * mp.cpu_count()) - 1
    m = mp.Manager()
    result_queue = m.Queue()
    pool = mp.pool.ThreadPool(pool_size)
    thread_pool_counter = 1
    output_df = pd.DataFrame()
    while len(batches_of_funds) > 0:
        njobs = min(pool_size, len(batches_of_funds))
        # Generate a list of arguments required for the threads
        batch_args = [(result_queue, mnemonics_chunk, df, column, index_column) for
                      mnemonics_chunk in batches_of_funds[0:njobs]]
        print(f"Triggering thread pool {thread_pool_counter} with {len(batch_args)} threads")
        pool.starmap(_feature_generator, batch_args)
        # Wait for the threads to finish working and put results into the queue
        for i in range(njobs):
            process_results = result_queue.get()
            output_df = output_df.append(process_results)
        batches_of_funds = batches_of_funds[njobs:]
        thread_pool_counter += 1
        # endTime = datetime.datetime.now()
    print(f"All features built with shape {output_df.shape}")
    return output_df


if __name__ == "__main__":
    df = pd.read_csv('data/training.csv')
    df = df[['start_time', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    index_data = df[df['ticker'] == 'BTCUSDT'][['start_time', 'close']].copy()
    index_data.rename(columns={'close': 'index_close'}, inplace=True)
    df = df[df['ticker'] != 'BTCUSDT'].copy()
    df = df.merge(index_data, on='start_time', how='left')
    print(df.shape)
    print(df.isnull().mean())
    df.rename(columns={'start_time': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    batches_of_funds = multiprocessing(df['ticker'].unique().tolist())
    # data
    print("Starting Parallel Processes")
    # completeDF = monthly_data_all(mongodb, equities_ids, days)
    complete_dF = feature_generator(batches_of_funds, df)

    complete_dF.to_csv(f'output/all_features_crypto.csv', index=False)

    print('Job complete')
