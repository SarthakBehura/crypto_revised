import numpy as np
import pandas as pd
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.others import *
from ta.volatility import *
from ta import *
import time
from stockstats import StockDataFrame as sdf

import warnings

warnings.filterwarnings("ignore")


# not used
def get_RSI(df, col_name, intervals):
    """
    stockstats lib seems to use 'close' column by default so col_name
    not used here.
    This calculates non-smoothed RSI
    """
    try:
        df_ss = sdf.retype(df)
        for i in intervals:
            df['rsi_' + str(i)] = df_ss['rsi_' + str(i)]

            del df['close_-1_s']
            del df['close_-1_d']
            del df['rs_' + str(i)]

            df['rsi_' + str(intervals[0])] = rsi(df['close'], i, fillna=True)
    #         print("RSI with stockstats done")
    except Exception as e:
        print('Error in get_RSI function')
        raise e


def get_RSI_smooth(df, col_name, intervals):
    """
    Momentum indicator
    As per https://www.investopedia.com/terms/r/rsi.asp
    RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
    RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )
    E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
    http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Verified!
    """
    try:
        prev_avg_gain = np.inf
        prev_avg_loss = np.inf
        rolling_count = 0

        def calculate_RSI(series, period):
            nonlocal prev_avg_gain
            nonlocal prev_avg_loss
            nonlocal rolling_count

            curr_gains = series.where(series >= 0, 0)  # replace 0 where series not > 0
            curr_losses = np.abs(series.where(series < 0, 0))
            avg_gain = curr_gains.sum() / period  # * 100
            avg_loss = curr_losses.sum() / period  # * 100
            rsi = -1

            if rolling_count == 0:
                # first RSI calculation
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
                # print(rolling_count,"rs1=",rs, rsi)
            else:
                # smoothed RSI
                # current gain and loss should be used, not avg_gain & avg_loss
                rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                         (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))

            rolling_count = rolling_count + 1
            prev_avg_gain = avg_gain
            prev_avg_loss = avg_loss
            return rsi

        diff = df[col_name].diff()[1:]  # skip na
        for period in intervals:
            df['rsi_' + str(period)] = np.nan
            rolling_count = 0
            res = diff.rolling(period).apply(calculate_RSI, args=(period,), raw=False)
            df['rsi_' + str(period)][1:] = res

        # df.drop(['diff'], axis = 1, inplace=True)
    #         print_time("Calculation of RSI Done", stime)
    except Exception as e:
        print('Error in get_RSI_smooth function')
        raise e


def get_williamR(df, col_name, intervals):
    """
    both libs gave same result
    Momentum indicator
    """
    try:
        for i in intervals:
            df["wr_" + str(i)] = wr(df['high'], df['low'], df['close'], i, fillna=True)
    except Exception as e:
        print('Error in get_williamR function')
        raise e


def get_williamR_single(df, col_name, i):
    try:
        df[f"wr_{i}"] = williams_r(df['high'], df['low'], df['close'], i, fillna=True)
    except Exception as e:
        print('Error in get_williamR_single function')
        raise e


def get_mfi(df, intervals):
    """
    momentum type indicator
    """
    try:
        for i in intervals:
            df['mfi_' + str(i)] = money_flow_index(df['high'], df['low'], df['close'], df['volume'], n=i, fillna=True)
    except Exception as e:
        print('Error in get_mfi function')
        raise e


def get_mfi_single(df, i):
    """
    momentum type indicator
    """
    try:
        df[f'mfi_{i}'] = money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=i, fillna=True)
    except Exception as e:
        print('Error in get_mfi_single function')
        raise e


def get_SMA(df, col_name, intervals):
    """
    Momentum indicator
    """
    try:
        df_ss = sdf.retype(df)
        for i in intervals:
            df[col_name + '_sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
            del df[col_name + '_' + str(i) + '_sma']
    except Exception as e:
        print('Error in get_SMA function')
        raise e


def get_EMA(df, col_name, intervals):
    """
    Needs validation
    Momentum indicator
    """
    try:
        df_ss = sdf.retype(df)
        for i in intervals:
            df['ema_' + str(i)] = df_ss[col_name + '_' + str(i) + '_ema']
            del df[col_name + '_' + str(i) + '_ema']
    except Exception as e:
        print('Error in get_EMA function')
        raise e


def get_WMA(df, col_name, intervals, hma_step=0):
    """
    Momentum indicator
    """
    try:
        stime = time.time()
        if hma_step == 0:
            # don't show progress for internal WMA calculation for HMA
            print("Calculating WMA")

        def wavg(rolling_prices, period):
            weights = pd.Series(range(1, period + 1))
            return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

        temp_col_count_dict = {}
        for i in intervals:
            res = df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
            if hma_step == 0:
                df['wma_' + str(i)] = res
            elif hma_step == 1:
                if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                    temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
                else:
                    temp_col_count_dict['hma_wma_' + str(i)] = 0
                # after halving the periods and rounding, there may be two intervals with same value e.g.
                # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
                df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
            elif hma_step == 3:
                import re
                expr = r"^hma_[0-9]{1}"
                columns = list(df.columns)
                # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
                df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

        # if hma_step == 0:
        #     print_time("Calculation of WMA Done", stime)
    except Exception as e:
        print('Error in get_WMA function')
        raise e


def get_HMA(df, col_name, intervals):
    try:
        import re
        #         stime = time.time()
        #         print("Calculating HMA")
        expr = r"^wma_.*"

        if len(list(filter(re.compile(expr).search, list(df.columns)))) > 0:
            print("WMA calculated already. Proceed with HMA")
        else:
            #             print("Need WMA first...")
            get_WMA(df, col_name, intervals)

        intervals_half = np.round([i / 2 for i in intervals]).astype(int)

        # step 1 = WMA for interval/2
        # this creates cols with prefix 'hma_wma_*'
        get_WMA(df, col_name, intervals_half, 1)
        # print("step 1 done", list(df.columns))

        # step 2 = step 1 - WMA
        columns = list(df.columns)
        expr = r"^hma_wma.*"
        hma_wma_cols = list(filter(re.compile(expr).search, columns))
        rest_cols = [x for x in columns if x not in hma_wma_cols]
        expr = r"^wma.*"
        wma_cols = list(filter(re.compile(expr).search, rest_cols))

        df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,
                                                fill_value=0)
        intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
        for i, col in enumerate(hma_wma_cols):
            # print("step 3", col, intervals_sqrt[i])
            get_WMA(df, col, [intervals_sqrt[i]], 3)
        df.drop(columns=hma_wma_cols, inplace=True)
    #         print_time("Calculation of HMA Done", stime)
    except Exception as e:
        print('Error in get_HMA function')
        raise e


def get_TRIX(df, col_name, intervals):
    """
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    try:
        for i in intervals:
            df['trix_' + str(i)] = trix(df['close'], i, fillna=True)

    except Exception as e:
        print('Error in get_TRIX function')
        raise e


def get_TRIX_single(df, col_name, i):
    try:
        df[f'trix_{i}'] = trix(df['close'], i, fillna=True)
    except Exception as e:
        print('Error in get_TRIX_single function')
        raise e


def get_DMI(df, col_name, intervals):
    """
    trend indicator
    TA gave same/wrong result
    """
    try:
        df_ss = sdf.retype(df)
        for i in intervals:
            df['dmi_' + str(i)] = df_ss['adx_' + str(i) + '_ema']

        drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
                        'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
        expr1 = r'dx_\d+_ema'
        expr2 = r'adx_\d+_ema'
        import re
        drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
        drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
        df.drop(columns=drop_columns, inplace=True)
    #         print_time("Calculation of DMI done", stime)
    except Exception as e:
        print('Error in get_DMI function')
        raise e


def get_DMI_single(df, col_name, i):
    try:
        df_ss = sdf.retype(df)
        df[f'dmi_{i}'] = df_ss[f'adx_{i}_ema']
    except Exception as e:
        print('Error in get_DMI_single function')
        raise e


def get_CCI(df, col_name, intervals):
    try:
        for i in intervals:
            df['cci_' + str(i)] = cci(df['high'], df['low'], df['close'], i, fillna=True)
    except Exception as e:
        print('Error in get_CCI function')
        raise e


def get_CCI_single(df, col_name, i):
    try:
        df['cci_' + str(i)] = cci(df['high'], df['low'], df['close'], i, fillna=True)
    except Exception as e:
        print('Error in get_CCI_single function')
        raise e


def get_BB_MAV(df, col_name, intervals):
    """
    volitility indicator
    """
    try:
        for i in intervals:
            df['bb_' + str(i)] = bollinger_mavg(df['close'], n=i, fillna=True)
    except Exception as e:
        print('Error in get_BB_MAV function')
        raise e


def get_BB_MAV_single(df, col_name, i):
    try:
        df[f'bb_{i}'] = bollinger_mavg(df['close'], window=i, fillna=True)
    except Exception as e:
        print('Error in get_BB_MAV_single function')
        raise e


def get_CMO(df, col_name, intervals):
    """
    Chande Momentum Oscillator
    As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
    CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
    range = +100 to -100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """
    try:
        #         print("Calculating CMO")
        #         stime = time.time()
        def calculate_CMO(series, period):
            sum_gains = series[series >= 0].sum()
            sum_losses = np.abs(series[series < 0].sum())
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            return np.round(cmo, 3)

        diff = df[col_name].diff()[1:]  # skip na
        for period in intervals:
            df['cmo_' + str(period)] = np.nan
            res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
            df['cmo_' + str(period)][1:] = res

    #         print_time("Calculation of CMO Done", stime)
    except Exception as e:
        print('Error in get_CMO function')
        raise e


def get_CMO_single(df, col_name, i):
    try:
        def calculate_CMO(series, period):
            sum_gains = series[series >= 0].sum()
            sum_losses = np.abs(series[series < 0].sum())
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            return np.round(cmo, 3)

        diff = df[col_name].diff()[1:]  # skip na
        df[f'cmo_{i}'] = np.nan
        res = diff.rolling(i).apply(calculate_CMO, args=(i,), raw=False)
        df[f'cmo_{i}'][1:] = res

    except Exception as e:
        print('Error in get_CMO_single function')
        raise e


# not used. on close(12,16): +3, ready to use
def get_MACD(df):
    """
    Not used
    Same for both
    calculated for same 12 and 26 periods on close only!! Not different periods.
    creates colums macd, macds, macdh
    """
    try:
        #         stime = time.time()
        #         print("Calculating MACD")
        df_ss = sdf.retype(df)
        df['macd'] = df_ss['macd']

        del df['macd_']
        del df['close_12_ema']
        del df['close_26_ema']
    #         print_time("Calculation of MACD done", stime)
    except Exception as e:
        print('Error in get_MACD function')
        raise e


# not implemented. period 12,26: +1, ready to use
def get_PPO(df, col_name, intervals):
    """
    As per https://www.investopedia.com/terms/p/ppo.asp
    uses EMA(12) and EMA(26) to calculate PPO value
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    calculated for same 12 and 26 periods only!!
    """
    try:
        stime = time.time()
        print("Calculating PPO")
        df_ss = sdf.retype(df)
        df['ema_' + str(12)] = df_ss[col_name + '_' + str(12) + '_ema']
        del df['close_' + str(12) + '_ema']
        df['ema_' + str(26)] = df_ss[col_name + '_' + str(26) + '_ema']
        del df['close_' + str(26) + '_ema']
        df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100

        del df['ema_12']
        del df['ema_26']
    except Exception as e:
        print('Error in get_PPO function')
        raise e


def get_ROC(df, col_name, intervals):
    """
    Momentum oscillator
    As per implement https://www.investopedia.com/terms/p/pricerateofchange.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    ROC = (close_price_n - close_price_(n-1) )/close_price_(n-1) * 100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """
    try:
        #         stime = time.time()
        #         print("Calculating ROC")

        def calculate_roc(series, period):
            return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

        for period in intervals:
            df['roc_' + str(period)] = np.nan
            # for 12 day period, 13th day price - 1st day price
            res = df['close'].rolling(period + 1).apply(calculate_roc, args=(period,), raw=False)
            # print(len(df), len(df[period:]), len(res))
            df['roc_' + str(period)] = res

    #         print_time("Calculation of ROC done", stime)
    except Exception as e:
        print('Error in get_ROC function')
        raise e


def get_ROC_single(df, col_name, i):
    try:
        def calculate_roc(series, period):
            return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

        df[f'roc_{i}'] = np.nan
        # for 12 day period, 13th day price - 1st day price
        res = df['close'].rolling(i + 1).apply(calculate_roc, args=(i,), raw=False)
        df[f'roc_{i}'] = res

    except Exception as e:
        print('Error in get_ROC_single function')
        raise e


def get_DPO(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """
    try:
        #         stime = time.time()
        #         print("Calculating DPO")
        for i in intervals:
            df['dpo_' + str(i)] = dpo(df['close'], n=i)

    #         print_time("Calculation of DPO done", stime)
    except Exception as e:
        print('Error in get_DPO function')
        raise e


def get_DPO_single(df, col_name, i):
    try:
        df[f'dpo_{i}'] = dpo(df['close'], window=i)
    except Exception as e:
        print('Error in get_DPO_single function')
        raise e


def get_kst(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """
    try:
        #         stime = time.time()
        #         print("Calculating KST")
        for i in intervals:
            df['kst_' + str(i)] = kst(df['close'], i)

    #         print_time("Calculation of KST done", stime)
    except Exception as e:
        print('Error in get_kst function')
        raise e


def get_kst_single(df, col_name, i):
    try:
        df[f'kst_{i}'] = kst(df['close'], i)
    except Exception as e:
        print('Error in get_kst_single function')
        raise e


def get_CMF(df, col_name, intervals):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    try:
        #         stime = time.time()
        #         print("Calculating CMF")
        for i in intervals:
            df['cmf_' + str(i)] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], i, fillna=True)

    #         print_time("Calculation of CMF done", stime)
    except Exception as e:
        print('Error in get_CMF function')
        raise e


def get_force_index(df, intervals):
    try:
        #         stime = time.time()
        #         print("Calculating Force Index")
        for i in intervals:
            df['fi_' + str(i)] = force_index(df['close'], df['volume'], 5, fillna=True)

    #         print_time("Calculation of Force Index done", stime)
    except Exception as e:
        print('Error in get_force_index function')
        raise e


def get_force_index_single(df, i):
    try:
        df[f'fi_{i}'] = force_index(df['close'], df['volume'], 5, fillna=True)
    except Exception as e:
        print('Error in get_force_index_single function')
        raise e


def get_OBV(df, intervals):
    try:
        #         stime = time.time()
        #         print("Calculating On Balance Volume")
        for i in intervals:
            df['obv_' + str(i)] = on_balance_volume(df['close'], df['volume'], fillna=True)

    #         print_time("Calculation of On Balance Volume done", stime)
    except Exception as e:
        print('Error in get_OBV function')
        raise e


def get_EOM(df, col_name, intervals):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    try:
        #         stime = time.time()
        #         print("Calculating EOM")
        for i in intervals:
            df['eom_' + str(i)] = ease_of_movement(df['high'], df['low'], df['volume'], n=i, fillna=True)

    #         print_time("Calculation of EOM done", stime)
    except Exception as e:
        print('Error in get_EOM function')
        raise e


def get_EOM_single(df, col_name, i):
    try:
        df[f'eom_{i}'] = ease_of_movement(df['high'], df['low'], df['volume'], n=i, fillna=True)
    except Exception as e:
        print('Error in get_EOM_single function')
        raise e


# not used. +1
def get_volume_delta(df):
    try:
        #         stime = time.time()
        #         print("Calculating volume delta")
        df_ss = sdf.retype(df)
        df_ss['volume_delta']

    #         print_time("Calculation of Volume Delta done", stime)
    except Exception as e:
        print('Error in get_volume_delta function')
        raise e


# not used. +2 for each interval kdjk and rsv
def get_kdjk_rsv(df, intervals):
    try:
        #         stime = time.time()
        #         print("Calculating KDJK, RSV")
        df_ss = sdf.retype(df)
        for i in intervals:
            df['kdjk_' + str(i)] = df_ss['kdjk_' + str(i)]

    #         print_time("Calculation of EMA Done", stime)
    except Exception as e:
        print('Error in get_kdjk_rsv function')
        raise e


def get_kdjk_rsv_single(df, i):
    try:
        df_ss = sdf.retype(df)
        df[f'kdjk_{i}'] = df_ss[f'kdjk_{i}']
    except Exception as e:
        print('Error in get_kdjk_rsv_single function')
        raise e


def get_sdf_feats(df, intervals):
    try:
        df_ss = sdf.retype(df)
        for i in intervals:
            df[f'kdjk_{i}'] = df_ss[f'kdjk_{i}']
            df[f'rsv_{i}'] = df_ss[f'rsv_{i}']
            df[f'dmi_{i}'] = df_ss[f'adx_{i}_ema']
    except Exception as e:
        print('Error in sdf features function')
        raise e
