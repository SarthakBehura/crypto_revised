BINANCE_API_KEY = "x4bPxAATtZwrR7DlNs15a8ye8T1PGCYRiOuKGH44ahlSOmxZ7yWkFPEfn05VT9wq"
BINANCE_SECRET_KEY = "AzF5Ur0yXCyjV4xmSC2VwKsAwOXUTSyYeXPlgCKKlscdMuNjhHBBnbNv8sT1eC5a"

TICKERS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'BCCUSDT', 'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT',
           'EOSUSDT', 'TUSDUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VENUSDT',
           'NULSUSDT', 'VETUSDT', 'PAXUSDT', 'BCHABCUSDT', 'BCHSVUSDT', 'USDCUSDT', 'LINKUSDT', 'WAVESUSDT',
           'BTTUSDT', 'USDSUSDT', 'ONGUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT', 'BATUSDT', 'XMRUSDT',
           'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'NANOUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT',
           'MITHUSDT', 'MATICUSDT', 'ATOMUSDT', 'TFUELUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'USDSBUSDT',
           'GTOUSDT', 'ERDUSDT', 'DOGEUSDT', 'DUSKUSDT', 'ANKRUSDT', 'WINUSDT', 'COSUSDT', 'NPXSUSDT', 'COCOSUSDT',
           'MTLUSDT', 'TOMOUSDT', 'PERLUSDT', 'DENTUSDT', 'MFTUSDT', 'KEYUSDT', 'STORMUSDT', 'DOCKUSDT', 'WANUSDT',
           'FUNUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'BUSDUSDT', 'BEAMUSDT']

NUM_FEATURES = 144
FEATURES = ['number_of_trades', 'index_close', 'avg_trade_size', 'close_returns_30_days', 'index_returns_30_days',
            'close_rel_returns_30_days', 'close_returns_60_days', 'close_rel_returns_60_days', 'close_returns_90_days',
            'index_returns_90_days', 'close_returns_180_days', 'close_returns_270_days', 'index_returns_270_days',
            'close_rel_returns_270_days', 'close_std_1m', 'close_std_1m_sortino', 'close_std_2m_sortino',
            'close_std_3m', 'close_std_3m_sortino', 'close_std_6m', 'close_std_9m', 'close_std_9m_sortino',
            'close_volitality_1m', 'close_volitality_2m', 'close_volitality_9m', 'close_sharpe_1m', 'close_sharpe_2m',
            'close_sharpe_3m', 'close_sharpe_6m', 'close_sharpe_9m', 'close_sharpe_2m_sortino',
            'close_sharpe_3m_sortino', 'close_sharpe_6m_sortino', 'close_sharpe_9m_sortino', 'close_q1_return',
            'close_q3_return', 'close_max_drawdown_2months', 'close_max_drawdown_3months', 'close_max_drawdown_6months',
            'close_max_drawdown_9months', 'close_max_drawdown_12months', 'close_tracking_error_2_m',
            'close_tracking_error_3_m', 'close_beta_2m', 'close_beta_3m', 'close_information_ratio_1_m',
            'close_information_ratio_2_m', 'close_information_ratio_3_m', 'close_downside_deviation_9m',
            'close_r_square_1_m', 'close_r_square_2_m', 'close_r_square_3_m', 'CUMLOGRET_1', 'CUMPCTRET_1', 'NVI_1',
            'OBV', 'OHLC4', 'PDIST', 'PPO_12_26_9', 'PPOs_12_26_9', 'PSL_12', 'PVI_1', 'PVOL', 'UO_7_14_28', 'VIDYA_14',
            'VTXM_14', 'VWMA_10', 'WCP', 'WILLR_14', 'WMA_10', 'ZL_EMA_10', 'ABER_ZG_5_15', 'ABER_SG_5_15',
            'ABER_ATR_5_15', 'ACCBM_20', 'AD', 'AMATe_LR_2', 'OBV_min_2', 'OBV_max_2', 'OBVe_4', 'OBVe_12', 'AOBV_SR_2',
            'AROONOSC_14', 'BBM_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'STOCHRSIk_14_14_3_3', 'SUPERT_7_3.0', 'SUPERTd_7_3.0', 'SWMA_10', 'T3_10_0.7', 'TEMA_10',
            'THERMOma_20_2_0.5', 'TRIMA_10', 'TRIXs_30_9', 'TRUERANGE_1', 'UI_14', 'EMA_10', 'FISHERT_9_1',
            'FISHERTs_9_1', 'FWMA_10', 'HA_open', 'HA_close', 'HILO_13_21', 'HL2', 'HLC3', 'HMA_10', 'HW-UPPER',
            'HWMA_0.2_0.1_0.1', 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'PVT', 'PWMA_10', 'QQE_14_5_4.236', 'RMA_10',
            'ROC_10', 'RSI_14', 'RVGIs_14_4', 'RVI_14', 'SINWMA_14', 'SMA_10', 'SMI_5_20_5', 'CG_10', 'CHOP_14_1_100',
            'CKSPl_10_1_9', 'CKSPs_10_1_9', 'CMO_14', 'COPC_11_14_10', 'LDECAY_5', 'DEC_1', 'DEMA_10', 'DCL_20_20',
            'DCU_20_20', 'INC_1', 'KAMA_10_2_30', 'KCBe_20_2', 'KCUe_20_2', 'KST_10_15_20_30_10_10_10_15', 'MAD_30',
            'MIDPOINT_2']

SCALER = 'resources/mm_scaler_crypto_v2.pkl'
MODEL = 'resources/cnn_long_v3_floss'
