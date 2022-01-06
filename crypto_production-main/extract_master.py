import pandas as pd
from binance.client import Client

from config import BINANCE_API_KEY, BINANCE_SECRET_KEY

DATA_COL = ['start_time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'Ignore']
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

completed = []
batch = 1

res = client.get_exchange_info()
ls = []
for dicts in res.get('symbols'):
    temp = dicts.get('symbol'), dicts.get('status'), dicts.get('baseAsset'), dicts.get('baseAssetPrecision'), \
           dicts.get('quoteAsset'), dicts.get('quotePrecision'), dicts.get('quoteAssetPrecision'), \
           dicts.get('baseCommissionPrecision'), dicts.get('quoteCommissionPrecision'), dicts.get('orderTypes'),\
           dicts.get('icebergAllowed'), dicts.get('ocoAllowed'), dicts.get('quoteOrderQtyMarketAllowed'), \
           dicts.get('isSpotTradingAllowed'), dicts.get('isMarginTradingAllowed'), dicts.get('permissions'), \
           dicts.get('filters')
    ls.append(temp)
col_names = ['ticker', 'status', 'baseAsset', 'baseAssetPrecision', 'quoteAsset', 'quotePrecision',
             'quoteAssetPrecision', 'baseCommissionPrecision', 'quoteCommissionPrecision', 'orderTypes',
             'icebergAllowed', 'ocoAllowed', 'quoteOrderQtyMarketAllowed', 'isSpotTradingAllowed',
             'isMarginTradingAllowed', 'permissions', 'filters']
pd.DataFrame(ls, columns=col_names).to_csv('master.csv', index=False)

