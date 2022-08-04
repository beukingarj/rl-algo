#%%
import yfinance as yf
from itertools import groupby
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from ta import add_all_ta_features

symbols = list(['BTC-EUR', 'SOL-EUR', 'NVDA']) # , 'AVAX-EUR''ETH-EUR', 'USDT-EUR', 'BNB-EUR', 'USDC-EUR', 'XRP-EUR', 'SOL-EUR', 'ADA-EUR'

delta = 365
interval = '1d'

stock_data_temp1 = yf.download(symbols, start=str(datetime.date.today()+datetime.timedelta(days=-delta)), interval = interval, group_by='ticker')
symbols = np.array(stock_data_temp1.columns.get_level_values(0).unique())
if len(symbols)==1:
    stock_data_temp1.columns = pd.MultiIndex.from_product([symbols, stock_data_temp1.columns])

valid_days = ~np.all(np.isnan(stock_data_temp1),axis=1)
stock_data_temp1 = stock_data_temp1.loc[valid_days,:]

stock_data_temp1 = stock_data_temp1.loc[:,(symbols,['Open','High','Low','Close','Volume'])]

idx_zero_i, idx_zero_j = np.where(stock_data_temp1.loc[:,(slice(None),'Volume')]==0)
for i in range(len(idx_zero_i)):
    stock_data_temp1.loc[:,(slice(None),'Volume')].iloc[idx_zero_i[i], idx_zero_j[i]] = np.nan
stock_data_temp1.loc[:,(slice(None),'Volume')].interpolate(inplace=True)

#%%
dhr = yf.Ticker(symbols[0])
info = dhr.info
info.keys()