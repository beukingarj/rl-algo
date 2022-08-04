from itertools import groupby
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from ta import add_all_ta_features

class extract_data():
    def __init__(self, df=None):
        # self.extract_stock_data(no_stocks=3, delta=365*4,interval='1d') #640
        # self.extract_features()
        # self.split_and_scale()
        pass

    def extract_stock_data(self, symbols=None, interval='1d', delta = 365*6, no_stocks=100):
        if symbols == None:
            # table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            # symbols = list(np.sort(table[0].Symbol))
            # symbols.remove('AMCR')
            # symbols.remove('BF.B')
            # symbols.remove('BRK.B')  
            
            symbols = list(['BTC-EUR', 'SOL-EUR', 'NVDA']) # , 'AVAX-EUR''ETH-EUR', 'USDT-EUR', 'BNB-EUR', 'USDC-EUR', 'XRP-EUR', 'SOL-EUR', 'ADA-EUR'
            symbols = symbols[:no_stocks]
        
        stock_data_temp1 = yf.download(symbols, start=str(datetime.date.today()+datetime.timedelta(days=-delta)), interval = interval, group_by='ticker')
        if len(symbols)==1:
            stock_data_temp1.columns = pd.MultiIndex.from_product([symbols, stock_data_temp1.columns])
        self.symbols = np.array(stock_data_temp1.columns.get_level_values(0).unique())
        stock_data_temp1 = stock_data_temp1.swaplevel(axis=1)

        valid_days = ~np.all(np.isnan(stock_data_temp1),axis=1)
        stock_data_temp1 = stock_data_temp1.loc[valid_days,:]

        stock_data_temp1 = stock_data_temp1.loc[:,(['Open','High','Low','Close','Volume'])]

        idx_zero_i, idx_zero_j = np.where(stock_data_temp1.Volume==0)
        for i in range(len(idx_zero_i)):
            stock_data_temp1.Volume.iloc[idx_zero_i[i], idx_zero_j[i]] = np.nan
        stock_data_temp1.Volume.interpolate(inplace=True)
        self.stock_data = stock_data_temp1.swaplevel(axis=1)

    def extract_features(self):
        for i, symbol in enumerate(self.symbols):
            df_temp = self.stock_data.loc[:,symbol]
            # df_temp = df_temp[~np.all(np.isnan(df_temp),1)]
            df_temp.dropna(axis=0, inplace=True)
            self.add_features(df_temp)

            if i==0:
                self.df = pd.DataFrame(index=self.stock_data.index, columns=pd.MultiIndex.from_product([self.symbols, df_temp.columns]))
            
            df_temp.columns = pd.MultiIndex.from_product([[symbol], df_temp.columns])
            self.df.loc[:,(symbol,slice(None))] = df_temp

        # VIX = yf.download('^VIX', start=str(datetime.date.today()+datetime.timedelta(days=-self.delta)), interval = self.interval).shift(-1)
        # self.df['VIX'] = VIX.loc[VIX.index[np.isin(VIX.index, self.df.index)],'Open']
        # self.df.loc[:,'VIX'].interpolate(inplace=True)

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # self.df.drop(columns=self.df.columns[self.df.isnull().all(0)], inplace=True)
        # print(self.df.index[0],self.df.index[1])
        # self.df.dropna(inplace=True)
        # print(self.df.index[0],self.df.index[1])
        # self.df.to_csv('./data/df.csv')        

    def add_features(self, df):
        add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)
        
        cor_col = ['momentum_kama','volume_adi','volume_obv','volume_em','volume_sma_em',
            'volume_vwap','volume_nvi','volatility_bbm','volatility_bbh','volatility_bbl',
            'volatility_kcc','volatility_kch','volatility_kcl','volatility_dcl','volatility_dch',
            'volatility_dcm','volatility_atr','trend_sma_slow','trend_sma_fast','trend_ema_slow',
            'trend_ema_fast','trend_ichimoku_conv','trend_ichimoku_base','trend_ichimoku_a',
            'trend_ichimoku_b','trend_visual_ichimoku_a','trend_visual_ichimoku_b','others_cr']
        df[cor_col] = df[cor_col] / np.repeat(np.expand_dims(df.Open.values, 1),len(cor_col),1)
        df['momentum_stoch_rsi_kd'] = df.momentum_stoch_rsi_d - df.momentum_stoch_rsi_k
        df[['trend_psar_up', 'trend_psar_down']] = ((df[['trend_psar_up', 'trend_psar_down']]>0).values.astype(int))
        df[['Openpct','Highpct','Lowpct','Closepct','Volumepct']] = df[['Open','High','Low','Close','Volume']].pct_change()
        df.drop(columns=['High','Low','Close','Volume'], inplace=True)
        df.loc[:,'Open'] = df.loc[:, 'Open'].shift(-1)
        
        return df

    def split_and_scale(self):

        self.train = self.df.loc[:,(['BTC-EUR'],slice(None))]
        self.val = self.df.loc[:,('SOL-EUR',slice(None))]
        self.test = self.df.loc[:,('NVDA',slice(None))]

        self.train.dropna(axis=0, inplace=True)
        self.val.dropna(axis=0, inplace=True)
        self.test.dropna(axis=0, inplace=True)

        # self.train = self.train[~np.all(np.isnan(self.train),1)]
        # self.val = self.val[~np.all(np.isnan(self.val),1)]
        # self.test = self.test[~np.all(np.isnan(self.test),1)]
        # train_split_perc = 0.7
        # train_split = int(train_split_perc * self.df.shape[0])

        # self.train = self.df.iloc[:train_split,:].copy()
        # self.test = self.df.iloc[train_split:,:].copy()
        self.X_train = self.train.loc[:, ~np.isin(self.train.columns.get_level_values(1), ['Open'])]
        self.X_val = self.val.loc[:, ~np.isin(self.val.columns.get_level_values(1), ['Open'])]
        self.X_test = self.test.loc[:, ~np.isin(self.test.columns.get_level_values(1), ['Open'])]
        self.y_train = self.train.loc[:,(slice(None),'Open')]/self.train.loc[:,(slice(None),'Open')].iloc[0]*10000
        self.y_val = self.val.loc[:,(slice(None),'Open')]/self.val.loc[:,(slice(None),'Open')].iloc[0]*10000
        self.y_test = self.test.loc[:,(slice(None),'Open')]/self.test.loc[:,(slice(None),'Open')].iloc[0]*10000