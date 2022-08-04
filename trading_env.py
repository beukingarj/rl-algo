import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import skimage
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, X, y):
        assert X.ndim == 2

        self.seed()
        self.X = X
        self.y = y
        # self.train = train
        # self.prices, self.signal_features = self._process_data()
        self.dates = self.X.index
        self.date_len = self.dates.shape[0]
        self.stocks_symbols = list(self.y.columns.get_level_values(0).unique())
        self.no_stocks = len(self.stocks_symbols)
        self.no_features = self.X.loc[:,(self.stocks_symbols,slice(None))].shape[1]
        # self.no_features = self.X[np.array(self.X.columns.get_level_values(0).unique())[[0,-1]]].shape[1]
        
        # spaces
        # self.action_space = spaces.Box(low = -1, high = 1,shape = (1,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.no_features,))

        # episode
        # if self.train==False:
        self._start_tick = -1
        self._end_tick = self.date_len - 2
        self._done = None
        self._current_tick = None
        self._last_trade_price = None
        self._shares_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.balance = None
        self.start_balance = float(10000) #np.array([10000],dtype=np.float64)
        self._balance_history = None
        self.transcosts = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # selection = np.random.choice(self.stocks_symbols)
        # self.prices = self.y[selection]
        # self.signal_features = self.X[[selection,'VIX']]
        # print("selection: ",selection)
        # self.signal_features = self.X[[np.random.choice(self.stocks_symbols),'VIX']]
        self.prices = self.y
        self.signal_features = self.X

        self._done = False
        self._current_tick = self._start_tick
        self.update_time()
        self.start = True

        self._last_trade_price = np.zeros(1)
        self.shares = float(0)
        # self._position_history = (self.window_size * [None]) + [self.shares]
        # self._position_history = np.concatenate([np.nan*np.ones([self.window_size,self.no_stocks]), self.shares[np.newaxis,...]])
        self._shares_history = list() #self.shares[np.newaxis,...]
        self._total_reward = 0
        self.kas = self.start_balance #.copy()
        self.balance = self.start_balance #.copy()
        self.transcosts = float(0.5)
        # self._total_profit = 10000  # unit
        # self._balance_history = (self.window_size * [None]) + [self.balance]
        self._balance_history = [self.balance]
        self.step_reward = float(0)
        self.action_list = list()
        
        self._first_rendering = True
        self.history = {}
        
        return self._get_observation()

    def update_time(self):
        # self._prev_price = self.prices.loc[self._current_time, :]
        self._current_tick += 1
        self._current_time = self.dates[self._current_tick]
        self._current_price = float(self.prices.loc[self._current_time, :])
        self._next_price = float(self.prices.loc[self.dates[self._current_tick+1], :])

    def step(self, action):
        self.action_list.append(action)
        
        if self.start == False:
            self.update_time()

        if self._current_tick == self._end_tick:
            self._done = True
            action = 0
        
        self._calculate_reward(action)
        
        self._shares_history.append(self.shares)
        self._balance_history.append(self.balance)
        observation = self._get_observation()
        info = dict(
            date = self._current_time.strftime('%Y-%m-%d'),
            actions = action,
            price = int(self._current_price),
            total_reward = float(np.round(self._total_reward,2)),
            step_reward = float(np.round(self.step_reward,2)),
            kas = int(self.kas),
            balance = int(self.balance),
            shares = float(np.round(self.shares,2)),
        )
        # print(info)
        self._update_history(info)

        self.start = False

        return observation, self.step_reward, self._done, info


    def _get_observation(self):
        # a = self.signal_features[(self._current_tick-self.window_size):self._current_tick,:]

        # a = self.balance / self._current_price
        # b = self._current_price
        # c = self.shares
        d = np.array(self.signal_features.loc[self._current_time,:],dtype="float32")
        # e = np.concatenate([a,b,c,d])
        return d

    def _calculate_reward(self, action):
        self.balance = self.kas + self._current_price * self.shares
        self.virt_shares = (self.balance - self.transcosts) / self._current_price

        if action==0 and self.shares>0:
            self.kas += float((self._current_price * self.shares) - self.transcosts)
            self.shares = float(0)
        
        if action==1 and self.kas>0:
            self.virt_shares = float(0)
            self.kas = float(0)
            self.shares = (self.balance - self.transcosts) / self._current_price
            
        self.update_profit(action)

    def update_profit(self, action):
        
        b = skimage.measure.label(np.array(self.action_list)+1, connectivity=1)
        x = np.sum(b==b[-1])
        beta = 1 - 1 / (1 + np.exp(-x/4 + 5))
        # beta = 1
        gamma = - 0.1 / (1 + np.exp(-x/4 + 5))
        # gamma = 0
        
        if action==1:
            next_balance = self._next_price * self.shares
            self.step_reward = beta * (next_balance - self.balance) / self.balance + gamma

        elif action==0:
            next_balance = self._next_price * self.virt_shares
            self.step_reward = - beta * (next_balance - self.balance) / self.balance + gamma

        # if self._done == True:
        #     if np.mean(self.action_list)==0 or np.mean(self.action_list)==1:
        #         self.step_reward = -1e6
        
        # self.step_reward /= float(abs(self.prices.diff()).sum())
        # self.step_reward *= 10000
        self._total_reward += self.step_reward      

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self, mode='human'):
        print("Account balance: {:.2f}".format(self.balance))
        eind_weighted = np.sum((self.start_balance)/self.prices.iloc[0,:]*self.prices.iloc[-2,:])
        print("Account balance weighted prices: {:.2f}".format(eind_weighted))
        
        y = relativedelta(self.prices.index[-1], self.prices.index[0]).years
        m = relativedelta(self.prices.index[-1], self.prices.index[0]).months / 12
        d = relativedelta(self.prices.index[-1], self.prices.index[0]).days / 365
        total_years = y+m+d
        
        print("Rendement per jaar: {:.2f}%".format((((self.balance/self.start_balance)**(1/total_years)-1)*100)))

        plt.figure(figsize=[10,10])
        
        print("Relative reward against weighted prices: {:.1f}%".format((self._balance_history[-1]/eind_weighted*100-100)))
        plt.plot(self._balance_history,'b')
        plt.plot(np.sum(((self.start_balance)/self.prices.iloc[0,:])*self.prices[:-1],1).values,'r')
        # for i in range(self.no_stocks):
        #     plt.plot((self.start_balance/self.prices.iloc[0,i]*self.prices.iloc[:,i]).values)
        plt.legend(['balance','weighted prices']+self.stocks_symbols)

    def render(self, mode='human'):
        self.render_all()