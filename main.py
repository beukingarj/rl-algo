#%%
%load_ext autoreload
%autoreload 2

from extract_data import extract_data
from train_and_evaluate import train_and_evaluate
import torch as th
import numpy as np


#%%
data_kwargs = dict(
    symbols = list(['AAPL']),
    interval='1d', 
    delta = 365*30, 
    no_stocks=100,
    df = None
)

data = extract_data(**data_kwargs)
data.extract_stock_data()
data.extract_features_new()
data.split_and_scale()


#%%
import matplotlib.pyplot as plt
a = data.stock_data.pct_change(1).iloc[:20]
a
# plt.imshow(a)

#%%

# data = 1
a = train_and_evaluate(data, 'A2C')
# print(a.evaluate(dataset='train', verbose=0)[0])

for i in np.logspace(-3,-6,4):
    model_params = dict(
        learning_rate = i,
        policy_kwargs = dict(
                            activation_fn=th.nn.LeakyReLU,
                            net_arch = [256, dict(vf=[256], pi=[256])],
                            optimizer_class=th.optim.RMSprop,
                            ),
        
        ent_coef = 0.8,
        create_eval_env = True,
        # n_steps = 5,
        # train_freq = (4, "step")
        )

model = a.train(total_timesteps=100000, cont=False, model_params=model_params)