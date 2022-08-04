#%%
from extract_data import extract_data
from train_and_evaluate import train_and_evaluate
import torch as th
import numpy as np


#%%
%load_ext autoreload
%autoreload 2


data = extract_data()
data.extract_stock_data()
data.extract_features()
data.split_and_scale()

#%%
%load_ext autoreload
%autoreload 2


# data = 1
a = train_and_evaluate(data, 'A2C')
# print(a.evaluate(dataset='train', verbose=0)[0])

for i in np.logspace(-3,-6,4):
    kwargs = dict(
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

model = a.train(total_timesteps=100000 , cont=False, **kwargs)