#%%
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3 import A2C, DQN, TD3
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, plot_curves
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
import tensorflow as tf
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np
import gym
import os
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import skimage
import time
from os.path import exists
from IPython import display

from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from skimage.measure import label
import pandas as pd

import datetime

#%%
%load_ext autoreload
%autoreload 2

from extract_data import extract_data
data = extract_data()
data.extract_stock_data()
data.stock_data