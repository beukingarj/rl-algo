import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from stable_baselines3 import A2C, DQN, TD3
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from evaluate_policy import evaluate_policy
from IPython import display

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, DummyEnv_train, DummyEnv_val, data, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.DummyEnv_train = DummyEnv_train
        self.DummyEnv_val = DummyEnv_val
        self.data = data
        self.best_model_path = os.path.join(log_dir, 'best_model.zip')
        self.best_env_path = os.path.join(log_dir, 'best_env')
        self.best_mean_reward = -np.inf
        self.reward_train_list = list()
        self.reward_val_list = list()
        self.num_timesteps_list = list()

    def _init_callback(self) -> None:
        if exists(self.best_model_path) and exists(self.best_env_path):
            model = A2C.load(self.best_model_path)

            env_train = VecMonitor(VecNormalize.load(self.best_env_path, self.DummyEnv_train))
            env_val = VecMonitor(VecNormalize.load(self.best_env_path, self.DummyEnv_val))
            env_train.training = False
            env_val.training = False
            env_train.norm_reward = False
            env_val.norm_reward = False
            self.total_reward_val_comp = evaluate_policy(model, env_val, n_eval_episodes=1, deterministic=True)[0]/data.X_val.shape[0]
            self.total_reward_train_comp = evaluate_policy(model, env_train, n_eval_episodes=1, deterministic=True)[0]/data.X_train.shape[0]
            # self.total_reward_comp, _ = a.evaluate(dataset='train', verbose=0)

            self.reward_train_list.append(float(self.total_reward_train_comp))
            self.reward_val_list.append(float(self.total_reward_val_comp))
            self.num_timesteps_list.append(float(0))
        else:
            self.total_reward_val_comp = -np.inf
            self.total_reward_train_comp = -np.inf

        # Create folder if needed
        if self.best_model_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.best_env_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

        display.clear_output(wait=True)
        plt.figure(figsize=(10,10))

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          self.model.save(os.path.join(self.log_dir, 'best_model_{}'.format(self.num_timesteps)))
          self.model.env.save(os.path.join(self.log_dir, 'best_env_{}'.format(self.num_timesteps)))

          env_val = VecMonitor(VecNormalize.load(os.path.join(self.log_dir, 'best_env_{}'.format(self.num_timesteps)), self.DummyEnv_val))
          self.model.env.training = False
          env_val.training = False
          self.model.env.norm_reward = False
          env_val.norm_reward = False

          reward_train = evaluate_policy(self.model, self.model.env, n_eval_episodes=1, deterministic=True)[0]/self.data.X_train.shape[0]
          reward_val = evaluate_policy(self.model, env_val, n_eval_episodes=1, deterministic=True)[0]/self.data.X_val.shape[0]

          self.model.env.training = True
          self.model.env.norm_reward = True
          # mean_reward, info = a.evaluate(model=self.model, dataset='train', verbose=0)
          
          self.reward_train_list.append(float(reward_train))
          self.reward_val_list.append(float(reward_val))
          self.num_timesteps_list.append(self.num_timesteps)
          
          plt.clf()
          plt.plot(self.num_timesteps_list,self.reward_val_list,'b',label='val')
          plt.plot(self.num_timesteps_list,self.reward_train_list,'g',label='train')
          plt.legend()
          plt.grid('on')
          # plt.axhline(self.total_reward_comp,C='r',label='highest')
          display.clear_output(wait=True)
          display.display(plt.gcf())
          
          # print(f"Num timesteps: {self.num_timesteps} -- best reward (this run): {self.best_mean_reward:.2f} -- best reward (overall): {self.total_reward_comp:.2f} -- test reward: {mean_reward:.2f}")
          #  -- test balance (331700): {info[0]['balance']:.0f}

          
          
          # if mean_reward_test > self.best_mean_reward:
          #     self.best_mean_reward = mean_reward_test

          if reward_val > self.total_reward_val_comp and reward_train > reward_val:
              self.total_reward_val_comp = reward_val
              self.model.save(os.path.join(self.log_dir, 'best_model.zip'))
              self.model.env.save(os.path.join(self.log_dir, 'best_env'))

        return True