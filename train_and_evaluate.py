import os
from os.path import exists
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3 import A2C, DQN, TD3
from trading_env import TradingEnv
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback

class train_and_evaluate():
    def __init__(self, data, model_type):
        self.model_type = model_type
        self.log_dir = './logs/best_model/'+model_type

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

        self.env_train = TradingEnv(X=data.X_train, y=data.y_train)
        self.DummyEnv_train = DummyVecEnv([lambda: self.env_train])
        self.env_train_norm = VecMonitor(VecNormalize(self.DummyEnv_train, norm_obs=True, norm_reward=False, clip_obs=10.))
        self.env_val = TradingEnv(X=data.X_val, y=data.y_val)
        self.DummyEnv_val = DummyVecEnv([lambda: self.env_val])
        self.data = data

    def load_best_model(self, **kwargs):
        
        if exists(os.path.join(self.log_dir, 'best_model.zip')):
            self.env_train_norm = VecMonitor(VecNormalize.load(self.log_dir+'/best_env', self.DummyEnv_train))
            self.env_val_norm = VecMonitor(VecNormalize.load(self.log_dir+'/best_env', self.DummyEnv_val))
            model = globals()[self.model_type].load(self.log_dir+'/best_model.zip', **kwargs)
        else:
            model = None
        return model

    def train(self, total_timesteps, cont, model_params):
        if cont:
            model = self.load_best_model(model_params)
        else:
            model = self.set_model(model_params)

        model.set_env(self.env_train_norm)

        # callback = EvalCallback(
        #     self.env_train_norm,
        #     best_model_save_path=self.log_dir,
        #     log_path='./logs/results',
        #     eval_freq=500, 
        #     n_eval_episodes=1,
        #     deterministic=True)

        callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=self.log_dir, DummyEnv_train=self.DummyEnv_train, DummyEnv_val=self.DummyEnv_val, data=self.data)
        model.learn(total_timesteps=total_timesteps)

        return model

    def evaluate(self, model=None, dataset='test', verbose=1):
        if model==None:
            model = self.load_best_model()

        if model == None:
            total_reward = 0
            info = [dict()]
        else:
            if dataset=='train':
                env = self.env_train_norm
            if dataset=='test':
                env = self.env_test_norm
                env.training = False
                env.norm_reward = False
                # print("reward_test: {}".format(evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)[0]/0.3))

            env.training = False
            env.norm_reward = False

            obs = env.reset()
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                if done:
                    total_reward = info[0]['total_reward']
                    if verbose:
                        env.render_all()
                    break

        return total_reward, info

    def set_model(self, model_params):
        model_params.update(dict(
            policy = 'MlpPolicy', # MlpLstmPolicy
            env = self.env_train_norm,
            verbose=0,
            # lr_schedule='linear_schedule',
            tensorboard_log="./logs/tensorboard",
        ))

        if self.model_type=='A2C':
            model = A2C(**model_params)

        elif self.model_type=='DQN':
            model = DQN(**model_params)

        return model