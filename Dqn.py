from gymnasium import Env
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from config import DEFAULT_CONFIG
from Envs import rlenv
from utils import *
import pickle 
from datetime import datetime
from colorama import Fore, Back
import os 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn
import torch.nn.functional as f
from multiprocessing import freeze_support  # Import freeze_support

models_dir = "models/DQN"
logdir = "logs"
TIMESTEPS = 200000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log training information
        if self.num_timesteps % 1000 == 0:
            print(f"Time steps: {self.num_timesteps}, Mean episode reward: {self.training_env.get_attr('episode_rewards')}")
        
        return True  # Continue training
# policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                      net_arch=dict(pi=[32,32], vf=[32,32]))
def main():
    freeze_support()  # Call freeze_support() here
    num_cpu = 6
    env =rlenv()# SubprocVecEnv([lambda: rlenv() for _ in range(num_cpu)])  # Corrected usage of SubprocVecEnv
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    net_arch = [200, 1000, 500,25]
    )
    # policy_kwargs=policy_kwargs ,
    model = DQN("MlpPolicy", env, learning_rate=0.0003, learning_starts= 5000,tau=1,batch_size=2048, gamma=1, exploration_fraction=0.80,verbose=1, tensorboard_log=logdir)
    print(model.policy)
    callback = CustomCallback()
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"DQN {current_time}", progress_bar=True,log_interval=10)
    # model.save(f"{models_dir}/DQN{TIMESTEPS}_{current_time}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)  # Changed model.get_env() to env
    obs = env.reset()

if __name__ == "__main__":
    main()
