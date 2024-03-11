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

from multiprocessing import freeze_support  # Import freeze_support

models_dir = "models/DQN"
logdir = "logs"
TIMESTEPS = 500000

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

def main():
    freeze_support()  # Call freeze_support() here
    num_cpu = 6
    env =rlenv()# SubprocVecEnv([lambda: rlenv() for _ in range(num_cpu)])  # Corrected usage of SubprocVecEnv

    model = DQN("MlpPolicy", env, learning_rate=0.00001, batch_size=64, gamma=1, exploration_fraction=0.80, verbose=2, tensorboard_log=logdir)
    callback = CustomCallback()
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="DQN", progress_bar=True)
    model.save(f"{models_dir}/DQN{TIMESTEPS}_{current_time}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)  # Changed model.get_env() to env
    obs = env.reset()

if __name__ == "__main__":
    main()
