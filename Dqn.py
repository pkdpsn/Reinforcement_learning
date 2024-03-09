from gymnasium import Env
from stable_baselines3 import A2C,PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from config import DEFAULT_CONFIG
from Envs import rlenv
from utils import *
import pickle 
from datetime import datetime
from colorama import Fore , Back
import os 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from multiprocessing import freeze_support  # Import freeze_support

models_dir = "models/DQN"
logdir = "logs"
TIMESTEPS = 500000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def main():
    freeze_support()  # Call freeze_support() here
    num_cpu = 6
    env = SubprocVecEnv([lambda: rlenv() for _ in range(num_cpu)])  # Corrected usage of SubprocVecEnv

    model = DQN("MlpPolicy", env, learning_rate=0.00001, batch_size=64, gamma=1,exploration_fraction=0.80, verbose=2, tensorboard_log=logdir)
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="DQN", progress_bar=True)
    model.save(f"{models_dir}/DQN{TIMESTEPS}_{current_time}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)  # Changed model.get_env() to env
    obs = env.reset()

if __name__ == "__main__":
    main()  # Call the main function inside the if __name__ == '__main__' block


# mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)  # Changed model.get_env() to vec_env
# obs = vec_env.reset()

# # from stable_baselines3 import A2C   exploration_fraction=0.95, exploration_initial_eps=1.0, exploration_final_eps=0.01 ,

# # model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# # model.learn(total_timesteps=10_000, tb_log_name="first_run")
# # # Pass reset_num_timesteps=False to continue the training curve in tensorboard
# # # By default, it will create a new curve
# # # Keep tb_log_name constant to have continuous curve (see note below)
# # model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)
# # model.learn(total_timesteps=10_000, tb_log_name="third_run", reset_num_timesteps=False)
