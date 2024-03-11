import gymnasium as gym
from stable_baselines3 import PPO
import os


models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


env = gym.make('LunarLander-v2') 
env.reset()

model = PPO('MlpPolicy', env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")