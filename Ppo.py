from stable_baselines3 import PPO , A2C , DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
from multiprocessing import freeze_support
from Envs2 import rlenv
import threading




models_dir = "models/PPO"
logdir = "logs"
TIMESTEPS = 50000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def main():
    freeze_support()
    num_cpu = 6
    env =rlenv()# SubprocVecEnv([lambda: rlenv() for _ in range(num_cpu)])
    i=0 
    model = PPO("MlpPolicy", env,batch_size= 2048,n_epochs = 100,gamma=1,ent_coef=0.0001 ,verbose=1, tensorboard_log=logdir)
    print(model.policy)
    ## make a model of A2C in next line
    # model = A2C("MlpPolicy", env,gamma=1,ent_coef=0.0001 ,verbose=2, tensorboard_log=logdir)
    
    while True:

        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO{current_time} {i}", log_interval=1,reset_num_timesteps=False,progress_bar=True)

        model.save(f"{models_dir}/PPO_{TIMESTEPS}_{current_time}__{i}")
        i+=1          

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

        for episode in range(10):
            obs,_ = env.reset()
            done = False
            turncated = False
            j=0
            while not done and not turncated and j<5000:
                action, _ = model.predict(obs)
                obs, reward, done,truncated, _ = env.step(action)
                print(f"ACTION {action} {j}")
                j+=1
            if done:
                    env.render()
       

if __name__ == "__main__":
    main()
