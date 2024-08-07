from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from datetime import datetime
import gymnasium as gym
from multiprocessing import freeze_support
from Env import rlEnvs
import threading
import torch as th
from torch import nn
# from torchviz import make_dot
from stable_baselines3.common.callbacks import BaseCallback

models_dir = "models/PPO_CNN_lenet"
logdir = "logs"
TIMESTEPS =50000000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=64)
        self.extractors = {}
        self.total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                print(subspace.shape)
                self.extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 6, kernel_size=5, stride=1),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5, stride=1),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 120, kernel_size=5, stride=1),
                    nn.Tanh(),
                    nn.Flatten()
                )
                self.total_concat_size += self._get_conv_output(subspace)
            elif key == "vector":
                self.extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                self.total_concat_size += 32

        self.extractors = nn.ModuleDict(self.extractors)
        self._features_dim = self.total_concat_size

    def _get_conv_output(self, shape):
        batch_size = 1
        shape = tuple(shape.shape)
        # shape = new_tuple = (1, shape[0], shape[1])
        print(f"shape1 {shape}")
        input = th.autograd.Variable(th.rand(batch_size, *shape))
        output_feat = self.extractors["image"](input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

class SaveOnBestRewardAndStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -float('inf')
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_steps += 1
        # Save model every 20,000 steps
        if self.total_steps % 600000 == 0:
            try:
                self.model.save(f"{models_dir}/PPO_LENET5{TIMESTEPS}_{current_time}_{self.total_steps}")
            except OSError as e:
                print(f"Error saving model: {e}")
        return True

    def _on_evaluation_end(self) -> None:
        # Evaluate mean reward
        mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}")

        # Save model if new local high reward is achieved
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            try:
                self.model.save(f"{models_dir}/PPO_best_lenet{current_time}_step{self.total_steps}")
            except OSError as e :
                print(f"Error saving  best model: {e}")


def main():
    freeze_support()
    num_cpu = 6
    env =  rlEnvs()#SubprocVecEnv([lambda: rlEnvs() for _ in range(num_cpu)])
    i = 0
    model = PPO("MultiInputPolicy", env, batch_size=64, n_epochs=100, gamma=0.7, ent_coef=0.01,
                verbose=1, tensorboard_log=logdir, policy_kwargs={"features_extractor_class": CustomCombinedExtractor,"activation_fn":th.nn.ReLU,"net_arch":dict(pi=[256, 256], vf=[256, 256])})
    
    print(model.policy)
    ## make a model of A2C in next line
    # model = A2C("MlpPolicy", env,gamma=1,ent_coef=0.0001 ,verbose=2, tensorboard_log=logdir)

    while True:

        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO_lenet{current_time} {i}", log_interval=1,
                    reset_num_timesteps=False, progress_bar=True)

        model.save(f"{models_dir}/PPO_{TIMESTEPS}_{current_time}__{i}")
        # obs = env.reset()
        # done , truncated = False , False
        # while not done or not truncated:
        #     action, _states = model.predict(obs)
        #     ## this is return of the step function return obs,self.reward, self.done,self.truncated, {} correct line
        #     obs, rewards, done, truncated, info = env.step(action)

        env.render()
        i = i + 1

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    main()
