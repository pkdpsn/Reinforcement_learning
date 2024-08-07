from stable_baselines3 import DQN
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
from stable_baselines3.common.callbacks import BaseCallback

models_dir = "models/DQN_LENET5_2"
logdir = "logs"
TIMESTEPS = 50000000

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
                self.model.save(f"{models_dir}/DQN_LENET5{TIMESTEPS}_{current_time}_{self.total_steps}")
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
                self.model.save(f"{models_dir}/DQN_best_lenet{current_time}_step{self.total_steps}")
            except OSError as e :
                print(f"Error saving  best model: {e}")

def main():
    freeze_support()
    num_cpu = 80
    env =  rlEnvs() #SubprocVecEnv([lambda: rlEnvs() for _ in range(num_cpu)])
    i = 0
    model = DQN("MultiInputPolicy", env, batch_size=128, gamma=0.9, learning_rate=0.001,exploration_fraction=0.1, buffer_size=1000000,
                learning_starts=10000, target_update_interval=10000, train_freq=4, verbose=1, tensorboard_log=logdir,
                policy_kwargs={"features_extractor_class": CustomCombinedExtractor, "activation_fn": th.nn.Tanh, "net_arch": [128,128]})

    print("Model Settings:")
    print(f"Policy: {model.policy_class}")
    print(f"Environment: {env}")
    print(f"Batch Size: {model.batch_size}")
    print(f"Gamma: {model.gamma}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Exploration Fraction: {model.exploration_fraction}")
    print(f"Buffer Size: {model.buffer_size}")
    print(f"Learning Starts: {model.learning_starts}")
    print(f"Target Update Interval: {model.target_update_interval}")
    print(f"Train Frequency: {model.train_freq}")
    print(f"Verbose: {model.verbose}")
    print(f"Tensorboard Log Directory: {model.tensorboard_log}")
    print(f"Policy Keyword Arguments: {model.policy_kwargs}")
    print(model.policy)

    callback = SaveOnBestRewardAndStepCallback()

    while True:
        model.learn(total_timesteps=TIMESTEPS,callback=callback ,tb_log_name=f"DQN_LENETBig{current_time}_{i}", log_interval=1,
                    reset_num_timesteps=False, progress_bar=True)

        model.save(f"{models_dir}/DQN_{TIMESTEPS}_{current_time}__{i}")

        env.render()
        i = i + 1

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    main()
