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

models_dir = "models/DQN_CNN"
logdir = "logs"
TIMESTEPS = 500000

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
                # print(subspace.shape)
                self.extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 32, kernel_size=1, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=1, stride=2),
                    nn.ReLU(),
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
        input = th.autograd.Variable(th.rand(batch_size, *shape))
        output_feat = self.extractors["image"](input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def main():
    freeze_support()
    num_cpu = 6
    env = rlEnvs()  # SubprocVecEnv([lambda: rlEnvs() for _ in range(num_cpu)])
    i = 0
    model = DQN("MultiInputPolicy", env, batch_size=64, gamma=0.7, learning_rate=0.0005,exploration_fraction=0.1, buffer_size=100000,
                learning_starts=1000, target_update_interval=1000, train_freq=4, verbose=1, tensorboard_log=logdir,
                policy_kwargs={"features_extractor_class": CustomCombinedExtractor, "activation_fn": th.nn.ReLU, "net_arch": [256, 256,256,64]})

    print(model.policy)

    while True:
        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"DQN_{current_time}_{i}", log_interval=1,
                    reset_num_timesteps=False, progress_bar=True)

        model.save(f"{models_dir}/DQN_{TIMESTEPS}_{current_time}__{i}")

        env.render()
        i = i + 1

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    main()
