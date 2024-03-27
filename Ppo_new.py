from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from datetime import datetime
import gymnasium as gym
from multiprocessing import freeze_support
from Envs import rlenv
import threading
import torch as th
from torch import nn
from torchviz import make_dot

models_dir = "models/PPO"
logdir = "logs"
TIMESTEPS = 5000000

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

        # Iterate over each observation space
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                print(subspace.shape)
                # Feature extractor for image space
                self.extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 32, kernel_size=1, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=1, stride=2),
                    nn.ReLU(),
                    nn.Flatten()
                )
                self.total_concat_size += self._get_conv_output(subspace)
            elif key == "vector":
                # Feature extractor for vector space
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
        shape = tuple(shape.shape)  # Convert Box object to tuple of shape dimensions
        input = th.autograd.Variable(th.rand(batch_size, *shape))
        output_feat = self.extractors["image"](input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # Apply each feature extractor to its corresponding observation
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate the output of all feature extractors
        return th.cat(encoded_tensor_list, dim=1)


def main():
    freeze_support()
    num_cpu = 6
    env = rlenv()# SubprocVecEnv([lambda: rlenv() for _ in range(num_cpu)])
    i = 0
    model = PPO("MultiInputPolicy", env, batch_size=2048, n_epochs=1000, gamma=1, ent_coef=0.001,
                verbose=1, tensorboard_log=logdir, policy_kwargs={"features_extractor_class": CustomCombinedExtractor,"activation_fn":th.nn.ReLU,"net_arch":dict(pi=[256, 256], vf=[256, 256])})
    
    print(model.policy)
    ## make a model of A2C in next line
    # model = A2C("MlpPolicy", env,gamma=1,ent_coef=0.0001 ,verbose=2, tensorboard_log=logdir)

    while True:

        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO{current_time} {i}", log_interval=1,
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
