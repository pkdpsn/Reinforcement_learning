# Import required libraries and modules
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DQN
import numpy as np
import random
from math import sqrt
from matplotlib import pyplot as plt
from typing import Any, SupportsFloat
from config import DEFAULT_CONFIG
from utils import *

# Define the environment class rlenv inheriting from Env
class rlenv(Env):
    # Define helper functions
    def _to_s(self, row, col):
        # Convert row and column indices to a state index
        return row * self.W + col

    def _move(self, action, row, col):
        # Move the agent based on the action
        if action == 0:  # Move left
            col = max(col - 1, 0)
        elif action == 1:  # Move down
            row = min(row + 1, self.H - 1)
        elif action == 2:  # Move right
            col = min(col + 1, self.W - 1)
        elif action == 3:  # Move up
            row = max(row - 1, 0)
        return row, col

    def _get_surroundings(self, i, j):
        # Get the surroundings of the agent
        half_size = self.visibility // 2
        start_i = max(0, i - half_size)
        start_j = max(0, j - half_size)
        end_i = min(self.grid.shape[0], i + half_size + 1)
        end_j = min(self.grid.shape[1], j + half_size + 1)
        subsection = np.full((self.visibility, self.visibility), 1e9)
        start_i_sub, start_j_sub = half_size - (i - start_i), half_size - (j - start_j)
        end_i_sub, end_j_sub = start_i_sub + (end_i - start_i), start_j_sub + (end_j - start_j)
        subsection[start_i_sub:end_i_sub, start_j_sub:end_j_sub] = self.grid[start_i:end_i, start_j:end_j]
        return subsection.flatten()

    def _ijtoxy(self, i, j):
        # Convert row and column indices to x, y coordinates
        return self.conf["X"][0] + j / self.conf["Resolution"], self.conf["Y"][1] - i / self.conf["Resolution"]

    def _xytoij(self, x, y):
        # Convert x, y coordinates to row and column indices
        return int((x - self.conf["X"][0]) * self.conf["Resolution"]), int(abs(y - self.conf["Y"][1]) * self.conf["Resolution"])

    # Initialize the environment
    def __init__(self, conf=None):
        super().__init__()
        # Set configuration
        if conf == None:
            self.conf = DEFAULT_CONFIG
        else:
            self.conf = conf
        # Calculate width and height of the grid
        self.W = int((self.conf["X"][1] - self.conf["X"][0]) * self.conf["Resolution"]) + 1
        self.H = int((self.conf["Y"][1] - self.conf["Y"][0]) * self.conf["Resolution"]) + 1
        print(self.H, self.W)
        # Set visibility and step size
        self.visibility = self.conf["visiblity"]
        self.stepsize = int(1 / self.conf["Resolution"])
        # Set gravity and other parameters
        self.gravity = 9.81
        self.truncated = False
        self.done = False
        # Define action and observation space
        self.action_space = Discrete(4)
        self.observation_space = Box(-1e9, 1e9, shape=([self.visibility**2 + 8]), dtype=np.float64)
        # Create grid and initialize velocity, reward, and collective variables
        self.grid = np.zeros((self.H, self.W))
        self.grid = np.array(create_grid())
        self.velocity = 1
        self.reward = 0
        self.collective = 0
        # Convert finish and start positions to row and column indices
        self.finish_col, self.finish_row = self._xytoij(self.conf["end"][0], self.conf["end"][1])
        self.start_col, self.start_row = self._xytoij(self.conf["start"][0], self.conf["start"][1])
        print(f"starting ", self.start_col, self.start_row)
        print(f"finishing ", self.finish_col, self.finish_row)
        # Initialize state trajectory and reward trajectory
        self.state_trajectory = []
        self.reward_trajectory = []
        self.random_init = bool(self.conf["randomstart"])
        # Set initial state
        self.state = self._to_s(int(self.start_row), int(self.start_col))

    # Perform a step in the environment
    def step(self, action):
        row, col = divmod(self.state, self.W)
        prev_row, prev_col = row, col
        h_prev = self.grid[row, col]
        if self.velocity == 0 or self.collective < -30:  # Check termination conditions
            self.reward = -10
            self.truncated = True
        elif (row == int(self.finish_col) and col == int(self.finish_row)):
            self.reward = 0
            self.done = True
        else:
            self.done = False
            row, col = self._move(action, row, col)
            if ((prev_col - col + prev_row - row) == 0):
                self.reward = -1  # In case it keeps bumping with wall
            else:
                self.reward = -float(1 / (self.conf["Resolution"] * self.velocity))
            h_new = self.grid[row, col]
            update_vel = self.velocity**2 + 2 * self.gravity * (h_prev - h_new)
            if update_vel < 0:
                self.velocity = 0
            else:
                self.velocity = sqrt(update_vel)
            self.collective += self.reward
            self.state = self._to_s(row, col)
        self.reward_trajectory.append(self.reward)
        self.state_trajectory.append([row, col])
        obs = [self.state, row, col, self.finish_row, self.finish_col, int(self.truncated), int(self.done),
               int(self.velocity)] + list(self._get_surroundings(col, row))
        obs = np.array(obs)
        return obs, self.reward, self.done, self.truncated, {}

    # Reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_init:
            random_y = np.random.uniform(self.conf["rand_Y"][0], self.conf["rand_Y"][1])
            random_x = np.random.uniform(self.conf["rand_X"][0], self.conf["rand_X"][1])
            random_row, random_col = self._xytoij(random_x, random_y)
            self.state = self._to_s(random_row, random_col)
        else:
            self.state = self._to_s(int(self.start_row), int(self.start_col))
        row, col = divmod(self.state, self.W)
        self.velocity = 1
        self.reward = 0
        self.truncated = False
        self.done = False
        self.collective = 0
        self.state_trajectory = []
        self.reward_trajectory = []
        self.state_trajectory.append([row, col])
        obs = [self.state, row, col, self.finish_row, self.finish_col, int(self.truncated), int(self.done),
               int(self.velocity)] + list(self._get_surroundings(col, row))
        obs = np.array(obs)
        return obs, {}

    # Render the environment
    def render(self, seed=None, options=None):
        print(self.state_trajectory)
        print_grid_and_path(self.grid, self.state_trajectory, conf=None, save_path='Render/', plotting=False,
                            graph_title=None)
        return


# Create an instance of the environment
env = rlenv()
# Reset the environment
env.reset()
# Render the environment
env.render()
# Check if the environment is compatible with stable-baselines
check_env(env)
