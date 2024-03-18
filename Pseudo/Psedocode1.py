
# importing required libraries 

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

class rlenv(Env):
    # Function to convert row and column indices to a state representation
    def _to_s(self, row, col):
        return row * self.W + col
    
    # Function to perform a movement action
    def _move(self, action, row, col):
        # Perform left, down, right, or up movement based on action
        # Returns the updated row and column indices
        # 0: Left, 1: Down, 2: Right, 3: Up
        ...

    # Function to get the surroundings of a cell in the grid
    def _get_surroundings(self, i, j):
        # Calculate the bounds of the subsection around cell (i, j)
        # Create an array to store the subsection
        # Fill the subsection with values from the grid
        # this function is basically the vision the smaller x*x matrix that the agent can see as an observation 
        ...

    # Function to convert row and column indices to x and y coordinates
    def _ijtoxy(self, i, j):
        # Convert row and column indices to x and y coordinates
        ...

    # Function to convert x and y coordinates to row and column indices
    def _xytoij(self, x, y):
        # Convert x and y coordinates to row and column indices
        ...

    # Constructor to initialize the environment
    def __init__(self, conf=None):
        # Initialize environment parameters and spaces
        # all variable initialized here can be accessed anywhere even outside the code so its neccessary for proper initialization og variables and grid here 
        # Create the grid based on configuration
        ...

    # Function to perform a step in the environment
    def step(self, action):
        # Execute an action and return the new state, reward, done flag, and additional info
        # move the agent get the reward update velocity add the trajectory if required finish the episode or truncate it 
        ...

    # Function to reset the environment to its initial state
    def reset(self, seed=None, options=None):
        # Reset the environment and return the initial observation
        # things like setting the start point , velocity , collrected rewards ,done , truncated paths trajectory  ect to iniital value 
        ...

    # Function to render the environment
    def render(self, seed=None, options=None):
        # Visualize the environment grid and agent trajectory
        ...

# Create an instance of the custom environment on which we run the episodes 
env = rlenv()

# Reset the environment and visualize it
env.reset()
env.render()

# Check if the environment meets the requirements of stable baselines , we need input and output in a specifeid format inorder for it to work with stable-baselines 
# this methods check if our env is compartible initially positive check doesnt really mean that there are no errors in our env its just that its compartible with stable baselines
check_env(env)
