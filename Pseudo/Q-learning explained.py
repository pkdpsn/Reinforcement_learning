# Import necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG
from Envs import rlenv
from utils import *
import pickle
import time
import sys
import os
from tqdm import tqdm
from colorama import Fore, Back

# Function to save the Q-table to a file
def save_q_table(Q, filename):
    # Open the file in binary write mode
    with open(filename, 'wb') as f:
        # Serialize and save the Q-table using pickle
        pickle.dump(Q, f)

# Function to run the Q-learning algorithm
def run(EPISODES, verbose, epsilon_value, print_val, q, env, filename):
    # Define learning parameters
    learning_rate = 0.9
    discount_factor = 1
    epsilon = epsilon_value  # Initial value of epsilon for epsilon-greedy policy
    epsilon_decay_rate = 0.0000015  # Rate of decay for epsilon
    rng = np.random.default_rng()  # Random number generator
    
    # Array to store rewards per episode
    reward_per_episodes = np.zeros(EPISODES)

    # Loop over the specified number of episodes
    for i in tqdm(range(EPISODES)):
        # Reset the environment and initialize episode variables
        state, _ = env.reset()
        done = False
        truncated = False

        # Main loop for each episode
        while not done and not truncated:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Choose a random action
            else:
                max_actions = np.where(q[int(state[0]), :] == np.max(q[int(state[0]), :]))[0]
                action = rng.choice(max_actions)  # Choose the action with maximum Q-value

            # Take the chosen action and observe the new state, reward, and termination status
            new_state, reward, done, truncated, _ = env.step(action)

            # Update Q-value of the current state-action pair using the Q-learning update rule
            q[int(state[0]), action] += learning_rate * (
                    reward + discount_factor * np.max(q[int(new_state[0]), :]) - q[int(state[0]), action])

            # Update the current state and accumulate the reward
            state = new_state
            reward_per_episodes[i] += reward

            # Decay epsilon
            epsilon = max(epsilon - epsilon_decay_rate, 0.15)

            # Print verbose information if required
            if verbose:
                print(f"New state: {divmod(int(state[0]), env.W)} DONE: {done} TRUNCATED: {truncated} Rewards: {reward}")

        # Save Q-table and print progress at specified intervals
        if i % print_val == 0:
            save_q_table(q, 'q_table.pkl')
            print(f"Episode: {i}, Epsilon: {epsilon}")

    # Save the final Q-table
    save_q_table(q, 'q_table.pkl')

# Main function
def main():
    # Initialize the environment
    env = rlenv()

    # Initialize variables for Q-table filename and Q-table
    q_table_filename = "IDK"
    q = np.zeros((env.W * env.H, env.action_space.n))

    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        q_table_filename = sys.argv[1]  # Get the filename of Q-table

        # Check if the Q-table file exists
        if os.path.exists(q_table_filename):
            print("Q-table file found:", q_table_filename)
            # Load the Q-table from the file
            with open(q_table_filename, 'rb') as f:
                q = pickle.load(f)
        else:
            # Create a new Q-table if the file does not exist
            with open(q_table_filename, 'wb') as f:
                pickle.dump(q, f)
    else:
        q = np.zeros((env.W * env.H, env.action_space.n))  # Initialize a new Q-table

    # Run the Q-learning algorithm
    run(800000, False, 0.05, 1000, q, env, filename="IDK")

# Entry point of the script
if __name__ == "__main__":
    main()
