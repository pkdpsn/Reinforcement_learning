import numpy as np 
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG
from Envs import rlenv
from utils import *
import pickle 
import time
from colorama import Fore , Back

def run(EPISODES,verbose,epsilon_value,print_val,q,env):
    # q = np.zeros((env.observation_space.n,env.action_space.n))
    learning_rate=0.7
    discount_factor=1
    epsilon= epsilon_value ## 100% random actions
    epsilon_decay_rate=0.0001
    rng = np.random.default_rng()
    reward_per_episodes=np.zeros(EPISODES)
    # env= rlenv()
    
    for i in range (EPISODES):
        if (i %print_val==0):
            print(i, epsilon)
        state,_ = env.reset()
        done= False
        truncated= False
        while (not done and  not truncated):
            # print("*********************",state[0])
            if rng.random() <epsilon:
                action= env.action_space.sample()
            else:
                # action= np.argmax(q[state,:])
                max_actions = np.where(q[int(state[0]), :] == np.max(q[int(state[0]), :]))[0]
                action = rng.choice(max_actions)
            new_state,reward,done,truncated,info= env.step(action)
            q[int(state[0]),action] = q[int(state[0]),action] + learning_rate * (reward + discount_factor * np.max(q[int(new_state[0]),:])-q[int(state[0]),action])
            if verbose == True:
                print(f"new state {divmod(int(state[0]),env.W)} DONE {done}  TRUNCATEDDD {truncated}  Rewards {reward}")
            state = new_state
            reward_per_episodes[i]+=reward
            # print(q)
            # break
        epsilon= max(epsilon-epsilon_decay_rate,0)
    print(q)

def main():
    print("a")
    env = rlenv()
    q = np.zeros((env.W*env.H,env.action_space.n))
    run(11000,False,1,1000,q,env)
    ## get values of system argvv



if __name__ == "__main__":
    main()