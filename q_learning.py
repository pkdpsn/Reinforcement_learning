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
from colorama import Fore , Back

def save_q_table(Q, filename):
        with open(filename, 'wb') as f:
            pickle.dump(Q, f)

def run(EPISODES,verbose,epsilon_value,print_val,q,env,filename):
    # q = np.zeros((env.observation_space.n,env.action_space.n))
    learning_rate=0.7
    discount_factor=1
    epsilon= epsilon_value ## 100% random actions
    epsilon_decay_rate=0.000001
    rng = np.random.default_rng()
    reward_per_episodes=np.zeros(EPISODES)
    # env= rlenv()
    
    # Load Q-table from a file
    # def load_q_table(filename):
    #     with open(filename, 'rb') as f:
    #         Q = pickle.load(f)
    #     return Q

    for i in tqdm(range(EPISODES)):
        
        state,_ = env.reset()
        done= False
        truncated= False
        if (i%1000==0):
            print(i)
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
        epsilon= max(epsilon-epsilon_decay_rate,0.05)
        if (i %print_val==0):
            print(i, epsilon)
        ##------idhar bhi save wali cheez daalni hai------##
            save_q_table(q, 'q_table.pkl')     
        #----yahan tak----#
            print(f"Truncated {truncated} DONE {done}")
            print_grid_and_path(env.grid,env.state_trajectory ,conf=None,save_path='Q-learning/', plotting=False)
        
        ##---------edit karna hai----------#
        # pbar_outer.set_description(f"Episode {i}")
        # pbar_outer.update(1)
        # break
        
    #save line daalni hai#
    save_q_table(q, 'q_table.pkl')          
    # print(q)

def main():
    print("a")
    env = rlenv()
    q_table_filename= "IDK"
    filename= "IDK"
    if len(sys.argv) >1 :
        q_table_filename = sys.argv[1]
        # q = np.load(q_table_filename)
        if os.path.exists(q_table_filename):
            with open(q_table_filename, 'rb') as f:
                q = pickle.load(f)
        else:
            q = np.zeros((env.W*env.H,env.action_space.n))
            with open(q_table_filename, 'wb') as f:
                pickle.dump(q, f)
    else:
        q = np.zeros((env.W*env.H,env.action_space.n))

    run(1000000,False,1,10000,q,env,filename)
    ## get values of system argvv



if __name__ == "__main__":
    main()