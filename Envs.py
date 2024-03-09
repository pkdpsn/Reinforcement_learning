from gymnasium import Env
from gymnasium.spaces import Discrete,Box 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C,PPO, DQN
import numpy as np
import random
from math import sqrt
from matplotlib import pyplot as plt
from typing import Any, SupportsFloat
from config import DEFAULT_CONFIG
from utils import *

class rlenv(Env):
    def _to_s(self, row, col):
        return row * self.W + col
    def _move(self,action,row,col):
        if action == 0:  # Move left
            col = max(col - 1, 0)
        elif action == 1:  # Move down
            row = min(row + 1, self.H - 1)
        elif action == 2:  # Move right
            col = min(col + 1, self.W - 1)
        elif action == 3:  # Move up
            row = max(row - 1, 0)
        return row ,col
    def _get_surroundings(self,i, j):
    # Calculate the bounds of the subsection
        half_size = self.visibility // 2
        start_i = max(0, i - half_size)
        start_j = max(0, j - half_size)
        end_i = min(self.grid.shape[0], i + half_size + 1)
        end_j = min(self.grid.shape[1], j + half_size + 1)

        # Create an array to store the subsection
        subsection = np.full((self.visibility, self.visibility), 1e9)

        # Fill the subsection with values from the self.grid
        start_i_sub, start_j_sub = half_size - (i - start_i), half_size - (j - start_j)
        end_i_sub, end_j_sub = start_i_sub + (end_i - start_i), start_j_sub + (end_j - start_j)
        subsection[start_i_sub:end_i_sub, start_j_sub:end_j_sub] = self.grid[start_i:end_i, start_j:end_j]

        return subsection.flatten()
    def _ijtoxy(self,i,j):
        return self.conf["X"][0]+j/self.conf["Resolution"],self.conf["Y"][1]-i/self.conf["Resolution"]
        return x,y
    def _xytoij(self,x,y):
        return int((x-self.conf["X"][0])*self.conf["Resolution"]) , int(abs(y-self.conf["Y"][1])*self.conf["Resolution"])
        

    

    def __init__(self, conf=None):
        super().__init__()
        if conf==None:
            self.conf = DEFAULT_CONFIG
        else:
            self.conf = conf
        self.W= (self.conf["X"][1]-self.conf["X"][0])*self.conf["Resolution"]+1
        self.H =(self.conf["Y"][1]-self.conf["Y"][0])*self.conf["Resolution"]+1
        self.visibility= self.conf["visiblity"]
        # print(self.H, self.W)
        self.stepsize=int(1/self.conf["Resolution"])
        self.gravity=9.81
        self.truncated=False
        self.done=False
        self.action_space = Discrete(4)
        self.observation_space = Box(-1e9,1e9,shape=([self.visibility**2+7]),dtype=np.float64)
        # print(f"shape",[self.visibility**2+7])
        self.grid = np.zeros((self.H,self.W))
        self.grid = np.array(create_grid())
        self.velocity = 1
        self.reward=0
        self.collective=0
        self.finish_row,self.finish_col=self._xytoij(self.conf["end"][0],self.conf["end"][1])
        self.start_row,self.start_col=self._xytoij(self.conf["start"][0],self.conf["start"][1])
        print(f"starting ",self.start_col,self.start_row)
        print(f"finishing ",self.finish_col,self.finish_row)
        self.state_trajectory = []
        self.reward_trajectory = []
        # print(f"grid",self.grid)
        self.state = self._to_s(int(self.start_col),int(self.start_row))
    
    def step(self, action):
        row, col = divmod(self.state, self.W)
        prev_row,prev_col= row,col
        h_prev=self.grid[row,col]
        if self.velocity == 0 or self.collective < -30 :  # Check termination conditions
            self.reward=-10
            self.truncated = True
            
        elif  (row == int(self.finish_col) and col == int(self.finish_row)): 
            self.reward=0
            self.done= True
        else:
            self.done = False
            row , col = self._move(action,row,col)
            if ((prev_col-col+prev_row-row)==0):
                self.reward = -1 ####incase it keeps bumping with wall
            else:
                self.reward = -float(1/(self.conf["Resolution"] *self.velocity))
            h_new = self.grid[row, col]

            # we update vel make afunction for further usage with new equations
            update_vel = self.velocity**2 + 2 * self.gravity * (h_prev - h_new)
            if update_vel < 0:
                self.velocity = 0
            else:
                self.velocity = sqrt(update_vel)
            self.collective+=self.reward
            self.state = self._to_s(row, col)
        self.reward_trajectory.append(self.render)
        self.state_trajectory.append([row,col])
        # print(self.state)
        obs = [self.state,row,col,self.finish_row,self.finish_col,int(self.truncated),int(self.done)]+list(self._get_surroundings(col,row))
        obs=np.array(obs)
        return obs,self.reward, self.done,self.truncated, {}

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        # print_grid(self.grid,None)
        # print("I am in reset")
        self.state=self._to_s(int((self.H)/2),int(0.25*(self.W)))####make this random later
        row, col = divmod(self.state, self.W)
        self.velocity = 1
        self.reward = 0
        self.truncated= False
        self.done= False
        self.collective=0
        # print(self.state_trajectory)
        self.state_trajectory = []
        self.reward_trajectory = []
        obs = [self.state,row,col,self.finish_row,self.finish_col,int(self.truncated),int(self.done)]+list(self._get_surroundings(col,row))        
        obs=np.array(obs)
     
        return obs , {}
    
    
    
    
    # def render(self) -> RenderFrame | list[RenderFrame] | None:
    #     pass
    #     return super().render()

env=rlenv()
# for row in range(env.W):
#     for col in range(env.W):
#         state_index = env._to_s(row, col)
#         X,Y = env._ijtoxy(row,col)
#         Col, Row = env._xytoij(X,Y)
#         print(f"({row}, {col}) --> {Row} {Col} -> State Index: {state_index} at X = {X} and Y = {Y}")
#     # break
check_env(env)
# model = DQN("MlpPolicy", env,learning_rate=0.001,buffer_size=1000 ,verbose=1)
# model.learn(total_timesteps=2000,progress_bar=True)

# episodes=1
# for episode in range (1, episodes+1):
#     state,_ = env.reset()
#     # print(state)
#     done = False
#     truncated=False
#     score =0
#     ac=[2,2,2,2,2,2,2,2,2,2,2,2,2]
#     # while not done:
#     for i in range (0,10):
#         action= ac[i]#env.action_space.sample()
#         print(f"State now {divmod(int(state[0]),env.W)}")
#         state,reward,done,truncated,info = env.step(action)
#         print(state)
#         row, col = divmod(int(state[0]),env.W)
        
#         a="left"
#         if action==0:
#             a="l"
#         if action==1:
#             a="d"
#         if action==2:
#             a="r"
#         if action==3:
#             a="u"
        
#         print(f"Action {a} State {row,col, int(state[0])} Reward {reward}")
#         score=reward
#     print("Episode:{} Score{}\n\n".format(episode,score))
