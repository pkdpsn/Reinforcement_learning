from gymnasium import Env 
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from typing import Any, SupportsFloat
from math import sqrt ,cos,sin
from matplotlib import pyplot as plt
from config import DEFAULT_CONFIG
from stable_baselines3.common.env_checker import check_env
from utils import *
'''
This is a continous grid world environment where the agent is placed in a grid world with a visibility range of d. 
The agent can move in 8 directions and the goal is to reach the target.
This is truely continous environment where the agent be in any position in the grid world.
Also we can toggle the noise in the environment by setting the noise to True.




'''

grid = make_grid(None) ## for plotting purpose only 

class rlEnvs(Env):

    def _newvelocity(self,x_old,y_old,x_new,y_new):
        ds = sqrt((x_new-x_old)**2 + (y_new-y_old)**2)
        dx = x_new - x_old
        dy = y_new - y_old
        if ds == 0:
            return -1
        if dx !=0:
            du_dx =(potential(self.conf['function'],x_new,y_old)-potential(self.conf['function'],x_old,y_old))/dx
        else:
            du_dx = 0
        if dy !=0:
            du_dy =(potential(self.conf['function'],x_old,y_new)-potential(self.conf['function'],x_old,y_old))/dy
        else:
            du_dy = 0
        f = sqrt(du_dx**2 + du_dy**2)
        if (f**2-((dx/ds)*du_dx + (dy/ds)*du_dy)**2)>1:
            return -1
        # print(f"dx: {dx} dy: {dy} ds: {ds} du_dx: {du_dx} du_dy: {du_dy} f: {f} {f**2-((dx/ds)*du_dx + (dy/ds)*du_dy)**2}")
        return -((dx/ds)*du_dx + (dy/ds)*du_dy)+sqrt(1-(f**2-((dx/ds)*du_dx + (dy/ds)*du_dy)**2))



    def __init__(self, conf: dict = None):
        if conf==None:
            self.conf = DEFAULT_CONFIG
        else:
            self.conf = conf
        self.d = self.conf['d']
        self.visiblitity = self.conf['visiblitity']
        self.truncated = False 
        self.done = False
        self.action_space = Discrete(8)
        self.observation_space = Box(-1e9,1e9,shape=([10**2+8]),dtype=np.float64)
        self.velocity = 1
        self.noise = self.conf['noise']
        self.reward = 0 
        self.delt= self.conf['delt']
        self.total_time = 0
        self.time_limit = 10
        self.del_r = self.conf['delta_r']
        self.function = self.conf['function']
        self.start_x, self.start_y = self.conf['start'][0], self.conf['start'][1]
        self.target_x, self.target_y = self.conf['end'][0], self.conf['end'][1]   
        print("Starting position: ", self.start_x, self.start_y)
        print("Target position: ", self.target_x, self.target_y)
        self.trajectory = []
        self.reqward_trajectory = []
        self.random_start = self.conf['random_start']
        self.state = [self.start_y, self.start_x]
        
        

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if self.random_start:
            self.state = [random.uniform(self.conf["start"][0],self.conf["end"][0]),random.uniform(self.conf["start"][0],self.conf["end"][0])]
        else:
            self.state = [self.start_y, self.start_x]
        self.done = False
        self.reward = 0
        self.total_time = 0
        self.velocity = 1
        self.truncated = False
        self.trajectory = []
        self.reward_trajectory = []
        self.trajectory.append(self.state)

                
        return np.array([self.state])# what all to return here

    def step(self, action: int):
        y_old,x_old =self.state
        x_new, y_new = x_old, y_old
        # print("Taking step")
        ##check if the agent is within the target range 
        if sqrt((self.target_x-x_old)**2 + (self.target_y-y_old)**2) < self.del_r:
            self.done = True
            self.reward = 100
            # return np.array([self.state]), self.reward, self.done, {} 
        ## check for truncation
        elif self.velocity<=0 or self.total_time>=10 or (x_new-self.target_x)>0.001  :
            self.reward = -100
            self.truncated = True
        # elif True:
        theta = (action-1)*np.pi/4
        # if self.noise:
        #     theta = theta +sqrt(2*self.d*self.delt)*np.random.normal(0,1)
        
        x_new = x_old + self.velocity*np.cos(theta)*self.delt
        y_new = y_old + self.velocity*np.sin(theta)*self.delt
        # print(f"internal state: {self.state}")
        if self.noise :
            # print("Noise added")
            x_new += sqrt(2*self.d*self.delt)*np.random.normal(0,1)
            y_new += sqrt(2*self.d*self.delt)*np.random.normal(0,1)

        #change velocity 
        self.velocity = self._newvelocity(x_old,y_old,x_new,y_new)
        print(f"Velocity: {self.velocity:.5f} state: [{self.state[0]:.3f}, {self.state[1]:.3f}] , {self.done==False} ,{self.truncated==False} ,{(x_new-self.target_x)}")
        self.state = [y_new, x_new]
        '''
        3 types of rewards can be given
        -ve time for spending more time
        +ve reward for reaching the midtargets

        '''
        self.reward = -self.delt +1.5-sqrt((x_new-self.target_x)**2 + (y_new-self.target_y)**2)
        #return the visible part of env 



        self.reward_trajectory.append(self.reward)
        self.trajectory.append(self.state)



        return self.state, self.reward, self.done,self.truncated, {}

    def render(self, mode='human'):
        # print(self.trajectory)
        #plot the trajectory of the agent
        plot_trajectory(self.trajectory,grid,conf = None , save_path = "Render/",plot= False, Title = "Trajectory of the agent", time = self.total_time) 
        return self.trajectory



# env = rlEnvs()

# check_env(env)