from gymnasium import Env
from gymnasium.spaces import Discrete,Box 
import numpy as np
import random
from math import sqrt
from matplotlib import pyplot as plt
import pygame
from typing import Any, SupportsFloat 

class rlEnv(Env):
    
    def __init__(self,W,H):
        #super().__init__()
        # define the grid
        self.W= W
        self.H =H
        self.stepsize=(2/self.W)
        self.gravity=9.81
        self.truncated=False
        self.done=False
        # Actions we can take
        self.action_space = Discrete(4)
        # Observation state 
        self.observation_space= Discrete(W*H)
        self.grid = np.zeros((H, W))
        for i in range(W):
            for j in range(H):
                x = -3 + i * (6 / self.W)
                y = -3 + j * (6 / self.H)
                self.grid[j, i] = 16*0.4 * (x**2 + y**2 - 1/4)**2 if x**2 + y**2 < 1/4 else 0
                # self.grid[j,i]=0.3*(1-x)**2*np.exp(-x**2-(y+1)**2) - (0.2*x - x**3 - y**5)*np.exp(-x**2 - y**2) - 1/30*np.exp(-(x+1)**2 - y**2)
            # print(f"{(self.grid[i,:])}")
        self.screen_width=800
        self.screen_height=800
        self.screen= None
        self.clock= None
        self.render_mode= None


        self.state = self._to_s(int((self.H)/2),int(0.25*(self.W))) ## Change
        self.velocity = 1
        self.reward=0
        self.collective=0
        
    def _to_s(self, row, col):
        return row * self.W + col
    def step(self, action):
        row, col = divmod(self.state, self.W)
        # print(f"INternalllll {row,col}")
        prev_row,prev_col= row,col
        h_prev=self.grid[row,col]
        if self.velocity == 0 or self.collective < -30 :  # Check termination conditions
            self.reward=-10
            self.truncated = True
            
        elif  (row == int((self.H)/2) and col == int(0.75*self.W)):
            self.reward=0
            self.done= True
            
        else:
            self.done = False
            if action == 0:  # Move left
                col = max(col - 1, 0)
            elif action == 1:  # Move down
                row = min(row + 1, self.H - 1)
            elif action == 2:  # Move right
                col = min(col + 1, self.W - 1)
            elif action == 3:  # Move up
                row = max(row - 1, 0)
            if ((prev_col-col+prev_row-row)==0):
                self.reward =-1
            else:
                self.reward = -float(self.stepsize / self.velocity)
            h_new = self.grid[row, col]
            # print(F"High_Old {2 * self.gravity * (h_prev - h_new)} Vel {self.velocity}")
            update_vel = self.velocity**2 + 2 * self.gravity * (h_prev - h_new)
            # print(self.velocity, self.reward , self.done , self.truncated)
            if update_vel < 0:
                self.velocity = 0
            else:
                self.velocity = sqrt(update_vel)
            
            self.state = self._to_s(row, col)
            self.collective+=self.reward
           
        return self.state, self.reward, self.done,self.truncated, {}
    #  pygame.init()
    #     self.width,self.height=800,800
    #     self.screen = pygame.display.set_mode((self.width, self.height))
    #     pygame.display.set_caption("Grid World with Hill")
    #     self.font = pygame.font.Font(None, 24)
    def render(self) :
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.screen.fill((0, 0, 0))
            for i in range(self.W):
                for j in range(self.H):
                    x = i * (self.screen_width / self.W)
                    y = j * (self.screen_height / self.H)
                    # value = self.grid[i, j]
                    # normalized_value = (value - self.min_value) / (self.max_value - self.min_value)

                    # # Use the colormap to get the color corresponding to the normalized value
                    # color = self.cmap(normalized_value)[:3]  # Ignore alpha channel
                    color_index = int(self.grid[i, j] * (len(colors) - 1))
                    color = colors[color_index]

                    pygame.draw.rect(self.screen, color,
                                     (x, y, self.screen_width / self.W, self.screen_height / self.H))
            # add position of active agent
            agent_row, agent_col = divmod(self.state, self.W)
            agent_x = agent_col * (self.screen_width / self.W)
            agent_y = agent_row * (self.screen_height / self.H) + (self.screen_height / self.H) / 2
            pygame.draw.circle(self.screen, (0, 0, 0), (int(agent_x), int(agent_y)), 10)
            pygame.display.flip()
        
        else:
            agent_row, agent_col = divmod(self.state, self.W)
            agent_x = agent_col * (self.screen_width / self.W)
            agent_y = agent_row * (self.screen_height / self.H) + (self.screen_height / self.H) / 2
            pygame.draw.circle(self.screen, (0, 0, 0), (int(agent_x), int(agent_y)), 10)
            pygame.display.flip()

        return
        # return super().render()
    
    def close(self):
        if self.screen is not None:
            # pygame.display.quit()
            pygame.quit()
            self.screen=None



    def reset(self):
        self.state=self._to_s(int((self.H)/2),int(0.25*(self.W)))
        self.velocity = 1
        self.reward = 0
        self.truncated= False
        self.done= False
        self.collective=0
        # print("resetted")
        # check for time
        return self.state