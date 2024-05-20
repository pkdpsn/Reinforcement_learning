import numpy as np 
import gymnasium as gym 
import random
from Envs import rlEnvs
import matplotlib.pyplot as plt

env = rlEnvs()

paths =[]

EPISODES =50
for episode in range(1,EPISODES+1):
    state = env.reset()
    done = False
    score=0
    truncated = False
    while not done and not truncated:
        action = 1
        next_state,reward,done,truncated,_ = env.step(action)
        print(done,truncated)
        score+=reward
    paths.append(env.render())
    print(f'Episode: {episode}, Score: {score}') 


##plot each path one one graph \
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
for path in paths:
    # Unpack the y and x values from the path
    x = [state[1] for state in path]
    y = [state[0] for state in path]
    print("PATHS \n\n\n")
    print(x)
    print(y)
    plt.plot(x, y)

# Show the plot
plt.show()