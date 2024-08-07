import numpy as np 
import gymnasium as gym 
import random
from Envs_random import rlEnvs
import matplotlib.pyplot as plt

env = rlEnvs()

paths =[]
rewards = []
time = []
EPISODES =10
for episode in range(1,EPISODES+1):
    state = env.reset()
    done = False
    score=0
    truncated = False
    while not done and not truncated:
        action = np.random.randint(0,7)
        next_state,reward,done,truncated,_ = env.step(action)
        # print(done,truncated)
        score+=reward
    paths.append(env.render())
    rewards.append(score)
    time.append(env.total_time)
    print(f'Episode: {episode}, Score: {score:.3f} , Time {env.total_time:.3f}') 


##plot each path one one graph \
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
for path in paths:
    # Unpack the y and x values from the path
    x = [state[1] for state in path]
    y = [state[0] for state in path]
    # print("PATHS \n\n\n")
    # print(x)
    # print(y)
    plt.plot(x, y)
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], f'Point {x[i]:.4f}, {y[i]:.4f}')   


# Show the plot
# plt.show()
plt.savefig("Figure_7.png")
# Plot histograms of rewards
plt.figure()
plt.hist(rewards, bins=50,label='Rewards',color='blue',edgecolor='black')
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.title('Histogram of Rewards')
plt.show()

plt.figure()
plt.hist(time, bins=50,label='Rewards',color='green',edgecolor='black')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Histogram of time')
plt.show()

