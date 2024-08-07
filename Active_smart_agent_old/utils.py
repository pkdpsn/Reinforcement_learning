import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import DEFAULT_CONFIG



#################### functions ####################
#################### potential value at xy ####################
'''Define all the potential functions here '''

def potential(option,x,y):
    if option == 1:
        return np.where(x**2 + y**2 < 1/4, 16*0.22* (x**2 + y**2 - 1/4)**2, 0)
    elif option == 2:
        return 0.3*(1-x)**2*np.exp(-x**2-(y+1)**2) - (0.2*x - x**3 - y**5)*np.exp(-x**2 - y**2) - 1/30*np.exp(-(x+1)**2 - y**2)
    else :
        return 0 

def make_grid (conf):
    if conf == None:
        conf = DEFAULT_CONFIG
    else :  
        conf = conf
    x = np.linspace(-0.5+conf["start"][0],conf["end"][0]+0.5,100)
    y = np.linspace(-0.5+conf['start'][0],conf['end'][0]+0.5,100)
    X,Y = np.meshgrid(x,y)
    grid = potential(conf['function'],X,Y)
    return grid

#################### RENDER ####################

def plot_trajectory(trajectory,grid,conf , save_path =None,plot= False, Title = None, time = None):
    if conf == None:
        conf = DEFAULT_CONFIG
    else :
        conf = conf
    #plot the potential 
    max_abs_value = np.max(np.abs(grid))  # find the maximum absolute value in the grid
    plt.imshow(grid, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value, vmax=max_abs_value)
    num_ticks = 5
    x_ticks = np.linspace(0, grid.shape[1], num_ticks)
    y_ticks = np.linspace(0, grid.shape[0], num_ticks)
    x_labels = np.linspace(conf['start'][0], conf['end'][0], num_ticks)
    y_labels = np.linspace(conf['start'][0], conf['end'][0], num_ticks)
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.colorbar()

    x_coords = [state[1] for state in trajectory]
    y_coords = [state[0] for state in trajectory]

    # Calculate differences between consecutive points for arrow directions
    x_diff = np.diff(x_coords)
    y_diff = np.diff(y_coords)
    for i in range(len(x_coords)-1):
        plt.quiver(x_coords[i], y_coords[i], x_diff[i], y_diff[i], angles='xy', scale_units='xy', scale=1.5, color='red')
    
    plt.plot(x_coords, y_coords, color='black')
    plt.scatter(x_coords[0], y_coords[0], color='orange',s=20)
    plt.scatter(x_coords[1:-1], y_coords[1:-1], color='green',s=10)
    plt.scatter(x_coords[-1], y_coords[-1], color='blue',s=20)
    if plot:
        plt.show()
    if Title is not None:
        plt.title(Title)
    if time is not None:
        plt.text(0, 0, f'Time: {time}', fontsize=12, color='black')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if save_path is None:
        save_path = f'RL_agents/figure_{current_time}.png'
    else:
        directory = os.path.dirname(save_path)
        save_path = f'{directory}/figure_{current_time}.png'
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    plt.savefig(save_path)
    plt.close()

    return


# print(make_grid(None)) tested and working