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
        return np.where(x**2 + y**2 < 1/4, 16*0.4* (x**2 + y**2 - 1/4)**2, 0)
    elif option == 2:
        return 0.3*(1-x)**2*np.exp(-x**2-(y+1)**2) - (0.2*x - x**3 - y**5)*np.exp(-x**2 - y**2) - 1/30*np.exp(-(x+1)**2 - y**2)
    else :
        return 0 

def make_grid (conf):
    if conf == None:
        conf = DEFAULT_CONFIG
    else :  
        conf = conf
    x = np.linspace(-0.05+conf["start"][0],conf["end"][0]+0.05,10)
    y = np.linspace(-0.05+conf['start'][0],conf['end'][0]+0.05,10)
    X,Y = np.meshgrid(x,y)
    grid = potential(conf['function'],X,Y)
    return grid

#####################VISION####################
def vision_xy(x,y,conf=None):
    if conf == None:
        conf = DEFAULT_CONFIG
    else :
        conf = conf
    vision = conf['visiblitity']
    space = conf['space']
    x = np.linspace(x-space*vision,x+space*vision,2*vision+1) 
    y = np.linspace(y-space*vision,y+space*vision,2*vision+1)
    x_limits = [conf['start'][0], conf['end'][0]]
    y_limits = [conf['start'][0], conf['end'][0]]

    X,Y = np.meshgrid(x,y)
    Z = potential(conf['function'],X,Y) ## if X , Y are in limits X = [conf[start][0],conf[end][0]] otherwise its 1e9
    Z = np.where( (X >= x_limits[0]) & (X <= x_limits[1]) & (Y >= y_limits[0]) & (Y <= y_limits[1]),Z,1e9)
    # print(Z.dtype)
    # print(x)

    return Z 

#################### RENDER ####################

def plot_trajectory(trajectory,grid,conf , save_path =None,plot= False, Title = None, time = None):
    if conf == None:
        conf = DEFAULT_CONFIG
    else :
        conf = conf
    #plot the potential 
    max_abs_value = np.max(np.abs(grid))  # find the maximum absolute value in the grid
    print(grid.shape)
    # plt.imshow(grid ,cmap='RdBu', interpolation='nearest', vmin=-max_abs_value, vmax=max_abs_value,extent=[-grid.shape[0]/1., grid.shape[0]/1., -grid.shape[0]/1., grid.shape[0]/1. ])
    # num_ticks = 5
    # x_ticks = np.linspace(0, grid.shape[1], num_ticks)
    # y_ticks = np.linspace(0, grid.shape[0], num_ticks)
    # x_labels = np.linspace(conf['start'][0], conf['end'][0], num_ticks)
    # y_labels = np.linspace(conf['start'][0], conf['end'][0], num_ticks)
    # plt.xticks(x_ticks, x_labels)
    # plt.yticks(y_ticks, y_labels)
    # plt.colorbar()

    x_coords = [state[1] for state in trajectory]
    y_coords = [state[0] for state in trajectory]
    # y_coords = [grid.shape[0] - y for y in y_coords]
    # x_coords = [grid.shape[1] - x for x in x_coords]
    # for i in range(len(x_coords)-1):
    #     print(x_coords[i], y_coords[i])
    # Calculate differences between consecutive points for arrow directions
    x_diff = np.diff(x_coords)
    y_diff = np.diff(y_coords)
    plt.xlim(conf['start'][0], conf['end'][0])
    plt.ylim(conf['start'][0], conf['end'][0])
    # for i in range(len(x_coords)-1):
    #     plt.quiver(x_coords[i], y_coords[i], x_diff[i], y_diff[i], angles='xy', scale_units='xy', scale=1.5, color='red')
    
    # plt.plot(x_coords, y_coords, color='black')
    # plt.scatter(x_coords[0], y_coords[0], color='orange',s=200)
    # plt.scatter(x_coords[1:-1], y_coords[1:-1], color='green',s=100)
    # plt.scatter(x_coords[-1], y_coords[-1], color='blue',s=200)
    # plt.scatter([-0.4,0, 0.4],[0,0,0], color='black',s=100)
    
    plt.plot(x_coords, y_coords, marker='s', linestyle='-', color='b')
    # plt.plot(0, 0, marker='o', color='g', markersize=20)
    # plt.plot(3, 4, marker='o', color='g', markersize=20)
    # for i in range(len(x_coords)):
    #     plt.text(x_coords[i], y_coords[i], f'Point {x_coords[i]:.2f}, {y_coords[i]:.2f}')   

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

# if __name__ == "__main__":
#     # Call the vision_xy function with test inputs
#     result = vision_xy(-0.45,0,None)

#     # Print the result
#     print(result)

# tested and working 