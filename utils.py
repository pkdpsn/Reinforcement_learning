import numpy as np
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG

def create_grid(conf=None):
    if conf==None:
            conf = DEFAULT_CONFIG
    else:
        conf = conf
    x_first = conf["X"][0]
    x_second = conf["X"][1]
    y_first = conf["Y"][0]
    y_second = conf["Y"][1]
    Resolution = conf["Resolution"]
    rows = Resolution*(x_second - x_first) + 1
    cols = Resolution*(y_second - y_first) + 1
    option = conf["options"]
    def potential1(x, y):
        # Mexican Hat
        return 16*0.4 * (x**2 + y**2 - 1/4)**2 if x**2 + y**2 < 1/4 else 0
    def potential2(x, y):
        # The one with multiple hills and valleys
        return 0.3*(1-x)**2*np.exp(-x**2-(y+1)**2) - (0.2*x - x**3 - y**5)*np.exp(-x**2 - y**2) - 1/30*np.exp(-(x+1)**2 - y**2)
    def potential3(x,y):
        return 0
    x_values = np.linspace(x_first, x_second, cols)
    y_values = np.linspace(y_first, y_second, rows)
    
    if option == 1:
        return [[potential1(x, y) for x in x_values] for y in y_values]
    elif option == 2:
        return [[potential2(x, y) for x in x_values] for y in y_values]
    elif option == 3:
        return [[potential3(x, y) for x in x_values] for y in y_values]
    return 

# states = [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13]]

def print_grid(grid,conf): 
    if conf == None:
        conf = DEFAULT_CONFIG
    else:
        conf = conf
    x_first = conf["X"][0]
    x_second = conf["X"][1]
    y_first = conf["Y"][0]
    y_second = conf["Y"][1]
    Resolution = conf["Resolution"]
    rows = Resolution*(x_second - x_first)+1
    cols = Resolution*(y_second - y_first)+1
    option = conf["options"]
    max_abs_value = np.max(np.abs(grid))  # find the maximum absolute value in the grid
    plt.imshow(grid, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value, vmax=max_abs_value)
    
    num_ticks = 5
    x_ticks = np.linspace(0, cols-1, num_ticks)
    x_labels = np.linspace(x_first, x_second, num_ticks)
    plt.xticks(x_ticks, x_labels)
    y_ticks = np.linspace(0, rows-1, num_ticks)
    y_labels = np.linspace(y_first, y_second, num_ticks)
    plt.yticks(y_ticks, y_labels)

    plt.colorbar()
    plt.show()

def print_grid_and_path(grid, states, conf):
    if conf==None:
        conf = DEFAULT_CONFIG
    else:
        conf = conf
    x_first = conf["X"][0]
    x_second = conf["X"][1]
    y_first = conf["Y"][0]
    y_second = conf["Y"][1]
    Resolution = conf["Resolution"]
    rows = Resolution*(x_second - x_first)+1
    cols = Resolution*(y_second - y_first)+1
    option = conf["options"]
    max_abs_value = np.max(np.abs(grid))  # find the maximum absolute value in the grid
    plt.imshow(grid, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value, vmax=max_abs_value)
    
    num_ticks = 5
    x_ticks = np.linspace(0, cols-1, num_ticks)
    x_labels = np.linspace(x_first, x_second, num_ticks)
    plt.xticks(x_ticks, x_labels)
    y_ticks = np.linspace(0, rows-1, num_ticks)
    y_labels = np.linspace(y_first, y_second, num_ticks)
    plt.yticks(y_ticks, y_labels)
    plt.colorbar()

    x_coords = [state[0] for state in states]
    y_coords = [state[1] for state in states]

    # Plot path
    plt.plot(x_coords, y_coords, color='black')
    plt.scatter(x_coords, y_coords, color='green')
    plt.show()