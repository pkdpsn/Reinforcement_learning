import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
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
        return 0*16*0.4 * (x**2 + y**2 - 1/4)**2 if x**2 + y**2 < 1/4 else 0
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

def print_grid_and_path(grid, states, conf, save_path=None, plotting = False, graph_title = None):
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

    x_coords = [state[1] for state in states]
    y_coords = [state[0] for state in states]

    # Calculate differences between consecutive points for arrow directions
    x_diff = np.diff(x_coords)
    y_diff = np.diff(y_coords)

    # Add arrows
    for i in range(len(x_coords)-1):
        plt.quiver(x_coords[i], y_coords[i], x_diff[i], y_diff[i], angles='xy', scale_units='xy', scale=1.5, color='red')
   
    # if plotting==True:
    plt.plot(x_coords, y_coords, color='black')
    # plt.scatter(x_coords, y_coords, color='green')
    # Plot first point in a different color
    plt.scatter(x_coords[0], y_coords[0], color='orange',s=20)

    # Plot the rest of the points
    plt.scatter(x_coords[1:-1], y_coords[1:-1], color='green',s=10)
    plt.scatter(x_coords[-1], y_coords[-1], color='blue',s=20)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if save_path is None:
        save_path = f'RL_agents/figure_{current_time}.png'
    else:
        directory = os.path.dirname(save_path)
        save_path = f'{directory}/figure_{current_time}.png'
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    if plotting==True:
        
        if graph_title != None:
            plt.title(graph_title, wrap=True, pad = 8, loc = "center", size = 10)
            # plt.text(Resolution*(x_second - x_first)+11.5, 0, test_string , fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            # plt.subplots_adjust(right=0.7)
        plt.show()   
    plt.close() 

#-----------------Test-----------------
# states = [[1, 2], [0, 2], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [2, 4], [2, 4], [3, 4], [3, 3], [4, 3], [4, 3], [4, 2], [3, 2], [2, 2], [1, 2], [2, 2], [2, 1], [2, 2], [3, 2], [4, 2], [4, 2], [4, 1], [4, 0], [4, 0], [4, 1], [4, 2], [3, 2], [2, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 3], [4, 2], [4, 2], [4, 3], [3, 3], [4, 3], [4, 4], [3, 4], [4, 4], [3, 4], [2, 4], [2, 3], [2, 3]]

# grid = create_grid(None)
# print_grid_and_path(grid, states, conf = None, save_path = None, plotting =True, graph_title = 'This is a test')
