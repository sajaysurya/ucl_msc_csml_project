'''
some utility functions
'''
import numpy as np
from tabulate import tabulate

def get_location(obs, char):
    '''
    finds the char on the board and return the unraveled index
    '''
    shape = obs[0].board.shape
    loc = np.where(obs[0].layers[char])
    loc = np.squeeze(loc)
    index = np.ravel_multi_index(loc, shape)
    return index

def is_wall(state, obs, wall_char='#'):
    '''
    checks if a wall is there at the given location
    '''
    loc = np.unravel_index(state, obs[0].board.shape)
    return obs[0].layers[wall_char][loc]

def get_dungeon_state(obs):
    '''
    the state is encoded visually
    player has be char 'O'
    0 - left end
    1 - right end
    2 - middle
    3 - confusion
    '''
    shape = obs[0].board.shape
    loc = np.where(obs[0].layers['O'])
    loc = np.squeeze(loc)
    if loc[1] == 1:
        return 0
    if loc[1] == 3:
        return 2
    if loc[1] == 5:
        return 1
    return 3

def make_latex_table(obs, policy):
    '''
    a simple function used to make tables for reports
    use this only for stochastic policy
    '''
    wall = obs[0].layers['#']
    policy = np.reshape(np.copy(policy), (4, 9, 11))
    start_loc = np.where(obs[0].layers['O'])
    goal_loc = np.where(obs[0].layers['X'])
    canvas = np.empty_like(wall, dtype=object)
    canvas[np.where(wall == True)] = r"$\blacksquare$"
    canvas[goal_loc] = r"\texttt{G}"
    current_loc = np.squeeze(np.array(start_loc))
    while canvas[current_loc[0], current_loc[1]] != r"\texttt{G}":
        action = np.argmax(policy[:, current_loc[0], current_loc[1]])
        if action == 0:
            canvas[current_loc[0], current_loc[1]] = r"$\leftarrow$"
            current_loc[1] -= 1
        elif action == 1:
            canvas[current_loc[0], current_loc[1]] = r"$\rightarrow$"
            current_loc[1] += 1
        elif action == 2:
            canvas[current_loc[0], current_loc[1]] = r"$\uparrow$"
            current_loc[0] -= 1
        elif action == 3:
            canvas[current_loc[0], current_loc[1]] = r"$\downarrow$"
            current_loc[0] += 1
    print(tabulate(canvas, tablefmt="latex_raw"))
