'''
Simple gridworld as in http://ai.stanford.edu/~nir/Papers/DFR1.pdf

Key for ascii art
| - wall
O - player location

Has no goal state, episodes never end

Reward:
    see the paper for details
'''

import curses
import numpy as np
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites


GRID_WORLD = ['O....@']


def get_params():
    '''
    gives the parameter of the environment
    1. number of actions
    2. number of states
    3. shape of gridworld
    '''
    params = {
        'num_actions': 2,
        'num_states' : 5,
        'shape' : (1, 5)
        }
    return params


def make_game():
    '''
    builds the game and returns the engine
    '''
    return ascii_art.ascii_art_to_game(
        GRID_WORLD,
        what_lies_beneath='.',  # space character
        sprites={'O': PlayerSprite})


class PlayerSprite(prefab_sprites.MazeWalker):
    '''
    Class for player

    Defines movements and associated rewards
    '''

    def __init__(self, corner, position, character):
        '''
        initialize as per superclass instructions and make walls impassable
        '''
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='@')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused

        if actions == 0:    # go east (mostly)
            # add reward as per location
            if self._position[1] == 4:
                the_plot.add_reward(200.0)
            if np.random.random() > 0.2:
                self._east(board, the_plot)
            else:
                self._teleport((0, 0))
        elif actions == 1:  # go west (mostly)
            if self._position[1] == 0:
                the_plot.add_reward(20.0)
            else:
                the_plot.add_reward(2.0)
            if np.random.random() < 0.2:
                self._east(board, the_plot)
            else:
                self._teleport((0, 0))
        elif actions == 4:  # stay (for human player)
            self._stay(board, the_plot)
        elif actions == 5:    # quit (for termination)
            the_plot.terminate_episode()


def main():
    '''
    C-style main function
    '''

    # Build the game
    game = make_game()

    # Create user interface
    user_interface = human_ui.CursesUi(
        keys_to_actions={curses.KEY_LEFT: 1,
                         curses.KEY_RIGHT: 0,
                         -1: 4,  # for dummy stayput action
                         'q': 5, 'Q': 5},  # to exit
        delay=200)

    # Run the game
    user_interface.play(game)


if __name__ == '__main__':
    main()
