'''
Comparing the performance of following in solving room MDP
1. Expectation - Maximization
2. Policy iteration
3. Gradient Ascent
'''
import ipdb
import numpy as np
import room as fl
import agent
import utils

def main():
    '''
    C-style main function
    This is an MDP
    both reward and transitions are known
    '''
    # make the game to get details
    game = fl.make_game()
    obs = game.its_showtime()

    # parameters
    world_param = fl.get_params()
    num_states = world_param['num_states']
    num_actions = world_param['num_actions']
    len_episodes = 100
    discount = 1
    # reward function
    reward = np.zeros((num_states, num_actions))
    goal_loc = utils.get_location(obs, 'X')
    # in all the following cases, we have 100 -> enters the goal state
    reward[goal_loc+1, 0] = 100
    reward[goal_loc-1, 1] = 100
    reward[goal_loc+11, 2] = 100
    reward[goal_loc-11, 3] = 100
    # start distribution - always current start location
    start_loc = utils.get_location(obs, 'O')
    start_dist = np.zeros(num_states)
    start_dist[start_loc] = 1

    # transition distribution (just alpha counts)
    alpha = np.zeros((num_states, num_states, num_actions))
    # go through all states
    for state in range(num_states):
        # change if it is not a wall
        if not utils.is_wall(state, obs):
            # if there is no wall, give 1
            if not utils.is_wall(state-1, obs):
                alpha[state-1, state, 0] = 1
            else:  # stay if there is wall
                alpha[state, state, 0] = 1

            if not utils.is_wall(state+1, obs):
                alpha[state+1, state, 1] = 1
            else:  # stay if there is wall
                alpha[state, state, 1] = 1

            if not utils.is_wall(state-11, obs):
                alpha[state-11, state, 2] = 1
            else:  # stay if there is wall
                alpha[state, state, 2] = 1

            if not utils.is_wall(state+11, obs):
                alpha[state+11, state, 3] = 1
            else:  # stay if there is wall
                alpha[state, state, 3] = 1
        else: # if inside wall, you will stay inside wall
            alpha[state, state, :] = 1

    #alpha = np.ones((num_states, num_states, num_actions))
    # create agent
    bond = agent.MLEM(num_actions,
                      num_states,
                      discount,
                      len_episodes)
    # set transition distribution
    bond.alpha = np.copy(alpha)
    bond.update_theta()
    # set reward details
    bond.reward = np.copy(reward)
    # set start distribution
    bond.start_dist = np.copy(start_dist)

    # change policy to local minima
    bond.policy[:, :] = 1
    bond.policy[3, 24] = 10
    bond.policy[3, 35] = 10
    bond.policy[3, 46] = 10
    bond.policy[1, 57] = 10
    bond.policy[1, 58] = 10
    bond.policy[1, 59] = 10
    bond.policy[1, 60] = 10
    bond.policy[1, 61] = 10
    bond.policy[1, 62] = 10
    bond.policy[2, 63] = 10
    bond.policy[2, 52] = 10
    bond.policy[2, 41] = 10
    bond.policy[2, 30] = 10
    bond.policy /= np.sum(bond.policy, axis=0)  # normalize to make dist.

    # print initial policy
    print(np.reshape(np.round(bond.policy[0], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[1], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[2], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[3], 2), (9,11)))

    # perform inference
    bond.learn(nconv=0.001, progress=True)  # 0 -> till convergence

    # print final policy
    print(np.reshape(np.round(bond.policy[0], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[1], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[2], 2), (9,11)))
    print(np.reshape(np.round(bond.policy[3], 2), (9,11)))

if __name__ == "__main__":
    main()
