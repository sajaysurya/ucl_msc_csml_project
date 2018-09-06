'''
canvas - starting point of execution
'''
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import agent
import chain as fland


def get_state(obs):
    '''
    finds the "O" on the board and return the unraveled index
    '''
    shape = obs[0].board.shape
    loc = np.where(obs[0].layers['O'])
    loc = np.squeeze(loc)
    index = np.ravel_multi_index(loc, shape)
    return index


# pretify
np.set_printoptions(precision=2)
sns.set_style('whitegrid')

# parameters
len_episodes = 100  # TODO: vary
num_episodes = 10  # TODO: vary
num_states = 5
num_actions = 2
discount = 1  # TODO: vary
num_experiments = 10  # TODO: vary
reward_data = []
# reward = r(s,a) = n(s) x n(a) matrix
# expected reward - as reward is also stochastic)
reward = np.array([[0.4, 1.6],
                   [0.4, 1.6],
                   [0.4, 1.6],
                   [0.4, 1.6],
                   [8.8, 3.6]])
# start_dist (always starts in state 0)
start_dist = np.array([1, 0, 0, 0, 0])

print("Searching for a policy using MLEM")
for _ in tqdm(range(num_experiments)):
    # make agent bond
    bond = agent.VBEM(num_actions,
                      num_states,
                      discount,
                      len_episodes)
    bond.reward = reward
    bond.start_dist = start_dist
    # for every episode
    avg_reward_history = []
    for _ in range(num_episodes):
        # reset total reward per episode count
        total_reward = 0
        bond.reset()
        game = fland.make_game()
        obs = game.its_showtime()
        for _ in range(len_episodes):
            action = bond.play(get_state(obs))
            obs = game.play(action)
            total_reward += obs[1]
        # learn after the end of episode
        bond.learn(nconv=0.01, progress=True)
        # add average reward to list
        avg_reward_history.append(total_reward/len_episodes)
        # quit game
        game.play(5)
    reward_data.append(avg_reward_history)

reward_data = np.array(reward_data)
# plot rewards
print("Improvement in average reward per episode")
sns.pointplot(data=reward_data, ci=90)
plt.show()
print("The current best policy is")
print(np.round(bond.policy, 2))
