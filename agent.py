'''
contains all agents under consideration
'''
import os
import sys
import ipdb
from tqdm import tqdm
import numpy as np
from scipy.special import digamma
import tensorflow as tf
from tensorflow_probability import distributions as tfd
# disable compile option reminder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# disable warnings about einsum
tf.logging.set_verbosity(tf.logging.ERROR)


class MDP_GD:
    '''
    agent that learns using gradient descent
    can only solve MDP. cannot be used for RL
    '''
    def __init__(self,
                 num_actions,
                 num_states,
                 discount,
                 len_episodes):
        '''
        initialized the agent
        '''
        # parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.len_episodes = len_episodes

        # initialize rewards randomly
        self.reward = np.zeros((num_states, num_actions))

        # initializing distributions
        # policy = p(a|s) = n(a) x n(s) matrix
        self.policy = np.random.random((num_actions, num_states))
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.

        # priors = uniform prior with all alpha values as 1
        # theta = transition dist param = n(s) x n(s) x n(a) matrix
        self.alpha = np.ones((num_states, num_states, num_actions))
        # theta = transition distribution = p(s'|s,a) = n(s) x n(s) x n(a) mat.
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize
        # uniform prior for start distribution
        self.start_dist = np.ones(num_states)
        self.start_dist /= np.sum(self.start_dist)

    def update_theta(self):
        '''
        finds and stores theta using alpha
        call after updating alpha
        '''
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize

    def expected_reward(self, reward, theta, start_dist, policy, discount):
        '''
        takes in reward, theta, start_dist, policy, discount
        and returns the expected reward
        '''
        g_fun = reward
        for _ in range(self.len_episodes-1):
            g_fun = reward + discount * tf.einsum(
                'ij,ji,jxy->xy',
                # take abs and normalize to make it a dist
                tf.abs(policy)/tf.reduce_sum(tf.abs(policy), axis=0),
                g_fun,
                theta)
        return tf.einsum('s,as,sa->',
                         start_dist,
                         # take abs and normalize to make it a dist
                         tf.abs(policy)/tf.reduce_sum(tf.abs(policy), axis=0),
                         g_fun)

    def build(self, step_size):
        '''
        to build the tf compute graph
        tf ops are used only in this space
        '''
        # reset the graph space
        tf.reset_default_graph()
        # make a copy of reward, theta, start_dist
        reward = tf.constant(self.reward, dtype=tf.float32)
        theta = tf.constant(self.theta, dtype=tf.float32)
        start_dist = tf.constant(self.start_dist, dtype=tf.float32)
        discount = tf.constant(self.discount, dtype=tf.float32)
        # create variable and initialize with policy
        self.tfpolicy = tf.get_variable(
            name="policy",
            shape=self.policy.shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(self.policy))
        self.neg_exp_reward = -1 * self.expected_reward(
            reward,
            theta,
            start_dist,
            self.tfpolicy,
            discount)
        self.update = tf.train.GradientDescentOptimizer(
            learning_rate=step_size).minimize(self.neg_exp_reward)
        self.init = tf.global_variables_initializer()

    def learn(self, niter=None, nconv=0.01, progress=False, step_size=0.001):
        '''
        calls the build function to build the graph
        uses gradient descent to
        if nconv is given looks for convergence in expected_reward
        '''
        # build the tf compute graph
        self.build(step_size=step_size)
        # run the optimization update
        with tf.Session() as sess:
            # initialize the variables
            sess.run(self.init)
            # if progress has to be shown
            if progress:
                # GD updates for policy
                if niter is None:  # update till convergence
                    norm = np.inf
                    pbar = tqdm()
                    while norm > nconv:
                        past_rew = sess.run(self.neg_exp_reward)
                        sess.run(self.update)
                        new_rew = sess.run(self.neg_exp_reward)
                        norm = np.absolute(past_rew - new_rew)
                        pbar.update(1)
                    pbar.close()
                else:  # update for given number of steps
                    for _ in tqdm(range(int(niter))):
                        sess.run(self.update)
            else:
                # GD updates for policy
                if niter is None:  # update till convergence
                    norm = np.inf
                    while norm > nconv:
                        past_rew = sess.run(self.neg_exp_reward)
                        sess.run(self.update)
                        new_rew = sess.run(self.neg_exp_reward)
                        norm = np.absolute(past_rew - new_rew)
                else:  # update for given number of steps
                    for _ in range(int(niter)):
                        sess.run(self.update)

            # give back the best policy
            self.policy = sess.run(tf.abs(self.tfpolicy)/tf.reduce_sum(tf.abs(self.tfpolicy), axis=0))

    def play(self, state):
        '''
        just to select an action
        '''
        action = np.random.choice(self.num_actions,
                                  p=self.policy[:, state])
        return action


class VBEM:
    '''
    agent that learns using EM with variational E-step
    '''
    def __init__(self,
                 num_actions,
                 num_states,
                 discount,
                 len_episodes):
        '''
        initialized the agent
        '''
        # parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.len_episodes = len_episodes

        # buffers
        self.last_state = None
        self.last_action = None

        # initialize rewards randomly
        self.reward = np.zeros((num_states, num_actions))

        # initializing distributions
        # policy = p(a|s) = n(a) x n(s) matrix
        self.policy = np.random.random((num_actions, num_states))
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        # alpha messages
        self.a_mes = np.zeros((len_episodes, num_states))
        # beta messages
        self.b_mes = np.zeros((len_episodes, num_states, num_actions))
        # priors = uniform prior with all plpha values as 1
        # theta = transition dist param = n(s) x n(s) x n(a) matrix
        self.alpha = np.ones((num_states, num_states, num_actions))
        # theta = transition distribution = p(s'|s,a) = n(s) x n(s) x n(a) mat.
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize
        # uniform prior for start distribution
        self.start_dist = np.ones(num_states)
        self.start_dist /= np.sum(self.start_dist)

    def reset(self):
        '''
        call at the beginning of every episode
        to flush last state/action buffers
        '''
        self.last_state = None
        self.last_action = None

    def update_theta(self):
        '''
        dummy function
        functionality not needed as it directly works with alphs
        '''
        print("Its unnecessary to update theta in VBEM")
        pass

    def estep(self):
        '''
        the variational e-step
        '''
        # initialize local messages
        # alpha messages
        a_mes = np.zeros((self.len_episodes-1, self.num_states))
        # beta messages
        b_mes = np.zeros((self.len_episodes-1, self.num_states))
        # start by assuming that omega is 0
        # omega = (r) in paper = n(s') x n(s) x n(a) matrix
        omega = np.zeros((self.num_states, self.num_states, self.num_actions))
        # loop till convergence
        for _ in range(10):  # TODO: change to convergence check
            # initialize the q distribution
            q_dist = np.zeros((self.num_states,
                               self.num_states,
                               self.num_actions,
                               self.len_episodes-1,
                               self.len_episodes-1))
            # calculate explog (with current values of alpha and omega)
            explog = np.exp(
                digamma(self.alpha+omega) -
                digamma(np.sum(self.alpha+omega, axis=0))
                )
            # calculate local a_mes and b_mes
            for i in range(self.len_episodes-1):  # note reduced length
                if i == 0:
                    b_mes[i, :] = np.einsum('as,sa->s',
                                            self.policy,
                                            self.reward)
                    a_mes[i, :] = self.start_dist
                else:
                    b_mes[i, :] = np.einsum('i,ijk,kj->j',
                                            b_mes[i-1, :],
                                            explog,
                                            self.policy)

                    a_mes[i, :] = np.einsum('xij,ji,i->x',
                                            explog,
                                            self.policy,
                                            a_mes[i-1, :])

            # calculate q distirubtion q(s',s,a,tau,t)
            for time in range(1, self.len_episodes):
                q_dist[:, :, :, 0:time, time-1] = self.discount**(time) * np.einsum(
                    'tp,ap,npa,tn->npat',
                    a_mes[:time, :],
                    self.policy,
                    explog,
                    np.flip(b_mes[:time, :], axis=0))

            # normalize the q distribution
            q_dist /= np.sum(q_dist)

            # calculate omega
            omega = np.einsum(
                'npaut->npa',  # sum across tau and t
                q_dist)

        # finally calculate global a_mes and b_mes for mstep
        for i in range(self.len_episodes):
            if i == 0:
                self.b_mes[i, :, :] = self.reward
                self.a_mes[i, :] = self.start_dist
            else:
                self.b_mes[i, :, :] = np.einsum('ij,ixy,ji->xy',
                                                self.b_mes[i-1, :, :],
                                                explog,
                                                self.policy)

                self.a_mes[i, :] = np.einsum('xij,ji,i->x',
                                             explog,
                                             self.policy,
                                             self.a_mes[i-1, :])

    def mstep(self):
        '''
        the m-step
        update the policy
        '''
        # take a copy of old policy for comparison
        old = np.copy(self.policy)
        # calculate summation over t and tau
        sum_mat = np.zeros((self.num_actions, self.num_states))
        for time in range(1, self.len_episodes+1):
            sum_mat += (self.discount**(time-1) *
                        np.einsum('txa,tx->ax',
                                  np.flip(self.b_mes[:time, :, :], axis=0),
                                  self.a_mes[:time, :]))
        self.policy *= sum_mat
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        return np.linalg.norm(old-self.policy, ord=1)

    def learn(self, niter=None, nconv=0.01, progress=False):
        '''
        here the actual learning (theta and policy updates) happens
        1. theta is updated based on counts
            - add counts to alpha (while playing during episode)
            - normalize by calling update_theta
        2. EM updates in a loop till policy converges
        '''
        if progress:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                pbar = tqdm()
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
                    pbar.update(1)
                pbar.close()
            else:  # update for given number of steps
                for _ in tqdm(range(int(niter))):
                    self.estep()
                    self.mstep()
        else:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
            else:  # update for given number of steps
                for _ in range(int(niter)):
                    self.estep()
                    self.mstep()
    def play(self, state):
        '''
        given the current state, it returns the best action
        as per the current policy
        '''
        action = np.random.choice(self.num_actions,
                                  p=self.policy[:, state])
        if (self.last_state is not None) and (self.last_action is not None):
            # update count (wont be updated the first time)
            self.alpha[state, self.last_state, self.last_action] += 1

        # update last action and state
        self.last_state = state
        self.last_action = action

        return action


class PoliQ:
    '''
    agent that learns using policy iteration
    or Q -learning
    (in my own wierd equations)
    '''
    def __init__(self,
                 num_actions,
                 num_states,
                 discount,
                 len_episodes):
        '''
        initialize the agent
        '''
        # parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.len_episodes = len_episodes

        # buffers
        self.last_state = None
        self.last_action = None

        # initialize rewards as zeros
        self.reward = np.zeros((num_states, num_actions))

        # initialize values randomly
        self.value = np.random.random((num_states, num_actions))

        # policy = d(s) vector (has action values as entries)
        self.policy = np.argmax(self.value, axis=1)

        # initializing distributions
        # priors = uniform prior with all plpha values as 1
        # theta = transition dist param = n(s) x n(s) x n(a) matrix
        self.alpha = np.ones((num_states, num_states, num_actions))
        # theta = transition distribution = p(s'|s,a) = n(s) x n(s) x n(a) mat.
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize

    def update_theta(self):
        '''
        finds and stores theta using alpha
        call after updating alpha
        '''
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize

    def reset(self):
        '''
        call at the beginning of every episode
        to flush last state/action buffers
        '''
        self.last_state = None
        self.last_action = None

    def estep(self):
        '''
        the e-step of policy iteration
        does one update for values, not till convergence
        '''
        # make a copy of old value function
        old = np.copy(self.value)
        # create necessary matrices
        f_vec = self.value[np.arange(self.num_states), self.policy]
        r_vec = self.reward[np.arange(self.num_states), self.policy]
        # update values
        #ipdb.set_trace()
        self.value = self.reward + np.einsum(
            'ijk,i->jk',
            self.theta,
            self.discount*f_vec)
        return np.linalg.norm(old-self.value, ord=1)

    def mstep(self):
        '''
        the m-step of policy iteration
        simply takes argmax to find policy
        '''
        self.policy = np.argmax(self.value, axis=1)

    def learn(self, niter=None, nconv=0.01, progress=False):
        '''
        policy iteration happens here
        '''
        if progress:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                pbar = tqdm()
                while norm > nconv:
                    self.mstep()
                    norm = self.estep()
                    pbar.update(1)
                pbar.close()
            else:  # update for given number of steps
                for _ in tqdm(range(int(niter))):
                    self.mstep()
                    self.estep()
        else:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                while norm > nconv:
                    self.mstep()
                    norm = self.estep()
            else:  # update for given number of steps
                for _ in range(int(niter)):
                    self.mstep()
                    self.estep()

    def qlearn(self):
        '''
        code for q-learning
        '''
        sys.exit("Q-learning not yet implemented")

    def play(self, state):
        '''
        TO BE used only with Q-learning
        given the current state, it returns the best action
        as per the current policy
        '''
        action = self.policy[state]
        if (self.last_state is not None) and (self.last_action is not None):
            # update count (wont be updated the first time)
            self.alpha[state, self.last_state, self.last_action] += 1

        # update last action and state
        self.last_state = state
        self.last_action = action

        return action


class MLEM:
    '''
    agent that learns using traditional maximum likelihood based
    expectation maximation algorithm
    '''
    def __init__(self,
                 num_actions,
                 num_states,
                 discount,
                 len_episodes):
        '''
        initialized the agent
        '''
        # parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.len_episodes = len_episodes

        # buffers
        self.last_state = None
        self.last_action = None

        # initialize rewards randomly
        self.reward = np.zeros((num_states, num_actions))

        # initializing distributions
        # policy = p(a|s) = n(a) x n(s) matrix
        self.policy = np.random.random((num_actions, num_states))
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        # alpha messages
        self.a_mes = np.zeros((len_episodes, num_states))
        # beta messages
        self.b_mes = np.zeros((len_episodes, num_states, num_actions))
        # priors = uniform prior with all plpha values as 1
        # theta = transition dist param = n(s) x n(s) x n(a) matrix
        self.alpha = np.ones((num_states, num_states, num_actions))
        # theta = transition distribution = p(s'|s,a) = n(s) x n(s) x n(a) mat.
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize
        # uniform prior for start distribution
        self.start_dist = np.ones(num_states)
        self.start_dist /= np.sum(self.start_dist)

    def update_theta(self):
        '''
        finds and stores theta using alpha
        call after updating alpha
        '''
        self.theta = self.alpha / np.sum(self.alpha, axis=0)  # normalize

    def reset(self):
        '''
        call at the beginning of every episode
        to flush last state/action buffers
        '''
        self.last_state = None
        self.last_action = None

    def estep(self):
        '''
        the e-step
        calculate alpha and beta messages
        '''
        for i in range(self.len_episodes):
            if i == 0:
                self.b_mes[i, :, :] = self.reward
                self.a_mes[i, :] = self.start_dist
            else:
                self.b_mes[i, :, :] = np.einsum('ij,ixy,ji->xy',
                                                self.b_mes[i-1, :, :],
                                                self.theta,
                                                self.policy)

                self.a_mes[i, :] = np.einsum('xij,ji,i->x',
                                             self.theta,
                                             self.policy,
                                             self.a_mes[i-1, :])

    def mstep(self):
        '''
        the m-step
        update the policy
        '''
        # make a copy of old matrix for comparison
        old = np.copy(self.policy)
        # calculate summation over t and tau
        sum_mat = np.zeros((self.num_actions, self.num_states))
        for time in range(1, self.len_episodes+1):
            sum_mat += (self.discount**(time-1) *
                        np.einsum('txa,tx->ax',
                                  np.flip(self.b_mes[:time, :, :], axis=0),
                                  self.a_mes[:time, :]))
        self.policy *= sum_mat
        # for some state every action is useless - put uniform probability there
        self.policy += 1e-300  # very small number, wont affect other probabilities
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        return np.linalg.norm(old-self.policy, ord=1)

    def learn(self, niter=None, nconv=0.01, progress=False):
        '''
        here the actual learning (theta and policy updates) happens
        1. theta is updated based on counts
            - add counts to alpha (while playing during episode)
            - normalize by calling update_theta
        2. EM updates in a loop till policy converges
        '''
        # update theta
        self.update_theta()
        if progress:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                pbar = tqdm()
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
                    pbar.update(1)
                pbar.close()
            else:  # update for given number of steps
                for _ in tqdm(range(int(niter))):
                    self.estep()
                    self.mstep()
        else:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
            else:  # update for given number of steps
                for _ in range(int(niter)):
                    self.estep()
                    self.mstep()


    def play(self, state):
        '''
        given the current state, it returns the best action
        as per the current policy
        '''
        action = np.random.choice(self.num_actions,
                                  p=self.policy[:, state])
        if (self.last_state is not None) and (self.last_action is not None):
            # update count (wont be updated the first time)
            self.alpha[state, self.last_state, self.last_action] += 1

        # update last action and state
        self.last_state = state
        self.last_action = action

        return action


class MontyBay:
    '''
    agent that learns using bayesian updates
    expectation maximation algorithm
    with monte-carlo integration
    NOTE: eager must be enabled before using this
    '''
    def __init__(self,
                 num_actions,
                 num_states,
                 discount,
                 len_episodes,
                 num_sams=100):  # no of samples for montecarlo integration
        '''
        initialized the agent
        '''
        # parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.len_episodes = len_episodes
        self.num_sams = num_sams

        # buffers
        self.last_state = None
        self.last_action = None

        # initialize rewards randomly
        self.reward = np.zeros((num_states, num_actions))

        # initializing distributions
        # policy = p(a|s) = n(a) x n(s) matrix
        self.policy = np.random.random((num_actions, num_states))
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        # alpha messages
        self.a_mes = np.zeros((len_episodes, num_states, num_sams))
        # beta messages
        self.b_mes = np.zeros((len_episodes, num_states, num_actions, num_sams))
        # priors = uniform prior with all plpha values as 1
        # theta dim order is different from MLEM
        # states, action -> state (ordering expected like this by tensorflow)
        self.alpha = np.ones((num_states, num_actions, num_states))
        # uniform prior for start distribution
        self.start_dist = np.ones(num_states)
        self.start_dist /= np.sum(self.start_dist)

    def update_theta(self):
        '''
        update theta method is not required as theta is sampled in this algo
        '''
        print("Its unnecessary to update theta in MontyBay")

    def reset(self):
        '''
        call at the beginning of every episode
        to flush last state/action buffers
        '''
        self.last_state = None
        self.last_action = None

    def estep(self):
        '''
        the e-step
        calculate alpha and beta messages
        tensorflow is required for sampling dirichlet
        '''
        # using the mean (no change from EM)
        dir_dist = tfd.Dirichlet(self.alpha)
        theta = dir_dist.sample(self.num_sams)
        # perform computation
        for i in range(self.len_episodes):
            if i == 0:
                self.b_mes[i, :, :, :] = np.repeat(self.reward[..., None],
                                                   self.num_sams, -1)
                self.a_mes[i, :, :] = np.repeat(self.start_dist[..., None],
                                                self.num_sams, -1)
            else:
                self.b_mes[i, :, :, :] = np.einsum('ijo,oxyi,ji->xyo',
                                                self.b_mes[i-1, :, :, :],
                                                theta,
                                                self.policy)

                self.a_mes[i, :, :] = np.einsum('oijx,ji,io->xo',
                                             theta,
                                             self.policy,
                                             self.a_mes[i-1, :, :])

    def mstep(self):
        '''
        the m-step
        update the policy
        '''
        # make a copy of old matrix for comparison
        old = np.copy(self.policy)
        # calculate summation over t and tau
        sum_mat = np.zeros((self.num_actions, self.num_states, self.num_sams))
        for time in range(1, self.len_episodes+1):
            sum_mat += (self.discount**(time-1) *
                        np.einsum('txao,txo->axo',
                                  np.flip(self.b_mes[:time, :, :, :], axis=0),
                                  self.a_mes[:time, :, :]))
        self.policy *= np.mean(sum_mat, axis=-1)  # marginalize out theta
        # for some state every action is useless - put uniform probability there
        self.policy += 1e-300  # very small number, wont affect other probabilities
        self.policy /= np.sum(self.policy, axis=0)  # normalize to make dist.
        return np.linalg.norm(old-self.policy, ord=1)

    def learn(self, niter=None, nconv=0.01, progress=False):
        '''
        here the actual learning (theta and policy updates) happens
        1. theta is updated based on counts
            - add counts to alpha (while playing during episode)
            - normalize by calling update_theta
        2. EM updates in a loop till policy converges
        '''
        if progress:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                pbar = tqdm()
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
                    pbar.update(1)
                pbar.close()
            else:  # update for given number of steps
                for _ in tqdm(range(int(niter))):
                    self.estep()
                    self.mstep()
        else:
            # EM updates for policy
            if niter is None:  # update till convergence
                norm = np.inf
                while norm > nconv:
                    self.estep()
                    norm = self.mstep()
            else:  # update for given number of steps
                for _ in range(int(niter)):
                    self.estep()
                    self.mstep()


    def play(self, state):
        '''
        given the current state, it returns the best action
        as per the current policy
        '''
        action = np.random.choice(self.num_actions,
                                  p=self.policy[:, state])
        if (self.last_state is not None) and (self.last_action is not None):
            # update count (wont be updated the first time)
            self.alpha[self.last_state, self.last_action, state] += 1

        # update last action and state
        self.last_state = state
        self.last_action = action

        return action


def test_mdp(agent_type):
    '''
    test EM algorithm in a simple MDP (not RL)
    so reward and transition are fixed

    expected result: the agent selects action 1 in all except last step
    '''
    # parameters
    len_episodes = 100  # TODO: vary
    num_states = 5
    num_actions = 2
    discount = 0.9  # TODO: vary
    # reward = r(s,a) = n(s) x n(a) matrix
    reward = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 10]])
    # start_dist (always starts in state 0)
    start_dist = np.array([1, 0, 0, 0, 0])

    # make agent bond
    bond = agent_type(num_actions,
                      num_states,
                      discount,
                      len_episodes)
    # fixe theta (transitions) to test EM implementation
    alpha = np.empty((num_states, num_states, num_actions))
    alpha[0, :, :] = np.array(
        [[2, 8],
         [2, 8],
         [2, 8],
         [2, 8],
         [2, 8]])
    alpha[1, :, :] = np.array(
        [[8, 2],
         [0, 0],
         [0, 0],
         [2, 0],
         [0, 0]])
    alpha[2, :, :] = np.array(
        [[0, 0],
         [8, 2],
         [0, 0],
         [0, 0],
         [0, 0]])
    alpha[3, :, :] = np.array(
        [[0, 0],
         [0, 0],
         [8, 2],
         [0, 0],
         [0, 0]])
    alpha[4, :, :] = np.array(
        [[0, 0],
         [0, 0],
         [0, 0],
         [8, 2],
         [8, 2]])

    if agent_type == MontyBay:
        tf.enable_eager_execution()
        bond.alpha = np.einsum('ijk->jki', alpha)
    else:
        bond.alpha = alpha

    # set transition distribution
    bond.update_theta()
    # set reward details
    bond.reward = reward

    # print the initial random policy
    print(np.round(bond.policy, 2))

    # perform inference
    #bond.learn(nconv=0.01, progress=True)
    bond.learn(niter=1e3, progress=True)

    # print final policy
    print(np.round(bond.policy, 2))


if __name__ == "__main__":
    # perform tests
    # manually check the final policies
    # the optimum is to select 0 everywhere and at last 1
    #print("Testing MLEM")
    #test_mdp(MLEM)
    #print("Testing PoliQ")
    #test_mdp(PoliQ)
    #print("Testing VBEM")
    #test_mdp(VBEM)
    #print("Testing MDP_GD")
    #test_mdp(MDP_GD)
    print("Testing MontyBay")
    test_mdp(MontyBay)
