import numpy as np
import math
import scipy.stats as stats
from abc import ABCMeta, abstractmethod
import distributions 
import utils_math
from problems import ABC_problems
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time


class LV_Problem(ABC_problems.ABC_Problem):

    '''
    The Lotka-Volterra problem with 3 parameters.
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                           # number of parameter samples
        self.n = n                                                           # number of data samples in each simulation

        self.prior = distributions.uniform
        self.prior_args = np.array([[-5,-1], [-1, 1], [-1, 1]])
        self.simulator_args = ['theta1', 'theta2', 'theta3']                 # just for information
        self.K = 3                                                           # number of parameters
        self.n_stat_per_dim = 40                                             # number of stat per each dimension

        self.true_theta1 = np.log(0.01)
        self.true_theta2 = np.log(0.5)
        self.true_theta3 = np.log(1)
        self.true_theta4 = np.log(0.01)

    def get_true_theta(self):
        return np.array([self.true_theta1, self.true_theta2, self.true_theta3])

    def statistics(self, data, theta=None):                                  # Sub-sampling time-series as S.S
        # some preparation
        states = data
        stat = np.zeros([1, 2*self.n_stat_per_dim])
        x, y = states[:, 0].copy(), states[:, 1].copy()

        # take (x_t, y_t) every ? timestamps
        idx = np.linspace(1, 40, self.n_stat_per_dim).astype(int)
        xx = x[idx]
        yy = y[idx]

        stat[:, 0:self.n_stat_per_dim] = xx
        stat[:, self.n_stat_per_dim:2*self.n_stat_per_dim] = yy
        return stat

    def simulator(self, theta):
        # set up
        init_state = [50, 100]                                                # initial population state [X1, X2]
        real_theta = np.exp(np.array([theta[0], theta[1], theta[2], theta[0]]))

        # simulate population evolution
        lv = LotkaVolterraProcess(init_state, real_theta)
        states = lv.sim_time(dt=0.2, duration=8, max_n_steps=30000)
        return states

    def sample_from_prior(self):
        sample_theta1 = self.prior.draw_samples(self.prior_args[0,0], self.prior_args[0,1],  1)[0]
        sample_theta2 = self.prior.draw_samples(self.prior_args[1,0], self.prior_args[1,1],  1)[0]
        sample_theta3 = self.prior.draw_samples(self.prior_args[2,0], self.prior_args[2,1],  1)[0]
        return np.array([sample_theta1, sample_theta2, sample_theta3])

    def visualize(self, states):
        # have a look at the problem
        plt.figure()
        t = np.linspace(0, 40, 41).astype(int)
        plt.plot(t, states[:,0], '-',mfc='none', color='blue')
        plt.plot(t, states[:,1], '-',mfc='none', color='red')

        idx = np.linspace(1, 40, self.n_stat_per_dim).astype(int)
        plt.xlabel('time t')
        plt.ylabel('number of creatures n')
        plt.plot(idx, states[idx,0], 'o',mfc='none', color='blue')
        plt.plot(idx, states[idx,1], '^',mfc='none', color='red')
        plt.legend(['predator', 'prey'])

        idx2 = np.linspace(0, 40, 11).astype(int)
        plt.xticks(idx2, idx2/5)
        plt.show()




class LotkaVolterraProcess(object):

    '''
    Util class. Implementation of the Lotka-Volterra process.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, init, params):

        self.state = np.asarray(init)
        self.params = np.asarray(params)
        self.time = 0.0

    def _calc_propensities(self):
        x, y = self.state
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):
        if reaction == 0:
            self.state[0] += 1
        elif reaction == 1:
            self.state[0] -= 1
        elif reaction == 2:
            self.state[1] += 1
        elif reaction == 3:
            self.state[1] -= 1
        else:
            raise ValueError('Unknown reaction.')

    def sim_time(self, dt, duration, max_n_steps=float('inf')):

        # > Simulates the process with Gillespie's algorithm

        num_rec = int(duration / dt) + 1
        states = np.zeros([num_rec, self.state.size])
        cur_time = self.time
        n_steps = 0

        for i in range(num_rec):
            while cur_time > self.time:

                # rate & total rate
                rates = self.params * self._calc_propensities()
                total_rate = rates.sum()

                # if predator or prey die out
                if total_rate == 0:
                    self.time = float('inf')
                    break

                # t ~ exp(total_rate)
                self.time += distributions.exponential.draw_samples(lamda=total_rate, N=1)[0]

                # sample from multinomial distribution
                reaction = utils_math.discrete_sample(rates / total_rate)[0]
                self._do_reaction(reaction)

                # step ++
                n_steps += 1
                if n_steps > max_n_steps:
                    return None

            states[i,:] = self.state.copy()
            cur_time += dt
        return np.array(states)