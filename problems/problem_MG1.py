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


class MG1_Problem(ABC_problems.ABC_Problem):

    '''
    The M/G/1 problem with three parameters: <service time> ~ [a, a+delta], <incoming time> ~ lambda
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                              # number of parameter samples
        self.n = n                                                              # number of data samples in each simulation

        self.prior = [distributions.uniform, distributions.uniform, distributions.uniform]
        self.prior_args = np.array([[0, 4], [2, 6], [0, 0.33]])
        self.simulator_args = ['alpha', 'delta', 'lambda']                      # just for information
        self.K = 3                                                              # number of parameters

        self.true_alpha = 1
        self.true_delta = 4
        self.true_lambda = 0.2

    def get_true_theta(self):
        return np.array([self.true_alpha, self.true_delta, self.true_lambda])

    def statistics(self, data, theta=None):
        # quantile as summary statistics
        n_quantiles = 20
        dim = data.shape[1]
        prob = np.linspace(0.025, 0.975, n_quantiles)
        stat = np.zeros([1, n_quantiles*dim])
        for k in range(dim):
            quantiles = stats.mstats.mquantiles(data[:, k], prob)
            stat_k = quantiles
            stat[0, k*n_quantiles : (k+1)*n_quantiles] = np.array(stat_k)
        return stat
 
    def simulator(self, theta):
        # get the params
        Alpha = theta[0]
        Delta = theta[1]
        Lambda = theta[2]

        # service times (uniformly distributed)
        sts = distributions.uniform.draw_samples(Alpha, Alpha + Delta, self.n)

        # interarrival times (exponentially distributed)
        iats = distributions.exponential.draw_samples(Lambda, self.n)

        # arrival times
        ats = np.cumsum(iats)

        # interdeparture and departure times
        idts = np.empty(self.n)
        dts = np.empty(self.n)

        idts[0] = sts[0] + ats[0]
        dts[0] = idts[0]

        for i in range(1, self.n):
            idts[i] = sts[i] + max(0.0, ats[i] - dts[i-1])
            dts[i] = dts[i-1] + idts[i]

        return np.atleast_2d(idts).T

    def sample_from_prior(self):
        sample_alpha = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        sample_delta = self.prior[1].draw_samples(self.prior_args[1, 0], self.prior_args[1, 1],  1)[0]
        sample_lambda = self.prior[2].draw_samples(self.prior_args[2, 0], self.prior_args[2, 1],  1)[0]
        return np.array([sample_alpha, sample_delta, sample_lambda])

    def visualize(self):

        # have a look at the problem

        plt.figure()
        t = np.linspace(0, self.n, self.n).astype(int)
        x = self.data_obs.reshape(-1)
        plt.rcParams["patch.force_edgecolor"] = True
        n, bins, patches = plt.hist(x, bins=80, facecolor='orchid', alpha=0.5)
        plt.xlabel(r'inter-departure time $\Delta t$')
        plt.ylabel(r'data y')
        plt.show()