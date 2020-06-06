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


class SIR_Problem(ABC_problems.ABC_Problem):

    '''
    SIR epidemic model with two parameters. Please see ICML18' - Black-box Variational Inference for Stochastic Differential Equations
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                                       # number of parameter samples
        self.n = n                                                                       # number of data samples in each simulation

        self.prior = [distributions.uniform, distributions.uniform]
        self.prior_args = np.array([[1.5, 2.4], [0.2, 0.6]])
        self.simulator_args = ['theta1', 'theta2']                                       # just for information
        self.K = 2                                                                       # number of parameters

        self.true_theta1 = 1.75
        self.true_theta2 = 0.475
        self.T = 8

    def get_true_theta(self):
        return np.array([self.true_theta1, self.true_theta2])

    def statistics(self, data, theta=None):
        # some preparation
        n_stat_per_dim = int(self.n/4)
        stat = np.zeros([1, 2*n_stat_per_dim])
        x, y = data[:, 0].copy(), data[:, 1].copy()

        # take (x_t, y_t) every 2 timestamps
        idx = np.linspace(1, self.n-1, n_stat_per_dim).astype(int)
        xx = x[idx]
        yy = y[idx]

        stat[:, 0:n_stat_per_dim] = xx
        stat[:, n_stat_per_dim:2*n_stat_per_dim] = yy
        return stat

    def simulator(self, theta):
        # get the params
        theta1 = theta[0]
        theta2 = theta[1]

        # noises
        T, d = self.T, self.n+1
        dt = T/d

        # data
        x = np.zeros([self.n, 2])
        x[0, 0], x[0, 1] = 100, 1
        N = x[0, 0] + x[0, 1]
        eps = distributions.normal_nd.draw_samples(mean=[0,0], cov=np.array([[1, 0], [0, 1]]), N=self.n)
        for t in range(self.n-1):
            S, I = x[t, 0], x[t, 1]
            s, i = S/N, I/N
            a, b = theta1*s*i, theta2*i
            alpha = np.array([-a, a - b])
            beta_sqrt = np.array([[a**0.5, 0], [-a**0.5, b**0.5]])
            dx = alpha*dt + (1/N**0.5)*(dt**0.5)*np.matmul(beta_sqrt, eps[t, :])
            tmp = x[t, :] + dx*N
            x[t+1, :] = np.log(np.exp(tmp)+1)
        return x/N

    def sample_from_prior(self):
        sample_theta1 = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        sample_theta2 = self.prior[1].draw_samples(self.prior_args[1, 0], self.prior_args[1, 1],  1)[0]
        return np.array([sample_theta1, sample_theta2])
    
    def visualize(self):
        plt.figure()
        t = np.linspace(0, self.n, self.n).astype(int)
        plt.plot(t, self.data_obs[:, 0], '-',mfc='none', color='r', label='S')
        plt.plot(t, self.data_obs[:, 1], '-',mfc='none', color='b', label='I')
        plt.plot(t, self.data_obs[:, 0], 'o',mfc='none', color='r')
        plt.plot(t, self.data_obs[:, 1], '^',mfc='none', color='b')
        plt.xlabel('time t')
        plt.ylabel('data x')
        plt.legend()
        plt.show()