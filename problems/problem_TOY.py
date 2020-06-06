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


class TOY_Problem(ABC_problems.ABC_Problem):

    '''
    toy model in SNL paper
    '''

    def __init__(self, N=100, n=50):

        # self.N = N                                                                       # number of posterior samples
        self.n = n                                                                       # length of the data vector x = {x_1, ..., x_n}

        self.x_dim = 20     # in SNL paper, this is 8

        self.prior = [distributions.uniform, distributions.uniform, distributions.uniform, distributions.uniform, distributions.uniform]                      # uniform prior
        self.prior_args = np.array([[-3., 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]])
        self.simulator_args = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5']
        self.K = 5                                                                       # number of parameters

        self.true_theta1 = 0.7
        self.true_theta2 = 2.9
        self.true_theta3 = -1.
        self.true_theta4 = -0.9
        self.true_theta5 = 0.6

        # self.G = 0.5
        self.T = 10.0                                                                    # total time of the process

    def get_true_theta(self):
        return np.array([self.true_theta1, self.true_theta2, self.true_theta3, self.true_theta4, self.true_theta5])

    def statistics(self, data, theta=None):
        # idx = np.arange(self.n * 2)
        # stat = data[idx]
        return np.reshape(data, (1, -1))

    def simulator(self, theta):
        mean = np.array([theta[0], theta[1]])
        rho = np.tanh(theta[4])
        s1, s2 = theta[2] ** 2, theta[3] ** 2
        cov = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
        x = np.random.multivariate_normal(mean, cov, int(self.x_dim * self.n / 2))
        return x.reshape(self.n, self.x_dim)
        # return x.reshape(1, -1)

    def sample_from_prior(self):
        sample_theta1 = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        sample_theta2 = self.prior[1].draw_samples(self.prior_args[1, 0], self.prior_args[1, 1],  1)[0]
        sample_theta3 = self.prior[2].draw_samples(self.prior_args[2, 0], self.prior_args[2, 1],  1)[0]
        sample_theta4 = self.prior[3].draw_samples(self.prior_args[3, 0], self.prior_args[3, 1],  1)[0]
        sample_theta5 = self.prior[4].draw_samples(self.prior_args[4, 0], self.prior_args[4, 1],  1)[0]
        return np.array([sample_theta1, sample_theta2, sample_theta3, sample_theta4, sample_theta5])
    
    def log_likelihood(self, theta):
        # get the params
        mean = np.array([theta[0], theta[1]])
        rho = np.tanh(theta[4])
        s1, s2 = theta[2] ** 2, theta[3] ** 2
        cov = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])

        x = self.data_obs.reshape(int(self.x_dim * self.n / 2), 2)
        return np.log(stats.multivariate_normal.pdf(x, mean=mean, cov=cov)).sum()

    def visualize(self):
        raise NotImplementedError