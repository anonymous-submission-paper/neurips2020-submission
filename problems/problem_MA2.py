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


class MA2_Problem(ABC_problems.ABC_Problem):

    '''
    The MA2 problem with two parameters: y_t = w_t + theta1 * w_(t-1) + theta2 * w_(t-2)
    '''

    def __init__(self, N=100, n=50):

        self.N = N                                                                       # number of parameter samples
        self.n = n                                                                       # number of data samples in each simulation

        self.prior = [distributions.uniform, distributions.uniform]
        self.prior_args = np.array([[0, 1], [0, 1]])
        self.simulator_args = ['theta1', 'theta2']                                       # just for information
        self.K = 2                                                                       # number of parameters

        self.true_theta1 = 0.60
        self.true_theta2 = 0.20

    def get_true_theta(self):
        return np.array([self.true_theta1, self.true_theta2])

    def statistics(self, data, theta=None):
        return data.reshape(1, -1)

    def simulator(self, theta):
        # get the params
        theta1 = theta[0]
        theta2 = theta[1]

        # noises
        w = np.atleast_2d(distributions.normal.draw_samples(0, 1, self.n)).T
        
        # data
        assert self.n > 2
        x = np.zeros([self.n, 1])
        x[0, :] = w[0:1, :]
        x[1, :] = w[1:2, :] + theta1 * w[0:1, :]
        x[2:, :] = w[2:, :] + theta1 * w[1:-1, :] + theta2 * w[:-2, :]
        return x

    def log_likelihood(self, theta):
        # get the params
        theta1 = theta[0]
        theta2 = theta[1]

        assert self.n > 2
        x = self.data_obs
        z = np.zeros((self.n, 1))
        z[0, 0] = x[0, 0]
        z[1, 0] = x[1, 0] - theta1*z[0, 0]
        
        # solve z first
        for t in range(self.n-2):
            j = t+2
            z[j, 0] = x[j,:] - theta1*z[j-1, 0] - theta2*z[j-2, 0]
        
        # p(x) = \prod p(x_d|x<d) 
        ret = 0
        for j in range(self.n):
            ret += distributions.normal.logpdf(z[j, 0], 0., 1.)
        return ret

    def sample_from_prior(self):
        sample_theta1 = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        sample_theta2 = self.prior[1].draw_samples(self.prior_args[1, 0], self.prior_args[1, 1],  1)[0]
        return np.array([sample_theta1, sample_theta2])

    def visualize(self):
        plt.figure()
        t = np.linspace(0, self.n, self.n).astype(int)
        plt.plot(t, self.data_obs, '-',mfc='none', color='darkviolet')
        plt.xlabel('time t')
        plt.ylabel('data y')
        plt.show()