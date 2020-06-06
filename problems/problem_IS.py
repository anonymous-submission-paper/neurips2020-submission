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


class IS_Problem(ABC_problems.ABC_Problem):

    '''
    A d*d Ising model                                                           H(x; theta) = alpha* \sum_<i,j> xi*xj + beta* \sum_i x_i 
    '''

    def __init__(self, N=100, n=1):

        self.N = N                                                              # number of posterior samples
        self.n = n                                                              # length of the data vector x = {x_1, ..., x_n}
        self.d = 8
        
        self.prior = [distributions.uniform]
        self.prior_args =  np.array([[0, 1.5]])                            
        self.simulator_args = ['theta1']                                       
        self.K = 1                                                              # number of parameters
        self.sweep = 100
        self.sampler = 'Gibbs'
        
        self.true_theta1 = 0.30
        self.true_theta2 = 0.10

    def get_true_theta(self):
        return np.array([self.true_theta1])

    def statistics(self, data, theta=None, is_sufficient=False):
        return data
    
    def sufficient_stat(self, data, dimensionality=1):                          # sufficient stat = {\sum_<i,j> xi*xj, \sum_i x_i }
        d = self.d
        x = data.reshape((d, d))
        A, B = np.zeros((d,d)), np.zeros((d,d))
        A[:, 0:d-1] = x[:, 1:d]
        B[0:d-1, :] = x[1:d, :]
        w, v = np.sum(A*x)+np.sum(B*x), np.sum(x)
        if dimensionality==1:
            stat = np.array([w])
        else:
            stat = np.array([w,v])
        return stat
        
    def simulator(self, theta):
        # get the params
        alpha = theta[0]
        beta = self.true_theta2
        
        # Metropolis-Hasting
        if self.sampler == 'MH':
            d = self.d
            x = np.sign(np.random.rand(d*d) - 0.5) 
            for _ in range(self.sweep):
                for i in range(d*d): # randomly pick one index
                    x_new, x_old = np.array(x), np.array(x)
                    idx = np.random.randint(low=0,high=d*d)
                    x_new[idx] = -x_old[idx]
                    e_old = self.energy(x_old, alpha, beta)
                    e_new = self.energy(x_new, alpha, beta)
                    e_delta = (e_new - e_old)
                    if e_delta > 0: 
                        e_u = 0.0
                        x = x_new
                    else:
                        e_u = np.log(np.random.rand(1))
                        x = x_new if e_u<e_delta else x_old
        # Gibbs sampling
        if self.sampler == 'Gibbs':     
            d = self.d
            x = np.sign(np.random.rand(d*d, 1) - 0.5) 
            for _ in range(self.sweep):
                for i in range(d*d): # sequentially select index
                    x_A, x_B = np.array(x), np.array(x)
                    x_A[i], x_B[i] = +1, -1
                    e_A = self.energy(x_A.reshape((d,d)), alpha, beta)
                    e_B = self.energy(x_B.reshape((d,d)), alpha, beta)
                    p_A = np.exp(e_A)/(np.exp(e_A)+np.exp(e_B))
                    p_B = np.exp(e_B)/(np.exp(e_A)+np.exp(e_B))
                    u = np.random.rand(1)
                    if u < p_A:
                        x = x_A
                    else:
                        x = x_B                
        return x.reshape((1,d*d))
      
    def sample_from_prior(self):
        sample_theta1 = self.prior[0].draw_samples(self.prior_args[0, 0], self.prior_args[0, 1],  1)[0]
        return np.array([sample_theta1])
    
    def energy(self, x, alpha, beta):        
        stats = self.sufficient_stat(x, dimensionality=2)
        w, v = stats[0], stats[1]
        H = alpha*w + beta*v
        return H
    
    def visualize(self):
        d = self.d
        M = self.y_obs.reshape((d,d))
        print('x_obs=', M)

        fig=plt.figure(figsize=(5,5))
        ax=fig.add_subplot(231)
        plt.title('')
        CS=plt.imshow(M, cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
        

