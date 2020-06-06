from abc import ABCMeta, abstractmethod

import numpy as np
import torch 
import os, sys, time, math
import scipy.stats as stats
import matplotlib.pyplot as plt

import utils_math, utils_os
import distributions
import discrepancy
import algorithms.ABC_algorithms as ABC_algorithms
from copy import deepcopy
from nn import MDN, MAF


class SNP_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequential Neural Posterior Estimate (ver.B)
    '''
    
    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.
        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            discrepency: function pointer
                The data discrepency
            hyperparams: 1-D array
                The hyper-parameters [epsilon, num-samples]
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(SNP_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_array = []
        self.all_stats = []
        self.all_samples = []
        self.all_weights = []

    def fit_nde(self):
        print('> fitting nde')
        all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats))).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
        all_weights = torch.tensor(np.vstack(self.all_weights)).float().to(self.device)  
        [n, dim] = all_stats.size()
        if self.hyperparams.nde == 'MDN':
            net = MDN.MDN(n_in=self.y_obs.shape[1], n_hidden=50, n_out=self.problem.K, K=8)
        if self.hyperparams.nde == 'MAF':
            net = MAF.MAF(n_blocks=5, n_inputs=self.problem.K, n_hidden=50, n_cond_inputs=self.y_obs.shape[1])
        net.train().to(self.device)
        net.learn(inputs=all_samples, cond_inputs=all_stats, weights=all_weights)
        net = net.cpu()
        self.nde_net = net
        self.nde_array.append(net)
              
    def logpdf_nde(self, theta):
        n, d = theta.size()
        y_obs = self.convert_stat(self.whiten(self.y_obs))
        y_obs = torch.tensor(y_obs).float().repeat(n, 1)
        net = self.nde_net
        log_pdf = net.log_probs(inputs=theta, cond_inputs=y_obs)
        return log_pdf
    
    def sample_from_nde(self):
        net = self.nde_net
        net.eval()
        y_obs = self.convert_stat(self.whiten(self.y_obs))
        y_obs = torch.tensor(y_obs).float().view(1, -1)
        x = net.sample(cond_inputs=y_obs)
        return x.detach().cpu().numpy()
    
    def compute_weight(self, samples):
        '''
           > pi(theta)/p_r(theta)
        '''
        [n, d] = samples.shape
        samples = torch.tensor(samples).float()
        if self.l == 0:
            weights = torch.zeros(n, 1) + 1.0
        else:
            weights = 1.0/self.logpdf_nde(samples).exp().view(n, 1)
        return weights.detach().cpu().numpy()
        
    def log_likelihood(self, theta):
        '''
           > log_q_r(theta|x^o)
        '''
        theta = torch.tensor(theta).float().view(1, self.problem.K)
        B = self.logpdf_nde(theta).item()
        return B
              
    def run(self):
        '''
            main pipeline for the algorithm
        '''
        # initialization
        self.prior = self.problem.sample_from_prior
        self.prior_args = np.array(self.problem.prior_args)
        
        # iterations
        L = self.hyperparams.L
        total_num_sim = self.num_sim 
        self.num_sim = int(total_num_sim/L)
        for l in range(L):
            print('iteration ', l, 'prior=', self.prior_args, 'num_sim=', self.num_sim)
            self.l = l
            self.max_ll = None
            self.simulate()
            self.all_stats.append(self.stats)
            self.all_samples.append(self.samples)
            self.all_weights.append(self.compute_weight(self.samples))
            self.fit_nde()
            self.prior = self.sample_from_nde
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()