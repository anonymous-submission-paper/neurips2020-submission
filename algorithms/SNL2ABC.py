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
from nn import MAF,MSN
from copy import deepcopy


class SNL2_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequential neural likelihood (with sufficient stat).
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
        super(SNL2_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_net = None                             # the learned q(x|theta)
        self.vae_net = None                             # the learned s(x)
        self.nde_array = []                             
        self.vae_array = []                             
        self.proposal_array = []                        # the proposal used at each round
        self.hidden_ratio = hyperparams.hidden_ratio    # dimensionality of s.s 
        
    def convert_stat(self, x): 
        # no autoencoder, directly return s
        if self.vae_net is None: 
            s = x
            return s
        # convert raw data to summary stat: s = S(x)
        else:
            s = self.vae_net.encode(torch.tensor(x).float())
            return s.detach().cpu().numpy()
    
    def fit_nde(self):
        print('\n > fitting nde')
        all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats))).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
        [n, dim] = all_stats.size()
        net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=50, n_cond_inputs=self.problem.K)
        net.train().to(self.device)
        net.learn(inputs=all_stats, cond_inputs=all_samples)
        net = net.eval().cpu()
        self.nde_net = net
        self.nde_array.append(net)

    def fit_vae(self):
        print('\n > fitting encoder')
        all_stats = torch.tensor(np.vstack(self.all_stats)).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
        [n, dim] = all_stats.size()
        h = int(dim*self.hidden_ratio)
        print('summary statistic dim =', h, 'original dim =', dim)
        architecture = [dim] + [100, 100, h]    
        print('architecture', architecture)
        net = MSN.MSN(architecture, dim_y=self.problem.K)
        net.train().to(self.device)
        net.learn(x=all_stats, y=all_samples)
        net = net.eval().cpu()
        self.vae_net = net
        self.vae_array.append(net)
        
    def sample_from_nde(self):
        net = self.nde_net
        net.eval()
        # pilot run
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(40000):
                theta = self.problem.sample_from_prior()
                ll = self.log_likelihood(theta)
                if ll > self.max_ll: self.max_ll = ll
        # rejection sampling
        while True:
            theta = self.problem.sample_from_prior()
            prob_accept = self.log_likelihood(theta) - self.max_ll
            u = distributions.uniform.draw_samples(0, 1, 1)[0]
            if np.log(u) < prob_accept: break
        return theta
        
    def log_likelihood(self, theta):
        '''
            log p(theta|x_o) = log q(x_o|theta)     (note: uniform prior)
        '''
        net = self.nde_net
        net.eval()
        y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
        y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
        log_probs = net.log_probs(inputs=y_obs, cond_inputs=theta)
        return log_probs.item()
    
    def log_mixture_likelihood(self, theta):
        '''
            log p(theta|x_o) = 1/K \sum log q_{r-k}(x_o|theta)   k = 0,1,..K
        '''
        K = 3
        max_idx, min_idx = self.l, max(self.l-K+1, 0)
        nde_array = self.nde_array[min_idx:max_idx+1]
        vae_array = self.vae_array[min_idx:max_idx+1]
        prob = 1e-12
        L = len(nde_array)
        for l in range(L):
            self.nde_net = nde_array[l].eval()
            self.vae_net = vae_array[l].eval()
            y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            probs = self.nde_net.log_probs(inputs=y_obs, cond_inputs=theta).exp()
            prob += probs.item()/L
        return np.log(prob)
    
    def learn_proposal(self):
        '''
            p_r(theta) = argmin_{q: q \in Gaussian} KL(q_{r-1}(theta|x_o), q)
        '''
        # minimize KL
        print('\n > fitting proposal')
        samples = np.vstack([self.sample_from_nde() for i in range(200)])
        [n, dim] = samples.shape
        mu = samples.mean(axis=0, keepdims=True)
        M = np.mat(samples - mu)
        cov = np.matmul(M.T, M)/n
        print('mu=', mu)
        print('cov=', cov)
        
        # proposal = inflated Gaussian approx to q
        alpha = 1.5
        mu, cov = torch.tensor(mu).float(), alpha*torch.tensor(cov).float()
        gaussian = torch.distributions.MultivariateNormal(mu, cov)
        self.proposal = gaussian
        self.proposal_array.append(gaussian)
        
    def sample_from_proposal(self):
        '''
           > theta ~ p_r(theta)
        '''
        theta = self.proposal.sample().view(-1)
        return theta.detach().cpu().numpy()

    def run(self):
        '''
            main pipeline for the algorithm
        '''
        # initialization
        self.prior = self.problem.sample_from_prior
        
        # iterations
        L = self.hyperparams.L
        total_num_sim = self.num_sim 
        self.num_sim = int(total_num_sim/L)
        self.all_stats = []
        self.all_samples = []
        for l in range(L):
            print('iteration ', l)
            self.l = l
            self.max_ll = None
            self.simulate()
            self.all_stats.append(self.stats)
            self.all_samples.append(self.samples)
            self.fit_vae()
            self.fit_nde()
            self.learn_proposal()
            self.prior = self.sample_from_proposal
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()

        
        
        
        
        
