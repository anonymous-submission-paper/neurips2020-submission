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


class TP2_ABC(ABC_algorithms.Base_ABC):

    '''
    True posterior approximation via SNP.
    '''
    
    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        super(TP2_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_array = []
        self.proposal_array = [Uniform(self.problem.prior_args)]
        self.all_stats = []
        self.all_samples = []

    def fit_nde(self):
        print('> fitting nde')
        all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats))).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
        [n, dim] = all_stats.size()
        net = MAF.MAF(n_blocks=5, n_inputs=self.problem.K, n_hidden=50, n_cond_inputs=self.y_obs.shape[1])
        net.train().to(self.device)
        net.learn(inputs=all_samples, cond_inputs=all_stats)
        net = net.eval().cpu()
        self.nde_net = net
        self.nde_array.append(net)
                      
    def logpdf_nde(self, theta):
        n, d = theta.size()
        y_obs = self.convert_stat(self.whiten(self.y_obs))
        y_obs = torch.tensor(y_obs).float().repeat(n, 1)
        net = self.nde_array[self.l]
        log_pdf = net.log_probs(inputs=theta, cond_inputs=y_obs)
        return log_pdf
            
    def learn_proposal(self):
        '''
            p_r(theta) = argmin_{q: q \in Gaussian} KL(q_{r-1}(theta|x_o), q)
        '''
        # minimize KL
        print('> fitting proposal')
        samples = np.vstack([self.sample_from_likelihood() for i in range(250)])
        [n, dim] = samples.shape
        mu = samples.mean(axis=0, keepdims=True)
        M = np.mat(samples - mu)
        cov = 1.50*np.matmul(M.T, M)/n
        
        # add to proposal list
        mu, cov = torch.tensor(mu).float(), torch.tensor(cov).float()
        gaussian = Gaussian(dim=self.problem.K, mu=mu, cov=cov)
        gaussian.print()
        self.proposal = gaussian
        self.proposal_array.append(gaussian)
        
    def sample_from_proposal(self):
        '''
           > theta ~ p_r(theta)
        '''
        theta = self.proposal.sample().view(-1)
        return theta.detach().cpu().numpy()
    
    def log_proposal(self, theta):
        '''
           > log_p_r(theta)
        '''
        theta = torch.tensor(theta).float()
        log_prob = self.proposal.log_probs(theta)
        return log_prob.view(-1).detach().cpu().numpy()
            
    def log_accumulate_proposal(self, theta):
        '''
           > p(theta) = 1/n * ∑ p_r(theta)
        '''
        pdf = torch.zeros(len(theta))
        proposal_array = self.proposal_array[0:self.l+1]
        for proposal in proposal_array: 
            pdf += proposal.log_probs(theta).exp()
        pdf = (pdf/len(proposal_array) + 1e-10)
        return pdf.log()
                
    def log_likelihood(self, theta):
        '''
           > log_q_r(theta|x^o)
        '''
        theta = torch.tensor(theta).float().view(1, self.problem.K)
        B = self.logpdf_nde(theta).item()
        C = self.log_accumulate_proposal(theta).item()
        return B-C
    
    def sample_from_likelihood(self):
        net = self.nde_net
        net.eval()
        # pilot run
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(10000):
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
    
    def fit_mixture_nde(self):
        '''
            fit K NDE with different train/val set
        '''
        K = 2
        nde_mixture_array = []
        for _ in range(K):
            print('> fitting nde')
            all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats))).float().to(self.device)
            all_samples = torch.tensor(np.vstack(self.all_samples)).float().to(self.device)
            [n, dim] = all_stats.size()
            net = MAF.MAF(n_blocks=5, n_inputs=self.problem.K, n_hidden=50, n_cond_inputs=self.y_obs.shape[1])
            net.train().to(self.device)
            net.learn(inputs=all_samples, cond_inputs=all_stats)
            net = net.eval().cpu()
            nde_mixture_array.append(net)
        self.nde_mixture_array = nde_mixture_array
        
    def log_mixture_likelihood(self, theta):
        '''
            log p(theta|x_o) = 1/K \sum log q_{r-k}(x_o|theta)   k = 0,1,..K
        '''
        prob = 1e-12
        nde_array = self.nde_mixture_array
        K = len(nde_array)
        for l in range(K):
            self.nde_net = nde_array[l].eval()
            log_prob = self.log_likelihood(theta)
            prob += np.exp(log_prob)/K
        return np.log(prob)
        
    def get_true_samples(self):
        n = self.hyperparams.num_samples
        true_samples = np.vstack([self.sample_from_likelihood() for i in range(n)])
        return true_samples
         
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
            self.fit_nde()
            self.learn_proposal()
            self.prior = self.sample_from_proposal
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()
        
        
# ------------------------------------------------------------------------------------ #  
    

class Gaussian(torch.nn.Module):
    def __init__(self, dim, mu, cov):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.Parameter(torch.Tensor(1, dim))
        self.V = torch.nn.Parameter(torch.Tensor(dim, dim))
        self.mu.data = mu
        self.V.data = cov

    def log_probs(self, x):
        mu, V = self.mu, self.V
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(mu, V)
        return mvn.log_prob(x).view(-1)
    
    def sample(self):
        mu, V = self.mu, self.V
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(mu, V)
        return mvn.sample()
    
    def print(self):
        print('mu=', self.mu.data)
        print('cov=', self.V.data)
 
            
class Uniform(torch.nn.Module):
    def __init__(self, ranges):
        super(Uniform, self).__init__()
        K, _ = ranges.shape
        self.ranges = ranges
        self.volume = 1.0
        for k in range(K): self.volume = self.volume*(self.ranges[k,1] - self.ranges[k,0])

    def log_probs(self, x):
        n, d = x.size()
        pdf = torch.zeros(n, 1) + 1.0/self.volume
        return pdf.log().view(-1)
 
    

        
        
        
        
        
