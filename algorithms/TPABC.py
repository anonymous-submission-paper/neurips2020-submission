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
from nn import MAF


class TP_ABC(ABC_algorithms.Base_ABC):

    '''
    True posterior approximation via <rejection sampling + sufficient stat>
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
        super(TP_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        self.posteriors = []
        
    def convert_stat(self, x):
        x = np.atleast_2d(x)
        n = len(x)
        stats = []
        for i in range(n):
            stat = self.problem.sufficient_stat(x[i])
            stats.append(stat)
        s = np.vstack(stats)
        return s
        
    def sort_samples(self):
        
        # > Sort the samples according to the corresponding |s - s^o|
        
        all_stats = self.stats
        all_samples = self.samples
        all_discrepancies = self.recompute_discrepancy(all_stats)
        self.rej_stats = np.zeros((self.num_samples, self.stats.shape[1]), float)
        self.rej_samples = np.zeros((self.num_samples, self.num_theta), float)
        idxes = np.argsort(all_discrepancies)
        for i in range(self.num_samples):
            idx = idxes[i]
            self.rej_stats[i, :], self.rej_samples[i, :] = all_stats[idx, :], all_samples[idx, :]

    def _fit(self, samples):
        if self.l == -1:
            # Gaussian
            [n, dim] = samples.shape
            mu = np.mean(samples, axis=0)
            M = np.mat(samples - mu)
            cov = 1.5*np.matmul(M.T, M)/n
            return [mu, cov]
        else:
            # Copula
            copula = distributions.copula()
            copula.fit(samples)
            return [copula]
        
    def _sample(self, distribution):
        if len(distribution) == 2:
            # Gaussian
            mu, cov = distribution[0], distribution[1]
            theta = distributions.normal_nd.draw_samples(mu, cov, 1)
        else:
            # Copula
            copula = distribution[0]
            theta = copula.sample()
        return theta.flatten()
        
    def _pdf(self, sample, distribution):
        if len(distribution) == 2:
            mu, cov = distribution[0], distribution[1]
            log_pdf = distributions.normal_nd.logpdf(sample, mu, cov)
        else:
            copula = distribution[0]
            log_pdf = copula.logpdf(sample)
        return np.exp(log_pdf)
     
    def learn_posterior(self):
        '''
            p_r(theta|x^o) ∝ p_r(theta)p(x^o|theta)
        '''
        print('> learning fake posterior ')
        while len(self.posteriors) < 10:
            posterior = self._fit(self.rej_samples)
            gc_cov = posterior[0].gc_cov
            cov_diagonal = gc_cov.diagonal()
            if np.abs(cov_diagonal.mean() - 1.0) > 0.05:
                continue
            self.posteriors.append(posterior)
            print('gc_cov=', posterior[0].gc_cov)
            print('\n')
        
    def log_likelihood(self, theta):
        '''
           > log_q_r(theta|x^o)
        '''
        A = 0.0
        for posterior in self.posteriors:
            A += self._pdf(theta, posterior)
        A = A/len(self.posteriors)
        return np.log(A + 1e-12)
    
    def sample_from_fake_posterior(self):
        return 0
    
    def sample_from_likelihood(self):
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
        
    def recompute_discrepancy(self, stats):
        new_stats = self.convert_stat(stats)
        y_obs = self.convert_stat(self.y_obs)
        [n, dim] = new_stats.shape
        discrepancies = np.zeros(n)
        for i in range(n):
            y = new_stats[i]
            discrepancies[i] = self.discrepancy(y_obs, y)
        return discrepancies
    
    def get_true_samples(self):
        n = self.hyperparams.num_samples
        true_samples = np.vstack([self.sample_from_likelihood() for i in range(n)])
        return true_samples
    
    def run(self):
        '''
           > main pipeline for the algorithm
        '''
        
        # initialization
        self.prior = self.problem.sample_from_prior
        total_num_sim = self.num_sim
        
        # two rounds learning
        ratios = [1.0, 0.0]
        self.l = 0
        self.max_ll = None
        self.num_sim = int(total_num_sim*ratios[self.l]) 
        self.simulate()
        self.sort_samples()
        self.learn_posterior() 
        print('\n')
        self.save_results()

        
        
        
        
        
