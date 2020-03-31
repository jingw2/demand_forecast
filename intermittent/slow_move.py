#!/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Probabilistic Forecasting: Slow-Moving Modeling Approach
Paper: https://isiarticles.com/bundles/Article/pre/pdf/20710.pdf

Author: Jing Wang
Email: jingw2@foxmail.com
Date: 03/30/2020
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma, factorial, loggamma
from scipy.stats import nbinom
import math
import random
import matplotlib.pyplot as plt 
import collections

# random.seed(123)
# np.random.seed(123)

class SlowMoveForcaster:

    def __init__(self, 
        distribution="negative_binomial",
        recursive_method="damped"):
    
        self.distribution = distribution
        self.recursive_method = recursive_method
    
    def fit(self, x):
        x = x.ravel()
        self.x = x 
        mu, long_run_mean = x[0], x[0]
        if mu == 0:
            mu = 1
        theta = [2, 0.1, 0.1]
        epsilon = 1e-2
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - epsilon}, 
            {'type': 'ineq', 'fun': lambda x: x[1] - epsilon},
            {'type': 'ineq', 'fun': lambda x: x[2] - epsilon},
            {'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + 1})

        wopt = minimize(
                        fun=self.cost,
                        x0=theta, 
                        method='SLSQP', # SLSQP, COBYLA
                        args=(x, mu, long_run_mean),
                        constraints=cons
                    )
        self.theta_opt = wopt.x
        res, sample_values, self.long_run_mean, self.mu = \
            negative_binomial_logprob(self.theta_opt, x, mu, long_run_mean, is_predict=True)
        self.counter = len(x) - 1 
        self.update(x[-1])
        y50 = [np.mean(s) for s in sample_values]
        y25 = [np.quantile(s, 0.25) for s in sample_values]
        y90 = [np.quantile(s, 0.9) for s in sample_values]
        return y50, y25, y90

    def cost(self, theta, x, mu, long_run_mean):
        res, sample_values, long_run_mean, mu = \
            negative_binomial_logprob(theta, x, mu, long_run_mean)
        return res 

    def predict(self, size=100):
        b, alpha, phi = self.theta_opt
        a = b * self.mu 
        p = b / (1 + b)
        rv = nbinom(a, p).rvs(size=size)
        y50 = np.mean(rv)
        y25 = np.quantile(rv, 0.25)
        y90 = np.quantile(rv, 0.9)
        return y50, y25, y90
    
    def update(self, x):
        b, alpha, phi = self.theta_opt
        mu = (1 - phi - alpha) * self.long_run_mean + phi * self.mu + alpha * x
        long_run_mean = (self.long_run_mean * self.counter + x) / (self.counter + 1)
        self.counter += 1 
        self.mu = mu 
        self.long_run_mean = long_run_mean

def negative_binomial_logprob(theta, x, mu, long_run_mean, is_predict=False):
    likelihood = 0
    b, alpha, phi = theta
    p = b / (1 + b)
    sample_values = []
    for i in range(1, len(x)):
        mu = (1 - phi - alpha) * long_run_mean + phi * mu + alpha * x[i-1]
        a = b * mu 
        y = x[i]
        if is_predict:
            rv = nbinom(a, p).rvs(size=500)
            sample_values.append(rv)
        logprob = (loggamma(a+y) - loggamma(a) - loggamma(y+1)) + \
            a * (math.log(b) - math.log(1+b)) - y * math.log(1+b)
        likelihood += logprob
        long_run_mean = (long_run_mean * i + x[i]) / (i+1)
    res = -likelihood / len(x)
    return res, sample_values, long_run_mean, mu

def example(
    num_points=200,
    num_demand_points=20,
    low=1,
    high=20,
    train_ratio=0.7,
    num_sample_points=500
    ):
    idxs = random.sample(range(num_points), num_demand_points)
    # ts = np.insert(a, idxs, val)
    ts = np.random.randint(size=(num_points,), low=low, high=high)
    ts[idxs] = 0
    smf = SlowMoveForcaster()
    train_size = int(num_points * train_ratio)
    test_size = num_points - train_size
    y50, y25, y90 = smf.fit(ts[:train_size])
    ypred, ypred25, ypred90 = [], [], []
    for _ in range(test_size):
        p50, p25, p90 = smf.predict(size=num_sample_points)
        ypred.append(p50)
        ypred25.append(p25)
        ypred90.append(p90)
    plt.plot(range(1, len(ts)), ts[1:], "b-")
    plt.plot(range(1, len(ts[:train_size])), y50, "r-")
    plt.fill_between(x=range(1, len(ts[:train_size])), \
            y1=y25, y2=y90, alpha=0.2, color="orange")
    plt.plot(range(train_size, len(ts)), ypred, "g-")
    plt.fill_between(x=range(train_size, len(ts)), \
            y1=ypred25, y2=ypred90, alpha=0.2, color="red")
    ymin, ymax = plt.ylim()
    plt.vlines(train_size, ymin, ymax, color="k", linestyles="dashed", linewidth=2)
    plt.ylim(ymin, ymax)
    plt.legend(["true", "yfit"])
    plt.show()

if __name__ == "__main__":
    example()
