#!/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Kalman Filter and Smoother

Author: Jing Wang
Email: jingw2@foxmail.com
Date: 03/25/2020

Reference: https://jwmi.github.io/ASM/6-KalmanFilter.pdf
'''

import math 
import numpy as np 
import matplotlib.pyplot as plt 

class KalmanFilter:
    
    def __init__(self, F, Q, H, R):
        self.F, self.Q, self.H, self.R = F, Q, H, R
        self.d = self.F.shape[0]
        self.D = self.H.shape[0]

    def setup(self, x1, mu0, V0):
        K = V0.dot(self.H.T).dot(np.linalg.inv(self.H.dot(V0).dot(self.H.T) + self.R)) # d * D
        mu = mu0 + K.dot(x1 - self.H.dot(mu0)) # d 
        V = (np.eye(V0.shape[0]) - K.dot(self.H)).dot(V0)
        P = self.F.dot(V).dot(self.F.T) + self.Q # d * d 
        return (K, mu, V, P)
    
    def fit(self, x):
        mu0 = np.zeros((self.d, 1))
        V0 = np.eye((self.d))
        K, mu, V, P = self.setup(x[0], mu0, V0)
        mus = [mu]
        Vs = [V]
        Ps = [P]
        for i in range(1, len(x)):
            K = P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(P).dot(self.H.T) + self.R))
            mu = self.F.dot(mu) + K.dot(x[i] - self.H.dot(self.F.dot(mu))) # d 
            V = (np.eye(V.shape[0]) - K.dot(self.H)).dot(P)
            P = self.F.dot(V).dot(self.F.T) + self.Q  # d * d 
            mus.append(mu)
            Vs.append(V) 
            Ps.append(P)
        self.mus = mus 
        self.Vs = Vs
        self.Ps = Ps
        return mus, Vs
    
    def predict(self):
        last_mu, last_V = self.mus[-1], self.Vs[-1]
        pred = np.random.multivariate_normal(last_mu.ravel(), last_V)
        return pred 

class KalmanSmoother:

    def __init__(self, F, Q, H, R):
        self.F, self.Q, self.H, self.R = F, Q, H, R
        self.d = self.F.shape[0]
        self.D = self.H.shape[0]
    
    def fit(self, x):
        kf = KalmanFilter(self.F, self.Q, self.H, self.R)
        kf.fit(x)
        n = len(kf.mus)
        mu_hat = kf.mus[-1]
        V_hat = kf.Vs[-1]
        mus_hat = [mu_hat]
        Vs_hat = [V_hat]
        for j in range(n-2, -1, -1):
            mu = kf.mus[j]
            V = kf.Vs[j]
            P = kf.Ps[j]
            C = V.dot(self.F.T).dot(np.linalg.inv(P))
            mu_hat = mu + C.dot(mu_hat - self.F.dot(mu))
            V_hat = V + C.dot(V_hat - P).dot(C.T)
            mus_hat = [mu_hat] + mus_hat
            Vs_hat = [V_hat] + Vs_hat
        self.mus = mus_hat
        self.Vs = Vs_hat
        return mus_hat, Vs_hat
    
    def predict(self):
        last_mu, last_V = self.mus[-1], self.Vs[-1]
        pred = np.random.multivariate_normal(last_mu.ravel(), last_V)
        return pred 

def example():
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2) + np.random.normal(0, 5, 100)

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    kf_predictions = []

    kf.fit(measurements)
    for mu in kf.mus:
        kf_predictions.append(H.dot(mu)[0])
    
    ks_predictions = []
    ks = KalmanSmoother(F, Q, H, R)
    ks.fit(measurements)
    for mu in ks.mus:
        ks_predictions.append(H.dot(mu)[0])

    plt.plot(range(len(measurements)), measurements, label='Measurements')
    plt.plot(range(len(measurements)), np.array(kf_predictions), label='Kalman Filter Prediction')
    plt.plot(range(len(measurements)), np.array(ks_predictions), label='Kalman Smoother Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    example()
