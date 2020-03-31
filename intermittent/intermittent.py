#!/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Probabilistic Forecasting: Model for Intermitent Demand

Author: Jing Wang
Email: jingw2@foxmail.com
Date: 03/23/2020
'''

import numpy as np 
import random 
import croston 
import adjust_croston
import kalman
import matplotlib.pyplot as plt
import slow_move
import scipy.stats as st 
import math 

class IntermitentForcaster:

    @staticmethod
    def croston(y, forecast_period):
        fit_pred = croston.fit(y, forecast_period)
        # model setup
        # model = fit_pred['crosto']
        ypred = fit_pred['croston_forecast']
        yfit = fit_pred['croston_fittedvalues']
        return yfit, ypred 
    
    @staticmethod
    def adjust_croston(y, forecast_period, forecast_hyperbolic=True):
        fit_pred = adjust_croston.fit(y, forecast_period, forecast_hyperbolic)
        # model setup
        # model = fit_pred['crosto']
        ypred = fit_pred['croston_forecast']
        yfit = fit_pred['croston_fittedvalues']
        return yfit, ypred 
    
    @staticmethod
    def kalman_filter(y, forecast_period, F=None, H=None, Q=None, R=None):
        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)
        yfit = []
        ypred = []
        kf = kalman.KalmanFilter(F, Q, H, R)
        kf.fit(y)
        for mu in kf.mus:
            yfit.append(H.dot(mu)[0])
        
        for mu, var in zip(kf.mus, kf.Vs):
            st.norm.cdf(0.25, loc=mu, scale=math.sqrt(var))
        for _ in range(forecast_period):
            ypred.append(H.dot(kf.predict()))
        yfit = np.asarray(yfit).reshape((-1, 1))
        ypred = np.asarray(ypred).reshape((-1, 1))
        return yfit, ypred
    
    @staticmethod
    def kalman_smoother(y, forecast_period, F=None, H=None, Q=None, R=None):
        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)
        yfit = []
        ypred = []
        ks = kalman.KalmanSmoother(F, Q, H, R)
        ks.fit(y)
        for mu in ks.mus:
            yfit.append(H.dot(mu)[0])
        for _ in range(forecast_period):
            ypred.append(H.dot(ks.predict()))
        yfit = np.asarray(yfit).reshape((-1, 1))
        ypred = np.asarray(ypred).reshape((-1, 1))
        return yfit, ypred
    
    @staticmethod
    def slow_mover(y, forecast_period, num_sample_points=200):
        smf = slow_move.SlowMoveForcaster()
        y50, y25, y90 = smf.fit(y)
        ypred, ypred25, ypred90 = [], [], []
        for _ in range(forecast_period):
            p50, p25, p90 = smf.predict(size=num_sample_points)
            ypred.append(p50)
            ypred25.append(p25)
            ypred90.append(p90)
        return y50, ypred

if __name__ == "__main__":
    # a = np.zeros(100) 
    # val = np.array(random.sample(range(10,200), 20)) 
    num_points = 100
    idxs = random.sample(range(num_points), 60)
    # ts = np.insert(a, idxs, val)
    tot_ts = np.random.randint(size=(100,), low=1, high=20)
    tot_ts[idxs] = 0
    train_size = 80
    ts = tot_ts[:train_size]
    forecast_period = num_points - train_size
    yfit, ypred = IntermitentForcaster.adjust_croston(ts, forecast_period)
    yhat1 = np.concatenate([yfit, ypred])

    yfit, ypred = IntermitentForcaster.adjust_croston(ts, forecast_period, False)
    yhat2 = np.concatenate([yfit, ypred])

    yfit, ypred = IntermitentForcaster.croston(ts, forecast_period)
    yhat3 = np.concatenate([yfit, ypred])

    yfit, ypred = IntermitentForcaster.kalman_filter(ts, forecast_period)
    yhat4 = np.concatenate([yfit, ypred])

    yfit, ypred = IntermitentForcaster.kalman_smoother(ts, forecast_period)
    yhat5 = np.concatenate([yfit, ypred])

    yfit, ypred = IntermitentForcaster.slow_mover(ts, forecast_period)
    yhat6 = np.concatenate([yfit, ypred])

    plt.plot(tot_ts) 
    plt.plot(yhat1)
    plt.plot(yhat2)
    plt.plot(yhat3)
    plt.plot(yhat4)
    plt.plot(yhat5)
    plt.plot(yhat6)
    plt.legend(["True", "adjusted_croston_hyper", "adjust_croston", "croston", "filter", "smoother", "slow_move"])
    ymin, ymax = plt.ylim()
    plt.vlines(train_size, ymin, ymax, color="k", linestyles="dashed", linewidth=2)
    plt.ylim(ymin, ymax)
    plt.show()
