#!/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Probabilistic Forecasting: Latent State Forecaster—Multi-stage Likelihood

Author: Jing Wang
Email: jingw2@foxmail.com
Date: 04/02/2020
'''

from torch import nn
import torch 
from progressbar import *
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import random 

class MultiStageGLM:

    def fit(self, X, y):
        stage1_y = y.copy()
        stage1_y[stage1_y != 0] = 1 
        self.stage1_lr = LogisticRegression()
        self.stage1_lr.fit(X, stage1_y)

        stage2_y = y[y > 0]
        stage2_y[stage2_y == 1] = 0 
        stage2_y[stage2_y != 0] = 1
        stage2_X = X[y > 0]
        self.stage2_lr = None
        if len(np.unique(stage2_y)) > 2:
            self.stage2_lr = LogisticRegression()
            self.stage2_lr.fit(stage2_X, stage2_y)

        stage3_y = y[y >= 2]
        stage3_X = X[y >= 2]
        stage3_y -= 2
        self.glm = sm.GLM(stage3_y, stage3_X, family=sm.families.Poisson())
        self.res = self.glm.fit()
    
    def predict(self, X):
        ypred = np.zeros((X.shape[0]))
        stage1_ypred = self.stage1_lr.predict(X)
        ypred[stage1_ypred == 0] = 0 

        pass_stage1_y = ypred[stage1_ypred > 0]
        stage2_X = X[stage1_ypred > 0]
        if self.stage2_lr is not None:
            stage2_ypred = self.stage2_lr.predict(stage2_X)
            pass_stage1_y[stage2_ypred == 0] = 1
        else:
            stage2_ypred = np.ones((stage2_X.shape[0]))

        stage3_X = stage2_X[stage2_ypred > 0]
        dist = self.glm.get_distribution(self.res.params, self.res.mu, exog=stage3_X)
        stage3_ypred = dist.rvs()
        pass_stage1_y[stage2_ypred > 0] = stage3_ypred + 2 
        ypred[stage1_ypred > 0] = pass_stage1_y
        return ypred 

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

# def example():

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    data = pd.read_csv("intermittent_data.csv", parse_dates=["date"])
    data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
    data["month"] = data["date"].apply(lambda x: x.month)
    data["year"] = data["date"].apply(lambda x: x.year)

    data = data[:500]
    X = np.c_[np.asarray(pd.get_dummies(data["day_of_week"])), np.asarray(pd.get_dummies(data["month"]))]
    y = np.asarray(data["sales"]).ravel()

    # num_masks = 150
    # idx = random.sample(range(len(y)), num_masks)
    # y[idx] = 0

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    train_ratio = 0.8
    train_size = int(X.shape[0] * train_ratio)
    Xtrain, ytrain, Xtest, ytest = X[:train_size], y[:train_size], X[train_size:], y[train_size:]
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    idx = list(range(Xtrain.shape[0]))
    random.shuffle(idx)
    Xtrain = Xtrain[idx]
    ytrain = ytrain[idx]
    
    mlglm = MultiStageGLM()
    mlglm.fit(Xtrain, ytrain)
    ypred_list = []
    for _ in range(100):
        ypred = mlglm.predict(Xtest)
        ypred_list.append(ypred.reshape((-1, 1)))
    ypred = np.concatenate(ypred_list, axis=1)

    p50 = np.quantile(ypred, 0.5, axis=1)
    p25 = np.quantile(ypred, 0.25, axis=1)
    p90 = np.quantile(ypred, 0.90, axis=1)

    plt.plot(range(len(y)), y, label="true")
    plt.plot(range(len(ytrain), len(y)), p50, label="pred")
    plt.fill_between(x=range(len(ytrain), len(y)), \
            y1=p25, y2=p90, alpha=0.5, color="red")
    ymin, ymax = plt.ylim()
    plt.vlines(train_size, ymin, ymax, color="k", linestyles="dashed", linewidth=2)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()
