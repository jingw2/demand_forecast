#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Pytorch Implementation of Deep Factors For Forecasting
Paper Link: https://arxiv.org/pdf/1905.12417.pdf
Author: Jing Wang (jingw2@foxmail.com)
'''

import torch
from torch import nn
import torch.nn.functional as F 
from torch.optim import Adam

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
import util
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
import argparse
from datetime import date
from progressbar import *

class DeepFactor(nn.Module):

    def __init__(self, input_size, global_nlayers, global_hidden_size, n_global_factors):
        super(DeepFactor, self).__init__()
        self.lstm = nn.LSTM(input_size, global_hidden_size, global_nlayers, \
                    bias=True, batch_first=True)
        self.factor = nn.Linear(global_hidden_size, n_global_factors)

    def forward(self, X):
        num_ts, num_features = X.shape
        X = X.unsqueeze(1)
        _, (h, c) = self.lstm(X)
        ht = h[-1, :, :] # num_ts, global factors
        ht = F.relu(ht)
        gt = ht
        return gt.view(num_ts, -1)

class Noise(nn.Module):

    def __init__(self, input_size, noise_nlayers, noise_hidden_size):
        super(Noise, self).__init__()
        self.lstm = nn.LSTM(input_size, noise_hidden_size, 
                noise_nlayers, bias=True, batch_first=True)
        self.affine = nn.Linear(noise_hidden_size, 1)

    def forward(self, X):
        num_ts, num_features = X.shape
        X = X.unsqueeze(1)
        _, (h, c) = self.lstm(X)
        ht = h[-1, :, :] # num_ts, global factors
        ht = F.relu(ht)
        sigma_t = self.affine(ht)
        sigma_t = torch.log(1 + torch.exp(sigma_t))
        return sigma_t.view(-1, 1)

class DFRNN(nn.Module):

    def __init__(self, input_size, noise_nlayers, noise_hidden_size, 
            global_nlayers, global_hidden_size, n_global_factors):
        super(DFRNN, self).__init__()
        self.noise = Noise(input_size, noise_hidden_size, noise_nlayers)
        self.global_factor = DeepFactor(input_size, global_nlayers, 
                    global_hidden_size, n_global_factors)
        self.embed = nn.Linear(global_hidden_size, n_global_factors)
    
    def forward(self, X,):
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
        num_ts, num_periods, num_features = X.size()
        mu = []
        sigma = []
        for t in range(num_periods):
            gt = self.global_factor(X[:, t, :])
            ft = self.embed(gt)
            ft = ft.sum(dim=1).view(-1, 1)
            sigma_t = self.noise(X[:, t, :])
            mu.append(ft)
            sigma.append(sigma_t)
        mu = torch.cat(mu, dim=1).view(num_ts, num_periods)
        sigma = torch.cat(sigma, dim=1).view(num_ts, num_periods) + 1e-6
        return mu, sigma
    
    def sample(self, X, num_samples=100):
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
        mu, var = self.forward(X)
        num_ts, num_periods = mu.size()
        z = torch.zeros(num_ts, num_periods)
        for _ in range(num_samples):
            dist = torch.distributions.normal.Normal(loc=mu, scale=var)
            zs = dist.sample().view(num_ts, num_periods)
            z += zs
        z = z / num_samples
        return z

def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

def train(
    X, 
    y,
    args
    ):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    # rho = args.quantile
    num_ts, num_periods, num_features = X.shape
    model = DFRNN(num_features, args.noise_nlayers, 
        args.noise_hidden_size, args.global_nlayers, 
        args.global_hidden_size, args.n_factors)
    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities 
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    cnt = 0

    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    progress = ProgressBar()
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    for epoch in progress(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(args.step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, 
                        seq_len, args.batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()  
            yf = torch.from_numpy(yf).float()
            mu, sigma = model(Xtrain_tensor)
            loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test 
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    
    result = []
    n_samples = args.sample_size
    for _ in tqdm(range(n_samples)):
        y_pred = model.sample(Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))
    
    result = np.concatenate(result, axis=1)
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)

    mape = util.MAPE(yf_test, p50)
    print("P50 MAPE: {}".format(mape))
    mape_list.append(mape)

    if args.show_plot:
        plt.figure(1, figsize=(20, 5))
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], p50, "r-")
        plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)], \
            y1=p10, y2=p90, alpha=0.5)
        plt.title('Prediction uncertainty')
        yplot = yte[-1, -seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses, mape_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--global_hidden_size", "-ghs", type=int, default=50)
    parser.add_argument("--global_nlayers", "-gn", type=int, default=1)
    parser.add_argument("--noise_hidden_size", "-nhs", type=int, default=5)
    parser.add_argument("--noise_nlayers", "-nn", type=int, default=1)
    parser.add_argument("--n_factors", "-f", type=int, default=10)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=1)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=20)

    args = parser.parse_args()

    if args.run_test:
        data_path = util.get_data_path()
        data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"])
        data["year"] = data["date"].apply(lambda x: x.year)
        data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
        data = data.loc[(data["date"] >= date(2014, 1, 1)) & (data["date"] <= date(2014, 3, 1))]

        features = ["hour", "day_of_week"]
        # hours = pd.get_dummies(data["hour"])
        # dows = pd.get_dummies(data["day_of_week"])
        hours = data["hour"]
        dows = data["day_of_week"]
        X = np.c_[np.asarray(hours), np.asarray(dows)]
        num_features = X.shape[1]
        num_periods = len(data)
        X = np.asarray(X).reshape((-1, num_periods, num_features))
        y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        losses, mape_list = train(X, y, args)
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
