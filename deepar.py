#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
DeepAR Model (Pytorch Implementation)
Paper Link: https://arxiv.org/abs/1704.04110
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
from sklearn.preprocessing import StandardScaler
import util
from datetime import date
import argparse

class GaussianLikelihood(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(GaussianLikelihood, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t

class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)
    
    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t

def gaussian_likelihood(ytrue, mu, sigma):
    '''
    Gaussian Loss Functions -log P(y|X)
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log 
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
            torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    return likelihood

def negative_binomial_likelihood(ytrue, mu, alpha):
    '''
    Negative Binomial Loss
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return torch.exp(likelihood)

class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size+input_size, hidden_size, \
                num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = GaussianLikelihood(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood
    
    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (seq_len, input_size)
        y (array like): shape (seq, 1)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        seq_len, _ = X.size()
        output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[s, :].view(-1, 1)
                yembed = self.input_embed(ynext).view(1, -1)
                x = X[s, :].view(1, -1)
            else:
                yembed = self.input_embed(ynext).view(1, -1)
                x = Xf[s-seq_len, :].view(1, -1)
            x = torch.cat([x, yembed], dim=1)
            inp = x.unsqueeze(0)
            out, (hs, c) = self.encoder(inp) # h size (num_layers, 1, hidden_size)
            hs = hs[-1, :, :]
            mu, sigma = self.likelihood_layer(hs)
            if self.likelihood == "g":
                ynext = gaussian_likelihood(ynext, mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_likelihood(ynext, mu_t, alpha_t)
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(-1, 1)
        return ypred
    
def batch_generator(X, y, num_obs_to_train, seq_len):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    '''
    num_periods, _ = X.shape
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    X_train_batch = X[t-num_obs_to_train:t]
    y_train_batch = y[t-num_obs_to_train:t]
    Xf = X[t:t+seq_len]
    yf = y[t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

def train(
    X, 
    y,
    args,
    rho=0.5
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
    num_periods, num_features = X.shape
    model = DeepAR(num_features, args.embedding_size, 
        args.hidden_size, args.n_layers, args.lr, args.likelihood)
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
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    for epoch in range(args.num_epoches):
        print("Epoch {} starts...".format(epoch))
        for step in tqdm(range(args.step_per_epoch)):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()  
            yf = torch.from_numpy(yf).float()
            ypred = model(Xtrain_tensor, ytrain_tensor, Xf)
            ypred_rho = ypred.view(-1, 1)
            e = ypred_rho - yf
            loss = torch.max(rho * e, (rho - 1) * e).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test 
    mape_list = []
    # select skus with most top K
    X_test = Xte[-seq_len-num_obs_to_train:-seq_len].reshape((-1, num_features))
    Xf_test = Xte[-seq_len:].reshape((-1, num_features))
    y_test = yte[-seq_len-num_obs_to_train:-seq_len].reshape((-1, 1))
    yf_test = yte[-seq_len:]
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    y_pred = model(X_test, y_test, Xf_test)
    y_pred = y_pred.data.numpy().ravel()
    if yscaler is not None:
        y_pred = yscaler.inverse_transform(y_pred)
    
    mape = util.MAPE(yf_test, y_pred)
    print("MAPE: {}".format(mape))
    mape_list.append(mape)

    if args.show_plot:
        plt.figure(1)
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], y_pred, "r-")
        plt.title('Prediction uncertainty')
        yplot = yte[-seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
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
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--embedding_size", "-es", type=int, default=64)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")

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
        X = np.asarray(X).reshape((num_periods, num_features))
        y = np.asarray(data["MT_200"]).reshape((num_periods, 1))
        losses, mape_list = train(X, y, args)
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
