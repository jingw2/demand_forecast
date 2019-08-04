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
        n_layers, seq_len, hidden_size = h.size()
        mu = []
        sigma = []
        for s in range(seq_len):
            ht = h[-1, s, :]
            sigma_t = torch.log(1 + torch.exp(self.sigma_layer(ht))) + 1e-6
            sigma_t = sigma_t.unsqueeze(0)
            mu_t = self.mu_layer(ht).unsqueeze(0)
            mu.append(mu_t)
            sigma.append(sigma_t)
        mu = torch.cat(mu, dim=0)
        # avoid zero by adding small value
        sigma = torch.cat(sigma, dim=0)
        return mu, sigma

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
        n_layers, seq_len, hidden_size = h.size()
        mu = []
        alpha = []
        for s in range(seq_len):
            ht = h[-1, s, :]
            alpha_t = torch.log(1 + torch.exp(self.sigma_layer(ht))) + 1e-6
            alpha_t = alpha_t.view(1, -1)
            mu_t = torch.log(1 + torch.exp(self.mu_layer(ht)))
            mu_t = mu_t.view(1, -1)
            mu.append(mu_t)
            alpha.append(alpha_t)
        mu = torch.cat(mu, dim=0)
        # avoid zero by adding small value
        alpha = torch.cat(alpha, dim=0)
        return mu, alpha

def gaussian_loss(ytrue, mu, sigma):
    '''
    Gaussian Loss Functions -log P(y|X)
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log 
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))

    multiplication to sum 
        - log l_{G} (z|mu, sigma) = 0.5 * log (2 * pi * sigma^2) + 0.5 * (z - mu)^2 / sigma^2
    '''
    loss = torch.log(sigma) + \
            0.5 * (ytrue - mu) * (ytrue - mu) / (sigma * sigma) + 20

    return loss.mean()

def negative_binomial_loss(ytrue, mu, alpha):
    '''
    Negative Binomial Loss
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return -likelihood.mean()

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
        # self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, bias=True, batch_first=True)
    
    def forward(self, X, y):
        '''
        Args:
        X (array like): shape (seq, input_size)
        X (array like): shape (seq, 1)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        y = self.input_embed(y)
        x = torch.cat([X, y], dim=1)
        x = x.unsqueeze(1)
        out, (h, c) = self.encoder(x) # h size (batch_size, num_layers, hidden_size)
        mu, sigma = self.likelihood_layer(h)
        return mu, sigma
    
    def predict(self, X, y):
        '''
        Predict 
        Args:
        X (array like): shape (batch_size, seq_len, num_features)
        '''
        if type(X) == type(np.empty((1, 1))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
        seq_len, num_features = X.size()
        mu, sigma = self.forward(X, y)
        ypred = []
        for sl in range(seq_len):
            if self.likelihood == "g":
                dist = torch.distributions.Normal(loc=mu[sl, :].reshape((-1, 1)), \
                        scale=sigma[sl, :].reshape((-1, 1)))
                ypred.append(dist.sample())
            elif self.likelihood == "nb":
                alpha_t = sigma[sl, :].view((1, -1))
                mu_t = mu[sl, :].view((1, -1))
                r = 1. / alpha_t.detach()
                p = r / (r + mu_t.detach())
                dist = torch.distributions.NegativeBinomial(r, p)
                sample_data = dist.sample()
                ypred.append(sample_data)
        ypred = torch.cat(ypred, dim=0).view(1, seq_len)
        return ypred

def batch_generator(X, y, seq_len):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    seq_len (int): sequence/encoder/decoder length
    '''
    num_periods, _ = X.shape
    t = random.choice(range(1, num_periods-seq_len))
    Xtrain = X[t:t+seq_len]
    ytrain = y[t-1:t+seq_len-1]
    yf = y[t:t+seq_len]
    return Xtrain, ytrain, yf

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
    ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = args.seq_len
    for epoch in range(args.num_epoches):
        print("Epoch {} starts...".format(epoch))
        for step in tqdm(range(args.step_per_epoch)):
            Xtrain, ytrain, yf = batch_generator(Xtr, ytr, seq_len)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()  
            mu, sigma = model(Xtrain_tensor, ytrain_tensor)
            if args.likelihood == "g":
                loss = gaussian_loss(ytrain_tensor, mu, sigma)
            elif args.likelihood == "nb":
                loss = negative_binomial_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test 
    mape_list = []
    # select skus with most top K
    X_test = Xte[1:].reshape((-1, num_features))
    yf = yte[:-1].reshape((-1, 1))
    y_test = yte[1:].reshape((-1, 1))
    y_pred_list = []
    for t in tqdm(range(args.num_results_to_sample)):
        y_test = yscaler.transform(y_test)
        y_pred = model.predict(X_test, y_test)
        y_pred = y_pred.data.numpy().ravel()
        y_pred = yscaler.inverse_transform(y_pred)
        y_pred_list.append(y_pred)
    tot_res = pd.DataFrame(y_pred_list).T
    tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)

    mape = util.MAPE(yf, tot_res.mu)
    print("MAPE: {}".format(mape))
    mape_list.append(mape)

    tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
    tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
    tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
    tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
    if args.show_plot:
        plt.figure(1)
        plt.plot([i for i in range(len(tot_res))], tot_res.mu, 'r-', linewidth=2)
        # plt.fill_between(x=tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
        # plt.fill_between(x=tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
        plt.title('Prediction uncertainty')
        plt.plot(range(len(yf)), yf, "k-")
        plt.legend(["prediction", "true"])
        plt.show()
    return losses, mape_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("--num_periods", "-np", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--embedding_size", "-es", type=int, default=64)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=168)
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
