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
import argparse

class GaussianLikelihood(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(GaussianLikelihood, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        mu = []
        sigma = []
        for s in range(seq_len):
            xt = x[:, s, :].view(batch_size, hidden_size)
            sigma_t = torch.log(1 + torch.exp(self.sigma_layer(xt))) + 1e-6
            sigma_t = sigma_t.unsqueeze(1)
            mu_t = self.mu_layer(xt).unsqueeze(1)
            mu.append(mu_t)
            sigma.append(sigma_t)
        mu = torch.cat(mu, dim=1)
        # avoid zero by adding small value
        sigma = torch.cat(sigma, dim=1)
        return mu.squeeze(2), sigma.squeeze(2)

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
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        mu = []
        alpha = []
        for s in range(seq_len):
            xt = x[:, s, :].view(batch_size, hidden_size)
            alpha_t = torch.log(1 + torch.exp(self.sigma_layer(xt))) + 1e-6
            alpha_t = alpha_t.unsqueeze(1)
            mu_t = torch.log(1 + torch.exp(self.mu_layer(xt)))
            mu_t = mu_t.unsqueeze(1)
            mu.append(mu_t)
            alpha.append(alpha_t)
        mu = torch.cat(mu, dim=1)
        # avoid zero by adding small value
        alpha = torch.cat(alpha, dim=1)
        return mu.squeeze(2), alpha.squeeze(2)

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
    likelihood = 0.5 * torch.log(2 * np.pi * sigma ** 2) + 0.5 * (ytrue - mu) ** 2 / sigma ** 2
    return likelihood.mean() 

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
        self.input_embed = nn.Linear(input_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = GaussianLikelihood(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood
        # self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, bias=True, batch_first=True)
    
    def forward(self, X):
        '''
        Args:
        X (array like): shape (batch_size, seq, input_size)

        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        x = self.input_embed(X)
        out, (h, c) = self.encoder(x) # h size (batch_size, num_layers, hidden_size)
        mu, sigma = self.likelihood_layer(out)
        return mu, sigma
    
    def predict(self, X):
        '''
        Predict 
        Args:
        X (array like): shape (batch_size, seq_len, num_features)
        '''
        if type(X) == type(np.empty((1, 1))):
            X = torch.from_numpy(X).float()
        batch_size, seq_len, num_features = X.size()
        mu, sigma = self.forward(X)
        ypred = []
        for sl in range(seq_len):
            if self.likelihood == "g":
                dist = torch.distributions.Normal(loc=mu[:, sl].reshape((-1, 1)), \
                        scale=sigma[:, sl].reshape((-1, 1)))
                ypred.append(dist.sample())
            elif self.likelihood == "nb":
                alpha_t = sigma[:, sl].view((batch_size, -1))
                mu_t = mu[:, sl].view((batch_size, -1))
                r = 1. / alpha_t.detach()
                p = r / (r + mu_t.detach())
                dist = torch.distributions.NegativeBinomial(r, p)
                sample_data = dist.sample()
                ypred.append(sample_data)
        ypred = torch.cat(ypred, dim=1).view(batch_size, seq_len)
        return ypred

def batch_generator(X, y, batch_size, seq_len):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    batch_size (int)
    seq_len (int): sequence/encoder/decoder length
    '''
    num_samples, _, num_periods = X.shape
    random_indice = random.sample(range(num_samples), batch_size)
    t = random.choice(range(1, num_periods-seq_len))
    X_train_batch = X[random_indice, :, t:t+seq_len]
    y_train_batch = y[random_indice, t-1:t+seq_len-1]
    y_train_batch = np.expand_dims(y_train_batch, axis=1)
    X_train = np.concatenate((X_train_batch, y_train_batch), axis=1)
    y_train = y[random_indice, t:t+seq_len]
    return X_train, y_train

def train(
    X, 
    y,
    epoches,
    step_per_epoch,
    batch_size=64,
    hidden_size=64,
    num_layers=2,
    embedding_size=64,
    lr=1e-3,
    seq_len=8,
    likelihood="g",
    num_skus_to_show=2,
    num_results_to_sample=100,
    show_plot=False,
    train_ratio=0.7
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
    num_samples, num_features, num_periods = X.shape
    model = DeepAR(num_features+1, embedding_size, hidden_size, num_layers, lr, likelihood)
    optimizer = Adam(model.parameters(), lr=lr)
    random.seed(2)
    # select sku with most top n quantities 
    train_periods = int(num_periods * train_ratio)
    Xtr = X[:, :, :train_periods]
    ytr = y[:, :train_periods]
    Xte = X[:, :, train_periods:]
    yte = y[:, train_periods:]
    losses = []
    cnt = 0

    # training
    for epoch in range(epoches):
        print("Epoch {} starts...".format(epoch))
        for step in tqdm(range(step_per_epoch)):
            X_train, y_train = batch_generator(Xtr, ytr, batch_size, seq_len)
            X_train = X_train.transpose(0, 2, 1)
            X_train_tensor = torch.from_numpy(X_train).float()
            y_train_tensor = torch.from_numpy(y_train).float()  
            mu, sigma = model(X_train_tensor)
            if likelihood == "g":
                loss = gaussian_loss(y_train_tensor, mu, sigma)
            elif likelihood == "nb":
                loss = negative_binomial_loss(y_train_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test 
    mape_list = []
    for i in range(num_skus_to_show):
        # select skus with most top K
        sample_idx = np.argsort(np.sum(yte, axis=1))[-i]
        test_periods = num_periods - train_periods
        X_test = Xte[sample_idx, :, 1:].reshape((-1, num_features, test_periods-1))
        y_test = np.expand_dims(yte[sample_idx, :-1].reshape((-1, test_periods-1)), axis=1)
        X_test = np.concatenate((X_test, y_test), axis=1)
        y_pred_list = []
        for t in tqdm(range(num_results_to_sample)):
            y_pred = model.predict(X_test.transpose(0, 2, 1))
            y_pred = y_pred.data.numpy().ravel().tolist()
            y_pred_list.append(y_pred)
        tot_res = pd.DataFrame(y_pred_list).T
        tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)

        smape = util.SMAPE(y[sample_idx], tot_res.mu)
        print("SMAPE: {}%".format(smape))
        mape_list.append(smape)

        tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
        tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
        tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
        tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)
        if show_plot:
            plt.figure(i)
            plt.plot([i + train_periods + 1 for i in range(len(tot_res))], tot_res.mu, 'bo-', linewidth=2)
            plt.fill_between(x=tot_res.index+train_periods+1, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
            plt.fill_between(x=tot_res.index+train_periods+1, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
            plt.title('Prediction uncertainty')
            plt.plot(range(len(y[sample_idx])), y[sample_idx], "r-")
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
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_skus_to_show", "-nss", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=100)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", action="store_true")

    args = parser.parse_args()

    if args.run_test:
        num_periods = args.num_periods
        num_features = 1
        num_samples = 200
        X = np.zeros((num_samples, num_features, num_periods))
        y = np.zeros((num_samples, num_periods))
        for ns in range(num_samples):
            d = [t * np.sin(t/6) / 3 + np.sin(t*2) for t in range(num_periods)]
            X[ns, :, :] = d
            y[ns] = d
        losses, mape_list = train(X, y, args.num_epoches, args.step_per_epoch, args.batch_size, \
                args.hidden_size, args.n_layers, args.hidden_size, args.lr, args.seq_len, \
                args.likelihood, args.num_skus_to_show, args.num_results_to_sample, args.show_plot)
        print("Average MAPE in test skus: ", np.mean(mape_list))
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.show()
    # # scaler = StandardScaler()
    # X = pickle.load(open("beijing_x.pkl", "rb"))
    # y = pickle.load(open("beijing_y.pkl", "rb"))
    # # normalize scale
    # # y = scaler.transform(y)
    # # scaler.fit(y)
    # # scale handling from paper
    # # y = y / np.mean(y, axis=1).reshape((-1, 1))  
    # num_samples, num_features, num_periods = X.shape
    # sample_idx = 10
    # num_periods = 180
    # # X = np.repeat(X[:, :, :num_periods].reshape((-1, num_features, num_periods)), 100, axis=0)
    # # y = np.repeat(y[:, :num_periods].reshape((-1, num_periods)), 100, axis=0)
    # X = X[:, :, :num_periods].reshape((-1, num_features, num_periods))
    # y = y[:, :num_periods].reshape((-1, num_periods))
    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)
    # epoches = 1000
    # step_per_epoch = 5
    # losses, mape_list = train(X, y, epoches, step_per_epoch, likelihood="g", seq_len=7)
    # print("Average MAPE in test skus: ", np.mean(mape_list))
    # if show_plot:
    #     plt.plot(range(len(losses)), losses, "k-")
    #     plt.show()


