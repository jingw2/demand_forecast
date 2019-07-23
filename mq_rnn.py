#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Pytorch Implementation of MQ-RNN
Paper Link: https://arxiv.org/abs/1711.11053
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

class Decoder(nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.global_mlp = nn.Linear(output_horizon * hidden_size, (output_horizon+1) * hidden_size)
        self.local_mlp = nn.Linear(hidden_size, output_size)
    
    def forward(self, ht, xf):
        '''
        Args:
        ht (tensor): (batch_size, 1, hidden_size)
        xf (tensor): (batch_size, output_horizon, hidden_size)
        '''
        batch_size, output_horizon, num_features = xf.size()
        _, _, hidden_size = ht.size()
        xf = self.embed(xf) # batch_size, output_horizon, hidden_size
        ht = ht.expand(batch_size, output_horizon, -1)
        inp = (xf + ht).view(batch_size, -1) # batch_size, hidden_size, output_horizon
        contexts = self.global_mlp(inp)
        contexts = contexts.view(batch_size, -1, hidden_size)
        ca = contexts[:, -1, :]
        C = contexts[:, :-1, :]
        C = F.relu(C)
        y = []
        for i in range(output_horizon):
            ci = C[:, i, :].view((batch_size, -1))
            xfi = xf[:, i, :].view((batch_size, -1))
            inp = xfi + ci + ca
            out = self.local_mlp(inp)
            y.append(out.unsqueeze(1))
        y = torch.cat(y, dim=1) # batch_size, output_horizon, quantiles
        return y 


class MQRNN(nn.Module):

    def __init__(self, output_horizon, num_quantiles, input_size, hidden_size=64, n_layers=3):
        '''
        Args:
        output_horizon (int): output horizons to output in prediction
        num_quantiles (int): number of quantiles interests, e.g. 0.25, 0.5, 0.75
        input_size (int): feature size
        embedding_size (int): embedding size
        hidden_size (int): hidden size in layers
        n_layers (int): number of layers used in model
        '''
        super(MQRNN, self).__init__()
        self.output_horizon = output_horizon
        self.hidden_size = hidden_size
        self.input_embed = nn.Linear(input_size+1, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, n_layers, bias=True, batch_first=True)
        self.decoder = Decoder(input_size, output_horizon, hidden_size, num_quantiles)
    
    def forward(self, X, y, Xf):
        '''
        Args:
        X (tensor like): shape (batch_size, num_features, num_periods)
        y (tensor like): shape (batch_size, num_periods)
        Xf (tensor like): shape (batch_size, num_features, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        X = X.permute(0, 2, 1)
        Xf = Xf.permute(0, 2, 1) # batch_size, output_horizon, num_features
        y = y.permute(0, 2, 1) # batch_size, num_periods, 1
        batch_size, num_periods, num_features = X.shape
        x = torch.cat([X, y], dim=2)
        # encoder
        x = self.input_embed(x)
        _, (h, c) = self.encoder(x)
        h = h.permute(1, 0, 2) # batch_size, num_layers, hidden_size
        # global mlp
        ht = h[:, -1, :].unsqueeze(1) # shape (batch_size, 1, hidden_size)
        ht = F.relu(ht)
        ypred = self.decoder(ht, Xf)
        return ypred

def batch_generator(X, y, batch_size, seq_len):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    epoches (int)
    batch_size (int)
    seq_len (int): sequence/encoder/decoder length
    '''
    num_samples, _, num_periods = X.shape
    t = random.choice(range(seq_len, num_periods-seq_len))
    random_indice = random.sample(range(num_samples), batch_size)
    X_train_batch = X[random_indice, :, :t]
    y_train_batch = y[random_indice, :t]
    y_train_batch = np.expand_dims(y_train_batch, axis=1)
    Xf = X[random_indice, :, t:t+seq_len]
    yf = y[random_indice, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

def train(
    X, 
    y,
    epoches,
    step_per_epoch,
    batch_size=64,
    hidden_size=128,
    n_layers=3,
    lr=1e-3,
    seq_len=7,
    num_skus_to_show=10,
    show_plot=False,
    train_ratio=0.7,
    quantiles=[0.1, 0.5, 0.9]
    ):
    num_samples, num_features, num_periods = X.shape
    mean_y = np.mean(y, axis=1).reshape((-1, 1))
    y_norm = y 
    num_quantiles = len(quantiles)
    model = MQRNN(seq_len, num_quantiles, num_features, hidden_size, n_layers)
    optimizer = Adam(model.parameters(), lr=lr)
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    # random_idx = random.sample(range(num_samples), 50)
    Xtr = X[:, :, :train_periods]
    ytr = y_norm[:, :train_periods]
    Xte = X[:, :, train_periods:]
    yte = y_norm[:, train_periods:]
    losses = []
    for epoch in range(epoches):
        print("Epoch {} start...".format(epoch))
        for step in tqdm(range(step_per_epoch)):
            X_train_batch, y_train_batch, Xf, yf = batch_generator(Xtr, ytr, batch_size, seq_len)
            X_train_tensor = torch.from_numpy(X_train_batch).float()
            y_train_tensor = torch.from_numpy(y_train_batch).float() 
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            ypred = model(X_train_tensor, y_train_tensor, Xf)
            loss = util.quantile_loss2(yf, ypred, quantiles)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    mape_list = []
    for i in range(num_skus_to_show):
        sample_idx = np.argsort(np.sum(yte, axis=1))[-i]
        X_test = Xte[sample_idx, :, :-seq_len].reshape((1, num_features, -1))
        Xf_test = Xte[sample_idx, :, -seq_len:].reshape((1, num_features, -1))
        y_test = np.expand_dims(yte[sample_idx, :-seq_len].reshape((1, -1)), axis=1)
        yf_test = yte[sample_idx, -seq_len:]
        y_pred = model(X_test, y_test, Xf_test) # (1, num_quantiles, output_horizon)
        y_pred = y_pred.data.numpy()
        # y_pred = np.maximum(0, y_pred)
        mape = util.MAPE(yf_test, y_pred[0, :, 1])
        print("MAPE: {}%".format(mape))
        mape_list.append(mape)

        if show_plot:
            plt.figure(i)
            plt.plot([i + num_periods - seq_len for i in range(seq_len)], y_pred[0, :, 1], "r-")
            plt.fill_between(x=[i + num_periods - seq_len for i in range(seq_len)], \
                y1=y_pred[0, :, 0], y2=y_pred[0, :, 2], alpha=0.5)
            plt.title('Prediction uncertainty')
            plt.plot(range(len(y[sample_idx])), y[sample_idx])
            plt.legend(["0.5", "true"])
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
                args.hidden_size, args.n_layers, args.lr, args.seq_len, \
                args.num_skus_to_show, args.show_plot)
        print("Average MAPE in test skus: ", np.mean(mape_list))
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.show()
