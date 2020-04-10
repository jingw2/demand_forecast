#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Pytorch Implementation of Extreme Event Forecasting at Uber
http://roseyu.com/time-series-workshop/submissions/TSW2017_paper_3.pdf
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

class AutoEncoder(nn.Module):

    def __init__(self, input_size, encoder_hidden_units):
        super(AutoEncoder, self).__init__()
        self.layers = []
        self.dropout = nn.Dropout()
        last_ehu = None
        for idx, ehu in enumerate(encoder_hidden_units):
            if idx == 0:
                layer = nn.LSTM(input_size, ehu, 1, bias=True, batch_first=True)
            else:
                layer = nn.LSTM(last_ehu, ehu, 1, bias=True, batch_first=True)
            last_ehu = ehu
            self.layers.append(layer)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        for layer in self.layers:
            hs = []
            for s in range(seq_len):
                _, (h, c) = layer(x)
                h = h.permute(1, 0, 2)
                h = F.relu(h)
                h = self.dropout(h)
                hs.append(h)
            x = torch.cat(hs, dim=1)
        return x
    
class Forecaster(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers):
        super(Forecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mu):
        '''
        Args:
        x (tensor): 
        mu (tensor): model uncertainty
        '''
        batch_size, seq_len, hidden_size = x.size()
        out = []
        for s in range(seq_len):
            xt = x[:, s, :].unsqueeze(1)
            xt = torch.cat([xt, mu], dim=1)
            _, (h, c) = self.lstm(xt)
            ht = h[-1, :, :].unsqueeze(0)
            h = ht.permute(1, 0, 2)
            h = F.relu(h)
            h = self.dropout(h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        return out

class ExtremeModel(nn.Module):

    def __init__(
        self, 
        input_size, 
        encoder_hidden_units=[512, 128, 64], 
        hidden_size_forecaster=512,
        n_layers_forecaster=3
    ):
        super(ExtremeModel, self).__init__()
        self.embed = nn.Linear(input_size, encoder_hidden_units[-1])
        self.auto_encoder = AutoEncoder(encoder_hidden_units[-1], encoder_hidden_units)
        self.forecaster = Forecaster(encoder_hidden_units[-1],
                hidden_size_forecaster, n_layers_forecaster)
    
    def forward(self, xpast, xnew):
        if isinstance(xpast, type(np.empty(1))):
            xpast = torch.from_numpy(xpast).float()
        if isinstance(xnew, type(np.empty(1))):
            xnew = torch.from_numpy(xnew).float()
        xpast = self.embed(xpast)
        xnew = self.embed(xnew)
        # auto-encoder
        ae_out = self.auto_encoder(xpast)
        ae_out = torch.mean(ae_out, dim=1).unsqueeze(1)
        # concatenate x
        # x = torch.cat([xnew, ae_out], dim=1)
        x = self.forecaster(xnew, ae_out)
        return x

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

    num_ts, num_periods, num_features = X.shape
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    progress = ProgressBar()
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train

    model = ExtremeModel(num_features)
    optimizer = Adam(model.parameters(), lr=args.lr)
    losses = []
    cnt = 0
    for epoch in progress(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(args.step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, 
                        seq_len, args.batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()  
            yf = torch.from_numpy(yf).float()
            ypred = model(Xtrain_tensor, Xf)
            loss = F.mse_loss(ypred, yf)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    ypred = model(X_test, Xf_test)
    ypred = ypred.data.numpy()
    if yscaler is not None:
        ypred = yscaler.inverse_transform(ypred)
    
    mape = util.MAPE(yf_test, ypred)
    print("MAPE: {}".format(mape))
    mape_list.append(mape)

    if args.show_plot:
        plt.figure(1)
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], ypred[-1], "r-")
        plt.title('Prediction uncertainty')
        yplot = yte[-1, -seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["forecast", "true"], loc="upper left")
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
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=1)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)

    args = parser.parse_args()
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

    losses, mape_list = train(X, y, args)
    if args.show_plot:
        plt.plot(range(len(losses)), losses, "k-")
        plt.xlabel("Period")
        plt.ylabel("Loss")
        plt.show()
