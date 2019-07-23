# Demand Forecast Models

Pytorch Implementation of DeepAR, MQ-RNN, Deep Factor Models and TPA-LSTM. Others, see LSTNet[https://github.com/laiguokun/LSTNet]

## Requirements
Please install [Pytorch](https://pytorch.org/) before run it, and 

```shell
pip install -r requirement.txt
```

## DeepAR
It is used to predict in rolling ways. DeepAR has to apply ground truth of previous historical days to predict the next few days. For example, if you would like to predict demands in 2019-07-17, you has to use ground truth in 2019-07-16.

#### Arguments

| name          | AKA | explanation   | default  |
| ------------- |---|-------------| ----- |
| --num_epoches  | -e | number of epoches to run | 1000 |
| --step_per_epoch | -spe | step per epoch to run | 2 |
| --num_periods | -np | number of periods in data used | 100|
| -lr |  | learning rate | 1e-3 |
| --n_layers | -nl | number of layers in networks | 3 |
| --batch_size | -b | batch size | 64 |
| --hidden_size | -hs | hidden size | 64 |
| --likelihood | -l | likelihood to use | "g" means gaussian, <br>  "nb" means negative binomial |
| --seq_len | -sl | sequence length to feed | 7 |
| --num_skus_to_show | -nss | number of skus to show in test | 1 |
| --num_results_to_sample | -nrs | number of results to sample in test | 1 |
| --show_plot | -sp | show figures | apply -sp to use |
| --run_test | -rt | run test examples | apply -rt to use |

#### Examples
Run in terminal
```shell
python deepar.py -e 100 -np 100 -nl 2 -l g -nrs 1 -sp -rt
```

## MQ-RNN
Sequence to Sequence Structure. Encoder is LSTM layer, Decoder has global and local Multi-layer Perceptron. It can be used to predict demands in the next few days based on data in previous days. 

#### Arguments
| name          | AKA | explanation   | default  |
| ------------- |---|-------------| ----- |
| --num_epoches  | -e | number of epoches to run | 1000 |
| --step_per_epoch | -spe | step per epoch to run | 2 |
| --num_periods | -np | number of periods in data used | 100|
| -lr |  | learning rate | 1e-3 |
| --n_layers | -nl | number of layers in networks | 3 |
| --batch_size | -b | batch size | 64 |
| --hidden_size | -hs | hidden size | 64 |
| --seq_len | -sl | sequence length to feed | 7 |
| --num_skus_to_show | -nss | number of skus to show in test | 1 |
| --show_plot | -sp | show figures | apply -sp to use |
| --run_test | -rt | run test examples | apply -rt to use |

#### Examples
Run in terminal
```shell
python mq_rnn.py -e 100 -np 100 -nl 2 -sp -rt
```

#### TO DO
* [ ] Deep Factor Model
* [ ] TPA-LSTM pytorch 

# Reference
* [DeepAR](https://arxiv.org/abs/1704.04110)
* [MQ-RNN](https://arxiv.org/abs/1711.11053)
