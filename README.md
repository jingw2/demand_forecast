# Demand Forecast Models

Implementation of DeepAR, MQ-RNN and Deep Factor Models. 

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
