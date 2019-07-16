# demand_forecast

Implementation of DeepAR, MQ-RNN and Deep Factor Models. 

## Requirements
Please install [Pytorch](https://pytorch.org/) before run it, and 

```shell
pip install -r requirement.txt
```

## DeepAR

#### Arguments

| name          | AKA | explanation   | default  |
| ------------- |:---:|:-------------:| -----:|
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
| --num_results_to_sample | -nrs | number of results to sample in test | 100 |
| --show_plot | -sp | show figures | apply -sp to use |
| --run_test | -rt | run test examples | apply -rt to use |

## MQ-RNN
