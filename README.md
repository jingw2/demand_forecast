# Deep Demand Forecast Models

Pytorch Implementation of DeepAR, MQ-RNN, Deep Factor Models, LSTNet, and TPA-LSTM. Furthermore, combine all these model to deep demand forecast model API.

## Requirements
Please install [Pytorch](https://pytorch.org/) before run it, and 

```python
pip install -r requirement.txt
```

## Run tests
```python
# DeepAR
pythonw deepar.py -e 100 -spe 3 -nl 1 -l g -not 168 -sp -rt -es 10 -hs 50  -sl 60 -ms

# MQ-RNN
pythonw mq_rnn.py -e 100 -spe 3 -nl 1 -sp -sl 72 -not 168 -rt -ehs 50 -dhs 20 -ss -es 10 -ms

# Deep Factors
pythonw deep_factors.py -e 100 -spe 3 -rt -not 168 -sp -sl 168 -ms
```
DeepAR \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/deepar.png) \
MQ-RNN \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/mq_rnn.png) \
Deep Factors \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/deep_factors.png)

## Arguments
|  Arguments   | Details  |
|  ----  | ----  |
| -e  | number of episodes |
| -spe  | steps per episode |
| -sl | sequence length |
| -not | number of observations to train|
| -ms | mean scaler on y|
| -nl | number of layers|
| -l | likelihood to select, "g" or "nb"|
| -sample_size | sample size to sample after </br> training in deep factors/deepar, default 100|

#### TO DO
* [X] Deep Factor Model
* [ ] TPA-LSTM pytorch 
* [ ] LSTNet pytorch
* [ ] Model API

# Reference
* [DeepAR](https://arxiv.org/abs/1704.04110)
* [MQ-RNN](https://arxiv.org/abs/1711.11053)
* [Deep Factor](https://arxiv.org/pdf/1905.12417.pdf)
* [LSTNet](https://arxiv.org/abs/1703.07015)
* [TPA-LSTM](https://arxiv.org/pdf/1809.04206v2.pdf)
* [LSTNet Github](https://github.com/laiguokun/LSTNet)
* [TPA-LSTM Github](https://github.com/gantheory/TPA-LSTM)
