# Deep Demand Forecast Models

Pytorch Implementation of DeepAR, MQ-RNN, Deep Factor Models, LSTNet, and TPA-LSTM. Furthermore, combine all these model to deep demand forecast model API.

## Requirements
Please install [Pytorch](https://pytorch.org/) before run it, and 

```python
pip install -r requirements.txt
```

## Run tests
```python
# DeepAR
python deepar.py -e 100 -spe 3 -nl 1 -l g -not 168 -sp -rt -es 10 -hs 50  -sl 60 -ms

# MQ-RNN
python mq_rnn.py -e 100 -spe 3 -nl 1 -sp -sl 72 -not 168 -rt -ehs 50 -dhs 20 -ss -es 10 -ms

# Deep Factors
python deep_factors.py -e 100 -spe 3 -rt -not 168 -sp -sl 168 -ms

# TPA-LSTM
python tpa_lstm.py -e 1000 -spe 1 -nl 1 -not 168 -sl 30 -sp -rt -max
```
DeepAR \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/deepar.png) \
MQ-RNN \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/mq_rnn.png) \
Deep Factors \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/deep_factors.png) \
TPA-LSTM \
![alt text](https://github.com/jingw2/demand_forecast/blob/master/pic/tpa_lstm.png)

## Arguments
|  Arguments   | Details  |
|  ----  | ----  |
| -e  | number of episodes |
| -spe  | steps per episode |
| -sl | sequence length |
| -not | number of observations to train|
| -ms | mean scaler on y|
| -max | max scaler on y|
| -nl | number of layers|
| -l | likelihood to select, "g" or "nb"|
| -rt | run test data |
| -sample_size | sample size to sample after </br> training in deep factors/deepar, default 100|

#### TO DO
* [X] Deep Factor Model
* [X] TPA-LSTM pytorch 
* [ ] LSTNet pytorch
* [ ] Debug Uber Extreme forcaster
* [ ] Modeling Extreme Events in TS
* [X] Intermittent Demand Forecasting
* [ ] Model API

# Demand Forecast Dataset Resources
* [Solar Energy](https://www.nrel.gov/grid/solar-power-data.html)
* [Traffic](http://pems.dot.ca.gov/)
* [Electricity](https://arxiv.org/pdf/1809.04206v2.pdf)
* [MuseDATA](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)

# Reference
* [DeepAR](https://arxiv.org/abs/1704.04110)
* [MQ-RNN](https://arxiv.org/abs/1711.11053)
* [Deep Factor](https://arxiv.org/pdf/1905.12417.pdf)
* [LSTNet](https://arxiv.org/abs/1703.07015)
* [TPA-LSTM](https://arxiv.org/pdf/1809.04206v2.pdf)
* [LSTNet Github](https://github.com/laiguokun/LSTNet)
* [TPA-LSTM Github](https://github.com/gantheory/TPA-LSTM)
* [Uber Extreme Event Forecast 1](http://roseyu.com/time-series-workshop/submissions/TSW2017_paper_3.pdf)
* [Uber Extreme Event Forecast 2](https://forecasters.org/wp-content/uploads/gravity_forms/7-c6dd08fee7f0065037affb5b74fec20a/2017/07/Laptev_Nikolay_ISF2017.pdf)
* [Modeling Extreme Events in Time Series Prediction](http://staff.ustc.edu.cn/~hexn/papers/kdd19-timeseries.pdf)
