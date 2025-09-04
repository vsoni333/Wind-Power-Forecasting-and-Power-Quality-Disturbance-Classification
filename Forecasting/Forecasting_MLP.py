# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 07:47:40 2025

@author: My Account
"""

from pandas import read_csv

dataset = read_csv('Task15_W_Zone6.csv')
values = dataset.values
#2.tranform data to [0,1]  3个属性，第4个是待预测量

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
XY= scaler.fit_transform(values)


XY= values
Y = XY[:,4]
n_train_hours1 =15000
x_train=XY[:n_train_hours1,0:4]
y_train =Y[:n_train_hours1]
x_test =XY[n_train_hours1:, 0:4]
y_test =Y[n_train_hours1:]
#LSTM的输入格式要3维，因此先做变换

print("x_train.shape", x_train.shape)  # 输出训练集的形状Output the shape of the training set. #(60000, 28, 28, 1)
print(x_train.shape[0], "train samples")  # 输出训练样本数量 output training sample quantity #60000 train samples
print(x_test.shape[0], "test samples")  # 输出测试样本数量 output test sample quantity #10000 test samples

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout

print()
print("Build the model")  # 构建模型 build a model
# 构建模型 build a model
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(4,)))
model.add(Dropout(0.25))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()  # 输出模型摘要 Generate model summary


model.compile(loss='mae', optimizer='adam')
# 6. fit the network
history = model.fit(x_train,y_train, epochs=50, batch_size=128,validation_data=(x_test, y_test), verbose=1, shuffle=False)

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#8. make a prediction and invert scaling for forecast
from pandas import concat
import numpy as np
forecasttestY0 = model.predict(x_test)
#forecasttestY= np.expand_dims(a, axis=1)

inv_yhat =np.concatenate((x_test,forecasttestY0), axis=1)
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,3]



