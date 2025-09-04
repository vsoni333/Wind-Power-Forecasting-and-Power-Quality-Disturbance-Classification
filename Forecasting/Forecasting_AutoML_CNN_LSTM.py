# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:28:16 2025

@author: My Account
"""

from pandas import read_csv

dataset = read_csv('Task15_W_Zone6.csv')
values = dataset.values
#2.tranform data to [0,1]  3个属性，第4个是待预测量

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
XY= scaler.fit_transform(values)

X= XY[:,0:3]    
Y = XY[:,3]
#3.split into train and test sets 950个训练集，剩下的都是验证集
n_train_hours = 9000
trainX = X[:n_train_hours, :]
trainY =Y[:n_train_hours]
testX = X[n_train_hours:, :]
testY =Y[n_train_hours:]

trainX = trainX.reshape((9000, 1, 3))
testX = testX.reshape((7800, 1, 3))


import keras
from keras import layers
import keras_tuner

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=hp.Int("units", min_value=32, max_value=512, step=32), kernel_size=1, padding='same', strides=1,
                            activation='relu', input_shape=(1,3)))
    model.add(layers.MaxPooling1D(pool_size=1))
    model.add(layers.LSTM(units=3, return_sequences=True, activation='relu'))
    model.add(layers.LSTM(units=3, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(units=1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam')
    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
)

tuner.search(trainX, trainY, epochs=5, batch_size=10, validation_data=(testX, testY), verbose=1, shuffle=False)
best_model = tuner.get_best_models(num_models=1)[0]

model = best_model
model.summary

history = model.fit(trainX,trainY, epochs=100, batch_size=20,validation_data=(testX, testY), verbose=1, shuffle=False)

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

from pandas import concat
import numpy as np
forecasttestY0 = model.predict(testX)

inv_yhat = XY[n_train_hours:, :].copy()
inv_yhat[:, 3] = forecasttestY0.ravel()
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,3]
