# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 08:29:03 2025

@author: My Account
"""

import keras_tuner 
import keras
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from pandas import read_csv 
batch_size = 64
epochs = 40

dataset = read_csv('Task15_W_Zone6.csv')
values = dataset.values
 
# input image dimensions
# 输入图像维度
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
XY= scaler.fit_transform(values)

# the data, shuffled and split between train and test sets
# 用于训练和测试的数据集，经过了筛选（清洗、数据样本顺序打乱）和分割（分割为训练和测试集）
X= XY[:,0:4]    
Y = XY[:,4]
n_train_hours1 =9000
x_train=XY[:n_train_hours1,0:4]
y_train =Y[:n_train_hours1]
x_test =XY[n_train_hours1:, 0:4]
y_test =Y[n_train_hours1:]



def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Tune number of units.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # Tune the activation function to use.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(x_train,y_train, epochs=5, validation_data=(x_test,y_test))
best_model = tuner.get_best_models()[0]

 
best_model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
from pandas import concat
import numpy as np
forecasttestY0 = best_model.predict(x_test)
#forecasttestY= np.expand_dims(a, axis=1)

inv_yhat =np.concatenate((x_test,forecasttestY0), axis=1)
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,3]