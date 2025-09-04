import numpy as np
import keras 
from keras import layers 
from pandas import read_csv

input_shape = (2,2,1)

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
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),  # 输入层，输入形状为 Input layer, input shape is 28x28x1
        layers.Conv2D(32, kernel_size=(1, 1), activation="relu"),  # 卷积层，32个3x3卷积核，激活函数为ReLU Convolutional layer, 32 3x3 convolutional kernels, activation function is ReLU
        layers.MaxPooling2D(pool_size=(1, 1)),  # 最大池化层，池化窗口大小为2x2  Max pooling layer, pooling window size of 2x2
        layers.Conv2D(64, kernel_size=(1, 1), activation="relu"),  # 卷积层，64个3x3卷积核，激活函数为ReLU  Convolutional layer, 64 3x3 convolutional kernels, activation function is ReLU
        layers.MaxPooling2D(pool_size=(1, 1)),  # 最大池化层，池化窗口大小为2x2 Max pooling layer, pooling window size of 2x2
        layers.Flatten(),  # 将卷积层的输出展平成一维向量 Flatten the output of the convolutional layer into a one-dimensional vector.
        layers.Dropout(0.5),  # Dropout层，丢弃50%的节点，用于防止过拟合 Dropout layer, which discards 50% of the nodes, is used to prevent overfitting.
        layers.Dense(1, activation="sigmoid")  # 全连接层，输出节点数为类别数，激活函数为Softmax  Fully connected layer, with the number of output nodes equal to the number of categories, and the activation function being Softmax.
    ]
)
model.summary()

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



