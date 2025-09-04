import numpy as np
import keras
from keras import layers
from pandas import read_csv

# 参数
num_classes = 8
timesteps = 28   # 序列长度（每张图片的行数）
input_dim = 28   # 每个时间步的特征数（每行像素）
batch_size = 128
epochs = 15


dataset = read_csv('ZerosOnePowerQuality.csv', header=None)
values = dataset.values
XY = values
Y = XY[:,28*28]
n_train_hours1 =9000
x_train = XY[:n_train_hours1, 0:28*28].reshape(n_train_hours1, 28, 28)
y_train = Y[:n_train_hours1]
x_test = XY[n_train_hours1:, 0:784].reshape(XY.shape[0] - n_train_hours1, 28, 28)
y_test = Y[n_train_hours1:]


print("x_train.shape:", x_train.shape)  # (60000, 28, 28)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建 LSTM 模型
model = keras.Sequential([
    keras.Input(shape=(timesteps, input_dim)),
    layers.LSTM(128, return_sequences=True),  # LSTM 层
    layers.LSTM(128, return_sequences=True),  # LSTM 层
    layers.LSTM(128, return_sequences=False),  # LSTM 层
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")  # 输出分类
])
model.summary()

# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print(f"LSTM 在 MNIST 上的准确率: {score[1]:.4f}")
