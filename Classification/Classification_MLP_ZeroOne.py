
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from pandas import read_csv 
batch_size = 64
num_classes = 8
epochs = 320
 
# input image dimensions
# 输入图像维度

# the data, shuffled and split between train and test sets
# 用于训练和测试的数据集，经过了筛选（清洗、数据样本顺序打乱）和分割（分割为训练和测试集）
dataset = read_csv('ZerosOnePowerQuality.csv',header=None)
values = dataset.values
XY= values
Y = XY[:,784]
n_train_hours1 =9000
x_train=XY[:n_train_hours1,0:784]
trainY =Y[:n_train_hours1]
x_test =XY[n_train_hours1:, 0:784]
testY =Y[n_train_hours1:]
y_train = keras.utils.to_categorical(trainY, num_classes)
y_test = keras.utils.to_categorical(testY, num_classes)
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(784,)))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()
 
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adamax(),metrics=['accuracy'])
 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])