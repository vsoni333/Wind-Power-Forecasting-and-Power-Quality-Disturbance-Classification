import numpy as np
import keras 
from keras import layers 

# 模型和数据参数 Models and data parameters
num_classes = 10  # 类别数，共10类 Number of categories, totaling 10 categories
input_shape = (28,28,1)  # 输入数据的形状 The shape of input data(28, 28, 1)

# 加载数据并将其分为训练集和测试集 Load the data and divide it into training and testing sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 将图像像素值缩放到 [0, 1] 范围 Scale image pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 确保图像具有形状 (28, 28, 1) Ensure that the image has the shape (28, 28, 1).
x_train = np.expand_dims(x_train, -1)  # 在最后添加一个维度 Add one more dimension at the end.
x_test = np.expand_dims(x_test, -1)  # 在最后添加一个维度 Add one more dimension at the end.

print("x_train.shape", x_train.shape)  # 输出训练集的形状Output the shape of the training set. #(60000, 28, 28, 1)
print(x_train.shape[0], "train samples")  # 输出训练样本数量 output training sample quantity #60000 train samples
print(x_test.shape[0], "test samples")  # 输出测试样本数量 output test sample quantity #10000 test samples

# 将类别向量转换为二进制类别矩阵 Convert category vectors to binary category matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print()
print("Build the model")  # 构建模型 build a model
# 构建模型 build a model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),  # 输入层，输入形状为 Input layer, input shape is 28x28x1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),  # 卷积层，32个3x3卷积核，激活函数为ReLU Convolutional layer, 32 3x3 convolutional kernels, activation function is ReLU
        layers.MaxPooling2D(pool_size=(2, 2)),  # 最大池化层，池化窗口大小为2x2  Max pooling layer, pooling window size of 2x2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # 卷积层，64个3x3卷积核，激活函数为ReLU  Convolutional layer, 64 3x3 convolutional kernels, activation function is ReLU
        layers.MaxPooling2D(pool_size=(2, 2)),  # 最大池化层，池化窗口大小为2x2 Max pooling layer, pooling window size of 2x2
        layers.Flatten(),  # 将卷积层的输出展平成一维向量 Flatten the output of the convolutional layer into a one-dimensional vector.
        layers.Dropout(0.5),  # Dropout层，丢弃50%的节点，用于防止过拟合 Dropout layer, which discards 50% of the nodes, is used to prevent overfitting.
        layers.Dense(num_classes, activation="softmax")  # 全连接层，输出节点数为类别数，激活函数为Softmax  Fully connected layer, with the number of output nodes equal to the number of categories, and the activation function being Softmax.
    ]
)
model.summary()  # 输出模型摘要 Generate model summary

print()
print("Train the model")  # 训练模型 train a model
# 训练模型 train a model
batch_size = 128  # 每个批次的样本数 The number of samples per batch
epochs = 15  # 训练轮数 training rounds

# 编译模型，指定损失函数、优化器和评估指标 Compile the model, specifying the loss function, optimizer, and evaluation metrics.
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型，划分10%的训练数据用于验证 Train the model, using 10% of the training data for validation.
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

print()
print("Evaluate the model")  # 评估模型 evaluate model
# 评估训练后的模型 Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)  # 在测试集上评估模型 Evaluate the trained model
print("Test loss:", score[0])  #Test loss: 0.025095002725720406
print("Test accuracy:", score[1])  #Test accuracy: 0.991100013256073