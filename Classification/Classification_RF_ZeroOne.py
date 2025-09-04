import numpy as np
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas import read_csv 

dataset = read_csv('ZerosOnePowerQuality.csv', header=None)
values = dataset.values
XY= values
Y = XY[:,784]
n_train_hours1 =9000
x_train=XY[:n_train_hours1,0:784]
y_train =Y[:n_train_hours1]
x_test =XY[n_train_hours1:, 0:784]
y_test =Y[n_train_hours1:]

# 展开为 784 维向量
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 定义随机森林分类器
rf = RandomForestClassifier(
    n_estimators=200,      # 树的数量
    max_depth=None,        # 不限制深度
    n_jobs=-1,             # 用全部 CPU 核
    random_state=42
)

# 训练模型
rf.fit(x_train, y_train)

# 预测
y_pred = rf.predict(x_test)

# 评估准确率
acc = accuracy_score(y_test, y_pred)
print(f"随机森林在 MNIST 上的准确率: {acc:.4f}")
