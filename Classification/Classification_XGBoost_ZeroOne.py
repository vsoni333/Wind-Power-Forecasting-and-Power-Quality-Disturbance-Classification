import numpy as np
from tensorflow import keras
import xgboost as xgb
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


# 展开为 784 维向量 Expand into a 784-dimensional vector
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 转换为 DMatrix 格式（XGBoost 高效输入格式） Convert to DMatrix format (an efficient input format for XGBoost)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# XGBoost 参数 parameter
params = {
    'objective': 'multi:softmax',   # 多分类
    'num_class': 10,               # 类别数
    'max_depth': 6,                # 树的最大深度
    'eta': 0.3,                    # 学习率
    'subsample': 0.8,              # 子样本比例
    'colsample_bytree': 0.8,       # 每棵树的特征采样比例
    'eval_metric': 'merror'        # 评价指标
}


bst = xgb.train(params, dtrain, num_boost_round=200)


y_pred = bst.predict(dtest)


acc = accuracy_score(y_test, y_pred)
print(f"XGBoost 在 MNIST 上的准确率: {acc:.4f}")