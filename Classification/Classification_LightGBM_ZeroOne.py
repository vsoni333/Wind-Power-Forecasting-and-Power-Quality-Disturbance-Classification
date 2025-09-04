import numpy as np
from tensorflow import keras
import lightgbm as lgb
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

# 转换为 LightGBM 数据集
dtrain = lgb.Dataset(x_train, label=y_train)
dtest = lgb.Dataset(x_test, label=y_test, reference=dtrain)

# LightGBM 参数
params = {
    'objective': 'multiclass',     # 多分类
    'num_class': 8,              # 类别数
    'boosting_type': 'gbdt',      # 梯度提升树
    'max_depth': -1,              # -1 表示不限制深度
    'num_leaves': 64,             # 叶子数
    'learning_rate': 0.1,         # 学习率
    'feature_fraction': 0.8,      # 特征采样
    'bagging_fraction': 0.8,      # 数据采样
    'bagging_freq': 5,
    'metric': 'multi_error'       # 评价指标
}

# 训练模型
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=200,
    valid_sets=[dtest],
    callbacks=[lgb.log_evaluation(period=50)]
)
# 预测
y_pred_proba = bst.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 评估准确率
acc = accuracy_score(y_test, y_pred)
print(f"LightGBM 在 MNIST 上的准确率: {acc:.4f}")




