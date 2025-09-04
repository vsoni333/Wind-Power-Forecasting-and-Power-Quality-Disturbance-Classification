import numpy as np
from tensorflow import keras
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas import read_csv 

dataset = read_csv('Task15_W_Zone6.csv')
values = dataset.values

# （删除无效的归一化）
XY = values
Y = XY[:,4].astype(float)
n_train_hours1 = 15000
x_train = XY[:n_train_hours1, 0:4].astype(float)
y_train = Y[:n_train_hours1].astype(float)
x_test  = XY[n_train_hours1:, 0:4].astype(float)
y_test  = Y[n_train_hours1:].astype(float)

print("x_train.shape", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_test, label=y_test)
dtest  = dvalid  # 保持你的变量用法最小改动

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost 回归 — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

dtest_one = xgb.DMatrix(x_test[0].reshape(1, -1))
y_pred_one = bst.predict(dtest_one, iteration_range=(0, bst.best_iteration + 1))[0]
print("单点预测:", float(y_pred_one))
print("真实值:", float(y_test[0]))
