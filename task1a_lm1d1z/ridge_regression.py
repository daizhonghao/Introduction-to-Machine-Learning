import numpy as np
from sklearn import datasets, linear_model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
csv = np.genfromtxt ('train.csv', delimiter=",")
data = csv[1:]
x_train = data[:,2:]
y_train = data[:,1]
# print(y_train.shape)

alp = [0.1,1,10,100,1000]
# reg = linear_model.Ridge (alpha = 1000)
# reg.fit(x_train,y_train)
# y_predict = reg.predict(x_train)
# RMSE = mean_squared_error(y_train, y_predict)**0.5
# print(RMSE)

kf = KFold(n_splits=10)

for a in alp:
    result = 0
    for train_index, test_index in kf.split(x_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train_sub, x_test_sub = x_train[train_index], x_train[test_index]
        y_train_sub, y_test_sub = y_train[train_index], y_train[test_index]
        reg = linear_model.Ridge(alpha=a)
        reg.fit(x_train_sub, y_train_sub)
        y_predict = reg.predict(x_test_sub)
        RMSE = mean_squared_error(y_test_sub, y_predict) ** 0.5
        result = result + RMSE
    result  = result/10.0
    print(result)
