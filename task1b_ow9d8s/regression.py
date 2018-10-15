import numpy as np
from sklearn import datasets, linear_model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
csv = np.genfromtxt ('train.csv', delimiter=",")
data = csv[1:]
x_train = data[:,2:]
print(x_train.shape)
y_train = data[:,1]

x_train_new = np.zeros((900,21))

print(x_train_new.shape)
for i in range(0,900):
    x_train_new[i,0:4] = x_train[i,0:4]
    x_train_new[i,5:9] = (x_train[i, 0:4])**2
    x_train_new[i,10:14] = np.exp((x_train[i, 0:4]))
    x_train_new[i,15:19] = np.cos((x_train[i, 0:4]))
    x_train_new[i,20] = 1

# regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha=1000)

# regr = linear_model.RidgeCV(alphas=(0.01,0.1, 1.0, 10.0,100,1000),cv=10)
regr.fit(x_train_new, y_train)
y_predict = regr.predict(x_train_new)
# RMSE = mean_squared_error(y_train, y_predict)**0.5
# print(RMSE)
coe = np.array(regr.coef_).transpose()
result = pd.DataFrame(columns=['coe'])
result['coe'] = coe
result.to_csv('result.csv',index=False,header=False)
