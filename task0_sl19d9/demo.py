import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
csv = np.genfromtxt ('train.csv', delimiter=",")
print(csv.shape)
data = csv[1:]
print(data.shape)
x_train = data[:,2:]
y_train = data[:,1]
# print(x_train)
# print(y_train)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

csv = np.genfromtxt ('test.csv', delimiter=",")
x_test = csv[1:,1:]
ID = np.array(csv[1:,0]).astype(int)
# print(x_test)
# y_test = regr.predict(x_test)
y_test = np.average(x_test,axis=1)
print(y_test)
# num, = y_test.shape
# result = np.zeros((num,2))
# result[:,0] = ID
# result[:,1] = y_test
result = pd.DataFrame(columns=['Id','y'])
result['Id'] = ID
result['y']= y_test
print(result)
result.to_csv('result.csv',index=False)
# np.savetxt("result.csv", result, delimiter=",")