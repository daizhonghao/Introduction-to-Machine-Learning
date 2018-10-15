import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
# print(train)
train_arr = np.array(train)
y_train = train_arr[:,0]
x_train = train_arr[:,1:]
pre_arr = np.array(test)
# print(y_train.shape)
kf = KFold(n_splits=5)
acc = 0
# for train_index, test_index in kf.split(x_train):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     x_train_sub, x_test_sub = x_train[train_index], x_train[test_index]
#     y_train_sub, y_test_sub = y_train[train_index], y_train[test_index]
#     clf = MLPClassifier(activation='tanh',solver='sgd', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
#     clf.fit(x_train_sub, y_train_sub)
#     y_predict = clf.predict(x_test_sub)
#     acc = acc + accuracy_score(y_test_sub, y_predict)
# print(acc/5)
#clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(900,),momentum=0.9, early_stopping=True,random_state=1,validation_fraction=0.1,tol =1e-4 )
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,200,100,50),momentum=0.9,random_state=2,learning_rate='adaptive',validation_fraction=0.1,tol =1e-4,max_iter=500,batch_size=500)
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,200,200,100,50),momentum=0.9,random_state=2,learning_rate='adaptive',tol =1e-4,max_iter=500,batch_size=500) # 0.839
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(300,200,200,100,50),momentum=0.9,random_state=2,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # 0.8636
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(300,200,200,200,100,100,50),momentum=0.9,random_state=2,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # 0.896
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(300,300,300,200,200,200,100),momentum=0.9,random_state=2,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # ??
# 0.9092
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(300,200,200,100,100,100,50),momentum=0.9,random_state=4,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # ?try
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_train)
# print(accuracy_score(y_train, y_predict))
# y_test = clf.predict(pre_arr)
# result = pd.DataFrame(columns=['Id','y'])
# result['y'] = np.transpose(y_test)
# result['Id'] = range(45324,53461)
# result.to_csv('MPL_1_layer.csv',index=False,header=True)

# 0.9135
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(300,200,300,200,100,100,50),momentum=0.9,random_state=4,learning_rate='adaptive',tol =1e-5,max_iter=900,batch_size=500) # ?try
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_train)
# print(accuracy_score(y_train, y_predict))
# y_test = clf.predict(pre_arr)
# result = pd.DataFrame(columns=['Id','y'])
# result['y'] = np.transpose(y_test)
# result['Id'] = range(45324,53461)
# result.to_csv('MPL_2.csv',index=False,header=True)

# 0.9071
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(400,300,200,200,100,100,50,50),momentum=0.9,random_state=4,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # ?try
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_train)
# print(accuracy_score(y_train, y_predict))
# y_test = clf.predict(pre_arr)
# result = pd.DataFrame(columns=['Id','y'])
# result['y'] = np.transpose(y_test)
# result['Id'] = range(45324,53461)
# result.to_csv('MPL_3.csv',index=False,header=True)

# 0.8760
# clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(400,300,200,200,100,100,100),momentum=0.9,random_state=6,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # ?try
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_train)
# print(accuracy_score(y_train, y_predict))
# y_test = clf.predict(pre_arr)
# result = pd.DataFrame(columns=['Id','y'])
# result['y'] = np.transpose(y_test)
# result['Id'] = range(45324,53461)
# result.to_csv('MPL_8.csv',index=False,header=True)

clf = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(400,300,400,300,200,100,100),momentum=0.9,random_state=6,learning_rate='adaptive',tol =1e-5,max_iter=800,batch_size=500) # ?try
clf.fit(x_train, y_train)
y_predict = clf.predict(x_train)
print(accuracy_score(y_train, y_predict))
y_test = clf.predict(pre_arr)
result = pd.DataFrame(columns=['Id','y'])
result['y'] = np.transpose(y_test)
result['Id'] = range(45324,53461)
result.to_csv('MPL_10.csv',index=False,header=True)
