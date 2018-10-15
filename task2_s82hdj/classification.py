import pandas as pd
import sklearn
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import RidgeClassifier,Perceptron,SGDClassifier,LogisticRegression
from sklearn.neural_network import MLPClassifier,BernoulliRBM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.neighbors import  KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,RandomForestClassifier,VotingClassifier
from scipy import stats

csv = np.genfromtxt ('train.csv', delimiter=",")
csv_test = np.genfromtxt ('test.csv', delimiter=",")
x_train = csv[1:,2:]
y_train = csv[1:,1]
x_test = csv_test[1:,1:]
print(np.shape(y_train))

# k_foler
kf = KFold(n_splits=5)

clf0 = SVC(C=100, kernel='rbf', degree=6, class_weight='balanced', gamma=1 / 4000)
clf1 = GradientBoostingClassifier(n_estimators=400, min_samples_split=10, max_features='sqrt')
clf2 = RandomForestClassifier(n_estimators=300, max_features='sqrt')
clf3 = RandomForestClassifier(n_estimators=500)
clf4 = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', alpha=0.0001, max_iter=500, batch_size=300)  # 可选
clf5 = ExtraTreesClassifier(n_estimators=400,random_state=10)  # it is good
clf6 = ExtraTreesClassifier(n_estimators=300,random_state=20)  # it is good
clf7 = ExtraTreesClassifier(n_estimators=200,random_state=100)  # it is good
clf8 = ExtraTreesClassifier(n_estimators=200,random_state=300)
clf9 = ExtraTreesClassifier(n_estimators=200,random_state=200)
clf10 = ExtraTreesClassifier(n_estimators=200,random_state=2)
clf11 = ExtraTreesClassifier(n_estimators=200,random_state=3)
clf_v = VotingClassifier(estimators=[('1', clf1), ('2', clf2), ('3', clf3), ('4', clf4), ('5', clf0), ('6', clf5),('7', clf6),('8', clf7),('9',clf8),('10',clf9),('11',clf10),('12',clf11)],weights=[1,1,1,1,1,2,2,2,2,2,3,3])

# acc = 0
# for train_index, test_index in kf.split(x_train):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     x_train_sub, x_test_sub = x_train[train_index], x_train[test_index]
#     y_train_sub, y_test_sub = y_train[train_index], y_train[test_index]
#     y_predict_array = np.zeros((200,5))
#     # clf0 = SVC(C=100,kernel='rbf',degree=6,class_weight ='balanced',gamma = 1/4000)
#     # clf1 = GradientBoostingClassifier(n_estimators=400,min_samples_split=10,max_features='sqrt')
#     # clf2 = RandomForestClassifier(n_estimators=300,max_features='sqrt')
#     # clf3 = RandomForestClassifier(n_estimators=500)
#     # #clf1 = RidgeClassifier(alpha=1)
#     # #clf1 = Perceptron(alpha=10000)
#     # clf4 = MLPClassifier(hidden_layer_sizes=(100, ),activation='tanh',alpha=0.0001,max_iter = 500,batch_size=300) # 可选
#     # clf5 = ExtraTreesClassifier(n_estimators=400) # it is good
#     #clf5.fit(x_train_sub, y_train_sub)
#     #y_predict = clf5.predict(x_test_sub)
#     #clf1 = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh',solver='sgd',alpha=0.0001, max_iter=500,
#                          #batch_size=200,learning_rate='adaptive')  # 可选
#     #clf1 = GradientBoostingClassifier() # 可选
#     #clf1 = SGDClassifier(loss='log',penalty='l2',alpha=1)
#     # clf1 = KNeighborsClassifier(n_neighbors=10,weights='distance')
#     # clf1 = LogisticRegression(penalty='l2',C=10,multi_class= 'multinomial',solver='newton-cgxx')
#     # clf1 = GaussianProcessClassifier(multi_class='one_vs_one')
#     #clf1 = DecisionTreeClassifier(min_samples_split=10)
#     #clf1 = RadiusNeighborsClassifier(radius=1000, weights='distance', algorithm='kd_tree')
#
#
#     clf_v = VotingClassifier(estimators=[('1',clf1),('2',clf2),('3',clf3),('4',clf4),('5',clf0),('6',clf5)],weights=[1,1,1,1,1,2])
#     clf_v.fit(x_train_sub, y_train_sub)
#     y_predict = clf_v.predict(x_test_sub)
#     acc = acc + accuracy_score(y_test_sub, y_predict)
#
# print(acc/5)


clf_v.fit(x_train,y_train)
y_predict = clf_v.predict(x_train)
print(accuracy_score(y_predict, y_train))
y_test = clf_v.predict(x_test)
result = pd.DataFrame(columns=['Id','y'])
result['y'] = np.transpose(y_test)
result['Id'] = range(2000,5000)
result.to_csv('result_vote12.csv',index=False,header=True)

