from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling2D,Convolution1D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K # turn image into speficifc shape
import pandas as pd
import numpy as np
import keras
# load data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
# convert to numpy
train_arr = np.array(train)
pre_arr = np.array(test)
print(train_arr.shape)

# data pre-processing

# no validatio data

x_train = train_arr[5000:,1:].reshape(40324,1,100)
x_val = train_arr[0:5000,1:].reshape(5000,1,100)
y_train = train_arr[5000:,0]
y_val = train_arr[0:5000:,0]
print(y_val.shape)
print("======")
y_train = keras.utils.to_categorical(y_train,5).reshape(40324,1,5)# Converts a class vector (integers) to binary class matrix.
y_val = keras.utils.to_categorical(y_val,5).reshape(5000,1,5)#
print ("===y===")
# validation data
# y_train = train_arr[0:40000,0]
# x_train = train_arr[0:40000,1:]
# x_test = train_arr[40001:,1:]
# y_test = train_arr[40001:,0]
# y_train = keras.utils.to_categorical(y_train,5)
# y_test = keras.utils.to_categorical(y_test,5)

#---- model-1 provdies acc~= 0.90-0.92------
model = Sequential()
model.add(Conv1D(80, 3, activation='relu', border_mode='same', input_shape=(1,100)))####
model.add(Conv1D(90, 3, activation='relu', border_mode='same'))####
model.add(Dense(100, activation='selu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0,1))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0,1))
# model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,epochs=800,batch_size=200,validation_data=(x_val,y_val)) # 461 epch
score = model.evaluate(x_train, y_train,batch_size=40000)
print(score[0],score[1])


#---------------------
# model-2 provdies acc~= 0.90-0.92
# model = Sequential()
# model.add(Dense(100, activation='relu',input_dim=100))
# # model.add(Dropout(0,1))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0,1))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(5, activation='softmax'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,epochs=80,batch_size=200)
# score = model.evaluate(x_train, y_train,batch_size=40000)
# print(score[0],score[1])

#-------------
# epch = range(5,20)
# score_arr = np.zeros((len(epch),1))
# for i in epch:
#     a = 0
#     epcho = i*10
#     model.fit(x_train, y_train,epochs=epcho,batch_size=200)
#     score = model.evaluate(x_train, y_train,batch_size=40000)
#     # print(score[0],score[1])
#     score_arr[a] = score[1]
#     a = a+1

# batch_size = 100
# epochs = 100
# model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
# score = model.evaluate(x_train,y_train,verbose=0)
#
x_test = pre_arr.reshape(8137,1,100)
y_prob = model.predict(x_test).reshape(8137,5)
y_classes = y_prob.argmax(axis=-1)
# print(y_classes)
result = pd.DataFrame(columns=['Id','y'])
result['y'] = np.transpose(y_classes)
result['Id'] = range(45324,53461)
result.to_csv('tf_14.csv',index=False,header=True)
