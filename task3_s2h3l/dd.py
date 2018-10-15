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

# data pre-processing

# no validatio data
y_train = train_arr[:,0]
x_train = train_arr[:,1:]
print(x_train.shape)
y_train = keras.utils.to_categorical(y_train,5)# Converts a class vector (integers) to binary class matrix.

# validation data
# y_train = train_arr[0:40000,0]
# x_train = train_arr[0:40000,1:]
# x_test = train_arr[40001:,1:]
# y_test = train_arr[40001:,0]
# y_train = keras.utils.to_categorical(y_train,5)
# y_test = keras.utils.to_categorical(y_test,5)

# model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(None,45324)))####这一行
model.add(Dense(100, activation='relu',input_dim=100))
model.add(Dropout(0,1))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0,1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])