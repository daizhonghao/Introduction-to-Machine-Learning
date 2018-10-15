from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling2D,Convolution1D
from keras.optimizers import SGD,Adam,RMSprop
from keras.layers import Dense, Activation, Dropout, Flatten, normalization
from keras.datasets import mnist
from keras import backend as K # turn image into speficifc shape
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
import keras
# load data
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
train_labeled = np.random.permutation(train_labeled)
test = pd.read_hdf("test.h5", "test")

# y_train_original = train_labeled.values[2000:,0]
# X_train_original = train_labeled.values[2000:,1:129].reshape(7000,1,128)
# y_val =  train_labeled.values[0:2000,0]
# X_val =  train_labeled.values[0:2000,1:129].reshape(2000,1,128)
y_train_original = train_labeled[2000:,0]
X_train_original = train_labeled[2000:,1:129].reshape(7000,1,128)
y_val =  train_labeled[0:2000,0]
X_val =  train_labeled[0:2000,1:129].reshape(2000,1,128)
X_unlabeled = train_unlabeled.values[:,0:128].reshape(21000,1,128)
y_train_original = keras.utils.to_categorical(y_train_original,10).reshape(7000,1,10)
y_val =  keras.utils.to_categorical(y_val,10).reshape(2000,1,10)
X_valid = test.values[:,0:128].reshape(8000,1,128)


#
# print("====last_time===")
model = Sequential()
model.add(BatchNormalization(input_shape=(1,128)))
model.add(Conv1D(500, 5, activation='relu', border_mode='same', input_shape=(1,128)))####
model.add(Dropout(0.5))
# model.add(Conv1D(500, 10, activation='relu', border_mode='same', input_shape=(1,128)))####
# model.add(Dropout(0.5))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(400, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

adam = Adam()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_original, y_train_original, epochs=50, verbose=1, batch_size=100)  # 461 epch
score = model.evaluate(X_val, y_val)
print("==acc last = first ===")
acc_last = score[1]
print(acc_last)

# predict
y_unlabelled = model.predict_proba(X_unlabeled, batch_size=32, verbose=0)
y_add_train = model.predict_classes(X_unlabeled)
y_max_p = y_unlabelled.max(axis=1).max(axis =1)
indice = y_max_p > (np.mean(y_max_p)+2.58 * np.std(y_max_p)/(len(y_max_p)**0.5))
X_add_train = X_unlabeled[indice, :]
y_add_train = keras.utils.to_categorical(y_add_train[indice], 10).reshape(len(y_add_train[indice]),1,10)
X_train_original = np.concatenate((X_train_original, X_add_train), axis=0)
y_train_original = np.concatenate((y_train_original, y_add_train), axis=0)
print(y_train_original.shape)
X_unlabeled = X_unlabeled[~indice,:]


acc_now  =  acc_last
print("==acc now = first ===")
print(acc_now)
i = 0
while acc_now >= acc_last:
    # y_train_original = train_labeled.values[2000:, 0]
    # X_train_original = train_labeled.values[2000:, 1:129]
    acc_last = acc_now
    print("traing data size")
    print(y_train_original.shape)
    print("====update=====")
    model = Sequential()
    model.add(BatchNormalization(input_shape=(1, 128)))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    adam = Adam()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train_original, y_train_original, epochs=60+5*i, verbose=0, batch_size=100)  # 461 epch
    score = model.evaluate(X_val, y_val)
    acc_now = score[1]

    # predict
    y_unlabelled = model.predict_proba(X_unlabeled, batch_size=32, verbose=0)
    y_add_train = model.predict_classes(X_unlabeled)
    y_max_p = y_unlabelled.max(axis=1).max(axis=1)
    indice = y_max_p > (np.mean(y_max_p) + 2.58 * np.std(y_max_p)/(len(y_max_p**0.5)))
    X_add_train = X_unlabeled[indice, :]
    y_add_train = keras.utils.to_categorical(y_add_train[indice], 10).reshape(len(y_add_train[indice]), 1, 10)
    X_train_original = np.concatenate((X_train_original, X_add_train), axis=0)
    y_train_original = np.concatenate((y_train_original, y_add_train), axis=0)
    print("training data now")
    print(y_train_original.shape)
    if len(X_unlabeled[~indice, :]) == 0:
        break
    else:
        X_unlabeled = X_unlabeled[~indice, :]
    print("acc last---  acc now")
    print(acc_last, acc_now)
    if acc_now<=acc_last:
        break

    i = i + 1

    y_prob = model.predict(X_valid).reshape(8000,10)
    y_classes = y_prob.argmax(axis=-1)
    result = pd.DataFrame(columns=['Id', 'y'])
    result['y'] = np.transpose(y_classes)
    result['Id'] = range(30000, 38000)
    result.to_csv('labelled'+str(i)+'.csv', index=False, header=True)






