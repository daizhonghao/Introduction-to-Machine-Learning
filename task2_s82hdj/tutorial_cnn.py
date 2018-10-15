import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.datasets import mnist
from keras import backend as K # turn image into speficifc shape
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
np.random.seed(100)

mnist.load_data()
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape) #60000 instances, each 28*28 matrix
print(x_test.shape)
# depth of image = different color channels , here 1 deepth
if(K.image_data_format()=='channels_first'):
    input_shape  = (1,28,28)
else:
    input_shape = (28,28,1)

#pre -process
print(y_train.shape)
y_train = keras.utils.to_categorical(y_train,10) # turn numerical class(0-10) to categorical
print(y_train.shape)
y_test = keras.utils.to_categorical(y_test,10)
x_train = x_train.reshape(x_train.shape[0],*input_shape)
x_test = x_test.reshape(x_test.shape[0],*input_shape)
print(x_train.shape)
print(x_train.shape[0])
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255 # normalize from 0-1
x_test /= 255
CNN = Sequential()
CNN.name = 'CNN'
# add convolution layer
CNN.add(Conv2D(32,kernel_size=(2,2),activation='relu',input_shape=input_shape)) # first layer
print(CNN.output.shape) # n = 28,p = 0, s=1, f =2 ; (28-2)/1  + 1, m = 32???? kernel / depth = 32
CNN.add(Conv2D(30,kernel_size=(3,3),activation='tanh'))  # sencond layer
# add pooling layer
CNN.add(MaxPooling2D(pool_size=(3,3)))
# add dropout layer
CNN.add(Dropout(0,1))
CNN.add(Flatten()) # Flatten layer , need to add before fully connected layer
CNN.add(Dense(100,activation='relu')) # fuller connceted layer arg1  = # output from that layer
CNN.add(Dense(10,activation='softmax')) # output
CNN.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
batch_size = 80
epochs = 1
CNN.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
score = CNN.evaluate(x_test,y_test,verbose=0)
print(score[0],score[1]) #loss and acc