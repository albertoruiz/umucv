# run as follows to use the CPU
# CUDA_VISIBLE_DEVICES="" python cnntest.py

import matplotlib.pyplot as plt
import numpy             as np
import numpy.linalg      as la
import os

if not os.path.exists("mnist.npz"):
    os.system("wget robot.inf.um.es/material/mnist.npz")

mnist = np.load("mnist.npz")

xl,yl,xt,yt = [mnist[d] for d in ['xl', 'yl', 'xt', 'yt']]

cl = np.argmax(yl,axis=1)
ct = np.argmax(yt,axis=1)

print(xl.shape, yl.shape, cl.shape)
print(xt.shape, yt.shape, ct.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Softmax, Flatten

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(5,5), strides=1,
                 padding='same', use_bias=True, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=1,
                 padding='same', use_bias=True, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(xl.reshape(-1,28,28,1), yl, epochs=5, batch_size=500)

