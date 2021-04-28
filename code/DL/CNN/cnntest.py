#!/usr/bin/env python

# run as follows to use the CPU
# CUDA_VISIBLE_DEVICES="" ./cnntest.py

import numpy             as np
import numpy.linalg      as la

from tensorflow.keras.datasets import mnist

(xl,cl), (xt,ct) = mnist.load_data()
print(xl.shape, cl.shape)
print(xt.shape, ct.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Softmax, Flatten

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(5,5), strides=1, padding='same', use_bias=True, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=1, padding='same', use_bias=True, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xl.reshape(-1,28,28,1)/255, cl, epochs=5, batch_size=500)

model.evaluate(xt.reshape(-1,28,28,1)/255,ct, batch_size=500)

model.save('digits.keras')
