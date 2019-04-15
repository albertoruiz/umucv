#!/usr/bin/env python

# pip install tensorflow
# pip install keras

import cv2   as cv
import numpy as np
from threading import Thread
import time
from umucv.util import putText
from umucv.stream import autoStream


import keras
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image

model = InceptionV3(weights='imagenet')

def classify(img):
    h,w,_ = img.shape
    dw = (w-299)//2
    dh = (h-299)//2
    win = img[dh:299+dh,dw:299+dw]
    arr = preprocess_input(np.expand_dims(win.astype(np.float32), axis=0))
    preds = model.predict(arr)    
    _,lab,p = decode_predictions(preds, top=3)[0][0]
    if p < 0.5: lab = ''
    return win, lab


# la primera llamada debe hacerse fuera del hilo
classify(np.zeros((480,640,3), np.uint8))

frame = None
goon = True
win  = None

def work():
    global win,lab
    while goon:
        if frame is not None:
            t0 = time.time()
            win,lab = classify(frame)    
            t1 = time.time()
            putText(win, '{:.0f}ms  {}'.format(1000*(t1-t0), lab))

t = Thread(target=work,args=())
t.start()

for key, frame in autoStream():
    cv.imshow('cam',frame)
    if win is not None:
            cv.imshow('inception', win)
            win = None

goon = False

