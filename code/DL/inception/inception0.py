#!/usr/bin/env python

# pip install tensorflow
# pip install keras

import cv2   as cv
import numpy as np
import time
from umucv.util import putText
from umucv.stream import autoStream

MODEL=0

if MODEL==0:
    from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
    model = InceptionV3(weights='imagenet')
    S = 299
if MODEL==1:
    from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    model = VGG16(weights='imagenet')
    S = 224
if MODEL==2:
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    model = ResNet50(weights='imagenet')
    S = 224


def classify(img):
    arr = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))
    preds = model.predict(arr)    
    _,lab,p = decode_predictions(preds, top=3)[0][0]
    if p < 0.5:
        lab = ''
    return lab


for key, frame in autoStream():
    
    h,w,_ = frame.shape
    dw = (w-S)//2
    dh = (h-S)//2
    trozo = frame[dh:S+dh,dw:S+dw]

    t0 = time.time()
    lab = classify(trozo)    
    t1 = time.time()
    putText(frame, '{:.0f}ms  {}'.format(1000*(t1-t0), lab))
    cv.rectangle(frame, (dw,dh),(dw+S,dh+S), (0,0,0), 3)
    cv.imshow('inception', frame)

