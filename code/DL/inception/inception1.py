#!/usr/bin/env python

import cv2   as cv
import numpy as np
from threading import Thread
import time
from umucv.util import putText
from umucv.stream import autoStream

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

model = InceptionV3(weights='imagenet')

def classify(img):
    h,w,_ = img.shape
    dw = (w-299)//2
    dh = (h-299)//2
    win = img[dh:299+dh,dw:299+dw]
    arr = preprocess_input(np.expand_dims(win.astype(np.float32), axis=0))
    preds = model.predict(arr)    
    r = decode_predictions(preds, top=5)[0]
    _,lab,p = r[0]
    if p < 0.5: lab = ''
    #print(r)
    print([l for _,l,_ in r])
    return win, lab


frame = None
goon = True
win  = None

def GUI():
    global frame, goon, win
    for key, frame in autoStream():
        cv.imshow('cam',frame)
        if win is not None:
            cv.imshow('inception', win)
            win = None
    goon = False

t = Thread(target=GUI,args=())
t.start()

while frame is None: pass

while goon:    
    
    t0 = time.time()

    win,lab = classify(frame)
    
    t1 = time.time()
    putText(win, '{:.0f}ms  {}'.format(1000*(t1-t0), lab))

