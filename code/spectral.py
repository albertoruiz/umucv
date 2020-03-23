#!/usr/bin/env python

import numpy             as np
import numpy.fft         as fft
import cv2               as cv
from umucv.stream import autoStream

def center(x):
    r,c = x.shape
    y = x[list(range(r//2,r)) + list(range(r//2)) ,:]
    y = y[:, list(range(c//2,c)) + list(range(c//2))]
    return y

def f(x):
    r,c = x.shape[:2]
    c2 = (c-r)//2
    g = cv.cvtColor(x[:,c2:c2+r] , cv.COLOR_BGR2GRAY)
    f = g.astype(float)/255
    y = abs(fft.fft2(f))
    y[0,0] = 0
    y = np.log(1+y)
    y = y/np.max(y)
    return center(y)
    
for key, frame in autoStream():

    cv.imshow('input', frame)
    cv.imshow('FFT2D', f(frame) )

