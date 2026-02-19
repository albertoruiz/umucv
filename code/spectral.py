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

def showF(f):
    y = abs(f)
    y[0,0] = 0
    y = np.log(1+y)
    y = y/np.max(y)
    return center(y)

for key, frame in autoStream():
    r,c = frame.shape[:2]
    c2 = (c-r)//2
    g = cv.cvtColor(frame[:,c2:c2+r] , cv.COLOR_BGR2GRAY)
    x = g.astype(float)/255
    f = fft.fft2(x)
    cv.imshow('input', x)
    cv.imshow('FFT2D', showF(f) )
