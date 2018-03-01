#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream

from collections import deque

d = deque(maxlen=20)

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY).astype(float)/255

for key,frame in autoStream():
    d.appendleft(bgr2gray(frame))
    cv.imshow('input',d[0])
    cv.imshow('delay',d[-1])
    cv.imshow('ghost',np.mean(d,axis=0))

cv.destroyAllWindows()

