#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream

from collections import deque

d = deque(maxlen=20)

for key,frame in autoStream():
    d.append(frame)
    cv.imshow('input',d[-1])
    cv.imshow('delay',d[0])
    cv.imshow('ghost',np.mean(d,axis=0)/255)

