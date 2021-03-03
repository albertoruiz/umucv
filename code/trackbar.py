#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream

def nada(v):
    pass

cv.namedWindow("binary")
cv.createTrackbar("umbral", "binary", 128, 255, nada)

for key, frame in autoStream():
    h = cv.getTrackbarPos('umbral','binary')
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #binary = (gray > h).astype(np.float)
    binary = (gray > h).astype(np.uint8)*255
    cv.imshow('binary', binary )

cv.destroyAllWindows()

