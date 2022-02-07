#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream

def nada(v):
    pass

cv.namedWindow("binary")
cv.createTrackbar("umbral", "binary", 128, 255, nada)

for key, frame in autoStream():
    #print(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #print(gray)
    h = cv.getTrackbarPos('umbral','binary')
    logica = gray > h
    #print(logica)
    #binary = logica.astype(np.uint8)*128
    binary = logica.astype(np.float)
    #print(binary)
    cv.imshow('binary', binary )

cv.destroyAllWindows()

