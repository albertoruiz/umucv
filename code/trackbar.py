#!/usr/bin/env python

import cv2   as cv
import numpy as np

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY)

h = 128

def update(v):
    global h
    h = v

cv.namedWindow("binary")
cv.createTrackbar("umbral", "binary", h, 255, update)

cap = cv.VideoCapture(0)
assert cap.isOpened()

while(cv.waitKey(1) & 0xFF != 27):
    ret, frame = cap.read()
    cv.imshow('binary', (bgr2gray(frame) > h).astype(np.float) )

cv.destroyAllWindows()

