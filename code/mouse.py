#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)


for key, frame in autoStream():
    cv.imshow('webcam',frame)

cv.destroyAllWindows()

