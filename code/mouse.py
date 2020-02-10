#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)

cv.namedWindow("webcam", cv.WINDOW_NORMAL)
cv.setMouseCallback("webcam", fun)

#cap = cv.VideoCapture(0)

#while(cv.waitKey(1) & 0xFF != 27):
#    ret, frame = cap.read()

for key, frame in autoStream():
    cv.imshow('webcam',frame)
    
cv.destroyAllWindows()

