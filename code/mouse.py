#!/usr/bin/env python

import cv2 as cv

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

cap = cv.VideoCapture(0)

while(cv.waitKey(1) & 0xFF != 27):
    ret, frame = cap.read()
    cv.imshow('webcam',frame)
    
cv.destroyAllWindows()

