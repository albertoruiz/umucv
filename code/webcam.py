#!/usr/bin/env python

import cv2   as cv

cap = cv.VideoCapture(0)

while(cv.waitKey(1) != 27):
    ret, frame = cap.read()
    
    cv.imshow('webcam',frame)

cv.destroyAllWindows()

