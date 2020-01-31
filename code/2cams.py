#!/usr/bin/env python

import numpy as np
import cv2   as cv

cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

while(cv.waitKey(1) & 0xFF != 27):
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    cv.imshow('c1',frame1)
    cv.imshow('c2',frame2)

    gray1 = cv.cvtColor(frame1, cv.COLOR_RGB2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_RGB2GRAY)

    cv.imshow('frame', gray1//2 + gray2//2)

cv.destroyAllWindows()

