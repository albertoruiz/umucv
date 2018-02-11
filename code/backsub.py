#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

kernel = np.ones((3,3),np.uint8)

for key,frame in autoStream():
    bgs = bgsub.apply(frame)
    
    bgs = cv.erode(bgs,kernel,iterations = 1)
    bgs = cv.medianBlur(bgs,3)

    cv.imshow('original',frame)
    cv.imshow('mask', bgs)

cv.destroyAllWindows()

