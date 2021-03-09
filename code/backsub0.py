#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

kernel = np.ones((3,3),np.uint8)

for key,frame in autoStream():
    fgmask = bgsub.apply(frame)
    
    cv.imshow('original',frame)
    cv.imshow('mask', fgmask)

cv.destroyAllWindows()

