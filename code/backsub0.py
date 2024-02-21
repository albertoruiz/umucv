#!/usr/bin/env python

# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from umucv.stream import autoStream

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)


for key,frame in autoStream():
    fgmask = bgsub.apply(frame)
    
    cv.imshow('original',frame)
    cv.imshow('mask', fgmask)

cv.destroyAllWindows()

