#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)


for key,frame in autoStream():
    fgmask = bgsub.apply(frame)
    
    cv.imshow('original',frame)
    cv.imshow('mask', fgmask)

cv.destroyAllWindows()

