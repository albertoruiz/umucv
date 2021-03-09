#!/usr/bin/env python

# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, mkStream

virt = mkStream(dev='dir:../images/cube3.png')
virtbgf = next(virt)

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

kernel = np.ones((3,3),np.uint8)

update = True

for key,frame in autoStream():

    if key == ord('c'):
        update = not update
        if not update: cv.imshow('background',bgsub.getBackgroundImage())

    fgmask = bgsub.apply(frame, learningRate = -1 if update else 0)
    
    fgmask = cv.erode(fgmask,kernel,iterations = 1)
    fgmask = cv.medianBlur(fgmask,3)

    if update: cv.circle(frame,(15,15),6,(0,0,255),-1)
    cv.imshow('original',frame)
    cv.imshow('mask', fgmask)

    masked = frame.copy()
    masked[fgmask==0] = 0
    cv.imshow('object', masked)
    
    #virtbg = next(virt)
    virtbg = virtbgf.copy()
    virtbg[fgmask!=0] = frame[fgmask!=0]
    cv.imshow('virt',virtbg)

cv.destroyAllWindows()

