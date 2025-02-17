#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import Slider

A = Slider('alpha', 'mask', 0.05, 0, 1, 0.01)


import skimage.io as io
path = "https://raw.githubusercontent.com/albertoruiz/umucv/master/images/"
background = io.imread(path+"palmeras.jpg")
background = cv.cvtColor(background, cv.COLOR_RGB2BGR)

bgsub = cv.createBackgroundSubtractorMOG2(500, varThreshold=16, detectShadows=False)

update = True
size = None
auto = True

for key,frame in autoStream():

    if size is None:
        size = (H,W,_) = frame.shape
        background = cv.resize(background, (W,H))

    if key == ord('a'): auto = not auto

    if key == ord('c'):
        update = not update
        if not update: cv.imshow('background',bgsub.getBackgroundImage())

    alpha = -1 if auto else A.value
    fgmask = bgsub.apply(frame, learningRate = alpha if update else 0)
    
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv.erode(fgmask,kernel,iterations=1)
    fgmask = cv.medianBlur(fgmask,3)

    if update: cv.circle(frame,(15,15),6,(0,0,255),-1)
    if auto:   cv.circle(frame,(30,15),6,(0,255,0),-1)
    cv.imshow('original',frame)
    cv.imshow('mask', fgmask)

    fgmask = np.expand_dims(fgmask,2)
    final = np.where(fgmask, frame, background)
    cv.imshow('virt',final)

