#!/usr/bin/env python

import cv2          as cv
import numpy        as np

from umucv.stream import autoStream
from umucv.htrans import homog

samples = []
targets = []

SZ = 40
DX = DY = np.arange(-20,20+2,2)

def mkSamples(img,x,y):
    return zip(*[(img[y-dy-SZ:y-dy+SZ,x-dx-SZ:x-dx+SZ], np.array([dx,dy])) for dx in DX for dy in DY ])

def solve(imgs, desp):
    X = np.array([ x.flatten() for x in imgs ])
    print(X.shape)
    Y = np.vstack(desp)
    print(Y.shape)
    sol = np.linalg.lstsq(homog(X),Y)
    W = sol[0]
    print(W.shape)
    print(sol[1])
    return W

POS = ()

def fun(event, x, y, flags, param):
    global samples, targets, weights, POS
    if event == cv.EVENT_LBUTTONDOWN:
        X,Y = mkSamples(frame,x,y)
        #print(X[0].shape)
        samples += X
        targets += Y
        weights = solve(samples,targets)
        POS = (x,y)
    if event == cv.EVENT_RBUTTONDOWN:
        POS = (x,y)


cv.namedWindow("input")
cv.setMouseCallback("input", fun)

for key,frame in autoStream():
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if key == ord('x'):
        POS = []
        samples = []
        targets = []
                
    if POS:
        x,y = POS
        dx = 0
        dy = 0
        reg = frame[y-dy-SZ:y-dy+SZ,x-dx-SZ:x-dx+SZ]
        try:
            xp,yp = homog(reg.flatten()) @ weights
            x = int(x+xp)
            y = int(y+yp)
            POS = x,y
            cv.circle(frame,(x,y), 3, (0,0,255))
            cv.circle(frame,(x,y), SZ, (0,0,255))
        except:
            pass

    cv.imshow('input',frame)

