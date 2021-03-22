#!/usr/bin/env python

import numpy             as np
import cv2               as cv
from umucv.stream import Camera

from umucv.util   import ROI, putText, Help

from skimage.feature import hog


help = Help(
"""
HOG object recognition

c: capture roi
g: show HOG
x: clear model

h: show/hide help
""")


cv.namedWindow('cam')
roi = ROI('cam')

PPC = 16
CPB = 2

SHOW = False

cam = Camera()

def feat(region):
    r = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    return hog( r, visualize = SHOW, transform_sqrt=True, feature_vector=False,
                   orientations = 8,
                   pixels_per_cell = (PPC,PPC),
                   cells_per_block = (CPB,CPB),
                   block_norm = 'L1' )

def dist(u,v):
    return sum(abs(u-v))/len(u)

MODEL = None

while True:
    key = cv.waitKey(1) & 0xFF
    help.show_if(key, ord('h'))
    
    if key == 27: break
    
    if key == ord('x'):
        MODEL = None

    if key == ord('g'):
        SHOW = not SHOW
 
    x = cam.frame.copy()

    if SHOW:
        h, sh = feat(x)
        putText(sh,str(h.shape))
        cv.imshow('hog', sh)
    else:
        h = feat(x)


    if MODEL is not None:
        h1,w1 = h.shape[:2]
        h2,w2 = MODEL.shape[:2]
        detected = []
        for j in range(h1-h2):
            for k in range(w1-w2):
                vr = h[j:j+h2 , k: k+w2].flatten()
                v  = MODEL.flatten()
                detected.append( (dist(vr,v), j, k) ) 
                

        dmin, jmin, kmin = min(detected)


        x1 = kmin*PPC
        y1 = jmin*PPC
        x2 = x1+(w2+CPB-1)*PPC
        y2 = y1+(h2+CPB-1)*PPC
        
        if dmin < 0.04:
            cv.rectangle(x, (x1,y1), (x2,y2), color=(255,255,0), thickness=2)
        putText(x,'{:.3f}'.format(dmin),(6,18),(0,128,255))

    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        reg = x[y1:y2, x1:x2].copy()
        
        if key == ord('c'):
            if SHOW:
                MODEL, sh = feat(reg)
                putText(sh,str(MODEL.shape))
                cv.imshow('model', sh)
            else:
                MODEL = feat(reg)
            roi.roi = []

        cv.rectangle(x, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    cv.imshow('cam', x)

cam.stop()

