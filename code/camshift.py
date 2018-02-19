#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI

REDU = 16
def rgbh(xs):
    def normhist(x): return x / np.sum(x)

    def h(rgb):
        return cv.calcHist([rgb]
                           , [0, 1, 2]
                           , None
                           , [256//REDU, 256//REDU, 256//REDU]
                           , [0, 256] + [0, 256] + [0, 256]
                           )

    return normhist(sum(map(h, xs)))

def smooth(s,x):
    return cv.GaussianBlur(x,(0,0),s)

his = None

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )


cv.namedWindow("input")
roi = ROI('input')


for key, x in autoStream():

    if his is not None:
        r,g,b = np.floor_divide( x , REDU).transpose(2,0,1)
        l = his[r,g,b]
        maxl = l.max()

        cv.imshow("likelihood", np.clip((1*l/maxl*255),0,255).astype(np.uint8))
        ret, track_window = cv.CamShift(l, track_window, term_crit)

        pts = cv.boxPoints(ret).astype(int)
        cv.ellipse(x,ret,(0,128,255),2)
        cv.polylines(x,[pts],True, (0,128,255),2)

    if roi.roi:
        his = None
        [x1,y1,x2,y2] = roi.roi
        
        if key == ord('t'):
            his = smooth(1,rgbh([x[y1:y2,x1:x2]]))
            r,h,c,w = y1,y2-y1,x1,x2-x1
            track_window = (c,r,w,h)
            roi.roi = []

        roipoly = [np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])]
        cv.polylines(x,roipoly, isClosed=True, color=(0,255,255), thickness=2)

    cv.imshow('input',x)

