#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI
import scipy.ndimage.filters as fil

def normhist(x):
    return x / np.sum(x)

def smooth(s,x):
    return cv.GaussianBlur(x,(0,0),s)

def hist(xs, redu):
    return cv.calcHist(xs
                       , [0, 1, 2]
                       , None
                       , [256//redu, 256//redu, 256//redu]
                       , [0, 256] + [0, 256] + [0, 256]
                       )

class Model:
    def __init__(self, xs, levels=16):
        self.redu = 256//levels
        self.H = fil.gaussian_filter(hist(xs,self.redu),1)
        print(self.H.shape)
    
    def __call__(self,img):
        r,g,b = np.floor_divide(x , self.redu).transpose(2,0,1)
        return self.H[r,g,b]


cv.namedWindow("input")
roi = ROI('input')

model = None

for key, x in autoStream():

    if model is not None:
        l = model(x)
        
        maxl = l.max()
        cv.imshow("likelihood", np.clip((1*l/maxl*255),0,255).astype(np.uint8))

        term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
        ret, track_window = cv.CamShift(l, track_window, term_crit)
        
        cv.ellipse(x,ret,(0,128,255),2)
        #pts = cv.boxPoints(ret).astype(int)
        #cv.polylines(x,[pts],True, (0,128,255),2)

    if roi.roi:
        model = None
        [x1,y1,x2,y2] = roi.roi
        
        if key == ord('t'):
            model = Model([x[y1:y2+1,x1:x2+1]])
            r,h,c,w = y1,y2-y1+1,x1,x2-x1+1
            track_window = (c,r,w,h)
            roi.roi = []

        cv.rectangle(x, (x1,y1), (x2,y2) , color=(0,255,255), thickness=2)

    cv.imshow('input',x)

cv.destroyAllWindows()

