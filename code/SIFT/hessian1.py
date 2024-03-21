#!/usr/bin/env python

import cv2 as cv
import numpy as np
import scipy.ndimage
from umucv.stream import autoStream

def hessian(x):
    gxx = cv.Sobel(x,-1,2,0)
    gyy = cv.Sobel(x,-1,0,2)
    gxy = cv.Sobel(x,-1,1,1)
    return gxx*gyy-gxy**2

N = 3+2
BASE = 2
SIGN = 1

cv.namedWindow("original")
cv.createTrackbar("sigma","original",3,N-1, lambda _: ())
cv.createTrackbar("thres","original",10,100, lambda _: ())
cv.createTrackbar("scale","original",10,100, lambda _: ())

for key, frame in autoStream():
    if key==ord("+"):
        SIGN *= -1

    sc = cv.getTrackbarPos('sigma','original')
    thres = cv.getTrackbarPos('thres','original') / 100
    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(float)/255
    
    hs = []
    
    scales = [BASE**t for t in range(N)]

    for sigma in scales:
        s = cv.GaussianBlur(g,(-1,-1),sigma)
        h = sigma**4/16 * hessian(s)
        hs.append(h)

    hs = np.array(hs) * SIGN

    ks = scipy.ndimage.maximum_filter(hs,3)

    S,R,C = np.where( (ks==hs) & (hs > thres*hs.max()) )
    
    for s,r,c in zip(S,R,C):
        cv.circle(frame, (c,r), int(np.sqrt(2)*scales[s]), color=(0,0,255));

    scale = cv.getTrackbarPos('scale','original') *10
    cv.imshow("hessian", 0.5+ hs[sc]*scale)
    
    cv.imshow("original",frame)

