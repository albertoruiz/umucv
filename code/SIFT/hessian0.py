#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream

def hessian(x):
    gxx = cv.Sobel(x,-1,2,0)
    gyy = cv.Sobel(x,-1,0,2)
    gxy = cv.Sobel(x,-1,1,1)
    return gxx*gyy-gxy**2

cv.namedWindow("original")
cv.createTrackbar("sigma","original",40,100, lambda _: ())
cv.createTrackbar("scale","original",500,1000, lambda _: ())

SHOW = 0

for key, frame in autoStream():
    if key == ord('+'):
        SHOW = (SHOW+1)%3

    sigma = 1.05**cv.getTrackbarPos('sigma','original')
    scale = cv.getTrackbarPos('scale','original') / 10
    
    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(float)/255
    s = cv.GaussianBlur(g,(-1,-1),sigma) if sigma>0 else g
    
    h = (sigma**4/16) * hessian(s)
    
    cv.imshow("debug",[lambda:0.5+scale*h, lambda:scale*abs(h), lambda:s][SHOW]()) 
    cv.imshow("original",frame)

