#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText, Slider

def hessian(x):
    gxx = cv.Sobel(x,-1,2,0)
    gyy = cv.Sobel(x,-1,0,2)
    gxy = cv.Sobel(x,-1,1,1)
    return gxx*gyy-gxy**2

s_sigma = Slider("sigma","debug",40,0,100,1)
s_scale = Slider("scale","debug",50,0,100,1)

SHOW = 0

for key, frame in autoStream():
    if key == ord('+'):
        SHOW = (SHOW+1)%3

    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(float)/255

    sigma = 1.05**s_sigma.value
    s = cv.GaussianBlur(g,(0,0),sigma)
    
    h = (sigma**4/16) * hessian(s)
    
    scale = s_scale.value
    cv.imshow("debug",[lambda:0.5+scale*h, lambda:scale*abs(h), lambda:s][SHOW]()) 
    putText(frame,f"sigma = {sigma:.2f}")
    cv.imshow("original",frame)

