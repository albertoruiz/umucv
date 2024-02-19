#!/usr/bin/env python


import cv2          as cv
from umucv.stream import autoStream
from umucv.util import putText
import time

cv.namedWindow("smooth")
SIGMA = [3]
cv.createTrackbar('sigma', 'smooth', SIGMA[0]*10, 200, lambda v: SIGMA.insert(0,v/10) ) 

cv.namedWindow("media")
RAD = [40]
cv.createTrackbar('radius', 'media', RAD[0], 200, lambda v: RAD.insert(0,v) ) 


for key,frame in autoStream():
    # pasamos a monocromo y float entre 0 y 1
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(float)/255
    #frame = frame.astype(float)/255

    cv.imshow('input',frame)


    t1 = time.time()   
    smooth = cv.GaussianBlur(frame, (0,0), SIGMA[0]) if SIGMA[0] > 0 else frame.copy()
    t2 = time.time()
    
    putText(smooth,f"sigma={SIGMA[0]:.1f}")
    putText(smooth, f'{1000*(t2-t1):5.1f} ms',orig=(5,35))
    cv.imshow('smooth',smooth)

    t1 = time.time()   
    media = cv.boxFilter(frame, -1, (RAD[0],RAD[0])) if RAD[0] > 0 else frame.copy()
    t2 = time.time()

    putText(media, f"radius={RAD[0]:.1f}")
    putText(media, f'{1000*(t2-t1):5.1f} ms',orig=(5,35))
    cv.imshow('media',media)

