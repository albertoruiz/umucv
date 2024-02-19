#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import putText

cv.namedWindow("media")
RAD = [22]
cv.createTrackbar('radius', 'media', RAD[0], 200, lambda v: RAD.insert(0,v) ) 

cv.namedWindow("premask")
RAD2 = [40]
cv.createTrackbar('radius', 'premask', RAD2[0], 200, lambda v: RAD2.insert(0,v) )

cv.namedWindow("mask")
H = [0.06]
cv.createTrackbar('umbral', 'mask', int(H[0]*1000), 500, lambda v: H.insert(0,v/1000) )


for key,frame in autoStream():
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(float)/255

    cv.imshow('input',frame)

    media = cv.boxFilter(frame, -1, (RAD[0],RAD[0])) if RAD[0] > 0 else frame.copy()

    dif = cv.absdiff(frame, media)
    cv.imshow('dif',2*dif)
    
    putText(media, f"radius={RAD[0]:.1f}")
    cv.imshow('media',media)

    premask = cv.boxFilter(dif, -1, (RAD2[0],RAD2[0])) if RAD2[0] > 0 else dif.copy()
    cv.imshow('premask', 4*premask)

    mask = premask < H[0]
    cv.imshow('mask',mask.astype(float))
    
    masked = frame.copy()
    masked[mask] = 0
    cv.imshow('masked', masked)

