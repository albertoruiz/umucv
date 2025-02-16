#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import putText, Slider

RAD1 = Slider('radius1', 'masked', 22,   1, 200)
RAD2 = Slider('radius2', 'masked', 40,   1, 200)
H    = Slider('umbral',  'masked', 0.06, 0, 0.5, 0.01)

def mean(x,n):
    return cv.boxFilter(x,-1,(n,n))

for key,frame in autoStream():
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(float)/255

    cv.imshow('input',frame)

    media = mean(gray, RAD1.value)
    cv.imshow('media',media)

    dif = cv.absdiff(gray, media)
    cv.imshow('dif',2*dif)

    premask = mean(dif, RAD2.value)
    cv.imshow('premask', 4*premask)

    mask = premask < H.value
    cv.imshow('mask',mask.astype(float))
    
    masked = frame.copy()
    masked[mask] = 0
    putText(masked, f"R1={RAD1.value} R1={RAD2.value} H={H.value:.2f}")
    cv.imshow('masked', masked)

