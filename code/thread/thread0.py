#!/usr/bin/env python

import cv2   as cv
from threading import Thread

frame = None

goon = True

def fun():
    global frame
    cap = cv.VideoCapture(0)
    while goon:
        _,frame = cap.read()
        print('+')

t = Thread(target=fun,args=())
t.start()

while frame is None: pass

while(cv.waitKey(1) & 0xFF != 27):
    cv.imshow('webcam',cv.medianBlur(frame,5))
    print('.')

goon=False

