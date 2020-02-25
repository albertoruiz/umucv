#!/usr/bin/env python

import cv2   as cv
from threading import Thread

def work(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return r


frame = None

goon = True

def capture():
    global frame
    cap = cv.VideoCapture(0)
    while goon:
        _,frame = cap.read()
        print('cap')

t = Thread(target=capture, args=())
t.start()

while frame is None: pass

while(cv.waitKey(1) & 0xFF != 27):
    cv.imshow('webcam',work(frame,20))
    print('work')

goon=False

