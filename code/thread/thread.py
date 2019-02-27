#!/usr/bin/env python

import cv2   as cv
from threading import Thread
import time
from umucv.util import putText
from umucv.stream import autoStream


def work(img):
    r = img
    for _ in range(10):
      r = cv.medianBlur(r,17)
    return r


frame = None
goon = True

def fun():
    global frame, goon, key
    for key,frame in autoStream():
        cv.imshow('cam',frame)
    goon = False

t = Thread(target=fun,args=())
t.start()

while frame is None: pass

while goon:    
    
    t0 = time.time()

    result = work(frame)
    
    t1 = time.time()
    putText(result, '{:.0f}ms'.format(1000*(t1-t0)))

    cv.imshow('work', result)

