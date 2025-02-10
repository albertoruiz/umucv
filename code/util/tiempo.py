#!/usr/bin/env python

import time

import cv2            as cv
from umucv.stream import autoStream

t0 = time.time()
for key,frame in autoStream():
    t1 = time.time()
    cv.imshow('input',frame)
    print(f"dt={(t1-t0)*1000:.0f}ms")
    t0 = t1

