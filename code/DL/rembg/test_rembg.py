#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
from rembg import remove

for key,frame in autoStream():
    cv.imshow('input',frame)
    if key == ord('r'):
        fx = 240/frame.shape[1]
        small = cv.resize(frame, (0,0), fx=fx, fy=fx)
        mask = remove(small, only_mask=False)
        cv.imshow("mask",mask)

