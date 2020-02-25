#!/usr/bin/env python

import cv2   as cv
from umucv.stream import autoStream


def work(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return r


for key,frame in autoStream():
    cv.imshow('cam',frame)
    result = work(frame, 20)
    cv.imshow('work', result)

