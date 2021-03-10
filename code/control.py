#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import mkStream
from os import system


def focus(v):
    cmd = f'v4l2-ctl -d /dev/video{DEV} -c focus_absolute={v}'
    system(cmd)

def exposure(v):
    cmd = f'v4l2-ctl -d /dev/video{DEV} -c exposure_absolute={v}'
    system(cmd)

def dev(v):
    global DEV
    DEV = v
    system(f'v4l2-ctl -d /dev/video{DEV} -c focus_auto=0')
    system(f'v4l2-ctl -d /dev/video{DEV} -c exposure_auto=1')


dev(0)

cv.namedWindow("control")
cv.createTrackbar("camera", "control", 0, 4, dev)
cv.createTrackbar("focus", "control", 0, 255, focus)
cv.createTrackbar("exposure", "control", 250, 2047, exposure)


while True:
    if cv.waitKey(100) & 0xFF == 27:
        break

cv.destroyAllWindows()

