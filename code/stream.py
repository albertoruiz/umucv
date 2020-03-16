#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    cv.imshow('input',frame)

cv.destroyAllWindows()

