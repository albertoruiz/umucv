#!/usr/bin/env python

import cv2          as cv
from umucv.stream import mkStream, withKey

stream = mkStream( dev = 'glob:../images/ccorr/scenes/*.png' )

for _, frame in withKey(stream,0):
    cv.imshow('input',frame)

cv.destroyAllWindows()

