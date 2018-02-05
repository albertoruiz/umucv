#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=0 --size=800x600
# > ./stream.py --dev=file:../images/rot4.mjpg
# > ./stream.py --dev=https://155.54.x.x/video
# > ./stream.py --dev=glob:../images/ccorr/scenes/*.png
# > ./stream.py --dev=picam

import cv2          as cv
import numpy        as np
import argparse

from umucv.stream import mkStream, withKey
from umucv.util   import sourceArgs

parser = argparse.ArgumentParser()
sourceArgs(parser)
args = parser.parse_args()

stream = mkStream(args.size, args.dev)

for key,frame in withKey(stream):
    cv.imshow('input',frame)

cv.destroyAllWindows()

