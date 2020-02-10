#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=0 --size=800x600  --resize=200x0
# > ./stream.py --dev=file:path/to/video  [--loop]
# > ./stream.py --dev=http://155.54.X.Y:8080/video
# > ./stream.py --dev=glob:../images/ccorr/scenes/*.png [--step]
# > ./stream.py --dev=dir:../images/ccorr/scenes/*.png
# > ./stream.py --dev=picam

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    cv.imshow('input',frame)

cv.destroyAllWindows()

