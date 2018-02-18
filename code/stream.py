#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=0 --size=800x600
# > ./stream.py --dev=file:../images/rot4.mjpg
# > ./stream.py --dev=http://155.54.X.Y:8080/video
# > ./stream.py --dev=glob:../images/ccorr/scenes/*.png --pause
# > ./stream.py --dev=picam

# > ffmpeg -i video.avi -c:v mjpeg -q:v 3 -huffman optimal -an video.mjpg

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    cv.imshow('input',frame)

cv.destroyAllWindows()

