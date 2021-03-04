#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import zoomWindow

zoom = zoomWindow('zoom', W=600, H=600, zink=ord('+'), zoutk=ord('-'))

for key,frame in autoStream():
    zoom.update(key,frame)

cv.destroyAllWindows()

