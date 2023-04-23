#!/usr/bin/env python

# rectificamos solo la distorsión radial, dejando la misma K

import numpy as np
import cv2   as cv
from umucv.stream import autoStream

calibdata = np.loadtxt("calib.txt")

K = calibdata[:9].reshape(3,3)
D = calibdata[9:]

orig = False

for n, (key,img) in enumerate(autoStream()):
    if n==0:
        h,w = img.shape[:2]
        map1, map2 = cv.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv.CV_16SC2)            

    if key==ord('u'): orig = not orig

    if not orig:
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    cv.imshow("clean",img)

