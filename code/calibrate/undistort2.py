#!/usr/bin/env python

# rectificamos con otra matriz de calibración y resolución

import numpy as np
import cv2   as cv
from umucv.stream import autoStream

def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])

newres = (320,240)
K2 = Kfov(newres,90)

calibdata = np.loadtxt("calib.txt")

K = calibdata[:9].reshape(3,3)
D = calibdata[9:]

orig = False

for n, (key,img) in enumerate(autoStream()):
    if n==0:
        h,w = img.shape[:2]
        map1, map2 = cv.initUndistortRectifyMap(K, D, np.eye(3), K2, newres, cv.CV_16SC2)            

    if key==ord('u'): orig = not orig
    if not orig:
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    cv.imshow("clean",img)

