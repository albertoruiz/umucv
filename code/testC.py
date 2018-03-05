#!/usr/bin/env python

import numpy             as np
import cv2               as cv
import matplotlib.pyplot as plt
import time

from cfuns import nms

def readfloat(file):
    return cv.cvtColor( cv.imread("../images/"+file), cv.COLOR_BGR2GRAY).astype(float)

def grad(x):
    gx =  cv.Sobel(x,-1,1,0)/8
    gy =  cv.Sobel(x,-1,0,1)/8
    return gx,gy

x   = readfloat('cube3.png')
print(x.shape)

gx,gy = grad(cv.GaussianBlur(x,(0,0),5))

gm = np.sqrt(gx**2+gy**2)
ga = np.arctan2(gy,gx)
gad = (np.round(ga / np.pi * 4) % 4).astype(np.uint8)

t0 = time.time()
cnms = nms(gm,gad)
t1 = time.time()
print('{:.0f}ms'.format(1000*(t1-t0)))

plt.imshow(cnms, 'gray', interpolation='bicubic');

plt.show()

