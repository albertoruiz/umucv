#!/usr/bin/env python

import numpy as np
import cv2 as cv
import sys

filename = sys.argv[1]

img = cv.imread(filename)

print(filename)

status = [False]
points = []
def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        status[0] = True
    if event == cv.EVENT_LBUTTONUP:
        status[0] = False
    if status[0] and event == 0:
        points.append((x,y))
    #print(event)

cv.namedWindow("mask")
cv.setMouseCallback("mask", manejador)

while True:
    key = cv.waitKey(100) & 0xFF
    if key == 27: break
    
    if key==ord('x') and points:
        del points[-1]
    
    cosa = img.copy()
    
    for (x,y) in points:
        cv.circle(cosa, (x,y), 10, color=(0,0,255), thickness=-1)
    
    cv.imshow("mask",cosa)

result = np.zeros_like(img)
for (x,y) in points:
    cv.circle(result, (x,y), 10, color=(255,255,255), thickness=-1)

cv.imwrite("mask_"+filename, result)

