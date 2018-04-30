#!/usr/bin/env python

import numpy as np
import cv2   as cv

K= np.array([[ 535.79688857,    0.,          308.50044065],
             [   0.,          536.56590112,  220.02740752],
             [   0.,            0.,            1.        ]])
d = np.array([ 0.0737991,  -0.28470474,  0.00037098, -0.00081263,  0.20919068])

img = cv.imread("mylogitech/20150309-091713.png")
h, w = img.shape[:2]

# undistort
newcamera, roi = cv.getOptimalNewCameraMatrix(K, d, (w,h), 0) 
newimg = cv.undistort(img, K, d, None, newcamera)
newimg2 = cv.undistort(img, K, d, None)
newimg3 = cv.undistort(img, K, d, None, np.array([[320*1.7,0,320],[0,320*1.7,240],[0,0,1]]))

print(newcamera)
print(roi)

if True:
    cv.imwrite("original.jpg", img)
    cv.imwrite("undistorted.jpg", newimg)
    cv.imwrite("undistorted2.jpg", newimg2)
    cv.imwrite("undistorted3.jpg", newimg3)


