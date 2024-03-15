#!/usr/bin/env python

# Detector de corners de OpenCV

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.1, minDistance=10).reshape(-1,2)
    
    for x,y in corners:
        cv.circle(frame,(int(x),int(y)), 3, (0,0,255), -1, cv.LINE_AA)
    cv.imshow('input',frame)


