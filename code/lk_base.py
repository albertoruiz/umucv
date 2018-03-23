#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import putText

n = 0
for key,frame in autoStream():
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if n == 0 or key == ord('c'):
        corners = cv.goodFeaturesToTrack(gray, 50, 0.1, 10).reshape(-1,2)
        nextPts = corners
        prevgray = gray
    n += 1
    
    nextPts, status, err = cv.calcOpticalFlowPyrLK(prevgray, gray, nextPts, None)
    prevgray = gray
    
    for (x,y), ok, (x0,y0) in zip(nextPts, status, corners):
        if ok:
            cv.circle(frame,(x0,y0), 2, (0,0,128), -1, cv.LINE_AA)
            cv.circle(frame, (x,y), 3, (0,0,255), -1, cv.LINE_AA)
            cv.line(frame, (x0,y0), (x,y), (0,0,255), 1, cv.LINE_AA)
        else:
            cv.circle(frame,(x0,y0), 3, (128,128,128), -1, cv.LINE_AA)


    putText(frame, '{}'.format(len(corners)), (5,20), color=(0,255,255))
    cv.imshow('input',frame)
    

cv.destroyAllWindows()

