#!/usr/bin/env python

# https://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf

import cv2 as cv
from umucv.util import ROI
from umucv.stream import autoStream

cv.namedWindow("input")
region = ROI("input")

ok = False

for key, frame in autoStream():
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            tracker = cv.TrackerKCF_create()
            tracker.init(frame,(x1,y1,x2-x1+1,y2-y1+1))
            ok = True
            region.roi = []
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=1)

    if ok:
        ok, (x1,y1,w,h) = tracker.update(frame)
        cv.rectangle(frame, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(255,255,0), thickness=2)

    cv.imshow('input',frame)

