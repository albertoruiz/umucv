#!/usr/bin/env python

# Calculamos la posición de las esquinas en sucesivos frames
# pero no calculándolas de nuevo, sino estimando la posición
# a la que se mueve cada una mediante cv.calcOpticalFlowPyrLK

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
import time


corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # resetear el punto inicial del tracking
    if n==0 or key==ord('c'):
        corners = cv.goodFeaturesToTrack(gray, **corners_params).reshape(-1,2)    
        nextPts = corners
        prevgray = gray
    
    
    t0 = time.time()
    
    # encontramos la posición siguiente a partir de la anterior
    nextPts , status, err = cv.calcOpticalFlowPyrLK(prevgray, gray, nextPts, None, **lk_params)
    prevgray = gray
    
    t1 = time.time()
    
    # Unimos la primera y la última posición de cada trayectoria
    for (x0,y0), (x,y), ok in zip(corners, nextPts, status):
        if ok:
            cv.circle(frame, (int(x), int(y)) , radius=3 , color=(0,0,255), thickness=-1, lineType=cv.LINE_AA )
            cv.line(frame, (int(x0), int(y0)), (int(x), int(y)), color=(0,0,255), thickness=1, lineType=cv.LINE_AA )
    
    putText(frame, f'{len(corners)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)

