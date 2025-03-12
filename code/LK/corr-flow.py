#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

prev = None
DELTA=30

for key, frame in autoStream():

    if prev is not None:
       i0 = prev[DELTA:-DELTA, DELTA:-DELTA]
       i1 = frame
       
       cc = cv.matchTemplate(i1, i0, cv.TM_CCORR_NORMED)
       min_val, max_val, _, max_loc = cv.minMaxLoc(cc)
       dx = max_loc[0] - DELTA
       dy = max_loc[1] - DELTA 

       cv.imshow('correlation',(cc-min_val)/(max_val-min_val))

       H,W,_ = frame.shape
       cv.arrowedLine(frame,(W//2,H//2),(W//2-dx*10,H//2-dy*10), color=(0,255,255), thickness=2)
       
       putText(frame,f"{dx} {dy}")

    prev = frame
    cv.imshow('input',frame)

