#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream


cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")

model = None


for key, frame in autoStream():
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY).astype(np.float32)/255
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            model = gray[y1:y2+1, x1:x2+1]
            cv.imshow("model", model)
            region.roi = []

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    if model is not None:
        cc = cv.matchTemplate(gray, model, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cc)
        #mr,mc = divmod(cc.argmax(),cc.shape[1])
        #cv.imshow('CC',cc)
        putText(cc,f'max correlation {max_val:.2f}')
        cv.imshow('CC',(cc-min_val)/(max_val-min_val))
        x1,y1 = max_loc
        h,w = model.shape[:2]
        x2 = x1+w; y2 = y1+h
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

   

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

