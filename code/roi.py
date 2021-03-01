# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI
from umucv.stream import autoStream

cv.namedWindow("input")
roi = ROI("input")

for key, frame in autoStream():
    
    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    cv.imshow('input',frame)

