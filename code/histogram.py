#!/usr/bin/env python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)

while(True):
    if cv.waitKey(1) & 0xFF == 27: break

    ret, frame = cap.read()
    
    x = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    h,b = np.histogram(x, np.arange(257))
    
    # ajustamos la escala del histograma para que se vea bien en la imagen
    kk = np.array([2*b[1:], 480-h*(480/10000)]).T.astype(int)
    cv.polylines(x, [kk], isClosed=False, color=(0,0,255), thickness=2)
    cv.imshow('histogram',x)

cv.destroyAllWindows()

