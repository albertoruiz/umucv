#!/usr/bin/env python

# Una computación muy costosa hace que la operación de captura
# deje de ir en tiempo real y la aplicación va a saltos

import cv2   as cv
from umucv.stream import autoStream


def heavywork(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return r


for key,frame in autoStream():
    cv.imshow('cam',frame)
    result = heavywork(frame, 20)
    cv.imshow('work', result)

