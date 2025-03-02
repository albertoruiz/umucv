#!/usr/bin/env python

import cv2
from umucv.stream import autoStream
from umucv.util import Slider, putText

sigma          = Slider("sigma", "canny", 2, 0.1, 5, 0.1)
low_threshold  = Slider("low threshold", "canny", 40, 0, 255)
high_threshold = Slider("high threshold", "canny", 100, 0, 255)

for key,frame in autoStream():
    cv2.imshow('input',frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(0,0),sigma.value)
    frame = cv2.Canny(frame, low_threshold.value, high_threshold.value, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    info = f"sigma={sigma.value:.1f} low={low_threshold.value} high={high_threshold.value}"
    putText(frame,info)
    cv2.imshow('canny',frame)

