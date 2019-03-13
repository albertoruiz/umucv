#!/usr/bin/env python

import cv2 as cv

cap = cv.VideoCapture(0)
assert cap.isOpened(), "conecta la webcam"
w   = cap.get(cv.CAP_PROP_FRAME_WIDTH)
h   = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)

print(f'{w:.0f}x{h:.0f} {fps}fps')

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    
    ok, frame = cap.read()
    cv.imshow('webcam',frame)

cv.destroyAllWindows()

