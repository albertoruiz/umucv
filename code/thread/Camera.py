#!/usr/bin/env python

import cv2   as cv
from umucv.stream import Camera

def work(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return r

#cam = Camera((640,480),'0',debug=True)
cam = Camera(debug=True)

t = 0

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    
    if cam.time == t: continue
    t = cam.time
    
    cv.imshow('webcam', work(cam.frame, 20) )
    print('WORK {:.0f}'.format(cam.clock.time()))

cam.stop()
cv.destroyAllWindows()

