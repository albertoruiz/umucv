#!/usr/bin/env python

import cv2   as cv
from umucv.stream import Camera

#cam = Camera((640,480),'0',debug=True)
cam = Camera()

t = 0

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    
    if cam.time == t: continue
    t = cam.time
    
    cv.imshow('webcam',cv.medianBlur(cam.frame,5))
    #print('WORK {:.0f}'.format(cam.clock.time()))

cam.stop()
cv.destroyAllWindows()

