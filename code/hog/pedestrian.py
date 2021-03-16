#!/usr/bin/env python

import cv2 as cv
import time
from umucv.util import putText
from umucv.stream import autoStream

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

MULTISCALE = True

for key, image in autoStream():
    
    if key==ord('m'):
        MULTISCALE = not MULTISCALE
    
    t0 = time.time()
    if MULTISCALE:
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)
    else:
        (rects, weights) = hog.detect(image, winStride=(4, 4), padding=(8, 8))
    t1 = time.time()    

    if len(rects) > 0:
            for rect, p in zip(rects,weights.flatten()):
                if MULTISCALE:
                    x,y,w,h = rect
                else:
                    x,y = rect; w,h = 64,128
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                putText(image,'{:.1f}'.format(p),(x+2,y-7),(0,128,255))

    putText(image,'{:.0f} ms'.format((t1-t0)*1000),(7,18),(0,255,128))
    cv.imshow('pedestrian',image)

cv.destroyAllWindows()

