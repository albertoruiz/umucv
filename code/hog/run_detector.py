#!/usr/bin/env python

# adaptado de
# http://dlib.net/train_object_detector.py.html


import sys
import dlib
import cv2 as cv

from umucv.stream import autoStream

detector = dlib.simple_object_detector(sys.argv[1])
# We can look at the HOG filter we learned.
win_det = dlib.image_window()
win_det.set_image(detector)


for key, img in autoStream():
    dets = detector(img)
    for k, d in enumerate(dets):
        cv.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (128,128,255), 3 )
    cv.imshow("object detection", img)

