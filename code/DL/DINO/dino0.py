#!/usr/bin/env python

from dinoutil import resize_for_dino, DINOv2_from_image
from umucv.stream import autoStream
import cv2 as cv
import time

for key, frame in autoStream():
    img = resize_for_dino(frame)
    t0 = time.time()
    cl, dinos = DINOv2_from_image(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    t1 = time.time()
    print(frame.shape, img.shape, dinos.shape, F"{(t1-t0)*1000:.0f}ms")
    cv.imshow("dino",img)

