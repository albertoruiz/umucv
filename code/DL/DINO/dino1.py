#!/usr/bin/env python

from dinoutil import resize_for_dino, DINOv2_from_image
from umucv.stream import autoStream
import cv2 as cv
import time
from threading import Thread

goon      = True
img_input = None
result    = None

def work():
    global result
    global used_input
    while img_input is None:
        time.sleep(0.010)
    while goon:
        used_input = img_input
        print("work...")
        cl, dinos = DINOv2_from_image(cv.cvtColor(used_input, cv.COLOR_BGR2RGB))
        print("ok")
        result = dinos

t = Thread(target=work,args=())
t.start()

for key, frame in autoStream():
    img_input = resize_for_dino(frame)
    cv.imshow("source",img_input)

    if result is not None:
        cv.imshow("dino",used_input)
        print(result.shape)
        result = None

goon = False
