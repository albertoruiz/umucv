#!/usr/bin/env python

from dinoutil import resize_for_dino, DINOv2_from_image
from umucv.stream import autoStream
from umucv.util import putText
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
        cl, dinos = DINOv2_from_image(cv.cvtColor(used_input, cv.COLOR_BGR2RGB))
        result = dinos

t = Thread(target=work,args=())
t.start()


point = [None]
cl = 1

seeds = []
cls   = []

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        point[0]=(x,y)

cv.namedWindow("masked")
cv.setMouseCallback("masked", manejador)

cv.namedWindow("source")
cv.setMouseCallback("source", manejador)


for key, frame in autoStream():
    if key==ord('+'):
        cl*=-1

    img_input = resize_for_dino(frame).copy()
    putText(frame,"POS" if cl == 1 else "NEG")
    cv.imshow("source",frame)

    if result is not None:

        if point[0] is not None:
            rx = point[0][0]//14
            ry = point[0][1]//14
            point[0]=None
            dino = result[ry,rx]
            seeds.append(dino)
            cls.append(cl)
            print(cls)

        cv.imshow("masked",used_input)
        result = None

goon = False
