#!/usr/bin/env python

# Captura asíncrona

# Utilizamos un hilo para realizar los cálculos

# (Parece que dentro de un hilo solo se puede hacer computación pura,
#  cv.imshow y otra operaciones dan problemas)

import cv2   as cv
from threading import Thread
from umucv.stream import autoStream
import numpy as np
import time


def heavywork(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return img, r


frame      = None
result     = None
goon       = True

def in_thread():
    global result
    while frame is None:
        time.sleep(0.01)
    while goon:
        print('work starts')
        result = heavywork(frame, 20)
        print('work ends')
        # cv.imshow('otherwork',result)  # Aquí NOOO

t = Thread(target=in_thread, args=())
t.start()

#cv.setNumThreads(8)

# capture
for key, frame in autoStream():
    cv.imshow('input',frame)
    print('capture')
    if result is not None:
        cv.imshow('work',np.hstack(result))
        print('display')
        result = None

goon=False

