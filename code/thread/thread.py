#!/usr/bin/env python

# Captura asíncrona

# Utilizamos un hilo para realizar los cálculos

# (Parece que dentro de un hilo solo se puede hacer computación pura,
#  cv.imshow y otra operaciones dan problemas)

import cv2   as cv
from threading import Thread
from umucv.stream import autoStream

def heavywork(img, n):
    r = img
    for _ in range(n):
        r = cv.medianBlur(r, 17)
    return r


frame = None
result = None

goon = True

def in_thread():
    global result
    while goon:
        if frame is not None:
            print('work starts')
            result = heavywork(frame,20)
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
        cv.imshow('work',result)
        print('display')
        result = None

goon=False

