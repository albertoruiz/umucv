#!/usr/bin/env python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27: break

    ret, frame = cap.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    h,b = np.histogram(gray, np.arange(257))

    # ajustamos la escala del histograma para que se vea bien en la imagen
    # usaremos cv.polylines, que admite una lista de listas de puntos x,y enteros
    # las coordenadas x son los bins del histograma (quitando el primero)
    # y las coordenadas y son el propio histograma escalado y desplazado
    xs = 2*b[1:]
    ys = 480-h*(480/10000)
    # trasponemos el array para emparejar cada x e y
    xys = np.array([xs,ys]).T.astype(int)

    cv.polylines(gray, [xys], isClosed=False, color=0, thickness=2)
    cv.imshow('histogram',gray)

cv.destroyAllWindows()

