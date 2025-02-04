#!/usr/bin/env python

# Ejemplo de reproyección de histograma

# Seleccionando un roi y pulsando c se captura el modelo de
# color de la región en forma de histograma y se muestra
# la verosimilitud de cada pixel de la imagen en ese modelo.

import numpy as np
import cv2 as cv

from umucv.util import ROI
from umucv.stream import autoStream

cv.namedWindow("input")
roi = ROI("input")

def hist(x, redu=16):
    return cv.calcHist([x],
                       [0,1,2], # canales a considerar
                       None,    # posible máscara
                       [redu , redu, redu],  # número de cajas en cada canal
                       [0,256] + [0,256] + [0,256])  # intervalo a considerar en cada canal

H = None

for key, frame in autoStream():
    
    if H is not None:
        b,g,r = np.floor_divide(frame, 16).transpose(2,0,1)
        L = H[b,g,r]  # indexa el array H simultáneamente en todos los
                      # pixels de la imagen.       
        cv.imshow("likelihood", L/L.max())

    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
            H = hist( trozo )
            print(H.shape)    
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    cv.imshow('input',frame)



