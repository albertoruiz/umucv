#!/usr/bin/env python

# Este programa tiene 4 objetivos:
# 1) Ilustrar la construcción de una máscara que indica
#    la localización de objetos de un color caractéristico,
#    mediante una sencilla umbralización.
# 2) Comprobar que el espacio de color HSV es útil para esto.

# 3) (En la versión completa
# 4)  inrange.py)

# python inrange0.py --dev=dir:../images/naranjas/*.jpg --resize=0x400
# python inrange0.py --dev=dir:../images/demos/fruits.png

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


# Creamos sliders para elegir interactivamente los umbrales
# de segmentación.

# umbrales iniciales
h1, s1, v1, h2, s2, v2 = 6,71,0,26,255,255

cv.namedWindow("mask")# , cv.WINDOW_NORMAL)

def nothing(x): pass

cv.createTrackbar("h1", "mask", h1, 180, nothing)
cv.createTrackbar("h2", "mask", h2, 180, nothing)
cv.createTrackbar("s1", "mask", s1, 255, nothing)
cv.createTrackbar("s2", "mask", s2, 255, nothing)
cv.createTrackbar("v1", "mask", v1, 255, nothing)
cv.createTrackbar("v2", "mask", v2, 255, nothing)


for key, frame in autoStream():
    cv.imshow('input',frame)

    # Convertimos al espacio de color deseado
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # separamos los canales
    h,s,v = hsv.transpose(2,0,1)
    
    # o también
    # h,s,v = cv.split(hsv)

    # Leemos los sliders
    h1 = cv.getTrackbarPos('h1','mask')
    h2 = cv.getTrackbarPos('h2','mask')
    s1 = cv.getTrackbarPos('s1','mask')
    s2 = cv.getTrackbarPos('s2','mask')
    v1 = cv.getTrackbarPos('v1','mask')
    v2 = cv.getTrackbarPos('v2','mask')
    
    # opencv genera máscaras np.uint8 con el convenio 0 (No) - 255 (Sí)
    # que se ven Negro - Blanco en un cv.imshow.
    mask0  = cv.inRange(hsv, (h1,s1,v1), (h2,s2,v2) ) # cambiar a condición explícita con numpy
    cv.imshow('mask', mask0)

    # otra forma de hacer lo mismo con numpy
    # numpy genera máscaras True-False
    mask = (h1 <= h) & (h <= h2) & (s1 <= s) & (s <= s2) & (v1 <= v) & (v <= v2)
    
    # para verlas con imshow hay que convertirlas a números
    cv.imshow('npmask',mask.astype(np.uint8)*255)
    

