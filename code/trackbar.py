#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream


cv.namedWindow("binary")
cv.createTrackbar("umbral", "binary", 128, 255, lambda _:())

for key, frame in autoStream():
    cv.imshow("original", frame)
    #print(frame)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # otras posibilidades de convertir en monocromo
    #gray = frame[:,:,1]  # coger el canal verde y punto
    gray = (frame @ [0.5,0.5,0.2]).astype(np.uint8) # contracción de numpy
    cv.imshow("gray",gray)
    #print(gray)
    
    # aprovechamos para ver ejemplos de indexado con numpy
    #recorte = gray[50:160:2, 100:200]
    #cv.imshow("recorte",recorte)
    #flipped = gray[::-1, :] 
    #cv.imshow("flipado",flipped)
    
    h = cv.getTrackbarPos('umbral','binary')
    logica = gray > h
    #print(logica)
    # para mostrar un array de bool en opencv hay que convertir a
    # byte con valores entre 0-255
    #binary = logica.astype(np.uint8)*128
    # o a float con valores entre 0 y 1
    binary = logica.astype(float)
    #print(binary)
    cv.imshow('binary', binary )

cv.destroyAllWindows()

