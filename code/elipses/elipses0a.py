#!/usr/bin/env python

# Detección de elipses (método frecuencial)
# pruébalo con:

# ./elipses0a.py --dev=dir:*.png


# paquetes habituales
from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.util import mkParam


# Extraemos los contornos igual que en el capítulo de reconocimiento de formas
def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r

def extractContours(image, minarea=10):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    contours = [ c.reshape(-1,2) for c in contours if cv.contourArea(c) > minarea ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


from numpy.fft import fft

# una elipse perfecta solo tiene frecuencias 0, 1, -1.
# Cuanto mayores sean los demás coeficientes de Fourier, 
# menos se parecerá el contorno a una elipse
def error_elipse(c):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)
    
    s = fa[1] + fa[-1]
    
    fa[0] = fa[1] = fa[-1] = 0
    p = np.sum(fa)

    # devolvemos la proporción de frecuencias "malas"
    return p / s



# esto es una utilidad para poner cómodamente trackbars
cv.namedWindow("source")
param = mkParam("source")
param.addParam("err",30,50)
param.addParam("area",20,100)


black = True

for key,frame in autoStream():
    cs = extractContours(frame, minarea=param.area)

    cv.polylines(frame, cs, color=(0,255,0), isClosed=True, thickness=1, lineType=cv.LINE_AA)
    
    # seleccionamos los contornos que se aproximan muy bien a una elipse
    els = [ c for c in cs if error_elipse(c) < param.err/100 ]

    cv.polylines(frame, els, color=(0,0,255), isClosed=True, thickness=2, lineType=cv.LINE_AA)

    cv.imshow('source',frame)

