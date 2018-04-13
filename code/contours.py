#!/usr/bin/env python

# ./contours.py --dev=dir:../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np
from numpy.fft    import fft
from numpy.linalg import norm
from umucv.util   import putText


def readrgb(file):
    return cv.cvtColor( cv.imread('../images/'+file), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def extractContours(g):
    #gt = (g > 128).astype(np.uint8)*255
    (_, gt) = cv.threshold(g,128,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #gt = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,-10)
    (_, contours, _) = cv.findContours(gt.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    return [ c.reshape(-1,2) for c in contours ]

def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)                     # para conseguir invarianza a rotación 
                                    # y punto de partida
    s = fa[1] + fa[-1]              # el tamaño global de la figura para normalizar la escala
    v = np.zeros(2*wmax+1)          # espacio para el resultado
    v[:wmax] = fa[2:wmax+2];        # cogemos las componentes de baja frecuencia, positivas
    v[wmax:] = fa[-wmax-1:];        # y negativas. Añadimos también la frecuencia -1, que tiene
                                    # que ver con la "redondez" global de la figura
   
    if fa[-1] > fa[1]:              # normalizamos el sentido de recorrido
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]
    
    return v / s

# invertimos la imagen para evitar el rectángulo exterior blanco
mod = extractContours(255-rgb2gray(readrgb('shapes/trebol.png')))[0].reshape(-1,2)
imod = invar(mod)


model = np.zeros((200,200), np.uint8)
cv.drawContours(model, [mod], -1, 255, 1)
cv.imshow('model',model)


def razonable(c):
    return cv.arcLength(c, closed=True) >= 50

for (key,frame) in autoStream():

    g = rgb2gray(frame)
    # sería bueno eliminar contornos que tocan el borde de la imagen, etc.
    contours = extractContours(g)
    
    ok = [c for c in contours if razonable(c) ]
    
    found = [c for c in ok if norm(invar(c)-imod) < 0.15 ]
    #print(len(contours), len(ok), len(found))
    
    #cv.drawContours(frame, contours, -1, (255,128,128), 1)
    #cv.drawContours(frame, ok, -1, (0,0,255), 1)
    cv.drawContours(frame, found, -1, (0,255,0), cv.FILLED)

    cv.imshow('shapes',frame)

cv.destroyAllWindows()

