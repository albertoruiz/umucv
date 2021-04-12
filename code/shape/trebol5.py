#!/usr/bin/env python

# Paso 5: Finalmente comparamos los invariantes de los contornos 
#         encontrados en la imagen con el modelo y señalamos
#         los que son muy parecidos
          

# ./trebol5.py --dev=dir:../../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np
# import necesario
from umucv.util import putText


from numpy.fft import fft


def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)

    s = fa[1] + fa[-1]

    v = np.zeros(2*wmax+1)
    v[:wmax] = fa[2:wmax+2]
    v[wmax:] = fa[-wmax-1:]

   
    if fa[-1] > fa[1]:
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]

    return v / s


def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r

def extractContours(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours

def razonable(c):
    return 100**2 >= cv.contourArea(c) >= 10**2

def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0


black = True

shcont = False

model = extractContours(cv.imread('../../images/shapes/trebol.png'))[0]

modelaux = np.zeros([200,200], np.uint8)
cv.drawContours(modelaux, [model], -1, 255, cv.FILLED)
cv.imshow('model', modelaux)

invmodel = invar(model)

MAXDIST = 0.15

for (key,frame) in autoStream():

    if key == ord('c'):
        shcont = not shcont

    contours = extractContours(frame)
    
    ok = [c for c in contours if razonable(c) and not orientation(c)]

    # seleccionamos los contornos con un descriptor muy parecido al del modelo     
    found = [c for c in ok if np.linalg.norm(invar(c)-invmodel) < MAXDIST ]


    if shcont:
        result = np.zeros_like(frame)
        cp = [c for c in contours if orientation(c) ]
        cn = [c for c in contours if not orientation(c) ]
        
        cv.drawContours(result, cp, contourIdx=-1, color=(255,128,128), thickness=1, lineType=cv.LINE_AA)
        cv.drawContours(result, cn, -1, (128,128,255), 1)
    else:
        result = frame
        # en este modo de visualización mostramos solo los detectados
        cv.drawContours(result, found, -1, (0,255,0), cv.FILLED)
    
    # y en ambos modos mostramos la similitud (y el área)
    for c in found:
        s = np.linalg.norm(invar(c)-invmodel)
        a = cv.contourArea(c)
        #info = f'{s:.2f} {a}'
        info = f'{s:.2f}'
        putText(result ,info,c.mean(axis=0).astype(int))

    cv.imshow('shape recognition',result)

cv.destroyAllWindows()

# puedes añadir un trackbar para controlar el umbral de detección
# se pueden evitar la repetición de operaciones en putText


