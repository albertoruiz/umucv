#!/usr/bin/env python

# Paso 3: Vamos a cargar una silueta que servirá de
#         modelo a reconocer.

# puedes probarlo con
# ./trebol3.py --dev=dir:../../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r

# metemos dentro la conversión a gris y binarización    
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

# cargamos una imagen con la silueta deseada y directamente extraemos
# el primer contorno encontrado, (el mayor de todos)
model = extractContours(cv.imread('../../images/shapes/trebol.png'))[0]

# Lo mostramos en una ventana 
modelaux = np.zeros([200,200], np.uint8)
cv.drawContours(modelaux, [model], -1, 255, cv.FILLED)
cv.imshow('model', modelaux)


# el resto prácticamente igual

for (key,frame) in autoStream():

    if key == ord('c'):
        shcont = not shcont

    contours = extractContours(frame)
    
    ok = [c for c in contours if razonable(c) and not orientation(c)]

    if shcont:
        result = np.zeros_like(frame)
        cp = [c for c in contours if orientation(c) ]
        cn = [c for c in contours if not orientation(c) ]
        
        cv.drawContours(result, cp, contourIdx=-1, color=(255,128,128), thickness=1, lineType=cv.LINE_AA)
        cv.drawContours(result, cn, -1, (128,128,255), 1)
    else:
        result = frame
        cv.drawContours(result, ok, -1, (255,0,0), cv.FILLED)
    
    cv.imshow('shape recognition',result)

cv.destroyAllWindows()

