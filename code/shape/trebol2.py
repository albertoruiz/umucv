#!/usr/bin/env python

# Paso 2: seleccionamos los contornos "prometedores"
#         y usamos una sola ventana

# puedes probarlo con
# ./trebol2.py --dev=dir:../../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r
    
def extractContours(g):  
    contours, _ = cv.findContours(g.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


# Para eliminar contornos claramente erróneos
# (pueden añadirse más criterios)
def razonable(c):
    return 100**2 >= cv.contourArea(c) >= 10**2

# esta función nos indica si el contorno se recorre en sentido horario o
# antihorario. La función findContours
# recorre en sentido antihorario las regiones True de la máscara binaria,
# (que con black=True serán las manchas oscuras) y en sentido horario
# los agujeros y las regiones claras.
def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0


black = True

# para elegir si mostramos la imagen original o todos los contornos
shcont = True


for (key,frame) in autoStream():

    # cambiamos el modo de visualización
    if key == ord('c'):
        shcont = not shcont

    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    
    b = binarize(g)
    contours = extractContours(b)

    # seleccionamos contornos oscuros de tamaño medio
    ok = [c for c in contours if razonable(c) and not orientation(c)]

    if shcont:
        # en este modo de visualización mostramos en colores distintos
        # las manchas oscuras y las claras
        result = np.zeros_like(frame)
        cp = [c for c in contours if orientation(c) ]
        cn = [c for c in contours if not orientation(c) ]
        
        cv.drawContours(result, cp, contourIdx=-1, color=(255,128,128), thickness=1, lineType=cv.LINE_AA)
        cv.drawContours(result, cn, -1, (128,128,255), 1)
    else:
        # en este modo de visualización mostramos los contornos que pasan el primer filtro
        result = frame
        cv.drawContours(result, ok, -1, (255,0,0), cv.FILLED)
    
    cv.imshow('shape recognition',result)

cv.destroyAllWindows()


