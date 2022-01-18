#!/usr/bin/env python

# Detección de polígonos negros

# Puedes probarlo con
#  ./polygon0.py --dev=../../images/rot4.mjpg
#  ./polygon0.py --dev=dir:../../images/marker.png
# o con la cámara, imprimiendo el marcador ../../images/ref.png
# o poniéndolo en la pantalla del ordenador o del móvil.

# paquetes habituales
from umucv.stream   import autoStream
import cv2 as cv
import numpy as np

# Extraeremos los contornos de las manchas oscuras más destacadas.
# Lo haremos igual que en el ejercicio trebol.py de reconocimiento de
# siluetas. Ese código está disponible en la siguiente función:
from umucv.contours import extractContours

# La estrategia será extraer los contornos y reducir sus nodos con el método
# que comentamos brevemente en shapes.ipynb. Nos quedamos con los que
# se queden con el número de vértices deseado.

# reducimos una polilínea con una tolerancia dada
def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

# filtramos una lista de contornos con aquellos
# que quedan reducidos a n vértices con la precisión deseada.
def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]


for key,frame in autoStream():

    # extraemos los contornos con el método habitual: umbral automático,
    # y eliminación de los contornos poco prometedores
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    # seleccionamos los que se pueden reducir sin mucho error a 6 vértices
    good = polygons(cs, n=6, prec=3)
 
    # dibujamos en cyan todos los contornos interesantes
    cv.drawContours(frame,[c.astype(int) for c in cs], -1, (255,255,0), 1, cv.LINE_AA)
    # dibujamos en rojo más grueso los hexágonos encontrados
    cv.drawContours(frame,[c.astype(int) for c in good], -1, (0,0,255), 3, cv.LINE_AA)
    cv.imshow('source',frame)

