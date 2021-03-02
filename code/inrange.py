#!/usr/bin/env python

# Este programa tiene 4 objetivos:
# 1) Ilustrar la construcción de una máscara que indica
#    la localización de objetos de un color caractéristico,
#    mediante una sencilla umbralización.
# 2) Comprobar que el espacio de color HSV es útil para esto.
# 3) Separar diferentes objetos mediante componentes conexas
#    obteniendo también su bounding box, area y centro.
# 4) Realizar la misma separación mediante extracción de
#    contornos.

# python inrange.py --dev=dir:../images/naranjas/*.jpg --resize=0x400
# python inrange.py --dev=dir:../images/demos/fruits.png

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
    
    # Leemos los sliders
    h1 = cv.getTrackbarPos('h1','mask')
    h2 = cv.getTrackbarPos('h2','mask')
    s1 = cv.getTrackbarPos('s1','mask')
    s2 = cv.getTrackbarPos('s2','mask')
    v1 = cv.getTrackbarPos('v1','mask')
    v2 = cv.getTrackbarPos('v2','mask')
    
    # opencv genera máscaras np.uint8 con el convenio 0 (No) - 255 (Sí)
    # que se ven Negro - Blanco en un cv.imshow.
    mask0  = cv.inRange(hsv, (h1,s1,v1), (h2,s2,v2) )
    cv.imshow('mask', mask0)

    # Si hay pequeños puntos activos de ruido, pueden eliminarse con un
    # filtro de imagen (lo explicaremos más adelante)
    mask = cv.morphologyEx( mask0, cv.MORPH_OPEN, np.ones([3,3], np.uint8) )
    cv.imshow('clean', mask)
    # dependiendo de la aplicación, puede ser necesario aplicar otros filtros
    # para rellenar agujeros, etc.


    # mask2 = scipy.ndimage.morphology.binary_opening(mask)
    # Otros módulos de proceso de imagen pueden generar int 0-1
    # y dependiendo del uso que hagamos puede ser necesario convertirlas
    

    # Una vez que tenemos la máscara, podemos "extraer" los objetos,
    # mostrándolos sobre fondo negro. Se puede hacer de varias formas:
    # multiplicando por la máscara (con valores 0-1, y hay que hace np.expand_dims),
    # asignando la imagen en otra vacía en la zona válida,
    # o anulando la zona de fondo:
    result = frame.copy()
    result[mask ==0] = (0,0,0)
    cv.imshow('masked', result)
   
    # Lo importante realmente es separar los objetos. Para ello usamos el algoritmo
    # de componentes conexas, en su variante más completa:    
    rois = frame.copy()
    n, cc, st, cen = cv.connectedComponentsWithStats(mask)

    # En principio, podemos mostrar la imagen de etiquetas, pero se verán niveles de gris
    # muy parecidos. En matplotlib queda bien con las paletas que tiene, con opencv multiplicamos
    # para separar un poco los tonos de gris:
    cv.imshow('CC',5*cc.astype(np.uint8))

    # Es mejor dibujar las bounding box, opcionalmente filtrando por área:
    for x in range(n):
        if 5**2 < st[x][cv.CC_STAT_AREA] < 50**2:
            x1 = st[x][cv.CC_STAT_LEFT]
            y1 = st[x][cv.CC_STAT_TOP]
            x2 = st[x][cv.CC_STAT_WIDTH] + x1
            y2 = st[x][cv.CC_STAT_HEIGHT] + y1
            cv.rectangle(rois, (x1,y1), (x2,y2), (255,0,0))
    cv.imshow('rois',rois)


    # Alternativamente, podemos extraer contornos:
    contours, _ = cv.findContours( mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]    
    # Quitamos la dimensión superflua y los contornos muy cortos
    ok = [ x.reshape(-1,2) for x in contours if len(x) > 10 ]

    otra = frame.copy()
    # Hay dos formas de dibujar los contornos
    cv.polylines(otra, ok, isClosed=True, color=(255,0,255))
    #cv.drawContours(otra, ok, -1 , (255,0,128) )
    cv.imshow('contours', otra)
    

