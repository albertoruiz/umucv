#!/usr/bin/env python

# Detección del marcador ../../images/ref.png

# Puedes probarlo con
#  ./polygon1.py --dev=../../images/rot4.mjpg

# en esta imagen hay figuras parecidas pero que no son el marcador:

#  ./polygon1.py --dev=dir:../../images/polis.png


from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.contours import extractContours
from umucv.htrans import htrans

def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

# coordenadas reales del marcador
marker = np.array(
       [[0,   0  ],
        [0,   1  ],
        [0.5, 1  ],
        [0.5, 0.5],
        [1,   0.5],
        [1,   0  ]])

# calculamos la homografía que relaciona un polígono observado con el marcador
# y devolvemos también el error de ajuste
def errorMarker(c):
    H,_ = cv.findHomography(c, marker)
    err = abs(htrans(H,c) - marker).sum()
    return err, H

# genera todas las posibles ordenaciones de puntos
def rots(c):
    return [np.roll(c,k,axis=0) for k in range(len(c))]

# el primer vértice del polígono detectado puede ser cualquiera
# probamos todas las asociaciones y nos quedamos con la que tenga menor error
def bestRot(c):
    return min( [ (errorMarker(r), r) for r in rots(c) ] )


for key,frame in autoStream():

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    good = polygons(cs, n=6, prec=3)
    
    # filtramos los polígonos detectados dejando solo los que se ajusten
    # muy bien al marcador
    ok = [ (c,H) for (err, H), c in map(bestRot, good) if err < 0.1 ]
    
    # dibujamos en cyan todos los contornos interesantes
    cv.drawContours(frame,[c.astype(int) for c in cs], -1, (255,255,0), 1, cv.LINE_AA)
    # dibujamos en rojo los hexágonos encontrados
    cv.drawContours(frame,[c.astype(int) for c in good], -1, (0,0,255), 1, cv.LINE_AA)
    # dibujamos en naranja grueso los marcadores
    cv.drawContours(frame,[c.astype(int) for c,_ in ok], -1, (0,128,255), 3, cv.LINE_AA)
    cv.imshow('source',frame)

