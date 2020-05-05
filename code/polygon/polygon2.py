#!/usr/bin/env python

# Rectificación del marcador ../../images/ref.png

# Puedes probarlo con el vídeo
#  ./polygon2.py --dev=../../images/rot4.mjpg

from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.contours import extractContours
from umucv.htrans import htrans, desp, scale

def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

marker = np.array(
       [[0,   0  ],
        [0,   1  ],
        [0.5, 1  ],
        [0.5, 0.5],
        [1,   0.5],
        [1,   0  ]])

def errorMarker(c):
    H,_ = cv.findHomography(c, marker)
    err = abs(htrans(H,c) - marker).sum()
    return err, H


def rots(c):
    return [np.roll(c,k,axis=0) for k in range(len(c))]


def bestRot(c):
    return min( [ (errorMarker(r), r) for r in rots(c) ] )


for key,frame in autoStream():

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    good = polygons(cs, n=6, prec=3)
    
    ok = [ (c,H) for (err, H), c in map(bestRot, good) if err < 0.1 ]
    
    if ok:
        # cogemos la homografía de rectificación del primer marcador detectado
        c,H = ok[0]
        
        # La combinamos con un escalado y desplazamiento para que la imagen
        # resultante quede de un tamaño adecuado
        T = desp([100,100]) @ scale([100,100]) @ H
        
        # rectificamos
        rectif = cv.warpPerspective(frame, T, (500,500))
        
        cv.imshow('rectif', rectif)
        

