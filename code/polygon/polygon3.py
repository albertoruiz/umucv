#!/usr/bin/env python

# Sistema de referencia virtual

# Puedes probarlo con el vídeo
#  ./polygon3.py --dev=../../images/rot4.mjpg

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

# coordenadas de las líneas que vamos a dibujar
horiz = np.array( [ [[x,0],[x,100]] for x in range(0,110,10)] ) + (50,50)
vert  = np.array( [ [[0,y],[100,y]] for y in range(0,110,10)] ) + (50,50)

for key,frame in autoStream():

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    good = polygons(cs, n=6, prec=3)
    
    ok = [ (c,H) for (err, H), c in map(bestRot, good) if err < 0.1 ]
    
    if ok:
        # elegimos la homografía del primer marcador detectado
        c,H = ok[0]

        # la adaptamos igual que antes
        T = desp([100,100]) @ scale([100,100]) @ H
        
        # calculamos la transformación inversa        
        IH = np.linalg.inv(T)
        
        # y nos llevamos las líneas del mundo real a la imagen
        thoriz = htrans(IH, horiz.reshape(-1,2)).reshape(-1,2,2)
        tvert  = htrans(IH,  vert.reshape(-1,2)).reshape(-1,2,2)
        
        
        cv.polylines(frame, thoriz.astype(int), False, (0,255,0), 1, cv.LINE_AA) 
        cv.polylines(frame,  tvert.astype(int), False, (0,255,0), 1, cv.LINE_AA) 
        
    cv.imshow('source',frame)

