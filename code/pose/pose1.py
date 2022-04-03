#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# Esta versión sustituye el código anterior por las mismas funciones
# disponibles en umucv

# pruébalo con el vídeo de siempre

# ./pose1.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose1.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import lineType, cube, showCalib
from umucv.contours import extractContours, redu


def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)

K = Kfov( size, 60 )

print(K)

marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])


def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]


def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

# (esta versión devuelve un objeto, no una tupla)
def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]



for key,frame in stream:

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)

    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]


    # rellenamos el marcador (podríamos intentar "borrarlo")
    cv.drawContours(frame,[htrans(M,marker).astype(int) for M in poses], -1, (0,128,255), -1, lineType)

    # mostramos un objeto 3D virtual en verde
    cv.drawContours(frame,[ htrans(M,cube).astype(int) for M in poses ], -1, (0,128,0), 3, lineType)

    cv.imshow('source',frame)
    
