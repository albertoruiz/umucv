#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# En esta versión añadimos una imagen fuera del plano

# pruébalo con el vídeo de siempre

# ./pose3.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose3.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.contours import extractContours, redu


# la imagen que queremos añadir
imvirt = cv.imread('../../images/ccorr/models/thing.png')


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


K = Kfov( size, 60 )


marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

square = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [1,   1,   0],
        [1,   0,   0]])



def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]


for n, (key,frame) in enumerate(stream):

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    for M in poses:

        # vamos a poner verticalmente la imagen imvirt cargada más arriba
        
        # las coordenadas de sus 4 esquinas
        # (se pueden sacar del bucle de captura)
        h,w = imvirt.shape[:2]
        src = np.array([[0,0],[0,h],[w,h],[w,0]])
        
        # decidimos dónde queremos poner esas esquinas en el sistema de referencia del marcador
        # (si no cambian se puede sacar del bucle de captura)
        world = np.array([[0.25,0.5,0],[.75,0.5,0],[.75,0.5,1],[0.25,0.5,1]])
        
        # calculamos dónde se proyectarán en la imagen esas esquinas
        # usamos la matriz de cámara estimada
        dst = htrans(M, world)

        # calculamos la transformación
        #H, _ = cv.findHomography(src,dst)
        # igual que findHomography pero solo con 4 correspondencias
        H = cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        # la aplicamos encima de la imagen de cámara
        cv.warpPerspective(imvirt,H,size,frame,0,cv.BORDER_TRANSPARENT)
        
        # tenemos también la distancia la marcador
        # print(np.linalg.norm(p.C))


    cv.imshow('source',frame)
    
