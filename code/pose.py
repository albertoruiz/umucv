#!/usr/bin/env python

# pip install --upgrade https://robot.inf.um.es/material/umucv.tar.gz
# python pose.py --dev=file:../images/rot4.mjpg

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose, kgen, f_from_hfov
from umucv.util     import lineType, cube, showCalib
from umucv.contours import extractContours, redu

# para situarla en la escena
imvirt = cv.imread('../images/ccorr/models/thing.png')

# consultamos la resolución de la cámara para formar correctamente K
stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)

K = kgen( size, f_from_hfov(np.radians(60)) ) # fov aprox 60 degree

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

# prueba distintas colocaciones de los puntos de referencia
def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]

nf=0

for key,frame in stream:
    nf += 1
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    # buscamos polígonos con los mismos lados que el marcador
    good = polygons(cs,6,3)
    #print(len(good))

    # bestPose nos da la cámara e información adicional
    # nos quedamos con las de menor error de reproyección
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        #print(err,Me)
        if p.rms < 2:
            poses += [p]
            #print(Me)

    cosa = cube + 0

    # todos los contornos 
    #cv.drawContours(frame,[c.astype(int) for c in cs], -1, (255,255,0), 1, lineType)

    # posibles marcadores
    #cv.drawContours(frame,[c.astype(int) for c in good], -1, (0,0,255), 3, lineType)

    # reproyección del marcador con las cámaras estimadas
    #cv.drawContours(frame,[htrans(p.M,marker).astype(int) for p in poses], -1, (0,255,255), 1, lineType)

    # mostramos un objeto 3D virtual
    cv.drawContours(frame,[htrans(p.M,cosa).astype(int) for p in poses], -1, (0,128,0), 3, lineType)

    # proyectamos una "textura" (imagen) donde deseemos
    for p in poses[:1]:
        # los extremos de la "imagen virtual" que vamos a proyectar
        h,w = imvirt.shape[:2]
        src = np.array([[0,0],[0,h],[w,h],[w,0]]).astype(np.float32)
        
        # dónde queremos ponerla en el sistema de referencia del marcador
        world = np.array([[0.25,0.5,0],[.75,0.5,0],[.75,0.5,1],[0.25,0.5,1]])
        
        # dónde se verá en la imagen de cámara
        dst = htrans(p.M, world).astype(np.float32)

        # calculamos la transformación
        #H, _ = cv.findHomography(src,dst)
        # igual que findHomography pero solo con 4 correspondencias
        H = cv.getPerspectiveTransform(src,dst)
        # la aplicamos encima de la imagen de cámara
        cv.warpPerspective(imvirt,H,size,frame,0,cv.BORDER_TRANSPARENT)
        
        # tenemos también la distancia la marcador
        # print(np.linalg.norm(p.C))


    showCalib(K,frame)
    cv.imshow('source',frame)
    
