#!/usr/bin/env python

# pip install --upgrade https://robot.inf.um.es/material/umucv.tar.gz
# python pose.py --dev=file:../images/rot4.mjpg

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose, kgen
from umucv.util     import lineType, cube, showCalib
from umucv.contours import extractContours, redu

imvirt = cv.resize(cv.imread('../images/ccorr/models/thing.png'),(200,200))

stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)

K = kgen(size,1.7) # fov aprox 60 degree

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

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]


kk = 0

for key,frame in stream:
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    #print(len(good))

    # como list comprehension:
    #poses = [ Me for err, Me in [bestPose(K,g,marker)] for g in good if err < 2 ] 
    # con un bucle
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        #print(err,Me)
        if p.rms < 2:
            poses += [p.M]
            #print(Me)

    cosa = cube + 0

    #cv.drawContours(frame,[c.astype(int) for c in cs], -1, (255,255,0), 1, lineType)

    #cv.drawContours(frame,[c.astype(int) for c in good], -1, (0,0,255), 3, lineType)

    cv.drawContours(frame,[htrans(M,marker).astype(int) for M in poses], -1, (0,255,255), 1, lineType)

    #cv.drawContours(frame,[htrans(M,cosa).astype(int) for M in poses], -1, (0,128,0), 3, lineType)

    for p in poses[:0]:
        src = np.array([[0.,0],[0,200],[200,200],[200,0]]).astype(np.float32)
        dst = htrans(p,np.array([[0.25,0.5,0],[.75,0.5,0],[.75,0.5,1],[0.25,0.5,1]])).astype(np.float32)
        H = cv.getPerspectiveTransform(src,dst) #(es la homografía plana)
        cv.warpPerspective(imvirt,H,size,frame,0,cv.BORDER_TRANSPARENT)

    showCalib(K,frame)
    cv.imshow('source',frame)
    
