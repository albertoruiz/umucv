#!/usr/bin/env python

# pip install --upgrade https://robot.inf.um.es/material/umucv.tar.gz
# conda install pyqtgraph pyopengl

# python pose3D.py --dev=file:../images/rot4.mjpg


import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

import cv2   as cv
import numpy as np
import numpy.linalg as la

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose, kgen, rotation, desp, sepcam, jc, jr, col, row
from umucv.contours import extractContours, redu


def cameraTransf(M):
    K,R,C = sepcam(M)
    rt = jr(jc(R, -R @ col(C)),
            row(0,0,0,1))
    return la.inv(rt)

# esquema en 3d de una cámara
def cameraOutline2(f, sc=0.3):    
    x = 1
    y = x
    z = f

    ps = [ x, y, z,
          -x, y, z,
          -x,-y, z,
           x,-y, z,
           x, y, z,
           0, 0, 0,
          -x, y, z,
          -x,-y, z,
           0, 0, 0,
           x, -y, z ]

    ps = np.array(ps).reshape(-1,3)
    return sc*ps




def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    sp = sorted(poses,key=lambda p: p.rms)
    return sp[0]


marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)


f = 1.7
K = kgen(size,f) # fov aprox 60 degree
print(K)



## Create a GL View widget to display data
app = QtGui.QApplication([])
win = gl.GLViewWidget()
win.show()
win.setWindowTitle('pose')
win.setCameraPosition(distance=20)

## grid
g = gl.GLGridItem()
win.addItem(g)


ax = gl.GLAxisItem(glOptions='opaque')
ax.setSize(2,2,2)

win.addItem(ax)
ax.setTransform(QtGui.QMatrix4x4(*(rotation((1,0,0),0.0001,homog=True).flatten())))
ax.translate(0,0,-0.02)

axc = gl.GLAxisItem(glOptions='opaque')
axc.setSize(1,1,1)
#axc.translate(0,0,0.02)
win.addItem(axc)



# imagen
view = gl.GLImageItem(data=np.zeros([100,100,4]))
win.addItem(view)


# marker
gmark = gl.GLLinePlotItem(pos=np.vstack([marker,marker[0]]),color=(255,0,0,1),antialias=True,width=3)
gmark.setGLOptions('opaque')
gmark.translate(0,0,0.01)
win.addItem(gmark)

# camera
cam = gl.GLLinePlotItem(pos=np.array([[0,0,0]]),color=(255,255,255,1),antialias=True,width=2)
cam.setGLOptions('opaque')
win.addItem(cam)



camsize = 0.5
drawCam = cameraOutline2(f,camsize)

W2 = WIDTH/2
H2 = HEIGHT/2

A = desp((0,0,f*camsize)) @ np.diag([1,1,1,W2/camsize]) @ desp((-W2,-H2,0))
        
def img2tex(image):
    x = image.transpose(1,0,2)
    texture,_ = pg.makeARGB(x, useRGBA=True)
    return texture


def update():
    key, img = next(stream)
    g = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    
    poses = []
    for c in good:
        p = bestPose(K,c,marker)    
        if p.rms < 2:
            poses += [p.M]
            cv.polylines(img,[c],True,(255,255,0),3)

    if poses:
        p = poses[0]
        T = cameraTransf(p)
        cam.setData(pos= htrans(T, drawCam ) )
        view.setData(data=img2tex(img))
        m = T @ A
        view.setTransform(QtGui.QMatrix4x4(*(m.flatten())))
        axc.setTransform(QtGui.QMatrix4x4(*(T.flatten())))
        #print(p)

    cv.imshow('input', img)
    #key = cv.waitKey(1);
    

    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

QtGui.QApplication.instance().exec_()

