#!/usr/bin/env python

# en esta versión usamos pyqtgraph para mostrar la posición
# de la cámara en 3D

# python pose3D.py --dev=../../images/rot4.mjpg


import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

import cv2   as cv
import numpy as np
import numpy.linalg as la

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose, kgen, rotation, desp, scale, sepcam, jc, jr, col, row
from umucv.contours import extractContours, redu


# a partir de la matriz de cámara construye la matriz de cambio de base
# para dibujar la cámara en el espacio
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



# detectamos la pose de la cámara con el marcador igual que antes

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

K = Kfov(size,60)


## Create a GL View widget to display data
app = QtGui.QApplication([])
win = gl.GLViewWidget()
win.show()
win.setWindowTitle('pose')
win.setCameraPosition(distance=20)

# a partir de aquí añadimos los elementos de la escena
# (en algunos solo se crea espacio para ellos,
# los datos verdaderos se actualizarán en el bucle de captura)


## añadimos a la escena un grid en el plano base
g = gl.GLGridItem()
win.addItem(g)

# añadimos los ejes de coordenadas del mundo
# (los perturbamos un poco para que se vean mejor)
ax = gl.GLAxisItem(glOptions='opaque')
ax.setSize(2,2,2)
#ax.setTransform(QtGui.QMatrix4x4(*(rotation((1,0,0),0.0001,homog=True).flatten())))
ax.translate(0,0,+0.02)
win.addItem(ax)

# añadimos los ejes de la cámara
axc = gl.GLAxisItem(glOptions='opaque')
axc.setSize(1,1,1)
axc.translate(0,0,0.02)
win.addItem(axc)

# añadimos la imagen en vivo, que situaremos en el esqueleto de la cámara
view = gl.GLImageItem(data=np.zeros([100,100,4]))
win.addItem(view)

# añadimos la silueta del marcador
gmark = gl.GLLinePlotItem(pos=np.vstack([marker,marker[0]]),color=(255,0,0,1),antialias=True,width=3)
gmark.setGLOptions('opaque')
gmark.translate(0,0,0.01)
win.addItem(gmark)

# añadimos el esqueleto de la cámara
cam = gl.GLLinePlotItem(pos=np.array([[0,0,0]]),color=(255,255,255,1),antialias=True,width=2)
cam.setGLOptions('opaque')
win.addItem(cam)


# añadimos la imagen rectificada del plano del marcador
world = gl.GLImageItem(data=np.zeros([100,100,4]))
win.addItem(world)


# generamos el esqueleto de la cámara con el tamaño deseado
camsize = 0.5
f = 1.6
drawCam = cameraOutline2(f,camsize)

W2 = WIDTH/2
H2 = HEIGHT/2

# transformación para que la cámara quede bien en la imagen
A = desp((0,0,f*camsize)) @ np.diag([1,1,1,W2/camsize]) @ desp((-W2,-H2,0))

# para mostrar imágenes en la escena 3D hay que pasarlas a un formato de "textura"
def img2tex(image):
    x = image.transpose(1,0,2)
    texture,_ = pg.makeARGB(x, useRGBA=True)
    return texture

# utilidad para convertir los arrays de numpy al tipo que usa pyqtgraph
def transform(H,obj):
    obj.setTransform(QtGui.QMatrix4x4(*(H.flatten())))
  
# Este es el bucle de captura de pyqtgraph. Es un un callback que se llama
# con un timer
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
            #cv.polylines(img,[c],True,(255,255,0),3)

    if poses:
        M = poses[0]
        
        # hasta aquí todo es igual que antes.
        # Tenemos la matriz de cámara M
        
        # sacamos la tranformación que nos permite situar en 3D
        # el esqueleto de la cámara, sus ejes, y la imagen que se ve en ella
        T = cameraTransf(M)
        
        transform(T,axc)
        cam.setData(pos= htrans(T, drawCam ) )

        view.setData(data=img2tex(img))
        m = T @ A
        transform(m,view)

        
        # A partir de la matriz de cámara sacamos la homografía del plano
        # (esto también se puede hace como en el capítulo anterior)
        # La homografía del plano z=0 se puede extraer de la matriz de cámara
        # simplemente quitando la 3 columna (la que corresponde a la coordenada z).
        # Hay que escalar para que encaje con el tamaño de la escena 3D.
        s = 1/50
        S = scale((s,s))
        HZ0 = desp((250,250)) @ la.inv(M[:,[0,1,3]] @ S)
        # rectificamos la imagen
        rectif = cv.warpPerspective(img, HZ0,(500,500))
        
        # la situamos en la escena con la escala adecuada
        world.setData(data=img2tex(rectif))
        transform( scale((s,s,1)) @ desp((-250,-250,0)) , world)


    cv.imshow('input', img)


# arrancamos la aplicación    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

QtGui.QApplication.instance().exec_()

