#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# Esta versión incluye la implementación que vimos en el notebook

# pruébalo con el vídeo de siempre

# ./pose0.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose0.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans
from umucv.util     import lineType
from umucv.contours import extractContours, redu


# Utilidades que necesitaremos luego

# crea una matriz columna a partir de elementos o de un vector 1D
def col(*args):
    a = args[0]
    n = len(args)
    if n==1 and type(a) == np.ndarray and len(a.shape) == 1:
        return a.reshape(len(a),1)
    return np.array(args).reshape(n,1)

# crea una matriz fila
def row(*args):
    return col(*args).T

# juntar columnas
def jc(*args):
    return np.hstack(args)

# juntar filas
def jr(*args):
    return np.vstack(args)


# matriz de calibración sencilla dada la
# resolución de la imagen y el fov horizontal en grados
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


# mide el error de una transformación (p.ej. una cámara)
# rms = root mean squared error
# "reprojection error"
def rmsreproj(view, model, transf):
    err = view - htrans(transf,model)
    return np.sqrt(np.mean(err.flatten()**2))


# recupera la posición de la cámara a partir de coordenadas en la imagen
# de un objeto conocido. Necesita la matriz de calibración.
def pose(K, image, model):
    ok,rvec,tvec = cv.solvePnP(model, image, K, (0,0,0,0))
    if not ok:
        return 1e6, None
    R,_ = cv.Rodrigues(rvec)
    M = K @ jc(R,tvec)
    rms = rmsreproj(image,model,M)
    return rms, M


# un objeto 3D que introduciremos en la escena
# (Es una polilínea que dibuja recorre los vértices de un cubo)
# Por supuesto, lo interesante sería usar un motor gráfico de verdad
# y mostrar un modelo 3D detallado basado en polígonos con textura)
cube = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,0],
    
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
        
    [1,0,1],
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [0,1,1],
    [0,1,0]
    ])


# consultamos la resolución de la cámara para formar correctamente K
stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)

K = Kfov( size, 60 ) # fov aprox 60 degree

print(K)

# este es nuestro marcador, define el sistema de coordenadas del mundo 3D
marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])


# seleccionamos los contornos que pueden reducirse al número de lados deseado
# exactamente igual que la semana pasada
def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

# generamos todos los posibles puntos de partida
def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]


# igual que la semana pasada, probamos todas las asociaciones de puntos
# imagen - modelo y nos quedamos con la que produzca menos error de ajuste
def bestPose(K,view,model):
    poses = [ pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p[0])[0]

# cambia esto para mostrar los contornos y el marcador detectado
debug = False

# empezamos el bucle de captura
for key,frame in stream:

    # extraemos los contornos como siempre, con la utilidad de umucv
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    # buscamos polígonos con los mismos lados que el marcador
    good = polygons(cs,6,3)
    #print(len(good))

    # bestPose nos da la cámara y el error
    # si encontramos varios posibles marcadores nos quedamos solo con los
    # que tengan un error de reproyección menor de 2 pixels
    # Si aparece un polígono por casualidad con 6 lados, es muy difícil que
    # sea consistente con una posible imagen del marcador.
    # Este critero elimina casi todos los falsos positivos
    poses = []
    for g in good:
        rms, M = bestPose(K,g,marker)
        if rms < 2:
            poses += [M]
            # guardamos las matrices de cámara encontradas
            # (tantas como marcadores se detecten, lo normal
            # es que en la escena haya una o ninguna).

    # es mejor usar el teclado cambiar la información que se muestra
    if debug:
        # dibujamos todos los contornos en morado
        cv.drawContours(frame,[c.astype(int) for c in cs], -1, (255,255,0), 1, lineType)

        # los polígonos de 6 lados (posibles marcadores) en rojo
        cv.drawContours(frame,[c.astype(int) for c in good], -1, (0,0,255), 3, lineType)

        # la reproyección del marcador con la cámaras estimada en amarillo
        cv.drawContours(frame,[htrans(M,marker).astype(int) for M in poses], -1, (0,255,255), 1, lineType)

    # mostramos un objeto 3D virtual en verde
    cv.drawContours(frame,[ htrans(M,cube).astype(int) for M in poses ], -1, (0,128,0), 3, lineType)

    cv.imshow('source',frame)
    
