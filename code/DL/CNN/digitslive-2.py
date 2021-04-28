#!/usr/bin/env python

# 2)  Vamos a normalizar el tamaño de las manchas de tinta de la misma forma
#     que los ejemplos MNIST. Se convierten en imágenes 28x28 siguiendo el
#     método explicado en el notebook machine-learning.ipynb.


import cv2 as cv
import numpy as np
from umucv.stream import autoStream


def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow('bin',r)
    return r

def ccs(mask):
    n, cc, st, cen = cv.connectedComponentsWithStats(mask)
    trozos = []
    LMIN=10
    LMAX=100
    for x in range(n):
        if LMIN**2 < st[x][cv.CC_STAT_AREA] < LMAX**2:
            x1 = st[x][cv.CC_STAT_LEFT]
            y1 = st[x][cv.CC_STAT_TOP]
            x2 = st[x][cv.CC_STAT_WIDTH] + x1
            y2 = st[x][cv.CC_STAT_HEIGHT] + y1
            trozos.append( ( cen[x], (cc[y1:y2+1, x1:x2+1]==x).astype(np.uint8)*255 ) )
    return trozos

def extractThings(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    return ccs(b)

###############################################################
# (hay otras formas de calcular el centro de masas)
def center(p):
    r,c = p.shape
    rs = np.outer(range(r),np.ones(c))
    cs = np.outer(np.ones(r),range(c))
    s = np.sum(p)
    my  = np.sum(p*rs) / s
    mx  = np.sum(p*cs) / s
    return mx,my

# La figura se escala a un tamaño 20x20
# respetando la proporción de tamaño
# El resultado se mete en una caja 28x28 de
# modo que la media quede en el centro
# y reescalamos los valores entre cero y uno
# (La clave está en la transformación H y la
# función warpAffine, que estudiaremos 
# en un tema siguiente)
def adaptsize(x):
    h,w = x.shape
    s = max(h,w)
    h2 = (s-h)//2
    w2 = (s-w)//2
    y = x
    if w2>0:
        z1 = np.zeros([s,w2],np.uint8)
        z2 = np.zeros([s,s-w-w2],np.uint8)
        y  = np.hstack([z1,x,z2])
    if h2>0:
        z1 = np.zeros([h2,s],np.uint8)
        z2 = np.zeros([s-h-h2,s],np.uint8)
        y  = np.vstack([z1,x,z2])
    y = cv.resize(y,(20,20))
    mx,my = center(y)
    H = np.array([[1.,0,4-(mx-9.5)],[0,1,4-(my-9.5)]])
    return cv.warpAffine(y,H,(28,28))/255
#################################################################

black = True

for key, frame in autoStream():

    cosas = extractThings(frame)

    # obtenemos la lista con los recortes normalizados
    nor = [adaptsize(c) for (x,y), c in cosas]

    # Quitamos la ventana auxiliar y
    # mostramos todas las detecciones (que quepan)
    # en la parte inferior de la imagen de entrada
    caben = frame.shape[1]//28
    for k,x in enumerate(nor[:caben]):
        frame[-28:,28*k:28*(k+1),:] = x.reshape(28,28,1)*255

    cv.imshow('digits',frame)

cv.destroyAllWindows()


