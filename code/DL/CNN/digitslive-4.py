#!/usr/bin/env python

# 4) Usamos la red convolucional que hemos entrenado en el notebook

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util   import putText
import time

##############################################################################

from tensorflow.keras.models import load_model

# el modelo preentrenado está aquí:
# wget https://robot.inf.um.es/material/va/digits.keras
model = load_model('../../../data/digits.keras')

def classifyN(xs):
    # ponemos la estructura de array que espera la red: una lista de imágenes de un canal
    t = np.array(xs).reshape(-1,28,28,1)
    # y hacemos lo mismo de antes, devolvemos la clase más probable y su probabilidad
    p = model.predict(t)
    r = np.argmax(p,axis=1)
    pm = np.max(p,axis=1)
    return r,pm

########################################################################################

# el resto del código es exactamente igual


def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
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


def center(p):
    r,c = p.shape
    rs = np.outer(range(r),np.ones(c))
    cs = np.outer(np.ones(r),range(c))
    s = np.sum(p)
    my  = np.sum(p*rs) / s
    mx  = np.sum(p*cs) / s
    return mx,my

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


black = True

for key, frame in autoStream():

    cosas = extractThings(frame)

    nor = [adaptsize(c) for (x,y), c in cosas]
    
    loc = [ c[0] for c in cosas ]

    t0 = time.time()
    if nor:
        clas,prob = classifyN(nor)
    else:
        clas,prob = [],[]
    t1 = time.time()

    for (x,y), label,pr in zip(loc, clas, prob):
        col = (0,255,255)
        if pr < 0.5:
            label = '?'
        if pr < 0.9:
            col = (0,160,160)
        putText(frame, str(label), (int(x),int(y)), color=col, div=1, scale=2, thickness=2)


    caben = frame.shape[1]//28
    for k,x in enumerate(nor[:caben]):
        frame[-28:,28*k:28*(k+1),:] = x.reshape(28,28,1)*255

    cv.imshow('digits',frame)

cv.destroyAllWindows()


