#!/usr/bin/env python

# Implementación sencilla del detector de corners
# basado en la distribución local del gradiente

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
import time

# suavizdo
def gaussian(s,x):
    return cv.GaussianBlur(x,(0,0), s)

# derivadas horizontal y vertical (gradiente de imagen)
def grad(x):
    gx = cv.Sobel(x,-1,1,0)
    gy = cv.Sobel(x,-1,0,1)
    return gx,gy

# supresión de no máximos
def nms(x, t = 0.1):
    m = cv.dilate(x, np.ones((5,5),np.uint8))  # filtro de máximo
    h = np.max(m)
    return (x == m) & (x > t*h)


for key, frame in autoStream():
    # pasamos a monocromo y con valores reales (no enteros)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)

    t0 = time.time()
    
    # gradiente    
    gx,gy = grad( gaussian(2, gray) )
    
    # los elementos de la matriz de covarianza
    gx2 = gx * gx
    gy2 = gy * gy
    xyg = gx * gy
    
    # sus medias en un entorno
    sx2 = gaussian(5,gx2)
    sy2 = gaussian(5,gy2)
    sxy = gaussian(5,xyg)
    
    # valor propio más pequeño en cada pixel, que indica la intensidad de esquina
    lmin = sx2 + sy2 - np.sqrt(sx2**2 + sy2**2 + 4*sxy**2 - 2*sx2*sy2)
    
    cv.imshow('lambda min',  lmin/lmin.max() )
    
    # extraemos los picos de respuesta
    cornermask = nms(lmin, t=0.1)
    py, px = np.where(cornermask)    
    corners = np.array([px,py]).T

    t1 = time.time()
    
    # y los mostramos encima de la imagen original
    for x,y in corners:
        cv.circle(frame, (int(x), int(y)) , radius=3 , color=(0,0,255), thickness=-1, lineType=cv.LINE_AA )
    
    putText(frame, f'{len(corners)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)

